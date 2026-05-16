"""
:class:`PathPyPI` — managed PEP 503 simple index at any :class:`Path` root.

A drop-in "package artifactory" that publishes Python distributions
under a yggdrasil :class:`~yggdrasil.io.path.path.Path` root —
local, workspace, S3, DBFS volumes, in-memory — and keeps a
PEP 503-shaped ``index.html`` per project so the result is consumable
by ``pip install --extra-index-url=<root>``.

Two ingestion shapes:

- :meth:`PathPyPI.publish` — build a real ``pip`` wheel for a local
  module (importable name or source path) and upload the versioned
  wheel under ``<root>/<normalized-project>/<wheel>.whl``. This is
  the pip-installable path: the wheel filename carries the version,
  the index page lists every uploaded wheel, and re-publishing the
  same source short-circuits when the target already exists.
- :meth:`PathPyPI.publish_archive` — zip a module via
  :meth:`Path.upload_module` and drop the archive under the same
  versioned layout. Used when the artefact ships as a raw zip
  (Spark ``addArtifacts(pyfile=True)``, ``sys.path`` extension)
  rather than a pip wheel.

Both shapes use the same versioning + subfolder convention; the
caller picks based on the consumer.

:class:`yggdrasil.databricks.jobs.workspace_pypi.WorkspacePyPI` is
a thin subclass that pins the root to ``/Workspace/Shared/.ygg/pypi/simple``
and binds the publisher to a workspace client.
"""
from __future__ import annotations

import importlib.metadata as ilm
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path as LocalPath
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from yggdrasil.io.path.path import Path


__all__ = [
    "PathPyPI",
    "parse_wheel_filename",
    "normalize_pep503_name",
]

LOGGER = logging.getLogger(__name__)

# PEP 503 normalization: any run of ``-_.`` collapses to ``-``; the
# whole project name lowers. Used for the per-project subfolder name
# so ``MyPackage`` / ``my_package`` / ``my-package`` all share one
# index folder.
_PEP503_NORMALIZE_RE = re.compile(r"[-_.]+")

# PEP 427 wheel filename layout:
#   {distribution}-{version}(-{build})?-{python}-{abi}-{platform}.whl
_WHEEL_FILENAME_RE = re.compile(
    r"^(?P<dist>[^-]+)-(?P<version>[^-]+)"
    r"(?:-[^-]+)?-[^-]+-[^-]+-[^-]+\.whl$",
)


def normalize_pep503_name(name: str) -> str:
    """Return the PEP 503-normalized form of a project name."""
    return _PEP503_NORMALIZE_RE.sub("-", name).strip("-").lower()


def parse_wheel_filename(name: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(distribution, version)`` from a wheel filename, or ``(None, None)``."""
    match = _WHEEL_FILENAME_RE.match(name)
    if not match:
        return None, None
    return match.group("dist"), match.group("version")


class PathPyPI:
    """A PEP 503 simple index hosted at *root* (any yggdrasil :class:`Path`).

    Parameters
    ----------
    root
        Where to publish. Anything :meth:`Path.from_` accepts —
        a :class:`Path` instance, a URL string
        (``"/Workspace/Shared/pypi/simple"``,
        ``"s3://my-bucket/pypi"``, ``"file:///tmp/pypi"``), or a
        :class:`pathlib.Path`. Strings without a scheme resolve to
        :class:`LocalPath`.

    Examples
    --------
    Local index for testing::

        pypi = PathPyPI("/tmp/my-index")
        pypi.publish("my_local_pkg")
        # → /tmp/my-index/my-local-pkg/my_local_pkg-0.1.0-py3-none-any.whl

    Layered S3 index::

        pypi = PathPyPI("s3://wheels.example.com/simple")
        pypi.publish("internal_lib")
    """

    def __init__(self, root: Union[str, "Path", LocalPath]) -> None:
        from yggdrasil.io.path.path import Path as _Path

        if isinstance(root, _Path):
            self.root: "Path" = root
        else:
            self.root = _Path.from_(root)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(root={self.root!r})"

    # ------------------------------------------------------------------ #
    # Wheel publishing — pip-installable artefacts
    # ------------------------------------------------------------------ #
    def publish(
        self,
        module: Any,
        *,
        source_path: Optional[str] = None,
        version: Optional[str] = None,
        rebuild: bool = False,
    ) -> "Path":
        """Build a wheel for *module* and upload it under :attr:`root`.

        *module* is anything :func:`resolve_local_lib_path` accepts —
        an importable module name (``"my_local_pkg"``), a directory
        path, or a pre-built ``.whl``. The wheel is built via
        ``python -m pip wheel --no-deps`` so we don't take a hard
        dependency on :mod:`build`. Pre-built wheels handed in via
        *source_path* are uploaded verbatim.

        Re-publishing the same source is a no-op unless *rebuild* is
        set — the wheel filename carries the version, so an
        already-uploaded target short-circuits.

        Returns the published :class:`Path` (full URL accessible via
        ``.full_path()`` or ``.url``).
        """
        from yggdrasil.environ.modules import resolve_local_lib_path

        local_root = (
            LocalPath(source_path).resolve()
            if source_path
            else resolve_local_lib_path(module)
        )
        if not local_root.exists():
            raise FileNotFoundError(
                f"{type(self).__name__}.publish: source {local_root!r} does "
                "not exist; pass source_path=<dir or wheel> or ensure the "
                "module is importable."
            )

        LOGGER.debug(
            "Publishing module %r from %r to %r",
            module, local_root, self,
        )

        if local_root.is_file() and local_root.suffix.lower() == ".whl":
            wheel_path = local_root
        else:
            wheel_path = self._build_wheel(local_root)

        dist, wheel_version = parse_wheel_filename(wheel_path.name)
        project = dist or (
            module if isinstance(module, str) else local_root.name
        )
        resolved_version = version or wheel_version or _module_version(module)
        normalized = normalize_pep503_name(project)
        project_root: "Path" = self.root / normalized
        target: "Path" = project_root / wheel_path.name

        if not rebuild and target.exists():
            LOGGER.debug(
                "Wheel %r already published at %r — skipping upload",
                wheel_path.name, target,
            )
        else:
            target.write_bytes(wheel_path.read_bytes())
            LOGGER.info(
                "Published wheel %r (project=%r, version=%r)",
                target, project, resolved_version,
            )

        self._refresh_index(project_root, project)
        return target

    # ------------------------------------------------------------------ #
    # Archive publishing — raw zip artefacts via Path.upload_module
    # ------------------------------------------------------------------ #
    def publish_archive(
        self,
        module: Any,
        *,
        name: Optional[str] = None,
        version: Optional[str] = None,
        rebuild: bool = True,
    ) -> "Path":
        """Zip *module* via :meth:`Path.upload_module` and place under :attr:`root`.

        Mirrors :meth:`publish` but emits a raw ``.zip`` (suitable for
        ``SparkSession.addArtifacts(pyfile=True)`` /
        ``sys.path`` extension) instead of a pip wheel. The version is
        derived from installed metadata when available; pass *version*
        to override. The archive filename follows
        ``{name}-{version}.zip`` so the per-project subfolder still
        sorts by version.

        ``rebuild=True`` (default) re-uploads each call — matches the
        cheap-and-idempotent contract of :meth:`Path.upload_module`.
        """
        if isinstance(module, str) and not LocalPath(module).exists():
            project = name or module
        elif name:
            project = name
        else:
            from yggdrasil.environ.modules import resolve_local_lib_path
            project = resolve_local_lib_path(module).name

        resolved_version = version or _module_version(module) or "0.0.0"
        normalized = normalize_pep503_name(project)
        project_root: "Path" = self.root / normalized
        archive_name = f"{normalized}-{resolved_version}.zip"
        target: "Path" = project_root / archive_name

        if not rebuild and target.exists():
            LOGGER.debug(
                "Archive %r already published at %r — skipping upload",
                archive_name, target,
            )
        else:
            # ``Path.upload_module`` handles the zipping + transport
            # in one round-trip; we just point it at the per-project
            # versioned filename.
            target.upload_module(module, name=archive_name, overwrite=True)
            LOGGER.info(
                "Published archive %r (project=%r, version=%r)",
                target, project, resolved_version,
            )

        self._refresh_index(project_root, project)
        return target

    # ------------------------------------------------------------------ #
    # Importing — pip-install from the index, then import
    # ------------------------------------------------------------------ #
    def import_module(
        self,
        module_name: str,
        *,
        version: Optional[str] = None,
        install: bool = False,
        cache_dir: Any = None,
    ) -> Any:
        """Locate and import the published *module_name* from this index.

        Locates the matching wheel / archive under
        ``<root>/<normalized-name>/``, prefers the requested *version*
        when set (matched against the wheel filename), otherwise picks
        the lexicographically-highest filename (the typical PEP 440
        ordering for the same project). The resolved
        :class:`Path` is then imported via
        :meth:`Path.import_module`.

        ``install=False`` (default) downloads the artefact and prepends
        the local archive to ``sys.path``. For ``.zip`` archives that's
        enough; for ``.whl`` wheels the import will fail with
        :class:`ModuleNotFoundError` unless the wheel is already
        installed locally — pass ``install=True`` to opt into the
        ``pip install`` fallback. The conservative default avoids
        mutating the active interpreter's site-packages on a casual
        lookup; callers that explicitly want the install step ask for
        it.

        Raises :class:`ModuleNotFoundError` when no artefact is
        published under that name, with a hint pointing at
        :meth:`publish` / :meth:`publish_archive`.
        """
        normalized = normalize_pep503_name(module_name)
        project_root: "Path" = self.root / normalized
        if not project_root.exists():
            raise ModuleNotFoundError(
                f"{type(self).__name__}.import_module: no artefact "
                f"published for {module_name!r} at {project_root!r}. "
                f"Call {type(self).__name__}.publish({module_name!r}) "
                "first."
            )

        artefacts = sorted(
            (
                p for p in project_root.iterdir()
                if p.name.endswith((".whl", ".zip", ".tar.gz"))
            ),
            key=lambda p: p.name,
        )
        if not artefacts:
            raise ModuleNotFoundError(
                f"{type(self).__name__}.import_module: index folder "
                f"{project_root!r} contains no wheels or archives for "
                f"{module_name!r}."
            )

        if version is not None:
            chosen = next(
                (
                    p for p in artefacts
                    if _artefact_version_matches(p.name, version)
                ),
                None,
            )
            if chosen is None:
                raise ModuleNotFoundError(
                    f"{type(self).__name__}.import_module: no artefact "
                    f"for {module_name!r} matches version {version!r}. "
                    f"Available: {[p.name for p in artefacts]}."
                )
        else:
            # Highest filename wins — PEP 440 versions sort lexically
            # the way callers expect for the common case (1.2 < 1.10
            # is the only well-known gotcha, and pip's resolver hits
            # the same constraint when scanning a simple index).
            chosen = artefacts[-1]

        LOGGER.debug(
            "Importing %r via %r from %r",
            module_name, chosen, self,
        )
        return chosen.import_module(
            module_name, install=install, cache_dir=cache_dir,
        )

    # ------------------------------------------------------------------ #
    # Index maintenance
    # ------------------------------------------------------------------ #
    def _refresh_index(self, project_root: "Path", project: str) -> None:
        """Rewrite the per-project ``index.html`` from current artefact files.

        One ``<a>`` tag per ``.whl`` / ``.zip``, sorted by name. The
        index sits next to the artefacts so pip's
        ``--extra-index-url=<root>/`` walk lands on the right page.
        Cheap to rebuild on every upload (one ``iterdir`` + one write).
        """
        entries = sorted(
            (
                p for p in project_root.iterdir()
                if p.name.endswith((".whl", ".zip", ".tar.gz"))
            ),
            key=lambda p: p.name,
        )
        rows = "\n".join(
            f'    <a href="{entry.name}">{entry.name}</a><br/>'
            for entry in entries
        )
        body = (
            "<!DOCTYPE html>\n<html><head>"
            f"<title>Links for {project}</title></head><body>\n"
            f"  <h1>Links for {project}</h1>\n"
            f"{rows}\n"
            "</body></html>\n"
        )
        (project_root / "index.html").write_bytes(body.encode("utf-8"))

    # ------------------------------------------------------------------ #
    # Wheel build
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_wheel(local_root: LocalPath) -> LocalPath:
        """Build a wheel for *local_root* via ``pip wheel --no-deps``.

        Uses pip (always present alongside the active interpreter) so
        we don't take a hard dependency on :mod:`build` / ``hatch``.
        The wheel lands in a temp dir; the caller reads + uploads
        immediately, after which the temp dir is reclaimed by the GC.
        """
        tmp = LocalPath(tempfile.mkdtemp(prefix="ygg-pypi-"))
        cmd = [
            sys.executable, "-m", "pip", "wheel",
            "--no-deps", "--quiet",
            "--wheel-dir", str(tmp),
            str(local_root),
        ]
        LOGGER.debug("Building wheel for %r via %s", local_root, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to build wheel for {local_root!r}: pip wheel exited "
                f"{exc.returncode}.\nstdout: {exc.stdout}\nstderr: {exc.stderr}"
            ) from exc
        wheels = sorted(tmp.glob("*.whl"))
        if not wheels:
            raise RuntimeError(
                f"pip wheel produced no .whl files under {tmp!r} for "
                f"{local_root!r}; check the package metadata."
            )
        return wheels[0]


def _artefact_version_matches(filename: str, version: str) -> bool:
    """Best-effort: does *filename* carry *version* in its PEP 427 slot?"""
    _, parsed = parse_wheel_filename(filename)
    if parsed is not None:
        return parsed == version
    # ``{name}-{version}.zip`` shape from :meth:`PathPyPI.publish_archive`.
    stem = filename.rsplit(".", 1)[0]
    if stem.endswith(".tar"):
        stem = stem[:-4]
    return stem.endswith(f"-{version}")


def _module_version(module: Any) -> Optional[str]:
    """Best-effort: pull the installed version of *module* from metadata."""
    if not isinstance(module, str):
        module = getattr(module, "__name__", None)
    if not module:
        return None
    top = module.split(".", 1)[0]
    # ``packages_distributions`` walks every site-packages dist —
    # share the cached snapshot maintained by ``environ.modules`` so
    # the scan runs at most once per process across every publisher
    # and introspection call site.
    from yggdrasil.environ.modules import packages_distributions_cached

    dists = packages_distributions_cached().get(top)
    if not dists:
        return None
    try:
        return ilm.distribution(dists[0]).version
    except Exception:  # noqa: BLE001 — best-effort
        return None
