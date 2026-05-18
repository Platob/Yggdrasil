""":class:`WorkspacePyPIRegistry` — workspace-backed shared wheel cache.

A lightweight PEP 503-shaped layout under a single
:class:`WorkspacePath` so colleagues on the same workspace can share
locally-built wheels without round-tripping through a real PyPI
server. The path is ``<base>/<pkg>/<pkg>-<version>-py3-none-any.whl``;
the registry is lazy (wheels only build when a consumer asks for
the dep) and idempotent (existing entries are reused unless the
caller explicitly forces a rebuild, or the dep is an editable
install — those re-publish every load with the local hostname
folded into the version so each developer's working copy lands at
a stable but unique slot).

Used by :meth:`DatabricksClient.spark` to wire serverless deps
into :class:`DatabricksEnv.withDependencies`:

- Public PyPI distributions go through verbatim
  (``"ygg==0.7.84"``); the cluster's pip handles them.
- Editable / private distributions get a wheel built locally
  (``pip wheel <name> --no-deps``), uploaded to the workspace
  registry once, then downloaded back to a local cache and
  declared via the ``local:`` prefix Databricks Connect understands.

The "is this a public PyPI package?" check is opt-in via
``check_public=True`` — an HTTPS HEAD against ``pypi.org/pypi/<name>/json``
— so an offline registry doesn't pay the latency by default. The
default classification reads ``direct_url.json`` straight off the
installed distribution (PEP 610) to spot editables, and otherwise
treats every dep as private (publish locally, share via the
workspace cache).
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path as _LocalPath
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Sequence, Tuple, Union

from yggdrasil.pickle import json as ygg_json

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.fs.workspace_path import WorkspacePath


__all__ = [
    "DependencyKind",
    "DependencyInfo",
    "WorkspacePyPIRegistry",
    "classify_dependency",
]


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class DependencyKind:
    """Enum-shaped marker for how a dep ships to the cluster."""

    PUBLIC: ClassVar[str] = "public"
    """Pip-installable from a public index — pass the spec straight
    to :meth:`DatabricksEnv.withDependencies`."""

    LOCAL: ClassVar[str] = "local"
    """Installed locally from a private source — build a wheel and
    share through the workspace registry."""

    EDITABLE: ClassVar[str] = "editable"
    """``pip install -e .`` install — version-stamp with the local
    hostname so each developer's working copy gets its own slot in
    the registry; always re-upload on resolve."""


@dataclass(slots=True)
class DependencyInfo:
    """Resolution of a single dep into a Databricks Connect spec.

    Attributes:
        name: PyPI project name (case-preserving). Used as both the
            install spec for public deps and the directory key under
            the registry for everything else.
        version: Resolved version string. Editable deps get a PEP
            440 local-version suffix (``+host-<hostname>``) so the
            same project from different machines lands at distinct
            registry entries.
        kind: One of :class:`DependencyKind` constants.
        source: Filesystem location of the source tree (editable /
            local) or ``None`` for public deps.
        spec: The pip install spec for public deps
            (``"ygg==0.7.84"``); ``None`` for everything else.
    """

    name: str
    version: Optional[str] = None
    kind: str = DependencyKind.LOCAL
    source: Optional[_LocalPath] = None
    spec: Optional[str] = None

    @property
    def is_public(self) -> bool:
        return self.kind == DependencyKind.PUBLIC

    @property
    def is_editable(self) -> bool:
        return self.kind == DependencyKind.EDITABLE

    @property
    def wheel_basename(self) -> str:
        """Conventional wheel filename under the registry layout.

        Always emits a pure-python wheel name —
        ``<name>-<version>-py3-none-any.whl`` — so reads from the
        registry don't have to guess at the tag. The build step is
        the authoritative source; this method is for cache lookup
        only.
        """
        return f"{self.name}-{self.version}-py3-none-any.whl"


def _looks_like_pip_spec(s: str) -> bool:
    """Distinguish bare names from PyPI install specs.

    A "spec" is anything pip understands as more than a project
    name on its own — a version operator (``"ygg==1.0"``,
    ``"numpy>=1"``) **or** an extras tag (``"ygg[data,databricks]"``).
    Bare names like ``"ygg"`` route through the installed-dist
    classifier so editable installs get caught; specs go straight
    to ``PUBLIC`` with the raw string handed to pip on the cluster.
    """
    return "[" in s or any(
        op in s for op in ("==", ">=", "<=", "!=", "~=", ">", "<")
    )


def _hostname_version(base_version: Optional[str]) -> str:
    """Build a PEP 440 local-version that embeds the host name.

    ``1.2.3`` → ``1.2.3+host.<host>``; ``None`` /
    unparseable → ``0.0.0+host.<host>``. The hostname is sanitized
    (PEP 440 local segments allow ``[a-zA-Z0-9.]`` only) and the
    timestamp slot is filled by the registry caller when it wants
    truly-unique-per-load identity. For the default editable flow,
    same-hostname collisions are fine: the registry overwrites on
    every load.
    """
    host = socket.gethostname() or "unknown"
    host = "".join(c if c.isalnum() else "-" for c in host).strip("-") or "unknown"
    base = (base_version or "0.0.0").split("+", 1)[0]
    return f"{base}+host.{host}"


def _detect_editable(dist) -> bool:
    """PEP 610 ``direct_url.json`` carries the editable flag."""
    try:
        text = dist.read_text("direct_url.json")
    except Exception:
        return False
    if not text:
        return False
    try:
        info = ygg_json.loads(text)
    except Exception:
        return False
    dir_info = info.get("dir_info") if isinstance(info, dict) else None
    if not isinstance(dir_info, dict):
        return False
    return bool(dir_info.get("editable"))


def _direct_url_path(dist) -> Optional[_LocalPath]:
    """Local filesystem path the dist points at, when known."""
    try:
        text = dist.read_text("direct_url.json")
    except Exception:
        return None
    if not text:
        return None
    try:
        info = ygg_json.loads(text)
    except Exception:
        return None
    url = info.get("url") if isinstance(info, dict) else None
    if isinstance(url, str) and url.startswith("file://"):
        return _LocalPath(url[len("file://"):])
    return None


def _is_public_pypi(name: str, *, timeout: float = 2.0) -> bool:
    """Best-effort ``HEAD pypi.org/pypi/<name>/json`` probe.

    Returns ``True`` iff the JSON endpoint exists. Any failure
    (network down, name unknown, timeout) is treated as "not
    public" so the registry falls back to publishing locally —
    publishing a private wheel is recoverable; pip-installing a
    name that doesn't resolve isn't.
    """
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://pypi.org/pypi/{name}/json", method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def classify_dependency(
    obj: Any,
    *,
    check_public: bool = False,
) -> DependencyInfo:
    """Resolve *obj* into a :class:`DependencyInfo`.

    Accepts the same shapes as the rest of the path API:

    - ``"ygg"`` / ``"ygg==0.7.84"`` — already a pip spec. Spec
      with an operator goes straight to ``PUBLIC``. A bare name
      consults ``importlib.metadata`` (and, when
      ``check_public=True``, ``pypi.org``) to pick between
      ``PUBLIC`` / ``EDITABLE`` / ``LOCAL``.
    - :class:`os.PathLike` — always ``LOCAL``; the path is the
      source tree.
    - Any object exposing ``__module__`` — resolves via its top
      package name.
    """
    import importlib.metadata as ilm
    from yggdrasil.environ.environment import safe_pip_name

    if isinstance(obj, os.PathLike) or hasattr(obj, "__fspath__"):
        path = _LocalPath(os.fspath(obj)).resolve()
        return DependencyInfo(
            name=path.name,
            kind=DependencyKind.LOCAL,
            source=path,
            version=None,
        )

    if isinstance(obj, str):
        spec_str = obj.strip()
        if _looks_like_pip_spec(spec_str):
            # Spec already carries the version / extras constraint;
            # trust it. Strip extras (``ygg[data]`` → ``ygg``) and
            # version operators so ``name`` is the canonical project
            # name for the cache key.
            name = spec_str.split("[", 1)[0]
            for op in ("==", ">=", "<=", "!=", "~=", ">", "<"):
                name = name.split(op, 1)[0]
            name = "".join(c for c in name if c.isalnum() or c in "-_.").strip()
            return DependencyInfo(
                name=name or spec_str,
                kind=DependencyKind.PUBLIC,
                spec=spec_str,
            )
        name = spec_str
    else:
        module_name = getattr(obj, "__module__", None) or getattr(obj, "__name__", None)
        if not module_name:
            raise ValueError(
                f"classify_dependency: cannot resolve {obj!r} — expected a "
                f"pip spec string, a path, or an object with __module__."
            )
        name = module_name.split(".", 1)[0]

    # Try to find a local installation matching the name.
    project_name = str(safe_pip_name(name))
    dist = None
    for candidate in (project_name, name):
        try:
            dist = ilm.distribution(candidate)
            break
        except ilm.PackageNotFoundError:
            continue

    if dist is None:
        # Not installed locally; trust caller and treat as a public spec.
        return DependencyInfo(
            name=project_name,
            kind=DependencyKind.PUBLIC,
            spec=project_name,
        )

    version = dist.version
    editable = _detect_editable(dist)
    source = _direct_url_path(dist)

    if editable:
        return DependencyInfo(
            name=str(dist.metadata["Name"] or project_name),
            version=_hostname_version(version),
            kind=DependencyKind.EDITABLE,
            source=source,
        )

    if check_public and _is_public_pypi(project_name):
        return DependencyInfo(
            name=str(dist.metadata["Name"] or project_name),
            version=version,
            kind=DependencyKind.PUBLIC,
            spec=f"{project_name}=={version}",
        )

    return DependencyInfo(
        name=str(dist.metadata["Name"] or project_name),
        version=version,
        kind=DependencyKind.LOCAL,
        source=source,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class WorkspacePyPIRegistry:
    """PEP 503-shaped wheel cache living under a :class:`WorkspacePath`.

    Layout::

        <base_path>/
            <package>/
                <package>-<version>-py3-none-any.whl
                <package>-<other-version>-py3-none-any.whl
                ...

    ``base_path`` defaults to
    ``/Workspace/Users/<me>/.ygg/pypi/simple`` so a single-user
    install Just Works without an admin first; pass an explicit
    workspace path (``/Workspace/Shared/...``) to share across a
    team.

    :meth:`publish` is the entry point. It classifies the dep,
    builds a wheel locally when needed, uploads it to the
    workspace under the layout above, and downloads it back to
    *local_cache* (default: a temp dir) so the caller can pass
    the result to
    :meth:`DatabricksEnv.withDependencies` with the ``local:``
    prefix Databricks Connect understands. Public PyPI specs
    short-circuit to ``"<name>==<version>"`` with no upload.

    The registry is lazy: existing entries are reused as-is, and
    only editable deps are re-uploaded on every call (their
    hostname-stamped version slot is the same on the same
    machine, so this is a single overwriting upload, not a leaky
    history).
    """

    client: "DatabricksClient"
    base_path: "WorkspacePath" = field(default=None)  # type: ignore[assignment]
    local_cache: _LocalPath = field(default=None)  # type: ignore[assignment]

    DEFAULT_BASE: ClassVar[str] = "/Workspace/Users/<me>/.ygg/pypi/simple"

    def __post_init__(self) -> None:
        from yggdrasil.databricks.fs.workspace_path import WorkspacePath

        if self.base_path is None:
            self.base_path = WorkspacePath(self.DEFAULT_BASE, client=self.client)
        elif not isinstance(self.base_path, WorkspacePath):
            self.base_path = WorkspacePath.from_(self.base_path, client=self.client)
        else:
            self.base_path = self.base_path.with_client(self.client)

        if self.local_cache is None:
            self.local_cache = _LocalPath(
                tempfile.gettempdir(),
            ) / "yggdrasil" / "pypi-cache"
        self.local_cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish_many(
        self,
        deps: Sequence[Any],
        *,
        check_public: bool = False,
    ) -> Tuple[list[str], list["WorkspacePath"]]:
        """Resolve a batch of deps. Returns ``(specs, remote_paths)``.

        ``specs`` is the list of strings to feed straight into
        :meth:`DatabricksEnv.withDependencies` — a mix of
        ``"name==version"`` (public) and ``"local:<path>"``
        (private). ``remote_paths`` collects the workspace
        entries (``None``-free, for the public deps that didn't
        upload anywhere) so callers can mirror / inspect the
        cache.
        """
        specs: list[str] = []
        remotes: list[Any] = []
        for dep in deps:
            spec, remote = self.publish(dep, check_public=check_public)
            specs.append(spec)
            if remote is not None:
                remotes.append(remote)
        return specs, remotes

    def publish(
        self,
        obj: Any,
        *,
        check_public: bool = False,
    ) -> Tuple[str, Optional["WorkspacePath"]]:
        """Resolve a single dep into a ``withDependencies``-ready spec.

        - ``PUBLIC`` → returns ``("<name>==<version>", None)``.
        - ``EDITABLE`` → builds a wheel with the hostname-stamped
          version, uploads under
          ``<base>/<name>/<wheel>``, downloads back to
          ``local_cache``, returns ``("local:<local-path>",
          <workspace-path>)``.
        - ``LOCAL`` → same as editable, but only uploads when the
          workspace entry is missing (the registry is lazy).
        """
        info = classify_dependency(obj, check_public=check_public)
        if info.is_public:
            assert info.spec is not None
            return info.spec, None

        # Build / fetch the wheel locally, then publish to workspace.
        wheel = self._materialize_wheel(info)
        remote = self.base_path / info.name / wheel.name

        try:
            existing = remote.exists()
        except Exception:
            existing = False

        if info.is_editable or not existing:
            remote.parent.mkdir(parents=True, exist_ok=True)
            remote.write_bytes(wheel.read_bytes())

        # Ensure a local copy exists for the local: prefix.
        local = self.local_cache / wheel.name
        if local != wheel:
            local.write_bytes(wheel.read_bytes())

        return f"local:{local}", remote

    def resolve_local(self, remote: "WorkspacePath") -> _LocalPath:
        """Pull *remote* into :attr:`local_cache` and return the path.

        Useful as a separate entry for consumers that already
        know the registry path (built earlier, or pre-populated by
        a teammate) and just want it on disk to feed
        ``withDependencies(local:...)``.
        """
        local = self.local_cache / remote.name
        local.write_bytes(remote.read_bytes())
        return local

    # ------------------------------------------------------------------
    # Internal — wheel build
    # ------------------------------------------------------------------

    def _materialize_wheel(self, info: DependencyInfo) -> _LocalPath:
        """Build (or copy) a wheel for *info* into :attr:`local_cache`.

        Order:

        1. *info.source* points at an existing ``.whl`` file →
           copy it to the cache and return.
        2. The installed distribution can be wheel-built —
           ``pip wheel <name> --no-deps -w <cache>`` — and we use
           the resulting filename. For editable installs the
           wheel name carries the original version; we rewrite it
           inside the wheel's METADATA isn't worth the
           complexity, so we instead **rename** the file to the
           hostname-stamped version so the workspace slot stays
           unique per host. Pip is happy to install the renamed
           wheel because ``--no-deps`` skips the constraint check.
        3. Fall back to :func:`build_module_archive` — a deflated
           zip of the package directory (no wheel metadata).
           Callers that hit this branch should know their dep is
           an installable shape; the zip is a last resort for
           bare module directories the user wants on the cluster
           via ``addArtifacts``, not via pip.
        """
        from yggdrasil.io.path._module_pack import build_module_archive

        if info.source is not None and info.source.is_file():
            suffix = info.source.suffix.lower()
            if suffix in (".whl", ".tar.gz", ".zip"):
                target = self.local_cache / info.source.name
                if target != info.source:
                    target.write_bytes(info.source.read_bytes())
                return target

        # Try ``pip wheel <name>`` for installed dists.
        try:
            return self._pip_wheel(info)
        except _WheelBuildError as exc:
            LOGGER.info(
                "pip wheel failed for %s (%s); falling back to zip archive",
                info.name, exc,
            )

        # Last resort — build a deflated zip from the source tree.
        if info.source is not None and info.source.is_dir():
            return _LocalPath(build_module_archive(info.source, dest=self.local_cache))

        raise RuntimeError(
            f"Cannot build a distribution for {info.name!r}: no source path "
            f"is known and ``pip wheel`` failed. Pass a path-shaped "
            f"dependency or install the project first."
        )

    def _pip_wheel(self, info: DependencyInfo) -> _LocalPath:
        """Run ``pip wheel`` and return the produced file."""
        with tempfile.TemporaryDirectory(prefix="ygg-pip-wheel-") as raw:
            outdir = _LocalPath(raw)
            cmd = [
                sys.executable, "-m", "pip", "wheel", info.name,
                "--no-deps", "-w", str(outdir),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                stderr = getattr(exc, "stderr", "") or str(exc)
                raise _WheelBuildError(stderr) from exc

            wheels = sorted(outdir.glob("*.whl"))
            if not wheels:
                raise _WheelBuildError(
                    "pip wheel produced no output (expected exactly one .whl)"
                )
            built = wheels[0]

            # Editable deps land at the dist's original version; rename
            # so the workspace slot is hostname-stamped.
            final_name = built.name
            if info.is_editable and info.version is not None:
                # Wheel filename: ``<name>-<version>-py3-none-any.whl``;
                # swap the version segment.
                parts = built.stem.split("-")
                if len(parts) >= 2:
                    parts[1] = info.version
                    final_name = "-".join(parts) + built.suffix

            target = self.local_cache / final_name
            target.write_bytes(built.read_bytes())
            return target


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _WheelBuildError(Exception):
    """Raised when :meth:`WorkspacePyPIRegistry._pip_wheel` cannot
    build a wheel; the registry then falls back to the zip path."""
