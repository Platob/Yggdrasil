"""Build wheels from the **live** package on disk and ship them to Databricks.

The wheel registry is the workspace's PyPI-like index — one folder per
distribution under ``/Workspace/Shared/pypi/<dist>/`` holding its version
binaries. This module owns the whole wheel lifecycle: synthesize a buildable
project from an installed package (:func:`synthesize_project`), build it
(:func:`build_wheel` / :func:`build_wheels_for_versions` / :func:`build_bundle`),
upload it (:func:`upload_wheel` / :func:`registry_upload`), and get-or-create it
(:func:`ensure_wheel` / :func:`ensure_ygg_wheels` / :func:`ensure_bundle`).

The :class:`Wheels` service (``dbc.wheels``) is the OO front door over these
functions, returning :class:`~yggdrasil.databricks.wheels.wheel.Wheel` handles.
"""
from __future__ import annotations

import importlib
import importlib.metadata as ilmd
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from ..service import DatabricksService
from .wheel import Wheel

if TYPE_CHECKING:
    from ..client import DatabricksClient

logger = logging.getLogger(__name__)

__all__ = [
    "WORKSPACE_PYPI_DIR",
    "WORKSPACE_WHL_DIR",
    "SERVERLESS_ENVIRONMENT_VERSION",
    "SERVERLESS_ENVIRONMENT_VERSIONS",
    "SERVERLESS_WHEEL_PLATFORMS",
    "BUNDLE_EXCLUDE",
    "SUPPORTED_PYTHONS",
    "serverless_environment_version",
    "environment_key_for",
    "wheel_for_python",
    "is_editable_install",
    "user_pypi_dir",
    "distribution_for",
    "import_packages_for",
    "synthesize_project",
    "build_wheel",
    "build_wheels_for_versions",
    "build_bundle",
    "build_project_wheel",
    "download_dependency_wheels",
    "upload_wheel",
    "registry_upload",
    "ensure_wheel",
    "ensure_wheels",
    "deployed_wheels",
    "ensure_ygg_wheel",
    "ensure_ygg_wheels",
    "ensure_bundle",
    "ensure_bundles",
    "Wheels",
]




def _run_build(cmd: "list[str]") -> None:
    """Run a wheel-build subprocess (uv / pip) **quietly**.

    The build tools are chatty — their progress belongs behind the CLI's
    spinner, not flooding the caller's stdout. Output is captured and, on a
    non-zero exit, the (trimmed) tail is folded into the raised
    :class:`RuntimeError` so a failure stays diagnosable. A missing executable
    still surfaces as :class:`FileNotFoundError` (so callers can fall back from
    ``uv`` to ``pip``)."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "").strip()
        logger.error("build command failed (%s): %s\n%s", exc.returncode, " ".join(cmd), tail)
        raise RuntimeError(
            f"build command failed (exit {exc.returncode}): "
            f"{' '.join(cmd[:4])} … — {tail[-2000:]}"
        ) from exc

#: Root of the workspace's PyPI-like wheel registry. Each distribution gets a
#: folder under it holding its version binaries (wheel / egg / …) — a flat,
#: PEP 503 "simple index"-style layout shared across the workspace::
#:
#:     /Workspace/Shared/pypi/<dist>/<dist>-<version>-py3-none-any.whl
#:
#: A new version drops alongside the others under the dist folder (no isolated
#: per-version subdir), so the registry is browsable and reusable like an index.
WORKSPACE_PYPI_DIR = "/Workspace/Shared/pypi"

#: Back-compat alias — the registry root (was an isolated ``.ygg/whl`` path).
WORKSPACE_WHL_DIR = WORKSPACE_PYPI_DIR

#: Latest serverless environment version — the fallback when the local Python
#: isn't one we map to an older runtime.
SERVERLESS_ENVIRONMENT_VERSION = "5"

#: Python minor versions we build wheels / environments for ("a wheel for every
#: Python version, same for environments"). Pure-python projects collapse to a
#: single ``py3-none-any`` wheel reused across all of them.
SUPPORTED_PYTHONS: "tuple[str, ...]" = ("3.10", "3.11", "3.12", "3.13")

#: Known serverless environment-version ↔ Python map. Configurable; a Python not
#: listed here resolves to the latest (:data:`SERVERLESS_ENVIRONMENT_VERSION`).
SERVERLESS_ENVIRONMENT_VERSIONS: "dict[str, str]" = {"3.10": "1", "3.11": "2"}

#: Linux-x86_64 manylinux platform tags the Databricks serverless / classic
#: compute can install. The bundled dependency closure is **pinned** to these so
#: it's platform-correct no matter what OS/arch deploys it — a macOS-arm64 or
#: linux-aarch64 host would otherwise emit wheels the compute rejects with
#: ``ERROR_WHEEL_INSTALLATION`` (platform tag mismatch). ``manylinux2014``
#: (glibc 2.17) is the broadly-shipped baseline; ``manylinux_2_28`` (glibc 2.28)
#: covers packages that only publish a newer tag (e.g. recent ``pyarrow``) and
#: still loads on every serverless runtime (glibc ≥ 2.31). Override via the
#: ``YGG_DATABRICKS_WHEEL_PLATFORMS`` env var (comma-separated) for a runtime
#: that needs a different/newer baseline.
SERVERLESS_WHEEL_PLATFORMS: "tuple[str, ...]" = (
    "manylinux2014_x86_64",
    "manylinux_2_28_x86_64",
)


def _serverless_wheel_platforms() -> "list[str]":
    raw = os.environ.get("YGG_DATABRICKS_WHEEL_PLATFORMS")
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    return list(SERVERLESS_WHEEL_PLATFORMS)


#: Distributions never shipped in the zero-PyPI dependency closure — the
#: Databricks runtime already provides them, and re-bundling our own copy has
#: broken the base-environment install (``certifi`` is pip-installed into the
#: serverless image; a second copy from the bundle conflicts). Resolved
#: transitively by pip from the runtime/index instead. Normalized (PEP 503)
#: names; override via ``YGG_DATABRICKS_BUNDLE_EXCLUDE`` (comma-separated).
BUNDLE_EXCLUDE: "frozenset[str]" = frozenset({"certifi"})


def _bundle_exclude() -> "set[str]":
    raw = os.environ.get("YGG_DATABRICKS_BUNDLE_EXCLUDE")
    if raw:
        return {_norm(p.strip()) for p in raw.split(",") if p.strip()}
    return set(BUNDLE_EXCLUDE)


def _py_minor(python: "str | None" = None) -> str:
    """Normalize a Python version to ``"3.X"`` (defaults to the local interpreter;
    accepts ``"3.11"``, ``"311"``, ``"py311"``, ``"3.11.7"``)."""
    if python is None:
        return f"3.{sys.version_info[1]}"
    digits = re.sub(r"[^0-9.]", "", python)
    parts = digits.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    if digits.startswith("3") and len(digits) >= 3:  # "311" → "3.11"
        return f"3.{digits[1:]}"
    return digits or f"3.{sys.version_info[1]}"


def serverless_environment_version(python: "str | None" = None) -> str:
    """The Databricks serverless environment version whose runtime **Python
    matches** *python* (default: the local interpreter).

    Matching matters twice over: a locally-built ygg wheel installs cleanly, and
    Python UDFs run (Spark Connect requires the client and server to share a
    minor Python version). Mapping comes from :data:`SERVERLESS_ENVIRONMENT_VERSIONS`
    (3.10 → ``"1"``, 3.11 → ``"2"``); anything else → the latest
    :data:`SERVERLESS_ENVIRONMENT_VERSION` (``"5"``)."""
    return SERVERLESS_ENVIRONMENT_VERSIONS.get(_py_minor(python), SERVERLESS_ENVIRONMENT_VERSION)


def environment_key_for(python: str) -> str:
    """The serverless ``environment_key`` for a Python version (``3.11`` →
    ``"py311"``)."""
    return "py" + _py_minor(python).replace(".", "")


def wheel_for_python(wheels: "list", python: "str | None" = None) -> str:
    """Pick the wheel matching *python* from *wheels* (paths/str): a version-tagged
    ``cp3XX`` build if present, else the universal ``py3-none-any`` wheel (a
    pure-python project), else the first. Returns a string path."""
    tag = "cp" + _py_minor(python).replace(".", "")
    items = [str(w) for w in wheels]
    return next(
        (w for w in items if tag in w),
        next((w for w in items if "-py3-none-any.whl" in w), items[0] if items else ""),
    )


def is_editable_install(dist: str) -> bool:
    """True when *dist* is installed in **editable / development** mode (``pip``
    or ``uv pip install -e``).

    Editable installs change under a fixed version, so their built wheel is sent
    to a per-user folder (:func:`user_pypi_dir`) and rebuilt on every deploy —
    rather than cached+shared in the workspace registry, where a stale build for
    the same version would shadow fresh code. The signal is ``direct_url.json``'s
    ``dir_info.editable`` (written by pip/uv for ``-e`` installs); an
    ``__editable__`` finder/``.pth`` is the fallback."""
    try:
        d = ilmd.distribution(dist)
    except ilmd.PackageNotFoundError:
        return False
    raw = d.read_text("direct_url.json")
    if raw:
        try:
            info = json.loads(raw)
        except ValueError:
            info = {}
        dir_info = info.get("dir_info")
        if isinstance(dir_info, dict) and dir_info.get("editable"):
            return True
    for f in d.files or []:
        if f.name.startswith("__editable__"):
            return True
    return False


def _workspace_text_unchanged(path: Any, body: str) -> bool:
    """``True`` when *path* exists and already holds exactly *body*.

    Lets the env-config writers (:func:`ensure_named_environment`,
    :func:`ensure_cluster_requirements`) be a true upsert — overwrite only when
    the content **differs** — so re-running a deploy doesn't churn unchanged
    ``.yml`` / ``.requirements.txt`` files. Any read failure (missing file,
    transient API error) is treated as "changed" so the write still happens."""
    try:
        if not path.exists():
            return False
        return path.read_text() == body
    except Exception:  # noqa: BLE001 — unreadable → fall through and (re)write
        return False


def user_pypi_dir(client: Any) -> str:
    """The current user's private PyPI-like wheel folder
    (``/Workspace/Users/<me>/pypi``) — where **editable / dev** builds land so a
    developer's iterations don't collide with others in the shared registry."""
    user = client.workspace_client().current_user.me().user_name
    return f"/Workspace/Users/{user}/pypi"


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _wheel_dist(filename: str) -> str:
    """The PEP 503-normalized distribution folder for a wheel *filename* — its
    first ``-``-delimited component, normalized (``databricks_sdk-0.114.0-...``
    → ``databricks-sdk``). Names the per-distribution folder a wheel lands in
    under the workspace pypi registry."""
    return _norm(Path(filename).name.split("-", 1)[0])


def _norm_version(version: str) -> str:
    """Collapse a version to a comparison key tolerant of wheel-filename
    escaping (``.``, ``_``, ``+`` all become ``-``) so ``0.8.45`` matches the
    ``0.8.45`` component of ``ygg-0.8.45-py3-none-any.whl``."""
    return re.sub(r"[^a-z0-9]+", "-", version.lower())


def distribution_for(package: str) -> str:
    """The distribution (pip) name providing the import *package* (``yggdrasil``
    → ``ygg``). Falls back to *package* itself when unmapped."""
    dists = ilmd.packages_distributions().get(package)
    return dists[0] if dists else package


def import_packages_for(dist: str) -> list[str]:
    """The top-level import packages a distribution provides — the inverse of
    :func:`distribution_for` (``ygg`` → ``["yggdrasil"]``). Empty when *dist*
    is not an installed distribution or ships no ``top_level.txt``."""
    try:
        top = ilmd.distribution(dist).read_text("top_level.txt")
    except ilmd.PackageNotFoundError:
        return []
    return [line.strip() for line in (top or "").splitlines() if line.strip()]


def _project_dependencies(dist: str, extras: "set[str]") -> list[str]:
    """Base requirements + those gated by the requested *extras* (flattened),
    dropping other-extra-only deps."""
    out: list[str] = []
    for req in ilmd.requires(dist) or []:
        head, _, marker = req.partition(";")
        head, marker = head.strip(), marker.strip()
        extra_match = re.search(r'extra\s*==\s*["\']([^"\']+)["\']', marker)
        if extra_match is None:
            out.append(req)              # base dep (keep any non-extra marker)
        elif extra_match.group(1) in extras:
            out.append(head)             # requested extra → flatten in
    return out


def _console_scripts(dist: str) -> dict[str, str]:
    """``{entry-point name: module:attr}`` console scripts of *dist*."""
    out: dict[str, str] = {}
    for ep in ilmd.entry_points(group="console_scripts"):
        ep_dist = getattr(ep, "dist", None)
        if ep_dist is None or _norm(ep_dist.name) == _norm(dist):
            out[ep.name] = ep.value
    return out


def synthesize_project(
    name: str,
    *,
    extras: "tuple[str, ...] | list[str]" = (),
    dest_dir: "str | Path | None" = None,
) -> Path:
    """Create a buildable project from the **installed** package — copy its
    on-disk files and write a ``pyproject.toml`` reconstructed from the
    distribution metadata (version, console scripts, dependencies incl. the
    requested *extras*). Returns the project dir.

    *name* may be the import package (``yggdrasil``) or the distribution /
    pip name (``ygg``) — both resolve to the same project. An import name
    is used directly; a distribution name is resolved to its top-level
    import package via :func:`import_packages_for`."""
    try:
        module = importlib.import_module(name)
        package, dist = name, distribution_for(name)
    except ModuleNotFoundError:
        # ``name`` is a distribution (pip) name, not an importable package —
        # resolve the import package it ships and build that.
        packages = import_packages_for(name)
        if not packages:
            raise
        package, dist = packages[0], name
        module = importlib.import_module(package)
    # A regular package exposes ``__file__`` (its ``__init__``); an editable /
    # namespace package served by a finder may not — fall back to the ``__path__``
    # entry that actually holds the package's ``__init__`` (an editable finder can
    # also surface the *project root*, whose basename matches — copying that would
    # double-nest the package, so the ``__init__`` check is what disambiguates).
    pkg_file = getattr(module, "__file__", None)
    if pkg_file:
        pkg_dir = Path(pkg_file).resolve().parent
    else:
        candidates = [Path(p).resolve() for p in getattr(module, "__path__", []) or []]
        pkg_dir = next(
            (p for p in candidates if (p / "__init__.py").exists()),
            candidates[0] if candidates else None,
        )
        if pkg_dir is None:
            raise ModuleNotFoundError(f"cannot locate on-disk files for package {package!r}")
    # An editable finder can hand back the *project root* (no ``__init__`` here, but
    # ``<root>/<package>/__init__.py`` one level down) — descend so the copy below
    # doesn't double-nest the package (``pkg/pkg/__init__.py``).
    if not (pkg_dir / "__init__.py").exists() and (pkg_dir / package / "__init__.py").exists():
        pkg_dir = pkg_dir / package
    meta = ilmd.metadata(dist)

    out = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="ygg-synth-"))
    shutil.copytree(pkg_dir, out / package, dirs_exist_ok=True)

    deps = _project_dependencies(dist, set(extras))
    scripts = _console_scripts(dist)
    (out / "pyproject.toml").write_text(
        _render_pyproject(meta["Name"], meta["Version"], package, deps, scripts)
    )
    logger.info("synthesized project for %s (%s) at %s", package, dist, out)
    return out


def _render_pyproject(name: str, version: str, package: str, deps: list[str], scripts: dict[str, str]) -> str:
    dep_block = "\n".join(f'  "{d}",' for d in deps)
    script_block = "\n".join(f'{k} = "{v}"' for k, v in scripts.items())
    return (
        "[build-system]\n"
        'requires = ["setuptools>=61"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[project]\n"
        f'name = "{name}"\n'
        f'version = "{version}"\n'
        "dependencies = [\n"
        f"{dep_block}\n"
        "]\n\n"
        "[project.scripts]\n"
        f"{script_block}\n\n"
        "[tool.setuptools.packages.find]\n"
        f'include = ["{package}*"]\n'
    )


def build_wheel(
    package: str,
    *,
    extras: "tuple[str, ...] | list[str]" = (),
    requirements: "tuple[str, ...] | list[str]" = (),
    dest_dir: "str | Path | None" = None,
    no_deps: bool = False,
) -> list[Path]:
    """Build the live *package* (synthesized project) via an isolated
    ``pip wheel`` — returns the produced ``.whl`` files.

    With ``no_deps=True`` builds **only the project wheel** (a pure-python
    ``py3-none-any`` wheel, no platform-specific dependency wheels) — what the
    ygg image ships, since deps resolve from the index on the cluster. This is
    built with **uv** (``uv build --wheel``; no separate pip needed), falling
    back to ``pip wheel --no-deps`` only if uv isn't on PATH.

    With ``no_deps=False`` (legacy) the project is built **with its
    dependencies** + any extra *requirements* via ``pip wheel`` — uv build
    doesn't bundle dependencies."""
    project = synthesize_project(package, extras=extras)
    out = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="ygg-wheel-"))

    if no_deps and not requirements:
        logger.info("building wheel for %s into %s (uv)", package, out)
        try:
            _run_build(["uv", "build", "--wheel", "--out-dir", str(out), str(project)])
        except FileNotFoundError:
            logger.info("uv not found — falling back to pip for %s", package)
            _run_build(
                [sys.executable, "-m", "pip", "wheel", str(project),
                 "--no-deps", "--wheel-dir", str(out)]
            )
    else:
        logger.info("building wheel (+ dependencies) for %s into %s (pip)", package, out)
        # Bundling the dependency closure needs pip (uv build can't); uv-created
        # venvs ship without it, so bootstrap one in-place via ensurepip first.
        if subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
        ).returncode != 0:
            logger.info("pip not present in %s — bootstrapping via ensurepip", sys.executable)
            _run_build([sys.executable, "-m", "ensurepip", "--upgrade"])
        _run_build(
            [sys.executable, "-m", "pip", "wheel", str(project), *requirements,
             "--wheel-dir", str(out)]
        )

    wheels = sorted(out.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"no wheels produced in {out}")
    return wheels


def build_wheels_for_versions(
    package: str,
    *,
    versions: "tuple[str, ...] | list[str]" = SUPPORTED_PYTHONS,
    extras: "tuple[str, ...] | list[str]" = (),
    dest_dir: "str | Path | None" = None,
) -> list[Path]:
    """Build *package* **once per Python version** (``uv build --python X.Y``) and
    return the unique wheels.

    A pure-python project yields a single ``py3-none-any`` wheel — built once and
    reused for every version (we stop after the first universal wheel). A package
    with native extensions yields a distinct ``cp3XX`` wheel per Python, so the
    registry carries a wheel for every version. Needs ``uv`` (it downloads the
    requested interpreters); without it, falls back to one wheel for the current
    interpreter."""
    project = synthesize_project(package, extras=extras)
    out = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="ygg-wheels-"))
    seen: set[str] = set()
    wheels: list[Path] = []
    for version in versions:
        try:
            _run_build(
                ["uv", "build", "--wheel", "--python", version,
                 "--out-dir", str(out), str(project)]
            )
        except FileNotFoundError:
            logger.info("uv not found — building one wheel for the current interpreter")
            _run_build(
                [sys.executable, "-m", "pip", "wheel", str(project),
                 "--no-deps", "--wheel-dir", str(out)]
            )
        for whl in sorted(out.glob("*.whl")):
            if whl.name not in seen:
                seen.add(whl.name)
                wheels.append(whl)
        # A universal wheel is identical for every Python — no need to rebuild.
        if any(n.endswith("-py3-none-any.whl") for n in seen):
            break
    if not wheels:
        raise FileNotFoundError(f"no wheels produced in {out}")
    return wheels


def build_bundle(
    package: str,
    *,
    python: "str | None" = None,
    extras: "tuple[str, ...] | list[str]" = ("databricks",),
    dest_dir: "str | Path | None" = None,
) -> list[Path]:
    """Build *package*'s project wheel **plus its whole dependency closure** as
    wheels for a single Python version — the input to a *zero-PyPI* serverless /
    cluster environment.

    The closure must install on Databricks compute — **Linux x86_64** — no matter
    what OS/arch deploys it. So the two halves are fetched independently of the
    host:

    * the **project wheel** is built with ``uv build --wheel --python X.Y`` (or a
      ``pip wheel --no-deps`` fallback); pure-python ``ygg`` comes out
      ``py3-none-any``, host-independent by construction.
    * the **dependency closure** is *downloaded* — not built — via ``pip
      download --only-binary=:all:`` with explicit ``--python-version`` /
      ``--implementation cp`` / ``--abi`` / ``--platform`` tags
      (:data:`SERVERLESS_WHEEL_PLATFORMS`), so pip pulls the manylinux wheels the
      serverless runtime installs even from a macOS-arm64 / linux-aarch64 box.
      pip auto-includes universal (``py3-none-any`` / ``abi3``) wheels for the
      pure-python deps.

    The previous "``pip wheel`` inside a uv venv" path built the deps for the
    *deploying host's* platform, so a non-Linux-x86_64 host poisoned the bundle
    with wheels the compute rejects (``ERROR_WHEEL_INSTALLATION``)."""
    project = synthesize_project(package, extras=extras)
    out = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="ygg-bundle-"))
    py = _py_minor(python)
    abi = "cp" + py.replace(".", "")          # 3.12 → cp312

    logger.info("building %s project wheel for Python %s into %s", package, py, out)
    try:
        _run_build(
            ["uv", "build", "--wheel", "--python", py,
             "--out-dir", str(out), str(project)]
        )
    except FileNotFoundError:
        logger.info("uv not found — building the project wheel with the host interpreter")
        _run_build(
            [sys.executable, "-m", "pip", "wheel", str(project),
             "--no-deps", "--wheel-dir", str(out)]
        )

    # Download the dependency closure as Linux-x86_64 wheels for the *serverless
    # runtime* (the target Python + manylinux), not the deploying host. pip
    # resolves the transitive closure from the top-level requirements and only
    # accepts wheels (``--only-binary=:all:`` is mandatory alongside ``--platform``).
    deps = _project_dependencies(distribution_for(package), set(extras))
    if deps:
        platform_args: list[str] = []
        for platform_tag in _serverless_wheel_platforms():
            platform_args += ["--platform", platform_tag]
        logger.info(
            "downloading %d top-level dep requirement(s) as %s wheels (py%s)",
            len(deps), "/".join(_serverless_wheel_platforms()), py,
        )
        _run_build(
            [sys.executable, "-m", "pip", "download",
             "--only-binary=:all:",
             "--python-version", py,
             "--implementation", "cp",
             "--abi", abi, "--abi", "abi3", "--abi", "none",
             *platform_args,
             "--dest", str(out), *deps]
        )

    # Drop runtime-provided distributions (e.g. ``certifi``) so the bundle never
    # ships a second copy the base-environment install conflicts on — pip
    # resolves them from the serverless image / index at install time.
    exclude = _bundle_exclude()
    wheels: list[Path] = []
    for whl in sorted(out.glob("*.whl")):
        if _wheel_dist(whl.name) in exclude:
            logger.info("excluding runtime-provided %s from bundle", whl.name)
            whl.unlink(missing_ok=True)
            continue
        wheels.append(whl)
    if not wheels:
        raise FileNotFoundError(f"no wheels produced in {out}")
    return wheels


def upload_wheel(client: Any, wheel: "str | Path", *, workspace_dir: str = WORKSPACE_WHL_DIR) -> str:
    """Upload *wheel* to *workspace_dir*; return its workspace path."""
    from yggdrasil.databricks.path import DatabricksPath

    wheel = Path(wheel)
    dest = f"{workspace_dir.rstrip('/')}/{wheel.name}"
    path = DatabricksPath.from_(dest, client=client)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wheel.read_bytes())
    logger.info("uploaded wheel to %s", dest)
    return dest


def registry_upload(
    client: Any,
    wheel: "str | Path",
    *,
    workspace_dir: str = WORKSPACE_PYPI_DIR,
    overwrite: bool = False,
) -> str:
    """Upload *wheel* into its **own distribution folder** under the PyPI-like
    registry root — ``<workspace_dir>/<dist>/<wheel>`` — so the registry stays a
    browsable PEP 503 "simple index" (one folder per distribution, versions
    side-by-side), rather than a flat per-image bundle directory.

    Dependency wheels are version-pinned and immutable, so an already-present
    target is left untouched and its path returned (the upload is skipped) —
    unless *overwrite*, which the caller sets for a project's **own** wheel whose
    code can change under a fixed version (an editable install)."""
    from yggdrasil.databricks.path import DatabricksPath

    wheel = Path(wheel)
    dest = f"{workspace_dir.rstrip('/')}/{_wheel_dist(wheel.name)}/{wheel.name}"
    path = DatabricksPath.from_(dest, client=client)
    if not overwrite and path.exists():
        logger.info("reusing deployed wheel %s", dest)
        return dest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wheel.read_bytes())
    logger.info("uploaded wheel to %s", dest)
    return dest


def ensure_wheel(
    client: Any,
    package: str,
    *,
    workspace_dir: str = WORKSPACE_WHL_DIR,
    extras: "tuple[str, ...] | list[str]" = (),
    requirements: "tuple[str, ...] | list[str]" = (),
    no_deps: bool = False,
) -> list[str]:
    """Build the live *package* (:func:`build_wheel`) and upload every produced
    wheel to *workspace_dir*; return their workspace paths. ``no_deps=True``
    builds only the pure-python project wheel (deps resolve at install time on
    the target). Built fresh each call so the deployed job ships current code."""
    wheels = build_wheel(
        package, extras=extras, requirements=requirements, no_deps=no_deps,
    )
    return [upload_wheel(client, w, workspace_dir=workspace_dir) for w in wheels]


def ensure_wheels(
    client: Any,
    package: str,
    *,
    versions: "tuple[str, ...] | list[str]" = SUPPORTED_PYTHONS,
    workspace_dir: str = WORKSPACE_WHL_DIR,
    extras: "tuple[str, ...] | list[str]" = (),
) -> list[str]:
    """Build the live *package* **for every Python version** (:func:`build_wheels_for_versions`)
    and upload every produced wheel to *workspace_dir*; return their workspace
    paths. Pure-python packages collapse to a single ``py3-none-any`` wheel. Built
    fresh each call so the deployed job ships current code."""
    wheels = build_wheels_for_versions(package, versions=versions, extras=extras)
    return [upload_wheel(client, w, workspace_dir=workspace_dir) for w in wheels]


def deployed_wheels(
    client: Any,
    dist: str,
    version: str,
    *,
    workspace_dir: str,
    dist_only: bool = False,
) -> list[str]:
    """Workspace paths of wheels already deployed for *dist* *version* under
    *workspace_dir*, or ``[]`` when *dist*'s own wheel for *version* is absent.

    The deploy counts as present only when *dist*'s wheel for *version* is there
    — a directory holding just dependency wheels (a never-built or half-finished
    upload) is treated as absent so the caller rebuilds. With ``dist_only=True``
    only *dist*'s own wheel(s) are returned (the current pure-python image is a
    single wheel; deps resolve from the index at install). Otherwise every
    ``.whl`` in the directory is returned (legacy full bundles)."""
    from yggdrasil.databricks.path import DatabricksPath

    folder = DatabricksPath.from_(workspace_dir, client=client)
    if not folder.exists():
        return []

    paths: list[str] = []
    dist_paths: list[str] = []
    want_dist, want_version = _norm(dist), _norm_version(version)
    for child in folder.iterdir():
        name = str(child.name)
        if not name.endswith(".whl"):
            continue
        full = child.full_path()
        paths.append(full)
        parts = name[:-4].split("-")  # drop ".whl"; <dist>-<version>-<tags...>
        if (
            len(parts) >= 2
            and _norm(parts[0]) == want_dist
            and _norm_version(parts[1]) == want_version
        ):
            dist_paths.append(full)
    if not dist_paths:
        return []
    return dist_paths if dist_only else paths


def ensure_ygg_wheel(
    client: Any,
    *,
    workspace_dir: str = WORKSPACE_PYPI_DIR,
    rebuild: bool = False,
) -> list[str]:
    """Get-or-build the **pure-python ygg wheel** for the current version.

    Builds *only* the live ``yggdrasil`` package as a ``py3-none-any`` wheel
    (uv ``build --wheel``) — no platform-specific dependency wheels — and deploys
    it into the PyPI-like registry under *workspace_dir*, in the distribution's
    own folder (``<workspace_dir>/ygg/ygg-<version>-py3-none-any.whl``) alongside
    any other versions. On the first call for a version the wheel is built and
    uploaded; later calls find and reuse it (:func:`deployed_wheels`). Pass
    ``rebuild=True`` to force a fresh build.

    Returns the workspace path of the ygg wheel — a serverless job installs it
    **by path** while resolving the runtime dependencies (see
    :func:`ygg_environment`) from the workspace index, so they land as
    platform-correct builds rather than wheels bundled from the deploying host
    (which a different serverless platform / python can't install)."""
    wheels = ensure_ygg_wheels(client, workspace_dir=workspace_dir, rebuild=rebuild)
    return [wheel_for_python(wheels)]


def ensure_ygg_wheels(
    client: Any,
    *,
    versions: "tuple[str, ...] | list[str]" = SUPPORTED_PYTHONS,
    workspace_dir: str = WORKSPACE_PYPI_DIR,
    rebuild: bool = False,
) -> list[str]:
    """Get-or-build the live ``yggdrasil`` wheel **for every Python version** and
    deploy them all into the PyPI-like registry under ``<workspace_dir>/ygg/``.

    Pure-python ygg yields one ``py3-none-any`` wheel reused across versions; a
    native build would yield a ``cp3XX`` wheel each. On the first call for a
    version the wheels are built and uploaded; later calls find and reuse them
    (:func:`deployed_wheels`). ``rebuild=True`` forces a fresh build.

    Returns the workspace paths of all ygg wheels — a serverless env installs the
    one matching its Python **by path** (:func:`wheel_for_python`) while resolving
    runtime deps from the index (see :func:`ygg_environment`)."""
    version = ilmd.version("ygg")
    # PyPI-like: one folder per distribution; versions/tags are distinct files.
    dist_dir = f"{workspace_dir.rstrip('/')}/ygg"

    if not rebuild:
        existing = deployed_wheels(
            client, "ygg", version, workspace_dir=dist_dir, dist_only=True,
        )
        if existing:
            logger.info("reusing %d deployed ygg %s wheel(s) at %s", len(existing), version, dist_dir)
            return existing
        logger.info("no ygg %s wheel at %s — building for %s", version, dist_dir, list(versions))

    wheels = build_wheels_for_versions("ygg", versions=versions, extras=("databricks",))
    return [upload_wheel(client, w, workspace_dir=dist_dir) for w in wheels]


def ensure_bundle(
    client: Any,
    package: str = "ygg",
    *,
    python: "str | None" = None,
    extras: "tuple[str, ...] | list[str]" = ("databricks",),
    workspace_dir: str = WORKSPACE_PYPI_DIR,
    rebuild: bool = False,
) -> list[str]:
    """Build *package* **with its whole transitive dependency closure** for one
    Python version (default: the local interpreter) and deploy every wheel into
    the workspace pypi registry; return their workspace paths, project wheel
    first.

    Where :func:`ensure_ygg_wheels` ships only the project wheel (deps resolve
    from the index at install), this bundles everything — so a serverless /
    cluster environment that lists these wheel paths installs **entirely from
    them, with zero PyPI access** ("0 pip install"). The compiled dependency
    wheels carry the **serverless runtime's** Linux-x86_64 manylinux + *python*
    ``cp3XX`` tags (:func:`build_bundle`), independent of the deploying host.

    Each wheel lands in its **own distribution folder**
    (``<workspace_dir>/<dist>/<wheel>`` via :func:`registry_upload`) — the
    PEP 503 simple-index layout the registry already uses for the ygg wheel, so
    dependency wheels are shared across images and versions sit side-by-side (no
    per-image ``<dist>-bundle/`` directory).

    Cached per ``(dist, version, python)`` by a small ``.bundle`` manifest beside
    the project wheel (the dep wheels are scattered across distribution folders,
    so the manifest records the exact set): a bundle whose manifest is present
    and whose project wheel still exists is reused unless *rebuild*. Uploads are
    incremental — immutable dependency wheels already in the registry are reused;
    only the (small) project wheel is re-uploaded."""
    from yggdrasil.databricks.path import DatabricksPath

    dist = distribution_for(package)
    version = ilmd.version(dist)
    proj = _norm(dist)
    py = _py_minor(python)
    root = workspace_dir.rstrip("/")
    # The manifest stem carries a ``-linux_x86_64`` platform scheme so a bundle
    # built by the *old* host-platform code path (whose manifest lacked it) is
    # never reused — the first deploy after the platform fix rebuilds the closure
    # with serverless-correct manylinux wheels instead of silently serving the
    # poisoned cache.
    manifest = f"{root}/{proj}/{proj}-{version}-{environment_key_for(py)}-linux_x86_64.bundle"
    mpath = DatabricksPath.from_(manifest, client=client)

    if not rebuild and mpath.exists():
        paths = [ln.strip() for ln in mpath.read_text().splitlines() if ln.strip()]
        if paths and DatabricksPath.from_(paths[0], client=client).exists():
            logger.info(
                "reusing %d-wheel %s %s bundle for Python %s", len(paths), dist, version, py,
            )
            return paths

    logger.info("building %s %s bundle (project + deps) for Python %s -> %s", dist, version, py, root)
    wheels = build_bundle(package, python=py, extras=extras)
    # Project wheel(s) first (so the env lists ygg first), then the deps sorted.
    project_wheels = sorted(w for w in wheels if _wheel_dist(w.name) == proj)
    dep_wheels = sorted(w for w in wheels if _wheel_dist(w.name) != proj)
    paths = [
        registry_upload(client, w, workspace_dir=root, overwrite=True)
        for w in project_wheels
    ]
    paths += [registry_upload(client, w, workspace_dir=root) for w in dep_wheels]

    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text("\n".join(paths) + "\n")
    return paths


def ensure_bundles(
    client: Any,
    package: str = "ygg",
    *,
    versions: "tuple[str, ...] | list[str]" = SUPPORTED_PYTHONS,
    extras: "tuple[str, ...] | list[str]" = ("databricks",),
    workspace_dir: str = WORKSPACE_PYPI_DIR,
    rebuild: bool = False,
) -> "dict[str, list[str]]":
    """:func:`ensure_bundle` **for every Python version** — build + deploy a
    zero-PyPI wheel closure per Python and return ``{python: wheel paths}``.

    The pure-python project wheel is shared across versions (deployed once); only
    the compiled dependency wheels differ per Python (distinct ``cp3XX`` tags),
    each landing in its distribution folder. Feed each version's list to a
    matching serverless ``base_environment`` / cluster requirements file (one per
    Python — see the ``seed`` command)."""
    return {
        version: ensure_bundle(
            client, package, python=version, extras=extras,
            workspace_dir=workspace_dir, rebuild=rebuild,
        )
        for version in versions
    }


def build_project_wheel(
    project_dir: str | Path,
    *,
    python: str | None = None,
    dest_dir: str | Path | None = None,
) -> list[Path]:
    """Build the **on-disk project** at *project_dir* into a wheel (``uv build
    --wheel``, ``--python X.Y`` when given; ``pip wheel --no-deps`` fallback).

    Unlike :func:`build_wheel` — which synthesizes a project from an *installed*
    package's metadata — this builds the real discovered project from its own
    ``pyproject.toml``, so a user's source tree ships exactly as written."""
    project_dir = Path(project_dir).resolve()
    out = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="ygg-proj-"))
    cmd = ["uv", "build", "--wheel"]
    if python:
        cmd += ["--python", python]
    cmd += ["--out-dir", str(out), str(project_dir)]
    try:
        _run_build(cmd)
    except FileNotFoundError:
        logger.info("uv not found — building project %s with pip", project_dir)
        _run_build(
            [sys.executable, "-m", "pip", "wheel", str(project_dir),
             "--no-deps", "--wheel-dir", str(out)]
        )
    wheels = sorted(out.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"no wheel produced for project {project_dir}")
    return wheels


def download_dependency_wheels(
    dependencies: list[str] | tuple[str, ...],
    *,
    python: str | None = None,
    dest_dir: str | Path | None = None,
) -> list[Path]:
    """Download *dependencies* as **Linux-x86_64** wheels for the serverless /
    cluster runtime (the same platform pins :func:`build_bundle` uses), so a
    zero-PyPI environment can install entirely from them. Runtime-provided
    distributions (:func:`_bundle_exclude`) are dropped. Empty in → empty out."""
    deps = [d for d in dependencies if d]
    if not deps:
        return []
    out = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="ygg-deps-"))
    py = _py_minor(python)
    abi = "cp" + py.replace(".", "")                  # 3.12 → cp312
    platform_args: list[str] = []
    for platform_tag in _serverless_wheel_platforms():
        platform_args += ["--platform", platform_tag]
    logger.info("downloading %d dependency requirement(s) as linux wheels (py%s)", len(deps), py)
    _run_build(
        [sys.executable, "-m", "pip", "download", "--only-binary=:all:",
         "--python-version", py, "--implementation", "cp",
         "--abi", abi, "--abi", "abi3", "--abi", "none",
         *platform_args, "--dest", str(out), *deps]
    )
    exclude = _bundle_exclude()
    wheels: list[Path] = []
    for whl in sorted(out.glob("*.whl")):
        if _wheel_dist(whl.name) in exclude:
            whl.unlink(missing_ok=True)
            continue
        wheels.append(whl)
    return wheels


class Wheels(DatabricksService):
    """Build, upload, and browse wheels in the workspace PyPI-like registry.

    The OO front door (``dbc.wheels``) over this module's functions. CRUD verbs
    return :class:`Wheel` handles; the heavy build/upload work lives in the
    module-level functions they delegate to.
    """

    #: Default registry root (overridable per call via ``workspace_dir=``).
    default_dir: ClassVar[str] = WORKSPACE_PYPI_DIR

    def _wheel(self, path: str, *, dist: Optional[str] = None,
               version: Optional[str] = None) -> Wheel:
        return Wheel(self, path=path, dist=dist, version=version)

    # -- build (local) -----------------------------------------------------
    def build(
        self,
        package: str = "ygg",
        *,
        extras: "tuple[str, ...] | list[str]" = (),
        requirements: "tuple[str, ...] | list[str]" = (),
        no_deps: bool = False,
        all_versions: bool = False,
        dest_dir: "str | Path | None" = None,
    ) -> list[Path]:
        """Build *package*'s wheel(s) on disk (no upload) — one per supported
        Python with *all_versions*, else a single wheel for this interpreter."""
        if all_versions:
            return build_wheels_for_versions(package, extras=extras, dest_dir=dest_dir)
        return build_wheel(
            package, extras=extras, requirements=requirements,
            no_deps=no_deps, dest_dir=dest_dir,
        )

    # -- upload / deploy ---------------------------------------------------
    def upload(
        self,
        wheel: "str | Path",
        *,
        workspace_dir: Optional[str] = None,
        registry: bool = True,
        overwrite: bool = False,
    ) -> Wheel:
        """Upload a prebuilt *wheel* — into its ``<dist>/`` registry folder
        (default), or flat under *workspace_dir* with ``registry=False``."""
        root = workspace_dir or self.default_dir
        if registry:
            dest = registry_upload(self.client, wheel, workspace_dir=root, overwrite=overwrite)
        else:
            dest = upload_wheel(self.client, wheel, workspace_dir=root)
        return self._wheel(dest)

    def deploy(
        self,
        package: str = "ygg",
        *,
        extras: "tuple[str, ...] | list[str]" = (),
        requirements: "tuple[str, ...] | list[str]" = (),
        no_deps: bool = False,
        all_versions: bool = False,
        workspace_dir: Optional[str] = None,
    ) -> list[Wheel]:
        """Build the live *package* and upload every produced wheel; return the
        :class:`Wheel` handles. *all_versions* builds one wheel per supported
        Python (the matrix builder)."""
        root = workspace_dir or self.default_dir
        if all_versions:
            paths = ensure_wheels(self.client, package, workspace_dir=root, extras=extras)
        else:
            paths = ensure_wheel(
                self.client, package, workspace_dir=root, extras=extras,
                requirements=requirements, no_deps=no_deps,
            )
        return [self._wheel(p) for p in paths]

    def deploy_ygg(
        self,
        *,
        all_versions: bool = True,
        rebuild: bool = False,
        workspace_dir: Optional[str] = None,
    ) -> list[Wheel]:
        """Get-or-build the versioned ygg image wheel(s). *all_versions* (default)
        covers every supported Python; else just this interpreter's."""
        root = workspace_dir or self.default_dir
        if all_versions:
            paths = ensure_ygg_wheels(self.client, workspace_dir=root, rebuild=rebuild)
        else:
            paths = ensure_ygg_wheel(self.client, workspace_dir=root, rebuild=rebuild)
        return [self._wheel(p) for p in paths]

    def bundle(
        self,
        package: str = "ygg",
        *,
        python: Optional[str] = None,
        extras: "tuple[str, ...] | list[str]" = ("databricks",),
        all_versions: bool = False,
        rebuild: bool = False,
        workspace_dir: Optional[str] = None,
    ) -> "list[Wheel] | dict[str, list[Wheel]]":
        """Build + deploy *package*'s whole zero-PyPI dependency closure. Returns
        ``{python: [Wheel, …]}`` with *all_versions*, else a flat ``[Wheel, …]``."""
        root = workspace_dir or self.default_dir
        if all_versions:
            mapping = ensure_bundles(self.client, package, extras=extras,
                                     workspace_dir=root, rebuild=rebuild)
            return {py: [self._wheel(p) for p in paths] for py, paths in mapping.items()}
        paths = ensure_bundle(self.client, package, python=python, extras=extras,
                              workspace_dir=root, rebuild=rebuild)
        return [self._wheel(p) for p in paths]

    # -- read --------------------------------------------------------------
    def deployed(
        self,
        dist: str,
        version: str,
        *,
        workspace_dir: Optional[str] = None,
        dist_only: bool = False,
    ) -> list[Wheel]:
        """The wheels already deployed for *dist* *version* (empty when *dist*'s
        own wheel for that version is absent)."""
        root = (workspace_dir or self.default_dir).rstrip("/")
        dist_dir = f"{root}/{_norm(distribution_for(dist))}"
        paths = deployed_wheels(self.client, dist, version,
                                workspace_dir=dist_dir, dist_only=dist_only)
        return [self._wheel(p, dist=dist, version=version) for p in paths]

    def list(
        self,
        dist: Optional[str] = None,
        *,
        workspace_dir: Optional[str] = None,
    ) -> "list[Wheel] | list[str]":
        """Browse the registry: the :class:`Wheel`\\ s under *dist*'s folder, or
        the distribution folder names when *dist* is ``None``."""
        from ..path import DatabricksPath

        root = (workspace_dir or self.default_dir).rstrip("/")
        if dist is not None:
            folder = DatabricksPath.from_(f"{root}/{distribution_for(dist)}", client=self.client)
            if not folder.exists():
                return []
            return [self._wheel(c.full_path()) for c in folder.iterdir()
                    if str(c.name).endswith(".whl")]
        registry = DatabricksPath.from_(root, client=self.client)
        if not registry.exists():
            return []
        return [str(c.name) for c in registry.iterdir() if c.is_dir()]

    def get(
        self,
        dist: str,
        *,
        version: Optional[str] = None,
        python: Optional[str] = None,
        workspace_dir: Optional[str] = None,
    ) -> Optional[Wheel]:
        """A single deployed wheel for *dist* — the one matching *python* (or the
        first) at *version* (or any version present). ``None`` when none found."""
        wheels = self.list(dist, workspace_dir=workspace_dir)
        wheels = [w for w in wheels if isinstance(w, Wheel)]
        if version is not None:
            want = _norm_version(version)
            wheels = [w for w in wheels if w.version and _norm_version(w.version) == want]
        if not wheels:
            return None
        chosen = wheel_for_python([w.path for w in wheels], python=python)
        return next((w for w in wheels if w.path == chosen), wheels[0])
