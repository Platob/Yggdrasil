"""Build a wheel from the **live** package on disk and upload it for serverless jobs.

Instead of relying on a published release or a source checkout, the deploy
*synthesizes* a buildable project from the installed package's own files +
metadata (:func:`synthesize_project`) and builds it — so the cluster runs exactly
the code that's running now, whether the package is a dev checkout or pip-installed.

:func:`build_wheel` synthesizes the project, then ``pip wheel`` resolves it **with
its dependencies** into a directory of wheels; :func:`ensure_wheel` uploads them
all and returns their workspace paths (installed by path on the cluster — no index).

(``uv`` has no ``pip wheel`` equivalent, so the dependency build uses ``pip``.)
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from yggdrasil.enums.mode import Mode

logger = logging.getLogger(__name__)


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
    "synthesize_project",
    "build_wheel",
    "build_wheels_for_versions",
    "build_bundle",
    "upload_wheel",
    "registry_upload",
    "ensure_wheel",
    "ensure_wheels",
    "deployed_wheels",
    "ensure_ygg_wheel",
    "ensure_ygg_wheels",
    "ensure_bundle",
    "ensure_bundles",
    "ensure_named_environment",
    "ensure_cluster_requirements",
    "ensure_environment",
    "ensure_environments",
    "find_pyproject",
    "read_pyproject",
    "build_project_wheel",
    "download_dependency_wheels",
    "ensure_project_environment",
    "environment_folder",
    "environment_stem",
    "environment_folder_of",
    "ygg_base_environment_name",
    "deployed_environments",
    "ygg_runtime_dependencies",
    "ygg_environment",
    "ygg_environments",
]

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

#: Where reusable serverless **base environments** live. The layout mirrors the
#: wheel registry (:data:`WORKSPACE_PYPI_DIR`): one folder **per project**, named
#: for the project/dist (no version, no python tag), holding its *version-tagged*
#: spec files + the zero-PyPI wheel closure::
#:
#:     /Workspace/Shared/environment/<proj>/
#:         <proj>-<version>-py3XX.yml             serverless base_environment
#:         <proj>-<version>-py3XX.requirements.txt   classic-cluster requirements
#:
#: e.g. ``environment/ygg/ygg-0.8.57-py311.yml``. Every version / python lands in
#: the **same** ``<proj>/`` folder (the filenames carry the version + ``py3XX``
#: tag), exactly like ``pypi/<dist>/<dist>-<version>-…whl`` — uniform with the
#: wheel creations.
#:
#: A job references the ``.yml`` by file path via ``Environment.base_environment``
#: instead of inlining the whole dependency list (see
#: :func:`ensure_named_environment`); its dependencies are **wheels in the shared
#: pypi registry** (:data:`WORKSPACE_PYPI_DIR`, built by :func:`ensure_environment`),
#: so the env is self-describing and the runtime installs with zero PyPI access.
WORKSPACE_ENV_DIR = "/Workspace/Shared/environment"

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


def environment_folder(project: str = "ygg") -> str:
    """The workspace **folder** an environment lives in: the project/dist name,
    normalized (``yggdrasil`` / ``ygg`` → ``ygg``) — **no version, no python
    tag**, mirroring the wheel registry's ``<dist>/`` folder. Accepts an import
    package or a distribution name."""
    return _norm(distribution_for(project))


def environment_stem(
    project: str = "ygg",
    *,
    python: "str | None" = None,
    version: "str | None" = None,
) -> str:
    """The version-tagged **file stem** of an environment — ``<proj>-<version>-py3XX``
    (e.g. ``ygg-0.8.57-py311``).

    The ``.yml`` / ``.requirements.txt`` files inside the project folder carry
    this stem, uniform with the wheel registry's ``<dist>-<version>-…whl``.
    *version* defaults to the installed *project* version (falling back to the
    in-tree ygg version), *python* to the local interpreter."""
    dist = distribution_for(project)
    if version is None:
        try:
            version = ilmd.version(dist)
        except Exception:  # noqa: BLE001 — fall back to the in-tree version
            from yggdrasil.version import __version__ as version
    return f"{_norm(dist)}-{version}-{environment_key_for(python)}"


def environment_folder_of(stem: str) -> str:
    """The project folder a versioned environment *stem* belongs to: the stem with
    its trailing ``-<version>-py3XX`` stripped (``ygg-0.8.57-py311`` → ``ygg``,
    ``my-proj-1.2.3-py312`` → ``my-proj``).

    A bare name with no ``py3XX`` python-tag suffix (a hand-written named env) is
    returned unchanged, so the old ``<name>/<name>.yml`` layout still resolves."""
    parts = stem.split("-")
    if len(parts) >= 3 and re.fullmatch(r"py3\d+", parts[-1]):
        return "-".join(parts[:-2])
    return stem


def ygg_base_environment_name(python: "str | None" = None) -> str:
    """Canonical name (version-tagged file stem) of the reusable serverless
    **base environment** for the running ygg image — ``ygg-<version>-py3XX``.

    This is exactly the stem ``ygg databricks seed`` writes under
    :data:`WORKSPACE_ENV_DIR` in the project folder
    (``environment/ygg/ygg-<version>-py3XX.yml``), so a job that points its
    ``base_environment_name`` here reuses the seeded, wheel-built image when the
    seed has run — and self-provisions the identical file (same wheel closure,
    same path) when it hasn't. The version-pinned name is the single source of
    truth for "the correct ygg environment", replacing the old static
    ``yellow`` env."""
    return environment_stem("ygg", python=python)


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


def ensure_named_environment(
    client: Any,
    name: str = "yellow",
    *,
    dependencies: "list[str] | tuple[str, ...]",
    environment_version: "str | None" = None,
    workspace_dir: str = WORKSPACE_ENV_DIR,
    filename: "str | None" = None,
) -> str:
    """Create-or-update a reusable serverless **base environment** *name* as a
    YAML file in the workspace and return its path.

    A serverless job can reference this file via
    ``Environment.base_environment`` instead of inlining the whole dependency
    list — so one shared, named environment is defined once and every ygg job
    points at it. The file is the documented serverless env spec; its
    *dependencies* are **built wheels in the workspace pypi registry**
    (:func:`ensure_bundle`) so the runtime installs with zero PyPI access::

        environment_version: '5'
        dependencies:
          - /Workspace/Shared/pypi/ygg/ygg-0.8.54-py3-none-any.whl
          - /Workspace/Shared/pypi/pyarrow/pyarrow-...-cp312-...-.whl

    Here *name* is the **project folder** (e.g. ``ygg``); the file is
    ``<name>.env.yaml`` unless *filename* overrides it — the seed writes a
    version-pinned ``<proj>-<version>-py3XX.yml`` so jobs can point at an exact
    image and every version coexists in the one project folder. Written
    (overwritten) on every call — upsert semantics, so redeploying keeps the
    file pointing at the current image. *dependencies* are wheel workspace paths
    (and/or pip requirement lines, when an index resolve is wanted instead)."""
    from yggdrasil.databricks.path import DatabricksPath

    version = environment_version or serverless_environment_version()
    lines = [f"environment_version: '{version}'", "dependencies:"]
    lines += [f"  - {dep}" for dep in dependencies]
    body = "\n".join(lines) + "\n"

    # The environment lives under the project's ``<workspace_dir>/<name>/`` folder
    # (mirroring the wheel registry's ``<dist>/``); *filename* carries the version
    # + ``py3XX`` tag so every build coexists rather than overwriting a flat file.
    dest = f"{workspace_dir.rstrip('/')}/{name}/{filename or f'{name}.env.yaml'}"
    path = DatabricksPath.from_(dest, client=client)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    logger.info(
        "wrote serverless base environment %r -> %s (%d deps, env v%s)",
        name, dest, len(dependencies), version,
    )
    return dest


def ensure_cluster_requirements(
    client: Any,
    name: str = "yellow",
    *,
    dependencies: "list[str] | tuple[str, ...]",
    workspace_dir: str = WORKSPACE_ENV_DIR,
    filename: "str | None" = None,
) -> str:
    """Create-or-update a plain ``<name>.requirements.txt`` in the workspace and
    return its path — the **classic-cluster** counterpart of
    :func:`ensure_named_environment`.

    Serverless references a base environment by path (``environment_version`` +
    dependencies); a classic cluster has no such concept — it installs from a
    pip requirements file via ``Library(requirements=<path>)``. So the same ygg
    image is written here as a flat requirements list (wheel workspace paths +
    pinned index requirements, no ``environment_version`` line)::

        /Workspace/Shared/pypi/ygg/ygg-0.8.54-py3-none-any.whl
        pyarrow==...

    Written (overwritten) on every call — upsert semantics, so redeploying keeps
    *name* pointing at the current image. *dependencies* are wheel workspace
    paths and/or pip requirement lines (typically the same list fed to
    :func:`ensure_named_environment`)."""
    from yggdrasil.databricks.path import DatabricksPath

    body = "\n".join(str(dep) for dep in dependencies) + "\n"
    # Sits beside the serverless ``.yml`` in the project's ``<name>/`` folder;
    # *filename* (a version-tagged ``<stem>.requirements.txt``) overrides the
    # default flat name so every version coexists in the one project folder.
    dest = f"{workspace_dir.rstrip('/')}/{name}/{filename or f'{name}.requirements.txt'}"
    path = DatabricksPath.from_(dest, client=client)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    logger.info(
        "wrote cluster requirements %r -> %s (%d deps)",
        name, dest, len(dependencies),
    )
    return dest


def ensure_environment(
    client: Any,
    *,
    python: "str | None" = None,
    version: "str | None" = None,
    workspace_dir: str = WORKSPACE_ENV_DIR,
    rebuild: bool = False,
    mode: Mode = Mode.AUTO,
) -> "dict[str, Any]":
    """Build + persist one **self-contained** ygg base environment for a single
    Python version, returning a small descriptor of what was written.

    Lays the environment out under the **project folder** (``ygg``), mirroring
    the wheel registry — every version / python coexists in the one folder, the
    filenames carry the ``<version>-py3XX`` tag, and the env holds **only its
    spec files** (no per-env binaries)::

        <workspace_dir>/ygg/
            ygg-<version>-py3XX.yml             serverless base_environment
            ygg-<version>-py3XX.requirements.txt   classic-cluster requirements

    The wheel closure is built (:func:`build_bundle`) and uploaded into the
    **shared pypi registry** (:data:`WORKSPACE_PYPI_DIR` — ``pypi/<dist>/<wheel>``),
    so dependency wheels are shared across images/versions instead of being
    duplicated per environment. The serverless ``.yml`` and cluster
    ``requirements.txt`` list those pypi wheel paths, so the runtime installs
    with zero PyPI access.

    *mode* sets the env-config-file policy (the wheel closure is **get-or-create**
    unless *rebuild*): :data:`Mode.APPEND` writes the ``.yml`` / ``.requirements.txt``
    only when they don't exist yet; :data:`Mode.AUTO` / :data:`Mode.OVERWRITE`
    (re)write them so they track the current closure.

    Returns ``{python, key, env_name, env_dir, n_wheels, serverless, cluster}``
    where ``env_name`` is the version-tagged stem and ``env_dir`` the project
    folder.
    """
    from yggdrasil.databricks.path import DatabricksPath

    overwrite_env = Mode.from_(mode) is not Mode.APPEND
    version = version or ilmd.version("ygg")
    key = environment_key_for(python)
    folder = environment_folder("ygg")                     # project folder: ``ygg``
    env_name = f"{folder}-{version}-{key}"                 # versioned file stem
    env_dir = f"{workspace_dir.rstrip('/')}/{folder}"

    # Wheels go to the shared pypi registry (not a per-env ``binaries/``), so the
    # dependency closure is reused across images/versions; the env files just
    # reference those pypi paths.
    bundle = ensure_bundle(client, "ygg", python=python, rebuild=rebuild)
    serverless_dest = f"{env_dir}/{env_name}.yml"
    cluster_dest = f"{env_dir}/{env_name}.requirements.txt"
    if overwrite_env or not DatabricksPath.from_(serverless_dest, client=client).exists():
        serverless = ensure_named_environment(
            client, folder, dependencies=bundle,
            environment_version=serverless_environment_version(python),
            workspace_dir=workspace_dir, filename=f"{env_name}.yml",
        )
    else:
        serverless = serverless_dest
    if overwrite_env or not DatabricksPath.from_(cluster_dest, client=client).exists():
        cluster = ensure_cluster_requirements(
            client, folder, dependencies=bundle, workspace_dir=workspace_dir,
            filename=f"{env_name}.requirements.txt",
        )
    else:
        cluster = cluster_dest
    return {
        "python": python,
        "key": key,
        "env_name": env_name,
        "env_dir": env_dir,
        "n_wheels": len(bundle),
        "serverless": serverless,
        "cluster": cluster,
    }


def ensure_environments(
    client: Any,
    *,
    versions: "tuple[str | None, ...] | list[str | None]" = (None,),
    workspace_dir: str = WORKSPACE_ENV_DIR,
    rebuild: bool = False,
    mode: Mode = Mode.AUTO,
    max_workers: "int | None" = None,
) -> "list[dict[str, Any]]":
    """:func:`ensure_environment` for several Python versions, **in parallel**.

    Each version's environment is an independent folder with its own wheel
    closure, so the builds share nothing and run concurrently on a
    :class:`~concurrent.futures.ThreadPoolExecutor` (the work is subprocess-bound
    — uv / pip — so threads give real overlap). Results are returned in the input
    order regardless of completion order. A single version skips the pool and
    runs inline. *mode* is forwarded to each :func:`ensure_environment`."""
    versions = list(versions) or [None]
    if len(versions) == 1:
        return [ensure_environment(
            client, python=versions[0], workspace_dir=workspace_dir,
            rebuild=rebuild, mode=mode,
        )]

    results: "dict[Any, dict[str, Any]]" = {}
    with ThreadPoolExecutor(
        max_workers=max_workers or len(versions), thread_name_prefix="ygg-env",
    ) as pool:
        futures = {
            pool.submit(
                ensure_environment,
                client, python=py, workspace_dir=workspace_dir,
                rebuild=rebuild, mode=mode,
            ): py
            for py in versions
        }
        for future in as_completed(futures):
            py = futures[future]
            results[py] = future.result()
    return [results[py] for py in versions]


# ---------------------------------------------------------------------------
# Arbitrary on-disk projects — discover a pyproject.toml, build it, and write a
# project-named environment (the user-project counterpart of the ygg image).
# ---------------------------------------------------------------------------


def find_pyproject(start: str | Path | None = None) -> Path:
    """The nearest ``pyproject.toml`` at or above *start* (cwd by default).

    *start* may point at the file itself, at its directory, or at any nested
    directory — the search walks up to the first ``pyproject.toml``. Raises
    :class:`FileNotFoundError` when none exists on the way up to the root."""
    start_path = Path(start).resolve() if start else Path.cwd()
    if start_path.is_file():
        if start_path.name == "pyproject.toml":
            return start_path
        start_path = start_path.parent
    for directory in (start_path, *start_path.parents):
        candidate = directory / "pyproject.toml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no pyproject.toml found at or above {start_path}")


def read_pyproject(path: str | Path) -> dict[str, Any]:
    """Parse a ``pyproject.toml``'s ``[project]`` table into what the deploy
    needs: ``name``, ``version``, base ``dependencies``, ``optional_dependencies``
    (keyed by extra), ``requires_python``, and the project ``dir``."""
    try:
        import tomllib as toml_reader
    except ModuleNotFoundError:                       # Python 3.10 has no tomllib
        import tomli as toml_reader

    path = Path(path).resolve()
    data = toml_reader.loads(path.read_text(encoding="utf-8"))
    project = data.get("project") or {}
    name = project.get("name")
    if not name:
        raise ValueError(f"{path} has no [project].name — not a deployable project")
    return {
        "name": name,
        "version": project.get("version", "0.0.0"),
        "dependencies": list(project.get("dependencies") or []),
        "optional_dependencies": {
            extra: list(reqs)
            for extra, reqs in (project.get("optional-dependencies") or {}).items()
        },
        "requires_python": project.get("requires-python"),
        "dir": path.parent,
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


def ensure_project_environment(
    client: Any,
    pyproject: str | Path | None = None,
    *,
    python: str | None = None,
    extras: tuple[str, ...] | list[str] = (),
    bundle: bool = False,
    mode: Mode = Mode.AUTO,
    workspace_dir: str = WORKSPACE_ENV_DIR,
    pypi_dir: str = WORKSPACE_PYPI_DIR,
) -> dict[str, Any]:
    """Discover a project's ``pyproject.toml``, build its wheel, and write a
    serverless **base environment** + classic-cluster **requirements** named for
    the project (``<name>-<version>``) — the user-project counterpart of
    :func:`ensure_environment`.

    The environment's dependency list is the **project wheel** plus the
    project's own ``[project].dependencies`` (and any requested *extras*' deps).
    With ``bundle=True`` those dependencies are downloaded as Linux-x86_64 wheels
    into the **shared pypi registry** (``pypi/<dist>/<wheel>``) and listed by
    workspace path, so the runtime installs with zero PyPI access; otherwise
    they're listed as index requirements resolved at install time.

    *mode* (a :class:`~yggdrasil.enums.Mode`) sets the idempotency policy:

    - :data:`Mode.OVERWRITE` — rebuild the wheel(s) and **overwrite** everything
      (the deployed wheel and the env config files).
    - :data:`Mode.APPEND` — **add only what's missing**: reuse an already-deployed
      wheel, and write the env config files only when they don't exist yet.
    - :data:`Mode.AUTO` (default) — **get-or-create** the wheel(s) (reuse when
      already deployed, build when not) but always **overwrite** the env config
      files so they track the current dependency set.

    Returns a descriptor with the project name/version, the env name, the written
    file paths, the dependency list, and the resolved *mode*.
    """
    from yggdrasil.databricks.path import DatabricksPath

    mode = Mode.from_(mode)
    rebuild = mode is Mode.OVERWRITE           # OVERWRITE rebuilds wheels
    overwrite_env = mode is not Mode.APPEND    # OVERWRITE + AUTO rewrite env files

    meta = read_pyproject(find_pyproject(pyproject))
    name, version = meta["name"], meta["version"]
    proj = _norm(name)
    # Folder = project name (mirrors the wheel registry's ``<dist>/``); the env
    # files carry the version + ``py3XX`` tag, so versions coexist in the folder.
    key = environment_key_for(python)
    env_name = f"{proj}-{version}-{key}"
    env_dir = f"{workspace_dir.rstrip('/')}/{proj}"

    # The project's declared deps, with any requested extras flattened in.
    deps = list(meta["dependencies"])
    for extra in extras:
        deps += meta["optional_dependencies"].get(extra, [])

    if bundle:
        # Zero-PyPI: project wheel + dependency closure, all uploaded to the
        # shared pypi registry and listed by workspace path. A ``.manifest`` under
        # the project's registry folder records the full path set so a
        # get-or-create (non-OVERWRITE) deploy can reuse the closure without
        # rebuilding.
        manifest = DatabricksPath.from_(
            f"{pypi_dir.rstrip('/')}/{proj}/{env_name}.manifest", client=client,
        )
        reused = (
            [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
            if (not rebuild and manifest.exists()) else []
        )
        if reused and DatabricksPath.from_(reused[0], client=client).exists():
            logger.info("reusing %d-wheel project bundle for %s", len(reused), env_name)
            dependencies = reused
        else:
            dependencies = [
                registry_upload(client, w, workspace_dir=pypi_dir, overwrite=True)
                for w in build_project_wheel(meta["dir"], python=python)
            ]
            dependencies += [
                registry_upload(client, w, workspace_dir=pypi_dir)
                for w in download_dependency_wheels(deps, python=python)
            ]
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("\n".join(dependencies) + "\n")
    else:
        # Project wheel by path + its declared deps resolved from the index.
        proj_dir = f"{pypi_dir.rstrip('/')}/{proj}"
        existing = (
            [] if rebuild
            else deployed_wheels(client, name, version, workspace_dir=proj_dir, dist_only=True)
        )
        if existing:
            logger.info("reusing deployed project wheel(s) for %s", env_name)
            wheel_paths = existing
        else:
            wheel_paths = [
                registry_upload(client, w, workspace_dir=pypi_dir, overwrite=True)
                for w in build_project_wheel(meta["dir"], python=python)
            ]
        dependencies = wheel_paths + deps

    # Env config files: OVERWRITE/AUTO always rewrite; APPEND writes only the
    # ones that don't exist yet ("add missing").
    serverless_dest = f"{env_dir}/{env_name}.yml"
    cluster_dest = f"{env_dir}/{env_name}.requirements.txt"
    if overwrite_env or not DatabricksPath.from_(serverless_dest, client=client).exists():
        serverless = ensure_named_environment(
            client, proj, dependencies=dependencies,
            environment_version=serverless_environment_version(python),
            workspace_dir=workspace_dir, filename=f"{env_name}.yml",
        )
    else:
        serverless = serverless_dest
    if overwrite_env or not DatabricksPath.from_(cluster_dest, client=client).exists():
        cluster = ensure_cluster_requirements(
            client, proj, dependencies=dependencies, workspace_dir=workspace_dir,
            filename=f"{env_name}.requirements.txt",
        )
    else:
        cluster = cluster_dest
    return {
        "name": name,
        "version": version,
        "env_name": env_name,
        "env_dir": env_dir,
        "dependencies": dependencies,
        "n_wheels": len(dependencies),
        "serverless": serverless,
        "cluster": cluster,
        "requires_python": meta["requires_python"],
        "mode": mode.name,
    }


def deployed_environments(client: Any, *, workspace_dir: str = WORKSPACE_ENV_DIR) -> list[str]:
    """Workspace paths of persisted environment files under *workspace_dir* —
    serverless base environments (``*.env.yaml`` / ``*.yml``, e.g. the
    version-pinned ``ygg-<version>-py3XX.yml``) and cluster requirement files
    (``*.requirements.txt``).

    Each environment lives in its **own ``<env-name>/`` folder** now
    (:func:`ensure_environment`), so this descends one level into those folders;
    loose files left directly under *workspace_dir* by older deploys are still
    picked up for back-compat. The environment-layer counterpart of
    :func:`deployed_wheels`: lets ``ygg databricks seed --check`` report whether
    the reusable environment files were written. Empty when the directory is
    absent or holds none."""
    from yggdrasil.databricks.path import DatabricksPath

    folder = DatabricksPath.from_(workspace_dir, client=client)
    if not folder.exists():
        return []

    suffixes = (".env.yaml", ".yml", ".requirements.txt")
    found: list[str] = []
    for child in folder.iterdir():
        if str(child.name).endswith(suffixes):     # legacy flat file
            found.append(child.full_path())
        elif child.is_dir():                        # per-env folder
            for sub in child.iterdir():
                if str(sub.name).endswith(suffixes):
                    found.append(sub.full_path())
    return found


def ygg_runtime_dependencies() -> list[str]:
    """The ygg image's runtime dependency **requirements** (names + version
    pins), for a serverless env to resolve from the workspace index.

    The live ``yggdrasil`` package's declared dependencies plus its
    ``[databricks]`` extra (which pins the latest ``databricks-sdk``). Shipped
    as names — not wheels — so the serverless runtime installs platform-correct
    builds. ``pyarrow`` / ``numpy`` and other binary deps therefore resolve on
    the cluster instead of being bundled from the build host.

    A bare, unpinned name (e.g. ``"xxhash"``) is pinned to its installed
    version so it reads as an unambiguous **index** requirement — otherwise the
    Spark Connect registry mistakes an installed-but-unpinned dep for a *local*
    package and tries to build a wheel for it."""
    _OPS = ("==", ">=", "<=", "!=", "~=", "===", ">", "<")
    out: list[str] = []
    for dep in _project_dependencies("ygg", {"databricks"}):
        head = dep.split(";", 1)[0].strip()
        if "[" in head or any(op in head for op in _OPS):
            out.append(dep)
            continue
        try:
            out.append(f"{head}=={ilmd.version(head)}")
        except Exception:  # noqa: BLE001 - not locally installed; ship the bare name
            out.append(dep)
    return out


def ygg_environment(
    client: Any,
    *,
    environment_key: str = "default",
    environment_version: "str | None" = None,
    rebuild: bool = False,
    workspace_dir: str = WORKSPACE_PYPI_DIR,
) -> Any:
    """The serverless ``JobEnvironment`` for the **versioned ygg image**.

    Pairs the serverless runtime — ``environment_version``, defaulting to
    :func:`serverless_environment_version` so the cluster Python matches the
    local interpreter (the locally-built wheel installs and UDFs run) — with:

    - the get-or-created pure-python ygg wheel (:func:`ensure_ygg_wheel`),
      installed **by path**; and
    - its runtime dependencies (:func:`ygg_runtime_dependencies`) as **index
      requirements**, so ``pyarrow`` / ``polars`` / ``databricks-sdk`` / … land
      as platform-correct builds the serverless runtime can actually install.

    Drop this into any serverless job's ``environments=[...]`` so its
    python-wheel tasks run the ``ygg`` CLI against a pinned image."""
    from databricks.sdk.service.compute import Environment
    from databricks.sdk.service.jobs import JobEnvironment

    wheels = ensure_ygg_wheel(
        client, workspace_dir=workspace_dir, rebuild=rebuild,
    )
    dependencies = list(wheels) + ygg_runtime_dependencies()
    return JobEnvironment(
        environment_key=environment_key,
        spec=Environment(
            environment_version=environment_version or serverless_environment_version(),
            dependencies=dependencies,
        ),
    )


def ygg_environments(
    client: Any,
    *,
    versions: "tuple[str, ...] | list[str]" = SUPPORTED_PYTHONS,
    default_python: "str | None" = None,
    rebuild: bool = False,
    workspace_dir: str = WORKSPACE_PYPI_DIR,
) -> list:
    """A serverless ``JobEnvironment`` **for every Python version** — the matrix
    counterpart of :func:`ygg_environment`.

    Returns ``[default, py310, py311, py312, py313]``: a ``"default"`` env pinned
    to *default_python* (the local interpreter unless given) plus one keyed
    ``py3XX`` per :data:`SUPPORTED_PYTHONS`, each pairing the matching ygg wheel
    (:func:`wheel_for_python`, installed by path) with the runtime deps from the
    index. Attach the whole list to a job's ``environments=[...]`` and point each
    task at the ``environment_key`` for the Python it needs; the default keeps the
    local-matched behaviour."""
    from databricks.sdk.service.compute import Environment
    from databricks.sdk.service.jobs import JobEnvironment

    wheels = ensure_ygg_wheels(
        client, versions=versions, workspace_dir=workspace_dir, rebuild=rebuild,
    )
    runtime = ygg_runtime_dependencies()
    default_python = _py_minor(default_python)

    def _env(key: str, python: str) -> Any:
        return JobEnvironment(
            environment_key=key,
            spec=Environment(
                environment_version=serverless_environment_version(python),
                dependencies=[wheel_for_python(wheels, python)] + runtime,
            ),
        )

    envs = [_env("default", default_python)]
    envs += [_env(environment_key_for(v), v) for v in versions]
    return envs
