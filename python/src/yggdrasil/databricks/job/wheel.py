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
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "WORKSPACE_PYPI_DIR",
    "WORKSPACE_WHL_DIR",
    "SERVERLESS_ENVIRONMENT_VERSION",
    "SERVERLESS_ENVIRONMENT_VERSIONS",
    "SERVERLESS_WHEEL_PLATFORMS",
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

#: Where reusable serverless **base environments** (``<name>.env.yaml``) live —
#: a job references one by file path via ``Environment.base_environment`` instead
#: of inlining the whole dependency list (see :func:`ensure_named_environment`).
#: Their dependencies are **built wheels in the workspace pypi registry**
#: (:func:`ensure_bundle`), so the runtime installs with zero PyPI access.
WORKSPACE_ENV_DIR = "/Workspace/Shared/environments"

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


def ygg_base_environment_name(python: "str | None" = None) -> str:
    """Canonical name of the reusable serverless **base environment** for the
    running ygg image — ``ygg-<version>-py3XX``.

    This is exactly the stem ``ygg databricks seed`` writes under
    :data:`WORKSPACE_ENV_DIR` (``ygg-<version>-py3XX.yml``), so a job that points
    its ``base_environment_name`` here reuses the seeded, wheel-built image when
    the seed has run — and self-provisions the identical file (same wheel
    closure, same path) when it hasn't. The version-pinned name is the single
    source of truth for "the correct ygg environment", replacing the old static
    ``yellow`` env."""
    try:
        import importlib.metadata as _md
        version = _md.version("ygg")
    except Exception:  # noqa: BLE001 — fall back to the in-tree version
        from yggdrasil.version import __version__ as version
    return f"ygg-{version}-{environment_key_for(python)}"


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
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(out), str(project)],
                check=True,
            )
        except FileNotFoundError:
            logger.info("uv not found — falling back to pip for %s", package)
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", str(project),
                 "--no-deps", "--wheel-dir", str(out)],
                check=True,
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
            subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", str(project), *requirements,
             "--wheel-dir", str(out)],
            check=True,
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
            subprocess.run(
                ["uv", "build", "--wheel", "--python", version,
                 "--out-dir", str(out), str(project)],
                check=True,
            )
        except FileNotFoundError:
            logger.info("uv not found — building one wheel for the current interpreter")
            subprocess.run(
                [sys.executable, "-m", "pip", "wheel", str(project),
                 "--no-deps", "--wheel-dir", str(out)],
                check=True,
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
        subprocess.run(
            ["uv", "build", "--wheel", "--python", py,
             "--out-dir", str(out), str(project)],
            check=True,
        )
    except FileNotFoundError:
        logger.info("uv not found — building the project wheel with the host interpreter")
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", str(project),
             "--no-deps", "--wheel-dir", str(out)],
            check=True,
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
        subprocess.run(
            [sys.executable, "-m", "pip", "download",
             "--only-binary=:all:",
             "--python-version", py,
             "--implementation", "cp",
             "--abi", abi, "--abi", "abi3", "--abi", "none",
             *platform_args,
             "--dest", str(out), *deps],
            check=True,
        )

    wheels = sorted(out.glob("*.whl"))
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

    The file is ``<name>.env.yaml`` unless *filename* overrides it — the seed
    writes a version-pinned ``ygg-<version>.yml`` so jobs can point at an exact
    image. Written (overwritten) on every call — upsert semantics, so redeploying
    keeps the file pointing at the current image. *dependencies* are wheel
    workspace paths (and/or pip requirement lines, when an index resolve is
    wanted instead)."""
    from yggdrasil.databricks.path import DatabricksPath

    version = environment_version or serverless_environment_version()
    lines = [f"environment_version: '{version}'", "dependencies:"]
    lines += [f"  - {dep}" for dep in dependencies]
    body = "\n".join(lines) + "\n"

    dest = f"{workspace_dir.rstrip('/')}/{filename or f'{name}.env.yaml'}"
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
    dest = f"{workspace_dir.rstrip('/')}/{name}.requirements.txt"
    path = DatabricksPath.from_(dest, client=client)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    logger.info(
        "wrote cluster requirements %r -> %s (%d deps)",
        name, dest, len(dependencies),
    )
    return dest


def deployed_environments(client: Any, *, workspace_dir: str = WORKSPACE_ENV_DIR) -> list[str]:
    """Workspace paths of persisted environment files under *workspace_dir* —
    serverless base environments (``*.env.yaml`` / ``*.yml``, e.g. the
    version-pinned ``ygg-<version>.yml``) and cluster requirement files
    (``*.requirements.txt``).

    The environment-layer counterpart of :func:`deployed_wheels`: lets
    ``ygg databricks seed --check`` report whether :func:`ensure_named_environment`
    / :func:`ensure_cluster_requirements` have actually written the reusable
    environment files. Empty when the directory is absent or holds none."""
    from yggdrasil.databricks.path import DatabricksPath

    folder = DatabricksPath.from_(workspace_dir, client=client)
    if not folder.exists():
        return []
    return [
        child.full_path()
        for child in folder.iterdir()
        if str(child.name).endswith((".env.yaml", ".yml", ".requirements.txt"))
    ]


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
