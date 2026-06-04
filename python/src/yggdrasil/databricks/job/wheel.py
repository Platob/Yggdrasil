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
    "SUPPORTED_PYTHONS",
    "serverless_environment_version",
    "environment_key_for",
    "wheel_for_python",
    "is_editable_install",
    "user_pypi_dir",
    "synthesize_project",
    "build_wheel",
    "build_wheels_for_versions",
    "upload_wheel",
    "ensure_wheel",
    "ensure_wheels",
    "deployed_wheels",
    "ensure_ygg_wheel",
    "ensure_ygg_wheels",
    "ensure_bundle",
    "ensure_named_environment",
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
WORKSPACE_ENV_DIR = "/Workspace/Shared/ygg/environments"

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


def user_pypi_dir(client: Any) -> str:
    """The current user's private PyPI-like wheel folder
    (``/Workspace/Users/<me>/pypi``) — where **editable / dev** builds land so a
    developer's iterations don't collide with others in the shared registry."""
    user = client.workspace_client().current_user.me().user_name
    return f"/Workspace/Users/{user}/pypi"


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


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
    extras: "tuple[str, ...] | list[str]" = ("databricks",),
    workspace_dir: str = WORKSPACE_PYPI_DIR,
    rebuild: bool = False,
) -> list[str]:
    """Build *package* **with its whole transitive dependency closure** as
    wheels and upload every one; return their workspace paths.

    Where :func:`ensure_ygg_wheels` ships only the project wheel (deps resolve
    from the index at install), this bundles everything — so a serverless
    environment that lists these wheel paths installs **entirely from them, with
    zero PyPI access** ("0 pip install"). The dependency wheels are built for the
    deploying host's platform / Python via ``pip wheel``; pin the serverless
    environment to a matching Python (the deploy already matches the local one).

    Cached per ``(dist, version)`` in ``<workspace_dir>/<dist>-bundle/``: a
    bundle whose project wheel is already present is reused unless *rebuild*."""
    dist = distribution_for(package)
    version = ilmd.version(dist)
    dist_dir = f"{workspace_dir.rstrip('/')}/{_norm(dist)}-bundle"
    if not rebuild:
        existing = deployed_wheels(client, dist, version, workspace_dir=dist_dir)
        if existing:
            logger.info(
                "reusing %d-wheel %s %s bundle at %s",
                len(existing), dist, version, dist_dir,
            )
            return existing
    logger.info("building full %s %s bundle (project + deps) -> %s", dist, version, dist_dir)
    wheels = build_wheel(package, extras=extras, no_deps=False)
    return [upload_wheel(client, w, workspace_dir=dist_dir) for w in wheels]


def ensure_named_environment(
    client: Any,
    name: str = "yellow",
    *,
    dependencies: "list[str] | tuple[str, ...]",
    environment_version: "str | None" = None,
    workspace_dir: str = WORKSPACE_ENV_DIR,
) -> str:
    """Create-or-update a reusable serverless **base environment** *name* as an
    ``env.yaml`` in the workspace and return its path.

    A serverless job can reference this file via
    ``Environment.base_environment`` instead of inlining the whole dependency
    list — so one shared, named environment (e.g. ``yellow``) is defined once and
    every ygg job points at it. The file is the documented serverless env spec::

        environment_version: '5'
        dependencies:
          - /Workspace/Shared/pypi/ygg/ygg-0.8.52-py3-none-any.whl
          - pyarrow==...

    Written (overwritten) on every call — upsert semantics, so redeploying keeps
    *name* pointing at the current image. *dependencies* are wheel workspace paths
    and/or pip requirement lines."""
    from yggdrasil.databricks.path import DatabricksPath

    version = environment_version or serverless_environment_version()
    lines = [f"environment_version: '{version}'", "dependencies:"]
    lines += [f"  - {dep}" for dep in dependencies]
    body = "\n".join(lines) + "\n"

    dest = f"{workspace_dir.rstrip('/')}/{name}.env.yaml"
    path = DatabricksPath.from_(dest, client=client)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    logger.info(
        "wrote serverless base environment %r -> %s (%d deps, env v%s)",
        name, dest, len(dependencies), version,
    )
    return dest


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
