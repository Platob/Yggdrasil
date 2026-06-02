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
    "WORKSPACE_WHL_DIR",
    "SERVERLESS_ENVIRONMENT_VERSION",
    "serverless_environment_version",
    "synthesize_project",
    "build_wheel",
    "upload_wheel",
    "ensure_wheel",
    "deployed_wheels",
    "ensure_ygg_wheel",
    "ygg_runtime_dependencies",
    "ygg_environment",
]

#: Root for workspace wheels — one subfolder per version to keep each bundle
#: self-contained (and so a deploy can reuse an existing version's bundle).
WORKSPACE_WHL_DIR = "/Workspace/Shared/.ygg/whl"

#: Latest serverless environment version — the fallback when the local Python
#: isn't one we map to an older runtime.
SERVERLESS_ENVIRONMENT_VERSION = "5"


def serverless_environment_version() -> str:
    """The Databricks serverless environment version whose runtime **Python
    matches the local interpreter**.

    Matching matters twice over: a locally-built ygg wheel installs cleanly, and
    Python UDFs run (Spark Connect requires the client and server to share a
    minor Python version). Mapping: 3.10 → ``"0"``, 3.11 → ``"2"``, anything
    else → the latest :data:`SERVERLESS_ENVIRONMENT_VERSION` (``"5"``)."""
    minor = sys.version_info[1]
    if minor == 10:
        return "0"
    if minor == 11:
        return "2"
    return SERVERLESS_ENVIRONMENT_VERSION


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
    pkg_dir = Path(module.__file__).resolve().parent
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
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", str(project), *requirements,
             "--wheel-dir", str(out)],
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
    workspace_dir: str = WORKSPACE_WHL_DIR,
    rebuild: bool = False,
) -> list[str]:
    """Get-or-build the **pure-python ygg wheel** for the current version.

    Builds *only* the live ``yggdrasil`` package as a ``py3-none-any`` wheel
    (``pip wheel --no-deps``) — no platform-specific dependency wheels — deployed
    under a version-scoped subfolder of *workspace_dir* (``.../whl/<version>/``).
    On the first call for a version the wheel is built and uploaded; later calls
    find and reuse it (:func:`deployed_wheels`). Pass ``rebuild=True`` to force
    a fresh build.

    Returns the workspace path of the ygg wheel — a serverless job installs it
    **by path** while resolving the runtime dependencies (see
    :func:`ygg_environment`) from the workspace index, so they land as
    platform-correct builds rather than wheels bundled from the deploying host
    (which a different serverless platform / python can't install)."""
    version = ilmd.version("ygg")
    version_dir = f"{workspace_dir.rstrip('/')}/{version}"

    if not rebuild:
        existing = deployed_wheels(
            client, "ygg", version, workspace_dir=version_dir, dist_only=True,
        )
        if existing:
            logger.info("reusing deployed ygg %s wheel at %s", version, version_dir)
            return existing
        logger.info("no ygg %s wheel at %s — building", version, version_dir)

    return ensure_wheel(
        client, "ygg",
        workspace_dir=version_dir,
        extras=("databricks",),
        no_deps=True,
    )


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
    workspace_dir: str = WORKSPACE_WHL_DIR,
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
