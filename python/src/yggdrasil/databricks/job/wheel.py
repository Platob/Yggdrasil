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
    "synthesize_project",
    "build_wheel",
    "upload_wheel",
    "ensure_wheel",
    "deployed_wheels",
    "ensure_ygg_wheel",
    "ygg_environment",
]

#: Root for workspace wheels — one subfolder per version to keep each bundle
#: self-contained (and so a deploy can reuse an existing version's bundle).
WORKSPACE_WHL_DIR = "/Workspace/Shared/.ygg/whl"

#: Default serverless environment version — **v5 (latest)**, used everywhere a
#: serverless ``Environment`` is built (the ygg image, async loader job, flows).
SERVERLESS_ENVIRONMENT_VERSION = "5"


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
) -> list[Path]:
    """Build the live *package* (synthesized project) **with its dependencies**
    via an isolated ``pip wheel`` — returns every produced ``.whl`` (the project
    plus all transitive deps + any extra *requirements*)."""
    project = synthesize_project(package, extras=extras)
    out = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="ygg-wheel-"))
    logger.info("building wheel (+ dependencies) for %s into %s", package, out)
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(project), *requirements, "--wheel-dir", str(out)],
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
) -> list[str]:
    """Build the live *package* with its dependencies (:func:`build_wheel`) and
    upload every wheel to *workspace_dir*; return their workspace paths. Built
    fresh each call so the deployed job ships current code."""
    wheels = build_wheel(package, extras=extras, requirements=requirements)
    return [upload_wheel(client, w, workspace_dir=workspace_dir) for w in wheels]


def deployed_wheels(
    client: Any,
    dist: str,
    version: str,
    *,
    workspace_dir: str,
) -> list[str]:
    """Workspace paths of a wheel bundle already deployed for *dist* *version*
    under *workspace_dir*, or ``[]`` when none is present.

    The bundle counts as present only when *dist*'s own wheel for *version* is
    there — a directory holding just the dependency wheels (a never-built or
    half-finished upload) is treated as absent so the caller rebuilds. Every
    ``.whl`` in the directory is returned, since :func:`build_wheel` drops the
    distribution wheel and its transitive deps side by side."""
    from yggdrasil.databricks.path import DatabricksPath

    folder = DatabricksPath.from_(workspace_dir, client=client)
    if not folder.exists():
        return []

    paths: list[str] = []
    has_dist = False
    want_dist, want_version = _norm(dist), _norm_version(version)
    for child in folder.iterdir():
        name = str(child.name)
        if not name.endswith(".whl"):
            continue
        paths.append(child.full_path())
        parts = name[:-4].split("-")  # drop ".whl"; <dist>-<version>-<tags...>
        if (
            len(parts) >= 2
            and _norm(parts[0]) == want_dist
            and _norm_version(parts[1]) == want_version
        ):
            has_dist = True
    return paths if has_dist else []


def ensure_ygg_wheel(
    client: Any,
    *,
    workspace_dir: str = WORKSPACE_WHL_DIR,
    rebuild: bool = False,
) -> list[str]:
    """Get-or-build the **full ygg wheel** bundle for the current version.

    The live ``yggdrasil`` package with its ``[databricks]`` dependencies
    **plus the latest ``databricks-sdk``**, deployed under a version-scoped
    subfolder of *workspace_dir* (``.../whl/<version>/``). On the first call for
    a version the bundle is built (:func:`build_wheel`) and uploaded; later
    calls find that deployed bundle (:func:`deployed_wheels`) and reuse it
    rather than rebuilding. Pass ``rebuild=True`` to force a fresh build (e.g.
    after a dev edit that doesn't bump the version). Returns the workspace
    wheel paths a serverless job installs by path (no index) to run the ``ygg``
    CLI on the cluster."""
    version = ilmd.version("ygg")
    version_dir = f"{workspace_dir.rstrip('/')}/{version}"

    if not rebuild:
        existing = deployed_wheels(client, "ygg", version, workspace_dir=version_dir)
        if existing:
            logger.info(
                "reusing deployed ygg %s wheel bundle at %s (%d wheels)",
                version, version_dir, len(existing),
            )
            return existing
        logger.info("no ygg %s wheel bundle at %s — building", version, version_dir)

    return ensure_wheel(
        client, "ygg",
        workspace_dir=version_dir,
        extras=("databricks",),
        requirements=("databricks-sdk",),
    )


def ygg_environment(
    client: Any,
    *,
    environment_key: str = "default",
    environment_version: str = SERVERLESS_ENVIRONMENT_VERSION,
    rebuild: bool = False,
    workspace_dir: str = WORKSPACE_WHL_DIR,
) -> Any:
    """The serverless ``JobEnvironment`` for the **versioned ygg image**.

    Pairs the latest serverless runtime (``environment_version``, default
    :data:`SERVERLESS_ENVIRONMENT_VERSION` = ``"5"``) with the get-or-created
    ygg wheel bundle (:func:`ensure_ygg_wheel` — ygg CLI + ``databricks-sdk`` +
    transitive deps), installed by path. Drop this into any serverless job's
    ``environments=[...]`` so its python-wheel tasks can run the ``ygg`` CLI
    against a pinned, pre-installed image rather than resolving from an index."""
    from databricks.sdk.service.compute import Environment
    from databricks.sdk.service.jobs import JobEnvironment

    wheels = ensure_ygg_wheel(
        client, workspace_dir=workspace_dir, rebuild=rebuild,
    )
    return JobEnvironment(
        environment_key=environment_key,
        spec=Environment(
            environment_version=environment_version,
            dependencies=wheels,
        ),
    )
