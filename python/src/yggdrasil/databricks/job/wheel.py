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
    "synthesize_project",
    "build_wheel",
    "upload_wheel",
    "ensure_wheel",
    "ensure_ygg_wheel",
]

#: Root for workspace wheels — one subfolder per job to keep each self-contained.
WORKSPACE_WHL_DIR = "/Workspace/Shared/.ygg/whl"


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def distribution_for(package: str) -> str:
    """The distribution (pip) name providing the import *package* (``yggdrasil``
    → ``ygg``). Falls back to *package* itself when unmapped."""
    dists = ilmd.packages_distributions().get(package)
    return dists[0] if dists else package


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
    package: str,
    *,
    extras: "tuple[str, ...] | list[str]" = (),
    dest_dir: "str | Path | None" = None,
) -> Path:
    """Create a buildable project from the **installed** *package* — copy its
    on-disk files and write a ``pyproject.toml`` reconstructed from the
    distribution metadata (version, console scripts, dependencies incl. the
    requested *extras*). Returns the project dir."""
    module = importlib.import_module(package)
    pkg_dir = Path(module.__file__).resolve().parent
    dist = distribution_for(package)
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


def ensure_ygg_wheel(client: Any, *, workspace_dir: str = WORKSPACE_WHL_DIR) -> list[str]:
    """Build the **full ygg wheel** — the live ``yggdrasil`` package with its
    ``[databricks]`` dependencies **plus the latest ``databricks-sdk``** — and
    upload every produced wheel to *workspace_dir*; return their workspace
    paths. The bundle a serverless job installs by path (no index) to run any
    ``ygg-job`` task on the cluster."""
    return ensure_wheel(
        client, "yggdrasil",
        workspace_dir=workspace_dir,
        extras=("databricks",),
        requirements=("databricks-sdk",),
    )
