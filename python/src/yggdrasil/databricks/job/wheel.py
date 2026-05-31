"""Build a project (with its dependencies) into wheels and upload them to a
workspace dir, so a serverless :class:`~yggdrasil.databricks.job.Flow` installs
everything by path — no index access on the cluster.

:func:`build_wheel` runs ``pip wheel`` (isolated): the project at *source* — with
any extras — plus every transitive dependency are resolved into one directory of
wheels. :func:`ensure_wheel` uploads them all and returns their workspace paths.

(``uv`` has no ``pip wheel`` equivalent — ``uv build`` only produces the project
wheel — so the dependency-bundling build uses ``pip``.)
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "WORKSPACE_WHL_DIR",
    "find_project_root",
    "build_wheel",
    "upload_wheel",
    "ensure_wheel",
]

#: Root for workspace wheels — one subfolder per job to keep each self-contained.
WORKSPACE_WHL_DIR = "/Workspace/Shared/.ygg/whl"

#: Files that mark a buildable Python project root.
_PROJECT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg")


def find_project_root(start: "str | Path") -> Path:
    """Walk up from *start* (a file or dir) to the nearest project root — a dir
    holding a ``pyproject.toml`` / ``setup.py`` / ``setup.cfg``. Generic, so it
    locates whatever project a flow is defined in, not just ygg."""
    p = Path(start).resolve()
    candidates = (p, *p.parents) if p.is_dir() else p.parents
    for parent in candidates:
        if any((parent / marker).exists() for marker in _PROJECT_MARKERS):
            return parent
    raise FileNotFoundError(
        f"no Python project ({' / '.join(_PROJECT_MARKERS)}) found from {start!r}"
    )


def build_wheel(source: "str | Path", dest_dir: "str | Path | None" = None) -> Path:
    """Build a wheel for the project at (or above) *source* via an **isolated**
    ``python -m build`` (PEP 517 build env — independent of the current
    install). Works for any project. Returns the produced ``.whl`` path."""
    root = find_project_root(source)
    out = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="ygg-wheel-"))
    logger.info("building wheel from %s (isolated build)", root)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(root)],
        check=True,
    )
    wheels = sorted(out.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"no wheel produced in {out}")
    return wheels[-1]


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


def ensure_wheel(client: Any, source: "str | Path", *, workspace_dir: str = WORKSPACE_WHL_DIR) -> str:
    """Build the project at *source* (isolated) + upload the wheel; return its
    workspace path. Built fresh each call so the deployed job ships current code."""
def build_wheel(
    source: "str | Path",
    *,
    extras: "tuple[str, ...] | list[str]" = (),
    requirements: "tuple[str, ...] | list[str]" = (),
    dest_dir: "str | Path | None" = None,
) -> list[Path]:
    """Build the project at (or above) *source* **with its dependencies** —
    ``pip wheel`` resolves and builds/downloads a wheel for the project (with any
    *extras*) plus every transitive dependency (and any extra *requirements*)
    into one dir. Isolated; works for any project. Returns all ``.whl`` paths.
    """
    root = find_project_root(source)
    out = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="ygg-wheel-"))
    spec = str(root) + (f"[{','.join(extras)}]" if extras else "")
    logger.info("building wheel (+ dependencies) for %s into %s", spec, out)
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", spec, *requirements, "--wheel-dir", str(out)],
        check=True,
    )
    wheels = sorted(out.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"no wheels produced in {out}")
    return wheels


def ensure_wheel(
    client: Any,
    source: "str | Path",
    *,
    workspace_dir: str = WORKSPACE_WHL_DIR,
    extras: "tuple[str, ...] | list[str]" = (),
    requirements: "tuple[str, ...] | list[str]" = (),
) -> list[str]:
    """Build the project at *source* with its dependencies (:func:`build_wheel`)
    and upload every wheel to *workspace_dir*; return their workspace paths.
    Built fresh each call so the deployed job ships current code + deps."""
    wheels = build_wheel(source, extras=extras, requirements=requirements)
    return [upload_wheel(client, w, workspace_dir=workspace_dir) for w in wheels]
