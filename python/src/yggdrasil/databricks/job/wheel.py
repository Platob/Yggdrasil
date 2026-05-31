"""Build the ygg wheel and upload it to a workspace dir for serverless jobs.

A deployed :class:`~yggdrasil.databricks.job.Flow` doesn't pull ``ygg`` from an
index — it ships the wheel built from *this* source tree. :func:`ensure_wheel`
builds the wheel (``python -m build``) and uploads it under
``/Workspace/Shared/.ygg/whl/pkg/<job>/`` (the :class:`Flow` builds per-job),
and :func:`ensure_requirement_wheel` ships extra deps (latest ``databricks-sdk``)
as their own uploaded wheels — so the serverless environment installs everything
by path, no index access.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "WORKSPACE_WHL_DIR",
    "find_project_root",
    "build_wheel",
    "upload_wheel",
    "ensure_wheel",
    "download_wheel",
    "ensure_requirement_wheel",
]

#: Root for workspace wheels — one subfolder per package (the job name for the
#: built wheel, the lib name for external deps) to centralize versions.
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
    wheel = build_wheel(source)
    return upload_wheel(client, wheel, workspace_dir=workspace_dir)


def download_wheel(requirement: str, dest_dir: "str | Path | None" = None) -> Path:
    """``pip download <requirement> --no-deps`` the **latest** wheel for a pinned
    or unpinned requirement (e.g. ``"databricks-sdk"``); return the ``.whl``."""
    out = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="ygg-dep-"))
    logger.info("downloading wheel for %s", requirement)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "download", requirement,
            "--no-deps", "--only-binary=:all:", "--dest", str(out),
        ],
        check=True,
    )
    wheels = sorted(out.glob("*.whl"))
    if not wheels:
        raise FileNotFoundError(f"no wheel downloaded for {requirement!r} in {out}")
    return wheels[-1]


def _req_name(requirement: str) -> str:
    """Bare distribution name from a requirement (``databricks-sdk==1.2`` →
    ``databricks-sdk``)."""
    return re.split(r"[<>=!~;\s\[]", requirement, 1)[0].strip()


def _pinned_version(requirement: str) -> Optional[str]:
    m = re.search(r"==\s*([0-9][^,;\s]*)", requirement)
    return m.group(1) if m else None


def latest_version(name: str) -> Optional[str]:
    """Best-effort latest version on the index via ``pip index versions``."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", name],
            check=True, capture_output=True, text=True,
        ).stdout
    except Exception:  # noqa: BLE001 - best effort; fall back to download
        return None
    m = re.search(r"Available versions:\s*([^\n]+)", out)
    if m:
        return m.group(1).split(",")[0].strip()
    m = re.search(rf"{re.escape(name)}\s*\(([^)]+)\)", out)
    return m.group(1).strip() if m else None


def ensure_requirement_wheel(
    client: Any,
    requirement: str,
    *,
    workspace_dir: str = WORKSPACE_WHL_DIR,
) -> str:
    """Ship an external lib (e.g. ``databricks-sdk``) as a wheel under
    ``<workspace_dir>/<lib>/`` — centralized by lib name so every job shares
    one copy per version.

    Pre-checks: resolve the version (pinned, else latest on the index) and reuse
    the wheel when it already exists in the workspace; otherwise download +
    upload it. Returns the workspace path."""
    from yggdrasil.databricks.path import DatabricksPath

    name = _req_name(requirement)
    lib_dir = f"{workspace_dir.rstrip('/')}/{name}"
    version = _pinned_version(requirement) or latest_version(name)
    if version:
        predicted = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        dest = f"{lib_dir}/{predicted}"
        if DatabricksPath.from_(dest, client=client).exists():
            return dest  # version already uploaded — skip download
    # Not present (or version unknown): download + upload.
    wheel = download_wheel(requirement)
    dest = f"{lib_dir}/{wheel.name}"
    if DatabricksPath.from_(dest, client=client).exists():
        return dest
    return upload_wheel(client, wheel, workspace_dir=lib_dir)
