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
    "wheel_name",
    "build_wheel",
    "upload_wheel",
    "ensure_wheel",
    "download_wheel",
    "ensure_requirement_wheel",
]

#: Root for workspace wheels — one subfolder per package (the job name for the
#: built wheel, the lib name for external deps) to centralize versions.
WORKSPACE_WHL_DIR = "/Workspace/Shared/.ygg/whl"


def project_root() -> Path:
    """The directory holding ``pyproject.toml`` (the ygg source tree)."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "could not locate pyproject.toml — building the ygg wheel needs the "
        "source tree (deploy from a checkout, not an installed package)"
    )


def wheel_name() -> str:
    """The wheel filename for the current ygg version (``ygg-<v>-…whl``)."""
    from yggdrasil.version import __version__

    return f"ygg-{__version__}-py3-none-any.whl"


def build_wheel(dest_dir: "str | Path | None" = None) -> Path:
    """Build the ygg wheel via ``python -m build`` — returns the ``.whl`` path."""
    root = project_root()
    out = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="ygg-wheel-"))
    logger.info("building ygg wheel from %s", root)
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
    logger.info("uploaded ygg wheel to %s", dest)
    return dest


def ensure_wheel(
    client: Any,
    *,
    workspace_dir: str = WORKSPACE_WHL_DIR,
    rebuild: bool = False,
) -> str:
    """Build + upload the ygg wheel; return its workspace path.

    Idempotent by version: when the matching wheel already exists in
    *workspace_dir* it's reused (skipping the build) unless ``rebuild=True``.
    """
    from yggdrasil.databricks.path import DatabricksPath

    dest = f"{workspace_dir.rstrip('/')}/{wheel_name()}"
    if not rebuild and DatabricksPath.from_(dest, client=client).exists():
        return dest
    wheel = build_wheel()
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
