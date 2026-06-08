"""Read the session ``ygg databricks configure`` remembers.

The specialized Databricks Loki detects a workspace **only** from the
config that ``ygg databricks configure`` writes — the profile in
``~/.databrickscfg`` plus the snapshot under
``~/.config/databricks-sdk-py/sessions/<hostname>.json`` — never from raw
``DATABRICKS_*`` environment variables or hard-coded parameters. This module
loads that snapshot.
"""
from __future__ import annotations

import json
import pathlib
import socket
from typing import Any, Optional

__all__ = ["session_dir", "read_session"]


def session_dir() -> pathlib.Path:
    return pathlib.Path.home() / ".config" / "databricks-sdk-py" / "sessions"


def _host_file() -> pathlib.Path:
    host = socket.gethostname() or "default"
    host = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in host)
    return session_dir() / f"{host}.json"


def read_session() -> Optional[dict[str, Any]]:
    """The remembered configure session (this host's, else the newest), or None."""
    preferred = _host_file()
    if preferred.is_file():
        return _load(preferred)
    d = session_dir()
    if not d.is_dir():
        return None
    snapshots = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return _load(snapshots[0]) if snapshots else None


def _load(path: pathlib.Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
