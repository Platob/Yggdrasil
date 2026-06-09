"""Settings for the yggdrasil.node FastAPI backend.

A single pydantic model carries every tunable the node and its services
read at startup: identity, on-disk roots (node home, front home, logs,
saga catalog store), the remote-call gate, and the read/preview/spill
caps that keep the hot paths bounded.

Environment overrides are read with the ``YGG_NODE_`` prefix (e.g.
``YGG_NODE_NODE_ID``, ``YGG_NODE_HOME``, ``YGG_NODE_PORT``) so the
uvicorn-spawned benchmarks and ``ygg node serve`` configure the same way.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw) if raw else default


def _default_node_id() -> str:
    return os.environ.get("YGG_NODE_NODE_ID", "ygg-node")


def _default_node_home() -> Path:
    return _env_path("YGG_NODE_HOME", Path.home() / ".ygg" / "node")


def _default_front_home() -> Path:
    return _env_path("YGG_NODE_FRONT_HOME", Path.home() / ".ygg" / "front")


def _default_logs_root() -> Path:
    return _env_path("YGG_NODE_LOGS_ROOT", Path.home() / ".ygg" / "logs")


def _default_saga_home() -> Path:
    return _env_path("YGG_NODE_SAGA_HOME", Path.home() / ".ygg" / "saga")


class Settings(BaseModel):
    node_id: str = Field(default_factory=_default_node_id)
    node_home: Path = Field(default_factory=_default_node_home)
    front_home: Path = Field(default_factory=_default_front_home)
    logs_root: Path = Field(default_factory=_default_logs_root)
    saga_home: Path = Field(default_factory=_default_saga_home)
    allow_remote: bool = False
    max_read_bytes: int = 4 * 1024 * 1024
    tabular_preview_max_rows: int = 1000
    saga_result_spill_rows: int = 100_000

    class Config:
        arbitrary_types_allowed = True
