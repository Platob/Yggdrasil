from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel


class ReplicateRequest(StrictModel):
    target_node_url: str
    include_envs: bool = True
    include_funcs: bool = True
    include_dags: bool = True


class ReplicateStatus(StrictModel):
    source_node_id: str
    target_node_id: str
    envs_synced: int = 0
    funcs_synced: int = 0
    dags_synced: int = 0
    status: str = "pending"
    error: str | None = None


class NodeSnapshot(StrictModel):
    """Full snapshot of node assets — portable between nodes."""
    node_id: str
    envs: list[dict[str, Any]] = Field(default_factory=list)
    funcs: list[dict[str, Any]] = Field(default_factory=list)
    dags: list[dict[str, Any]] = Field(default_factory=list)
