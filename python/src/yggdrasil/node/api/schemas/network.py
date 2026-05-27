from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import NodeRole, StrictModel


class NodeMeta(StrictModel):
    node_id: str
    host: str
    port: int
    role: NodeRole = NodeRole.HYBRID
    version: str = ""
    lat: float | None = None
    lon: float | None = None
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    active_runs: int = 0
    gpu_count: int = 0


class PeerRegisterRequest(StrictModel):
    node_id: str
    host: str
    port: int
    role: NodeRole = NodeRole.HYBRID
    version: str = ""
    lat: float | None = None
    lon: float | None = None


class PeerRegisterResponse(StrictModel):
    node_id: str
    role: NodeRole
    peers: list[NodeMeta] = Field(default_factory=list)


class PeerListResponse(StrictModel):
    node_id: str
    peers: list[NodeMeta]


class DispatchRequest(StrictModel):
    func_id: int | None = None
    func_code: str | None = None
    func_name: str = "dispatch"
    env_id: int | None = None
    env_spec: dict[str, Any] | None = None
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    timeout: float | None = None


class DispatchResponse(StrictModel):
    run_id: int
    node_id: str
    status: str = "pending"
