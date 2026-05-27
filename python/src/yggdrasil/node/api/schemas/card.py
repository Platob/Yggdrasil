from __future__ import annotations

from pydantic import Field

from .common import NodeRole, StrictModel


class NodeCard(StrictModel):
    node_id: str
    host: str
    port: int
    url: str
    role: NodeRole = NodeRole.HYBRID
    version: str = ""
    hostname: str = ""
    platform: str = ""
    python_version: str = ""
    lat: float | None = None
    lon: float | None = None
    cpu_count: int = 0
    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    gpu_count: int = 0
    active_runs: int = 0
    total_runs: int = 0
    env_count: int = 0
    func_count: int = 0
    uptime_seconds: float = 0.0
    node_home: str = ""
    peers: list[str] = Field(default_factory=list)
