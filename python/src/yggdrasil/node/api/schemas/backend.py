from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import NodeRole, StrictModel


class GpuInfo(StrictModel):
    index: int = 0
    name: str = ""
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    utilization_percent: float = 0.0
    temperature_c: float = 0.0


class NetworkIO(StrictModel):
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0


class NodeBackend(StrictModel):
    node_id: str
    role: NodeRole = NodeRole.HYBRID
    hostname: str = ""
    platform: str = ""
    python_version: str = ""
    cpu_count: int = 0
    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_used_mb: float = 0.0
    disk_total_mb: float = 0.0
    gpus: list[GpuInfo] = Field(default_factory=list)
    network: NetworkIO = Field(default_factory=NetworkIO)
    uptime_seconds: float = 0.0
    active_runs: int = 0
    total_runs: int = 0
    timestamp: str = ""


class BackendResponse(StrictModel):
    backend: NodeBackend


class BackendStreamEvent(StrictModel):
    type: str = "snapshot"
    backend: NodeBackend
