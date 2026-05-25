from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel


class NetworkSnapshot(StrictModel):
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    timestamp: str = ""


class ResourceSnapshot(StrictModel):
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_percent: float = 0.0
    network: NetworkSnapshot = Field(default_factory=NetworkSnapshot)
    timestamp: str = ""


class MonitorResponse(StrictModel):
    node_id: str
    snapshot: ResourceSnapshot
    history: list[ResourceSnapshot] = Field(default_factory=list)
