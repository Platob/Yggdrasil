from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class NodeInfo(StrictModel):
    node_id: str
    host: str
    port: int
    version: str
    uptime: float = 0.0
    channels: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)


class HelloRequest(StrictModel):
    node_id: str
    host: str
    port: int
    version: str = "0.1.0"


class HelloResponse(StrictModel):
    node_id: str
    host: str
    port: int
    version: str
    peers: list[NodeInfo] = Field(default_factory=list)


class PeerListResponse(StrictModel):
    node_id: str
    peers: list[NodeInfo]
