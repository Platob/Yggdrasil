from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any

import httpx

from ...config import Settings
from ...ids import make_id
from ...transport import (
    CONTENT_TYPE_ARROW_STREAM,
    CONTENT_TYPE_PICKLE,
    deserialize_result,
    serialize_result,
)
from ..schemas.backend import NodeBackend
from ..schemas.common import NodeRole
from ..schemas.network import (
    DispatchRequest,
    DispatchResponse,
    NodeMeta,
    PeerListResponse,
    PeerRegisterRequest,
    PeerRegisterResponse,
)
from .backend import BackendService

LOGGER = logging.getLogger(__name__)

_PEER_TTL = 300


class _Peer:
    __slots__ = (
        "node_id", "host", "port", "role", "version",
        "lat", "lon", "cpu_percent", "memory_percent",
        "active_runs", "gpu_count", "last_seen",
    )

    def __init__(self, meta: NodeMeta) -> None:
        self.node_id = meta.node_id
        self.host = meta.host
        self.port = meta.port
        self.role = meta.role
        self.version = meta.version
        self.lat = meta.lat
        self.lon = meta.lon
        self.cpu_percent = meta.cpu_percent
        self.memory_percent = meta.memory_percent
        self.active_runs = meta.active_runs
        self.gpu_count = meta.gpu_count
        self.last_seen: float = time.monotonic()

    def update(self, meta: NodeMeta) -> None:
        self.host = meta.host
        self.port = meta.port
        self.role = meta.role
        self.version = meta.version
        self.lat = meta.lat
        self.lon = meta.lon
        self.cpu_percent = meta.cpu_percent
        self.memory_percent = meta.memory_percent
        self.active_runs = meta.active_runs
        self.gpu_count = meta.gpu_count
        self.last_seen = time.monotonic()

    def to_meta(self) -> NodeMeta:
        return NodeMeta(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            role=self.role,
            version=self.version,
            lat=self.lat,
            lon=self.lon,
            cpu_percent=self.cpu_percent,
            memory_percent=self.memory_percent,
            active_runs=self.active_runs,
            gpu_count=self.gpu_count,
        )


class NetworkService:
    """Manages peer discovery and inter-node dispatch.

    Nodes form a mesh: any node can be driver (dispatches work) or
    executor (accepts work) or hybrid (both). Communication uses
    Arrow IPC serialization for tabular payloads, pickle for the rest.
    """

    def __init__(
        self,
        settings: Settings,
        backend_service: BackendService,
    ) -> None:
        self.settings = settings
        self._backend = backend_service
        self._peers: dict[str, _Peer] = {}
        self._lock = Lock()

    async def register_peer(self, req: PeerRegisterRequest) -> PeerRegisterResponse:
        meta = NodeMeta(
            node_id=req.node_id,
            host=req.host,
            port=req.port,
            role=req.role,
            version=req.version,
            lat=req.lat,
            lon=req.lon,
        )
        with self._lock:
            self._purge_stale()
            existing = self._peers.get(req.node_id)
            if existing:
                existing.update(meta)
            else:
                self._peers[req.node_id] = _Peer(meta)
            peers = [p.to_meta() for p in self._peers.values()]

        return PeerRegisterResponse(
            node_id=self.settings.node_id,
            role=self._backend.role,
            peers=peers,
        )

    async def get_peers(self) -> PeerListResponse:
        with self._lock:
            self._purge_stale()
            peers = [p.to_meta() for p in self._peers.values()]
        return PeerListResponse(node_id=self.settings.node_id, peers=peers)

    async def get_self_meta(self) -> NodeMeta:
        snap = self._backend.snapshot()
        return NodeMeta(
            node_id=self.settings.node_id,
            host=self.settings.host,
            port=self.settings.port,
            role=self._backend.role,
            version=self.settings.app_version,
            cpu_percent=snap.cpu_percent,
            memory_percent=(
                round(snap.memory_used_mb / snap.memory_total_mb * 100, 1)
                if snap.memory_total_mb > 0 else 0.0
            ),
            active_runs=snap.active_runs,
            gpu_count=len(snap.gpus),
        )

    async def dispatch(self, req: DispatchRequest) -> DispatchResponse:
        """Dispatch execution to the best available executor node.

        Selection: prefer executors with lowest active_runs and cpu_percent.
        Falls back to local execution if no peers are available.
        """
        target = self._select_executor()
        if target is None:
            return DispatchResponse(
                run_id=0,
                node_id=self.settings.node_id,
                status="no_executor",
            )

        url = f"http://{target.host}:{target.port}/api/v2/pyfuncrun"
        payload: dict[str, Any] = {
            "func_id": req.func_id,
            "args": req.args,
            "kwargs": req.kwargs,
        }
        if req.func_code is not None:
            payload["func_code"] = req.func_code
        if req.env_id is not None:
            payload["env_id"] = req.env_id
        if req.timeout is not None:
            payload["timeout"] = req.timeout

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"X-YGG-Source-Node": self.settings.node_id},
                )
                resp.raise_for_status()
                data = resp.json()

            run_data = data.get("run", data)
            return DispatchResponse(
                run_id=run_data.get("id", 0),
                node_id=target.node_id,
                status=run_data.get("status", "dispatched"),
            )
        except Exception as exc:
            LOGGER.error("Dispatch to %s failed: %s", target.node_id, exc)
            return DispatchResponse(
                run_id=0,
                node_id=target.node_id,
                status="failed",
            )

    async def dispatch_arrow(self, data: bytes, target_node_id: str | None = None) -> bytes:
        """Send Arrow IPC payload to a peer and return the response bytes."""
        target = self._get_peer(target_node_id) if target_node_id else self._select_executor()
        if target is None:
            raise ValueError("No executor available")

        url = f"http://{target.host}:{target.port}/api/v2/network/arrow"
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                url,
                content=data,
                headers={
                    "Content-Type": CONTENT_TYPE_ARROW_STREAM,
                    "X-YGG-Source-Node": self.settings.node_id,
                },
            )
            resp.raise_for_status()
            return resp.content

    def set_role(self, role: NodeRole) -> None:
        self._backend.set_role(role)

    # -- internals ----------------------------------------------------------

    def _select_executor(self) -> _Peer | None:
        with self._lock:
            self._purge_stale()
            candidates = [
                p for p in self._peers.values()
                if p.role in (NodeRole.EXECUTOR, NodeRole.HYBRID)
            ]
        if not candidates:
            return None
        return min(candidates, key=lambda p: (p.active_runs, p.cpu_percent))

    def _get_peer(self, node_id: str) -> _Peer | None:
        with self._lock:
            return self._peers.get(node_id)

    def _purge_stale(self) -> None:
        now = time.monotonic()
        stale = [
            nid for nid, p in self._peers.items()
            if now - p.last_seen > _PEER_TTL
        ]
        for nid in stale:
            del self._peers[nid]
