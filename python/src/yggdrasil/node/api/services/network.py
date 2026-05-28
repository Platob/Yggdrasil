from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any

import httpx

from ...config import Settings
from ...transport import CONTENT_TYPE_ARROW_STREAM
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


class NetworkService:
    def __init__(self, settings: Settings, backend_service: BackendService) -> None:
        self.settings = settings
        self._backend = backend_service
        self._peers: dict[str, tuple[NodeMeta, float]] = {}
        self._lock = Lock()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0),
        )

    async def register_peer(self, req: PeerRegisterRequest) -> PeerRegisterResponse:
        meta = NodeMeta(
            node_id=req.node_id, host=req.host, port=req.port,
            role=req.role, version=req.version, lat=req.lat, lon=req.lon,
        )
        with self._lock:
            self._peers[req.node_id] = (meta, time.monotonic())
            peers = [m for m, ts in self._peers.values() if time.monotonic() - ts < _PEER_TTL]
        return PeerRegisterResponse(
            node_id=self.settings.node_id, role=self._backend.role, peers=peers,
        )

    async def get_peers(self) -> PeerListResponse:
        now = time.monotonic()
        with self._lock:
            peers = [m for m, ts in self._peers.values() if now - ts < _PEER_TTL]
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
        target = self._select_executor()
        if target is None:
            return DispatchResponse(run_id=0, node_id=self.settings.node_id, status="no_executor")

        payload: dict[str, Any] = {"func_id": req.func_id, "args": req.args, "kwargs": req.kwargs}
        if req.func_code is not None:
            payload["func_code"] = req.func_code
        if req.env_id is not None:
            payload["env_id"] = req.env_id
        if req.timeout is not None:
            payload["timeout"] = req.timeout

        try:
            resp = await self._client.post(
                f"http://{target.host}:{target.port}/api/v2/pyfuncrun",
                json=payload,
                headers={"X-YGG-Source-Node": self.settings.node_id},
            )
            resp.raise_for_status()
            run_data = resp.json().get("run", resp.json())
            return DispatchResponse(
                run_id=run_data.get("id", 0), node_id=target.node_id,
                status=run_data.get("status", "dispatched"),
            )
        except Exception as exc:
            LOGGER.error("Dispatch to %s failed: %s", target.node_id, exc)
            return DispatchResponse(run_id=0, node_id=target.node_id, status="failed")

    async def dispatch_arrow(self, data: bytes, target_node_id: str | None = None) -> bytes:
        target = self._get_peer(target_node_id) if target_node_id else self._select_executor()
        if target is None:
            raise ValueError("No executor available")
        resp = await self._client.post(
            f"http://{target.host}:{target.port}/api/v2/network/arrow",
            content=data,
            headers={"Content-Type": CONTENT_TYPE_ARROW_STREAM, "X-YGG-Source-Node": self.settings.node_id},
        )
        resp.raise_for_status()
        return resp.content

    def set_role(self, role: NodeRole) -> None:
        self._backend.set_role(role)

    def _select_executor(self) -> NodeMeta | None:
        now = time.monotonic()
        with self._lock:
            candidates = [
                m for m, ts in self._peers.values()
                if now - ts < _PEER_TTL and m.role in (NodeRole.EXECUTOR, NodeRole.HYBRID)
            ]
        if not candidates:
            return None
        return min(candidates, key=lambda m: (m.active_runs, m.cpu_percent))

    def _get_peer(self, node_id: str) -> NodeMeta | None:
        with self._lock:
            entry = self._peers.get(node_id)
        return entry[0] if entry else None
