"""Health / liveness / node-stats routes.

``/api/ping`` is the cheap liveness probe (no service touch); ``/health``
reports version and node identity; ``/stats`` reports uptime plus the fs
service's file/disk rollup. These are the endpoints the perf bench hammers,
so they stay allocation-light.
"""
from __future__ import annotations

import os
import time

from fastapi import APIRouter, Request

from yggdrasil.version import __version__

from ..schemas.base import now_ms

__all__ = ["ping_router", "router"]

# ``/api/ping`` lives outside the ``/api/v2`` prefix, so it gets its own router.
ping_router = APIRouter()
router = APIRouter(prefix="/v2", tags=["health"])


@ping_router.get("/ping")
async def ping(request: Request) -> dict:
    return {"ok": True, "ts": now_ms(), "node_id": request.app.state.settings.node_id}


@router.get("/health")
async def health(request: Request) -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "node_id": request.app.state.settings.node_id,
        "ts": now_ms(),
    }


@router.get("/stats")
async def stats(request: Request) -> dict:
    state = request.app.state
    fs_stats = await state.fs.get_stats()
    return {
        "node_id": state.settings.node_id,
        "version": __version__,
        "uptime_ms": now_ms() - state.started_at,
        "pid": os.getpid(),
        "files": fs_stats["file_count"],
        "disk_bytes": fs_stats["total_size"],
        "ts": now_ms(),
    }
