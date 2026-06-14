"""Health + backend introspection endpoints."""
from __future__ import annotations

import platform
import time

from fastapi import APIRouter

router = APIRouter()

_START = time.time()


@router.get("/ping")
async def ping():
    return {"status": "ok", "ts": time.time()}


@router.get("/v2/health")
async def health():
    return {"status": "ok", "ts": time.time()}


@router.get("/v2/stats")
async def stats():
    return {
        "uptime": time.time() - _START,
        "platform": platform.system(),
        "python": platform.python_version(),
    }


@router.get("/v2/backend")
async def backend():
    return {"name": "ygg-node", "version": "2.0", "status": "running"}


@router.get("/v2/backend/summary")
async def backend_summary():
    return {"name": "ygg-node", "version": "2.0", "status": "running", "endpoints": 20}
