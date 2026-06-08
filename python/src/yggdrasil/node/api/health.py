"""Health / introspection endpoints."""
from __future__ import annotations

import platform
import sys
import time

from fastapi import APIRouter

from yggdrasil.node import list_registered as _list_registered
from yggdrasil.version import __version__

router = APIRouter()

_START_TIME = time.time()


@router.get("/api/ping")
async def ping() -> dict:
    return {"pong": True, "ts": int(time.time() * 1000)}


@router.get("/api/v2/health")
async def health() -> dict:
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _START_TIME, 3),
        "version": __version__,
    }


@router.get("/api/v2/stats")
async def stats() -> dict:
    from .call import request_count

    return {
        "registered_functions": len(_list_registered()),
        "requests_total": request_count(),
        "uptime_s": round(time.time() - _START_TIME, 3),
    }


@router.get("/api/v2/backend")
async def backend() -> dict:
    import pyarrow as pa

    info = {
        "arrow": pa.__version__,
        "python": platform.python_version(),
        "platform": sys.platform,
    }
    try:
        import polars as pl

        info["polars"] = pl.__version__
    except ImportError:
        info["polars"] = None
    return info
