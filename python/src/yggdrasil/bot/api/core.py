"""Core endpoints: /api/ping, /api/v2/health, /api/v2/stats."""
from __future__ import annotations

import time

from fastapi import APIRouter

from yggdrasil.version import __version__

from ..market import cache_stats
from ..models import HealthResponse, PingResponse, StatsResponse
from ..ws import manager

router = APIRouter()

_start_ts = time.monotonic()
_req_total = 0


def _inc_req() -> None:
    global _req_total
    _req_total += 1


@router.get("/api/ping", response_model=PingResponse, tags=["core"])
async def ping() -> PingResponse:
    _inc_req()
    return PingResponse(version=__version__)


@router.get("/api/v2/health", response_model=HealthResponse, tags=["core"])
async def health() -> HealthResponse:
    _inc_req()
    size, _, _ = cache_stats()
    return HealthResponse(
        uptime_s=round(time.monotonic() - _start_ts, 2),
        market_cache_size=size,
        ws_connections=manager.active_count(),
    )


@router.get("/api/v2/stats", response_model=StatsResponse, tags=["core"])
async def stats() -> StatsResponse:
    _inc_req()
    size, hits, misses = cache_stats()
    return StatsResponse(
        requests_total=_req_total,
        ws_messages_sent=manager.messages_sent(),
        cache_hits=hits,
        cache_misses=misses,
        uptime_s=round(time.monotonic() - _start_ts, 2),
    )
