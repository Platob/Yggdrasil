from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from ..deps import get_monitor_service
from ..middleware import cached_response
from ..schemas.monitor import MonitorResponse, ResourceSnapshot
from ..services.monitor import MonitorService

router = APIRouter(tags=["monitor"])


@router.get("", response_model=MonitorResponse)
@cached_response(ttl_seconds=1.0)
async def get_monitor(
    limit: int = Query(60, ge=1, le=300),
    service: MonitorService = Depends(get_monitor_service),
) -> MonitorResponse:
    snap = service.snapshot()
    hist = service.history(limit)
    return MonitorResponse(
        node_id=service.settings.node_id,
        snapshot=snap,
        history=hist,
    )


@router.get("/snapshot", response_model=ResourceSnapshot)
async def get_snapshot(
    service: MonitorService = Depends(get_monitor_service),
) -> ResourceSnapshot:
    """Lightweight single-sample endpoint — no history, no caching layer.

    Use this for quick dashboard loads or polling clients that do not
    need the historical trace.
    """
    return service.snapshot()


@router.get("/info")
@cached_response(ttl_seconds=60.0)
async def get_info(
    service: MonitorService = Depends(get_monitor_service),
) -> dict[str, object]:
    """Static host facts (hostname, CPU count, total memory, ...).

    Cached for a minute since none of these change at runtime.
    """
    return {"node_id": service.settings.node_id, **service.system_info}


@router.get("/stream")
async def stream_monitor(
    interval: float = Query(1.0, ge=0.5, le=30.0, description="Seconds between samples."),
    service: MonitorService = Depends(get_monitor_service),
):
    async def _generate():
        while True:
            snap = service.snapshot()
            yield f"data: {json.dumps(snap.model_dump())}\n\n"
            await asyncio.sleep(interval)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
