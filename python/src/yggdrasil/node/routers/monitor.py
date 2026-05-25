from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from ..middleware import cached_response
from ..schemas.monitor import MonitorResponse, ResourceSnapshot
from ..services.monitor import MonitorService

router = APIRouter(tags=["monitor"])


def _get_service(request: Request) -> MonitorService:
    return request.app.state.monitor_service


@router.get("", response_model=MonitorResponse)
@cached_response(ttl_seconds=1.0)
async def get_monitor(
    limit: int = Query(60, ge=1, le=300),
    service: MonitorService = Depends(_get_service),
    request: Request = None,
) -> MonitorResponse:
    snap = service.snapshot()
    hist = service.history(limit)
    return MonitorResponse(
        node_id=request.app.state.settings.node_id,
        snapshot=snap,
        history=hist,
    )


@router.get("/stream")
async def stream_monitor(
    interval: float = Query(1.0, ge=0.25, le=10.0),
    service: MonitorService = Depends(_get_service),
):
    async def _generate():
        while True:
            snap = service.snapshot()
            yield f"data: {json.dumps(snap.model_dump())}\n\n"
            await asyncio.sleep(interval)

    return StreamingResponse(_generate(), media_type="text/event-stream")
