from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..deps import get_backend_service
from ..schemas.backend import BackendResponse, NodeBackend
from ..services.backend import BackendService

router = APIRouter(tags=["backend"])


@router.get("", response_model=BackendResponse)
async def get_backend(
    service: BackendService = Depends(get_backend_service),
) -> BackendResponse:
    snap = service.snapshot()
    return BackendResponse(backend=snap)


@router.get("/history")
async def get_history(
    limit: int = 60,
    service: BackendService = Depends(get_backend_service),
) -> list[NodeBackend]:
    return service.history(limit=limit)


@router.get("/stream")
async def stream_backend(
    service: BackendService = Depends(get_backend_service),
) -> StreamingResponse:
    """SSE stream of node resource snapshots every second."""
    import asyncio

    async def event_stream():
        while True:
            snap = service.snapshot()
            payload = snap.model_dump()
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
