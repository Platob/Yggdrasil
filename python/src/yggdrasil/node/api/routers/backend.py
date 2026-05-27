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


@router.get("/summary")
async def get_summary(
    service: BackendService = Depends(get_backend_service),
) -> dict:
    """Lightweight health check — CPU, memory, run counts only.

    Skips GPU collection, network IO, and disk stats for minimal latency.
    Designed for peer health polling where you just need the basics.
    """
    cpu_percent = 0.0
    mem_total = 0.0
    mem_used = 0.0
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        mem_used = mem.used / (1024 * 1024)
        mem_total = mem.total / (1024 * 1024)
    except ImportError:
        pass

    memory_percent = round(mem_used / mem_total * 100, 1) if mem_total > 0 else 0.0
    return {
        "node_id": service.settings.node_id,
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "active_runs": service._active_runs_fn(),
        "total_runs": service._total_runs_fn(),
    }


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
