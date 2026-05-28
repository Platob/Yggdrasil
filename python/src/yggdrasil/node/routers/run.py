from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..deps import get_run_service
from ..schemas.run import RunCreate, RunListResponse, RunResponse
from ..services.run import RunService

router = APIRouter(tags=["run"])


@router.get("", response_model=RunListResponse)
async def list_runs(
    service: RunService = Depends(get_run_service),
) -> RunListResponse:
    return await service.list()


@router.get("/active", response_model=RunListResponse)
async def list_active_runs(
    service: RunService = Depends(get_run_service),
) -> RunListResponse:
    """Return runs currently in 'pending' or 'running' state."""
    return await service.active()


@router.post("", response_model=RunResponse)
async def create_run(
    req: RunCreate,
    service: RunService = Depends(get_run_service),
) -> RunResponse:
    return await service.create(req)


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: int,
    service: RunService = Depends(get_run_service),
) -> RunResponse:
    entry = await service.get(run_id)
    return RunResponse(run=entry)


@router.delete("/{run_id}", response_model=RunResponse)
async def delete_run(
    run_id: int,
    service: RunService = Depends(get_run_service),
) -> RunResponse:
    return await service.delete(run_id)


@router.get("/{run_id}/logs")
async def stream_logs(
    run_id: int,
    service: RunService = Depends(get_run_service),
) -> StreamingResponse:
    # Validate run exists before starting the stream response,
    # so the exception handler can return a proper 404 JSON response.
    await service.get(run_id)

    async def event_stream():
        async for event in service.stream_logs(run_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
