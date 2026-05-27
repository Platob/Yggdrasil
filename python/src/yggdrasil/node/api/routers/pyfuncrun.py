from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from ...transport import (
    CONTENT_TYPE_ARROW_STREAM,
    serialize_result,
)
from ..deps import get_pyfuncrun_service
from ..schemas.pyfuncrun import (
    PyFuncRunCreate,
    PyFuncRunListResponse,
    PyFuncRunResponse,
)
from ..services.pyfuncrun import PyFuncRunService

router = APIRouter(tags=["pyfuncrun"])


@router.get("", response_model=PyFuncRunListResponse)
async def list_runs(
    func_id: int | None = None,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunListResponse:
    return await service.list(func_id=func_id)


@router.post("", response_model=PyFuncRunResponse)
async def create_run(
    req: PyFuncRunCreate,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    return await service.create(req)


@router.get("/{run_id}", response_model=PyFuncRunResponse)
async def get_run(
    run_id: int,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    entry = await service.get(run_id)
    return PyFuncRunResponse(run=entry)


@router.delete("/{run_id}", response_model=PyFuncRunResponse)
async def delete_run(
    run_id: int,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    return await service.delete(run_id)


@router.get("/{run_id}/logs")
async def stream_logs(
    run_id: int,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> StreamingResponse:
    await service.get(run_id)

    async def event_stream():
        async for event in service.stream_logs(run_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/{run_id}/state")
async def stream_state(
    run_id: int,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> StreamingResponse:
    """SSE stream of the full run state (progress, status, etc.) until done."""
    await service.get(run_id)

    async def event_stream():
        async for state in service.stream_state(run_id):
            yield f"data: {json.dumps(state)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/{run_id}/result")
async def get_result_arrow(
    run_id: int,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> Response:
    """Return the run result serialized as Arrow IPC (if tabular) or pickle."""
    entry = await service.get(run_id)
    if entry.result is None:
        return Response(status_code=204)
    data, content_type = serialize_result(entry.result)
    return Response(content=data, media_type=content_type)
