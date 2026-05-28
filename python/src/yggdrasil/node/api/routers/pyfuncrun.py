from __future__ import annotations

import orjson

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

_SSE_HEADERS = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}

from ...transport import serialize_result
from ..deps import get_pyfuncrun_service
from ..schemas.common import StrictModel
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
    status: str | None = None,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunListResponse:
    """List runs. ``status`` can be a comma-separated list (e.g. ``running,pending``)."""
    resp = await service.list(func_id=func_id)
    if status:
        wanted = {s.strip() for s in status.split(",") if s.strip()}
        resp.runs = [r for r in resp.runs if r.status in wanted]
    return resp


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


@router.head("/{run_id}")
async def head_run(
    run_id: int,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> dict:
    """Existence check by ID. Returns 200 if found, 404 otherwise."""
    await service.get(run_id)
    return {}


class _BulkDeleteRunRequest(StrictModel):
    """Body for bulk-delete: list of run IDs to delete."""
    ids: list[int]


@router.post("/bulk/delete")
async def bulk_delete_runs(
    req: _BulkDeleteRunRequest,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> dict:
    """Delete many runs in one request. Continues on per-ID failures."""
    deleted = 0
    failed: list[dict] = []
    for rid in req.ids:
        try:
            await service.delete(rid)
            deleted += 1
        except Exception as exc:
            failed.append({"id": rid, "error": str(exc)})
    return {"deleted": deleted, "failed": failed}


@router.post("/{run_id}/cancel", response_model=PyFuncRunResponse)
async def cancel_run(
    run_id: int,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    """Cancel a pending or running PyFuncRun. Returns the updated entry."""
    entry = await service.cancel(run_id)
    return PyFuncRunResponse(run=entry)


@router.delete("/{run_id}", response_model=PyFuncRunResponse)
async def delete_run(
    run_id: int,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    return await service.delete(run_id)


@router.get("/{run_id}/wait", response_model=PyFuncRunResponse)
async def wait_run(
    run_id: int,
    timeout: float = 600.0,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    """Block until run completes or timeout, then return final state."""
    entry = await service.wait(run_id, timeout=timeout)
    return PyFuncRunResponse(run=entry)


@router.get("/{run_id}/logs")
async def stream_logs(
    run_id: int,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> StreamingResponse:
    await service.get(run_id)

    async def event_stream():
        async for event in service.stream_logs(run_id):
            yield b"data: " + orjson.dumps(event) + b"\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.get("/{run_id}/state")
async def stream_state(
    run_id: int,
    service: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> StreamingResponse:
    """SSE stream of the full run state (progress, status, etc.) until done."""
    await service.get(run_id)

    async def event_stream():
        async for state in service.stream_state(run_id):
            yield b"data: " + orjson.dumps(state) + b"\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


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
