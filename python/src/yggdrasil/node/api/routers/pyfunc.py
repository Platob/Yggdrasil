from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import Field

from ..deps import get_pyfunc_service, get_pyfuncrun_service
from ..schemas.common import StrictModel
from ..schemas.pyfunc import (
    PyFuncCreate,
    PyFuncListResponse,
    PyFuncResponse,
    PyFuncUpdate,
)
from ..schemas.pyfuncrun import PyFuncRunCreate, PyFuncRunListResponse, PyFuncRunResponse
from ..services.pyfunc import PyFuncService
from ..services.pyfuncrun import PyFuncRunService

router = APIRouter(tags=["pyfunc"])


@router.get("", response_model=PyFuncListResponse)
async def list_funcs(
    service: PyFuncService = Depends(get_pyfunc_service),
) -> PyFuncListResponse:
    return await service.list()


@router.post("", response_model=PyFuncResponse)
async def create_func(
    req: PyFuncCreate,
    service: PyFuncService = Depends(get_pyfunc_service),
) -> PyFuncResponse:
    return await service.create(req)


@router.get("/by-name/{name}", response_model=PyFuncResponse)
async def get_func_by_name(
    name: str,
    service: PyFuncService = Depends(get_pyfunc_service),
) -> PyFuncResponse:
    """Resolve a function by name."""
    entry = await service.get_by_name(name)
    return PyFuncResponse(func=entry)


@router.head("/by-name/{name}")
async def head_func_by_name(
    name: str,
    service: PyFuncService = Depends(get_pyfunc_service),
) -> dict:
    """Existence check by name. Returns 200 if found, 404 otherwise."""
    await service.get_by_name(name)
    return {}


class _BulkDeleteFuncRequest(StrictModel):
    """Body for bulk-delete: list of func IDs to delete."""
    ids: list[int]


@router.post("/bulk/delete")
async def bulk_delete_funcs(
    req: _BulkDeleteFuncRequest,
    service: PyFuncService = Depends(get_pyfunc_service),
) -> dict:
    """Delete many functions in one request. Continues on per-ID failures."""
    deleted = 0
    failed: list[dict] = []
    for fid in req.ids:
        try:
            await service.delete(fid)
            deleted += 1
        except Exception as exc:
            failed.append({"id": fid, "error": str(exc)})
    return {"deleted": deleted, "failed": failed}


class _FuncRunRequest(StrictModel):
    """Inline body for the convenience run endpoint."""
    env_id: int | None = None
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    timeout: float | None = None
    max_memory_mb: int | None = None


@router.post("/by-name/{name}/run", response_model=PyFuncRunResponse)
async def run_func_by_name(
    name: str,
    req: _FuncRunRequest,
    pyfunc: PyFuncService = Depends(get_pyfunc_service),
    pyfuncrun: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    """Run a function by name. Resolves the name, creates a PyFuncRun, and
    returns immediately with the pending entry — the subprocess runs in the
    background. Use /{run_id}/wait, /logs, or /state to consume it."""
    func = await pyfunc.get_by_name(name)
    create_req = PyFuncRunCreate(
        func_id=func.id,
        env_id=req.env_id,
        args=list(req.args),
        kwargs=dict(req.kwargs),
        timeout=req.timeout,
        max_memory_mb=req.max_memory_mb,
    )
    return await pyfuncrun.create(create_req)


@router.get("/{func_id}", response_model=PyFuncResponse)
async def get_func(
    func_id: int,
    service: PyFuncService = Depends(get_pyfunc_service),
) -> PyFuncResponse:
    entry = await service.get(func_id)
    return PyFuncResponse(func=entry)


@router.head("/{func_id}")
async def head_func(
    func_id: int,
    service: PyFuncService = Depends(get_pyfunc_service),
) -> dict:
    """Existence check by ID. Returns 200 if found, 404 otherwise."""
    await service.get(func_id)
    return {}


@router.put("/{func_id}", response_model=PyFuncResponse)
async def update_func(
    func_id: int,
    req: PyFuncUpdate,
    service: PyFuncService = Depends(get_pyfunc_service),
) -> PyFuncResponse:
    return await service.update(func_id, req)


@router.patch("/{func_id}", response_model=PyFuncResponse)
async def patch_func(
    func_id: int,
    req: PyFuncUpdate,
    service: PyFuncService = Depends(get_pyfunc_service),
) -> PyFuncResponse:
    """Partial update on an existing function. Unlike PUT (upsert), 404s if missing."""
    return await service.update(func_id, req)


@router.delete("/{func_id}", response_model=PyFuncResponse)
async def delete_func(
    func_id: int,
    service: PyFuncService = Depends(get_pyfunc_service),
) -> PyFuncResponse:
    return await service.delete(func_id)


@router.post("/{func_id}/run", response_model=PyFuncRunResponse)
async def run_func(
    func_id: int,
    req: _FuncRunRequest,
    pyfuncrun: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    """Run a function by its ID. Returns immediately with the pending PyFuncRun;
    follow with GET /{run_id}/wait or GET /{run_id}/logs to consume the result."""
    create_req = PyFuncRunCreate(
        func_id=func_id,
        env_id=req.env_id,
        args=list(req.args),
        kwargs=dict(req.kwargs),
        timeout=req.timeout,
        max_memory_mb=req.max_memory_mb,
    )
    return await pyfuncrun.create(create_req)


@router.get("/{func_id}/runs", response_model=PyFuncRunListResponse)
async def list_func_runs(
    func_id: int,
    pyfuncrun: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunListResponse:
    """List all runs for a specific function."""
    return await pyfuncrun.list(func_id=func_id)


@router.get("/{func_id}/runs/{run_id}", response_model=PyFuncRunResponse)
async def get_func_run(
    func_id: int,
    run_id: int,
    pyfuncrun: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    """Get a specific run under a function. Validates the run belongs to the function."""
    entry = await pyfuncrun.get(run_id)
    if entry.func_id != func_id:
        from ...exceptions import NotFoundError
        raise NotFoundError(
            f"Run {run_id!r} does not belong to function {func_id!r}. "
            f"It belongs to function {entry.func_id!r}."
        )
    return PyFuncRunResponse(run=entry)


@router.post("/{func_id}/runs/{run_id}/cancel", response_model=PyFuncRunResponse)
async def cancel_func_run(
    func_id: int,
    run_id: int,
    pyfuncrun: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    """Cancel a pending or running run under a function."""
    entry = await pyfuncrun.cancel(run_id)
    return PyFuncRunResponse(run=entry)


@router.get("/{func_id}/runs/{run_id}/wait", response_model=PyFuncRunResponse)
async def wait_func_run(
    func_id: int,
    run_id: int,
    timeout: float = 600.0,
    pyfuncrun: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    """Wait for a specific run to complete under a function."""
    entry = await pyfuncrun.wait(run_id, timeout=timeout)
    return PyFuncRunResponse(run=entry)


@router.get("/{func_id}/runs/{run_id}/logs")
async def stream_func_run_logs(
    func_id: int,
    run_id: int,
    pyfuncrun: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> StreamingResponse:
    """SSE log stream for a run under a function."""
    await pyfuncrun.get(run_id)

    async def event_stream():
        async for event in pyfuncrun.stream_logs(run_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/{func_id}/runs/{run_id}/state")
async def stream_func_run_state(
    func_id: int,
    run_id: int,
    pyfuncrun: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> StreamingResponse:
    """SSE state stream for a run under a function."""
    await pyfuncrun.get(run_id)

    async def event_stream():
        async for state in pyfuncrun.stream_state(run_id):
            yield f"data: {json.dumps(state)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
