from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import Field

from ..deps import get_pyfunc_service, get_pyfuncrun_service
from ..schemas.common import StrictModel
from ..schemas.pyfunc import (
    PyFuncCreate,
    PyFuncListResponse,
    PyFuncResponse,
    PyFuncUpdate,
)
from ..schemas.pyfuncrun import PyFuncRunCreate, PyFuncRunResponse
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


@router.get("/{func_id}", response_model=PyFuncResponse)
async def get_func(
    func_id: int,
    service: PyFuncService = Depends(get_pyfunc_service),
) -> PyFuncResponse:
    entry = await service.get(func_id)
    return PyFuncResponse(func=entry)


@router.put("/{func_id}", response_model=PyFuncResponse)
async def update_func(
    func_id: int,
    req: PyFuncUpdate,
    service: PyFuncService = Depends(get_pyfunc_service),
) -> PyFuncResponse:
    return await service.update(func_id, req)


@router.delete("/{func_id}", response_model=PyFuncResponse)
async def delete_func(
    func_id: int,
    service: PyFuncService = Depends(get_pyfunc_service),
) -> PyFuncResponse:
    return await service.delete(func_id)


class _FuncRunRequest(StrictModel):
    """Inline body for the convenience run endpoint."""
    env_id: int | None = None
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    timeout: float | None = None
    max_memory_mb: int | None = None


@router.post("/{func_id}/run", response_model=PyFuncRunResponse)
async def run_func(
    func_id: int,
    req: _FuncRunRequest,
    pyfuncrun: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    """Run a function directly by its ID. Creates a PyFuncRun and returns the result."""
    create_req = PyFuncRunCreate(
        func_id=func_id,
        env_id=req.env_id,
        args=list(req.args),
        kwargs=dict(req.kwargs),
        timeout=req.timeout,
        max_memory_mb=req.max_memory_mb,
    )
    return await pyfuncrun.create(create_req)
