from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_pyfunc_service
from ..schemas.pyfunc import (
    PyFuncCreate,
    PyFuncListResponse,
    PyFuncResponse,
    PyFuncUpdate,
)
from ..services.pyfunc import PyFuncService

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
