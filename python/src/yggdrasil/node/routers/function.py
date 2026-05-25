from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_function_service, get_run_service
from ..schemas.function import (
    FunctionCreate,
    FunctionListResponse,
    FunctionResponse,
    FunctionUpdate,
)
from ..schemas.run import RunCreate, RunListResponse, RunResponse
from ..services.function import FunctionService
from ..services.run import RunService

router = APIRouter(tags=["function"])


# -- function CRUD ---------------------------------------------------------

@router.get("", response_model=FunctionListResponse)
async def list_functions(
    service: FunctionService = Depends(get_function_service),
) -> FunctionListResponse:
    return await service.list()


@router.post("", response_model=FunctionResponse)
async def create_function(
    req: FunctionCreate,
    service: FunctionService = Depends(get_function_service),
) -> FunctionResponse:
    return await service.create(req)


@router.get("/{func_id}", response_model=FunctionResponse)
async def get_function(
    func_id: str,
    service: FunctionService = Depends(get_function_service),
) -> FunctionResponse:
    entry = await service.get(func_id)
    return FunctionResponse(function=entry)


@router.put("/{func_id}", response_model=FunctionResponse)
async def update_function(
    func_id: str,
    req: FunctionUpdate,
    service: FunctionService = Depends(get_function_service),
) -> FunctionResponse:
    return await service.update(func_id, req)


@router.delete("/{func_id}", response_model=FunctionResponse)
async def delete_function(
    func_id: str,
    service: FunctionService = Depends(get_function_service),
) -> FunctionResponse:
    return await service.delete(func_id)


# -- run sub-resource (per function) ----------------------------------------

@router.post("/{func_id}/run", response_model=RunResponse)
async def trigger_function_run(
    func_id: str,
    service: RunService = Depends(get_run_service),
) -> RunResponse:
    req = RunCreate(function_id=func_id)
    return await service.create(req)


@router.get("/{func_id}/run", response_model=RunListResponse)
async def list_function_runs(
    func_id: str,
    service: RunService = Depends(get_run_service),
) -> RunListResponse:
    return await service.list(function_id=func_id)
