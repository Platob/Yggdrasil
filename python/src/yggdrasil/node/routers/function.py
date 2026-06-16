"""Python function endpoints (CRUD + run)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from yggdrasil.exceptions.node import NodeNotFoundError

from ..schemas.function import (
    FunctionCreate,
    FunctionData,
    FunctionResponse,
    RunData,
    RunResponse,
)
from ..services.function import FunctionService

router = APIRouter(tags=["function"])


def _service(request: Request) -> FunctionService:
    return request.app.state.function


@router.post("/function", response_model=FunctionResponse)
async def create_function(payload: FunctionCreate, request: Request) -> FunctionResponse:
    return await _service(request).create(payload)


@router.get("/function", response_model=list[FunctionData])
async def list_functions(request: Request) -> list[FunctionData]:
    return await _service(request).list()


@router.get("/function/{function_id}", response_model=FunctionResponse)
async def get_function(function_id: str, request: Request) -> FunctionResponse:
    try:
        func = await _service(request).get(function_id)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FunctionResponse(function=func)


@router.delete("/function/{function_id}")
async def delete_function(function_id: str, request: Request) -> dict:
    try:
        await _service(request).delete(function_id)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": function_id}


@router.post("/function/{function_id}/run", response_model=RunResponse)
async def run_function(function_id: str, request: Request) -> RunResponse:
    try:
        return await _service(request).run(function_id)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/run/{run_id}", response_model=RunData)
async def get_run(run_id: str, request: Request) -> RunData:
    try:
        return await _service(request).get_run(run_id)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
