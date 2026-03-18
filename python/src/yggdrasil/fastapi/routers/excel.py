"""Excel-specific router for DataFrame execution and cached parquet output.

Uses the global service layer, dependency injection, and shared schemas.
Gzip request decompression is handled via ``GzipRoute`` for all endpoints
on this router.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_python_service
from ..middleware import GzipRoute
from ..schemas.python import (
    ExcelExecuteRequest,
    ExcelExecuteResponse,
    ExcelPrepareRequest,
    ExcelPrepareResponse,
)
from ..services.python import PythonService

router = APIRouter(tags=["python", "excel"], route_class=GzipRoute)


@router.post("/execute", response_model=ExcelExecuteResponse)
async def execute_excel(
    req: ExcelExecuteRequest,
    service: PythonService = Depends(get_python_service),
) -> ExcelExecuteResponse:
    return await service.execute_excel(req)


@router.post("/prepare", response_model=ExcelPrepareResponse)
async def prepare_excel(
    req: ExcelPrepareRequest,
    service: PythonService = Depends(get_python_service),
) -> ExcelPrepareResponse:
    return await service.prepare_excel(req)
