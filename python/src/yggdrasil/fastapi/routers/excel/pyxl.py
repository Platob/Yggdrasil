"""Python/Excel execution routes.

Handles real-time DataFrame execution and cached parquet preparation.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...deps import get_python_service
from ...schemas.python import (
    ExcelExecuteRequest,
    ExcelExecuteResponse,
    ExcelPrepareRequest,
    ExcelPrepareResponse,
)
from ...services.python import PythonService

router = APIRouter(tags=["python", "excel"])


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

