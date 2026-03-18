from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_system_service
from ..schemas.system import HealthResponse, SystemInfoResponse
from ..services.system import SystemService

router = APIRouter(tags=["system"])


@router.get("/info", response_model=SystemInfoResponse)
async def info(
    service: SystemService = Depends(get_system_service),
) -> SystemInfoResponse:
    return await service.info()


@router.get("/healthz", response_model=HealthResponse)
async def healthz(
    service: SystemService = Depends(get_system_service),
) -> HealthResponse:
    return await service.health()
