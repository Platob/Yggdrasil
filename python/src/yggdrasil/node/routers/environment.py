from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_environment_service
from ..schemas.environment import (
    EnvironmentCloneRequest,
    EnvironmentCreate,
    EnvironmentListResponse,
    EnvironmentResponse,
    EnvironmentUpdate,
    InstallRequest,
)
from ..services.environment import EnvironmentService

router = APIRouter(tags=["environment"])


@router.get("", response_model=EnvironmentListResponse)
async def list_environments(
    service: EnvironmentService = Depends(get_environment_service),
) -> EnvironmentListResponse:
    return await service.list()


@router.post("", response_model=EnvironmentResponse)
async def create_environment(
    req: EnvironmentCreate,
    service: EnvironmentService = Depends(get_environment_service),
) -> EnvironmentResponse:
    return await service.create(req)


@router.get("/{env_id}", response_model=EnvironmentResponse)
async def get_environment(
    env_id: int,
    service: EnvironmentService = Depends(get_environment_service),
) -> EnvironmentResponse:
    entry = await service.get(env_id)
    return EnvironmentResponse(environment=entry)


@router.put("/{env_id}", response_model=EnvironmentResponse)
async def update_environment(
    env_id: int,
    req: EnvironmentUpdate,
    service: EnvironmentService = Depends(get_environment_service),
) -> EnvironmentResponse:
    return await service.update(env_id, req)


@router.delete("/{env_id}", response_model=EnvironmentResponse)
async def delete_environment(
    env_id: int,
    service: EnvironmentService = Depends(get_environment_service),
) -> EnvironmentResponse:
    return await service.delete(env_id)


@router.post("/{env_id}/clone", response_model=EnvironmentResponse)
async def clone_environment(
    env_id: int,
    req: EnvironmentCloneRequest | None = None,
    service: EnvironmentService = Depends(get_environment_service),
) -> EnvironmentResponse:
    new_name = req.name if req else None
    return await service.clone(env_id, new_name=new_name)


@router.post("/{env_id}/install", response_model=EnvironmentResponse)
async def install_packages(
    env_id: int,
    req: InstallRequest,
    service: EnvironmentService = Depends(get_environment_service),
) -> EnvironmentResponse:
    return await service.install(env_id, req)
