from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_pyenv_service
from ..schemas.pyenv import (
    PyEnvCreate,
    PyEnvListResponse,
    PyEnvResponse,
    PyEnvUpdate,
)
from ..services.pyenv import PyEnvService

router = APIRouter(tags=["pyenv"])


@router.get("", response_model=PyEnvListResponse)
async def list_envs(
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvListResponse:
    return await service.list()


@router.post("", response_model=PyEnvResponse)
async def create_env(
    req: PyEnvCreate,
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvResponse:
    return await service.create(req)


@router.get("/{env_id}", response_model=PyEnvResponse)
async def get_env(
    env_id: int,
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvResponse:
    entry = await service.get(env_id)
    return PyEnvResponse(env=entry)


@router.put("/{env_id}", response_model=PyEnvResponse)
async def update_env(
    env_id: int,
    req: PyEnvUpdate,
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvResponse:
    return await service.update(env_id, req)


@router.delete("/{env_id}", response_model=PyEnvResponse)
async def delete_env(
    env_id: int,
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvResponse:
    return await service.delete(env_id)
