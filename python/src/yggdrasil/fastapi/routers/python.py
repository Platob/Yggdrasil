from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..deps import get_python_service
from ..schemas.python import (
    CreateEnvRequest,
    DeleteEnvRequest,
    DeleteEnvResponse,
    EnvListResponse,
    EnvRefRequest,
    EnvResponse,
    ExecuteCodeRequest,
    ExecutionResponse,
    MutationResponse,
    PackageRequest,
    RequirementsResponse,
)
from ..services.python import PythonService

router = APIRouter(tags=["python"])


@router.get("/envs/current", response_model=EnvResponse)
async def current_env(
    prefer_uv: bool = True,
    service: PythonService = Depends(get_python_service),
) -> EnvResponse:
    return await service.current_env(prefer_uv=prefer_uv)


@router.get("/envs", response_model=EnvListResponse)
async def list_envs(
    prefer_uv: bool = True,
    service: PythonService = Depends(get_python_service),
) -> EnvListResponse:
    return await service.list_envs(prefer_uv=prefer_uv)


@router.post("/envs/resolve", response_model=EnvResponse)
async def resolve_env(
    req: EnvRefRequest,
    service: PythonService = Depends(get_python_service),
) -> EnvResponse:
    return await service.resolve_env(req)


@router.post("/envs", response_model=EnvResponse)
async def create_env(
    req: CreateEnvRequest,
    service: PythonService = Depends(get_python_service),
) -> EnvResponse:
    return await service.create_env(req)


@router.delete("/envs", response_model=DeleteEnvResponse)
async def delete_env(
    req: DeleteEnvRequest,
    service: PythonService = Depends(get_python_service),
) -> DeleteEnvResponse:
    return await service.delete_env(req)


@router.get("/requirements", response_model=RequirementsResponse)
async def requirements(
    identifier: str | None = Query(default="current"),
    cwd: str | None = Query(default=None),
    prefer_uv: bool = Query(default=True),
    with_system: bool = Query(default=False),
    service: PythonService = Depends(get_python_service),
) -> RequirementsResponse:
    return await service.requirements(
        identifier=identifier,
        cwd=cwd,
        prefer_uv=prefer_uv,
        with_system=with_system,
    )


@router.post("/packages/install", response_model=MutationResponse)
async def install_packages(
    req: PackageRequest,
    service: PythonService = Depends(get_python_service),
) -> MutationResponse:
    return await service.install_packages(req)


@router.post("/packages/update", response_model=MutationResponse)
async def update_packages(
    req: PackageRequest,
    service: PythonService = Depends(get_python_service),
) -> MutationResponse:
    return await service.update_packages(req)


@router.post("/packages/uninstall", response_model=MutationResponse)
async def uninstall_packages(
    req: PackageRequest,
    service: PythonService = Depends(get_python_service),
) -> MutationResponse:
    return await service.uninstall_packages(req)


@router.post("/execute", response_model=ExecutionResponse)
async def execute_code(
    req: ExecuteCodeRequest,
    service: PythonService = Depends(get_python_service),
) -> ExecutionResponse:
    return await service.execute_code(req)

