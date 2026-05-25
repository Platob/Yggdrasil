from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..deps import get_env_service
from ..schemas.env import EnvGetResponse, EnvSetRequest, EnvSetResponse
from ..services.env import EnvService

router = APIRouter(tags=["env"])


@router.get("", response_model=EnvGetResponse)
async def get_env(
    keys: str | None = Query(
        default=None,
        description="Comma-separated list of env var names. Omit for all.",
    ),
    service: EnvService = Depends(get_env_service),
) -> EnvGetResponse:
    parsed_keys = [k.strip() for k in keys.split(",") if k.strip()] if keys else None
    return await service.get_env(parsed_keys)


@router.post("", response_model=EnvSetResponse)
async def set_env(
    req: EnvSetRequest,
    service: EnvService = Depends(get_env_service),
) -> EnvSetResponse:
    return await service.set_env(req)
