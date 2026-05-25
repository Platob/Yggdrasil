from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import Response as FastAPIResponse

from ..deps import get_cmd_service
from ..schemas.cmd import CmdListResponse, CmdRequest, CmdResponse
from ..services.cmd import CmdService

router = APIRouter(tags=["cmd"])


@router.get("", response_model=CmdListResponse)
async def list_commands(
    service: CmdService = Depends(get_cmd_service),
) -> CmdListResponse:
    return await service.list_history()


@router.post("", response_model=CmdResponse)
async def execute_command(
    req: CmdRequest,
    service: CmdService = Depends(get_cmd_service),
) -> CmdResponse:
    return await service.execute(req)


@router.get("/{cmd_id}", response_model=CmdResponse)
async def get_command(
    cmd_id: str,
    service: CmdService = Depends(get_cmd_service),
) -> CmdResponse:
    return await service.get(cmd_id)


@router.delete("/{cmd_id}", response_model=CmdResponse)
async def delete_command(
    cmd_id: str,
    service: CmdService = Depends(get_cmd_service),
) -> CmdResponse:
    return await service.delete(cmd_id)
