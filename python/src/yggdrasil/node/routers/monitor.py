"""Monitor endpoint — current resource snapshot."""
from __future__ import annotations

from fastapi import APIRouter, Request

from ..services.monitor import MonitorService

router = APIRouter(tags=["monitor"])


def _service(request: Request) -> MonitorService:
    return request.app.state.monitor


@router.get("/monitor")
async def monitor(request: Request) -> dict:
    return _service(request).snapshot()
