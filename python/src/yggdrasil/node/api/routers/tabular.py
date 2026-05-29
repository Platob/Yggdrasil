from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_network_service, get_tabular_service
from ..schemas.tabular import TabularInspect, TabularPreview, TabularWriteRequest, TabularWriteResponse
from ..services.network import NetworkService
from ..services.tabular import TabularService

router = APIRouter(tags=["tabular"])


@router.get("/inspect", response_model=TabularInspect)
async def inspect(
    path: str,
    node: str | None = None,
    service: TabularService = Depends(get_tabular_service),
    network: NetworkService = Depends(get_network_service),
) -> TabularInspect:
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "GET", "/api/v2/tabular/inspect", params={"path": path})
    return await service.inspect(path)


@router.get("/preview", response_model=TabularPreview)
async def preview(
    path: str,
    limit: int = 100,
    node: str | None = None,
    service: TabularService = Depends(get_tabular_service),
    network: NetworkService = Depends(get_network_service),
) -> TabularPreview:
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "GET", "/api/v2/tabular/preview", params={"path": path, "limit": limit})
    return await service.preview(path, limit)


@router.post("/write", response_model=TabularWriteResponse)
async def write(
    req: TabularWriteRequest,
    node: str | None = None,
    service: TabularService = Depends(get_tabular_service),
    network: NetworkService = Depends(get_network_service),
) -> TabularWriteResponse:
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/tabular/write", json_body=req.model_dump())
    return await service.write(req)
