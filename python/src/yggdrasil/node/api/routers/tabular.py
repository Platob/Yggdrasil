from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from ...transport import CONTENT_TYPE_ARROW_STREAM
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
    offset: int = 0,
    node: str | None = None,
    service: TabularService = Depends(get_tabular_service),
    network: NetworkService = Depends(get_network_service),
) -> TabularPreview:
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "GET", "/api/v2/tabular/preview", params={"path": path, "limit": limit, "offset": offset})
    return await service.preview(path, limit, offset)


@router.get("/preview.arrow")
async def preview_arrow(
    path: str,
    limit: int = 200,
    offset: int = 0,
    node: str | None = None,
    service: TabularService = Depends(get_tabular_service),
    network: NetworkService = Depends(get_network_service),
) -> Response:
    """Bounded, paged preview as an Arrow IPC stream — the fast wire for the grid."""
    if node and node != service.settings.node_id:
        return StreamingResponse(
            network.proxy_stream(node, "/api/v2/tabular/preview.arrow", {"path": path, "limit": limit, "offset": offset}),
            media_type=CONTENT_TYPE_ARROW_STREAM,
        )
    data = await service.preview_arrow(path, limit, offset)
    return Response(content=data, media_type=CONTENT_TYPE_ARROW_STREAM)


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
