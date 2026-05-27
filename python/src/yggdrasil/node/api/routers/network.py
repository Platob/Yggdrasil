from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from ...transport import (
    CONTENT_TYPE_ARROW_STREAM,
    read_arrow_stream,
    write_arrow_stream_bytes,
)
from ..deps import get_network_service
from ..schemas.common import NodeRole
from ..schemas.network import (
    DispatchRequest,
    DispatchResponse,
    NodeMeta,
    PeerListResponse,
    PeerRegisterRequest,
    PeerRegisterResponse,
)
from ..services.network import NetworkService

router = APIRouter(tags=["network"])


@router.get("/ping")
async def ping(request: Request) -> dict:
    return {"pong": True, "node_id": request.app.state.settings.node_id}


@router.get("", response_model=NodeMeta)
async def get_self(
    service: NetworkService = Depends(get_network_service),
) -> NodeMeta:
    return await service.get_self_meta()


@router.post("/register", response_model=PeerRegisterResponse)
async def register_peer(
    req: PeerRegisterRequest,
    service: NetworkService = Depends(get_network_service),
) -> PeerRegisterResponse:
    return await service.register_peer(req)


@router.get("/peers", response_model=PeerListResponse)
async def list_peers(
    service: NetworkService = Depends(get_network_service),
) -> PeerListResponse:
    return await service.get_peers()


@router.put("/role")
async def set_role(
    role: NodeRole,
    service: NetworkService = Depends(get_network_service),
) -> NodeMeta:
    service.set_role(role)
    return await service.get_self_meta()


@router.post("/dispatch", response_model=DispatchResponse)
async def dispatch_execution(
    req: DispatchRequest,
    service: NetworkService = Depends(get_network_service),
) -> DispatchResponse:
    return await service.dispatch(req)


@router.post("/arrow")
async def receive_arrow(
    request: Request,
    service: NetworkService = Depends(get_network_service),
) -> Response:
    """Receive Arrow IPC payload from a peer node, echo back for now."""
    body = await request.body()
    table = read_arrow_stream(body)
    response_bytes = write_arrow_stream_bytes(table)
    return Response(
        content=response_bytes,
        media_type=CONTENT_TYPE_ARROW_STREAM,
        headers={"X-YGG-Node-Id": service.settings.node_id},
    )
