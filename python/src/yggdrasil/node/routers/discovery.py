from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_discovery_service
from ..middleware import cached_response
from ..schemas.discovery import HelloRequest, HelloResponse, NodeInfo, PeerListResponse
from ..services.discovery import DiscoveryService

router = APIRouter(tags=["discovery"])

_hello_get = cached_response(ttl_seconds=5.0)
_peers_get = cached_response(ttl_seconds=5.0)


@router.get("", response_model=NodeInfo)
@_hello_get
async def hello_get(service: DiscoveryService = Depends(get_discovery_service)) -> NodeInfo:
    """Quick health/identity check -- no peer registration."""
    return await service.get_self_info()


@router.post("", response_model=HelloResponse)
async def hello_post(req: HelloRequest, service: DiscoveryService = Depends(get_discovery_service)) -> HelloResponse:
    """Register caller as peer and return this node's info + known peers."""
    return await service.hello(req)


@router.get("/peers", response_model=PeerListResponse)
@_peers_get
async def list_peers(service: DiscoveryService = Depends(get_discovery_service)) -> PeerListResponse:
    """Return list of known peers."""
    return await service.get_peers()


@router.post("/discover")
async def discover_friends(
    targets: list[str],
    service: DiscoveryService = Depends(get_discovery_service),
) -> PeerListResponse:
    """Reach out to a list of bot URLs and discover peers."""
    return await service.discover_friends(targets)
