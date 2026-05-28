from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_replicate_service
from ..schemas.replicate import NodeSnapshot, ReplicateRequest, ReplicateStatus
from ..services.replicate import ReplicateService

router = APIRouter(tags=["replicate"])


@router.get("/export", response_model=NodeSnapshot)
async def export_snapshot(
    service: ReplicateService = Depends(get_replicate_service),
) -> NodeSnapshot:
    return await service.export_snapshot()


@router.post("/import", response_model=ReplicateStatus)
async def import_snapshot(
    snapshot: NodeSnapshot,
    service: ReplicateService = Depends(get_replicate_service),
) -> ReplicateStatus:
    return await service.import_snapshot(snapshot)


@router.post("/push", response_model=ReplicateStatus)
async def replicate_push(
    req: ReplicateRequest,
    service: ReplicateService = Depends(get_replicate_service),
) -> ReplicateStatus:
    """Push this node's assets to a target node."""
    return await service.replicate_to(req)


@router.post("/pull", response_model=ReplicateStatus)
async def replicate_pull(
    source_url: str,
    service: ReplicateService = Depends(get_replicate_service),
) -> ReplicateStatus:
    """Pull assets from a source node."""
    return await service.replicate_from(source_url)
