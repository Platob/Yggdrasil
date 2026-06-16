"""Discovery endpoints — node identity and known peers."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/hello", tags=["discovery"])


@router.get("")
async def hello(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "node_id": settings.node_id,
        "version": "2.0",
        "allow_remote": settings.allow_remote,
    }


@router.get("/peers")
async def peers(request: Request) -> dict:
    # Peer discovery is not wired in the in-memory node; report self only.
    settings = request.app.state.settings
    return {"peers": [], "self": settings.node_id}
