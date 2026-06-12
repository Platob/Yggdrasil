from __future__ import annotations
from datetime import datetime
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


@router.get("/")
async def root() -> dict:
    return {"service": "ygg-bot", "version": "0.1.0", "docs": "/docs"}
