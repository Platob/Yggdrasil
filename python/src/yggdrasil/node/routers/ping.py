"""Liveness ping."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/ping")
async def ping() -> dict:
    return {"pong": True, "version": "2.0"}
