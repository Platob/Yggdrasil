"""Python function + environment listing endpoints (mounted under /api/v2)."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/pyfunc")
async def list_pyfunc():
    return {"functions": []}


@router.get("/pyenv")
async def list_pyenv():
    return {"environments": []}
