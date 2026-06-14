"""Tabular inspection endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/inspect")
async def inspect(request: Request, path: str = Query(...)):
    res = await request.app.state.tabular.inspect(path)
    return res.model_dump()
