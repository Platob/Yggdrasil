"""Filesystem endpoints — list and read node-rooted paths."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/ls")
async def ls(
    request: Request,
    path: str = Query(""),
    offset: int = Query(0),
    limit: int | None = Query(None),
):
    res = await request.app.state.fs.ls(path, offset=offset, limit=limit)
    return res.model_dump()


@router.get("/read")
async def read(request: Request, path: str = Query(...)):
    res = await request.app.state.fs.read(path)
    return {"content": res.content.decode("utf-8", "replace"), "truncated": res.truncated}
