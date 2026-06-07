"""Filesystem routes — list node-local files and inspect parquet schemas."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

__all__ = ["router"]

router = APIRouter(prefix="/v2/fs", tags=["fs"])


@router.get("")
async def list_files(request: Request, path: str = "", glob: str = "*.parquet") -> list[dict]:
    try:
        return await request.app.state.fs.list_files(path, glob)
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Path outside node_home: {path!r}")


@router.get("/schema")
async def schema(request: Request, path: str) -> dict:
    try:
        return await request.app.state.fs.read_parquet_schema(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No such file {path!r}")
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Path outside node_home: {path!r}")
