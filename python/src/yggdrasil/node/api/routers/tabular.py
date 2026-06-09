"""Tabular router — /api/v2/tabular/

GET  /api/v2/tabular/inspect?path=   → InspectResult
GET  /api/v2/tabular/preview?path=&limit=  → PreviewResult
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

router = APIRouter(prefix="/api/v2/tabular", tags=["tabular"])


def _svc(request: Request):
    return request.app.state.tabular


@router.get("/inspect")
async def inspect(path: str, svc=Depends(_svc)) -> dict:
    try:
        result = await svc.inspect(path)
        return result.model_dump()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/preview")
async def preview(path: str, limit: int | None = None, svc=Depends(_svc)) -> dict:
    try:
        result = await svc.preview(path, limit=limit)
        return result.model_dump()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
