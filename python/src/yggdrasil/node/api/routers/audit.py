"""Audit log endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/audit")
async def audit(request: Request, limit: int = Query(100)):
    return {"entries": request.app.state.audit.get(limit=limit)}
