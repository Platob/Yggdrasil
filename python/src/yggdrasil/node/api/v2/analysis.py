"""Analysis routes — aggregate / series / OHLC / pivot over node files.

All four accept a JSON request body naming a node-local file and the
columns/measures to compute. Missing files become 404; a path escaping
``node_home`` becomes 403; an unsupported aggregate/format becomes 400.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..schemas.analysis import (
    AggregateRequest,
    AggregateResponse,
    OhlcRequest,
    OhlcResponse,
    PivotRequest,
    PivotResponse,
    SeriesRequest,
    SeriesResponse,
)

__all__ = ["router"]

router = APIRouter(prefix="/v2/analysis", tags=["analysis"])


def _translate(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=f"No such file {exc}")
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=f"Path outside node_home: {exc}")
    if isinstance(exc, (ValueError, KeyError)):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc


@router.post("/aggregate", response_model=AggregateResponse)
async def aggregate(request: Request, body: AggregateRequest) -> AggregateResponse:
    try:
        return await request.app.state.analysis.aggregate(body)
    except Exception as exc:
        raise _translate(exc)


@router.post("/series", response_model=SeriesResponse)
async def series(request: Request, body: SeriesRequest) -> SeriesResponse:
    try:
        return await request.app.state.analysis.series(body)
    except Exception as exc:
        raise _translate(exc)


@router.post("/ohlc", response_model=OhlcResponse)
async def ohlc(request: Request, body: OhlcRequest) -> OhlcResponse:
    try:
        return await request.app.state.analysis.ohlc(body)
    except Exception as exc:
        raise _translate(exc)


@router.post("/pivot", response_model=PivotResponse)
async def pivot(request: Request, body: PivotRequest) -> PivotResponse:
    try:
        return await request.app.state.analysis.pivot(body)
    except Exception as exc:
        raise _translate(exc)
