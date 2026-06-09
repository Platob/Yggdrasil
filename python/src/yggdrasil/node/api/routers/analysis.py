"""Analysis router — /api/v2/analysis/

POST /api/v2/analysis/aggregate
POST /api/v2/analysis/series
POST /api/v2/analysis/ohlc
POST /api/v2/analysis/pivot
POST /api/v2/analysis/forecast
GET  /api/v2/analysis/finance?path=&column=&timestamp=&risk_free_rate=
POST /api/v2/analysis/indicators
POST /api/v2/analysis/signals
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from yggdrasil.node.api.schemas.analysis import (
    AggregateRequest,
    FinanceRequest,
    ForecastRequest,
    IndicatorRequest,
    OhlcRequest,
    PivotRequest,
    SeriesRequest,
    SignalRequest,
)

router = APIRouter(prefix="/api/v2/analysis", tags=["analysis"])


def _svc(request: Request):
    return request.app.state.analysis


@router.post("/aggregate")
async def aggregate(req: AggregateRequest, svc=Depends(_svc)) -> dict:
    try:
        return (await svc.aggregate(req)).model_dump()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/series")
async def series(req: SeriesRequest, svc=Depends(_svc)) -> dict:
    try:
        return (await svc.series(req)).model_dump()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/ohlc")
async def ohlc(req: OhlcRequest, svc=Depends(_svc)) -> dict:
    try:
        return (await svc.ohlc(req)).model_dump()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/pivot")
async def pivot(req: PivotRequest, svc=Depends(_svc)) -> dict:
    try:
        return (await svc.pivot(req)).model_dump()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/forecast")
async def forecast(req: ForecastRequest, svc=Depends(_svc)) -> dict:
    try:
        return (await svc.forecast(req)).model_dump()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/finance")
async def finance(
    path: str = Query(..., description="Relative path to the price file"),
    column: str = Query("close", description="Price column name"),
    timestamp: str | None = Query(None, description="Timestamp column (optional)"),
    risk_free_rate: float = Query(0.0, description="Annual risk-free rate"),
    svc=Depends(_svc),
) -> dict:
    """Finance risk metrics: EMA, drawdown curve, and risk metrics.

    Returns ``ema[]``, ``drawdown[]``, and ``metrics{}``
    (total_return, cagr, ann_return, ann_volatility, sharpe, sortino,
    max_drawdown, calmar).
    """
    req = FinanceRequest(path=path, column=column, timestamp=timestamp,
                         risk_free_rate=risk_free_rate)
    try:
        return (await svc.finance(req)).model_dump()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/indicators")
async def indicators(req: IndicatorRequest, svc=Depends(_svc)) -> dict:
    try:
        return (await svc.indicators(req)).model_dump()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/signals")
async def signals(req: SignalRequest, svc=Depends(_svc)) -> dict:
    try:
        return (await svc.signals(req)).model_dump()
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
