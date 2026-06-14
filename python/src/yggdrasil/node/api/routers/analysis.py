"""Analysis endpoints — aggregate, series, OHLC, pivot, forecast, indicators."""
from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from yggdrasil.node.api.schemas.analysis import (
    AggregateRequest,
    ForecastRequest,
    OhlcRequest,
    PivotRequest,
    SeriesRequest,
)

router = APIRouter()


@router.post("/aggregate")
async def aggregate(request: Request, req: AggregateRequest):
    return (await request.app.state.analysis.aggregate(req)).model_dump()


@router.post("/series")
async def series(request: Request, req: SeriesRequest):
    return (await request.app.state.analysis.series(req)).model_dump()


@router.post("/ohlc")
async def ohlc(request: Request, req: OhlcRequest):
    return (await request.app.state.analysis.ohlc(req)).model_dump()


@router.post("/pivot")
async def pivot(request: Request, req: PivotRequest):
    return (await request.app.state.analysis.pivot(req)).model_dump()


@router.post("/forecast")
async def forecast(request: Request, req: ForecastRequest):
    return (await request.app.state.analysis.forecast(req)).model_dump()


class IndicatorsRequest(BaseModel):
    path: str
    column: str
    sma_periods: list[int] = [20, 50, 200]
    rsi_period: int = 14
    bbands_period: int = 20
    bbands_stddev: float = 2.0


@router.post("/indicators")
async def indicators(request: Request, req: IndicatorsRequest):
    """Compute technical indicators (SMA, RSI, Bollinger Bands) on a parquet column."""
    return await request.app.state.analysis.indicators(req)
