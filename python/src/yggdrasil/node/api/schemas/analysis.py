"""Analysis engine contracts: aggregate, series, ohlc, pivot, forecast,
finance risk metrics, plus the trading layer — technical indicators and
signal detection.

Finance endpoint: GET /api/v2/analysis/finance
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class AggMeasure(BaseModel):
    column: str
    agg: str  # "mean" | "sum" | "count" | "min" | "max"


class AggregateRequest(BaseModel):
    path: str
    group_by: list[str]
    measures: list[AggMeasure]


class AggregateResult(BaseModel):
    group_count: int
    data: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# series (adaptive downsample)
# ---------------------------------------------------------------------------

class SeriesRequest(BaseModel):
    path: str
    column: str
    points: int = 1000


class SeriesResult(BaseModel):
    x: list[Any]
    y: list[float]


# ---------------------------------------------------------------------------
# ohlc
# ---------------------------------------------------------------------------

class OhlcRequest(BaseModel):
    path: str
    column: str
    buckets: int = 100


class OhlcResult(BaseModel):
    bars: int
    data: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# pivot
# ---------------------------------------------------------------------------

class PivotRequest(BaseModel):
    path: str
    rows: list[str]
    columns: list[str]
    measures: list[AggMeasure]


class PivotResult(BaseModel):
    row_count: int
    col_count: int
    data: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    path: str
    column: str
    x: str
    group: str | None = None
    horizon: int = 24
    model: str = "ridge"  # "ridge" | "gbr" | "xgboost"
    period: int = 24


class ForecastSeries(BaseModel):
    group: str
    rmse: float
    forecast: list[float]


class ForecastResult(BaseModel):
    model_used: str
    series: list[ForecastSeries]


# ---------------------------------------------------------------------------
# trading: technical indicators
# ---------------------------------------------------------------------------

class IndicatorRequest(BaseModel):
    path: str
    column: str = "close"
    timestamp: str = "ts"
    indicators: list[str]  # ["rsi", "macd", "bb", "ema_20", "sma_50"]


class IndicatorResult(BaseModel):
    rows: int
    columns: list[str]
    data: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# trading: signal detection
# ---------------------------------------------------------------------------

class SignalRequest(BaseModel):
    path: str
    close_col: str = "close"
    timestamp_col: str = "ts"


class Signal(BaseModel):
    kind: str  # "golden_cross" | "death_cross"
    index: int
    timestamp: Any
    price: float


class SignalResult(BaseModel):
    count: int
    signals: list[Signal]


# ---------------------------------------------------------------------------
# finance risk metrics  (GET /api/v2/analysis/finance)
# ---------------------------------------------------------------------------

class FinanceRequest(BaseModel):
    path: str
    column: str = "close"
    timestamp: str | None = None
    risk_free_rate: float = 0.0


class FinanceMetrics(BaseModel):
    total_return: float
    cagr: float
    ann_return: float
    ann_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float


class FinanceResult(BaseModel):
    ema: list[float]
    drawdown: list[float]
    metrics: FinanceMetrics
