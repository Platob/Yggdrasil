"""Request/response schemas for the analysis endpoints."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class AggMeasure(BaseModel):
    column: str
    agg: Literal["mean", "sum", "min", "max", "count", "std"] = "mean"


class AggregateRequest(BaseModel):
    path: str
    group_by: list[str]
    measures: list[AggMeasure]


class AggregateResult(BaseModel):
    group_count: int
    rows: list[dict]


class SeriesRequest(BaseModel):
    path: str
    column: str
    points: int = 800
    time_column: Optional[str] = None


class SeriesResult(BaseModel):
    x: list[float]
    y: list[float]


class OhlcRequest(BaseModel):
    path: str
    column: str
    buckets: int = 120
    time_column: Optional[str] = None


class OhlcResult(BaseModel):
    bars: int
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    times: list[float]


class PivotRequest(BaseModel):
    path: str
    rows: list[str]
    columns: list[str]
    measures: list[AggMeasure]


class PivotResult(BaseModel):
    row_count: int
    col_count: int
    rows: list[str]
    columns: list[str]
    data: list[list[float]]


class ForecastRequest(BaseModel):
    path: str
    column: str
    x: str
    group: Optional[str] = None
    horizon: int = 24
    model: Literal["ridge", "gbr", "xgboost", "auto"] = "auto"
    period: Optional[int] = None


class ForecastSeries(BaseModel):
    group: str
    forecast: list[float]
    rmse: float


class ForecastResult(BaseModel):
    model_used: str
    series: list[ForecastSeries]


class FinanceRequest(BaseModel):
    path: str
    column: str
    time_column: Optional[str] = None
    risk_free_rate: float = 0.0
    periods_per_year: int = 252  # 252 trading days, 365 calendar, 52 weekly


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
    metrics: FinanceMetrics
    ema: list[float]
    drawdown: list[float]
    cum_return: list[float]
