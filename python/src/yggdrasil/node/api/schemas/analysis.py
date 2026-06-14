"""Analysis schemas — aggregate, series, OHLC, pivot, forecast."""
from __future__ import annotations

from pydantic import BaseModel


class AggMeasure(BaseModel):
    column: str
    agg: str = "mean"


class AggregateRequest(BaseModel):
    path: str
    group_by: list[str]
    measures: list[AggMeasure]


class AggregateResult(BaseModel):
    group_count: int
    data: list[dict]


class SeriesRequest(BaseModel):
    path: str
    column: str
    points: int = 800


class SeriesResult(BaseModel):
    x: list[float]
    y: list[float]


class OhlcRequest(BaseModel):
    path: str
    column: str
    buckets: int = 120


class OhlcResult(BaseModel):
    bars: int
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    timestamps: list[int]


class PivotRequest(BaseModel):
    path: str
    rows: list[str]
    columns: list[str]
    measures: list[AggMeasure]


class PivotResult(BaseModel):
    row_count: int
    col_count: int
    data: list[dict]


class ForecastRequest(BaseModel):
    path: str
    column: str
    x: str
    group: str | None = None
    horizon: int = 24
    model: str = "ridge"
    period: int = 24


class ForecastSeries(BaseModel):
    group: str | None
    values: list[float]
    rmse: float


class ForecastResult(BaseModel):
    model_used: str
    series: list[ForecastSeries]
