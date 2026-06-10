"""Request/response contracts for the analysis engine.

These shape the lazy-scan analytics: group-by aggregates, adaptive
downsampled series, OHLC resamples, cross-tab pivots, and forecasts over
engineered features. Every request names a ``path`` relative to the node
home; the service resolves it and reads through ``pl.scan_parquet`` so
projection pushdown only touches the columns named here.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AggMeasure(BaseModel):
    column: str
    agg: str  # "mean" | "sum" | "min" | "max" | "count" | "std"


class AggregateRequest(BaseModel):
    path: str
    group_by: list[str]
    measures: list[AggMeasure]


class AggregateResult(BaseModel):
    group_count: int
    columns: list[str]
    data: list[dict]


class SeriesRequest(BaseModel):
    path: str
    column: str
    points: int = 1000
    time_col: str | None = None


class SeriesResult(BaseModel):
    x: list[Any]
    y: list[float]
    column: str


class OhlcRequest(BaseModel):
    path: str
    column: str
    buckets: int = 100
    time_col: str | None = None


class OhlcBar(BaseModel):
    open: float
    high: float
    low: float
    close: float
    bucket: int


class OhlcResult(BaseModel):
    bars: int
    data: list[OhlcBar]


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
    model: str = "ridge"  # "ridge" | "gbr" | "xgboost"
    period: int | None = None


class ForecastSeries(BaseModel):
    group: str | None
    rmse: float
    x_future: list[Any]
    y_pred: list[float]


class ForecastResult(BaseModel):
    model_used: str
    series: list[ForecastSeries]
