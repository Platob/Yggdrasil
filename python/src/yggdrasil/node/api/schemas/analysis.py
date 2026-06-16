"""Analysis request/response schemas.

These drive the trading-focused parquet analytics in
:class:`yggdrasil.node.api.services.analysis.AnalysisService`: aggregate,
downsampled series, OHLC resample, cross-tab pivot, and forecast.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ----------------------------- requests --------------------------------- #

class AggMeasure(BaseModel):
    """A single aggregation: ``agg`` of ``column`` (mean/sum/min/max/...)."""

    column: str
    agg: str = "sum"


class AggregateRequest(BaseModel):
    path: str
    group_by: list[str]
    measures: list[AggMeasure]


class SeriesRequest(BaseModel):
    path: str
    column: str
    points: int = 1000


class OhlcRequest(BaseModel):
    path: str
    column: str
    buckets: int = 120


class PivotRequest(BaseModel):
    path: str
    rows: list[str]
    columns: list[str]
    measures: list[AggMeasure]


class ForecastRequest(BaseModel):
    path: str
    column: str
    x: str
    group: str | None = None
    horizon: int = 24
    model: str = "ridge"
    period: int = 24


# ----------------------------- responses -------------------------------- #

class AggregateResult(BaseModel):
    rows: list[dict]
    group_count: int


class SeriesResult(BaseModel):
    x: list
    y: list
    points: int


class OhlcResult(BaseModel):
    bars: int
    data: list[dict]


class PivotResult(BaseModel):
    row_count: int
    col_count: int
    data: list[dict]


class ForecastSeries(BaseModel):
    group: str | None = None
    rmse: float
    predictions: list


class ForecastResult(BaseModel):
    series: list[ForecastSeries]
    model_used: str
