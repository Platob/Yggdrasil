"""Analysis request/response schemas — aggregate, series, OHLC, pivot."""
from __future__ import annotations

from typing import Any

from .base import StrictModel

__all__ = [
    "AggregateRequest",
    "AggregateRow",
    "AggregateResponse",
    "SeriesRequest",
    "SeriesResponse",
    "OhlcRequest",
    "OhlcResponse",
    "PivotRequest",
    "PivotResponse",
    "ForecastRequest",
    "ForecastResponse",
    "ForecastSeries",
    "AggMeasure",
]


class AggMeasure(StrictModel):
    column: str
    agg: str


class AggregateRequest(StrictModel):
    """Simplified single-column aggregate used by the dashboard.

    ``column="*"`` with ``agg="count"`` returns the file schema (column names)
    in ``columns`` with empty ``rows`` — used for column discovery.
    ``agg="series"`` returns the raw values in row order, no grouping.
    """
    path: str
    column: str
    agg: str  # "count","sum","mean","min","max","std","median","series"
    group_by: str | None = None


class AggregateRow(StrictModel):
    group: str | int | float | None
    value: float | int | None


class AggregateResponse(StrictModel):
    path: str
    column: str
    agg: str
    group_by: str | None
    rows: list[AggregateRow]
    columns: list[str]
    elapsed_ms: float


class SeriesRequest(StrictModel):
    path: str
    column: str
    points: int = 500
    group_by: str | None = None


class SeriesResponse(StrictModel):
    x: list[float]
    y: list[float]
    label: str
    points: int
    elapsed_ms: float


class OhlcRequest(StrictModel):
    path: str
    column: str
    buckets: int = 120


class OhlcResponse(StrictModel):
    bars: int
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    elapsed_ms: float


class PivotRequest(StrictModel):
    path: str
    rows: list[str]
    columns: list[str]
    measures: list[AggMeasure]


class PivotResponse(StrictModel):
    data: list[dict[str, Any]]
    row_count: int
    col_count: int
    elapsed_ms: float


class ForecastRequest(StrictModel):
    path: str
    column: str  # target column
    x: str  # time/index column
    group: str | None = None  # optional grouping column
    horizon: int = 24  # bars to forecast
    model: str = "ridge"  # "ridge" | "gbr" | "xgboost"
    period: int | None = None  # seasonality period for feature engineering


class ForecastSeries(StrictModel):
    group: str | None
    forecast: list[float]
    rmse: float


class ForecastResponse(StrictModel):
    path: str
    column: str
    model_used: str  # actual model that ran (may differ if requested model unavailable)
    horizon: int
    series: list[ForecastSeries]
    elapsed_ms: float
