"""Request models for the analysis endpoints (lazy Polars over parquet/arrow)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Agg = Literal["mean", "sum", "count", "min", "max", "std"]


class AggMeasure(BaseModel):
    column: str
    agg: Agg


class AggregateRequest(BaseModel):
    path: str
    group_by: list[str]
    measures: list[AggMeasure]


class SeriesRequest(BaseModel):
    path: str
    column: str
    points: int = 800


class OhlcRequest(BaseModel):
    path: str
    column: str
    buckets: int = 120
    ts_column: str | None = None


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
    model: Literal["auto", "ridge", "gbr", "xgboost"] = "auto"
    period: int | None = None
