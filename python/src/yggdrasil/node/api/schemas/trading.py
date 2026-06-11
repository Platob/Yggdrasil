"""Request models for the trading endpoints (lazy Polars over parquet/arrow)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Strategy = Literal["ema_cross", "rsi_mean_reversion", "macd", "buy_and_hold"]


class IndicatorsRequest(BaseModel):
    path: str
    column: str
    ts_column: str | None = None
    max_points: int | None = 2000


class SignalsRequest(BaseModel):
    path: str
    column: str
    ts_column: str | None = None
    max_points: int | None = 2000


class BacktestRequest(BaseModel):
    path: str
    column: str
    strategy: Strategy = "ema_cross"
    initial_cash: float = 10_000.0
    ts_column: str | None = None
    max_points: int | None = 2000


class CorrelationRequest(BaseModel):
    paths: list[str]
    column: str = "close"


class PortfolioRequest(BaseModel):
    paths: list[str]
    weights: list[float] | None = None
    column: str = "close"
