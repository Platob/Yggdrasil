from __future__ import annotations
from typing import Any
from .common import StrictModel


class IndicatorSpec(StrictModel):
    type: str  # rsi|macd|bb|sma|ema|atr|vwap|obv|stoch
    period: int | None = None      # main period (RSI=14, SMA/EMA/BB=20, ATR=14, Stoch=14)
    fast: int | None = None        # MACD fast (12)
    slow: int | None = None        # MACD slow (26)
    signal: int | None = None      # MACD signal (9)
    std_dev: float | None = None   # BB std dev multiplier (2.0)
    d_period: int | None = None    # Stochastic %D period (3)


class TechnicalRequest(StrictModel):
    path: str                      # parquet/csv/ndjson file with price data
    close: str = "close"           # price/close column name
    high: str | None = None        # high column (ATR/VWAP/Stoch)
    low: str | None = None         # low column  (ATR/VWAP/Stoch)
    volume: str | None = None      # volume column (VWAP/OBV)
    x: str | None = None           # timestamp/index column
    indicators: list[IndicatorSpec]
    filters: list[Any] = []


class IndicatorSeries(StrictModel):
    name: str                      # e.g. "RSI(14)", "MACD(12,26,9)", "BB_upper(20)"
    series: list[float | None]


class TechnicalResult(StrictModel):
    node_id: str
    path: str
    x: list[Any]                   # index values (timestamp or row index)
    close: list[float | None]      # close prices (for reference)
    indicators: list[IndicatorSeries]
    source_rows: int
