"""Market data schemas — candles, ticks, order books, asset metadata."""
from __future__ import annotations

from .base import StrictModel

__all__ = [
    "Candle",
    "Tick",
    "OrderBook",
    "AssetInfo",
    "MarketDataResponse",
]


class Candle(StrictModel):
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    interval: str


class Tick(StrictModel):
    ts: int
    symbol: str
    price: float
    volume: float
    side: str


class OrderBook(StrictModel):
    ts: int
    symbol: str
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


class AssetInfo(StrictModel):
    symbol: str
    name: str
    type: str
    currency: str
    exchange: str | None = None


class MarketDataResponse(StrictModel):
    symbol: str
    candles: list[Candle]
    count: int
