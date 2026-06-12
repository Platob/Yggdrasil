from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Ticker(BaseModel):
    symbol: str
    name: str
    exchange: str
    asset_type: str  # stock, crypto, forex, etf


class Quote(BaseModel):
    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    market_cap: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OHLCV(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None


class OrderBook(BaseModel):
    symbol: str
    bids: list[tuple[float, float]]  # (price, size)
    asks: list[tuple[float, float]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
