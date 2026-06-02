from __future__ import annotations
from .common import StrictModel


class MarketQuote(StrictModel):
    symbol: str
    price: float | None = None
    prev_close: float | None = None
    change: float | None = None
    change_pct: float | None = None
    open: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    volume: float | None = None
    market_cap: float | None = None
    name: str | None = None
    currency: str | None = None


class OHLCVBar(StrictModel):
    t: str        # ISO timestamp string
    o: float | None
    h: float | None
    l: float | None
    c: float | None
    v: float | None


class MarketOHLCV(StrictModel):
    symbol: str
    period: str
    interval: str
    bars: list[OHLCVBar]
    currency: str | None = None


class MarketSearchResult(StrictModel):
    symbol: str
    name: str
    exchange: str | None = None
    type: str | None = None
