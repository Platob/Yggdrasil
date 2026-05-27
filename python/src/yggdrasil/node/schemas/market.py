from __future__ import annotations

from typing import Optional

from pydantic import Field

from .common import StrictModel


class FxRateEntry(StrictModel):
    source: str
    target: str
    pair: str            # "EUR/USD"
    value: float
    from_timestamp: str
    to_timestamp: str
    sampling: str


class FxLatestResponse(StrictModel):
    rates: list[FxRateEntry]
    cached: bool
    fetched_at: str


class FxHistoryPoint(StrictModel):
    from_timestamp: str
    to_timestamp: str
    value: float


class FxHistoryResponse(StrictModel):
    source: str
    target: str
    pair: str
    sampling: str
    points: list[FxHistoryPoint]


class FxConvertResponse(StrictModel):
    source: str
    target: str
    amount: float
    result: float
    rate: float
    at: Optional[str] = None


class WatchlistEntry(StrictModel):
    pair: str    # "EUR/USD"
    source: str
    target: str


class WatchlistResponse(StrictModel):
    pairs: list[WatchlistEntry]


class WatchlistAddRequest(StrictModel):
    pair: str    # "EUR/USD" format


class MarketErrorResponse(StrictModel):
    detail: str
    pairs_requested: list[str] = Field(default_factory=list)
