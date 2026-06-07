"""Wire contracts for the node API — strict pydantic request/response models.

Grouped by domain: :mod:`market` (candles/ticks/books), :mod:`portfolio`
(positions/orders/trades), :mod:`analysis` (polars-backed analytics). All
derive from :class:`StrictModel` (``extra="forbid"``) so a malformed client
payload fails loudly at the boundary.
"""
from __future__ import annotations

from .base import StrictModel, make_id, now_ms
from .market import AssetInfo, Candle, MarketDataResponse, OrderBook, Tick
from .portfolio import (
    CreateOrderRequest,
    Order,
    Portfolio,
    PortfolioSummary,
    Position,
    Trade,
)
from .analysis import (
    AggMeasure,
    AggregateRequest,
    AggregateResponse,
    OhlcRequest,
    OhlcResponse,
    PivotRequest,
    PivotResponse,
    SeriesRequest,
    SeriesResponse,
)

__all__ = [
    "StrictModel",
    "make_id",
    "now_ms",
    "Candle",
    "Tick",
    "OrderBook",
    "AssetInfo",
    "MarketDataResponse",
    "Position",
    "Order",
    "Trade",
    "Portfolio",
    "PortfolioSummary",
    "CreateOrderRequest",
    "AggMeasure",
    "AggregateRequest",
    "AggregateResponse",
    "SeriesRequest",
    "SeriesResponse",
    "OhlcRequest",
    "OhlcResponse",
    "PivotRequest",
    "PivotResponse",
]
