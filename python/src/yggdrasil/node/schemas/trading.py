from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import StrictModel


class PriceQuote(StrictModel):
    symbol: str
    price: float
    change_pct: float
    volume: int
    timestamp_ms: int


class Position(StrictModel):
    id: int
    symbol: str
    qty: float
    avg_price: float
    current_price: float
    pnl: float
    pnl_pct: float


class OrderCreate(StrictModel):
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = None


class Order(StrictModel):
    id: int
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    filled_qty: float
    order_type: Literal["market", "limit"]
    limit_price: float | None
    status: Literal["pending", "filled", "cancelled", "rejected"]
    avg_fill_price: float | None
    created_at: int
    filled_at: int | None


class PortfolioSummary(StrictModel):
    cash: float
    equity: float
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    positions: list[Position]


class TradingSignal(StrictModel):
    symbol: str
    signal: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    confidence: float
    reason: str
    indicators: dict[str, float]
    timestamp_ms: int


class WatchlistEntry(StrictModel):
    symbol: str
    added_at: int


class WatchlistAdd(StrictModel):
    symbol: str


class PriceAlert(StrictModel):
    id: int
    symbol: str
    condition: Literal["above", "below"]
    threshold: float
    triggered: bool
    created_at: int
    triggered_at: int | None


class PriceAlertCreate(StrictModel):
    symbol: str
    condition: Literal["above", "below"]
    threshold: float


class TradeHistoryEntry(StrictModel):
    order_id: int
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    price: float
    realized_pnl: float | None
    timestamp_ms: int


class PricesResponse(StrictModel):
    prices: list[PriceQuote]


class OrdersResponse(StrictModel):
    orders: list[Order]


class WatchlistResponse(StrictModel):
    entries: list[WatchlistEntry]


class SignalsResponse(StrictModel):
    signals: list[TradingSignal]


class AlertsResponse(StrictModel):
    alerts: list[PriceAlert]


class TradeHistoryResponse(StrictModel):
    trades: list[TradeHistoryEntry]


class OrderResponse(StrictModel):
    order: Order


class AlertResponse(StrictModel):
    alert: PriceAlert


class WatchlistEntryResponse(StrictModel):
    entry: WatchlistEntry


# Per-symbol base price for the deterministic simulator. Keeps the mock
# numbers in a believable range without external data.
DEFAULT_SYMBOLS: dict[str, float] = {
    "AAPL": 184.0,
    "MSFT": 412.0,
    "GOOGL": 168.0,
    "TSLA": 248.0,
    "AMZN": 178.0,
    "NVDA": 875.0,
    "BTC-USD": 67_500.0,
    "ETH-USD": 3_450.0,
    "SOL-USD": 152.0,
}
