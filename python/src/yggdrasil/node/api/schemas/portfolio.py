"""Portfolio / trading schemas — positions, orders, trades, portfolios."""
from __future__ import annotations

from .base import StrictModel

__all__ = [
    "Position",
    "Order",
    "Trade",
    "Portfolio",
    "PortfolioSummary",
    "CreateOrderRequest",
]


class Position(StrictModel):
    id: int
    symbol: str
    side: str
    qty: float
    avg_entry: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    opened_at: int


class Order(StrictModel):
    id: int
    symbol: str
    side: str
    type: str
    qty: float
    price: float | None
    status: str
    created_at: int
    filled_at: int | None


class Trade(StrictModel):
    id: int
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    pnl: float
    ts: int


class Portfolio(StrictModel):
    id: int
    name: str
    equity: float
    cash: float
    margin_used: float
    total_pnl: float
    daily_pnl: float
    positions: list[Position]
    open_orders: list[Order]
    updated_at: int


class PortfolioSummary(StrictModel):
    equity: float
    cash: float
    total_pnl: float
    daily_pnl: float
    position_count: int
    open_order_count: int
    win_rate: float


class CreateOrderRequest(StrictModel):
    symbol: str
    side: str
    type: str = "market"
    qty: float
    price: float | None = None
