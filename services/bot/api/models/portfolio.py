from __future__ import annotations
from datetime import datetime
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


class Trade(BaseModel):
    id: int
    symbol: str
    side: TradeSide
    quantity: float
    price: float
    fee: float = 0.0
    status: TradeStatus = TradeStatus.FILLED
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None

    @property
    def notional(self) -> float:
        return self.quantity * self.price


class Position(BaseModel):
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    weight: float = 0.0  # portfolio weight %

    def update_price(self, price: float) -> None:
        self.current_price = price
        self.market_value = self.quantity * price
        cost_basis = self.quantity * self.avg_cost
        self.unrealized_pnl = self.market_value - cost_basis
        self.unrealized_pnl_pct = (self.unrealized_pnl / cost_basis * 100) if cost_basis else 0.0


class PnL(BaseModel):
    total_value: float
    cash: float
    invested: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    total_pnl: float
    total_pnl_pct: float
    day_pnl: float = 0.0
    day_pnl_pct: float = 0.0


class Portfolio(BaseModel):
    id: int = 1
    name: str = "Main"
    cash: float = 100_000.0
    positions: dict[str, Position] = Field(default_factory=dict)
    trades: list[Trade] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def pnl(self) -> PnL:
        invested = sum(p.market_value for p in self.positions.values())
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        realized = sum(p.realized_pnl for p in self.positions.values())
        total_value = self.cash + invested
        cost_basis = sum(p.quantity * p.avg_cost for p in self.positions.values())
        total_pnl = unrealized + realized
        return PnL(
            total_value=total_value,
            cash=self.cash,
            invested=invested,
            unrealized_pnl=unrealized,
            unrealized_pnl_pct=(unrealized / cost_basis * 100) if cost_basis else 0.0,
            realized_pnl=realized,
            total_pnl=total_pnl,
            total_pnl_pct=(total_pnl / 100_000 * 100),
        )
