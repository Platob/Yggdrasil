from __future__ import annotations

from pydantic import BaseModel


class StrictModel(BaseModel):
    model_config = {"extra": "forbid"}


class TradeSignal(StrictModel):
    id: int
    func_id: int | None = None
    name: str
    symbol: str
    direction: str  # "buy" | "sell" | "hold"
    confidence: float  # 0.0–1.0
    price: float | None = None
    metadata: dict = {}
    created_at: str
    expires_at: str | None = None


class TradeSignalCreate(StrictModel):
    func_id: int | None = None
    name: str
    symbol: str
    direction: str
    confidence: float = 1.0
    price: float | None = None
    metadata: dict = {}
    expires_at: str | None = None


class Position(StrictModel):
    symbol: str
    qty: float
    avg_price: float
    current_price: float | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    opened_at: str
    updated_at: str


class PositionUpsert(StrictModel):
    symbol: str
    qty: float
    avg_price: float
    current_price: float | None = None


class Portfolio(StrictModel):
    positions: list[Position]
    total_pnl: float
    total_pnl_pct: float
    updated_at: str


class SignalListResponse(StrictModel):
    node_id: str
    signals: list[TradeSignal]


class PortfolioResponse(StrictModel):
    node_id: str
    portfolio: Portfolio
