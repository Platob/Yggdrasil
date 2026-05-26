from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class PriceQuote(StrictModel):
    symbol: str
    price: float
    currency: str = "USD"
    source: str = "unknown"
    timestamp: str
    stale: bool = False


class PricesResponse(StrictModel):
    prices: dict[str, PriceQuote]
    timestamp: str


class PriceHistoryResponse(StrictModel):
    symbol: str
    prices: list[float]
    timestamps: list[str]


class PortfolioPositionEntry(StrictModel):
    symbol: str
    quantity: float
    avg_cost: float
    currency: str = "USD"
    current_price: float | None = None
    pnl: float | None = None
    pnl_pct: float | None = None


class PortfolioResponse(StrictModel):
    positions: list[PortfolioPositionEntry]
    total_value: float
    total_pnl: float
    currency: str = "USD"
    timestamp: str


class PositionCreate(StrictModel):
    symbol: str
    quantity: float
    avg_cost: float
    currency: str = "USD"


class TechnicalIndicators(StrictModel):
    symbol: str
    sma_20: float | None = None
    sma_50: float | None = None
    ema_20: float | None = None
    rsi_14: float | None = None
    price: float | None = None
    timestamp: str


class PriceAlert(StrictModel):
    id: int
    symbol: str
    condition: str = Field(..., description="'above' or 'below'")
    price: float
    created_at: str
    triggered_at: str | None = None


class AlertCreate(StrictModel):
    symbol: str
    condition: str = Field(..., description="'above' or 'below'")
    price: float


class AlertsResponse(StrictModel):
    alerts: list[PriceAlert]
