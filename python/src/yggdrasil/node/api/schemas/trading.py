from __future__ import annotations

from typing import Any

from .common import StrictModel


class IndicatorRequest(StrictModel):
    path: str
    column: str                    # price column
    x: str | None = None           # time/order column
    indicators: list[str] = ["rsi", "macd", "bb"]  # which to compute
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14           # needs high/low columns
    high: str | None = None
    low: str | None = None
    limit: int = 2000
    filters: list[Any] = []        # FilterSpec-compatible


class IndicatorResult(StrictModel):
    node_id: str
    path: str
    column: str
    index: list[Any]
    value: list[float | None]
    rsi: list[float | None] | None = None
    macd: list[float | None] | None = None
    macd_signal: list[float | None] | None = None
    macd_hist: list[float | None] | None = None
    bb_upper: list[float | None] | None = None
    bb_mid: list[float | None] | None = None
    bb_lower: list[float | None] | None = None
    atr: list[float | None] | None = None
    source_rows: int
    truncated: bool


class CorrelationRequest(StrictModel):
    paths: list[str]               # 2–10 file paths
    column: str                    # the numeric column to use in each
    labels: list[str] = []         # display names (auto = basename)
    method: str = "pearson"        # pearson | spearman
    limit: int = 2000


class CorrelationResult(StrictModel):
    node_id: str
    labels: list[str]
    method: str
    matrix: list[list[float | None]]  # n×n correlation matrix
    source_rows: list[int]         # rows used per asset


class PortfolioAsset(StrictModel):
    path: str
    column: str
    label: str = ""
    weight: float = 1.0            # portfolio weight (will be normalized)


class PortfolioRequest(StrictModel):
    assets: list[PortfolioAsset]   # 2–20 assets
    periods_per_year: int = 252
    risk_free: float = 0.0
    limit: int = 2000


class PortfolioMetrics(StrictModel):
    total_return: float | None = None
    cagr: float | None = None
    ann_return: float | None = None
    ann_volatility: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown: float | None = None
    calmar: float | None = None
    beta: float | None = None       # vs equal-weight benchmark
    alpha: float | None = None


class PortfolioResult(StrictModel):
    node_id: str
    labels: list[str]
    weights: list[float]
    index: list[Any]
    portfolio_value: list[float | None]
    drawdown: list[float | None]
    individual_returns: list[list[float | None]]
    metrics: PortfolioMetrics
    correlation_matrix: list[list[float | None]]
    source_rows: int


class VaRRequest(StrictModel):
    path: str
    column: str
    method: str = "historical"     # historical | parametric | cornish_fisher
    confidence: float = 0.95       # VaR confidence level
    horizon: int = 1               # days ahead
    periods_per_year: int = 252
    limit: int = 2000


class VaRResult(StrictModel):
    node_id: str
    path: str
    column: str
    method: str
    confidence: float
    horizon: int
    var: float | None              # VaR as a fraction (negative = loss)
    cvar: float | None             # Conditional VaR (Expected Shortfall)
    var_pct: float | None          # var × 100 for display
    cvar_pct: float | None
    ann_volatility: float | None
    source_rows: int


class SignalRequest(StrictModel):
    path: str
    column: str
    x: str | None = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal_period: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    limit: int = 2000


class TradeSignal(StrictModel):
    index: Any                     # timestamp / bar index
    action: str                    # BUY | SELL | HOLD
    strength: float                # 0..1
    reasons: list[str]             # e.g. ["RSI oversold", "MACD crossover"]
    rsi: float | None = None
    macd_hist: float | None = None
    bb_position: float | None = None  # 0=at lower, 0.5=mid, 1=upper


class SignalResult(StrictModel):
    node_id: str
    path: str
    column: str
    signals: list[TradeSignal]
    last_action: str
    buy_count: int
    sell_count: int
    source_rows: int
