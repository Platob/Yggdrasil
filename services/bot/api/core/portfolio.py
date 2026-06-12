from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import xxhash

from ..models.portfolio import Portfolio, Position, Trade, TradeSide, TradeStatus

# In-memory store (production: replace with Databricks Delta or Postgres)
_portfolios: dict[int, Portfolio] = {}


def _make_trade_id(symbol: str) -> int:
    ts_ms = int(time.time() * 1000)
    h = xxhash.xxh32(symbol.encode()).intdigest()
    return (h << 32) | (ts_ms & 0xFFFFFFFF)


def get_portfolio(pid: int = 1) -> Portfolio:
    if pid not in _portfolios:
        _portfolios[pid] = Portfolio(id=pid, cash=100_000.0)
    return _portfolios[pid]


def execute_trade(pid: int, symbol: str, side: TradeSide, quantity: float,
                  price: float, fee: float = 0.0, notes: Optional[str] = None) -> Trade:
    portfolio = get_portfolio(pid)
    cost = quantity * price + fee

    if side == TradeSide.BUY:
        if portfolio.cash < cost:
            raise ValueError(f"Insufficient cash: need {cost:.2f}, have {portfolio.cash:.2f}")
        portfolio.cash -= cost
        pos = portfolio.positions.get(symbol)
        if pos:
            total_qty = pos.quantity + quantity
            pos.avg_cost = (pos.quantity * pos.avg_cost + quantity * price) / total_qty
            pos.quantity = total_qty
        else:
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
                market_value=quantity * price,
            )
    else:  # SELL
        pos = portfolio.positions.get(symbol)
        if not pos or pos.quantity < quantity:
            raise ValueError(f"Insufficient position in {symbol}")
        realized = (price - pos.avg_cost) * quantity - fee
        pos.realized_pnl += realized
        pos.quantity -= quantity
        portfolio.cash += quantity * price - fee
        if pos.quantity < 1e-8:
            del portfolio.positions[symbol]

    # Recompute weights
    total_mv = sum(p.market_value for p in portfolio.positions.values())
    for p in portfolio.positions.values():
        p.weight = (p.market_value / total_mv * 100) if total_mv else 0.0

    trade = Trade(
        id=_make_trade_id(symbol),
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
        status=TradeStatus.FILLED,
        notes=notes,
    )
    portfolio.trades.append(trade)
    portfolio.updated_at = datetime.utcnow()
    return trade


def update_prices(pid: int, prices: dict[str, float]) -> None:
    portfolio = get_portfolio(pid)
    total_mv = 0.0
    for symbol, price in prices.items():
        if symbol in portfolio.positions:
            portfolio.positions[symbol].update_price(price)
            total_mv += portfolio.positions[symbol].market_value
    for p in portfolio.positions.values():
        p.weight = (p.market_value / total_mv * 100) if total_mv else 0.0
