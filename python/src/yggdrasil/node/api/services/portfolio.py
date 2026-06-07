"""Portfolio service — in-memory trading book with live mark-to-market.

Holds portfolios in a dict keyed by int64 ID and seeds one demo book on
construction so the UI has something to render immediately. Positions are
marked to market against :class:`MarketDataService` current prices on every
read, so unrealized P&L and equity move with the synthetic feed. Orders are
matched immediately (synthetic venue): market orders fill at the current
price, limit/stop orders are accepted as ``open`` and left resting.

State is process-local and non-durable by design — this is a trading
sandbox, not a ledger.
"""
from __future__ import annotations

from ..schemas.base import make_id, now_ms
from ..schemas.portfolio import (
    CreateOrderRequest,
    Order,
    Portfolio,
    PortfolioSummary,
    Position,
    Trade,
)
from .market import MarketDataService

__all__ = ["PortfolioService", "DEMO_PORTFOLIO_ID"]

DEMO_PORTFOLIO_ID = make_id("demo", ts_ms=0)

# (symbol, side, qty, avg_entry)
_DEMO_POSITIONS: list[tuple[str, str, float, float]] = [
    ("BTC/USD", "long", 0.75, 61500.0),
    ("ETH/USD", "long", 8.0, 3250.0),
    ("AAPL", "long", 120.0, 188.0),
    ("EUR/USD", "short", 50000.0, 1.0920),
    ("XAU/USD", "long", 15.0, 2300.0),
]

# (symbol, side, qty, price, fee, pnl, ts_offset_ms)
_DEMO_TRADES: list[tuple[str, str, float, float, float, float]] = [
    ("BTC/USD", "buy", 0.25, 58000.0, 14.5, 0.0),
    ("BTC/USD", "sell", 0.25, 62000.0, 15.5, 970.0),
    ("ETH/USD", "buy", 4.0, 3100.0, 6.2, 0.0),
    ("AAPL", "buy", 50.0, 185.0, 4.6, 0.0),
    ("AAPL", "sell", 50.0, 199.0, 4.9, 690.0),
    ("EUR/USD", "sell", 50000.0, 1.0920, 2.7, 0.0),
    ("GOOGL", "buy", 30.0, 170.0, 2.5, 0.0),
    ("GOOGL", "sell", 30.0, 165.0, 2.4, -150.0),
]


class PortfolioService:
    def __init__(self, market: MarketDataService) -> None:
        self._market = market
        self._portfolios: dict[int, Portfolio] = {}
        self._trades: dict[int, list[Trade]] = {}
        self._seed_demo()

    def _seed_demo(self) -> None:
        now = now_ms()
        positions: list[Position] = []
        for i, (symbol, side, qty, avg_entry) in enumerate(_DEMO_POSITIONS):
            positions.append(
                Position(
                    id=make_id(f"demo-pos-{symbol}", ts_ms=i),
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    avg_entry=avg_entry,
                    current_price=avg_entry,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    opened_at=now - 86_400_000 * (i + 1),
                )
            )

        trades: list[Trade] = []
        for i, (symbol, tside, qty, price, fee, pnl) in enumerate(_DEMO_TRADES):
            trades.append(
                Trade(
                    id=make_id(f"demo-trade-{i}", ts_ms=i),
                    symbol=symbol,
                    side=tside,
                    qty=qty,
                    price=price,
                    fee=fee,
                    pnl=pnl,
                    ts=now - 3_600_000 * (len(_DEMO_TRADES) - i),
                )
            )

        portfolio = Portfolio(
            id=DEMO_PORTFOLIO_ID,
            name="Demo Book",
            equity=0.0,
            cash=100_000.0,
            margin_used=0.0,
            total_pnl=0.0,
            daily_pnl=0.0,
            positions=positions,
            open_orders=[],
            updated_at=now,
        )
        self._portfolios[DEMO_PORTFOLIO_ID] = portfolio
        self._trades[DEMO_PORTFOLIO_ID] = trades

    def _mark(self, portfolio: Portfolio) -> Portfolio:
        """Re-price positions and recompute equity/P&L against live prices."""
        margin_used = 0.0
        unrealized_total = 0.0
        for pos in portfolio.positions:
            price = self._market.current_price(pos.symbol)
            direction = 1.0 if pos.side == "long" else -1.0
            pos.current_price = price
            pos.unrealized_pnl = (price - pos.avg_entry) * pos.qty * direction
            unrealized_total += pos.unrealized_pnl
            margin_used += abs(pos.qty) * pos.avg_entry

        realized_total = sum(t.pnl for t in self._trades.get(portfolio.id, []))
        portfolio.margin_used = margin_used
        portfolio.total_pnl = realized_total + unrealized_total
        portfolio.daily_pnl = unrealized_total
        portfolio.equity = portfolio.cash + unrealized_total
        portfolio.updated_at = now_ms()
        return portfolio

    async def get_portfolio(self, portfolio_id: int) -> Portfolio:
        portfolio = self._portfolios.get(portfolio_id)
        if portfolio is None:
            raise KeyError(portfolio_id)
        return self._mark(portfolio)

    async def get_summary(self, portfolio_id: int) -> PortfolioSummary:
        portfolio = await self.get_portfolio(portfolio_id)
        closed = [t for t in self._trades.get(portfolio_id, []) if t.pnl != 0.0]
        wins = sum(1 for t in closed if t.pnl > 0.0)
        win_rate = (wins / len(closed)) if closed else 0.0
        return PortfolioSummary(
            equity=portfolio.equity,
            cash=portfolio.cash,
            total_pnl=portfolio.total_pnl,
            daily_pnl=portfolio.daily_pnl,
            position_count=len(portfolio.positions),
            open_order_count=len(portfolio.open_orders),
            win_rate=win_rate,
        )

    async def open_order(self, portfolio_id: int, req: CreateOrderRequest) -> Order:
        portfolio = self._portfolios.get(portfolio_id)
        if portfolio is None:
            raise KeyError(portfolio_id)
        if req.symbol not in self._market_symbols():
            raise ValueError(req.symbol)

        now = now_ms()
        order_id = make_id(f"order-{portfolio_id}-{req.symbol}-{now}", ts_ms=now)

        if req.type == "market":
            fill_price = self._market.current_price(req.symbol)
            order = Order(
                id=order_id,
                symbol=req.symbol,
                side=req.side,
                type=req.type,
                qty=req.qty,
                price=fill_price,
                status="filled",
                created_at=now,
                filled_at=now,
            )
            fee = fill_price * req.qty * 0.0004
            self._trades.setdefault(portfolio_id, []).append(
                Trade(
                    id=make_id(f"trade-{order_id}", ts_ms=now),
                    symbol=req.symbol,
                    side=req.side,
                    qty=req.qty,
                    price=fill_price,
                    fee=fee,
                    pnl=0.0,
                    ts=now,
                )
            )
            cash_delta = -fill_price * req.qty if req.side == "buy" else fill_price * req.qty
            portfolio.cash += cash_delta - fee
        else:
            order = Order(
                id=order_id,
                symbol=req.symbol,
                side=req.side,
                type=req.type,
                qty=req.qty,
                price=req.price,
                status="open",
                created_at=now,
                filled_at=None,
            )
            portfolio.open_orders.append(order)
        return order

    async def cancel_order(self, portfolio_id: int, order_id: int) -> bool:
        portfolio = self._portfolios.get(portfolio_id)
        if portfolio is None:
            raise KeyError(portfolio_id)
        before = len(portfolio.open_orders)
        portfolio.open_orders = [o for o in portfolio.open_orders if o.id != order_id]
        return len(portfolio.open_orders) < before

    async def get_trades(self, portfolio_id: int, limit: int) -> list[Trade]:
        if portfolio_id not in self._portfolios:
            raise KeyError(portfolio_id)
        trades = self._trades.get(portfolio_id, [])
        return sorted(trades, key=lambda t: t.ts, reverse=True)[:limit]

    def _market_symbols(self) -> set[str]:
        from .market import ASSETS

        return set(ASSETS)
