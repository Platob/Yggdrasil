"""Paper-trading service.

Self-contained: prices are generated from a deterministic sine-wave
formula keyed by symbol + wall-clock time, so the entire stack works
without any external market-data feed. Orders, positions, watchlist,
alerts and signals are kept in process memory with locking.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from threading import Lock

from ..config import Settings
from ..exceptions import BotError, NotFoundError
from ..ids import make_id, make_id_pair
from ..schemas.trading import (
    DEFAULT_SYMBOLS,
    Order,
    OrderCreate,
    PortfolioSummary,
    Position,
    PriceAlert,
    PriceAlertCreate,
    PriceQuote,
    TradeHistoryEntry,
    TradingSignal,
    WatchlistEntry,
)

LOGGER = logging.getLogger(__name__)

_STARTING_CASH = 100_000.0
_HISTORY_POINTS = 100  # synthetic price history depth for indicators
_MAX_TRADES = 500


def _now_ms() -> int:
    return int(time.time() * 1000)


def _symbol_seed(symbol: str) -> int:
    """Stable integer seed derived from the symbol — used to phase-shift waves."""
    return sum(ord(c) * (i + 1) for i, c in enumerate(symbol))


def _simulate_price(symbol: str, base: float, t: float) -> tuple[float, float]:
    """Return ``(price, change_pct)`` for a symbol at wall-clock time ``t``.

    Combines two sine waves of different periods plus a small high-freq
    jitter so the chart never looks flat. Returns the 24h change vs. base.
    """
    seed = _symbol_seed(symbol)
    # Period 1: slow drift (~10 minutes); period 2: medium swing (~90s).
    slow = math.sin((t / 600.0) + seed * 0.1) * 0.025
    medium = math.sin((t / 90.0) + seed * 0.3) * 0.012
    jitter = math.sin((t / 7.0) + seed) * 0.003
    drift = slow + medium + jitter
    price = base * (1.0 + drift)
    # change_pct: simulate "yesterday's close" with same formula offset 86_400s.
    prev = base * (1.0 + math.sin(((t - 86_400) / 600.0) + seed * 0.1) * 0.025
                   + math.sin(((t - 86_400) / 90.0) + seed * 0.3) * 0.012)
    change_pct = (price - prev) / prev * 100.0 if prev else 0.0
    return round(price, 4), round(change_pct, 3)


def _simulate_history(symbol: str, base: float, t: float, n: int = _HISTORY_POINTS,
                      step_s: float = 60.0) -> list[float]:
    """Return ``n`` synthetic price points ending at ``t``, ``step_s`` apart."""
    out: list[float] = []
    for i in range(n):
        ti = t - (n - 1 - i) * step_s
        price, _ = _simulate_price(symbol, base, ti)
        out.append(price)
    return out


def _ma(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def _rsi(values: list[float], period: int = 14) -> float | None:
    """Standard Wilder RSI on closing prices."""
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return round(100.0 - 100.0 / (1.0 + rs), 2)


class TradingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()

        self._cash: float = _STARTING_CASH
        self._positions: dict[str, dict] = {}  # symbol -> {qty, avg_price}
        self._orders: dict[int, Order] = {}
        self._watchlist: dict[str, WatchlistEntry] = {}
        self._alerts: dict[int, PriceAlert] = {}
        self._trades: deque[TradeHistoryEntry] = deque(maxlen=_MAX_TRADES)

        # Seed default symbols into the watchlist so the UI has content
        # on first load.
        now = _now_ms()
        for sym in DEFAULT_SYMBOLS:
            self._watchlist[sym] = WatchlistEntry(symbol=sym, added_at=now)

    # -- prices -----------------------------------------------------------

    def _base_for(self, symbol: str) -> float:
        sym = symbol.upper()
        base = DEFAULT_SYMBOLS.get(sym)
        if base is None:
            # Unknown ticker: derive a deterministic base from its name so
            # the UI can still trade it without us hardcoding every symbol.
            base = 50.0 + (_symbol_seed(sym) % 500)
        return base

    def get_price(self, symbol: str) -> PriceQuote:
        sym = symbol.upper()
        base = self._base_for(sym)
        t = time.time()
        price, change_pct = _simulate_price(sym, base, t)
        # Volume scales with abs(change_pct) for realism.
        seed = _symbol_seed(sym)
        volume = int(1_000_000 + (seed % 9_000_000) + abs(change_pct) * 250_000)
        return PriceQuote(
            symbol=sym, price=price, change_pct=change_pct,
            volume=volume, timestamp_ms=int(t * 1000),
        )

    def get_all_prices(self) -> list[PriceQuote]:
        """Vectorized in spirit: single time.time() call shared across symbols."""
        t = time.time()
        t_ms = int(t * 1000)
        out: list[PriceQuote] = []
        with self._lock:
            symbols = list(self._watchlist.keys())
        for sym in symbols:
            base = self._base_for(sym)
            price, change_pct = _simulate_price(sym, base, t)
            seed = _symbol_seed(sym)
            volume = int(1_000_000 + (seed % 9_000_000) + abs(change_pct) * 250_000)
            out.append(PriceQuote(
                symbol=sym, price=price, change_pct=change_pct,
                volume=volume, timestamp_ms=t_ms,
            ))
        return out

    # -- portfolio --------------------------------------------------------

    def get_portfolio(self) -> PortfolioSummary:
        positions: list[Position] = []
        equity = 0.0
        total_cost = 0.0
        with self._lock:
            pos_items = list(self._positions.items())
            cash = self._cash
        for sym, p in pos_items:
            q = self.get_price(sym)
            value = q.price * p["qty"]
            cost = p["avg_price"] * p["qty"]
            pnl = value - cost
            pnl_pct = (pnl / cost * 100.0) if cost else 0.0
            positions.append(Position(
                id=make_id_pair("position", sym),
                symbol=sym, qty=p["qty"], avg_price=round(p["avg_price"], 4),
                current_price=q.price, pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 3),
            ))
            equity += value
            total_cost += cost
        total_pnl = equity - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100.0) if total_cost else 0.0
        return PortfolioSummary(
            cash=round(cash, 2),
            equity=round(equity, 2),
            total_value=round(cash + equity, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_pct=round(total_pnl_pct, 3),
            positions=positions,
        )

    # -- orders -----------------------------------------------------------

    def place_order(self, req: OrderCreate) -> Order:
        sym = req.symbol.upper()
        if req.qty <= 0:
            raise BotError(f"Order qty must be positive, got {req.qty}", status_code=400)
        if req.order_type == "limit" and req.limit_price is None:
            raise BotError("Limit order requires limit_price", status_code=400)

        now = _now_ms()
        oid = make_id(f"order:{sym}:{now}")
        quote = self.get_price(sym)

        # Market orders fill at the simulated price immediately.
        # Limit orders queue and are filled lazily by _scan_limit_orders.
        if req.order_type == "market":
            fill_price = quote.price
            self._apply_fill(sym, req.side, req.qty, fill_price)
            order = Order(
                id=oid, symbol=sym, side=req.side, qty=req.qty,
                filled_qty=req.qty, order_type="market", limit_price=None,
                status="filled", avg_fill_price=fill_price,
                created_at=now, filled_at=now,
            )
        else:
            order = Order(
                id=oid, symbol=sym, side=req.side, qty=req.qty,
                filled_qty=0.0, order_type="limit", limit_price=req.limit_price,
                status="pending", avg_fill_price=None,
                created_at=now, filled_at=None,
            )

        with self._lock:
            self._orders[oid] = order
        LOGGER.info("Placed order %r %s %s %s @ %s", oid, req.side, req.qty, sym, quote.price)
        return order

    def _apply_fill(self, symbol: str, side: str, qty: float, price: float) -> None:
        """Apply a fill to cash + positions + trade history. Holds the lock."""
        with self._lock:
            pos = self._positions.get(symbol)
            realized: float | None = None
            if side == "buy":
                cost = qty * price
                if cost > self._cash:
                    raise BotError(
                        f"Insufficient cash: need ${cost:.2f}, have ${self._cash:.2f}",
                        status_code=400,
                    )
                self._cash -= cost
                if pos is None:
                    self._positions[symbol] = {"qty": qty, "avg_price": price}
                else:
                    total_qty = pos["qty"] + qty
                    pos["avg_price"] = (pos["avg_price"] * pos["qty"] + price * qty) / total_qty
                    pos["qty"] = total_qty
            else:  # sell
                if pos is None or pos["qty"] < qty:
                    have = pos["qty"] if pos else 0
                    raise BotError(
                        f"Insufficient position: need {qty} {symbol}, have {have}",
                        status_code=400,
                    )
                proceeds = qty * price
                realized = (price - pos["avg_price"]) * qty
                self._cash += proceeds
                pos["qty"] -= qty
                if pos["qty"] <= 1e-9:
                    del self._positions[symbol]

            self._trades.append(TradeHistoryEntry(
                order_id=make_id(f"trade:{symbol}:{time.monotonic_ns()}"),
                symbol=symbol, side=side, qty=qty, price=round(price, 4),
                realized_pnl=round(realized, 2) if realized is not None else None,
                timestamp_ms=_now_ms(),
            ))

    def get_orders(self) -> list[Order]:
        self._scan_limit_orders()
        with self._lock:
            return sorted(self._orders.values(), key=lambda o: o.created_at, reverse=True)

    def cancel_order(self, order_id: int) -> Order:
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                raise NotFoundError(f"Order {order_id!r} not found")
            if order.status != "pending":
                raise BotError(
                    f"Cannot cancel order in status {order.status!r}",
                    status_code=409,
                )
            cancelled = order.model_copy(update={"status": "cancelled"})
            self._orders[order_id] = cancelled
        LOGGER.info("Cancelled order %r", order_id)
        return cancelled

    def _scan_limit_orders(self) -> None:
        """Fill any pending limit orders whose trigger has been met."""
        with self._lock:
            pending = [o for o in self._orders.values() if o.status == "pending"]
        for o in pending:
            if o.limit_price is None:
                continue
            quote = self.get_price(o.symbol)
            triggered = (
                (o.side == "buy" and quote.price <= o.limit_price)
                or (o.side == "sell" and quote.price >= o.limit_price)
            )
            if not triggered:
                continue
            try:
                self._apply_fill(o.symbol, o.side, o.qty, quote.price)
            except BotError:
                with self._lock:
                    self._orders[o.id] = o.model_copy(update={"status": "rejected"})
                continue
            with self._lock:
                self._orders[o.id] = o.model_copy(update={
                    "status": "filled",
                    "filled_qty": o.qty,
                    "avg_fill_price": quote.price,
                    "filled_at": _now_ms(),
                })

    # -- watchlist --------------------------------------------------------

    def get_watchlist(self) -> list[WatchlistEntry]:
        with self._lock:
            return list(self._watchlist.values())

    def add_watchlist(self, symbol: str) -> WatchlistEntry:
        sym = symbol.upper()
        with self._lock:
            entry = self._watchlist.get(sym)
            if entry is None:
                entry = WatchlistEntry(symbol=sym, added_at=_now_ms())
                self._watchlist[sym] = entry
        return entry

    def remove_watchlist(self, symbol: str) -> WatchlistEntry:
        sym = symbol.upper()
        with self._lock:
            entry = self._watchlist.pop(sym, None)
        if entry is None:
            raise NotFoundError(f"Symbol {sym!r} not in watchlist")
        return entry

    # -- signals ----------------------------------------------------------

    def get_signal(self, symbol: str) -> TradingSignal:
        sym = symbol.upper()
        base = self._base_for(sym)
        t = time.time()
        history = _simulate_history(sym, base, t)
        ma20 = _ma(history, 20)
        ma50 = _ma(history, 50)
        rsi = _rsi(history, 14)
        last = history[-1]

        indicators = {
            "price": round(last, 4),
            "ma20": round(ma20, 4) if ma20 is not None else 0.0,
            "ma50": round(ma50, 4) if ma50 is not None else 0.0,
            "rsi14": rsi if rsi is not None else 50.0,
        }
        signal, conf, reason = _classify(indicators)
        return TradingSignal(
            symbol=sym, signal=signal, confidence=conf, reason=reason,
            indicators=indicators, timestamp_ms=int(t * 1000),
        )

    def get_all_signals(self) -> list[TradingSignal]:
        with self._lock:
            symbols = list(self._watchlist.keys())
        return [self.get_signal(s) for s in symbols]

    # -- alerts -----------------------------------------------------------

    def create_alert(self, req: PriceAlertCreate) -> PriceAlert:
        sym = req.symbol.upper()
        aid = make_id(f"alert:{sym}:{req.condition}:{req.threshold}")
        alert = PriceAlert(
            id=aid, symbol=sym, condition=req.condition, threshold=req.threshold,
            triggered=False, created_at=_now_ms(), triggered_at=None,
        )
        with self._lock:
            self._alerts[aid] = alert
        return alert

    def list_alerts(self) -> list[PriceAlert]:
        self._scan_alerts()
        with self._lock:
            return list(self._alerts.values())

    def delete_alert(self, alert_id: int) -> PriceAlert:
        with self._lock:
            alert = self._alerts.pop(alert_id, None)
        if alert is None:
            raise NotFoundError(f"Alert {alert_id!r} not found")
        return alert

    def _scan_alerts(self) -> None:
        with self._lock:
            pending = [a for a in self._alerts.values() if not a.triggered]
        for a in pending:
            quote = self.get_price(a.symbol)
            hit = (
                (a.condition == "above" and quote.price >= a.threshold)
                or (a.condition == "below" and quote.price <= a.threshold)
            )
            if hit:
                with self._lock:
                    self._alerts[a.id] = a.model_copy(update={
                        "triggered": True, "triggered_at": _now_ms(),
                    })

    # -- history ----------------------------------------------------------

    def get_trade_history(self) -> list[TradeHistoryEntry]:
        with self._lock:
            return list(reversed(self._trades))


def _classify(ind: dict[str, float]) -> tuple[str, float, str]:
    """Heuristic classifier: MA crossover + RSI."""
    price = ind["price"]
    ma20 = ind["ma20"]
    ma50 = ind["ma50"]
    rsi = ind["rsi14"]

    above_short = ma20 and price > ma20
    above_long = ma50 and price > ma50
    bull_cross = ma20 and ma50 and ma20 > ma50

    if rsi >= 70:
        return "strong_sell", 0.85, f"RSI overbought at {rsi:.1f}"
    if rsi <= 30:
        return "strong_buy", 0.85, f"RSI oversold at {rsi:.1f}"
    if bull_cross and above_short and above_long and rsi < 65:
        return "buy", 0.7, "MA20 > MA50 and price above both, RSI healthy"
    if not bull_cross and not above_short and not above_long and rsi > 35:
        return "sell", 0.65, "MA20 < MA50 and price below both"
    return "hold", 0.55, f"Neutral: RSI {rsi:.0f}, price {'above' if above_short else 'below'} MA20"
