from __future__ import annotations

import datetime as dt
import json
import logging
import time
import urllib.request
from collections import deque
from threading import Lock
from typing import AsyncIterator

from ..config import Settings
from ..exceptions import NotFoundError
from ..ids import make_id
from ..schemas.trading import (
    AlertCreate,
    AlertsResponse,
    PortfolioPositionEntry,
    PortfolioResponse,
    PriceAlert,
    PriceHistoryResponse,
    PriceQuote,
    PricesResponse,
    TechnicalIndicators,
)

LOGGER = logging.getLogger(__name__)

# Default symbols to track
DEFAULT_FX_SYMBOLS = ("EUR/USD", "GBP/USD", "JPY/USD", "CHF/USD", "AUD/USD", "CAD/USD")
DEFAULT_CRYPTO_SYMBOLS = ("BTC-USD", "ETH-USD", "SOL-USD")
DEFAULT_SYMBOLS = DEFAULT_FX_SYMBOLS + DEFAULT_CRYPTO_SYMBOLS

# API endpoints
_FRANKFURTER_URL = "https://api.frankfurter.app/latest"
_COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"

# Crypto ID mapping for CoinGecko
_CRYPTO_IDS = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
}

# FX base currency mapping (Frankfurter gives rates FROM a base)
_FX_PAIRS = {
    "EUR/USD": ("EUR", "USD"),
    "GBP/USD": ("GBP", "USD"),
    "JPY/USD": ("JPY", "USD"),
    "CHF/USD": ("CHF", "USD"),
    "AUD/USD": ("AUD", "USD"),
    "CAD/USD": ("CAD", "USD"),
}

_CACHE_TTL = 30.0  # seconds


class _CachedValue:
    __slots__ = ("value", "fetched_at")

    def __init__(self, value: object, fetched_at: float) -> None:
        self.value = value
        self.fetched_at = fetched_at

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.fetched_at) < _CACHE_TTL


class _PortfolioPosition:
    __slots__ = ("symbol", "quantity", "avg_cost", "currency")

    def __init__(self, symbol: str, quantity: float, avg_cost: float, currency: str) -> None:
        self.symbol = symbol
        self.quantity = quantity
        self.avg_cost = avg_cost
        self.currency = currency


class TradingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()

        # Price cache: symbol -> _CachedValue(float)
        self._price_cache: dict[str, _CachedValue] = {}

        # Portfolio: symbol -> _PortfolioPosition
        self._portfolio: dict[str, _PortfolioPosition] = {}

        # Price history: symbol -> deque of (timestamp_iso, price)
        self._price_history: dict[str, deque[tuple[str, float]]] = {}

        # Alerts: alert_id -> PriceAlert
        self._alerts: dict[int, PriceAlert] = {}

    # -- Market data fetching -------------------------------------------------

    def _fetch_fx_rates(self) -> dict[str, float]:
        """Fetch FX rates from Frankfurter API. Returns {symbol: price}."""
        results: dict[str, float] = {}
        try:
            # Get rates with USD as base -- gives us 1 USD = X foreign
            url = f"{_FRANKFURTER_URL}?from=USD"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            rates = data.get("rates", {})
            # Convert: EUR/USD means "how many USD per 1 EUR" = 1 / (USD->EUR rate)
            for symbol, (base, quote) in _FX_PAIRS.items():
                if base in rates and rates[base] != 0:
                    # rates[base] = how many BASE per 1 USD
                    # BASE/USD price = 1 / rates[base] (how many USD per 1 BASE)
                    results[symbol] = round(1.0 / rates[base], 6)
                elif base == "USD" and quote in rates:
                    results[symbol] = round(rates[quote], 6)
        except Exception as exc:
            LOGGER.warning("FX fetch failed: %s", exc)
        return results

    def _fetch_crypto_prices(self) -> dict[str, float]:
        """Fetch crypto prices from CoinGecko API. Returns {symbol: price}."""
        results: dict[str, float] = {}
        try:
            ids = ",".join(_CRYPTO_IDS.values())
            url = f"{_COINGECKO_URL}?ids={ids}&vs_currencies=usd"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            for symbol, gecko_id in _CRYPTO_IDS.items():
                if gecko_id in data and "usd" in data[gecko_id]:
                    results[symbol] = float(data[gecko_id]["usd"])
        except Exception as exc:
            LOGGER.warning("Crypto fetch failed: %s", exc)
        return results

    def _get_live_price(self, symbol: str) -> tuple[float | None, bool]:
        """Get price for symbol. Returns (price, stale)."""
        with self._lock:
            cached = self._price_cache.get(symbol)
            if cached is not None and cached.is_fresh:
                return cached.value, False

        # Need to fetch fresh data
        if symbol in _FX_PAIRS:
            prices = self._fetch_fx_rates()
        elif symbol in _CRYPTO_IDS:
            prices = self._fetch_crypto_prices()
        else:
            # Unknown symbol -- return stale if available
            with self._lock:
                if cached is not None:
                    return cached.value, True
            return None, False

        now = time.time()
        with self._lock:
            for sym, price in prices.items():
                self._price_cache[sym] = _CachedValue(price, now)
                # Record in history
                if sym not in self._price_history:
                    self._price_history[sym] = deque(maxlen=200)
                ts = dt.datetime.now(dt.timezone.utc).isoformat()
                self._price_history[sym].append((ts, price))

            cached = self._price_cache.get(symbol)
            if cached is not None:
                return cached.value, not cached.is_fresh
        return None, False

    def get_prices(self, symbols: tuple[str, ...] | None = None) -> PricesResponse:
        """Get current prices for given symbols."""
        symbols = symbols or DEFAULT_SYMBOLS
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        quotes: dict[str, PriceQuote] = {}

        for symbol in symbols:
            price, stale = self._get_live_price(symbol)
            if price is not None:
                source = "frankfurter" if symbol in _FX_PAIRS else "coingecko"
                quotes[symbol] = PriceQuote(
                    symbol=symbol,
                    price=price,
                    currency="USD",
                    source=source,
                    timestamp=now,
                    stale=stale,
                )

        return PricesResponse(prices=quotes, timestamp=now)

    def get_price_history(self, symbol: str) -> PriceHistoryResponse:
        """Get price history for a symbol (last 200 points)."""
        with self._lock:
            history = self._price_history.get(symbol)
            if history is None:
                return PriceHistoryResponse(symbol=symbol, prices=[], timestamps=[])
            timestamps = [ts for ts, _ in history]
            prices = [p for _, p in history]
        return PriceHistoryResponse(symbol=symbol, prices=prices, timestamps=timestamps)

    # -- Portfolio tracking ----------------------------------------------------

    def upsert_position(self, symbol: str, quantity: float, avg_cost: float, currency: str = "USD") -> None:
        """Add or update a portfolio position."""
        with self._lock:
            self._portfolio[symbol] = _PortfolioPosition(
                symbol=symbol,
                quantity=quantity,
                avg_cost=avg_cost,
                currency=currency,
            )
        LOGGER.info("Upserted position %r (qty=%s, cost=%s)", symbol, quantity, avg_cost)

    def remove_position(self, symbol: str) -> None:
        """Remove a portfolio position."""
        with self._lock:
            removed = self._portfolio.pop(symbol, None)
        if removed is None:
            raise NotFoundError(f"Position {symbol!r} not found in portfolio")
        LOGGER.info("Removed position %r", symbol)

    def get_portfolio(self) -> PortfolioResponse:
        """Get portfolio with live P&L computation."""
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock:
            positions = list(self._portfolio.values())

        entries: list[PortfolioPositionEntry] = []
        total_value = 0.0
        total_cost = 0.0

        for pos in positions:
            price, _ = self._get_live_price(pos.symbol)
            current_price = price
            pnl: float | None = None
            pnl_pct: float | None = None

            if current_price is not None:
                cost_basis = pos.quantity * pos.avg_cost
                market_value = pos.quantity * current_price
                pnl = round(market_value - cost_basis, 2)
                pnl_pct = round((pnl / cost_basis) * 100, 2) if cost_basis != 0 else 0.0
                total_value += market_value
                total_cost += cost_basis
            else:
                # Use cost basis as fallback
                total_value += pos.quantity * pos.avg_cost
                total_cost += pos.quantity * pos.avg_cost

            entries.append(PortfolioPositionEntry(
                symbol=pos.symbol,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                currency=pos.currency,
                current_price=current_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
            ))

        total_pnl = round(total_value - total_cost, 2)

        return PortfolioResponse(
            positions=entries,
            total_value=round(total_value, 2),
            total_pnl=total_pnl,
            currency="USD",
            timestamp=now,
        )

    # -- Technical indicators --------------------------------------------------

    def get_indicators(self, symbol: str) -> TechnicalIndicators:
        """Compute technical indicators for a symbol."""
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock:
            history = self._price_history.get(symbol)
            if history is None or len(history) == 0:
                # Try to fetch a price to seed history
                pass

        # Attempt to get current price (seeds history as side effect)
        price, _ = self._get_live_price(symbol)

        with self._lock:
            history = self._price_history.get(symbol)
            if history is None or len(history) == 0:
                return TechnicalIndicators(
                    symbol=symbol, price=price, timestamp=now,
                )
            prices = [p for _, p in history]

        sma_20 = self._sma(prices, 20)
        sma_50 = self._sma(prices, 50)
        ema_20 = self._ema(prices, 20)
        rsi_14 = self._rsi(prices, 14)

        return TechnicalIndicators(
            symbol=symbol,
            sma_20=round(sma_20, 6) if sma_20 is not None else None,
            sma_50=round(sma_50, 6) if sma_50 is not None else None,
            ema_20=round(ema_20, 6) if ema_20 is not None else None,
            rsi_14=round(rsi_14, 2) if rsi_14 is not None else None,
            price=price,
            timestamp=now,
        )

    @staticmethod
    def _sma(prices: list[float], n: int) -> float | None:
        """Simple moving average over last n prices."""
        if len(prices) < n:
            return None
        return sum(prices[-n:]) / n

    @staticmethod
    def _ema(prices: list[float], n: int) -> float | None:
        """Exponential moving average over last n prices."""
        if len(prices) < n:
            return None
        multiplier = 2.0 / (n + 1)
        # Seed with SMA of first n points
        ema = sum(prices[:n]) / n
        for price in prices[n:]:
            ema = (price - ema) * multiplier + ema
        return ema

    @staticmethod
    def _rsi(prices: list[float], n: int = 14) -> float | None:
        """Relative Strength Index. Needs at least n+1 data points."""
        if len(prices) < n + 1:
            return None
        # Compute gains and losses
        gains = 0.0
        losses = 0.0
        for i in range(1, n + 1):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains += change
            else:
                losses += abs(change)

        avg_gain = gains / n
        avg_loss = losses / n

        # Smooth with remaining prices
        for i in range(n + 1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                avg_gain = (avg_gain * (n - 1) + change) / n
                avg_loss = (avg_loss * (n - 1)) / n
            else:
                avg_gain = (avg_gain * (n - 1)) / n
                avg_loss = (avg_loss * (n - 1) + abs(change)) / n

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    # -- Price alerts ----------------------------------------------------------

    def set_alert(self, req: AlertCreate) -> PriceAlert:
        """Create a new price alert."""
        if req.condition not in ("above", "below"):
            raise ValueError(f"Condition must be 'above' or 'below', got {req.condition!r}")

        alert_id = make_id(f"{req.symbol}:{req.condition}:{req.price}")
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        alert = PriceAlert(
            id=alert_id,
            symbol=req.symbol,
            condition=req.condition,
            price=req.price,
            created_at=now,
            triggered_at=None,
        )
        with self._lock:
            self._alerts[alert_id] = alert
        LOGGER.info("Set alert %r: %s %s %s", alert_id, req.symbol, req.condition, req.price)
        return alert

    def check_alerts(self) -> list[PriceAlert]:
        """Check all alerts against current prices. Returns newly triggered alerts."""
        triggered: list[PriceAlert] = []
        now = dt.datetime.now(dt.timezone.utc).isoformat()

        with self._lock:
            pending = [a for a in self._alerts.values() if a.triggered_at is None]

        for alert in pending:
            price, _ = self._get_live_price(alert.symbol)
            if price is None:
                continue

            should_trigger = False
            if alert.condition == "above" and price >= alert.price:
                should_trigger = True
            elif alert.condition == "below" and price <= alert.price:
                should_trigger = True

            if should_trigger:
                # Create updated alert with triggered_at set
                updated = PriceAlert(
                    id=alert.id,
                    symbol=alert.symbol,
                    condition=alert.condition,
                    price=alert.price,
                    created_at=alert.created_at,
                    triggered_at=now,
                )
                with self._lock:
                    self._alerts[alert.id] = updated
                triggered.append(updated)
                LOGGER.info("Alert %r triggered: %s %s %s (current=%s)",
                            alert.id, alert.symbol, alert.condition, alert.price, price)

        return triggered

    def get_alerts(self) -> AlertsResponse:
        """Get all alerts."""
        with self._lock:
            alerts = list(self._alerts.values())
        return AlertsResponse(alerts=alerts)

    def remove_alert(self, alert_id: int) -> PriceAlert:
        """Remove an alert by ID."""
        with self._lock:
            alert = self._alerts.pop(alert_id, None)
        if alert is None:
            raise NotFoundError(f"Alert {alert_id!r} not found")
        LOGGER.info("Removed alert %r", alert_id)
        return alert

    # -- SSE streaming ---------------------------------------------------------

    async def stream_prices(self, symbols: tuple[str, ...] | None = None, interval_seconds: float = 5.0) -> AsyncIterator[str]:
        """Async generator yielding JSON price updates as SSE events."""
        import asyncio

        symbols = symbols or DEFAULT_SYMBOLS
        while True:
            response = self.get_prices(symbols)
            yield f"data: {response.model_dump_json()}\n\n"
            await asyncio.sleep(interval_seconds)
