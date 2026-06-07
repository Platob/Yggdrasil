"""Market data service — deterministic synthetic OHLC / ticks / order books.

There is no live feed behind the node; this service fabricates realistic
price action with a geometric Brownian-motion random walk seeded off the
symbol so a given ``(symbol, interval)`` always renders the same series
(stable across restarts, cache-friendly, reproducible in tests).

Price generation is fully vectorized with NumPy + polars — no Python loop
walks the bars. Per-asset price bases and volatilities give crypto its big
swings and FX its tight ranges. Ticks carry a 1-second cache so a burst of
clients polling the same symbol shares one computation.

Candles cache: the JSON bytes and Arrow table for each (symbol, interval, limit)
are memoised with a 2-second TTL so repeated polls from the dashboard are
served from memory without re-running the random walk.
"""
from __future__ import annotations

import orjson
import numpy as np

from yggdrasil.lazy_imports import polars as pl

from ..schemas.base import now_ms
from ..schemas.market import AssetInfo, Candle, OrderBook, Tick

__all__ = ["MarketDataService", "ASSETS", "INTERVAL_MS"]


# (symbol, name, type, currency, exchange, price_base, annualized_vol)
_ASSET_SPECS: list[tuple[str, str, str, str, str | None, float, float]] = [
    ("BTC/USD", "Bitcoin", "crypto", "USD", "Coinbase", 64000.0, 0.65),
    ("ETH/USD", "Ethereum", "crypto", "USD", "Coinbase", 3400.0, 0.75),
    ("AAPL", "Apple Inc.", "stock", "USD", "NASDAQ", 195.0, 0.28),
    ("GOOGL", "Alphabet Inc.", "stock", "USD", "NASDAQ", 175.0, 0.30),
    ("EUR/USD", "Euro / US Dollar", "fx", "USD", None, 1.08, 0.07),
    ("GBP/USD", "Pound / US Dollar", "fx", "USD", None, 1.27, 0.08),
    ("XAU/USD", "Gold Spot", "commodity", "USD", None, 2350.0, 0.15),
    ("SPY", "SPDR S&P 500 ETF", "stock", "USD", "NYSE", 530.0, 0.18),
]

ASSETS: dict[str, AssetInfo] = {
    s[0]: AssetInfo(symbol=s[0], name=s[1], type=s[2], currency=s[3], exchange=s[4])
    for s in _ASSET_SPECS
}

_PRICE_SPECS: dict[str, tuple[float, float]] = {s[0]: (s[5], s[6]) for s in _ASSET_SPECS}

INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# Fraction of one trading year that a single bar of each interval spans.
# Used to scale annualized vol down to per-bar sigma. 365d * 24h crypto basis.
_YEAR_MS = 365.0 * 86_400_000.0


class MarketDataService:
    def __init__(self) -> None:
        self._tick_cache: dict[str, tuple[int, Tick]] = {}
        # (symbol, interval, limit) → (cached_at_ms, json_bytes, arrow_table)
        self._candles_cache: dict[tuple[str, str, int], tuple[int, bytes, object]] = {}

    def _walk(self, symbol: str, interval: str, limit: int) -> pl.DataFrame:
        """Build *limit* OHLC bars ending at the most recent closed bar.

        Geometric Brownian motion on the log of the price: closes are a
        cumulative product of per-bar lognormal returns; highs/lows are the
        close plus a vectorized intrabar wick. Deterministic per symbol via a
        seeded NumPy ``Generator`` — no global RNG state, thread-safe.
        """
        base, ann_vol = _PRICE_SPECS.get(symbol, (100.0, 0.30))
        step_ms = INTERVAL_MS.get(interval, INTERVAL_MS["1h"])
        sigma = ann_vol * np.sqrt(step_ms / _YEAR_MS)

        seed = (abs(hash((symbol, interval))) & 0xFFFFFFFF) ^ 0x9E3779B1
        rng = np.random.default_rng(seed)

        n = max(int(limit), 1)
        # Tiny upward drift so series don't trend to zero over long windows.
        drift = -0.5 * sigma * sigma
        log_returns = drift + sigma * rng.standard_normal(n)
        close = base * np.exp(np.cumsum(log_returns))

        prev_close = np.empty(n, dtype=np.float64)
        prev_close[0] = base
        prev_close[1:] = close[:-1]
        open_ = prev_close

        body_hi = np.maximum(open_, close)
        body_lo = np.minimum(open_, close)
        wick = np.abs(rng.standard_normal(n)) * sigma * close
        high = body_hi + wick * rng.uniform(0.0, 1.0, n)
        low = body_lo - wick * rng.uniform(0.0, 1.0, n)
        low = np.maximum(low, 1e-9)

        # Volume scales with bar range so volatile bars look heavier.
        rng_pct = (high - low) / np.maximum(close, 1e-9)
        volume = (1.0 + rng_pct * 50.0) * rng.uniform(500.0, 1500.0, n)

        now = now_ms()
        last_open = now - (now % step_ms)
        ts = last_open - step_ms * np.arange(n - 1, -1, -1, dtype=np.int64)

        return pl.DataFrame(
            {
                "ts": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def get_candles_cached(
        self, symbol: str, interval: str, limit: int
    ) -> tuple[bytes, object]:
        """Return ``(json_bytes, arrow_table)`` for ``/candles``, cached for 2 s.

        Bypasses per-row Pydantic construction — we control the data so
        validation is unnecessary. The Arrow table is kept alongside JSON so
        the route can serve ``Accept: arrow`` without a second walk.
        """
        if symbol not in ASSETS:
            raise KeyError(symbol)
        now = now_ms()
        key = (symbol, interval, limit)
        hit = self._candles_cache.get(key)
        if hit is not None and now - hit[0] < 2000:
            return hit[1], hit[2]

        df = self._walk(symbol, interval, limit)
        rows = df.to_dicts()
        candles = [
            {
                "ts": int(r["ts"]),
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r["volume"],
                "symbol": symbol,
                "interval": interval,
            }
            for r in rows
        ]
        payload = {"symbol": symbol, "candles": candles, "count": len(candles)}
        json_bytes = orjson.dumps(payload)

        import pyarrow as pa
        arrow_table = pa.table(
            {
                "ts": df["ts"].to_list(),
                "open": df["open"].to_list(),
                "high": df["high"].to_list(),
                "low": df["low"].to_list(),
                "close": df["close"].to_list(),
                "volume": df["volume"].to_list(),
            }
        )
        self._candles_cache[key] = (now, json_bytes, arrow_table)
        return json_bytes, arrow_table

    async def get_candles(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        if symbol not in ASSETS:
            raise KeyError(symbol)
        df = self._walk(symbol, interval, limit)
        rows = df.to_dicts()
        return [
            Candle.model_construct(
                ts=int(r["ts"]),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]),
                symbol=symbol,
                interval=interval,
            )
            for r in rows
        ]

    async def get_tick(self, symbol: str) -> Tick:
        if symbol not in ASSETS:
            raise KeyError(symbol)
        now = now_ms()
        cached = self._tick_cache.get(symbol)
        if cached is not None and now - cached[0] < 1000:
            return cached[1]

        df = self._walk(symbol, "1m", 2)
        last = df.row(-1, named=True)
        prev = df.row(-2, named=True)
        price = float(last["close"])
        side = "buy" if price >= float(prev["close"]) else "sell"
        tick = Tick(
            ts=now,
            symbol=symbol,
            price=price,
            volume=float(last["volume"]) / 60.0,
            side=side,
        )
        self._tick_cache[symbol] = (now, tick)
        return tick

    async def get_book(self, symbol: str, depth: int = 10) -> OrderBook:
        if symbol not in ASSETS:
            raise KeyError(symbol)
        tick = await self.get_tick(symbol)
        mid = tick.price
        base, ann_vol = _PRICE_SPECS[symbol]
        # Half-spread proportional to volatility, floored so FX still ticks.
        half = max(mid * ann_vol * 1e-4, mid * 1e-5)

        seed = (abs(hash((symbol, "book"))) & 0xFFFFFFFF) ^ 0x85EBCA77
        rng = np.random.default_rng(seed)
        levels = np.arange(1, depth + 1, dtype=np.float64)
        bid_px = mid - half * levels
        ask_px = mid + half * levels
        bid_qty = rng.uniform(0.5, 5.0, depth) * levels
        ask_qty = rng.uniform(0.5, 5.0, depth) * levels

        return OrderBook(
            ts=tick.ts,
            symbol=symbol,
            bids=[(float(bid_px[i]), float(bid_qty[i])) for i in range(depth)],
            asks=[(float(ask_px[i]), float(ask_qty[i])) for i in range(depth)],
        )

    async def get_assets(self) -> list[AssetInfo]:
        return list(ASSETS.values())

    def current_price(self, symbol: str) -> float:
        """Synchronous last price used by the portfolio service for mark-to-market."""
        df = self._walk(symbol, "1m", 1)
        return float(df.row(-1, named=True)["close"])
