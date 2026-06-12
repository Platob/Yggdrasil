from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import pyarrow as pa
import polars as pl

from ..models.market import OHLCV, Quote, Ticker

# TTL cache: (symbol, period) → (data, fetched_at)
_ohlcv_cache: dict[tuple[str, str], tuple[list[OHLCV], float]] = {}
_quote_cache: dict[str, tuple[Quote, float]] = {}
_QUOTE_TTL = 15.0   # seconds
_OHLCV_TTL = 60.0

# well-known symbols for search
_KNOWN: list[Ticker] = [
    Ticker(symbol="AAPL",  name="Apple Inc.",          exchange="NASDAQ", asset_type="stock"),
    Ticker(symbol="MSFT",  name="Microsoft Corp.",      exchange="NASDAQ", asset_type="stock"),
    Ticker(symbol="GOOGL", name="Alphabet Inc.",        exchange="NASDAQ", asset_type="stock"),
    Ticker(symbol="AMZN",  name="Amazon.com Inc.",      exchange="NASDAQ", asset_type="stock"),
    Ticker(symbol="NVDA",  name="NVIDIA Corp.",         exchange="NASDAQ", asset_type="stock"),
    Ticker(symbol="META",  name="Meta Platforms Inc.",  exchange="NASDAQ", asset_type="stock"),
    Ticker(symbol="TSLA",  name="Tesla Inc.",           exchange="NASDAQ", asset_type="stock"),
    Ticker(symbol="BRK-B", name="Berkshire Hathaway",   exchange="NYSE",   asset_type="stock"),
    Ticker(symbol="JPM",   name="JPMorgan Chase",       exchange="NYSE",   asset_type="stock"),
    Ticker(symbol="V",     name="Visa Inc.",            exchange="NYSE",   asset_type="stock"),
    Ticker(symbol="SPY",   name="SPDR S&P 500 ETF",     exchange="NYSE",   asset_type="etf"),
    Ticker(symbol="QQQ",   name="Invesco QQQ Trust",    exchange="NASDAQ", asset_type="etf"),
    Ticker(symbol="BTC-USD", name="Bitcoin USD",        exchange="CCC",    asset_type="crypto"),
    Ticker(symbol="ETH-USD", name="Ethereum USD",       exchange="CCC",    asset_type="crypto"),
    Ticker(symbol="SOL-USD", name="Solana USD",         exchange="CCC",    asset_type="crypto"),
]


def search_tickers(query: str, limit: int = 10) -> list[Ticker]:
    q = query.upper()
    return [t for t in _KNOWN if q in t.symbol or q in t.name.upper()][:limit]


async def fetch_quote(symbol: str) -> Quote:
    """Fetch a live quote, with 15-second TTL cache."""
    now = time.monotonic()
    cached = _quote_cache.get(symbol)
    if cached and now - cached[1] < _QUOTE_TTL:
        return cached[0]

    try:
        import yfinance as yf
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        info = await asyncio.to_thread(lambda: ticker.info)
        fast = await asyncio.to_thread(lambda: ticker.fast_info)

        price = float(fast.get("last_price") or info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        prev = float(fast.get("previous_close") or info.get("previousClose") or price)
        change = price - prev
        change_pct = (change / prev * 100) if prev else 0.0

        q = Quote(
            symbol=symbol.upper(),
            price=price,
            change=change,
            change_pct=change_pct,
            volume=int(fast.get("three_month_average_volume") or info.get("volume") or 0),
            market_cap=info.get("marketCap"),
            bid=info.get("bid"),
            ask=info.get("ask"),
            high_52w=fast.get("year_high") or info.get("fiftyTwoWeekHigh"),
            low_52w=fast.get("year_low") or info.get("fiftyTwoWeekLow"),
        )
    except Exception:
        # Fallback: return zero quote so API stays healthy even without network
        q = Quote(symbol=symbol.upper(), price=0.0, change=0.0, change_pct=0.0, volume=0)

    _quote_cache[symbol] = (q, now)
    return q


async def fetch_ohlcv(symbol: str, period: str = "1mo", interval: str = "1d") -> list[OHLCV]:
    """Fetch OHLCV bars with 60-second TTL cache."""
    key = (symbol, period, interval)
    now = time.monotonic()
    cached = _ohlcv_cache.get(key)  # type: ignore[arg-type]
    if cached and now - cached[1] < _OHLCV_TTL:
        return cached[0]

    try:
        import yfinance as yf
        df = await asyncio.to_thread(yf.download, symbol, period=period, interval=interval,
                                     progress=False, auto_adjust=True)
        # Build OHLCV list efficiently via numpy arrays (no iterrows overhead)
        timestamps = df.index.to_pydatetime()
        opens  = df["Open"].to_numpy()
        highs  = df["High"].to_numpy()
        lows   = df["Low"].to_numpy()
        closes = df["Close"].to_numpy()
        vols   = df["Volume"].to_numpy()
        rows: list[OHLCV] = [
            OHLCV(symbol=symbol, timestamp=t,
                  open=float(o), high=float(h), low=float(l), close=float(c), volume=int(v))
            for t, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, vols)
        ]
    except Exception:
        rows = []

    _ohlcv_cache[key] = (rows, now)  # type: ignore[assignment]
    return rows


OHLCV_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("ms")),
    ("open",  pa.float64()),
    ("high",  pa.float64()),
    ("low",   pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
])

# Arrow table cache: same key as _ohlcv_cache — avoids re-building on repeated Arrow requests
_arrow_cache: dict[tuple[str, str, str], tuple[pa.Table, float]] = {}


def ohlcv_to_arrow(bars: list[OHLCV]) -> pa.Table:
    """Convert OHLCV bars to Arrow table via pyarrow-native list comprehensions."""
    if not bars:
        return pa.table(
            {c.name: pa.array([], type=c.type) for c in OHLCV_SCHEMA}, schema=OHLCV_SCHEMA
        )
    return pa.table({
        "timestamp": pa.array([b.timestamp for b in bars], type=pa.timestamp("ms")),
        "open":   pa.array([b.open   for b in bars], type=pa.float64()),
        "high":   pa.array([b.high   for b in bars], type=pa.float64()),
        "low":    pa.array([b.low    for b in bars], type=pa.float64()),
        "close":  pa.array([b.close  for b in bars], type=pa.float64()),
        "volume": pa.array([b.volume for b in bars], type=pa.int64()),
    }, schema=OHLCV_SCHEMA)


async def fetch_ohlcv_arrow(symbol: str, period: str = "1mo", interval: str = "1d") -> pa.Table:
    """Fetch OHLCV as Arrow table with a separate cache layer (avoids repeated list → Arrow conversion)."""
    key = (symbol, period, interval)
    now = time.monotonic()
    cached = _arrow_cache.get(key)
    if cached and now - cached[1] < _OHLCV_TTL:
        return cached[0]
    bars = await fetch_ohlcv(symbol, period, interval)
    table = ohlcv_to_arrow(bars)
    _arrow_cache[key] = (table, now)
    return table


def ohlcv_to_polars(bars: list[OHLCV]) -> pl.DataFrame:
    """Convert OHLCV bars to Polars DataFrame."""
    return pl.from_arrow(ohlcv_to_arrow(bars))
