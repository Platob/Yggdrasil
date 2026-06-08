"""``/api/v2/market`` — live market data from Yahoo Finance.

Quotes and OHLCV come from Yahoo's public ``chart`` endpoint (no auth). Each
symbol's last response is cached in an :class:`ExpiringDict` keyed by
``(symbol, interval, range)`` so a burst of dashboard polls collapses to one
upstream fetch per TTL window.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from fastapi import APIRouter, Query

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.exceptions.api import NotFoundError, TimeoutError as APITimeoutError
from yggdrasil.node.config import Settings

router = APIRouter(prefix="/api/v2/market")

_YF_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Cache the parsed chart payload, not the quote/ohlcv view, so a quote poll and
# an ohlcv poll for the same window share one upstream fetch.
_CHART_CACHE: ExpiringDict[str, dict] = ExpiringDict(Settings.from_env().market_cache_ttl)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}

# Process-wide async client so a batch/scan reuses pooled keep-alive
# connections to Yahoo instead of opening a fresh TLS handshake per symbol.
# Created lazily on first fetch; closed by the app lifespan on shutdown.
_CLIENT: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None or _CLIENT.is_closed:
        _CLIENT = httpx.AsyncClient(timeout=15.0, headers=_BROWSER_HEADERS)
    return _CLIENT


async def fetch_chart(symbol: str, interval: str, range_: str) -> dict:
    """Fetch + parse a Yahoo Finance chart for *symbol* (cached per window).

    Returns the parsed ``result[0]`` dict: ``meta`` + ``timestamp`` +
    ``indicators.quote[0]``. Raises :class:`NotFoundError` for an unknown
    symbol and :class:`APITimeoutError` when Yahoo is unreachable.
    """
    key = f"{symbol.upper()}|{interval}|{range_}"
    cached = _CHART_CACHE.get(key)
    if cached is not None:
        return cached

    params = {"interval": interval, "range": range_}
    try:
        resp = await _client().get(_YF_CHART.format(symbol=symbol), params=params)
    except httpx.HTTPError as exc:
        raise APITimeoutError(
            f"Could not reach Yahoo Finance for {symbol!r}: {exc}. "
            f"Check connectivity and try again."
        ) from exc

    if resp.status_code == 404:
        raise NotFoundError(
            f"No market data for symbol {symbol!r}. "
            f"Use a valid ticker, e.g. 'AAPL', 'MSFT', 'BRK-B'."
        )

    body = resp.json()
    chart = (body or {}).get("chart") or {}
    err = chart.get("error")
    results = chart.get("result")
    if err or not results:
        desc = (err or {}).get("description") if isinstance(err, dict) else None
        raise NotFoundError(
            f"No market data for symbol {symbol!r}"
            + (f": {desc}" if desc else ".")
            + " Use a valid ticker, e.g. 'AAPL', 'MSFT', 'BRK-B'."
        )

    result = results[0]
    _CHART_CACHE.set(key, result)
    return result


def _quote_from_chart(symbol: str, result: dict) -> dict:
    """Project a chart payload down to the compact quote shape."""
    meta = result.get("meta") or {}
    price = meta.get("regularMarketPrice")
    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
    change = None
    change_pct = None
    if price is not None and prev:
        change = price - prev
        change_pct = (change / prev) * 100 if prev else None
    return {
        "symbol": (meta.get("symbol") or symbol).upper(),
        "price": price,
        "prev_close": prev,
        "change": round(change, 4) if change is not None else None,
        "change_pct": round(change_pct, 4) if change_pct is not None else None,
        "currency": meta.get("currency"),
        "exchange": meta.get("exchangeName"),
        "volume": meta.get("regularMarketVolume"),
        "ts": int((meta.get("regularMarketTime") or time.time())) * 1000,
    }


def _ohlcv_from_chart(symbol: str, interval: str, result: dict) -> dict:
    """Project a chart payload into the OHLCV series shape."""
    meta = result.get("meta") or {}
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    data: list[dict[str, Any]] = []
    for i, ts in enumerate(timestamps):
        c = closes[i] if i < len(closes) else None
        if c is None:  # Yahoo emits null rows for halted/missing bars.
            continue
        data.append(
            {
                "ts": int(ts) * 1000,
                "open": opens[i] if i < len(opens) else None,
                "high": highs[i] if i < len(highs) else None,
                "low": lows[i] if i < len(lows) else None,
                "close": c,
                "volume": volumes[i] if i < len(volumes) else None,
            }
        )
    return {
        "symbol": (meta.get("symbol") or symbol).upper(),
        "interval": interval,
        "currency": meta.get("currency"),
        "data": data,
    }


@router.get("/quote/{symbol}")
async def get_quote(symbol: str) -> dict:
    result = await fetch_chart(symbol, "1d", "1d")
    return _quote_from_chart(symbol, result)


@router.get("/ohlcv/{symbol}")
async def get_ohlcv(
    symbol: str,
    interval: str = "1d",
    range_: str = Query("1mo", alias="range"),
) -> dict:
    result = await fetch_chart(symbol, interval, range_)
    return _ohlcv_from_chart(symbol, interval, result)


@router.get("/batch")
async def get_batch_quotes(symbols: str) -> dict:
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not tickers:
        return {"quotes": {}, "errors": {}}

    async def _one(sym: str) -> tuple[str, dict | None, str | None]:
        try:
            result = await fetch_chart(sym, "1d", "1d")
            return sym, _quote_from_chart(sym, result), None
        except Exception as exc:  # one bad ticker shouldn't sink the batch
            return sym, None, str(exc)

    results = await asyncio.gather(*(_one(s) for s in tickers))
    quotes = {sym: q for sym, q, _ in results if q is not None}
    errors = {sym: e for sym, _, e in results if e is not None}
    return {"quotes": quotes, "errors": errors}
