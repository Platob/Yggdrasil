"""Market data service — yfinance backend with TTL cache.

yfinance is optional (add to pyproject.toml `market` extra). If not installed
the service still loads but every call raises a BadRequestError with an install
hint. The cache uses a simple dict with expiry timestamps so there is no extra
dependency.
"""
from __future__ import annotations

import time
from functools import partial

from fastapi.concurrency import run_in_threadpool

from yggdrasil.exceptions.api import BadRequestError

from ...config import Settings
from ..schemas.market import MarketOHLCV, MarketQuote, MarketSearchResult, OHLCVBar

# Cache: symbol+key -> (expires_at, data)
_QUOTE_TTL = 30       # 30 s for real-time quotes
_OHLCV_TTL = 300      # 5 min for historical bars


class MarketService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[str, tuple[float, object]] = {}

    def _get(self, key: str):
        entry = self._cache.get(key)
        if entry and time.monotonic() < entry[0]:
            return entry[1]
        return None

    def _set(self, key: str, value, ttl: float) -> None:
        self._cache[key] = (time.monotonic() + ttl, value)
        # evict stale entries when cache grows large
        if len(self._cache) > 512:
            now = time.monotonic()
            self._cache = {k: v for k, v in self._cache.items() if v[0] > now}

    async def quote(self, symbol: str) -> MarketQuote:
        return await run_in_threadpool(partial(self._quote, symbol.upper()))

    def _quote(self, symbol: str) -> MarketQuote:
        cached = self._get(f"quote:{symbol}")
        if cached is not None:
            return cached  # type: ignore[return-value]
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError:
            raise BadRequestError("yfinance is not installed; run: pip install yfinance")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            prev = getattr(info, "previous_close", None) or getattr(info, "regularMarketPreviousClose", None)
            price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            change = (price - prev) if (price is not None and prev is not None) else None
            change_pct = (change / prev * 100) if (change is not None and prev) else None
            result = MarketQuote(
                symbol=symbol,
                price=price,
                prev_close=prev,
                change=change,
                change_pct=change_pct,
                open=getattr(info, "open", None),
                day_high=getattr(info, "day_high", None),
                day_low=getattr(info, "day_low", None),
                volume=getattr(info, "three_month_average_volume", None),
                market_cap=getattr(info, "market_cap", None),
                currency=getattr(info, "currency", None),
            )
        except Exception as exc:
            raise BadRequestError(f"Failed to fetch quote for {symbol}: {exc}")
        self._set(f"quote:{symbol}", result, _QUOTE_TTL)
        return result

    async def ohlcv(self, symbol: str, period: str = "1y", interval: str = "1d") -> MarketOHLCV:
        return await run_in_threadpool(partial(self._ohlcv, symbol.upper(), period, interval))

    def _ohlcv(self, symbol: str, period: str, interval: str) -> MarketOHLCV:
        key = f"ohlcv:{symbol}:{period}:{interval}"
        cached = self._get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError:
            raise BadRequestError("yfinance is not installed; run: pip install yfinance")
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
            if df is None or df.empty:
                raise BadRequestError(f"No data returned for {symbol}")
            bars: list[OHLCVBar] = []
            for ts, row in df.iterrows():
                bars.append(OHLCVBar(
                    t=str(ts.isoformat()),
                    o=float(row["Open"]) if "Open" in row and row["Open"] == row["Open"] else None,
                    h=float(row["High"]) if "High" in row and row["High"] == row["High"] else None,
                    l=float(row["Low"]) if "Low" in row and row["Low"] == row["Low"] else None,
                    c=float(row["Close"]) if "Close" in row and row["Close"] == row["Close"] else None,
                    v=float(row["Volume"]) if "Volume" in row and row["Volume"] == row["Volume"] else None,
                ))
        except BadRequestError:
            raise
        except Exception as exc:
            raise BadRequestError(f"Failed to fetch OHLCV for {symbol}: {exc}")
        result = MarketOHLCV(symbol=symbol, period=period, interval=interval, bars=bars)
        self._set(key, result, _OHLCV_TTL)
        return result

    async def search(self, q: str) -> list[MarketSearchResult]:
        return await run_in_threadpool(partial(self._search, q))

    def _search(self, q: str) -> list[MarketSearchResult]:
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError:
            raise BadRequestError("yfinance is not installed; run: pip install yfinance")
        try:
            results = yf.Search(q, max_results=10)
            out: list[MarketSearchResult] = []
            for r in getattr(results, "quotes", []) or []:
                sym = r.get("symbol") or r.get("Symbol")
                if not sym:
                    continue
                out.append(MarketSearchResult(
                    symbol=sym,
                    name=r.get("longname") or r.get("shortname") or sym,
                    exchange=r.get("exchange"),
                    type=r.get("quoteType"),
                ))
            return out
        except BadRequestError:
            raise
        except Exception as exc:
            raise BadRequestError(f"Search failed: {exc}")
