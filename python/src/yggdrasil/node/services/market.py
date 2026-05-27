from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from functools import partial
from threading import Lock
from typing import Any

from fastapi.concurrency import run_in_threadpool

from ..config import Settings
from ..schemas.market import (
    FxConvertResponse,
    FxHistoryPoint,
    FxHistoryResponse,
    FxLatestResponse,
    FxRateEntry,
    WatchlistEntry,
    WatchlistResponse,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_WATCHLIST = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD"]


def _parse_pair(pair: str) -> tuple[str, str]:
    pair = pair.upper().replace("-", "/")
    if "/" not in pair:
        raise ValueError(f"Invalid pair format {pair!r}. Use 'EUR/USD'.")
    src, tgt = pair.split("/", 1)
    return src.strip(), tgt.strip()


class MarketService:
    def __init__(self, settings: Settings, *, cache_ttl: float = 60.0) -> None:
        self.settings = settings
        self._cache_ttl = cache_ttl
        self._rate_cache: dict[str, tuple[float, Any]] = {}
        self._watchlist: list[str] = list(DEFAULT_WATCHLIST)
        self._lock = Lock()

    def _cache_get(self, key: str) -> Any | None:
        with self._lock:
            item = self._rate_cache.get(key)
        if item is None:
            return None
        ts, val = item
        if time.monotonic() - ts > self._cache_ttl:
            return None
        return val

    def _cache_set(self, key: str, val: Any) -> None:
        with self._lock:
            self._rate_cache[key] = (time.monotonic(), val)

    # -- FX ------------------------------------------------------------------

    def _fx_latest_sync(self, pairs: list[tuple[str, str]]) -> list[FxRateEntry]:
        try:
            from yggdrasil.fxrate.session import FxRate
        except ImportError:
            raise RuntimeError("FxRate not available: install yggdrasil[http]")

        fx = FxRate()
        df = fx.latest(pairs=[(s, t) for s, t in pairs])
        rows: list[FxRateEntry] = []
        for row in df.iter_rows(named=True):
            rows.append(FxRateEntry(
                source=row["source"],
                target=row["target"],
                pair=f"{row['source']}/{row['target']}",
                value=float(row["value"]),
                from_timestamp=str(row["from_timestamp"]),
                to_timestamp=str(row["to_timestamp"]),
                sampling=row["sampling"],
            ))
        return rows

    async def get_latest(self, pairs: list[str]) -> FxLatestResponse:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        coerced = [_parse_pair(p) for p in pairs]
        cache_key = "latest:" + ",".join(f"{s}/{t}" for s, t in sorted(coerced))

        cached_val = self._cache_get(cache_key)
        if cached_val is not None:
            return FxLatestResponse(rates=cached_val, cached=True, fetched_at=now)

        rates = await run_in_threadpool(partial(self._fx_latest_sync, coerced))
        self._cache_set(cache_key, rates)
        return FxLatestResponse(rates=rates, cached=False, fetched_at=now)

    def _fx_history_sync(self, source: str, target: str, days: int) -> list[FxHistoryPoint]:
        try:
            from yggdrasil.fxrate.session import FxRate
        except ImportError:
            raise RuntimeError("FxRate not available")

        fx = FxRate()
        end = dt.datetime.now(dt.timezone.utc)
        start = end - dt.timedelta(days=days)
        df = fx.fetch(pairs=[(source, target)], start=start, end=end, sampling="1d")
        points = []
        for row in df.iter_rows(named=True):
            points.append(FxHistoryPoint(
                from_timestamp=str(row["from_timestamp"]),
                to_timestamp=str(row["to_timestamp"]),
                value=float(row["value"]),
            ))
        return points

    async def get_history(self, pair: str, days: int = 30) -> FxHistoryResponse:
        source, target = _parse_pair(pair)
        cache_key = f"history:{source}/{target}:{days}"
        cached_val = self._cache_get(cache_key)
        if cached_val is not None:
            return FxHistoryResponse(
                source=source, target=target,
                pair=f"{source}/{target}", sampling="1d",
                points=cached_val,
            )

        points = await run_in_threadpool(partial(self._fx_history_sync, source, target, days))
        self._cache_set(cache_key, points)
        return FxHistoryResponse(
            source=source, target=target,
            pair=f"{source}/{target}", sampling="1d",
            points=points,
        )

    def _fx_convert_sync(self, amount: float, source: str, target: str) -> float:
        try:
            from yggdrasil.fxrate.session import FxRate
        except ImportError:
            raise RuntimeError("FxRate not available")
        fx = FxRate()
        return fx.convert(amount, source, target)

    async def convert(self, amount: float, source: str, target: str) -> FxConvertResponse:
        cache_key = f"rate:{source}/{target}"
        cached_rate = self._cache_get(cache_key)
        if cached_rate is not None:
            return FxConvertResponse(
                source=source, target=target,
                amount=amount, result=amount * cached_rate, rate=cached_rate,
            )

        result = await run_in_threadpool(partial(self._fx_convert_sync, 1.0, source, target))
        self._cache_set(cache_key, result)
        return FxConvertResponse(
            source=source, target=target,
            amount=amount, result=amount * result, rate=result,
        )

    # -- Watchlist -----------------------------------------------------------

    def get_watchlist(self) -> WatchlistResponse:
        with self._lock:
            pairs = list(self._watchlist)
        entries = []
        for p in pairs:
            try:
                src, tgt = _parse_pair(p)
                entries.append(WatchlistEntry(pair=p, source=src, target=tgt))
            except ValueError:
                continue
        return WatchlistResponse(pairs=entries)

    def add_to_watchlist(self, pair: str) -> WatchlistResponse:
        norm_src, norm_tgt = _parse_pair(pair)
        norm = f"{norm_src}/{norm_tgt}"
        with self._lock:
            if norm not in self._watchlist:
                self._watchlist.append(norm)
        return self.get_watchlist()

    def remove_from_watchlist(self, pair: str) -> WatchlistResponse:
        norm_src, norm_tgt = _parse_pair(pair)
        norm = f"{norm_src}/{norm_tgt}"
        with self._lock:
            self._watchlist = [p for p in self._watchlist if p != norm]
        return self.get_watchlist()
