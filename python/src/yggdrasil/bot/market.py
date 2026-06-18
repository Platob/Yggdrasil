"""Market data service — ENTSOE day-ahead prices + FX rates.

ENTSOE: requires a free API token from transparency.entsoe.eu (env: ENTSOE_API_TOKEN).
FX:     hits the public frankfurter.app API (no token required, rate-limited).

Both results are TTL-cached in-memory so repeated API calls are cheap.
The service is synchronous (ENTSOE uses HTTPSession under the hood); callers
in async context wrap with asyncio.to_thread().
"""
from __future__ import annotations

import datetime as dt
import json
import threading
import time
from typing import Optional
from urllib.request import urlopen


class _TTLCache:
    """Thread-safe TTL dict: key → (value, expire_ts).

    get() is deliberately lock-free for the fast path (dict lookup is GIL-safe
    in CPython) — the lock is only taken on write and on expiry eviction to
    avoid tearing.
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[object, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> object | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, exp = entry
        if time.monotonic() < exp:
            return value
        # Expired — evict under lock to avoid two threads both evicting
        with self._lock:
            self._data.pop(key, None)
        return None

    def set(self, key: str, value: object, ttl: float) -> None:
        with self._lock:
            self._data[key] = (value, time.monotonic() + ttl)

    def size(self) -> int:
        now = time.monotonic()
        return sum(1 for _, exp in self._data.values() if now < exp)


_cache = _TTLCache()
_hits = 0
_misses = 0


def cache_stats() -> tuple[int, int, int]:
    """Return (size, hits, misses)."""
    return _cache.size(), _hits, _misses


def peek_prices(
    zone: str,
    series: str,
    days: int,
) -> list[dict] | None:
    """Synchronous cache-only lookup (no network, no thread). None on miss."""
    return _cache.get(f"entsoe:{zone}:{series}:{days}")  # type: ignore[return-value]


def peek_fx(base: str, targets: list[str]) -> dict | None:
    """Synchronous cache-only FX lookup. None on miss."""
    return _cache.get(f"fx:{base}:{','.join(sorted(targets))}")  # type: ignore[return-value]


def fetch_prices(
    zone: str = "DE_LU",
    series: str = "day_ahead_prices",
    days: int = 7,
    *,
    security_token: Optional[str] = None,
    cache_ttl: int = 300,
) -> list[dict]:
    """Fetch day-ahead prices for *zone* from ENTSOE.

    Returns a list of row dicts: {timestamp, value, unit, currency}.
    Falls back to an empty list (with a warning) when no token is available.
    """
    global _hits, _misses

    key = f"entsoe:{zone}:{series}:{days}"
    cached = _cache.get(key)
    if cached is not None:
        _hits += 1
        return cached  # type: ignore[return-value]
    _misses += 1

    try:
        from yggdrasil.loki.entsoe import fetch_frame, token as entsoe_token

        tok = security_token or entsoe_token()
        if not tok:
            _cache.set(key, [], ttl=min(cache_ttl, 30))   # cache "no token" briefly
            return []

        end = dt.datetime.now(dt.timezone.utc)
        start = end - dt.timedelta(days=days)
        frame = fetch_frame(series, zone, start, end, security_token=tok)

        rows = [
            {
                "timestamp": row["timestamp"],
                "value": row["value"],
                "unit": row.get("unit", "MWh"),
                "currency": row.get("currency", "EUR"),
            }
            for row in frame.to_dicts()
        ]
        _cache.set(key, rows, ttl=cache_ttl)
        return rows
    except Exception:
        return []


def fetch_fx(
    base: str = "EUR",
    targets: list[str] | None = None,
    *,
    cache_ttl: int = 60,
) -> dict:
    """Fetch latest FX rates from frankfurter.app (free, no key).

    Returns {"base": str, "date": str, "rates": {currency: float}}.
    Falls back to empty rates dict on network failure.
    """
    global _hits, _misses

    targets = targets or ["USD", "GBP", "CHF", "JPY", "CAD"]
    key = f"fx:{base}:{','.join(sorted(targets))}"
    cached = _cache.get(key)
    if cached is not None:
        _hits += 1
        return cached  # type: ignore[return-value]
    _misses += 1

    try:
        symbols = ",".join(targets)
        url = f"https://api.frankfurter.app/latest?from={base}&to={symbols}"
        with urlopen(url, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read())
        _cache.set(key, data, ttl=cache_ttl)
        return data
    except Exception:
        fallback = {"base": base, "date": str(dt.date.today()), "rates": {}}
        # Cache the failure briefly — avoids hammering an unreachable host on every request.
        _cache.set(key, fallback, ttl=min(cache_ttl, 10))
        return fallback
