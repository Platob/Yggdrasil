"""Frankfurter backend — free, public, no API key.

https://frankfurter.dev  (ECB daily rates, ~170 currencies).

Each fetch is a blocking HTTP call so the backend is lightweight:
one requests.get per source currency per date range.  Results are
cached for 60 seconds in the module-level ``_CACHE`` so repeated
calls within the same process don't hammer the upstream.
"""
from __future__ import annotations

import datetime as dt
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import FxQuote, FxRate

from .backends import Backend, BackendError

__all__ = ["FrankfurterBackend"]

_CACHE: dict[tuple, tuple[float, object]] = {}  # key -> (ts, value)
_TTL = 60.0  # seconds


def _get(url: str, params: dict) -> dict:
    try:
        import urllib.request, urllib.parse, json
        query = urllib.parse.urlencode(params)
        full = f"{url}?{query}" if params else url
        with urllib.request.urlopen(full, timeout=8) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        raise BackendError(f"Frankfurter HTTP error: {exc}") from exc


class FrankfurterBackend(Backend):
    """ECB daily rates via frankfurter.dev."""

    name = "frankfurter"
    base_url = "https://api.frankfurter.dev"
    default_sampling = "1d"

    def fetch_timeseries(
        self,
        session: FxRate,
        *,
        source: str,
        targets: list[str],
        start: dt.datetime,
        end: dt.datetime,
        sampling: str,
    ) -> list[FxQuote]:
        from .session import FxQuote
        start_s = start.strftime("%Y-%m-%d")
        end_s = end.strftime("%Y-%m-%d")
        targets_s = ",".join(targets)
        key = ("ts", source, targets_s, start_s, end_s)
        now = time.monotonic()
        if key in _CACHE and now - _CACHE[key][0] < _TTL:
            return list(_CACHE[key][1])  # type: ignore[arg-type]

        data = _get(f"{self.base_url}/{start_s}..{end_s}", {"base": source, "symbols": targets_s})
        rates_by_date: dict = data.get("rates", {})
        quotes: list[FxQuote] = []
        for date_str, day_rates in rates_by_date.items():
            day = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            next_day = day + dt.timedelta(days=1)
            for target, value in day_rates.items():
                quotes.append(FxQuote(
                    source=source, target=target,
                    from_timestamp=day, to_timestamp=next_day,
                    sampling="1d", value=float(value),
                ))
        _CACHE[key] = (now, quotes)
        return quotes

    def fetch_latest(
        self,
        session: FxRate,
        *,
        source: str,
        targets: list[str],
        at: dt.datetime,
    ) -> list[FxQuote]:
        from .session import FxQuote
        targets_s = ",".join(targets)
        key = ("latest", source, targets_s)
        now = time.monotonic()
        if key in _CACHE and now - _CACHE[key][0] < _TTL:
            return list(_CACHE[key][1])  # type: ignore[arg-type]

        data = _get(f"{self.base_url}/latest", {"base": source, "symbols": targets_s})
        day_str: str = data.get("date", at.strftime("%Y-%m-%d"))
        day = dt.datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        next_day = day + dt.timedelta(days=1)
        quotes: list[FxQuote] = [
            FxQuote(
                source=source, target=target,
                from_timestamp=day, to_timestamp=next_day,
                sampling="1d", value=float(value),
            )
            for target, value in data.get("rates", {}).items()
        ]
        _CACHE[key] = (now, quotes)
        return quotes
