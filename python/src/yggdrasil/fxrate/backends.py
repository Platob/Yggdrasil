"""FX rate backend implementations.

Data over code: each backend is a class with ``name``, ``base_url``,
``default_sampling`` attrs and two methods. Add a new upstream by adding
a class here — :class:`FxRate` picks it up via its ``backends`` arg.

Real backends ship with the module:

- :class:`FrankfurterBackend` — ``api.frankfurter.app`` (free, ECB data)
- :class:`FawazBackend` — jsDelivr-hosted Fawaz Ahmed exchange-rate CDN (free)

Both are stateless and safe to share across threads.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yggdrasil.http_ import HTTPSession
    from .session import FxQuote

__all__ = ["Backend", "BackendError", "FrankfurterBackend", "FawazBackend"]


class BackendError(Exception):
    """Raised when a backend fetch fails; :class:`FxRate` falls to the next one."""


class Backend(ABC):
    name: str
    base_url: str
    default_sampling: str = "1d"

    @abstractmethod
    def fetch_timeseries(
        self,
        session: HTTPSession,
        *,
        source: str,
        targets: list[str],
        start: dt.date,
        end: dt.date,
        sampling: str,
    ) -> list[FxQuote]: ...

    @abstractmethod
    def fetch_latest(
        self,
        session: HTTPSession,
        *,
        source: str,
        targets: list[str],
        at: dt.datetime | None,
    ) -> list[FxQuote]: ...


class FrankfurterBackend(Backend):
    """Frankfurter API — free, ECB-sourced daily FX rates.

    Endpoint: ``https://api.frankfurter.app/{start}..{end}?from=EUR&to=USD,GBP``
    Supports all ECB currency pairs; no API key required.
    """

    name = "frankfurter"
    base_url = "https://api.frankfurter.app"
    default_sampling = "1d"

    def fetch_timeseries(
        self,
        session: HTTPSession,
        *,
        source: str,
        targets: list[str],
        start: dt.date,
        end: dt.date,
        sampling: str,
    ) -> list[FxQuote]:
        from .session import FxQuote

        to_param = ",".join(targets)
        url = f"{self.base_url}/{start.isoformat()}..{end.isoformat()}?from={source}&to={to_param}"
        try:
            resp = session.get(url)
            data: dict[str, Any] = resp.json()
        except Exception as e:
            raise BackendError(f"frankfurter fetch failed: {e}") from e

        rates: dict[str, dict[str, float]] = data.get("rates", {})
        quotes: list[FxQuote] = []
        for date_str, day_rates in rates.items():
            day = dt.date.fromisoformat(date_str)
            from_ts = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
            to_ts = from_ts + dt.timedelta(days=1)
            for tgt, val in day_rates.items():
                quotes.append(FxQuote(
                    source=source,
                    target=tgt,
                    from_timestamp=from_ts,
                    to_timestamp=to_ts,
                    sampling="1d",
                    value=float(val),
                ))
        return quotes

    def fetch_latest(
        self,
        session: HTTPSession,
        *,
        source: str,
        targets: list[str],
        at: dt.datetime | None,
    ) -> list[FxQuote]:
        from .session import FxQuote

        to_param = ",".join(targets)
        if at is not None:
            date_str = at.date().isoformat()
            url = f"{self.base_url}/{date_str}?from={source}&to={to_param}"
        else:
            url = f"{self.base_url}/latest?from={source}&to={to_param}"
        try:
            resp = session.get(url)
            data: dict[str, Any] = resp.json()
        except Exception as e:
            raise BackendError(f"frankfurter latest failed: {e}") from e

        date_str = data.get("date", "")
        try:
            day = dt.date.fromisoformat(date_str)
        except ValueError:
            day = dt.date.today()
        from_ts = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
        to_ts = from_ts + dt.timedelta(days=1)
        quotes: list[FxQuote] = []
        for tgt, val in data.get("rates", {}).items():
            quotes.append(FxQuote(
                source=source,
                target=tgt,
                from_timestamp=from_ts,
                to_timestamp=to_ts,
                sampling="1d",
                value=float(val),
            ))
        return quotes


class FawazBackend(Backend):
    """Fawaz Ahmed currency-api CDN — free, community-maintained.

    Served via jsDelivr CDN from ``@fawazahmed0/currency-api``.
    Updated daily; supports 170+ currencies. No API key required.
    Endpoint: ``https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{date}/v1/currencies/{source}.json``
    """

    name = "fawaz"
    base_url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api"
    default_sampling = "1d"

    def _url(self, date: dt.date, source: str) -> str:
        return f"{self.base_url}@{date.isoformat()}/v1/currencies/{source.lower()}.json"

    def fetch_timeseries(
        self,
        session: HTTPSession,
        *,
        source: str,
        targets: list[str],
        start: dt.date,
        end: dt.date,
        sampling: str,
    ) -> list[FxQuote]:
        from .session import FxQuote

        quotes: list[FxQuote] = []
        current = start
        target_set = {t.lower() for t in targets}
        while current <= end:
            url = self._url(current, source)
            try:
                resp = session.get(url)
                data: dict[str, Any] = resp.json()
            except Exception as e:
                raise BackendError(f"fawaz fetch failed for {current}: {e}") from e

            rates: dict[str, float] = data.get(source.lower(), {})
            from_ts = dt.datetime(current.year, current.month, current.day, tzinfo=dt.timezone.utc)
            to_ts = from_ts + dt.timedelta(days=1)
            for tgt_lower, val in rates.items():
                if tgt_lower in target_set:
                    quotes.append(FxQuote(
                        source=source,
                        target=tgt_lower.upper(),
                        from_timestamp=from_ts,
                        to_timestamp=to_ts,
                        sampling="1d",
                        value=float(val),
                    ))
            current += dt.timedelta(days=1)
        return quotes

    def fetch_latest(
        self,
        session: HTTPSession,
        *,
        source: str,
        targets: list[str],
        at: dt.datetime | None,
    ) -> list[FxQuote]:
        from .session import FxQuote

        date = at.date() if at is not None else dt.date.today()
        url = self._url(date, source)
        try:
            resp = session.get(url)
            data: dict[str, Any] = resp.json()
        except Exception as e:
            raise BackendError(f"fawaz latest failed: {e}") from e

        rates: dict[str, float] = data.get(source.lower(), {})
        target_set = {t.lower() for t in targets}
        from_ts = dt.datetime(date.year, date.month, date.day, tzinfo=dt.timezone.utc)
        to_ts = from_ts + dt.timedelta(days=1)
        return [
            FxQuote(
                source=source,
                target=tgt.upper(),
                from_timestamp=from_ts,
                to_timestamp=to_ts,
                sampling="1d",
                value=float(val),
            )
            for tgt, val in rates.items()
            if tgt in target_set
        ]
