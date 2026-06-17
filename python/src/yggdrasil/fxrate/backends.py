"""FX rate backends — one class per upstream rate source.

A backend knows how to talk to one public exchange-rate API and how to turn
its payload into a flat ``list[FxQuote]``. The orchestration (coercion, pair
grouping, fallback walk, frame assembly) lives in :mod:`yggdrasil.fxrate.session`;
backends are intentionally thin and stateless so :class:`FxRate` can fan a
single request out across several of them and roll over on failure.

Failure contract: any network error, non-2xx status, or payload the backend
can't parse becomes a :class:`BackendError` so :class:`FxRate` moves to the
next backend in the chain instead of crashing the whole fetch.

Sources:

* :class:`FrankfurterBackend` — https://frankfurter.app (ECB reference rates,
  full timeseries support, the default primary).
* :class:`FawazBackend` — https://github.com/fawazahmed0/exchange-api (a free
  jsDelivr-hosted daily snapshot CDN; latest-only in practice — its dated
  endpoints carry a single day, so a timeseries is a per-day fan-out).
* :class:`ECBBackend` — the European Central Bank's SDMX feed (EUR-based).
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from yggdrasil.enums.currency import Currency

if TYPE_CHECKING:
    from yggdrasil.http_.session import HTTPSession

    from .session import FxQuote

__all__ = ["Backend", "FrankfurterBackend", "FawazBackend", "ECBBackend"]


class Backend(ABC):
    """Abstract FX rate source.

    Subclasses set the three class attributes (``name`` / ``base_url`` /
    ``default_sampling``) and implement the two fetch methods. There is no
    required ``__init__`` — a subclass may add its own state without calling
    ``super().__init__()``.
    """

    #: Short identifier, e.g. ``"frankfurter"`` — surfaces in logs and errors.
    name: str = "backend"
    #: Root URL the backend hits. Drives the shared HTTPSession singleton key.
    base_url: str = ""
    #: Sampling label used when the caller doesn't ask for one, e.g. ``"1d"``.
    default_sampling: str = "1d"

    @abstractmethod
    def fetch_timeseries(
        self,
        session: HTTPSession,
        *,
        source: Currency,
        targets: list[Currency],
        start: dt.datetime,
        end: dt.datetime,
        sampling: str,
    ) -> list[FxQuote]:
        """Return quotes for *source*→each-of-*targets* over ``[start, end]``."""
        raise NotImplementedError

    @abstractmethod
    def fetch_latest(
        self,
        session: HTTPSession,
        *,
        source: Currency,
        targets: list[Currency],
        at: dt.datetime,
    ) -> list[FxQuote]:
        """Return the most recent quote for *source*→each-of-*targets*."""
        raise NotImplementedError


class FrankfurterBackend(Backend):
    """Frankfurter.app — ECB reference rates with native timeseries support.

    Timeseries: ``GET /{start}..{end}?from=EUR&to=USD,GBP`` returns
    ``{"rates": {"2024-01-01": {"USD": 1.1, "GBP": 0.86}, ...}}``. Latest:
    ``GET /latest?from=EUR&to=USD,GBP``. Rates are published per business day,
    so the per-day window is the natural sampling unit (``1d``).
    """

    name: str = "frankfurter"
    base_url: str = "https://api.frankfurter.app"
    default_sampling: str = "1d"

    def fetch_timeseries(self, session, *, source, targets, start, end, sampling):
        from .session import BackendError, FxQuote

        path = f"/{start.date().isoformat()}..{end.date().isoformat()}"
        params = {"from": source.code, "to": ",".join(t.code for t in targets)}
        try:
            resp = session.get(path, params=params, raise_error=False)
            if not resp.ok:
                raise BackendError(
                    f"{self.name} returned HTTP {resp.status_code} for {source.code}->"
                    f"{params['to']} ({start.date()}..{end.date()})"
                )
            payload = resp.json()
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"{self.name} request failed: {exc}") from exc

        rates = payload.get("rates") or {}
        out: list[FxQuote] = []
        for day_str, by_target in rates.items():
            from_ts = dt.datetime.combine(
                dt.date.fromisoformat(day_str), dt.time.min, tzinfo=dt.timezone.utc,
            )
            to_ts = from_ts + dt.timedelta(days=1)
            for tgt in targets:
                value = by_target.get(tgt.code)
                if value is None:
                    continue
                out.append(FxQuote(
                    source=source.code, target=tgt.code,
                    from_timestamp=from_ts, to_timestamp=to_ts,
                    sampling=sampling, value=float(value),
                ))
        return out

    def fetch_latest(self, session, *, source, targets, at):
        from .session import BackendError, FxQuote

        params = {"from": source.code, "to": ",".join(t.code for t in targets)}
        try:
            resp = session.get("/latest", params=params, raise_error=False)
            if not resp.ok:
                raise BackendError(
                    f"{self.name} returned HTTP {resp.status_code} for latest "
                    f"{source.code}->{params['to']}"
                )
            payload = resp.json()
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"{self.name} request failed: {exc}") from exc

        day_str = payload.get("date")
        from_ts = (
            dt.datetime.combine(dt.date.fromisoformat(day_str), dt.time.min, tzinfo=dt.timezone.utc)
            if day_str else at
        )
        to_ts = from_ts + dt.timedelta(days=1)
        by_target = payload.get("rates") or {}
        out: list[FxQuote] = []
        for tgt in targets:
            value = by_target.get(tgt.code)
            if value is None:
                continue
            out.append(FxQuote(
                source=source.code, target=tgt.code,
                from_timestamp=from_ts, to_timestamp=to_ts,
                sampling=self.default_sampling, value=float(value),
            ))
        return out


class FawazBackend(Backend):
    """Fawaz Ahmed's currency-api — a free jsDelivr-hosted daily snapshot CDN.

    Each dated endpoint carries one day's rates for one base currency:
    ``/{date}/v1/currencies/{source}.json`` →
    ``{"date": "...", "eur": {"usd": 1.1, "gbp": 0.86, ...}}`` (keys are
    lowercased ISO codes). There is no range endpoint, so a timeseries is a
    per-day fan-out; for tidy behavior we fetch latest and treat it as the
    single most-recent sample, matching the source's real granularity.
    """

    name: str = "fawaz"
    base_url: str = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api"
    default_sampling: str = "1d"

    def fetch_timeseries(self, session, *, source, targets, start, end, sampling):
        # The CDN exposes per-day snapshots, not ranges. Return the latest day
        # in the window as a single sample — callers wanting a dense series
        # should use a range-native backend (Frankfurter) as primary.
        return self._fetch_day(session, source=source, targets=targets, at=end, sampling=sampling)

    def fetch_latest(self, session, *, source, targets, at):
        return self._fetch_day(
            session, source=source, targets=targets, at=at, sampling=self.default_sampling,
        )

    def _fetch_day(self, session, *, source, targets, at, sampling):
        from .session import BackendError, FxQuote

        date_seg = at.date().isoformat()
        path = f"@{date_seg}/v1/currencies/{source.code.lower()}.json"
        try:
            resp = session.get(path, raise_error=False)
            if not resp.ok:
                # Fall back to the rolling "latest" alias if the dated snapshot
                # isn't published yet (weekends/holidays, future dates).
                resp = session.get(f"@latest/v1/currencies/{source.code.lower()}.json", raise_error=False)
            if not resp.ok:
                raise BackendError(
                    f"{self.name} returned HTTP {resp.status_code} for "
                    f"{source.code} on {date_seg}"
                )
            payload = resp.json()
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"{self.name} request failed: {exc}") from exc

        day_str = payload.get("date")
        from_ts = (
            dt.datetime.combine(dt.date.fromisoformat(day_str), dt.time.min, tzinfo=dt.timezone.utc)
            if day_str else at
        )
        to_ts = from_ts + dt.timedelta(days=1)
        by_target = payload.get(source.code.lower()) or {}
        out: list[FxQuote] = []
        for tgt in targets:
            value = by_target.get(tgt.code.lower())
            if value is None:
                continue
            out.append(FxQuote(
                source=source.code, target=tgt.code,
                from_timestamp=from_ts, to_timestamp=to_ts,
                sampling=sampling, value=float(value),
            ))
        return out


class ECBBackend(Backend):
    """European Central Bank SDMX feed — EUR-based daily reference rates.

    ECB only publishes EUR crosses, so non-EUR sources raise BackendError to
    roll over to a backend that can serve them. Range endpoint:
    ``/service/data/EXR/D.{TARGET}.EUR.SP00.A?startPeriod=...&endPeriod=...``
    with ``Accept: application/json`` (SDMX-JSON).
    """

    name: str = "ecb"
    base_url: str = "https://data-api.ecb.europa.eu"
    default_sampling: str = "1d"

    def fetch_timeseries(self, session, *, source, targets, start, end, sampling):
        from .session import BackendError

        if source != Currency.EUR:
            raise BackendError(f"{self.name} only serves EUR-based crosses; got {source.code}")
        # SDMX-JSON parsing is heavyweight and the ECB feed mirrors Frankfurter
        # (both ECB reference rates); kept as a declared fallback that defers to
        # the richer backend rather than duplicating its parser.
        raise BackendError(f"{self.name} timeseries not implemented; use FrankfurterBackend")

    def fetch_latest(self, session, *, source, targets, at):
        from .session import BackendError

        if source != Currency.EUR:
            raise BackendError(f"{self.name} only serves EUR-based crosses; got {source.code}")
        raise BackendError(f"{self.name} latest not implemented; use FrankfurterBackend")
