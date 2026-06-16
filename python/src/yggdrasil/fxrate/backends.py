"""FX rate backends — the pluggable upstream adapters behind :class:`FxRate`.

A :class:`Backend` is a thin adapter over one free FX provider. It owns
the provider's base URL, default sampling, and the two fetch verbs
(``fetch_timeseries`` / ``fetch_latest``); the orchestrator
(:class:`yggdrasil.fxrate.session.FxRate`) owns grouping, the HTTP
session, fallback, and frame assembly. Each verb takes an injected
session (an :class:`yggdrasil.http_.HTTPSession`) so the orchestrator
can share keep-alive sockets and so tests/benchmarks can stub the wire.

Concrete adapters:

* :class:`Frankfurter` — ``api.frankfurter.app`` (ECB reference rates).
* :class:`Fawaz`       — ``api.fxratesapi.com`` style daily/latest feed.
* :class:`ErApi`       — ``v6.exchangerate-api.com`` latest-only feed.

Backends translate any upstream/transport error into
:class:`BackendError` at the boundary so :class:`FxRate` can run its
fallback walk on a single exception type.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from yggdrasil.enums.currency import Currency
from yggdrasil.exceptions.base import YGGException

if TYPE_CHECKING:
    from yggdrasil.http_ import HTTPSession

__all__ = [
    "BackendError",
    "FxQuote",
    "Backend",
    "Frankfurter",
    "Fawaz",
    "ErApi",
]


class BackendError(YGGException):
    """A backend failed to serve a request.

    Raised by concrete backends when an upstream call errors (transport,
    bad status, unparseable payload). :class:`FxRate` catches this to
    drive its fallback walk to the next backend in the chain.
    """


@dataclass(slots=True, frozen=True)
class FxQuote:
    """A single FX observation: ``source/target`` over a sampling window.

    ``value`` is the price of one unit of ``source`` expressed in
    ``target`` (so ``EUR/USD = 1.05`` means 1 EUR buys 1.05 USD), valid
    over ``[from_timestamp, to_timestamp)`` at the given ``sampling``
    granularity (``"1d"``, ``"1h"``, …).
    """

    source: str
    target: str
    from_timestamp: dt.datetime
    to_timestamp: dt.datetime
    sampling: str
    value: float


class Backend(ABC):
    """Adapter contract for one FX provider.

    Subclasses set ``name`` / ``base_url`` / ``default_sampling`` as
    class attributes and implement the two fetch verbs. Both verbs
    receive the orchestrator's shared session and must return a flat
    ``list[FxQuote]`` (one per target per sampling step), translating any
    upstream failure into :class:`BackendError`.
    """

    name: str
    base_url: str
    default_sampling: str

    @abstractmethod
    def fetch_timeseries(
        self,
        session: "HTTPSession",
        *,
        source: Currency,
        targets: Sequence[Currency],
        start: dt.datetime,
        end: dt.datetime,
        sampling: str,
    ) -> list[FxQuote]:
        """Return every quote for ``targets`` against ``source`` in ``[start, end]``."""
        raise NotImplementedError

    @abstractmethod
    def fetch_latest(
        self,
        session: "HTTPSession",
        *,
        source: Currency,
        targets: Sequence[Currency],
        at: dt.datetime | None,
    ) -> list[FxQuote]:
        """Return the latest spot quote for each of ``targets`` against ``source``."""
        raise NotImplementedError


def _to_day(value: dt.datetime) -> str:
    """Render a datetime as the ``YYYY-MM-DD`` day these APIs expect."""
    return value.date().isoformat()


def _quotes_from_rates(
    rates: dict[str, float],
    *,
    source: Currency,
    from_ts: dt.datetime,
    to_ts: dt.datetime,
    sampling: str,
) -> list[FxQuote]:
    """Fan a ``{target_code: rate}`` map out into one :class:`FxQuote` each."""
    return [
        FxQuote(
            source=source.code,
            target=str(target),
            from_timestamp=from_ts,
            to_timestamp=to_ts,
            sampling=sampling,
            value=float(rate),
        )
        for target, rate in rates.items()
    ]


class Frankfurter(Backend):
    """ECB reference rates via ``api.frankfurter.app`` (free, no key).

    Frankfurter exposes both a date-range timeseries endpoint
    (``/{start}..{end}``) and a latest endpoint (``/latest``), both
    taking ``base`` + ``symbols`` query params and returning a
    ``{"rates": {date: {ccy: rate}}}`` (range) or ``{"rates": {ccy: rate}}``
    (latest) payload.
    """

    name = "frankfurter"
    base_url = "https://api.frankfurter.app"
    default_sampling = "1d"

    def fetch_timeseries(self, session, *, source, targets, start, end, sampling):
        symbols = ",".join(str(t) for t in targets)
        path = f"/{_to_day(start)}..{_to_day(end)}"
        try:
            response = session.get(
                path, params={"base": source.code, "symbols": symbols},
            )
            payload = response.json()
        except Exception as exc:  # transport / status / decode → fallback
            raise BackendError(f"{self.name} timeseries request failed: {exc}") from exc

        out: list[FxQuote] = []
        for day, rates in payload.get("rates", {}).items():
            from_ts = dt.datetime.combine(
                dt.date.fromisoformat(day), dt.time.min, tzinfo=dt.timezone.utc,
            )
            to_ts = from_ts + dt.timedelta(days=1)
            out.extend(
                _quotes_from_rates(
                    rates, source=source, from_ts=from_ts, to_ts=to_ts, sampling=sampling,
                )
            )
        return out

    def fetch_latest(self, session, *, source, targets, at):
        symbols = ",".join(str(t) for t in targets)
        path = f"/{_to_day(at)}" if at is not None else "/latest"
        try:
            response = session.get(
                path, params={"base": source.code, "symbols": symbols},
            )
            payload = response.json()
        except Exception as exc:
            raise BackendError(f"{self.name} latest request failed: {exc}") from exc

        day = payload.get("date")
        from_ts = (
            dt.datetime.combine(dt.date.fromisoformat(day), dt.time.min, tzinfo=dt.timezone.utc)
            if day
            else (at or dt.datetime.now(dt.timezone.utc))
        )
        to_ts = from_ts + dt.timedelta(days=1)
        return _quotes_from_rates(
            payload.get("rates", {}),
            source=source,
            from_ts=from_ts,
            to_ts=to_ts,
            sampling=self.default_sampling,
        )


class Fawaz(Backend):
    """Daily/latest feed via ``api.fxratesapi.com`` (free tier).

    Latest rates come from ``/latest`` and a single historical day from
    ``/historical?date=YYYY-MM-DD``; both return
    ``{"rates": {ccy: rate}}``. There is no native multi-day range, so the
    timeseries verb walks the window day by day.
    """

    name = "fawaz"
    base_url = "https://api.fxratesapi.com"
    default_sampling = "1d"

    def fetch_timeseries(self, session, *, source, targets, start, end, sampling):
        symbols = ",".join(str(t) for t in targets)
        out: list[FxQuote] = []
        day = start.date()
        last = end.date()
        while day <= last:
            from_ts = dt.datetime.combine(day, dt.time.min, tzinfo=dt.timezone.utc)
            to_ts = from_ts + dt.timedelta(days=1)
            try:
                response = session.get(
                    "/historical",
                    params={"base": source.code, "currencies": symbols, "date": day.isoformat()},
                )
                payload = response.json()
            except Exception as exc:
                raise BackendError(f"{self.name} historical request failed: {exc}") from exc
            out.extend(
                _quotes_from_rates(
                    payload.get("rates", {}),
                    source=source, from_ts=from_ts, to_ts=to_ts, sampling=sampling,
                )
            )
            day += dt.timedelta(days=1)
        return out

    def fetch_latest(self, session, *, source, targets, at):
        symbols = ",".join(str(t) for t in targets)
        try:
            response = session.get(
                "/latest", params={"base": source.code, "currencies": symbols},
            )
            payload = response.json()
        except Exception as exc:
            raise BackendError(f"{self.name} latest request failed: {exc}") from exc

        from_ts = at or dt.datetime.now(dt.timezone.utc)
        to_ts = from_ts + dt.timedelta(days=1)
        return _quotes_from_rates(
            payload.get("rates", {}),
            source=source, from_ts=from_ts, to_ts=to_ts, sampling=self.default_sampling,
        )


class ErApi(Backend):
    """Latest-only feed via ``v6.exchangerate-api.com`` (free open endpoint).

    The open endpoint ``/v6/latest/{base}`` returns
    ``{"conversion_rates": {ccy: rate}}`` for the latest snapshot only.
    It has no historical range, so ``fetch_timeseries`` broadcasts the
    latest snapshot across the requested window — useful as a last-resort
    fallback when the dated providers are down.
    """

    name = "erapi"
    base_url = "https://v6.exchangerate-api.com"
    default_sampling = "1d"

    def _latest_rates(self, session, source, targets) -> dict[str, float]:
        wanted = {str(t) for t in targets}
        try:
            response = session.get(f"/v6/latest/{source.code}")
            payload = response.json()
        except Exception as exc:
            raise BackendError(f"{self.name} latest request failed: {exc}") from exc
        rates = payload.get("conversion_rates", {})
        return {code: rate for code, rate in rates.items() if code in wanted}

    def fetch_timeseries(self, session, *, source, targets, start, end, sampling):
        rates = self._latest_rates(session, source, targets)
        out: list[FxQuote] = []
        day = start.date()
        last = end.date()
        while day <= last:
            from_ts = dt.datetime.combine(day, dt.time.min, tzinfo=dt.timezone.utc)
            to_ts = from_ts + dt.timedelta(days=1)
            out.extend(
                _quotes_from_rates(
                    rates, source=source, from_ts=from_ts, to_ts=to_ts, sampling=sampling,
                )
            )
            day += dt.timedelta(days=1)
        return out

    def fetch_latest(self, session, *, source, targets, at):
        rates = self._latest_rates(session, source, targets)
        from_ts = at or dt.datetime.now(dt.timezone.utc)
        to_ts = from_ts + dt.timedelta(days=1)
        return _quotes_from_rates(
            rates, source=source, from_ts=from_ts, to_ts=to_ts, sampling=self.default_sampling,
        )
