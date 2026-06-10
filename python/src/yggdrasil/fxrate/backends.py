"""FX backends — one adapter per upstream, a common :class:`Backend` contract.

Each backend turns one upstream's JSON into a list of :class:`FxQuote`.
The orchestrator (:class:`~yggdrasil.fxrate.session.FxRate`) tries them in
order and walks to the next on :class:`BackendError`. The contract is two
methods — :meth:`fetch_timeseries` (a window) and :meth:`fetch_latest`
(spot) — both handed the shared :class:`HTTPSession` so connections,
retries, and the response cache are reused across calls.

Built-in upstreams:

- :class:`FrankfurterBackend` — frankfurter.app (ECB reference rates, free,
  no key). Primary.
- :class:`ExchangeRateBackend` — open.er-api.com (free, no key). Fallback;
  latest-only, so its timeseries fans the spot rate across the window.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from .exceptions import BackendError
from .quote import FxQuote

if TYPE_CHECKING:
    from yggdrasil.http_ import HTTPSession

__all__ = ["Backend", "FrankfurterBackend", "ExchangeRateBackend", "DEFAULT_BACKENDS"]


class Backend(ABC):
    name: str
    base_url: str
    default_sampling: str = "1d"

    @abstractmethod
    def fetch_timeseries(
        self,
        session: "HTTPSession",
        *,
        source: str,
        targets: Sequence[str],
        start: dt.datetime,
        end: dt.datetime,
        sampling: str,
    ) -> list[FxQuote]:
        """Quotes for *source → each target* across ``[start, end]``."""

    @abstractmethod
    def fetch_latest(
        self,
        session: "HTTPSession",
        *,
        source: str,
        targets: Sequence[str],
        at: dt.datetime,
    ) -> list[FxQuote]:
        """Spot quotes for *source → each target* as of *at*."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def _day(d: dt.datetime) -> dt.datetime:
    return dt.datetime.combine(d.date(), dt.time.min, tzinfo=dt.timezone.utc)


class FrankfurterBackend(Backend):
    name = "frankfurter"
    base_url = "https://api.frankfurter.app"
    default_sampling = "1d"

    def fetch_timeseries(self, session, *, source, targets, start, end, sampling):
        to = ",".join(targets)
        url = f"{self.base_url}/{start.date():%Y-%m-%d}..{end.date():%Y-%m-%d}"
        resp = session.get(url, params={"from": source, "to": to}, raise_error=False)
        if resp.status_code != 200:
            raise BackendError(f"frankfurter timeseries HTTP {resp.status_code} for {source}->{to}.")
        body = resp.json()
        rates = body.get("rates")
        if not rates:
            raise BackendError(f"frankfurter returned no rates for {source}->{to}.")
        out: list[FxQuote] = []
        for day_str, by_ccy in rates.items():
            frm = dt.datetime.fromisoformat(day_str).replace(tzinfo=dt.timezone.utc)
            to_ts = frm + dt.timedelta(days=1)
            for tgt, value in by_ccy.items():
                out.append(FxQuote(source, tgt, frm, to_ts, sampling, float(value)))
        return out

    def fetch_latest(self, session, *, source, targets, at):
        to = ",".join(targets)
        resp = session.get(f"{self.base_url}/latest", params={"from": source, "to": to}, raise_error=False)
        if resp.status_code != 200:
            raise BackendError(f"frankfurter latest HTTP {resp.status_code} for {source}->{to}.")
        body = resp.json()
        rates = body.get("rates")
        if not rates:
            raise BackendError(f"frankfurter returned no latest rates for {source}->{to}.")
        frm = _day(at)
        to_ts = frm + dt.timedelta(days=1)
        return [FxQuote(source, tgt, frm, to_ts, "latest", float(v)) for tgt, v in rates.items()]


class ExchangeRateBackend(Backend):
    name = "er-api"
    base_url = "https://open.er-api.com/v6"
    default_sampling = "1d"

    def fetch_timeseries(self, session, *, source, targets, start, end, sampling):
        # ER-API is latest-only — fan the spot rate across the requested window
        # so the fallback still produces a frame of the right shape.
        spot = {q.target: q.value for q in self.fetch_latest(session, source=source, targets=targets, at=end)}
        out: list[FxQuote] = []
        day = _day(start)
        last = _day(end)
        while day <= last:
            nxt = day + dt.timedelta(days=1)
            for tgt in targets:
                v = spot.get(tgt)
                if v is not None:
                    out.append(FxQuote(source, tgt, day, nxt, sampling, v))
            day = nxt
        if not out:
            raise BackendError(f"er-api produced no quotes for {source}.")
        return out

    def fetch_latest(self, session, *, source, targets, at):
        resp = session.get(f"{self.base_url}/latest/{source}", raise_error=False)
        if resp.status_code != 200:
            raise BackendError(f"er-api HTTP {resp.status_code} for {source}.")
        body = resp.json()
        if body.get("result") != "success":
            raise BackendError(f"er-api error for {source}: {body.get('error-type', 'unknown')}.")
        rates = body.get("rates", {})
        frm = _day(at)
        to_ts = frm + dt.timedelta(days=1)
        out = [
            FxQuote(source, tgt, frm, to_ts, "latest", float(rates[tgt]))
            for tgt in targets if tgt in rates
        ]
        if not out:
            raise BackendError(f"er-api had no requested targets for {source}.")
        return out


#: Default fallback chain: ECB reference rates first, ER-API as the spare.
DEFAULT_BACKENDS: tuple[Backend, ...] = (FrankfurterBackend(), ExchangeRateBackend())
