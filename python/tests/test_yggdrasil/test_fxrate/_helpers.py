"""Stub backend + session helpers for FX rate unit tests.

The :class:`StubFxBackend` is a deterministic in-memory backend that
records every call and emits pre-loaded quotes — useful for asserting
the fallback chain, group fan-out, frame assembly, and geography
enrichment without touching the network.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Sequence

from yggdrasil.enums.currency import Currency
from yggdrasil.fxrate.backends import Backend
from yggdrasil.fxrate.session import FxQuote, FxRate


__all__ = ["StubFxBackend", "make_quote"]


def make_quote(
    source: str,
    target: str,
    *,
    date: str = "2024-01-01",
    value: float = 1.0,
    sampling: str = "1d",
) -> FxQuote:
    from_ts = dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc)
    return FxQuote(
        source=source.upper(),
        target=target.upper(),
        from_timestamp=from_ts,
        to_timestamp=from_ts + dt.timedelta(days=1),
        sampling=sampling,
        value=value,
    )


@dataclass
class StubFxBackend(Backend):
    """In-memory FX backend.

    Per call shape:

    * if :attr:`raise_with` is set, raise it (use :exc:`BackendError`
      for the fallback path; any other exception is treated as a
      hard error by :class:`FxRate`'s orchestrator);
    * else return :attr:`quotes` (defaults to empty — which mimics
      a backend that has no data for the pair and triggers a
      fallback).

    Tracks every call on :attr:`calls` so tests can assert which
    drivers were consulted.
    """

    name: str = "stub"
    base_url: str = "stub://"
    default_sampling: str = "1d"
    quotes: Sequence[FxQuote] = ()
    raise_with: Exception | None = None
    calls: list[dict] = field(default_factory=list)

    def fetch_timeseries(
        self,
        session: FxRate,
        *,
        source: Currency,
        targets: Sequence[Currency],
        start: dt.datetime,
        end: dt.datetime,
        sampling: str,
    ) -> Sequence[FxQuote]:
        self.calls.append({
            "kind": "timeseries",
            "source": source.code,
            "targets": tuple(t.code for t in targets),
            "start": start,
            "end": end,
            "sampling": sampling,
        })
        if self.raise_with is not None:
            raise self.raise_with
        return tuple(self.quotes)

    def fetch_latest(
        self,
        session: FxRate,
        *,
        source: Currency,
        targets: Sequence[Currency],
        at: dt.datetime,
    ) -> Sequence[FxQuote]:
        self.calls.append({
            "kind": "latest",
            "source": source.code,
            "targets": tuple(t.code for t in targets),
            "at": at,
        })
        if self.raise_with is not None:
            raise self.raise_with
        return tuple(self.quotes)
