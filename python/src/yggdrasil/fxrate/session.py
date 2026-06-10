"""FxRate — the orchestrator that turns a pair request into a tidy frame.

One :class:`FxRate` owns an ordered backend chain and the shared
:class:`HTTPSession`. :meth:`fetch` coerces the inputs (currencies, dates,
pairs), groups pairs by *source* so one HTTP call covers all targets of a
source, walks the backend chain per group (falling back on
:class:`BackendError`), assembles the quotes into a long polars frame, and
optionally enriches it with geography. :meth:`latest` is the spot variant.

Singleton-by-config: instances are cached on ``(backend names, base url)``
so repeated ``FxRate(backends=...)`` calls reuse one live session.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from yggdrasil.enums.currency import Currency

from .backends import DEFAULT_BACKENDS, Backend
from .exceptions import AllBackendsFailed, BackendError, NoBackendsError
from .quote import FxQuote

if TYPE_CHECKING:
    import polars as pl

LOGGER = logging.getLogger("yggdrasil.fxrate")

__all__ = ["FxRate", "FxRateClient"]

_INSTANCES: dict[tuple, "FxRate"] = {}


# ---------------------------------------------------------------------------
# input coercion
# ---------------------------------------------------------------------------

def _coerce_currency(value: Any) -> Currency:
    if isinstance(value, Currency):
        return value
    return Currency.from_(value)


def _coerce_pair(pair: Any) -> tuple[Currency, Currency]:
    if isinstance(pair, str):
        # "EUR/USD" or "EURUSD"
        sep = "/" if "/" in pair else ("-" if "-" in pair else None)
        if sep:
            src, tgt = pair.split(sep, 1)
        elif len(pair) == 6:
            src, tgt = pair[:3], pair[3:]
        else:
            raise ValueError(f"Cannot parse FX pair {pair!r}. Use 'EUR/USD' or ('EUR','USD').")
        return _coerce_currency(src), _coerce_currency(tgt)
    src, tgt = pair
    return _coerce_currency(src), _coerce_currency(tgt)


def _coerce_datetime(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min, tzinfo=dt.timezone.utc)
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
    if isinstance(value, str):
        parsed = dt.datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    raise TypeError(f"Cannot coerce {type(value).__name__} to a datetime.")


def _group_pairs_by_source(pairs: Iterable[tuple[Currency, Currency]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for src, tgt in pairs:
        bucket = grouped.setdefault(src.code, [])
        if tgt.code not in bucket:
            bucket.append(tgt.code)
    return grouped


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

class FxRate:
    def __new__(cls, *, backends: Sequence[Backend] | None = None, base_url: str | None = None) -> "FxRate":
        chain = tuple(backends) if backends else DEFAULT_BACKENDS
        # Identity keys on the concrete backend objects, not just their names —
        # two stubs both named "stub" with different quote lists are different
        # sessions, while the shared DEFAULT_BACKENDS instances dedupe to one.
        key = (tuple(id(b) for b in chain), base_url)
        cached = _INSTANCES.get(key)
        if cached is not None:
            return cached
        inst = super().__new__(cls)
        inst._initialized = False
        _INSTANCES[key] = inst
        return inst

    def __init__(self, *, backends: Sequence[Backend] | None = None, base_url: str | None = None) -> None:
        if self._initialized:
            return
        self.backends: tuple[Backend, ...] = tuple(backends) if backends else DEFAULT_BACKENDS
        if not self.backends:
            raise NoBackendsError("FxRate needs at least one backend.")
        self._base_url = base_url
        self._session = None
        self._initialized = True

    @property
    def session(self):
        if self._session is None:
            from yggdrasil.http_ import HTTPSession

            self._session = HTTPSession(base_url=self._base_url) if self._base_url else HTTPSession()
        return self._session

    def fetch(
        self,
        *,
        pairs: Sequence[Any],
        start: Any,
        end: Any,
        sampling: str | None = None,
        lazy: bool = False,
        geo: bool = False,
    ) -> "pl.DataFrame | pl.LazyFrame":
        start_dt = _coerce_datetime(start)
        end_dt = _coerce_datetime(end)
        grouped = _group_pairs_by_source([_coerce_pair(p) for p in pairs])

        quotes: list[FxQuote] = []
        for source, targets in grouped.items():
            quotes.extend(self._walk(
                lambda b, s=source, t=targets: b.fetch_timeseries(
                    self.session, source=s, targets=t,
                    start=start_dt, end=end_dt, sampling=sampling or b.default_sampling,
                ),
                source,
            ))
        return self._assemble(quotes, lazy=lazy, geo=geo)

    def latest(
        self,
        *,
        pairs: Sequence[Any],
        at: Any = None,
        geo: bool = False,
    ) -> "pl.DataFrame":
        at_dt = _coerce_datetime(at) if at is not None else dt.datetime.now(dt.timezone.utc)
        grouped = _group_pairs_by_source([_coerce_pair(p) for p in pairs])
        quotes: list[FxQuote] = []
        for source, targets in grouped.items():
            quotes.extend(self._walk(
                lambda b, s=source, t=targets: b.fetch_latest(
                    self.session, source=s, targets=t, at=at_dt,
                ),
                source,
            ))
        return self._assemble(quotes, lazy=False, geo=geo)

    def _walk(self, call, source: str) -> list[FxQuote]:
        failures: dict[str, Exception] = {}
        for backend in self.backends:
            try:
                return call(backend)
            except BackendError as exc:
                failures[backend.name] = exc
                LOGGER.warning("Backend %s failed for %s, falling back (%s)", backend.name, source, exc)
        raise AllBackendsFailed(failures)

    def _assemble(self, quotes: list[FxQuote], *, lazy: bool, geo: bool) -> "pl.DataFrame | pl.LazyFrame":
        import polars as pl

        if quotes:
            # Build the frame columnar from the quote slots — no per-row dict.
            frame = pl.DataFrame({
                "source": [q.source for q in quotes],
                "target": [q.target for q in quotes],
                "from_timestamp": [q.from_timestamp for q in quotes],
                "to_timestamp": [q.to_timestamp for q in quotes],
                "sampling": [q.sampling for q in quotes],
                "value": [q.value for q in quotes],
            })
        else:
            frame = pl.DataFrame(schema={
                "source": pl.Utf8, "target": pl.Utf8,
                "from_timestamp": pl.Datetime, "to_timestamp": pl.Datetime,
                "sampling": pl.Utf8, "value": pl.Float64,
            })
        if geo:
            from .geo import enrich_frame

            frame = enrich_frame(frame)
        return frame.lazy() if lazy else frame


#: Backwards-friendly alias — the task brief names the client ``FxRateClient``.
FxRateClient = FxRate
