"""FX rate session — coerce inputs, route to backends, assemble a frame.

``FxRate`` is the public entry point: give it currency pairs in whatever
shape you have them (``Currency`` enums, ISO codes, symbols like ``"$"``)
and it returns a polars frame of quotes. It groups pairs by source so each
backend call fetches one source against many targets, and falls through to
the next backend on :class:`BackendError`.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from yggdrasil.enums.currency import Currency
from yggdrasil.exceptions.base import YGGException
from yggdrasil.fxrate.backends import Backend
from yggdrasil.lazy_imports import polars as pl

LOGGER = logging.getLogger(__name__)

_QUOTE_COLUMNS = ["source", "target", "from_timestamp", "to_timestamp", "sampling", "value"]


class BackendError(YGGException):
    """A backend failed to satisfy a request; the session tries the next one."""


@dataclass(slots=True)
class FxQuote:
    """A single FX observation: ``value`` units of ``target`` per ``source``."""

    source: str
    target: str
    from_timestamp: dt.datetime
    to_timestamp: dt.datetime
    sampling: str
    value: float


def _coerce_currency(x: object) -> Currency:
    """Coerce *x* (``Currency`` / ISO code / symbol alias) to a ``Currency``."""
    return Currency.from_(x)


def _coerce_datetime(x: object) -> dt.datetime:
    """Coerce *x* (ISO string / epoch-ms int / ``datetime``) to a ``datetime``."""
    if isinstance(x, dt.datetime):
        return x
    if isinstance(x, dt.date):
        return dt.datetime(x.year, x.month, x.day, tzinfo=dt.timezone.utc)
    if isinstance(x, (int, float)):
        return dt.datetime.fromtimestamp(x / 1000.0, tz=dt.timezone.utc)
    if isinstance(x, str):
        s = x.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return dt.datetime.fromisoformat(s)
        except ValueError as exc:
            raise BackendError(
                f"Cannot parse {x!r} as a datetime. Expected an ISO-8601 string "
                f"(e.g. '2024-01-31' or '2024-01-31T12:00:00Z'), an epoch-millisecond "
                f"int, or a datetime instance."
            ) from exc
    raise BackendError(
        f"Cannot coerce {type(x).__name__} to datetime; pass an ISO string, "
        f"epoch-ms int, or datetime."
    )


def _coerce_pair(pair: tuple) -> tuple[Currency, Currency]:
    """Coerce both legs of a ``(source, target)`` pair to ``Currency``."""
    if len(pair) != 2:
        raise BackendError(
            f"FX pair must be a 2-tuple (source, target), got {pair!r} "
            f"with {len(pair)} element(s)."
        )
    return _coerce_currency(pair[0]), _coerce_currency(pair[1])


def _group_pairs_by_source(pairs: list[tuple]) -> dict[Currency, list[Currency]]:
    """Group ``[(EUR,USD),(EUR,GBP),(USD,JPY)]`` into ``{EUR:[USD,GBP], USD:[JPY]}``."""
    grouped: dict[Currency, list[Currency]] = {}
    for pair in pairs:
        source, target = _coerce_pair(pair)
        bucket = grouped.setdefault(source, [])
        if target not in bucket:
            bucket.append(target)
    return grouped


class FxRate:
    """Multi-backend FX rate session returning polars frames."""

    def __init__(self, backends: list[Backend] | None = None) -> None:
        if backends is None:
            from yggdrasil.fxrate.frankfurter import FrankfurterBackend

            backends = [FrankfurterBackend()]
        self._backends = list(backends)
        self._session = None

    @property
    def session(self):
        if self._session is None:
            from yggdrasil.http_.session import HTTPSession

            self._session = HTTPSession()
        return self._session

    def fetch(
        self,
        *,
        pairs: list[tuple],
        start: object,
        end: object,
        sampling: str = "1d",
        lazy: bool = False,
        geo: bool = False,
    ):
        start_dt = _coerce_datetime(start)
        end_dt = _coerce_datetime(end)
        grouped = _group_pairs_by_source(pairs)

        quotes: list[FxQuote] = []
        for source, targets in grouped.items():
            quotes.extend(
                self._walk_backends(
                    lambda b: b.fetch_timeseries(
                        self.session,
                        source=source,
                        targets=targets,
                        start=start_dt,
                        end=end_dt,
                        sampling=sampling,
                    ),
                    source=source,
                    targets=targets,
                )
            )

        frame = self._to_frame(quotes)
        return frame.lazy() if lazy else frame

    def latest(self, *, pairs: list[tuple], at: object | None = None):
        at_dt = _coerce_datetime(at) if at is not None else None
        grouped = _group_pairs_by_source(pairs)

        quotes: list[FxQuote] = []
        for source, targets in grouped.items():
            quotes.extend(
                self._walk_backends(
                    lambda b: b.fetch_latest(
                        self.session, source=source, targets=targets, at=at_dt
                    ),
                    source=source,
                    targets=targets,
                )
            )
        return self._to_frame(quotes)

    def _walk_backends(self, call, *, source: Currency, targets: list[Currency]) -> list[FxQuote]:
        last: BackendError | None = None
        for backend in self._backends:
            try:
                return call(backend)
            except BackendError as exc:
                LOGGER.warning(
                    "Backend %r failed for %r->%r; trying next (%s)",
                    backend.name, source.code, [t.code for t in targets], exc,
                )
                last = exc
        raise BackendError(
            f"All {len(self._backends)} backend(s) failed for "
            f"{source.code}->{[t.code for t in targets]}. Last error: {last}"
        )

    def _to_frame(self, quotes: list[FxQuote]):
        if not quotes:
            return pl.DataFrame(schema={
                "source": pl.Utf8,
                "target": pl.Utf8,
                "from_timestamp": pl.Datetime,
                "to_timestamp": pl.Datetime,
                "sampling": pl.Utf8,
                "value": pl.Float64,
            })
        return pl.DataFrame(
            {
                "source": [q.source for q in quotes],
                "target": [q.target for q in quotes],
                "from_timestamp": [q.from_timestamp for q in quotes],
                "to_timestamp": [q.to_timestamp for q in quotes],
                "sampling": [q.sampling for q in quotes],
                "value": [q.value for q in quotes],
            }
        )
