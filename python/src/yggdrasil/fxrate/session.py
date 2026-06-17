"""FX rate orchestration — coercion, multi-backend fan-out, frame assembly.

:class:`FxRate` is the public entry point. It takes the shapes a caller
already has (``"EUR"`` / ``"$"`` / :class:`Currency`, ISO date strings /
epoch ints / datetimes), normalises them, groups pairs by source so one HTTP
call covers many targets, walks its backend chain with fallback on
:class:`BackendError`, and assembles the flat quote list into a tidy polars
frame using pre-built column arrays (no per-row dict churn).

    fx = FxRate()
    df = fx.fetch([("EUR", "USD"), ("EUR", "GBP")], "2024-01-01", "2024-01-31")
    df = fx.latest([("EUR", "USD")], geo=True)
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

from yggdrasil.enums.currency import Currency

from .backends import Backend, FawazBackend, FrankfurterBackend

if TYPE_CHECKING:
    import polars as pl

    from yggdrasil.http_.session import HTTPSession

__all__ = ["FxRate", "FxQuote", "BackendError"]

_log = logging.getLogger("yggdrasil.fxrate")

_UTC = dt.timezone.utc


class BackendError(Exception):
    """Raised by a backend when it can't serve a request.

    Signals :class:`FxRate` to fall back to the next backend in the chain
    rather than aborting the whole fetch.
    """


@dataclass(slots=True, frozen=True)
class FxQuote:
    """A single exchange-rate observation: 1 *source* = *value* *target*.

    Timestamps are timezone-aware UTC and bound the sample's validity window
    (``[from_timestamp, to_timestamp)``); ``sampling`` labels the granularity
    (``"1d"``, ``"1h"``, …).
    """

    source: str
    target: str
    from_timestamp: dt.datetime
    to_timestamp: dt.datetime
    sampling: str
    value: float


# ---------------------------------------------------------------------------
# Coercion helpers — accept what a caller has, normalise to canonical types.
# ---------------------------------------------------------------------------


def _coerce_currency(value: Currency | str) -> Currency:
    """``Currency`` / ISO code / alias (``"$"``, ``"€"``) → :class:`Currency`."""
    if type(value) is Currency:
        return value
    return Currency.from_(value)


def _coerce_pair(pair: Sequence[Currency | str]) -> tuple[Currency, Currency]:
    """``(source, target)`` in any currency shape → a pair of :class:`Currency`."""
    source, target = pair
    return _coerce_currency(source), _coerce_currency(target)


def _coerce_datetime(value: dt.datetime | dt.date | int | float | str) -> dt.datetime:
    """Any reasonable instant → a timezone-aware UTC :class:`datetime`.

    Accepts aware/naive datetimes (naive assumed UTC), dates (midnight UTC),
    epoch seconds (int/float), and ISO strings (date-only fast path, then full
    datetime with a trailing-``Z`` shim).
    """
    if type(value) is dt.datetime:
        return value.astimezone(_UTC) if value.tzinfo is not None else value.replace(tzinfo=_UTC)
    if isinstance(value, dt.datetime):
        return value.astimezone(_UTC) if value.tzinfo is not None else value.replace(tzinfo=_UTC)
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day, tzinfo=_UTC)
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value, _UTC)
    if isinstance(value, str):
        # Fast path: a bare ISO date ("2024-01-01") is the common caller shape.
        try:
            d = dt.date.fromisoformat(value)
            return dt.datetime(d.year, d.month, d.day, tzinfo=_UTC)
        except ValueError:
            pass
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(_UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=_UTC)
    raise TypeError(f"Cannot coerce {type(value).__name__} to datetime: {value!r}")


def _group_pairs_by_source(
    pairs: Sequence[tuple[Currency, Currency]],
) -> dict[Currency, list[Currency]]:
    """Collapse pairs into ``{source: [target, ...]}`` — one HTTP call per source.

    Preserves first-seen order of both sources and targets, and de-dupes
    repeated targets so a backend never fetches the same cross twice.
    """
    grouped: dict[Currency, list[Currency]] = {}
    for source, target in pairs:
        targets = grouped.get(source)
        if targets is None:
            grouped[source] = [target]
        elif target not in targets:
            targets.append(target)
    return grouped


#: Inline ISO-code → geography catalog for ``geo=True`` enrichment. Covers the
#: G10 plus the common Nordics/Antipodeans; unknown codes enrich to nulls.
_GEO_CATALOG: dict[str, dict[str, str]] = {
    "USD": {"country": "United States", "region": "North America"},
    "EUR": {"country": "Eurozone", "region": "Europe"},
    "GBP": {"country": "United Kingdom", "region": "Europe"},
    "JPY": {"country": "Japan", "region": "Asia"},
    "CHF": {"country": "Switzerland", "region": "Europe"},
    "CNY": {"country": "China", "region": "Asia"},
    "AUD": {"country": "Australia", "region": "Oceania"},
    "CAD": {"country": "Canada", "region": "North America"},
    "NZD": {"country": "New Zealand", "region": "Oceania"},
    "SEK": {"country": "Sweden", "region": "Europe"},
    "NOK": {"country": "Norway", "region": "Europe"},
    "DKK": {"country": "Denmark", "region": "Europe"},
}


class FxRate:
    """Multi-backend exchange-rate fetcher.

    Configure with a backend chain (defaults to Frankfurter → Fawaz); each
    fetch tries backends in order and rolls over on :class:`BackendError`.
    Shares one :class:`HTTPSession` per backend ``base_url`` for pooling.
    """

    def __init__(self, backends: Sequence[Backend] | None = None) -> None:
        # Default chain: range-native primary, snapshot CDN as fallback.
        self.backends: tuple[Backend, ...] = (
            tuple(backends) if backends is not None
            else (FrankfurterBackend(), FawazBackend())
        )
        # Lazily-built per-backend sessions and the geo lookup, populated on
        # first use so construction stays free of HTTP/polars imports.
        self._sessions: dict[str, HTTPSession] = {}
        self._geo_catalog: dict[str, dict[str, str]] | None = None

    def _session_for(self, backend: Backend) -> HTTPSession:
        session = self._sessions.get(backend.base_url)
        if session is not None:
            return session
        from yggdrasil.http_.session import HTTPSession

        session = HTTPSession(base_url=backend.base_url)
        self._sessions[backend.base_url] = session
        return session

    def fetch(
        self,
        pairs: Sequence[Sequence[Currency | str]],
        start: dt.datetime | dt.date | int | float | str,
        end: dt.datetime | dt.date | int | float | str,
        *,
        lazy: bool = False,
        geo: bool = False,
    ) -> "pl.DataFrame | pl.LazyFrame":
        """Fetch a rate timeseries for *pairs* over ``[start, end]``.

        Returns a tidy long frame (one row per source/target/timestamp). Set
        ``lazy=True`` for a ``LazyFrame``, ``geo=True`` to add source/target
        country + region columns.
        """
        start_dt = _coerce_datetime(start)
        end_dt = _coerce_datetime(end)
        grouped = _group_pairs_by_source(
            [_coerce_pair(p) for p in pairs]
        )

        quotes: list[FxQuote] = []
        for source, targets in grouped.items():
            quotes.extend(self._walk_backends(
                "fetch_timeseries", source, targets,
                start=start_dt, end=end_dt,
            ))

        frame = self._quotes_to_frame(quotes)
        if geo:
            frame = self._enrich_geo(frame)
        return frame.lazy() if lazy else frame

    def latest(
        self,
        pairs: Sequence[Sequence[Currency | str]],
        *,
        at: dt.datetime | dt.date | int | float | str | None = None,
        lazy: bool = False,
        geo: bool = False,
    ) -> "pl.DataFrame | pl.LazyFrame":
        """Fetch the most recent rate for *pairs* (optionally as of *at*)."""
        at_dt = _coerce_datetime(at) if at is not None else dt.datetime.now(_UTC)
        grouped = _group_pairs_by_source(
            [_coerce_pair(p) for p in pairs]
        )

        quotes: list[FxQuote] = []
        for source, targets in grouped.items():
            quotes.extend(self._walk_backends(
                "fetch_latest", source, targets, at=at_dt,
            ))

        frame = self._quotes_to_frame(quotes)
        if geo:
            frame = self._enrich_geo(frame)
        return frame.lazy() if lazy else frame

    def _walk_backends(
        self,
        method: str,
        source: Currency,
        targets: list[Currency],
        **kwargs: Any,
    ) -> list[FxQuote]:
        """Try each backend for one source group; fall back on BackendError."""
        last_error: BackendError | None = None
        for backend in self.backends:
            try:
                sampling = backend.default_sampling
                fn = getattr(backend, method)
                if method == "fetch_timeseries":
                    return fn(
                        self._session_for(backend),
                        source=source, targets=targets,
                        start=kwargs["start"], end=kwargs["end"],
                        sampling=sampling,
                    )
                return fn(
                    self._session_for(backend),
                    source=source, targets=targets, at=kwargs["at"],
                )
            except BackendError as exc:
                last_error = exc
                _log.warning(
                    "Fallback FxBackend %r (source=%s, targets=%s, reason=%s)",
                    backend.name, source.code,
                    ",".join(t.code for t in targets), exc,
                )
        raise BackendError(
            f"All {len(self.backends)} backend(s) failed for {source.code}->"
            f"{','.join(t.code for t in targets)}; last error: {last_error}"
        )

    def _quotes_to_frame(self, quotes: list[FxQuote]) -> "pl.DataFrame":
        """Build the long frame from quotes via column arrays (no list-of-dicts)."""
        import polars as pl

        schema = {
            "source": pl.Utf8,
            "target": pl.Utf8,
            "from_timestamp": pl.Datetime("us", "UTC"),
            "to_timestamp": pl.Datetime("us", "UTC"),
            "sampling": pl.Utf8,
            "value": pl.Float64,
        }
        if not quotes:
            return pl.DataFrame(schema=schema)

        # Transpose once into per-column lists — polars builds each column as a
        # contiguous array, far cheaper than inferring from a row-of-dicts.
        return pl.DataFrame(
            {
                "source": [q.source for q in quotes],
                "target": [q.target for q in quotes],
                "from_timestamp": [q.from_timestamp for q in quotes],
                "to_timestamp": [q.to_timestamp for q in quotes],
                "sampling": [q.sampling for q in quotes],
                "value": [q.value for q in quotes],
            },
            schema=schema,
        )

    def _build_geo_catalog(self) -> dict[str, dict[str, str]]:
        cached = self._geo_catalog
        if cached is not None:
            return cached
        self._geo_catalog = _GEO_CATALOG
        return _GEO_CATALOG

    def _enrich_geo(self, frame: "pl.DataFrame") -> "pl.DataFrame":
        """Add source/target country + region via vectorized map lookups."""
        import polars as pl

        catalog = self._build_geo_catalog()
        countries = {code: g["country"] for code, g in catalog.items()}
        regions = {code: g["region"] for code, g in catalog.items()}
        return frame.with_columns(
            pl.col("source").replace_strict(countries, default=None).alias("source_country"),
            pl.col("source").replace_strict(regions, default=None).alias("source_region"),
            pl.col("target").replace_strict(countries, default=None).alias("target_country"),
            pl.col("target").replace_strict(regions, default=None).alias("target_region"),
        )
