"""FxRate — multi-backend FX orchestration with polars output.

Coercion helpers are module-level so the benchmark can import and time
them individually (they're the per-call hot path before any I/O).
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any, Sequence

import polars as pl

from yggdrasil.enums.currency import Currency

log = logging.getLogger(__name__)

__all__ = ["FxQuote", "FxRate",
           "_coerce_currency", "_coerce_datetime", "_coerce_pair",
           "_group_pairs_by_source"]


@dataclass(slots=True)
class FxQuote:
    """One (source, target) exchange-rate observation for a time window."""

    source: str
    target: str
    from_timestamp: dt.datetime
    to_timestamp: dt.datetime
    sampling: str
    value: float


# ---------------------------------------------------------------------------
# Coercion helpers — benchmarked directly, keep these tight
# ---------------------------------------------------------------------------


def _coerce_currency(x: Any) -> Currency:
    return Currency.from_(x)


def _coerce_pair(pair: tuple[Any, Any]) -> tuple[Currency, Currency]:
    return (_coerce_currency(pair[0]), _coerce_currency(pair[1]))


def _coerce_datetime(x: Any) -> dt.datetime:
    if isinstance(x, dt.datetime):
        return x if x.tzinfo is not None else x.replace(tzinfo=dt.timezone.utc)
    if isinstance(x, dt.date):
        return dt.datetime(x.year, x.month, x.day, tzinfo=dt.timezone.utc)
    if isinstance(x, (int, float)):
        return dt.datetime.fromtimestamp(x, tz=dt.timezone.utc)
    if isinstance(x, str):
        # 10-char "YYYY-MM-DD" is the dominant case. ``date.fromisoformat`` +
        # ``datetime.combine`` is ~3x faster here than constructing a datetime
        # from sliced int fields (656ns → 207ns/call), and keeps the same
        # ValueError on a malformed date.
        if len(x) == 10:
            try:
                return dt.datetime.combine(
                    dt.date.fromisoformat(x), dt.time.min, dt.timezone.utc,
                )
            except ValueError:
                raise ValueError(f"Cannot parse datetime from string: {x!r}")
        # Full ISO with offset or Z.
        try:
            ts = dt.datetime.fromisoformat(x.replace("Z", "+00:00"))
            return ts if ts.tzinfo is not None else ts.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            raise ValueError(f"Cannot parse datetime from string: {x!r}")
    raise TypeError(f"Cannot coerce {type(x).__name__} to datetime; expected str/date/datetime/int")


def _group_pairs_by_source(
    pairs: Sequence[tuple[Currency, Currency]],
) -> dict[str, list[str]]:
    """Group (source, target) Currency pairs by source ISO code."""
    out: dict[str, list[str]] = {}
    for src, tgt in pairs:
        key = src.code
        if key not in out:
            out[key] = []
        out[key].append(tgt.code)
    return out


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------


def _quotes_to_frame(quotes: list[FxQuote]) -> pl.DataFrame:
    if not quotes:
        return pl.DataFrame(schema={
            "source": pl.Utf8,
            "target": pl.Utf8,
            "from_timestamp": pl.Datetime("us", "UTC"),
            "to_timestamp": pl.Datetime("us", "UTC"),
            "sampling": pl.Utf8,
            "value": pl.Float64,
        })
    return pl.DataFrame({
        "source": [q.source for q in quotes],
        "target": [q.target for q in quotes],
        "from_timestamp": [q.from_timestamp for q in quotes],
        "to_timestamp": [q.to_timestamp for q in quotes],
        "sampling": [q.sampling for q in quotes],
        "value": [q.value for q in quotes],
    })


#: Process-wide cache of the currency-code → (country, region) lookup frame.
#: Building it pulls the REST-countries catalog over HTTP and indexes ~250
#: zones — far too costly to repeat per ``geo=True`` call. Sentinel ``False``
#: means "not built yet"; ``None`` means "built but unavailable" (offline /
#: empty catalog) so we don't retry the failing fetch on every call.
_CURRENCY_META: pl.DataFrame | None | bool = False


def _currency_meta() -> pl.DataFrame | None:
    """The currency-code → (country, region) lookup frame, built once and cached.

    Returns ``None`` when the geozone catalog can't be assembled (e.g. offline);
    callers fall back to leaving the frame un-enriched. The first successful
    build is memoized for the life of the process, so repeated ``geo=True``
    fetches pay only the join cost, not the catalog fetch + index.
    """
    global _CURRENCY_META
    if _CURRENCY_META is not False:
        return _CURRENCY_META  # type: ignore[return-value]
    try:
        from yggdrasil.enums.geozone import GeoZoneCatalog
        catalog = GeoZoneCatalog.empty().with_country_geozones()
        seen: set[str] = set()
        rows: list[dict[str, str | None]] = []
        for z in catalog.zones:
            ccy = z.ccy
            if not ccy or ccy in seen:
                continue
            seen.add(ccy)
            rows.append({"currency": ccy, "country": z.country_iso or z.name, "region": z.region_iso})
        _CURRENCY_META = pl.DataFrame(rows) if rows else None
    except Exception:
        _CURRENCY_META = None
    return _CURRENCY_META  # type: ignore[return-value]


def _enrich_geo(frame: pl.DataFrame) -> pl.DataFrame:
    """Left-join country/region metadata for source and target currencies.

    The catalog can carry several countries per currency (EUR, USD); the lookup
    keeps the first seen per code so the join stays one-to-one. Gracefully
    degrades to the untouched frame when the catalog is unavailable — geo
    enrichment is a convenience, never a hard dependency.
    """
    meta = _currency_meta()
    if meta is None:
        return frame
    return frame.join(
        meta.rename({"currency": "source", "country": "source_country", "region": "source_region"}),
        on="source", how="left",
    ).join(
        meta.rename({"currency": "target", "country": "target_country", "region": "target_region"}),
        on="target", how="left",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class FxRate:
    """Multi-backend FX rate fetcher with polars output.

    Tries each backend in order and falls back on :class:`BackendError`.
    The HTTP session is created lazily so stubs/tests that never call a
    real endpoint pay zero session-setup cost.

    Usage::

        fx = FxRate()  # uses Frankfurter → Fawaz fallback chain
        df = fx.fetch(pairs=[("EUR", "USD"), ("EUR", "GBP")],
                      start="2024-01-01", end="2024-01-31")
        latest = fx.latest(pairs=[("EUR", "USD")])
    """

    def __init__(self, backends: Sequence[Any] | None = None) -> None:
        if backends is None:
            from .backends import FawazBackend, FrankfurterBackend
            backends = [FrankfurterBackend(), FawazBackend()]
        self._backends: list[Any] = list(backends)
        self._session: Any = None  # lazy: only created on first real HTTP call

    @property
    def _http(self) -> Any:
        if self._session is None:
            from yggdrasil.http_ import HTTPSession
            self._session = HTTPSession()
        return self._session

    def fetch(
        self,
        pairs: Sequence[tuple[Any, Any]],
        start: Any,
        end: Any,
        sampling: str = "1d",
        lazy: bool = False,
        geo: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Fetch a time series for the given currency pairs.

        Args:
            pairs: List of (source, target) pairs; each element is coerced
                by :func:`_coerce_pair` (accepts ISO strings, ``$``/``€``
                aliases, or :class:`Currency` instances).
            start: Start date — string ``"YYYY-MM-DD"``, :class:`datetime.date`,
                :class:`datetime.datetime`, or a Unix timestamp ``int/float``.
            end: End date — same forms as ``start``.
            sampling: Desired sampling interval (``"1d"`` default; backends
                may not support sub-daily granularity).
            lazy: Return a :class:`polars.LazyFrame` instead of collecting.
            geo: Enrich the frame with country/region metadata for each
                currency (joins from the geozone catalog; silently skipped
                if the catalog is unavailable).

        Returns:
            A polars DataFrame (or LazyFrame if ``lazy=True``) with columns
            ``source``, ``target``, ``from_timestamp``, ``to_timestamp``,
            ``sampling``, ``value``.
        """
        coerced = [_coerce_pair(p) for p in pairs]
        groups = _group_pairs_by_source(coerced)
        start_dt = _coerce_datetime(start)
        end_dt = _coerce_datetime(end)

        all_quotes: list[FxQuote] = []
        for source, targets in groups.items():
            quotes = self._fetch_with_fallback(
                "timeseries",
                source=source,
                targets=targets,
                start=start_dt.date(),
                end=end_dt.date(),
                sampling=sampling,
            )
            all_quotes.extend(quotes)

        frame = _quotes_to_frame(all_quotes)
        if geo:
            frame = _enrich_geo(frame)
        return frame.lazy() if lazy else frame

    def latest(
        self,
        pairs: Sequence[tuple[Any, Any]],
        at: Any = None,
    ) -> pl.DataFrame:
        """Fetch the most recent rate for each pair.

        Args:
            pairs: Currency pairs (same coercion as :meth:`fetch`).
            at: Optional point-in-time — same forms as ``start`` in
                :meth:`fetch`. Defaults to today.
        """
        coerced = [_coerce_pair(p) for p in pairs]
        groups = _group_pairs_by_source(coerced)
        at_dt = _coerce_datetime(at) if at is not None else None

        all_quotes: list[FxQuote] = []
        for source, targets in groups.items():
            quotes = self._fetch_with_fallback(
                "latest",
                source=source,
                targets=targets,
                at=at_dt,
            )
            all_quotes.extend(quotes)
        return _quotes_to_frame(all_quotes)

    def _fetch_with_fallback(self, mode: str, **kwargs: Any) -> list[FxQuote]:
        from .backends import BackendError

        last_err: Exception | None = None
        for backend in self._backends:
            try:
                if mode == "timeseries":
                    return backend.fetch_timeseries(self._http, **kwargs)
                else:
                    return backend.fetch_latest(self._http, **kwargs)
            except BackendError as e:
                log.warning("backend %s failed: %s", backend.name, e)
                last_err = e
        raise BackendError(f"all backends failed; last error: {last_err}")
