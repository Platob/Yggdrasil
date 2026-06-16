"""Input coercion + the :class:`FxRate` orchestrator.

:class:`FxRate` is the trading-facing entry point: hand it a list of
currency pairs and a date window, it fans the request out across the
configured :class:`~yggdrasil.fxrate.backends.Backend` chain (grouping
pairs by source currency so each backend is hit once per source),
falls back to the next backend on :class:`BackendError`, and assembles
the resulting quotes straight into a long-format polars frame.

The coercion helpers (``_coerce_currency`` / ``_coerce_pair`` /
``_coerce_datetime`` / ``_group_pairs_by_source``) are the per-call
input shaping the orchestrator runs before it ever touches the wire;
they are kept module-level (and benchmarked directly) because they are
the hot path that runs on every ``fetch``.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import TYPE_CHECKING, Iterable, Sequence

from yggdrasil.enums.currency import Currency

from .backends import Backend, BackendError, FxQuote

if TYPE_CHECKING:
    import polars

__all__ = ["FxRate"]

logger = logging.getLogger("yggdrasil.fxrate")

# Long-frame column order — every quote lands here regardless of which
# backend produced it, so downstream joins/aggregations are stable.
_QUOTE_COLUMNS: tuple[str, ...] = (
    "source",
    "target",
    "from_timestamp",
    "to_timestamp",
    "sampling",
    "value",
)


def _coerce_currency(value: Currency | str | None) -> Currency:
    """Normalise any currency shape to a :class:`Currency`.

    Identity short-circuits, plain ISO codes and aliases (``"$"`` →
    ``USD``) both route through :meth:`Currency.from_`, which keeps a
    prebuilt instance cache so the hot path pays no allocation.
    """
    return Currency.from_(value)


def _coerce_pair(pair: tuple[Currency | str, Currency | str]) -> tuple[Currency, Currency]:
    """Normalise a ``(source, target)`` pair to ``(Currency, Currency)``."""
    source, target = pair
    return Currency.from_(source), Currency.from_(target)


def _coerce_datetime(value: dt.datetime | dt.date | str | int | float) -> dt.datetime:
    """Normalise any datetime shape to a timezone-aware UTC ``datetime``.

    Accepts a live ``datetime`` (returned as-is when it already carries a
    tzinfo, stamped UTC when naive), a ``date``, an ISO-8601 string
    (``"2024-01-01"`` or ``"2024-01-01T10:00:00+00:00"``), or a POSIX
    epoch number.
    """
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min, tzinfo=dt.timezone.utc)
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)
    if isinstance(value, str):
        # ``date.fromisoformat`` rejects the time component, so try the
        # full datetime parse first and fall back to date-only.
        try:
            parsed = dt.datetime.fromisoformat(value)
        except ValueError:
            parsed = dt.datetime.combine(
                dt.date.fromisoformat(value), dt.time.min,
            )
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
    raise TypeError(
        f"Cannot coerce {type(value).__name__} to datetime; pass a "
        f"datetime, date, ISO-8601 string, or epoch number"
    )


def _group_pairs_by_source(
    pairs: Sequence[tuple[Currency, Currency]],
) -> dict[Currency, list[Currency]]:
    """Bucket ``(source, target)`` pairs into ``{source: [targets]}``.

    One backend call serves all targets of a given source, so this is
    the fan-out plan. Insertion order is preserved per source and
    duplicate targets are dropped.
    """
    grouped: dict[Currency, list[Currency]] = {}
    for source, target in pairs:
        targets = grouped.get(source)
        if targets is None:
            grouped[source] = [target]
        elif target not in targets:
            targets.append(target)
    return grouped


class FxRate:
    """Orchestrate FX quote retrieval across a fallback chain of backends.

    Construct with an ordered list of backends — the first one that
    answers wins, the rest are tried in order when a backend raises
    :class:`BackendError`. ``fetch`` returns a long-format polars frame
    (or :class:`polars.LazyFrame` when ``lazy=True``); ``latest`` is the
    spot-rate flavour.
    """

    def __init__(self, backends: Iterable[Backend]) -> None:
        self.backends: tuple[Backend, ...] = tuple(backends)
        if not self.backends:
            raise ValueError("FxRate needs at least one backend")
        # One reusable HTTPSession per backend, keyed by base_url so
        # concrete backends share keep-alive sockets across calls. Built
        # lazily on first use — stub backends in tests/benches never
        # touch it.
        self._sessions: dict[str, object] = {}

    def _session_for(self, backend: Backend) -> object:
        cached = self._sessions.get(backend.base_url)
        if cached is not None:
            return cached
        from yggdrasil.http_ import HTTPSession

        session = HTTPSession(base_url=backend.base_url)
        self._sessions[backend.base_url] = session
        return session

    def fetch(
        self,
        *,
        pairs: Sequence[tuple[Currency | str, Currency | str]],
        start: dt.datetime | dt.date | str | int | float,
        end: dt.datetime | dt.date | str | int | float,
        sampling: str | None = None,
        lazy: bool = False,
        geo: bool = False,
    ) -> "polars.DataFrame | polars.LazyFrame":
        """Fetch a timeseries of quotes for ``pairs`` over ``[start, end]``.

        Pairs are grouped by source currency and each group is served by
        the first backend that doesn't raise :class:`BackendError`. The
        collected quotes are assembled into a long frame; pass
        ``geo=True`` to left-join country geography per target currency,
        ``lazy=True`` to get a :class:`polars.LazyFrame` back.
        """
        coerced = [_coerce_pair(pair) for pair in pairs]
        start_dt = _coerce_datetime(start)
        end_dt = _coerce_datetime(end)
        grouped = _group_pairs_by_source(coerced)

        quotes: list[FxQuote] = []
        for source, targets in grouped.items():
            quotes.extend(
                self._call_with_fallback(
                    lambda backend, session: backend.fetch_timeseries(
                        session,
                        source=source,
                        targets=targets,
                        start=start_dt,
                        end=end_dt,
                        sampling=sampling or self.backends[0].default_sampling,
                    ),
                    source=source,
                    targets=targets,
                )
            )

        return self._assemble(quotes, lazy=lazy, geo=geo)

    def latest(
        self,
        *,
        pairs: Sequence[tuple[Currency | str, Currency | str]],
        at: dt.datetime | dt.date | str | int | float | None = None,
        lazy: bool = False,
        geo: bool = False,
    ) -> "polars.DataFrame | polars.LazyFrame":
        """Fetch the latest spot quote for each pair (optionally ``at`` a time)."""
        coerced = [_coerce_pair(pair) for pair in pairs]
        at_dt = _coerce_datetime(at) if at is not None else None
        grouped = _group_pairs_by_source(coerced)

        quotes: list[FxQuote] = []
        for source, targets in grouped.items():
            quotes.extend(
                self._call_with_fallback(
                    lambda backend, session: backend.fetch_latest(
                        session, source=source, targets=targets, at=at_dt,
                    ),
                    source=source,
                    targets=targets,
                )
            )

        return self._assemble(quotes, lazy=lazy, geo=geo)

    def _call_with_fallback(self, call, *, source: Currency, targets: list[Currency]) -> list[FxQuote]:
        """Walk the backend chain until one answers; raise if all fail."""
        last_error: BackendError | None = None
        for backend in self.backends:
            try:
                return call(backend, self._session_for(backend))
            except BackendError as exc:
                last_error = exc
                logger.warning(
                    "Fetch FxQuote %r failed on backend %r, falling back (targets=%d, error=%s)",
                    str(source), backend.name, len(targets), exc,
                )
        raise BackendError(
            f"All {len(self.backends)} backend(s) failed for source {source!s} "
            f"(targets={[str(t) for t in targets]}); last error: {last_error}"
        ) from last_error

    def _assemble(
        self, quotes: list[FxQuote], *, lazy: bool, geo: bool,
    ) -> "polars.DataFrame | polars.LazyFrame":
        """Build the long polars frame from collected quotes (+ optional geo)."""
        from yggdrasil.lazy_imports import polars as pl

        # Column-oriented construction — no per-row dict, no pandas
        # detour. Empty result still yields the typed empty frame so
        # downstream schema expectations hold.
        if quotes:
            frame = pl.DataFrame(
                {
                    "source": [q.source for q in quotes],
                    "target": [q.target for q in quotes],
                    "from_timestamp": [q.from_timestamp for q in quotes],
                    "to_timestamp": [q.to_timestamp for q in quotes],
                    "sampling": [q.sampling for q in quotes],
                    "value": [q.value for q in quotes],
                },
                schema={
                    "source": pl.Utf8,
                    "target": pl.Utf8,
                    "from_timestamp": pl.Datetime(time_zone="UTC"),
                    "to_timestamp": pl.Datetime(time_zone="UTC"),
                    "sampling": pl.Utf8,
                    "value": pl.Float64,
                },
            )
        else:
            frame = pl.DataFrame(
                schema={
                    "source": pl.Utf8,
                    "target": pl.Utf8,
                    "from_timestamp": pl.Datetime(time_zone="UTC"),
                    "to_timestamp": pl.Datetime(time_zone="UTC"),
                    "sampling": pl.Utf8,
                    "value": pl.Float64,
                }
            )

        if geo:
            frame = self._enrich_geo(frame)

        return frame.lazy() if lazy else frame

    def _enrich_geo(self, frame: "polars.DataFrame") -> "polars.DataFrame":
        """Left-join country geography onto each quote by target currency.

        Uses :meth:`GeoZoneCatalog.with_country_geozones` — a singleton,
        so the country index is built/fetched once and reused. Rows whose
        target currency has no catalog entry keep null geo columns.
        """
        from yggdrasil.enums.geozone.catalog import GeoZoneCatalog
        from yggdrasil.lazy_imports import polars as pl

        catalog = _country_catalog()
        rows = catalog.to_rows()
        if not rows:
            # No catalog data (network-restricted environment) — return frame
            # with null geo columns so the schema stays consistent.
            return frame.with_columns([
                pl.lit(None).cast(pl.Utf8).alias("country_iso"),
                pl.lit(None).cast(pl.Utf8).alias("region_iso"),
                pl.lit(None).cast(pl.Utf8).alias("sub_iso"),
                pl.lit(None).cast(pl.Utf8).alias("country_name"),
                pl.lit(None).cast(pl.Float64).alias("country_lat"),
                pl.lit(None).cast(pl.Float64).alias("country_lon"),
            ])
        # One geo row per currency (catalog can list several countries per
        # currency — e.g. EUR — so keep the first deterministic match).
        geo = (
            pl.DataFrame(rows)
            .filter(pl.col("ccy").is_not_null())
            .unique(subset=["ccy"], keep="first")
            .select(
                pl.col("ccy").alias("_geo_ccy"),
                pl.col("country_iso"),
                pl.col("region_iso"),
                pl.col("sub_iso"),
                pl.col("name").alias("country_name"),
                pl.col("lat").alias("country_lat"),
                pl.col("lon").alias("country_lon"),
            )
        )
        return frame.join(
            geo,
            left_on="target",
            right_on="_geo_ccy",
            how="left",
        )


def _country_catalog():
    """Build (once) the country-enriched catalog used for geo joins.

    Returns an empty catalog when the upstream REST Countries fetch fails
    (network-restricted environments) so geo=True degrades gracefully instead
    of raising.
    """
    global _COUNTRY_CATALOG
    if _COUNTRY_CATALOG is None:
        from yggdrasil.enums.geozone.catalog import GeoZoneCatalog
        try:
            _COUNTRY_CATALOG = GeoZoneCatalog.empty().with_country_geozones()
        except Exception:
            _COUNTRY_CATALOG = GeoZoneCatalog.empty()
    return _COUNTRY_CATALOG


# Module-level cache so the (HTTP-fetched) country catalog is assembled
# at most once per process — the bench warms it explicitly before timing
# geo=True so the steady-state cost is the join, not the fetch.
_COUNTRY_CATALOG = None
