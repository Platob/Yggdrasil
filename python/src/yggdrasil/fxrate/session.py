""":class:`FxRate` — single :class:`HTTPSession` that fetches FX rates
across multiple free public sources with automatic fallback.

One session, one connection pool, one retry policy, one cache tier
— but multiple wire-level data sources behind the same call.
:meth:`FxRate.fetch` tries each backend in :attr:`backends` order;
the first one that returns rows for a given ``(source, target window)``
group wins, and the rest are skipped. When a backend raises (HTTP
error, parse failure, unsupported currency) the session logs a
warning and rolls over to the next driver.

The public surface:

* :meth:`FxRate.fetch(pairs, start, end, sampling, geo=False, lazy=False)`
  → :class:`polars.DataFrame` long-format
  ``[source, target, from_timestamp, to_timestamp, sampling, value]``.
* :meth:`FxRate.latest(pairs, geo=False)` → same shape, one row per
  pair using the backend's cheapest snapshot endpoint.

Datetime arguments use the canonical
:func:`yggdrasil.data.cast.convert` registry — pass an ISO string,
a :class:`datetime`, epoch seconds, ``"now"`` / ``"utcnow"``, all
parse uniformly. Naive datetimes are read as UTC.

Geography enrichment routes through
:class:`yggdrasil.data.enums.geozone.GeoZoneCatalog` — the same
catalog every other yggdrasil module uses for country/region
lookup. ``geo=True`` triggers a lazy one-time fetch of the country
catalog (cached for the rest of the process), then joins per
currency's ``ccy`` token to surface country / lat / lon for each
side of the pair.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
from collections import defaultdict
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
)

from yggdrasil.data.cast import convert
from yggdrasil.data.enums.currency import Currency
from yggdrasil.io.http_.session import HTTPSession
from yggdrasil.io.url import URL

from .backends import Backend, BackendError, DEFAULT_BACKENDS

if TYPE_CHECKING:
    import polars as pl
    from yggdrasil.data.enums.geozone import GeoZone, GeoZoneCatalog


__all__ = [
    "FxRate",
    "FxQuote",
    "PairLike",
    "DateLike",
    "FX_FRAME_COLUMNS",
    "FX_FRAME_GEO_COLUMNS",
]


LOGGER = logging.getLogger(__name__)


#: ``(source, target)`` shape accepted by :meth:`FxRate.fetch`. Each
#: side accepts a :class:`Currency` instance, an ISO 4217 alpha-3
#: code, or any alias :meth:`Currency.parse_str` recognises
#: (``"$"`` → USD, ``"€"`` → EUR, ``"yen"`` → JPY, …).
PairLike = tuple[Union[Currency, str], Union[Currency, str]]


#: ``start`` / ``end`` shape: anything :func:`yggdrasil.data.cast
#: .convert` knows how to push into :class:`datetime.datetime`.
DateLike = Union[str, dt.datetime, dt.date, int, float, None]


#: Columns the public long frame ships, in order.
FX_FRAME_COLUMNS: tuple[str, ...] = (
    "source",
    "target",
    "from_timestamp",
    "to_timestamp",
    "sampling",
    "value",
)


#: Extra columns added when ``geo=True``. Lat/lon are WGS84
#: ``float64`` matching the codebase's geographic contract.
FX_FRAME_GEO_COLUMNS: tuple[str, ...] = (
    "source_country_iso",
    "source_lat",
    "source_lon",
    "target_country_iso",
    "target_lat",
    "target_lon",
)


class FxQuote(tuple):
    """One FX observation row.

    Tuple subclass with named accessors — keeps the row light enough
    that a few thousand of them assemble into a polars frame without
    extra copies, while still printing the way a row should in a REPL.
    """

    __slots__ = ()

    def __new__(
        cls,
        source: str,
        target: str,
        from_timestamp: dt.datetime,
        to_timestamp: dt.datetime,
        sampling: str,
        value: float,
    ) -> "FxQuote":
        return tuple.__new__(
            cls,
            (source, target, from_timestamp, to_timestamp, sampling, value),
        )

    @property
    def source(self) -> str: return self[0]  # noqa: E704
    @property
    def target(self) -> str: return self[1]  # noqa: E704
    @property
    def from_timestamp(self) -> dt.datetime: return self[2]  # noqa: E704
    @property
    def to_timestamp(self) -> dt.datetime: return self[3]  # noqa: E704
    @property
    def sampling(self) -> str: return self[4]  # noqa: E704
    @property
    def value(self) -> float: return self[5]  # noqa: E704

    def __repr__(self) -> str:
        return (
            f"FxQuote({self.source}->{self.target} "
            f"@ {self.from_timestamp.isoformat()} "
            f"sampling={self.sampling!r} value={self.value})"
        )


# ---------------------------------------------------------------------------
# Input coercion helpers
# ---------------------------------------------------------------------------


def _coerce_currency(value: Union[Currency, str]) -> Currency:
    if isinstance(value, Currency):
        return value
    try:
        return Currency.parse(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid currency {value!r}: {exc}. Pass an ISO 4217 alpha-3 "
            f"code (e.g. 'USD'), a :class:`Currency` instance, or a known "
            f"alias ('$', '€', '£', 'yen', 'franc')."
        ) from exc


def _coerce_pair(pair: PairLike) -> tuple[Currency, Currency]:
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError(
            f"FX pair must be a (source, target) tuple of length 2; got "
            f"{pair!r}. Example: ('EUR', 'USD')."
        )
    src = _coerce_currency(pair[0])
    tgt = _coerce_currency(pair[1])
    if src == tgt:
        raise ValueError(
            f"FX pair source and target are identical ({src.code!r}). "
            f"That'd be a constant 1.0 — drop the pair instead."
        )
    return src, tgt


def _coerce_datetime(
    value: DateLike, *, default: Optional[dt.datetime] = None,
) -> dt.datetime:
    if value is None:
        if default is None:
            raise ValueError("Datetime is required; got None and no default available.")
        return default
    parsed = convert(value, dt.datetime)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _group_pairs_by_source(
    pairs: Sequence[tuple[Currency, Currency]],
) -> dict[Currency, tuple[Currency, ...]]:
    grouped: dict[Currency, list[Currency]] = defaultdict(list)
    seen: dict[Currency, set[Currency]] = defaultdict(set)
    for src, tgt in pairs:
        if tgt in seen[src]:
            continue
        seen[src].add(tgt)
        grouped[src].append(tgt)
    return {src: tuple(tgts) for src, tgts in grouped.items()}


# ---------------------------------------------------------------------------
# Currency → GeoZone lookup (lazy, GeoZoneCatalog-backed)
# ---------------------------------------------------------------------------


_GEO_LOCK = threading.RLock()
_GEO_CATALOG: "GeoZoneCatalog | None" = None
_GEO_CATALOG_FAILED: bool = False


def _currency_geo_catalog() -> "GeoZoneCatalog | None":
    """Lazy-load the country-enriched :class:`GeoZoneCatalog`.

    The catalog ships a ``ccy`` field per country; we re-use the
    canonical loader so the lookup keys (token index) stay in sync
    with the rest of the codebase's geozone logic. The fetch goes
    over HTTP once per process — subsequent calls hit the cached
    instance. On network failure we cache the failure flag too so
    we don't retry the offline fetch on every FX call.
    """
    global _GEO_CATALOG, _GEO_CATALOG_FAILED
    if _GEO_CATALOG is not None:
        return _GEO_CATALOG
    if _GEO_CATALOG_FAILED:
        return None
    with _GEO_LOCK:
        if _GEO_CATALOG is not None:
            return _GEO_CATALOG
        if _GEO_CATALOG_FAILED:
            return None
        try:
            from yggdrasil.data.enums.geozone import load_geozones
            _GEO_CATALOG = load_geozones(include_countries=True)
        except Exception as exc:
            LOGGER.warning(
                "Loading country geozone catalog failed (%s) — falling back "
                "to the default catalog (sparse coverage).", exc,
            )
            try:
                from yggdrasil.data.enums.geozone import load_geozones
                _GEO_CATALOG = load_geozones()
            except Exception:
                _GEO_CATALOG_FAILED = True
                return None
        return _GEO_CATALOG


def _zone_for_currency(code: str) -> "GeoZone | None":
    """Find a :class:`GeoZone` whose ``ccy`` matches *code*.

    Strategy:

    1. ISO 4217 alpha-3 codes overwhelmingly start with the alpha-2
       ISO 3166 code of the issuing country (``USD`` → ``US``,
       ``GBP`` → ``GB``, ``JPY`` → ``JP``, ``CHF`` → ``CH``…).
       Look the alpha-2 prefix up first and accept the match when
       the country's ``ccy`` confirms the currency.
    2. Otherwise fall back to the catalog's currency-token index —
       which picks *some* country sharing the currency (the choice
       is deterministic for a given catalog instance but the
       semantics are "any holder"; useful for currencies with no
       single issuer like ``EUR`` / ``XOF`` / ``XAF``).
    """
    catalog = _currency_geo_catalog()
    if catalog is None:
        return None
    code_upper = code.upper()
    if len(code_upper) >= 2:
        iso2 = code_upper[:2]
        zone = catalog.lookup(iso2)
        if zone is not None and zone.ccy == code_upper:
            return zone
    return catalog.lookup(code_upper)


# ---------------------------------------------------------------------------
# FxRate session
# ---------------------------------------------------------------------------


class FxRate(HTTPSession):
    """:class:`HTTPSession` that fetches FX rates with multi-source fallback.

    Backends ship in priority order on :attr:`backends`. For each
    ``(source, [targets…])`` group, :meth:`fetch` walks the chain:
    the first backend that returns at least one row wins. When a
    backend raises a :class:`~yggdrasil.fxrate.backends.BackendError`
    (HTTP error, parse failure, unsupported currency) the session
    logs a warning and rolls over to the next driver. If every
    backend fails for a given group, the *last* error is re-raised
    so the caller sees something diagnostic.

    The session itself has **no** ``base_url`` — each backend builds
    its own absolute URL — so :class:`HTTPSession`'s singleton key
    collapses repeated ``FxRate()`` calls into one live connection
    pool regardless of which backends are configured.

    Construction:

    .. code-block:: python

        from yggdrasil.fxrate import FxRate

        fx = FxRate()                          # default backend chain
        # Or override the chain — e.g. skip Frankfurter, prefer Fawaz:
        from yggdrasil.fxrate.backends import FawazBackend, ErApiBackend
        fx_lite = FxRate(backends=(FawazBackend(), ErApiBackend()))

    Usage:

    .. code-block:: python

        df = fx.fetch(
            pairs=[("EUR", "USD"), ("EUR", "GBP")],
            start="2024-01-01",
            end="2024-01-10",
            sampling="1d",
        )
        # → polars.DataFrame
        #   source target from_timestamp        to_timestamp          sampling value
        #   EUR    GBP    2024-01-01T00:00:00Z  2024-01-02T00:00:00Z  1d       0.8671
        #   EUR    USD    2024-01-01T00:00:00Z  2024-01-02T00:00:00Z  1d       1.1064
        #   ...
    """

    DEFAULT_SAMPLING: str = "1d"

    def __init__(
        self,
        base_url: Optional[Union[URL, str]] = None,
        *,
        backends: Optional[Sequence[Backend]] = None,
        verify: bool = True,
        pool_maxsize: int = 10,
        headers: Optional[Mapping[str, str]] = None,
        waiting: Any = None,
        auth: Any = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            # Backends are transient — refresh them on re-entry so the
            # caller's intent wins over whatever the previous owner of
            # this singleton left behind. ``FxRate()`` (no kw) resets
            # to :data:`DEFAULT_BACKENDS`, ``FxRate(backends=...)``
            # pins the provided chain; the user-facing contract is
            # "what you pass is what you get".
            self.backends = tuple(backends) if backends is not None else DEFAULT_BACKENDS
            return
        kwargs: dict[str, Any] = {
            "base_url": base_url,
            "verify": verify,
            "pool_maxsize": pool_maxsize,
            "headers": dict(headers) if headers else None,
            "auth": auth,
        }
        if waiting is not None:
            kwargs["waiting"] = waiting
        super().__init__(**kwargs)
        self.backends: tuple[Backend, ...] = tuple(backends) if backends is not None else DEFAULT_BACKENDS

    # ``backends`` is runtime config, not identity. List it transient
    # so the singleton-key probe + pickle round-trip skip it and a
    # caller passing a different chain doesn't fragment the connection
    # pool. (Same shape :class:`ErrorNotifyingHTTPSession` uses.)
    _TRANSIENT_STATE_ATTRS = HTTPSession._TRANSIENT_STATE_ATTRS | {"backends"}

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #
    def fetch(
        self,
        pairs: Iterable[PairLike],
        start: DateLike = None,
        end: DateLike = None,
        sampling: Optional[str] = None,
        *,
        geo: bool = False,
        lazy: bool = False,
    ) -> "pl.DataFrame | pl.LazyFrame":
        """Fetch the FX timeseries for *pairs* between *start* and *end*.

        Args:
            pairs: Iterable of ``(source, target)`` couples. Each side
                accepts :class:`Currency`, an ISO 4217 alpha-3 string,
                or any alias :meth:`Currency.parse_str` accepts
                (``"$"``, ``"€"``, ``"yen"`` …).
            start: Window start, UTC. Anything :func:`yggdrasil.data
                .cast.convert` parses into a :class:`datetime.datetime`
                — ISO strings, :class:`datetime` / :class:`date`,
                epoch seconds/millis, ``"now"`` / ``"utcnow"``.
                Naive inputs are read as UTC. Defaults to one week
                before *end*.
            end: Window end, UTC. Same coercion as *start*. Defaults
                to ``datetime.now(UTC)``.
            sampling: Output ``sampling`` column. Defaults to
                :attr:`DEFAULT_SAMPLING` (``"1d"``) — the cadence
                every free public source publishes at.
            geo: When ``True``, splice in
                ``source_country_iso / lat / lon`` + target equivalents
                via :class:`yggdrasil.data.enums.geozone.GeoZoneCatalog`.
                Triggers a one-time country-catalog fetch the first
                time this argument is set in the process.
            lazy: Return a :class:`polars.LazyFrame` instead of an
                eager :class:`polars.DataFrame`.

        Returns:
            A long-format polars frame with the schema documented at
            module top, ordered by ``(from_timestamp, source, target)``.

        Raises:
            ValueError: empty *pairs* iterable, mis-shaped pair tuple,
                or *start* after *end*.
            BackendError: every configured backend failed for at least
                one ``(source, target window)`` group. The error carries
                the *last* backend's failure on ``__cause__``.
        """
        normalised_pairs = [_coerce_pair(p) for p in pairs]
        if not normalised_pairs:
            raise ValueError(
                "fetch() needs at least one (source, target) pair; got an "
                "empty iterable."
            )

        utcnow = dt.datetime.now(dt.timezone.utc)
        end_dt = _coerce_datetime(end, default=utcnow)
        start_dt = _coerce_datetime(start, default=end_dt - dt.timedelta(days=7))
        if start_dt > end_dt:
            raise ValueError(
                f"FX window start ({start_dt.isoformat()}) is after end "
                f"({end_dt.isoformat()}). Swap them or check the inputs."
            )

        bucket = sampling or self.DEFAULT_SAMPLING
        groups = _group_pairs_by_source(normalised_pairs)

        all_quotes: list[FxQuote] = []
        for src, targets in groups.items():
            LOGGER.debug(
                "Fetching FX timeseries source=%s targets=%s from=%s to=%s sampling=%s",
                src.code, [t.code for t in targets], start_dt.isoformat(),
                end_dt.isoformat(), bucket,
            )
            all_quotes.extend(
                self._fetch_group_with_fallback(
                    source=src,
                    targets=targets,
                    fetcher=lambda backend, _src=src, _tgts=targets: backend.fetch_timeseries(
                        self,
                        source=_src,
                        targets=_tgts,
                        start=start_dt,
                        end=end_dt,
                        sampling=bucket,
                    ),
                )
            )

        return self._to_frame(all_quotes, geo=geo, lazy=lazy)

    def latest(
        self,
        pairs: Iterable[PairLike],
        *,
        geo: bool = False,
        lazy: bool = False,
    ) -> "pl.DataFrame | pl.LazyFrame":
        """Single-point fetch of the most recent rate per pair.

        Same fallback semantics as :meth:`fetch` — backends are tried
        in :attr:`backends` order, the first one that returns rows
        wins, errors roll over.
        """
        normalised_pairs = [_coerce_pair(p) for p in pairs]
        if not normalised_pairs:
            raise ValueError(
                "latest() needs at least one (source, target) pair; got an "
                "empty iterable."
            )

        utcnow = dt.datetime.now(dt.timezone.utc)
        groups = _group_pairs_by_source(normalised_pairs)

        all_quotes: list[FxQuote] = []
        for src, targets in groups.items():
            LOGGER.debug(
                "Fetching FX latest source=%s targets=%s",
                src.code, [t.code for t in targets],
            )
            all_quotes.extend(
                self._fetch_group_with_fallback(
                    source=src,
                    targets=targets,
                    fetcher=lambda backend, _src=src, _tgts=targets: backend.fetch_latest(
                        self, source=_src, targets=_tgts, at=utcnow,
                    ),
                )
            )

        return self._to_frame(all_quotes, geo=geo, lazy=lazy)

    # ------------------------------------------------------------------ #
    # Backend orchestration
    # ------------------------------------------------------------------ #
    def _fetch_group_with_fallback(
        self,
        *,
        source: Currency,
        targets: Sequence[Currency],
        fetcher,
    ) -> Sequence[FxQuote]:
        """Walk :attr:`backends` until one returns rows or every one fails."""
        if not self.backends:
            raise BackendError(
                "FxRate has no configured backends. Pass `backends=` "
                "with at least one driver (see `yggdrasil.fxrate.backends`)."
            )

        last_exc: Optional[BaseException] = None
        attempts: list[str] = []
        for backend in self.backends:
            attempts.append(backend.name)
            try:
                quotes = fetcher(backend)
            except BackendError as exc:
                last_exc = exc
                LOGGER.warning(
                    "FX backend %r failed for %s->%s, falling back: %s",
                    backend.name, source.code,
                    [t.code for t in targets], exc,
                )
                continue
            except Exception as exc:
                last_exc = exc
                LOGGER.warning(
                    "FX backend %r raised %s for %s->%s, falling back: %s",
                    backend.name, type(exc).__name__, source.code,
                    [t.code for t in targets], exc,
                )
                continue

            if quotes:
                LOGGER.debug(
                    "FX backend %r returned %d rows for %s->%s",
                    backend.name, len(quotes), source.code,
                    [t.code for t in targets],
                )
                return quotes
            # An empty-but-no-error return means the backend doesn't
            # have data for this pair — try the next one rather than
            # silently dropping rows.
            LOGGER.debug(
                "FX backend %r returned 0 rows for %s->%s, falling back",
                backend.name, source.code, [t.code for t in targets],
            )

        # Every backend failed or returned nothing.
        if last_exc is not None:
            raise BackendError(
                f"All FX backends failed for {source.code}->"
                f"{[t.code for t in targets]} (tried: {attempts}). "
                f"Last error: {last_exc}"
            ) from last_exc
        # No errors, just empty data everywhere — that's a softer signal.
        LOGGER.warning(
            "All FX backends returned 0 rows for %s->%s (tried: %s).",
            source.code, [t.code for t in targets], attempts,
        )
        return ()

    # ------------------------------------------------------------------ #
    # Frame assembly + geography
    # ------------------------------------------------------------------ #
    def _to_frame(
        self,
        quotes: Sequence[FxQuote],
        *,
        geo: bool,
        lazy: bool,
    ) -> "pl.DataFrame | pl.LazyFrame":
        from yggdrasil.lazy_imports import polars as pl

        # Build the six per-column lists in a single pass — six
        # comprehensions each walking ``quotes`` was hot enough to
        # show up on the benchmark for large frames. Tuple-index
        # (``q[0]``) skips the property descriptor lookup that
        # ``q.source`` pays each time.
        n = len(quotes)
        sources: list[str] = [""] * n
        targets: list[str] = [""] * n
        from_ts: list[dt.datetime] = [None] * n  # type: ignore[list-item]
        to_ts: list[dt.datetime] = [None] * n  # type: ignore[list-item]
        samplings: list[str] = [""] * n
        values: list[float] = [0.0] * n
        for i, q in enumerate(quotes):
            sources[i] = q[0]
            targets[i] = q[1]
            from_ts[i] = q[2]
            to_ts[i] = q[3]
            samplings[i] = q[4]
            values[i] = q[5]
        data: dict[str, Any] = {
            "source": sources,
            "target": targets,
            "from_timestamp": from_ts,
            "to_timestamp": to_ts,
            "sampling": samplings,
            "value": values,
        }

        if geo:
            data.update(_geo_columns(sources, targets))

        df = pl.DataFrame(
            data,
            schema_overrides={
                "from_timestamp": pl.Datetime("us", time_zone="UTC"),
                "to_timestamp": pl.Datetime("us", time_zone="UTC"),
                "value": pl.Float64,
            },
        ).sort(["from_timestamp", "source", "target"])

        return df.lazy() if lazy else df


def _geo_columns(
    sources: Sequence[str],
    targets: Sequence[str],
) -> dict[str, list[Any]]:
    """Build the six geography columns via :class:`GeoZoneCatalog`.

    Memoises the per-currency lookup across the row set so a
    timeseries (which repeats the same pair on every date row) only
    pays one catalog hit per distinct currency.
    """
    cache: dict[str, "GeoZone | None"] = {}

    def _resolve(code: str) -> "GeoZone | None":
        if code not in cache:
            cache[code] = _zone_for_currency(code)
        return cache[code]

    s_country: list[Optional[str]] = []
    s_lat: list[Optional[float]] = []
    s_lon: list[Optional[float]] = []
    t_country: list[Optional[str]] = []
    t_lat: list[Optional[float]] = []
    t_lon: list[Optional[float]] = []
    for src, tgt in zip(sources, targets):
        s_zone = _resolve(src)
        t_zone = _resolve(tgt)
        s_country.append(s_zone.country_iso if s_zone else None)
        s_lat.append(float(s_zone.lat) if s_zone else None)
        s_lon.append(float(s_zone.lon) if s_zone else None)
        t_country.append(t_zone.country_iso if t_zone else None)
        t_lat.append(float(t_zone.lat) if t_zone else None)
        t_lon.append(float(t_zone.lon) if t_zone else None)
    return {
        "source_country_iso": s_country,
        "source_lat": s_lat,
        "source_lon": s_lon,
        "target_country_iso": t_country,
        "target_lat": t_lat,
        "target_lon": t_lon,
    }
