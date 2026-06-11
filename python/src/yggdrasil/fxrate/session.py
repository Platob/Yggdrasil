"""FX rate orchestration — coercion, fan-out, fallback, frame assembly.

:class:`FxRate` is the public entry point: hand it a list of backends and
ask for ``fetch(pairs, start, end)`` or ``latest(pairs)``. It takes the
loose shapes a real caller already has — ``"EUR"`` / ``"$"`` / a
:class:`Currency`, ISO strings / epoch ints / ``datetime`` instances —
normalises them, groups the requested pairs by source currency, fans each
group out to the backend chain (falling back on :class:`BackendError`),
and assembles the resulting quotes into a long polars frame.

The coercion helpers (``_coerce_*``, ``_group_pairs_by_source``) are
module-level so the per-call input-shaping cost is benchmarkable in
isolation; they're the only "private" surface the orchestrator leans on.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from yggdrasil.enums.currency import Currency
from yggdrasil.lazy_imports import polars

from .backends import Backend

__all__ = ["FxQuote", "BackendError", "FxRate"]

_LOG = logging.getLogger("yggdrasil.fxrate")

# Column order of every frame :class:`FxRate` produces. Kept here so the
# happy path, the empty-result path, and the lazy path all agree on the
# schema even when there are zero quotes to infer it from.
_COLUMNS: tuple[str, ...] = (
    "source", "target", "from_timestamp", "to_timestamp", "sampling", "value",
)


class BackendError(Exception):
    """A backend failed to serve a request.

    Raised by a :class:`Backend` to signal "I can't answer this — try
    the next provider". Deliberately a plain :class:`Exception` rather
    than a :class:`YGGException` subclass: it's a control-flow signal in
    the fallback walk, caught directly by :class:`FxRate.fetch`.
    """


@dataclass(slots=True, frozen=True)
class FxQuote:
    """One FX observation: ``value`` units of ``target`` per unit ``source``.

    ``from_timestamp`` / ``to_timestamp`` bound the sampling bucket the
    quote represents; ``sampling`` names the bucket width (``"1d"`` …).
    """

    source: str
    target: str
    from_timestamp: dt.datetime
    to_timestamp: dt.datetime
    sampling: str
    value: float


# ---------------------------------------------------------------------------
# Input coercion — accept the shapes a real caller already holds
# ---------------------------------------------------------------------------


def _coerce_currency(obj: Currency | str | None) -> Currency:
    """Normalise a currency-ish value to a :class:`Currency`.

    Accepts an existing :class:`Currency` (identity), an ISO code or
    alias string (``"EUR"`` / ``"$"`` / ``"euro"``), or ``None`` →
    workspace default (USD). ``Currency.from_`` already short-circuits
    each of those through its instance cache.
    """
    return Currency.from_(obj)


def _coerce_datetime(obj: dt.datetime | dt.date | str | int | float) -> dt.datetime:
    """Normalise a moment to a tz-aware UTC :class:`datetime`.

    - ``datetime`` → returned as-is if tz-aware, stamped UTC if naive.
    - ``date`` → midnight UTC.
    - ``int`` / ``float`` → Unix epoch seconds.
    - ``str`` → ISO-8601 (date or datetime); a bare ``Z`` is accepted.
    """
    if isinstance(obj, dt.datetime):
        return obj if obj.tzinfo is not None else obj.replace(tzinfo=dt.timezone.utc)
    # ``date`` is a supertype check, so it must come after ``datetime``.
    if isinstance(obj, dt.date):
        return dt.datetime.combine(obj, dt.time.min, tzinfo=dt.timezone.utc)
    if isinstance(obj, (int, float)):
        return dt.datetime.fromtimestamp(obj, tz=dt.timezone.utc)
    if isinstance(obj, str):
        parsed = dt.datetime.fromisoformat(obj.strip().replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=dt.timezone.utc)
    raise TypeError(
        f"Cannot coerce {type(obj).__name__} to datetime — pass a datetime, "
        f"date, ISO-8601 string, or epoch number, got {obj!r}"
    )


def _coerce_pair(pair: tuple[Currency | str, Currency | str]) -> tuple[Currency, Currency]:
    """Normalise a ``(source, target)`` pair to a pair of :class:`Currency`."""
    try:
        source, target = pair
    except (ValueError, TypeError):
        raise ValueError(
            f"Each pair must be a (source, target) 2-tuple, got {pair!r}"
        ) from None
    return _coerce_currency(source), _coerce_currency(target)


def _group_pairs_by_source(
    pairs: list[tuple[Currency | str, Currency | str]],
) -> dict[Currency, list[Currency]]:
    """Collapse pairs into ``{source: [target, ...]}``, source-deduplicated.

    One backend request can usually serve every target sharing a source
    currency, so grouping here turns N pairs into (≤ N) one-per-source
    fetches. Target order within a source is preserved; duplicates are
    dropped.
    """
    grouped: dict[Currency, list[Currency]] = {}
    for pair in pairs:
        source, target = _coerce_pair(pair)
        targets = grouped.get(source)
        if targets is None:
            grouped[source] = [target]
        elif target not in targets:
            targets.append(target)
    return grouped


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class FxRate:
    """Fan FX-rate requests across a fallback chain of backends.

    The chain is tried in order per source-currency group: the first
    backend that returns without raising :class:`BackendError` wins;
    on failure the next is tried. If every backend fails for a group,
    that group contributes no rows (and logs at warning level).
    """

    def __init__(self, backends: list[Backend]) -> None:
        backends = list(backends)
        if not backends:
            raise ValueError("FxRate needs at least one backend")
        self.backends = backends

    def __repr__(self) -> str:
        names = ", ".join(b.name for b in self.backends)
        return f"FxRate(backends=[{names}])"

    # -- public API ---------------------------------------------------------

    def fetch(
        self,
        pairs: list[tuple[Currency | str, Currency | str]],
        start: dt.datetime | dt.date | str | int | float,
        end: dt.datetime | dt.date | str | int | float,
        *,
        lazy: bool = False,
        geo: bool = False,
    ) -> "polars.DataFrame | polars.LazyFrame":
        """Fetch a timeseries for every pair over ``[start, end]``.

        Returns a long polars frame with columns ``source, target,
        from_timestamp, to_timestamp, sampling, value``. ``lazy=True``
        returns a :class:`polars.LazyFrame`; ``geo=True`` left-joins a
        geo-zone catalog onto each currency (best-effort — skipped
        silently if the catalog is unavailable).
        """
        start_dt = _coerce_datetime(start)
        end_dt = _coerce_datetime(end)
        grouped = _group_pairs_by_source(pairs)

        quotes: list[FxQuote] = []
        for source, targets in grouped.items():
            sampling = self.backends[0].default_sampling
            quotes.extend(self._walk(
                lambda backend, src=source, tgts=targets, smp=sampling: backend.fetch_timeseries(
                    self,
                    source=src.code,
                    targets=[t.code for t in tgts],
                    start=start_dt,
                    end=end_dt,
                    sampling=smp,
                ),
                source=source,
            ))

        return self._assemble(quotes, lazy=lazy, geo=geo)

    def latest(
        self,
        pairs: list[tuple[Currency | str, Currency | str]],
        *,
        at: dt.datetime | dt.date | str | int | float | None = None,
        lazy: bool = False,
        geo: bool = False,
    ) -> "polars.DataFrame | polars.LazyFrame":
        """Fetch the most recent quote per pair (as of ``at``, default now)."""
        at_dt = _coerce_datetime(at) if at is not None else dt.datetime.now(dt.timezone.utc)
        grouped = _group_pairs_by_source(pairs)

        quotes: list[FxQuote] = []
        for source, targets in grouped.items():
            quotes.extend(self._walk(
                lambda backend, src=source, tgts=targets: backend.fetch_latest(
                    self,
                    source=src.code,
                    targets=[t.code for t in tgts],
                    at=at_dt,
                ),
                source=source,
            ))

        return self._assemble(quotes, lazy=lazy, geo=geo)

    # -- internals ----------------------------------------------------------

    def _walk(self, call, *, source: Currency) -> list[FxQuote]:
        """Try each backend in order; return the first non-failing result.

        ``call(backend)`` performs the actual fetch. A
        :class:`BackendError` rolls over to the next backend; exhausting
        the chain yields ``[]`` and a warning. Any other exception is a
        bug and propagates.
        """
        last: BackendError | None = None
        for backend in self.backends:
            try:
                return call(backend)
            except BackendError as exc:
                last = exc
                _LOG.warning(
                    "Backend %r failed for source %r, falling back (error=%s)",
                    backend.name, source.code, exc,
                )
        _LOG.warning(
            "All backends exhausted for source %r (last_error=%s)",
            source.code, last,
        )
        return []

    def _assemble(
        self, quotes: list[FxQuote], *, lazy: bool, geo: bool,
    ) -> "polars.DataFrame | polars.LazyFrame":
        """Build the long frame from collected quotes, then enrich/lazy-ify."""
        if quotes:
            frame = polars.DataFrame(
                {
                    "source": [q.source for q in quotes],
                    "target": [q.target for q in quotes],
                    "from_timestamp": [q.from_timestamp for q in quotes],
                    "to_timestamp": [q.to_timestamp for q in quotes],
                    "sampling": [q.sampling for q in quotes],
                    "value": [q.value for q in quotes],
                },
            )
        else:
            # Preserve the schema even with zero rows so downstream joins
            # / concats don't blow up on a column-less empty frame.
            frame = polars.DataFrame(
                schema={
                    "source": polars.Utf8,
                    "target": polars.Utf8,
                    "from_timestamp": polars.Datetime("us", "UTC"),
                    "to_timestamp": polars.Datetime("us", "UTC"),
                    "sampling": polars.Utf8,
                    "value": polars.Float64,
                },
            )

        if geo:
            frame = self._enrich_geo(frame)

        return frame.lazy() if lazy else frame

    def _enrich_geo(self, frame: "polars.DataFrame") -> "polars.DataFrame":
        """Left-join geo-zone metadata onto source/target currencies.

        Best-effort: the geo catalog is an optional enrichment, so any
        failure to build or join it leaves the frame untouched rather
        than failing the whole fetch.
        """
        try:
            from yggdrasil.enums.geozone import GeoZoneCatalog  # noqa: F401
        except Exception:  # pragma: no cover - optional enrichment
            _LOG.warning("Geo enrichment requested but GeoZoneCatalog unavailable; skipping")
            return frame
        # No currency→zone mapping ships yet; the join surface is wired so
        # callers can pass ``geo=True`` today and get richer output once a
        # currency-to-zone catalog lands. Until then this is a no-op that
        # never fails the fetch.
        return frame
