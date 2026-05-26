"""Unified temporal module — types + framework-native cast helpers.

Nuclear-simplified version. Design rules:

* **Framework casts only.** Each engine (Arrow, Polars, Spark, pandas) gets
  one cast dispatcher that delegates straight to ``pc.cast`` / ``series.cast``
  / ``column.cast``. No multi-format coalesce, no ISO-duration regex, no
  per-row Python fallback, no fractional-second re-attachment.

* **ISO-8601 only for strings.** Strings parse via each framework's native
  ISO strptime (``pc.strptime`` with one ISO format, ``pl.Series.str.to_datetime``
  default, Spark ``to_timestamp`` default). Any non-ISO shape (``dd/MM/yyyy``,
  ``PT15M``, ``HH:MM:SS`` clock durations, etc.) is caller's problem — pre-parse
  before handing the array here.

* **Wall-clock reinterpret for naive→aware.** One rule, always: a naive
  timestamp cast to a tz-aware target keeps its wall-clock digits and stamps
  on the target zone. No ``unsafe_tz`` flag, no "assume UTC" mode, no DST
  null-on-nonexistent threading — we lean on whatever the framework does by
  default and accept the consequences.
"""

from __future__ import annotations

import datetime as dt
import decimal
import re
from abc import ABC
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Union

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.enums.timeunit import TimeUnit
from yggdrasil.enums.timezone import Timezone
from yggdrasil.enums import Mode

from .base import PrimitiveType
from ..id import DataTypeId
from yggdrasil.lazy_imports import polars_module, spark_sql_module

if TYPE_CHECKING:
    import polars  # noqa: F401
    import polars as pl  # noqa: F401
    import pyspark.sql as ps  # noqa: F401
    import pyspark.sql.types as pst  # noqa: F401
    from ..cast.options import CastOptions  # noqa: F401

__all__ = [
    # type classes
    "TemporalType",
    "DateType",
    "TimeType",
    "TimestampType",
    "DurationType",
    # engine dispatchers
    "arrow_cast",
    "spark_cast",
    "cast_polars_array_to_temporal",
]


# Polars only supports ms/us/ns units natively for Datetime/Duration —
# the membership test runs in tight cast-fast-path code. Members are
# :class:`TimeUnit` instances and subclass ``str``, so the set works
# for both ``TimeUnit.MICROSECOND`` and the bare ``"us"`` token.
_POLARS_UNITS = frozenset({
    TimeUnit.MILLISECOND,
    TimeUnit.MICROSECOND,
    TimeUnit.NANOSECOND,
})

# ---------------------------------------------------------------------------
# Minimal string format catalogues (ISO + Excel/CSV)
# ---------------------------------------------------------------------------
#
# Philosophy: one catalogue, four shapes, coalesced in priority order.
# Anything not in this list is caller's responsibility to pre-format.
#
# Order matters — ``pl.coalesce`` takes the first non-null. Day-first before
# month-first means ``01/02/2024`` parses as 1 Feb (Excel non-US default)
# rather than 2 Jan. ISO always wins when it matches because it's tried first.
#
# Polars uses chrono format strings. ``%.f`` = optional fractional seconds,
# ``%#z`` = flexible UTC offset (``+02``, ``+0200``, or ``+02:00``).

_POLARS_DATETIME_FORMATS: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S%.f%#z",       # ISO aware
    "%Y-%m-%dT%H:%M:%S%.f",           # ISO naive
    "%Y-%m-%d %H:%M:%S%.f",           # space-separated ISO
    "%d/%m/%Y %H:%M:%S",              # Excel EU day-first
    "%m/%d/%Y %H:%M:%S",              # Excel US month-first
    "%Y/%m/%d %H:%M:%S",              # Excel year-first
    "%d/%m/%Y",                        # bare CSV day-first
    "%m/%d/%Y",                        # bare CSV month-first
    "%Y/%m/%d",                        # bare CSV year-first
)

_POLARS_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",                        # ISO
    "%d/%m/%Y",                        # day-first (Excel EU)
    "%m/%d/%Y",                        # month-first (Excel US)
    "%Y/%m/%d",                        # year-first
)

_POLARS_TIME_FORMATS: tuple[str, ...] = (
    "%H:%M:%S%.f",                     # 24h with optional fractional
    "%H:%M:%S",                        # 24h plain
    "%H:%M",                           # hour:minute only
    "%I:%M:%S %p",                     # 12h (AM/PM)
)

# Calendar units in ISO-8601 durations (Y, M, W) have no fixed second count.
# We collapse them to whole-day defaults — the convention Postgres ``interval``,
# Java ``Duration.parse``, and ``isodate`` all converge on.
ISO_DURATION_DAYS_PER_YEAR = 365
ISO_DURATION_DAYS_PER_MONTH = 30
ISO_DURATION_DAYS_PER_WEEK = 7

_ISO_DURATION_REGEX = (
    r"^(?P<sign>[+-])?P"
    r"(?:(?P<years>\d+(?:[.,]\d+)?)Y)?"
    r"(?:(?P<months>\d+(?:[.,]\d+)?)M)?"
    r"(?:(?P<weeks>\d+(?:[.,]\d+)?)W)?"
    r"(?:(?P<days>\d+(?:[.,]\d+)?)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+(?:[.,]\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:[.,]\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:[.,]\d+)?)S)?"
    r")?$"
)
_RE_ISO_DURATION = re.compile(_ISO_DURATION_REGEX, re.IGNORECASE)

# ``HH:MM:SS[.ffffff]`` and ``HH:MM`` clock-style duration with optional sign.
_RE_CLOCK_DURATION = re.compile(
    r"^(?P<sign>[+-])?(?P<hours>\d+):(?P<minutes>\d{1,2})(?::(?P<seconds>\d{1,2}(?:\.\d+)?))?$"
)

UTC_ALIAS_TIMEZONES = frozenset({"Z", "Etc/UTC", "+00:00"})


def _parse_iso_duration(text: str) -> "dt.timedelta | None":
    """Scalar ``P[n]Y[n]M[n]W[n]D[T[n]H[n]M[n]S]`` → ``timedelta``.

    Calendar units (Y/M/W) collapse via the module-level day defaults.
    Returns ``None`` for non-ISO shapes so callers can chain with other
    parsers.
    """
    m = _RE_ISO_DURATION.fullmatch(text.strip())
    if m is None:
        return None
    parts = m.groupdict()
    if not any(parts[k] for k in ("years", "months", "weeks", "days", "hours", "minutes", "seconds")):
        return None
    sign = -1 if parts["sign"] == "-" else 1

    def _f(name: str) -> float:
        raw = parts[name]
        return 0.0 if raw is None else float(raw.replace(",", "."))

    total_seconds = (
        _f("years") * ISO_DURATION_DAYS_PER_YEAR * 86400.0
        + _f("months") * ISO_DURATION_DAYS_PER_MONTH * 86400.0
        + _f("weeks") * ISO_DURATION_DAYS_PER_WEEK * 86400.0
        + _f("days") * 86400.0
        + _f("hours") * 3600.0
        + _f("minutes") * 60.0
        + _f("seconds")
    )
    return dt.timedelta(seconds=sign * total_seconds)


def _parse_clock_duration(text: str) -> "dt.timedelta | None":
    """Parse ``[+-]HH:MM[:SS[.fff]]`` into a ``timedelta``.

    Returns ``None`` if the input doesn't match so callers can chain.
    """
    m = _RE_CLOCK_DURATION.fullmatch(text.strip())
    if m is None:
        return None
    sign = -1 if m.group("sign") == "-" else 1
    hours = int(m.group("hours"))
    minutes = int(m.group("minutes"))
    seconds_raw = m.group("seconds")
    seconds = float(seconds_raw) if seconds_raw is not None else 0.0
    total = hours * 3600.0 + minutes * 60.0 + seconds
    return dt.timedelta(seconds=sign * total)


def _coerce_str(value: Any) -> str | None:
    """Return *value* as ``str`` if it's string-like, else None.

    Covers ``str``, ``bytes``, and ``bytearray`` — enough to pass through
    CSV readers, serde round-trips, and loose Python dicts without
    surprise nulls. ``None`` falls through as ``None``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin-1")
    return None


# ---------------------------------------------------------------------------
# Arrow cast dispatcher — routed through polars
# ---------------------------------------------------------------------------


def _arrow_target_fits_polars(target: "pa.DataType") -> bool:
    """True when polars can natively represent the Arrow target dtype.

    Polars' Datetime / Duration top out at ms/us/ns — second-precision
    Timestamp / Duration can't round-trip. Everything else (date / time /
    tz-aware timestamp / non-temporal) is fine.
    """
    if pa.types.is_timestamp(target) or pa.types.is_duration(target):
        return target.unit in _POLARS_UNITS
    return True


def _arrow_to_polars_target(target: "pa.DataType") -> Any:
    """Map an Arrow dtype to the polars dtype instance ``cast_polars_array_to_temporal`` expects."""
    pl = polars_module()
    if pa.types.is_timestamp(target):
        return pl.Datetime(time_unit=target.unit, time_zone=target.tz)
    if pa.types.is_date(target):
        return pl.Date()
    if pa.types.is_time(target):
        return pl.Time()
    if pa.types.is_duration(target):
        return pl.Duration(time_unit=target.unit)
    # Non-temporal — not our concern here; callers check ``_is_temporal_target``.
    raise TypeError(f"Non-temporal Arrow target: {target!r}")


def _is_temporal_target(target: "pa.DataType") -> bool:
    return (
        pa.types.is_timestamp(target)
        or pa.types.is_date(target)
        or pa.types.is_time(target)
        or pa.types.is_duration(target)
    )


def _pc_cast_equivalent_to_polars(
    src_type: "pa.DataType", target: "pa.DataType"
) -> bool:
    """True when ``pc.cast(src → target)`` produces the same result as the
    polars route — letting us skip the Arrow → polars → Arrow round-trip.

    The polars detour exists for exactly two semantic features:

    * **ISO-8601 string parsing** via polars' chrono parser (and its
      coalesce-of-formats fallback for Excel / CSV shapes).
    * **Wall-clock tz reinterpret** for naive ↔ aware transitions and
      cross-tz conversion (``replace_time_zone`` / ``convert_time_zone``).

    Pyarrow's ``pc.cast`` does not reproduce either: it rejects naive →
    aware outright (needs ``assume_timezone``), and aware → naive goes
    through UTC rather than keeping the wall clock. So we only fast-
    path same-family casts where neither feature applies — pure unit
    conversion with matching tz, or tz-free families (date / time /
    duration). Cross-tz, naive↔aware, and string sources stay on the
    polars route.
    """
    # Same-tz timestamp: pure unit conversion. Different tz (including
    # naive↔aware) needs polars' wall-clock semantics.
    if pa.types.is_timestamp(src_type) and pa.types.is_timestamp(target):
        return src_type.tz == target.tz
    # Tz-free families — pyarrow handles unit conversion natively with
    # the same semantics polars would produce.
    if pa.types.is_duration(src_type) and pa.types.is_duration(target):
        return True
    if pa.types.is_date(src_type) and pa.types.is_date(target):
        return True
    if pa.types.is_time(src_type) and pa.types.is_time(target):
        return True
    return False


def arrow_cast(
    array: "pa.Array | pa.ChunkedArray",
    target: "pa.DataType",
    *,
    safe: bool = False,
) -> "pa.Array | pa.ChunkedArray":
    """Cast an Arrow array to *target* by routing through polars.

    Arrow array → polars Series → ``cast_polars_array_to_temporal`` →
    Arrow array. Polars' native chrono parser handles ISO-8601 strings,
    unit conversion, and wall-clock tz reinterpret in one pass.

    Falls back to ``pc.cast`` when:

    * the target is non-temporal,
    * polars can't represent the target (second-precision
      Datetime / Duration), or
    * the cast is a same-family unit conversion where pyarrow's cast
      semantics already match polars (no tz reinterpret, no string
      parsing) — see :func:`_pc_cast_equivalent_to_polars`.
    """
    # Non-temporal or polars-incompatible target — direct Arrow cast.
    if not _is_temporal_target(target) or not _arrow_target_fits_polars(target):
        return pc.cast(array, target, safe=safe)

    # Fast path: same-family cast where pyarrow's native ``pc.cast``
    # produces the same result as the polars round-trip. Skip building a
    # polars Series + going through ``cast_polars_array_to_temporal`` +
    # the back-conversion — saves the lion's share of cost on the
    # ``timestamp(us)→timestamp(ms)`` / duration / date / time hot paths.
    if _pc_cast_equivalent_to_polars(array.type, target):
        return pc.cast(array, target, safe=safe)

    pl = polars_module()
    pl_target = _arrow_to_polars_target(target)

    # ChunkedArray needs per-chunk handling because ``pl.from_arrow`` on a
    # ChunkedArray returns a Series, but the output dtype of each chunk
    # must match — simpler to map per chunk and rebuild.
    def _one(chunk: pa.Array) -> pa.Array:
        series = pl.from_arrow(chunk)
        if not isinstance(series, pl.Series):
            series = pl.Series(values=series)
        casted = cast_polars_array_to_temporal(
            series,
            source=series.dtype,
            target=pl_target,
            safe=safe,
            source_tz=series.dtype.time_zone
            if isinstance(series.dtype, pl.Datetime)
            else None,
            target_tz=target.tz if pa.types.is_timestamp(target) else None,
        )
        out = casted.to_arrow()
        # ── Double cast ────────────────────────────────────────────────
        # Polars' dtype system is coarser than Arrow's — ``pl.Date`` maps
        # to ``date32[day]`` only, never ``date64[ms]``; ``pl.Time`` is
        # always ``time64[ns]``. Arrow discriminates date32 vs date64,
        # time32[s/ms] vs time64[us/ns]. Pin the exact target so the
        # caller gets the type they asked for rather than polars'
        # canonical shape.
        #
        # ``safe=False`` here because ``time64[ns] → time64[us]`` is a
        # unit downcast that pyarrow considers lossy by default. The
        # semantic parse already happened in polars; this final hop
        # only adjusts storage layout.
        if out.type != target:
            out = pc.cast(out, target, safe=False)
        return out

    if isinstance(array, pa.ChunkedArray):
        chunks = [_one(chunk) for chunk in array.chunks]
        if not chunks:
            placeholder = _one(pa.array([], type=array.type))
            return pa.chunked_array([], type=placeholder.type)
        return pa.chunked_array(chunks, type=chunks[0].type)

    return _one(array)


# ---------------------------------------------------------------------------
# Polars cast dispatcher
# ---------------------------------------------------------------------------


def cast_polars_array_to_temporal(
    array: Union["pl.Series", "pl.Expr"],
    source: Any,
    target: Any,
    safe: bool,
    source_tz: str | None = None,
    target_tz: str | None = None,
    to_expr: bool = False,
    parent_name: str | None = None,
) -> Union["pl.Series", "pl.Expr"]:
    """Cast a polars Series / Expr to a temporal dtype via the native cast.

    Strings parse through ``str.to_datetime`` / ``str.to_date`` / ``str.to_time``
    with default ISO format. Naive→aware uses ``replace_time_zone``
    (wall-clock reinterpret). Everything else is ``.cast(target)``.
    """
    pl = polars_module()
    is_expr = isinstance(array, pl.Expr)
    series_name = "" if is_expr else (array.name or "")

    if parent_name and series_name:
        col_name = f"{parent_name}.{series_name}"
    else:
        col_name = series_name or parent_name or "__col__"

    working: Any = array if is_expr else pl.col(col_name)

    src_cls = source if isinstance(source, type) else source.__class__
    tgt_cls = target if isinstance(target, type) else target.__class__

    src_tz = (
        getattr(source, "time_zone", None) if not isinstance(source, type) else None
    ) or source_tz
    tgt_tz = (
        getattr(target, "time_zone", None) if not isinstance(target, type) else None
    ) or target_tz

    string_classes = (pl.String, pl.Utf8)

    def _coalesce_strptime(expr: Any, pl_dtype: Any, formats: tuple[str, ...]) -> Any:
        """Try each format in order, coalesce to the first match.

        ``strict=False`` is baked in — individual rows that don't match a
        format yield null; rows that match any format win via coalesce.
        Using explicit formats instead of polars' ``infer`` avoids the
        ``ComputeError: could not find an appropriate format`` that fires
        when the input contains zero parseable values (e.g. all "garbage").
        """
        if not formats:
            return expr.cast(pl_dtype, strict=safe)
        branches = [
            expr.str.strptime(pl_dtype, fmt, strict=False)
            for fmt in formats
        ]
        if len(branches) == 1:
            return branches[0]
        return pl.coalesce(branches)

    # --- Datetime target ------------------------------------------------
    if tgt_cls is pl.Datetime:
        tu = getattr(target, "time_unit", "us") or "us"
        if src_cls in string_classes:
            # Build tz-aware and naive branches separately so polars' coalesce
            # doesn't trip on dtype mismatch between offset-bearing and bare rows.
            #
            # Offset-bearing rows parse as ``Datetime[tu, UTC]`` (polars normalises
            # to UTC), naive rows as ``Datetime[tu]``. We homogenise:
            # - naive target: strip tz on the aware branch.
            # - aware target: reinterpret wall-clock on the naive branch.
            tz_fmts = tuple(f for f in _POLARS_DATETIME_FORMATS if "%#z" in f or "%z" in f)
            naive_fmts = tuple(f for f in _POLARS_DATETIME_FORMATS if f not in tz_fmts)

            tz_branch = _coalesce_strptime(working, pl.Datetime(tu, "UTC"), tz_fmts)
            naive_branch = _coalesce_strptime(working, pl.Datetime(tu), naive_fmts)

            if tgt_tz is None:
                working = pl.coalesce([tz_branch.dt.replace_time_zone(None), naive_branch])
            else:
                working = pl.coalesce([
                    tz_branch.dt.convert_time_zone(tgt_tz),
                    naive_branch.dt.replace_time_zone(tgt_tz),
                ])
        elif src_cls is pl.Datetime:
            # Unit change + tz rewrite.
            working = working.cast(pl.Datetime(tu, src_tz), strict=safe)
            if src_tz is None and tgt_tz is not None:
                working = working.dt.replace_time_zone(tgt_tz)
            elif src_tz is not None and tgt_tz is None:
                working = working.dt.replace_time_zone(None)
            elif src_tz != tgt_tz:
                working = working.dt.convert_time_zone(tgt_tz)
        else:
            working = working.cast(pl.Datetime(tu, tgt_tz), strict=safe)

    # --- Date target ----------------------------------------------------
    elif tgt_cls is pl.Date:
        if src_cls in string_classes:
            working = _coalesce_strptime(working, pl.Date, _POLARS_DATE_FORMATS)
        else:
            working = working.cast(pl.Date, strict=safe)

    # --- Time target ----------------------------------------------------
    elif tgt_cls is pl.Time:
        if src_cls in string_classes:
            working = _coalesce_strptime(working, pl.Time, _POLARS_TIME_FORMATS)
        else:
            working = working.cast(pl.Time, strict=safe)

    # --- Duration target ------------------------------------------------
    elif tgt_cls is pl.Duration:
        tu = getattr(target, "time_unit", "us") or "us"
        working = working.cast(pl.Duration(tu), strict=safe)

    else:
        raise TypeError(f"Unsupported temporal target type: {target!r}")

    if series_name:
        working = working.alias(series_name)

    if is_expr or to_expr:
        return working

    # Materialise expression against source series.
    out = array.to_frame(name=col_name).select(working).to_series()
    return out.rename(series_name) if series_name else out


# ---------------------------------------------------------------------------
# Spark cast dispatcher
# ---------------------------------------------------------------------------


def spark_cast(
    column: "ps.Column",
    target: Any,
    *,
    safe: bool = False,
    unit: str = "us",
    tz: str | None = None,
) -> "ps.Column":
    """Cast a Spark column to *target* via native ``column.cast``.

    Strings cast straight — Spark's ``cast(TimestampType())`` accepts ISO
    inputs natively, and rejects everything else (returns null in
    best-effort mode, raises in strict).
    """
    # unit/tz aren't really expressible in Spark's type system without custom
    # metadata; accept them as signatures for API parity but don't re-stamp.
    del unit, tz, safe
    return column.cast(target)


# ===========================================================================
# TemporalType base and concrete subclasses
# ===========================================================================


@dataclass(frozen=True, repr=False)
class TemporalType(PrimitiveType, ABC):
    """Base class for Date / Time / Timestamp / Duration.

    Holds shared fields (``unit`` / ``tz``) and cross-engine dispatch logic.
    Subclasses implement ``type_id``, the ``handles_*`` / ``from_*`` class
    methods, and ``to_arrow`` / ``to_polars`` / ``to_spark`` / ``to_spark_name``.
    """

    unit: TimeUnit = TimeUnit.MICROSECOND
    tz: str | None = None

    def __post_init__(self):
        # Funnel ``unit`` through :class:`TimeUnit` so any spelling —
        # short token, plural, long form, ``TimeUnit`` member — collapses
        # to a canonical enum instance. Subclasses with extra
        # __post_init__ logic call ``super().__post_init__()`` first.
        if self.unit is not None and not isinstance(self.unit, TimeUnit):
            normalized = TimeUnit.from_(self.unit, default=None)
            if normalized is not None:
                object.__setattr__(self, "unit", normalized)
        if self.tz and self.tz in UTC_ALIAS_TIMEZONES:
            object.__setattr__(self, "tz", "UTC")

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "DateType":
        try:
            return cls(
                byte_size=value.get("byte_size", 8),
                unit=value.get("unit", "d"),
                tz=value.get("tz"),
            )
        except Exception as e:
            if default is ...:
                raise ValueError(
                    f"Cannot construct {cls.__name__} from {value!r}"
                ) from e
            return default

    # ------------------------------------------------------------------
    # Cast helpers — one-liners over the engine dispatchers.
    # ------------------------------------------------------------------

    def _cast_arrow_array(self, array: "pa.Array", options: "CastOptions") -> "pa.Array":
        if not options.need_cast(array, self):
            return array
        casted = arrow_cast(array, self.to_arrow(), safe=options.safe)
        return options.fill_arrow_nulls(casted)

    def _cast_spark_column(self, column: Any, options: "CastOptions"):
        if not options.need_cast(column, self):
            return column
        casted = spark_cast(
            column,
            self.to_spark(),
            safe=options.safe,
            unit=self.unit,
            tz=str(self.tz) if self.tz else None,
        )
        return self.fill_spark_column_nulls(
            casted, nullable=self._target_nullable(options)
        )

    def _cast_pandas_series(self, series: Any, options: "CastOptions") -> Any:
        """Pandas path rides on Arrow.

        MATCH bypass: skip the ``pa.Array.from_pandas → cast → to_pandas``
        round-trip when the Series's dtype already corresponds to the
        target Arrow type. Without this, even a no-op cast pays the full
        Arrow conversion cost on both sides — measurable on tabular
        ingest where the same Schema is enforced batch after batch.
        """
        if not options.need_cast(series, self):
            return series
        arrow = pa.Array.from_pandas(series)
        casted = self._cast_arrow_array(arrow, options)
        return casted.to_pandas()

    # ------------------------------------------------------------------
    # Polars wiring
    # ------------------------------------------------------------------

    def _polars_dtype_instance(self):
        """``to_polars()`` normalized to an instance, widening unsupported units."""
        pl = polars_module()

        target = self
        if self.unit in {"s", "d"} and self.type_id in {
            DataTypeId.TIMESTAMP,
            DataTypeId.DURATION,
        }:
            target = replace(self, unit="ms")

        dtype = target.to_polars()
        if isinstance(dtype, type) and issubclass(dtype, pl.DataType):
            return dtype()
        return dtype

    def _needs_arrow_bridge(self) -> bool:
        """True when polars can't store the requested unit natively."""
        if self.type_id in (DataTypeId.TIMESTAMP, DataTypeId.DURATION):
            return self.unit == "s"
        return False

    def _polars_from_arrow(self, series: "polars.Series", options: "CastOptions"):
        pl = polars_module()
        arrow = series.to_arrow()
        casted = self._cast_arrow_array(arrow, options)
        return pl.Series(name=series.name, values=casted)

    def _cast_polars_series(self, series: "polars.Series", options: "CastOptions"):
        if not options.need_cast(series, self):
            return series
        if self._needs_arrow_bridge():
            return self._polars_from_arrow(series, options)

        pl = polars_module()
        source_dtype = series.dtype
        casted = cast_polars_array_to_temporal(
            series,
            source=source_dtype,
            target=self._polars_dtype_instance(),
            safe=options.safe,
            source_tz=source_dtype.time_zone
            if isinstance(source_dtype, pl.Datetime)
            else None,
            target_tz=str(self.tz) if self.tz else None,
        )
        return self.fill_polars_array_nulls(
            casted, nullable=self._target_nullable(options)
        )

    def _cast_polars_expr(self, expr: Any, options: "CastOptions"):
        if not options.need_cast(expr, self):
            return expr

        pl = polars_module()

        if self._needs_arrow_bridge():
            return expr.cast(self._polars_dtype_instance(), strict=options.safe)

        source_field = options.source
        source_dtype = (
            source_field.dtype.to_polars() if source_field is not None else pl.String
        )
        if isinstance(source_dtype, type) and issubclass(source_dtype, pl.DataType):
            source_dtype = source_dtype()

        casted = cast_polars_array_to_temporal(
            expr,
            source=source_dtype,
            target=self._polars_dtype_instance(),
            safe=options.safe,
            source_tz=getattr(source_dtype, "time_zone", None)
            if isinstance(source_dtype, pl.Datetime)
            else None,
            target_tz=str(self.tz) if self.tz else None,
            to_expr=True,
        )
        return self.fill_polars_array_nulls(
            casted, nullable=self._target_nullable(options)
        )

    # ------------------------------------------------------------------
    # Scalar parsing — ISO-8601 only.
    # ------------------------------------------------------------------

    def _parse_iso_scalar_string(self, value: str, kind: str, safe: bool):
        """Parse an ISO-8601 string to a native Python temporal object.

        ``kind`` is the user-visible name — ``"date" | "time" | "timestamp" | "duration"``.
        Returns ``None`` on failure (best-effort) or raises (safe). The
        *kind* string drives both error messages and dispatch.

        For ``"timestamp"`` we also try falling back to ``date.fromisoformat``
        when the full-timestamp parse fails — a bare ``"2024-01-15"`` should
        widen to midnight rather than null.

        For ``"date"`` we accept timestamp-shaped inputs like
        ``"2024-01-15T10:30:45"`` and truncate: the time component is
        discarded.
        """
        stripped = value.strip()
        if not stripped:
            if safe:
                raise ValueError(
                    f"Cannot parse {kind} from empty string for {type(self).__name__}."
                )
            return None
        try:
            if kind == "date":
                # Bare ISO date first.
                try:
                    return dt.date.fromisoformat(stripped)
                except ValueError:
                    pass
                # Fall back to timestamp shape — split off the time component.
                ts_str = stripped[:-1] + "+00:00" if stripped.endswith("Z") else stripped
                return dt.datetime.fromisoformat(ts_str).date()
            if kind == "time":
                return dt.time.fromisoformat(stripped)
            # timestamp: handle trailing Z since fromisoformat pre-3.11 doesn't.
            if stripped.endswith("Z"):
                stripped = stripped[:-1] + "+00:00"
            return dt.datetime.fromisoformat(stripped)
        except ValueError:
            if safe:
                raise ValueError(
                    f"Cannot parse {kind} from {value!r} for {type(self).__name__}."
                )
            return None

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: "Mode" = Mode.AUTO,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "TemporalType":
        base = super()._merge_with_same_id(other, mode, downcast, upcast)

        # Use :attr:`TimeUnit.order` directly — the rank table moved to
        # the enum so every temporal type agrees on the precedence.
        left = TimeUnit.from_(self.unit, default=TimeUnit.NANOSECOND)
        right = TimeUnit.from_(other.unit, default=TimeUnit.NANOSECOND)
        left_rank = left.order
        right_rank = right.order

        if downcast:
            unit = self.unit if left_rank <= right_rank else other.unit
            # Naive on either side collapses to naive — the conservative
            # downcast keeps the wall-clock rather than picking a side.
            tz = self.tz if self.tz == other.tz else None
        else:
            unit = self.unit if left_rank >= right_rank else other.unit
            # ``__bool__`` on Timezone is True for non-NAIVE — pick the
            # aware side when one of the two is naive.
            tz = self.tz if self.tz == other.tz else (self.tz or other.tz)

        return self.__class__(byte_size=base.byte_size, unit=unit, tz=tz)

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        # ``self.unit`` is a :class:`TimeUnit` (str subclass) and
        # ``self.tz`` is ``str`` on most temporal types but
        # :class:`Timezone` on :class:`TimestampType`. ``str(...)``
        # collapses both to the canonical token for serialization.
        unit_value = str(self.unit) if self.unit else None
        tz_value = str(self.tz) if self.tz else None
        return {**base, "unit": unit_value, "tz": tz_value}

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        if self.unit:
            tags[b"unit"] = str(self.unit).encode("utf-8")
        if self.tz:
            tags[b"tz"] = str(self.tz).encode("utf-8")
        return tags


# ======================================================================
# DateType
# ======================================================================


@dataclass(frozen=True, repr=False)
class DateType(TemporalType):
    unit: str = "d"
    tz: str | None = None

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "byte_size", 4)

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.DATE

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        return f"{pad}date[{self.unit}{f', tz={self.tz}' if self.tz else ''}]"

    @classmethod
    def handles_arrow_type(cls, dtype: "pa.DataType") -> bool:
        return pa.types.is_date(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: "pa.DataType") -> "DateType":
        if pa.types.is_date32(dtype):
            return cls(byte_size=4, unit="d")
        if pa.types.is_date64(dtype):
            return cls(byte_size=8, unit="ms")
        raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = polars_module()
        return dtype == pl.Date

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "DateType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        return cls(byte_size=4, unit="d")

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = spark_sql_module()
        return isinstance(dtype, spark.types.DateType)

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "DateType":
        if not cls.handles_spark_type(dtype):
            raise TypeError(f"Unsupported Spark data type: {dtype!r}")
        return cls(byte_size=4, unit="d")

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.DATE)

    def _default_pyhint(self) -> Any:
        import datetime as _dt
        return _dt.date

    def to_arrow(self) -> "pa.DataType":
        return pa.date64() if self.unit == "ms" else pa.date32()

    def to_polars(self) -> "polars.DataType":
        pl = polars_module()
        return pl.Date

    def to_spark(self) -> Any:
        spark = spark_sql_module()
        return spark.types.DateType()

    def to_spark_name(self) -> str:
        return "DATE"

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else dt.date(1970, 1, 1)

    def _convert_pyobj(self, value: Any, safe: bool = False) -> "dt.date | None":
        if value is None:
            return None
        if isinstance(value, dt.datetime):
            return value.date()
        if isinstance(value, dt.date):
            return value
        text = _coerce_str(value)
        if text is not None:
            return self._parse_iso_scalar_string(text, "date", safe)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return dt.date(1970, 1, 1) + dt.timedelta(days=int(value))
            except (OverflowError, ValueError):
                if safe:
                    raise ValueError(
                        f"Cannot convert {value!r} to date for {type(self).__name__}."
                    )
                return None
        if safe:
            raise ValueError(
                f"Cannot convert {type(value).__name__} to date "
                f"for {type(self).__name__}: {value!r}."
            )
        return None


# ======================================================================
# TimeType
# ======================================================================


@dataclass(frozen=True, repr=False)
class TimeType(TemporalType):
    unit: str = "us"
    tz: str | None = None

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "byte_size", 4 if self.unit in {"s", "ms"} else 8)

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.TIME

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        return f"{pad}time[{self.unit}{f', tz={self.tz}' if self.tz else ''}]"

    @classmethod
    def handles_arrow_type(cls, dtype: "pa.DataType") -> bool:
        return pa.types.is_time(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: "pa.DataType") -> "TimeType":
        if not pa.types.is_time(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")
        return cls(byte_size=dtype.bit_width // 8, unit=dtype.unit)

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = polars_module()
        return dtype == pl.Time

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "TimeType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        return cls(byte_size=8, unit="ns")

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        return False

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "TimeType":
        raise TypeError(f"Spark has no native time-only type: {dtype!r}")

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.TIME)

    def _default_pyhint(self) -> Any:
        import datetime as _dt
        return _dt.time

    def to_arrow(self) -> "pa.DataType":
        if self.unit in {"s", "ms"}:
            return pa.time32(self.unit)
        return pa.time64(self.unit)

    def to_polars(self) -> "polars.DataType":
        pl = polars_module()
        return pl.Time

    def to_spark(self) -> Any:
        spark = spark_sql_module()
        return spark.types.StringType()

    def as_spark(self):
        # Spark has no native ``TimeType`` — widen to ``StringType`` so
        # ``self.to_spark()`` round-trips through ISO-8601 text.
        from .string import StringType

        return StringType()

    def to_spark_name(self) -> str:
        return "STRING"

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else dt.time(0, 0, 0)

    def _convert_pyobj(self, value: Any, safe: bool = False) -> "dt.time | None":
        if value is None:
            return None
        if isinstance(value, dt.datetime):
            return value.time()
        if isinstance(value, dt.time):
            return value
        text = _coerce_str(value)
        if text is not None:
            return self._parse_iso_scalar_string(text, "time", safe)
        if safe:
            raise ValueError(
                f"Cannot convert {type(value).__name__} to time "
                f"for {type(self).__name__}: {value!r}."
            )
        return None


# ======================================================================
# TimestampType
# ======================================================================


@dataclass(frozen=True, repr=False)
class TimestampType(TemporalType):
    unit: TimeUnit = TimeUnit.MICROSECOND
    # Always a :class:`Timezone` instance — naive timestamps use the
    # :attr:`Timezone.NAIVE` sentinel rather than ``None`` so the field
    # type stays non-optional and call sites can ``if self.tz:`` to
    # mean "is timezone-aware". Constructor accepts plain IANA strings
    # / ``ZoneInfo`` / ``datetime.tzinfo`` / ``None`` and routes them
    # through :meth:`Timezone.from_` in ``__post_init__``.
    tz: Timezone = Timezone.NAIVE

    def __post_init__(self):
        if self.unit is not None and not isinstance(self.unit, TimeUnit):
            normalized = TimeUnit.from_(self.unit, default=None)
            if normalized is not None:
                object.__setattr__(self, "unit", normalized)
        if self.tz is None:
            object.__setattr__(self, "tz", Timezone.NAIVE)
        elif not isinstance(self.tz, Timezone):
            object.__setattr__(self, "tz", Timezone.from_(self.tz))
        object.__setattr__(self, "byte_size", 8)

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.TIMESTAMP

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        return f"{pad}timestamp[{self.unit}{f', tz={self.tz}' if self.tz else ''}]"

    @classmethod
    def handles_arrow_type(cls, dtype: "pa.DataType") -> bool:
        return pa.types.is_timestamp(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: "pa.TimestampType") -> "TimestampType":
        if not pa.types.is_timestamp(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

        return cls(byte_size=8, unit=dtype.unit, tz=dtype.tz or None)

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = polars_module()
        return isinstance(dtype, pl.Datetime)

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "TimestampType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        unit = getattr(dtype, "time_unit", "us") or "us"
        tz = getattr(dtype, "time_zone", None)
        return cls(byte_size=8, unit=unit, tz=tz)

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = spark_sql_module()
        return isinstance(
            dtype, (spark.types.TimestampType, spark.types.TimestampNTZType)
        )

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "TimestampType":
        spark = spark_sql_module()
        if isinstance(dtype, spark.types.TimestampType):
            return cls(byte_size=8, unit="us", tz="UTC")
        if isinstance(dtype, spark.types.TimestampNTZType):
            return cls(byte_size=8, unit="us", tz=None)
        raise TypeError(f"Unsupported Spark data type: {dtype!r}")

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.TIMESTAMP)

    @property
    def tz_iana(self) -> str | None:
        """IANA token for :attr:`tz`, or ``None`` when naive.

        Bridge for engine APIs (``pa.timestamp``, ``pl.Datetime``,
        Spark) that take a string. New code should prefer ``self.tz``
        directly — it carries :class:`Timezone` helpers like
        ``is_utc()`` and ``utc_offset()``.
        """
        return None if self.tz.is_naive() else self.tz.iana

    def _default_pyhint(self) -> Any:
        import datetime as _dt
        return _dt.datetime

    def to_arrow(self) -> "pa.DataType":
        return pa.timestamp(unit=self.unit, tz=self.tz_iana)

    def to_polars(self) -> "polars.DataType":
        pl = polars_module()
        return pl.Datetime(time_unit=self.unit, time_zone=self.tz_iana)

    def to_spark(self) -> Any:
        spark = spark_sql_module()
        if self.tz.is_naive() and hasattr(spark.types, "TimestampNTZType"):
            return spark.types.TimestampNTZType()
        return spark.types.TimestampType()

    def as_spark(self) -> "TimestampType":
        # Spark only has UTC-anchored ``TimestampType`` and naive
        # ``TimestampNTZType`` — non-UTC zones (``Europe/Paris``,
        # ``America/New_York`` …) lose their offset on round-trip.
        # Drop them to ``Timezone.NAIVE`` so the wall-clock survives
        # rather than silently shifting to UTC at write time.
        if self.tz.is_naive() or self.tz.is_utc():
            return self
        return TimestampType(unit=self.unit, tz=Timezone.NAIVE)

    def as_polars(self) -> "TimestampType":
        # Polars ``Datetime`` only supports ``ms`` / ``us`` / ``ns``;
        # second-precision timestamps widen to ms so the
        # ``to_polars()`` produces a dtype Polars actually stores.
        if str(self.unit) == "s":
            return TimestampType(unit=TimeUnit.MILLISECOND, tz=self.tz)
        return self

    def to_spark_name(self) -> str:
        # Databricks ``TIMESTAMP`` is UTC-anchored — only emit it when the tz is
        # UTC-equivalent. Anything else (naive or a real non-UTC zone like
        # ``Europe/Paris``) drops to ``TIMESTAMP_NTZ`` so the wall-clock value
        # round-trips without a silent shift.
        if self.tz.is_naive():
            return "TIMESTAMP_NTZ"
        return "TIMESTAMP" if self.tz.is_utc() else "TIMESTAMP_NTZ"

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        # ``Timezone.NAIVE`` is falsy via ``__bool__`` so ``if self.tz``
        # cleanly distinguishes naive from aware here.
        if self.tz:
            return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        return dt.datetime(1970, 1, 1)

    def _convert_pyobj(self, value: Any, safe: bool = False) -> "dt.datetime | None":
        if value is None:
            return None
        if isinstance(value, dt.datetime):
            return self._apply_tz(value)
        if isinstance(value, dt.date):
            return self._apply_tz(dt.datetime(value.year, value.month, value.day))
        text = _coerce_str(value)
        if text is not None:
            parsed = self._parse_iso_scalar_string(text, "timestamp", safe)
            return None if parsed is None else self._apply_tz(parsed)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            factor = TimeUnit.from_(self.unit, default=TimeUnit.MICROSECOND).seconds
            try:
                seconds = float(value) * factor
                ts = dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
            except (OverflowError, OSError, ValueError):
                if safe:
                    raise ValueError(
                        f"Cannot convert epoch value {value!r} to timestamp "
                        f"for {type(self).__name__}."
                    )
                return None
            if self.tz.is_naive():
                return ts.replace(tzinfo=None)
            return self._apply_tz(ts)
        if safe:
            raise ValueError(
                f"Cannot convert {type(value).__name__} to timestamp "
                f"for {type(self).__name__}: {value!r}."
            )
        return None

    def _apply_tz(self, value: "dt.datetime") -> "dt.datetime":
        # Naive-target: strip tz. Aware-target: attach UTC when given a naive value,
        # otherwise preserve; downstream Arrow cast normalizes to ``self.tz``.
        if self.tz.is_naive():
            if value.tzinfo is None:
                return value
            return value.astimezone(dt.timezone.utc).replace(tzinfo=None)
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value


# ======================================================================
# DurationType
# ======================================================================


@dataclass(frozen=True, repr=False)
class DurationType(TemporalType):
    unit: str = "us"
    tz: str | None = None

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "byte_size", 8)

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.DURATION

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        return f"{pad}duration[{self.unit}{f', tz={self.tz}' if self.tz else ''}]"

    @classmethod
    def handles_arrow_type(cls, dtype: "pa.DataType") -> bool:
        return pa.types.is_duration(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: "pa.DurationType") -> "DurationType":
        if not pa.types.is_duration(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")
        return cls(byte_size=8, unit=dtype.unit)

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = polars_module()
        return isinstance(dtype, pl.Duration)

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "DurationType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        unit = getattr(dtype, "time_unit", "us") or "us"
        return cls(byte_size=8, unit=unit)

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        return False

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "DurationType":
        raise TypeError(f"Spark has no native duration type: {dtype!r}")

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.DURATION)

    def _default_pyhint(self) -> Any:
        import datetime as _dt
        return _dt.timedelta

    def to_arrow(self) -> "pa.DataType":
        return pa.duration(self.unit)

    def to_polars(self) -> "polars.DataType":
        pl = polars_module()
        return pl.Duration(time_unit=self.unit)

    def to_spark(self) -> Any:
        spark = spark_sql_module()
        return spark.types.LongType()

    def as_spark(self):
        # Spark has no native interval type — widen to a 64-bit signed
        # integer to mirror :meth:`to_spark` (``LongType``).
        from .numeric import IntegerType

        return IntegerType(byte_size=8, signed=True)

    def as_polars(self) -> "DurationType":
        # Polars ``Duration`` only supports ``ms`` / ``us`` / ``ns``;
        # second-precision durations widen to ms so ``to_polars()``
        # produces a dtype Polars actually stores.
        if str(self.unit) == "s":
            return DurationType(unit=TimeUnit.MILLISECOND)
        return self

    def to_spark_name(self) -> str:
        return "BIGINT"

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else dt.timedelta(0)

    def _convert_pyobj(self, value: Any, safe: bool = False) -> "dt.timedelta | None":
        if value is None:
            return None
        if isinstance(value, dt.timedelta):
            return value
        if isinstance(value, bool):
            return self._timedelta_from_numeric(float(value))
        if isinstance(value, (int, float)):
            return self._timedelta_from_numeric(float(value))
        if isinstance(value, decimal.Decimal):
            return self._timedelta_from_numeric(float(value))
        text = _coerce_str(value)
        if text is not None:
            stripped = text.strip()
            if not stripped:
                if safe:
                    raise ValueError(
                        f"Cannot parse duration from empty string "
                        f"for {type(self).__name__}."
                    )
                return None
            # Cascade: ISO-8601 duration → clock-style → numeric count-of-unit.
            # Vectorised casts stay stripped-down (nuclear mode) but scalar
            # conversion keeps these shapes because callers that reach
            # ``_convert_pyobj`` (list-of-dict ingest, serde round-trip) get
            # loose Python values and can't pre-parse cheaply.
            iso = _parse_iso_duration(stripped)
            if iso is not None:
                return iso
            clock = _parse_clock_duration(stripped)
            if clock is not None:
                return clock
            try:
                return self._timedelta_from_numeric(float(stripped))
            except ValueError:
                if safe:
                    raise ValueError(
                        f"Cannot parse duration from {value!r} "
                        f"for {type(self).__name__}."
                    )
                return None
        if safe:
            raise ValueError(
                f"Cannot convert {type(value).__name__} to duration "
                f"for {type(self).__name__}: {value!r}."
            )
        return None

    def _timedelta_from_numeric(self, value: float) -> "dt.timedelta":
        factor = TimeUnit.from_(self.unit, default=TimeUnit.MICROSECOND).seconds
        return dt.timedelta(seconds=value * factor)