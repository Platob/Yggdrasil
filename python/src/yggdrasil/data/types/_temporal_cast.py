"""Vectorized best-effort temporal casting helpers.

User data hitting the library is messy — strings in a dozen formats, epoch
numbers in seconds/ms/us/ns, naive timestamps tagged with timezones after the
fact, durations spelled as plain integers. This module turns that chaos into
Arrow/Spark-native temporal arrays without ever looping in Python.

Everything here is vectorized: pyarrow.compute for Arrow arrays,
pyspark.sql.functions for Spark columns. Pure-Python loops are deliberately
avoided so ingestion stays fast even on wide, deep inputs.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from .support import get_spark_sql

if TYPE_CHECKING:
    import pyspark.sql as ps


__all__ = [
    "ARROW_DATETIME_FORMATS_TZ",
    "ARROW_DATETIME_FORMATS_NAIVE",
    "ARROW_DATE_FORMATS",
    "ARROW_TIME_FORMATS",
    "SPARK_DATETIME_FORMATS",
    "SPARK_DATE_FORMATS",
    "SPARK_TIME_FORMATS",
    "nullify_empty_strings",
    "arrow_str_to_timestamp",
    "arrow_str_to_date",
    "arrow_str_to_time",
    "arrow_str_to_duration",
    "arrow_numeric_to_timestamp",
    "arrow_numeric_to_date",
    "arrow_numeric_to_time",
    "arrow_numeric_to_duration",
    "arrow_cast_to_timestamp",
    "arrow_cast_to_date",
    "arrow_cast_to_time",
    "arrow_cast_to_duration",
    "arrow_temporal_to_string",
    "spark_to_timestamp",
    "spark_to_date",
    "spark_to_time_string",
    "spark_to_duration_seconds",
]


# ---------------------------------------------------------------------------
# Format catalogues
# ---------------------------------------------------------------------------

# strptime-style patterns. Arrow uses C strptime semantics — %.f is not valid
# there, %f is. Each format is attempted independently; non-matching rows are
# nulled via error_is_null=True so pc.coalesce can pick the first winner.

ARROW_DATETIME_FORMATS_TZ: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M%z",
    "%Y-%m-%d %H:%M%z",
)

ARROW_DATETIME_FORMATS_NAIVE: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y",
    "%Y%m%dT%H%M%S",
    "%Y%m%d%H%M%S",
    "%Y%m%d",
    "%d %b %Y %H:%M:%S",
    "%d %b %Y",
    "%b %d, %Y",
    "%Y-%m-%d",
)

ARROW_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y%m%d",
    "%d %b %Y",
    "%b %d, %Y",
    "%d.%m.%Y",
)

ARROW_TIME_FORMATS: tuple[str, ...] = (
    "%H:%M:%S",
    "%H:%M",
    "%I:%M:%S %p",
    "%I:%M %p",
)

# Spark uses Java DateTimeFormatter patterns (not strptime). Mirrors the Arrow
# catalogue in intent, not in letters.
SPARK_DATETIME_FORMATS: tuple[str, ...] = (
    "yyyy-MM-dd'T'HH:mm:ss.SSSXXX",
    "yyyy-MM-dd'T'HH:mm:ssXXX",
    "yyyy-MM-dd HH:mm:ss.SSSXXX",
    "yyyy-MM-dd HH:mm:ssXXX",
    "yyyy-MM-dd'T'HH:mm:ss.SSS",
    "yyyy-MM-dd'T'HH:mm:ss",
    "yyyy-MM-dd HH:mm:ss.SSS",
    "yyyy-MM-dd HH:mm:ss",
    "yyyy-MM-dd'T'HH:mm",
    "yyyy-MM-dd HH:mm",
    "yyyy/MM/dd HH:mm:ss",
    "yyyy/MM/dd",
    "dd/MM/yyyy HH:mm:ss",
    "dd/MM/yyyy",
    "MM/dd/yyyy HH:mm:ss",
    "MM/dd/yyyy",
    "dd-MM-yyyy HH:mm:ss",
    "dd-MM-yyyy",
    "yyyyMMdd'T'HHmmss",
    "yyyyMMddHHmmss",
    "yyyyMMdd",
    "dd MMM yyyy HH:mm:ss",
    "dd MMM yyyy",
    "MMM dd, yyyy",
    "yyyy-MM-dd",
)

SPARK_DATE_FORMATS: tuple[str, ...] = (
    "yyyy-MM-dd",
    "yyyy/MM/dd",
    "dd/MM/yyyy",
    "MM/dd/yyyy",
    "dd-MM-yyyy",
    "yyyyMMdd",
    "dd MMM yyyy",
    "MMM dd, yyyy",
    "dd.MM.yyyy",
)

SPARK_TIME_FORMATS: tuple[str, ...] = (
    "HH:mm:ss.SSS",
    "HH:mm:ss",
    "HH:mm",
    "hh:mm:ss a",
    "hh:mm a",
)


# ---------------------------------------------------------------------------
# Arrow helpers
# ---------------------------------------------------------------------------

_TIME_UNIT_PER_SECOND = {
    "s": 1,
    "ms": 1_000,
    "us": 1_000_000,
    "ns": 1_000_000_000,
}


def _is_chunked(array: Any) -> bool:
    return isinstance(array, pa.ChunkedArray)


def _apply_chunked(array: Any, fn):
    """Map *fn* across chunks, rebuilding a ChunkedArray with the result type."""
    if not _is_chunked(array):
        return fn(array)

    chunks = [fn(chunk) for chunk in array.chunks]
    if not chunks:
        # Keep caller-compatible: return a fresh, correctly-typed chunked array.
        placeholder = fn(pa.array([], type=array.type))
        return pa.chunked_array([], type=placeholder.type)
    return pa.chunked_array(chunks, type=chunks[0].type)


def _ensure_string(array: pa.Array) -> pa.Array:
    """Force an Arrow array to utf8 so strptime/utf8_* kernels accept it."""
    if pa.types.is_string(array.type):
        return array
    return pc.cast(array, pa.string())


def nullify_empty_strings(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Trim + treat empty strings as null.

    Best-effort parsing falls apart on blank cells, so normalize here.
    """

    def _one(chunk: pa.Array) -> pa.Array:
        if not pa.types.is_string(chunk.type) and not pa.types.is_large_string(
            chunk.type
        ):
            return chunk
        trimmed = pc.utf8_trim_whitespace(chunk)
        empty = pa.scalar("", type=trimmed.type)
        null_scalar = pa.scalar(None, type=trimmed.type)
        return pc.if_else(pc.equal(trimmed, empty), null_scalar, trimmed)

    return _apply_chunked(array, _one)


def _coalesce_strptime(
    array: pa.Array,
    formats: tuple[str, ...] | list[str],
    unit: str,
    strip_tz: bool = False,
) -> pa.Array:
    """Run *formats* through strptime and coalesce non-null parses row-wise.

    Arrow's strptime raises when the C format string contains directives it
    cannot parse (notably %z on some platforms). Those formats are skipped
    silently — a format that cannot compile cannot produce matches anyway.
    """
    target_type = pa.timestamp(unit)
    parsed: list[pa.Array] = []

    for fmt in formats:
        try:
            out = pc.strptime(array, format=fmt, unit=unit, error_is_null=True)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowKeyError):
            continue
        if strip_tz and out.type.tz:
            out = pc.cast(out, target_type)
        parsed.append(out)

    if not parsed:
        return pa.nulls(len(array), type=target_type)
    if len(parsed) == 1:
        return parsed[0]
    return pc.coalesce(*parsed)


def _numeric_epoch_scale(array: pa.Array, target_unit: str) -> pa.Array:
    """Scale an epoch-numeric array to int64 in *target_unit*.

    Users pass epoch in seconds, milliseconds, microseconds, or nanoseconds.
    We pick per-row based on magnitude against the current wall clock, so a
    single column can mix scales without breaking.
    """
    target_per_second = _TIME_UNIT_PER_SECOND[target_unit]

    as_float = pc.cast(array, pa.float64(), safe=False)
    abs_arr = pc.abs_checked(as_float)
    now_s = max(time.time(), 1.0)

    # Yeah, the thresholds look magic, but they match the scalar heuristic in
    # data/cast/datetime.py — keeping vectorized and scalar paths aligned keeps
    # users from getting different results depending on entry point.
    bound_s = pa.scalar(now_s * 100.0, type=pa.float64())
    bound_ms = pa.scalar(now_s * 100_000.0, type=pa.float64())

    is_s = pc.less(abs_arr, bound_s)
    is_ms = pc.and_(pc.greater_equal(abs_arr, bound_s), pc.less(abs_arr, bound_ms))

    scale_s = pa.scalar(float(target_per_second), type=pa.float64())
    scale_ms = pa.scalar(target_per_second / 1_000.0, type=pa.float64())
    scale_us = pa.scalar(target_per_second / 1_000_000.0, type=pa.float64())

    picked = pc.if_else(
        is_s,
        pc.multiply(as_float, scale_s),
        pc.if_else(
            is_ms,
            pc.multiply(as_float, scale_ms),
            pc.multiply(as_float, scale_us),
        ),
    )
    return pc.cast(picked, pa.int64(), safe=False)


def arrow_str_to_timestamp(
    array: pa.Array | pa.ChunkedArray,
    unit: str = "us",
    tz: str | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Best-effort parse a string array to ``timestamp[unit, tz]``."""

    def _one(chunk: pa.Array) -> pa.Array:
        chunk = _ensure_string(chunk)
        chunk = nullify_empty_strings(chunk)
        chunk = _strip_fractional_seconds(chunk)
        naive = _coalesce_strptime(
            chunk,
            ARROW_DATETIME_FORMATS_TZ + ARROW_DATETIME_FORMATS_NAIVE,
            unit=unit,
            strip_tz=True,
        )
        if tz:
            # The tz-aware branch produced UTC wall-clock. Naive parses have no
            # recorded tz, so we trust UTC here — that matches the scalar
            # parser's default in data/cast/datetime.py.
            stamped = pc.assume_timezone(naive, timezone="UTC")
            return pc.cast(stamped, pa.timestamp(unit, tz))
        return naive

    return _apply_chunked(array, _one)


def arrow_str_to_date(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Best-effort parse a string array to ``date32``."""

    def _one(chunk: pa.Array) -> pa.Array:
        chunk = _ensure_string(chunk)
        chunk = nullify_empty_strings(chunk)
        chunk = _strip_fractional_seconds(chunk)
        # Reuse the datetime catalogue too — users frequently hand us a full
        # datetime string and expect the date portion.
        parsed = _coalesce_strptime(
            chunk,
            ARROW_DATE_FORMATS + ARROW_DATETIME_FORMATS_NAIVE,
            unit="us",
            strip_tz=True,
        )
        return pc.cast(parsed, pa.date32())

    return _apply_chunked(array, _one)


def _strip_fractional_seconds(array: pa.Array) -> pa.Array:
    """Drop the fractional-second segment before strptime.

    Arrow's strptime is built on C ``strptime`` and doesn't understand ``%f``
    or ``%.f``. Sub-second precision is lost on this path — callers that need
    it should parse strings into timestamps upstream.
    """
    # Match a leading ``.`` plus up to 9 digits. Keep whatever follows (tz
    # offset, ``Z`` suffix, or end-of-string).
    return pc.replace_substring_regex(array, pattern=r"\.\d{1,9}", replacement="")


def arrow_str_to_time(
    array: pa.Array | pa.ChunkedArray,
    unit: str = "us",
) -> pa.Array | pa.ChunkedArray:
    """Best-effort parse a string array to ``time32/time64``.

    Arrow's strptime cannot parse a bare ``HH:MM:SS`` — it always wants a date
    component. We prepend a placeholder date before parsing, then cast the
    resulting timestamp back to ``time``.
    """
    time_type = pa.time64(unit) if unit in {"us", "ns"} else pa.time32(unit)

    def _one(chunk: pa.Array) -> pa.Array:
        chunk = _ensure_string(chunk)
        chunk = nullify_empty_strings(chunk)
        chunk = _strip_fractional_seconds(chunk)

        # Prepend a fixed date so strptime treats the input as a full timestamp.
        prefix = pa.scalar("1970-01-01T", type=chunk.type)
        prefixed = pc.binary_join_element_wise(prefix, chunk, "")

        full_formats = tuple(f"%Y-%m-%dT{fmt}" for fmt in ARROW_TIME_FORMATS)
        parsed = _coalesce_strptime(prefixed, full_formats, unit=unit)
        return pc.cast(parsed, time_type)

    return _apply_chunked(array, _one)


def arrow_str_to_duration(
    array: pa.Array | pa.ChunkedArray,
    unit: str = "us",
) -> pa.Array | pa.ChunkedArray:
    """Parse an integer-valued string array into ``duration[unit]``.

    Real duration string parsing (``"PT15M"``, ``"1d 02:30"``) is scalar-only
    via :mod:`yggdrasil.data.cast.datetime`. For bulk ingest we accept the
    common case where the column holds stringified counts in *unit*.
    """
    target = pa.duration(unit)

    def _one(chunk: pa.Array) -> pa.Array:
        chunk = _ensure_string(chunk)
        chunk = nullify_empty_strings(chunk)
        as_int = pc.cast(chunk, pa.int64(), safe=False)
        return pc.cast(as_int, target)

    return _apply_chunked(array, _one)


def arrow_numeric_to_timestamp(
    array: pa.Array | pa.ChunkedArray,
    unit: str = "us",
    tz: str | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Cast epoch-numeric arrays to ``timestamp[unit, tz]`` with unit detection."""

    def _one(chunk: pa.Array) -> pa.Array:
        scaled = _numeric_epoch_scale(chunk, unit)
        return pc.cast(scaled, pa.timestamp(unit, tz))

    return _apply_chunked(array, _one)


def arrow_numeric_to_date(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Cast epoch-numeric arrays to ``date32``."""

    def _one(chunk: pa.Array) -> pa.Array:
        scaled = _numeric_epoch_scale(chunk, "s")
        ts = pc.cast(scaled, pa.timestamp("s"))
        return pc.cast(ts, pa.date32())

    return _apply_chunked(array, _one)


def arrow_numeric_to_time(
    array: pa.Array | pa.ChunkedArray,
    unit: str = "us",
) -> pa.Array | pa.ChunkedArray:
    """Cast numeric arrays (treated as count-of-unit) to ``time``."""
    time_type = pa.time64(unit) if unit in {"us", "ns"} else pa.time32(unit)

    def _one(chunk: pa.Array) -> pa.Array:
        target_int = pa.int64() if unit in {"us", "ns"} else pa.int32()
        as_int = pc.cast(chunk, target_int, safe=False)
        return pc.cast(as_int, time_type)

    return _apply_chunked(array, _one)


def arrow_numeric_to_duration(
    array: pa.Array | pa.ChunkedArray,
    unit: str = "us",
) -> pa.Array | pa.ChunkedArray:
    """Cast numeric arrays to ``duration[unit]``.

    We do *not* auto-scale here: a plain integer meant as ``seconds`` would
    silently lose three orders of magnitude if we stole the timestamp
    heuristic. Explicit beats clever for durations.
    """
    target = pa.duration(unit)

    def _one(chunk: pa.Array) -> pa.Array:
        as_int = pc.cast(chunk, pa.int64(), safe=False)
        return pc.cast(as_int, target)

    return _apply_chunked(array, _one)


def arrow_cast_to_timestamp(
    array: pa.Array | pa.ChunkedArray,
    unit: str = "us",
    tz: str | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Dispatch any source Arrow type to a ``timestamp[unit, tz]`` array."""
    src = array.type
    target = pa.timestamp(unit, tz)

    if pa.types.is_timestamp(src):
        return pc.cast(array, target)

    if pa.types.is_date(src):
        return pc.cast(pc.cast(array, pa.timestamp(unit)), target)

    if pa.types.is_time(src):
        # Wall-clock time has no date — anchor at epoch so the caller at
        # least gets a stable ordering. pyarrow can't cast time→duration
        # directly, so bounce through int64 (count of the source unit).
        src_unit = src.unit
        as_int = pc.cast(array, pa.int64())
        as_dur = pc.cast(as_int, pa.duration(src_unit))
        aligned = pc.cast(as_dur, pa.duration(unit))
        epoch = pa.scalar(0, type=pa.timestamp(unit))
        shifted = pc.add(epoch, aligned)
        return pc.cast(shifted, target)

    if pa.types.is_duration(src):
        casted = pc.cast(array, pa.duration(unit))
        epoch = pa.scalar(0, type=pa.timestamp(unit))
        shifted = pc.add(epoch, casted)
        return pc.cast(shifted, target)

    if pa.types.is_string(src) or pa.types.is_large_string(src):
        return arrow_str_to_timestamp(array, unit=unit, tz=tz)

    if pa.types.is_integer(src) or pa.types.is_floating(src):
        return arrow_numeric_to_timestamp(array, unit=unit, tz=tz)

    if pa.types.is_null(src):
        return pa.nulls(len(array), type=target)

    return pc.cast(array, target)


def arrow_cast_to_date(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Dispatch any source Arrow type to ``date32``."""
    src = array.type

    if pa.types.is_date(src):
        return pc.cast(array, pa.date32())

    if pa.types.is_timestamp(src):
        # Casts that carry a tz drop it implicitly to the wall-clock date, same
        # as pyarrow. Timestamps without tz are already wall-clock.
        return pc.cast(array, pa.date32())

    if pa.types.is_string(src) or pa.types.is_large_string(src):
        return arrow_str_to_date(array)

    if pa.types.is_integer(src) or pa.types.is_floating(src):
        return arrow_numeric_to_date(array)

    if pa.types.is_null(src):
        return pa.nulls(len(array), type=pa.date32())

    return pc.cast(array, pa.date32())


def arrow_cast_to_time(
    array: pa.Array | pa.ChunkedArray,
    unit: str = "us",
) -> pa.Array | pa.ChunkedArray:
    """Dispatch any source Arrow type to ``time[unit]``."""
    src = array.type
    time_type = pa.time64(unit) if unit in {"us", "ns"} else pa.time32(unit)

    if pa.types.is_time(src):
        return pc.cast(array, time_type)

    if pa.types.is_timestamp(src):
        return pc.cast(array, time_type)

    if pa.types.is_duration(src):
        # Normalize into a day, then drop the day portion — same semantics as
        # casting a timestamp on 1970-01-01 to time.
        casted = pc.cast(array, pa.duration(unit))
        epoch = pa.scalar(0, type=pa.timestamp(unit))
        shifted = pc.add(epoch, casted)
        return pc.cast(shifted, time_type)

    if pa.types.is_string(src) or pa.types.is_large_string(src):
        return arrow_str_to_time(array, unit=unit)

    if pa.types.is_integer(src) or pa.types.is_floating(src):
        return arrow_numeric_to_time(array, unit=unit)

    if pa.types.is_null(src):
        return pa.nulls(len(array), type=time_type)

    return pc.cast(array, time_type)


def arrow_cast_to_duration(
    array: pa.Array | pa.ChunkedArray,
    unit: str = "us",
) -> pa.Array | pa.ChunkedArray:
    """Dispatch any source Arrow type to ``duration[unit]``."""
    src = array.type
    target = pa.duration(unit)

    if pa.types.is_duration(src):
        return pc.cast(array, target)

    if pa.types.is_timestamp(src):
        # Distance from epoch in the target unit — matches the Polars path.
        if src.tz:
            array = pc.cast(array, pa.timestamp(src.unit))
        aligned = pc.cast(array, pa.timestamp(unit))
        epoch = pa.scalar(0, type=pa.timestamp(unit))
        diff = pc.subtract(aligned, epoch)
        return pc.cast(diff, target)

    if pa.types.is_time(src):
        # Same trick as the timestamp path: time→int64 (count of source unit)
        # →duration then normalize to the requested unit.
        src_unit = src.unit
        as_int = pc.cast(array, pa.int64())
        as_dur = pc.cast(as_int, pa.duration(src_unit))
        return pc.cast(as_dur, target)

    if pa.types.is_date(src):
        as_ts = pc.cast(array, pa.timestamp(unit))
        epoch = pa.scalar(0, type=pa.timestamp(unit))
        diff = pc.subtract(as_ts, epoch)
        return pc.cast(diff, target)

    if pa.types.is_string(src) or pa.types.is_large_string(src):
        return arrow_str_to_duration(array, unit=unit)

    if pa.types.is_integer(src) or pa.types.is_floating(src):
        return arrow_numeric_to_duration(array, unit=unit)

    if pa.types.is_null(src):
        return pa.nulls(len(array), type=target)

    return pc.cast(array, target)


def arrow_temporal_to_string(
    array: pa.Array | pa.ChunkedArray,
    fmt: str | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Format a temporal array as strings. Durations get an ISO-8601-ish repr."""

    def _one(chunk: pa.Array) -> pa.Array:
        src = chunk.type

        if pa.types.is_date(src):
            return pc.strftime(chunk, format=fmt or "%Y-%m-%d")

        if pa.types.is_time(src):
            return pc.cast(chunk, pa.string())

        if pa.types.is_timestamp(src):
            default_fmt = "%Y-%m-%dT%H:%M:%S.%f%z" if src.tz else "%Y-%m-%dT%H:%M:%S.%f"
            return pc.strftime(chunk, format=fmt or default_fmt)

        if pa.types.is_duration(src):
            # strftime doesn't take duration. Render in the source unit so the
            # string round-trips cleanly through arrow_str_to_duration.
            return pc.cast(pc.cast(chunk, pa.int64()), pa.string())

        return pc.cast(chunk, pa.string())

    return _apply_chunked(array, _one)


# ---------------------------------------------------------------------------
# Spark helpers
# ---------------------------------------------------------------------------


def _spark_when(F: Any) -> Any:
    return F


def spark_to_timestamp(
    column: "ps.Column",
    unit: str = "us",
    tz: str | None = None,
) -> "ps.Column":
    """Best-effort cast a Spark column to a timestamp with tz handling."""
    spark = get_spark_sql()
    F = spark.functions
    T = spark.types

    source_type_name = None
    try:
        # In-expression columns don't carry a schema — so only used if present.
        source_type_name = column._jc.expr().dataType().typeName()  # type: ignore[attr-defined]
    except Exception:
        source_type_name = None

    is_string_source = source_type_name == "string"
    is_numeric_source = source_type_name in {
        "byte",
        "short",
        "integer",
        "long",
        "float",
        "double",
        "decimal",
    }

    if is_string_source:
        attempts = [F.to_timestamp(column, fmt) for fmt in SPARK_DATETIME_FORMATS]
        casted = F.coalesce(*attempts, column.cast(T.TimestampType()))
    elif is_numeric_source:
        seconds = _spark_epoch_to_seconds(column)
        casted = seconds.cast(T.TimestampType())
    else:
        casted = column.cast(T.TimestampType())

    if tz is None:
        if hasattr(T, "TimestampNTZType"):
            casted = casted.cast(T.TimestampNTZType())
        return casted

    # from_utc_timestamp doesn't change stored instant on tz-aware columns; it
    # returns wall-clock in the requested zone. Spark stores as instant anyway,
    # so returning as-is is the closest match to Arrow's tz-aware type.
    return casted


def _spark_epoch_to_seconds(column: "ps.Column") -> "ps.Column":
    """Pick seconds/ms/us/ns for an epoch number column, row-wise."""
    spark = get_spark_sql()
    F = spark.functions
    T = spark.types

    abs_col = F.abs(column.cast(T.DoubleType()))
    now_s = max(time.time(), 1.0)

    bound_s = F.lit(now_s * 100.0)
    bound_ms = F.lit(now_s * 100_000.0)

    as_double = column.cast(T.DoubleType())
    return (
        F.when(abs_col < bound_s, as_double)
        .when(abs_col < bound_ms, as_double / F.lit(1000.0))
        .otherwise(as_double / F.lit(1_000_000.0))
    )


def spark_to_date(column: "ps.Column") -> "ps.Column":
    """Best-effort cast a Spark column to DateType."""
    spark = get_spark_sql()
    F = spark.functions
    T = spark.types

    try:
        source_type_name = column._jc.expr().dataType().typeName()  # type: ignore[attr-defined]
    except Exception:
        source_type_name = None

    if source_type_name == "string":
        attempts = [F.to_date(column, fmt) for fmt in SPARK_DATE_FORMATS]
        attempts.extend(F.to_date(column, fmt) for fmt in SPARK_DATETIME_FORMATS)
        return F.coalesce(*attempts, column.cast(T.DateType()))

    if source_type_name in {
        "byte",
        "short",
        "integer",
        "long",
        "float",
        "double",
        "decimal",
    }:
        seconds = _spark_epoch_to_seconds(column)
        return seconds.cast(T.TimestampType()).cast(T.DateType())

    return column.cast(T.DateType())


def spark_to_time_string(column: "ps.Column") -> "ps.Column":
    """Render a Spark column as an HH:mm:ss.SSS string (Spark has no TIME)."""
    spark = get_spark_sql()
    F = spark.functions
    T = spark.types

    try:
        source_type_name = column._jc.expr().dataType().typeName()  # type: ignore[attr-defined]
    except Exception:
        source_type_name = None

    if source_type_name == "string":
        return column.cast(T.StringType())

    if source_type_name in {"timestamp", "timestamp_ntz"}:
        return F.date_format(column, "HH:mm:ss.SSS")

    if source_type_name in {
        "byte",
        "short",
        "integer",
        "long",
        "float",
        "double",
        "decimal",
    }:
        # treat numeric as seconds since midnight
        seconds = column.cast(T.DoubleType())
        base = F.to_timestamp(F.lit("1970-01-01 00:00:00"))
        shifted = (base.cast(T.DoubleType()) + seconds).cast(T.TimestampType())
        return F.date_format(shifted, "HH:mm:ss.SSS")

    return column.cast(T.StringType())


def spark_to_duration_seconds(column: "ps.Column", unit: str = "us") -> "ps.Column":
    """Best-effort cast a Spark column to an integer count-of-unit.

    Spark has no native duration type; yggdrasil lands durations in ``LongType``
    counting *unit* steps. This mirrors :meth:`DurationType.to_spark`.
    """
    spark = get_spark_sql()
    F = spark.functions
    T = spark.types

    try:
        source_type_name = column._jc.expr().dataType().typeName()  # type: ignore[attr-defined]
    except Exception:
        source_type_name = None

    scale = _TIME_UNIT_PER_SECOND[unit]

    if source_type_name == "string":
        return column.cast(T.LongType())

    if source_type_name in {"timestamp", "timestamp_ntz"}:
        epoch_seconds = column.cast(T.DoubleType())
        return (epoch_seconds * F.lit(float(scale))).cast(T.LongType())

    if source_type_name == "date":
        # days since epoch * seconds per day * unit scale
        epoch_seconds = F.unix_timestamp(column.cast(T.TimestampType()))
        return (epoch_seconds.cast(T.DoubleType()) * F.lit(float(scale))).cast(
            T.LongType()
        )

    return column.cast(T.LongType())
