"""Vectorized best-effort temporal casting helpers.

User data hitting the library is messy — strings in a dozen formats, epoch
numbers in seconds/ms/us/ns, naive timestamps tagged with timezones after the
fact, durations spelled as plain integers. This module turns that chaos into
Arrow / Polars / Spark-native temporal arrays without ever looping in Python.

Everything here is vectorized: pyarrow.compute for Arrow arrays, polars
expressions for polars Series / Expr, pyspark.sql.functions for Spark
columns. Pure-Python loops are deliberately avoided so ingestion stays fast
even on wide, deep inputs.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any, Union

import pyarrow as pa
import pyarrow.compute as pc

from .support import get_polars, get_spark_sql

if TYPE_CHECKING:
    import polars as pl  # noqa: F401
    import pyspark.sql as ps


__all__ = [
    "ARROW_DATETIME_FORMATS_TZ",
    "ARROW_DATETIME_FORMATS_NAIVE",
    "ARROW_DATE_FORMATS",
    "ARROW_TIME_FORMATS",
    "POLARS_DATETIME_FORMATS_TZ",
    "POLARS_DATETIME_FORMATS_NAIVE",
    "POLARS_DATE_FORMATS",
    "POLARS_TIME_FORMATS",
    "POLARS_PARSE_DATETIME_FORMATS_TZ",
    "POLARS_PARSE_DATETIME_FORMATS_NAIVE",
    "SPARK_DATETIME_FORMATS",
    "SPARK_DATE_FORMATS",
    "SPARK_TIME_FORMATS",
    "ISO_DURATION_DAYS_PER_YEAR",
    "ISO_DURATION_DAYS_PER_MONTH",
    "ISO_DURATION_DAYS_PER_WEEK",
    "nullify_empty_strings",
    "parse_iso_duration_to_nanos",
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
    "retimestamp_prefer_polars",
    "arrow_cast_to_string",
    "arrow_timestamp_to_string",
    "arrow_date_to_string",
    "arrow_time_to_string",
    "arrow_duration_to_string",
    "arrow_temporal_to_string",
    "attach_fractional_seconds",
    "cast_polars_array_to_temporal",
    "spark_to_timestamp",
    "spark_to_date",
    "spark_to_time_string",
    "spark_to_duration_seconds",
    "spark_temporal_to_string",
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

# Calendar units in ISO 8601 durations (Y, M, W) have no fixed second-count.
# Real calendars vary; for ingestion into a fixed-width ``duration[unit]``
# we collapse them to whole-day defaults — the convention most data sources
# (Postgres ``interval``, ``isodate``, Java ``Duration.parse``) settle on.
ISO_DURATION_DAYS_PER_YEAR = 365
ISO_DURATION_DAYS_PER_MONTH = 30
ISO_DURATION_DAYS_PER_WEEK = 7

# ``P[n]Y[n]M[n]W[n]D[T[n]H[n]M[n]S]``. Fields are independently optional but
# at least one must be present (handled in :func:`parse_iso_duration_to_nanos`
# — the regex alone would happily match a bare ``P``). Decimals are accepted
# on every component, including the ISO-permitted comma separator. A ``+`` /
# ``-`` sign prefix is a common extension; we accept both.
_RE_ISO_DURATION = re.compile(
    r"^(?P<sign>[+-])?P"
    r"(?:(?P<years>\d+(?:[.,]\d+)?)Y)?"
    r"(?:(?P<months>\d+(?:[.,]\d+)?)M)?"
    r"(?:(?P<weeks>\d+(?:[.,]\d+)?)W)?"
    r"(?:(?P<days>\d+(?:[.,]\d+)?)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+(?:[.,]\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:[.,]\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:[.,]\d+)?)S)?"
    r")?$",
    re.IGNORECASE,
)

# Looks-like-an-integer for the count-of-unit fast path. Accepts an optional
# sign and surrounding whitespace; rejects decimals so they fall through to
# the float-parse branch.
_RE_INT_LITERAL = re.compile(r"^[+-]?\d+$")


def parse_iso_duration_to_nanos(value: str) -> int | None:
    """Parse an ISO 8601 duration string to total nanoseconds.

    Recognises the standard ``P[n]Y[n]M[n]W[n]D[T[n]H[n]M[n]S]`` shape and a
    handful of common extensions: leading ``+`` / ``-`` sign, ``,`` as decimal
    separator, lowercase letters, and decimals on any component.

    Calendar fields collapse via the module-level day defaults
    (:data:`ISO_DURATION_DAYS_PER_YEAR` etc.) — ``P1Y`` becomes 365 days,
    ``P1M`` becomes 30 days, ``P1W`` stays at 7 days. The ISO spec leaves
    those values implementation-defined; we pick the widely-used "30/365"
    convention rather than tracking a reference date the caller doesn't have.

    Returns ``None`` when *value* is not a recognisable ISO duration, when
    every component is absent (``"P"`` / ``"PT"`` carry no information), or
    when *value* is empty / ``None``-ish.
    """
    if not value:
        return None

    m = _RE_ISO_DURATION.fullmatch(value.strip())
    if m is None:
        return None

    parts = {
        name: m.group(name)
        for name in ("years", "months", "weeks", "days", "hours", "minutes", "seconds")
    }
    if not any(parts.values()):
        # Bare "P" / "PT" — syntactically the regex permits it, but it
        # encodes no duration. Treat as unparseable so the caller can null it.
        return None

    sign = -1 if m.group("sign") == "-" else 1

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

    return int(round(sign * total_seconds * 1_000_000_000))


def _is_chunked(array: Any) -> bool:
    return isinstance(array, pa.ChunkedArray)


# ---------------------------------------------------------------------------
# Timezone helpers — polars-first, pyarrow fallback
# ---------------------------------------------------------------------------
#
# pyarrow's tz kernels (``pc.assume_timezone``, ``pc.cast`` across tz-aware
# timestamp types) resolve zones against Arrow C++'s tz database. On Windows
# that database has to be downloaded manually and corporate proxies routinely
# block it, producing ``ArrowInvalid: Unable to get Timezone database version``.
# polars ships its own tz database via ``chrono-tz`` and doesn't care about
# the system state, so we route tz transitions through polars whenever it's
# importable and only fall back to pyarrow when polars isn't installed (or
# doesn't support the requested unit — polars has no "s" time unit).
#
# String parsing follows the same preference: polars uses chrono format
# strings which understand ``%.f`` for optional fractional seconds and ``%#z``
# for timezone offsets in either ``+0200`` or ``+02:00`` form. Arrow's
# strptime is built on the platform's C strptime kernel and silently rejects
# patterns that work on glibc but not on MSVCRT — the colon in ``+02:00``
# being the most common breakage point. Routing parse through polars makes
# behavior identical across Linux / macOS / Windows.


_POLARS_UNITS = frozenset({"ms", "us", "ns"})


def _polars_or_none():
    try:
        import polars as pl

        return pl
    except ImportError:
        return None


# Polars chrono parse formats — ``%#z`` matches ``+0200`` *and* ``+02:00``;
# ``%.f`` matches an optional ``.fraction`` segment so a single format covers
# both ``HH:MM:SS`` and ``HH:MM:SS.123456`` rows.
POLARS_PARSE_DATETIME_FORMATS_TZ: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S%.f%#z",
    "%Y-%m-%d %H:%M:%S%.f%#z",
    "%Y-%m-%dT%H:%M%#z",
    "%Y-%m-%d %H:%M%#z",
    "%Y-%m-%dT%H:%M:%S%.fZ",
    "%Y-%m-%d %H:%M:%S%.fZ",
    "%Y-%m-%dT%H:%MZ",
    "%Y-%m-%d %H:%MZ",
)

POLARS_PARSE_DATETIME_FORMATS_NAIVE: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
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
    "%Y%m%d",
    "%d %b %Y %H:%M:%S",
    "%d %b %Y",
    "%b %d, %Y",
    "%d.%m.%Y",
    "%Y-%m-%d",
)


def _polars_parse_str_to_timestamp(
    array: pa.Array,
    unit: str,
    tz: str | None,
    *,
    unsafe_tz: bool,
    keep_fractional: bool = True,
    formats_tz: tuple[str, ...] | None = None,
    formats_naive: tuple[str, ...] | None = None,
) -> pa.Array | None:
    """Parse a string array to ``timestamp[unit, tz]`` via polars.

    Returns ``None`` when polars is unavailable or the unit isn't one polars
    can store natively — the caller is expected to fall back to pyarrow's
    strptime path in that case.

    Semantics:

    * tz-aware source rows (``+02:00``, ``+0200``, ``Z``, trailing ``UTC``)
      are parsed into ``Datetime[unit, UTC]`` and converted to *tz* (or
      replaced with naive when *tz* is ``None``).
    * naive source rows are parsed into ``Datetime[unit]`` and either
      reinterpreted as *tz* wall-clock (``unsafe_tz=True``) or assumed UTC
      and shifted to *tz* (``unsafe_tz=False``).
    * Per-row coalesce — tz-aware result wins when both branches resolve
      the same row (mixed-shape columns Just Work).
    """
    pl = _polars_or_none()
    if pl is None or unit not in _POLARS_UNITS:
        return None

    fmt_tz = formats_tz if formats_tz is not None else POLARS_PARSE_DATETIME_FORMATS_TZ
    fmt_naive = (
        formats_naive if formats_naive is not None else POLARS_PARSE_DATETIME_FORMATS_NAIVE
    )

    try:
        series = pl.from_arrow(array)
        # ``pl.from_arrow`` returns ChunkedArray-equivalent for ChunkedArray
        # inputs; both flavors carry a ``.cast`` that lands on Series.
        if not isinstance(series, pl.Series):
            series = pl.Series(values=series)

        # Cast to string up front so binary / large-string sources still parse
        # without us having to branch on the Arrow type tag.
        col = pl.col("v").cast(pl.String, strict=False).str.strip_chars()
        # Trim blanks; an empty string can never satisfy any chrono format
        # but ``strptime`` raises rather than returning null on some inputs.
        col = (
            pl.when(col.str.len_chars() == 0)
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(col)
        )
        # ``" UTC"`` / ``"UTC"`` suffix → ``+0000`` so it falls through the
        # numeric-offset branch in chrono.
        col_norm = col.str.replace(r"\s*UTC$", "+0000")

        utc_dtype = pl.Datetime(unit, "UTC")
        bare_dtype = pl.Datetime(unit)

        tz_branches = [
            col_norm.str.strptime(
                utc_dtype, fmt, strict=False, ambiguous="earliest"
            )
            for fmt in fmt_tz
        ]
        naive_branches = [
            col.str.strptime(
                bare_dtype, fmt, strict=False, ambiguous="earliest"
            )
            for fmt in fmt_naive
        ]

        tz_resolved_utc = pl.coalesce(tz_branches) if tz_branches else None
        naive_resolved = pl.coalesce(naive_branches) if naive_branches else None

        if tz is None:
            # Output naive: tz-aware → drop UTC tag (already wall-clock UTC),
            # then coalesce with naive branch.
            parts = []
            if tz_resolved_utc is not None:
                parts.append(tz_resolved_utc.dt.replace_time_zone(None))
            if naive_resolved is not None:
                parts.append(naive_resolved)
            result = pl.coalesce(parts) if len(parts) > 1 else parts[0]
        else:
            # Output tz-aware: tz-aware branch converts to *tz*, naive branch
            # either reinterprets wall-clock as *tz* (unsafe_tz=True) or
            # assumes UTC then shifts (unsafe_tz=False). Both end up in the
            # same Datetime[unit, tz] dtype so coalesce can merge them.
            parts = []
            if tz_resolved_utc is not None:
                parts.append(tz_resolved_utc.dt.convert_time_zone(tz))
            if naive_resolved is not None:
                if unsafe_tz:
                    parts.append(
                        naive_resolved.dt.replace_time_zone(
                            tz, ambiguous="earliest"
                        )
                    )
                else:
                    parts.append(
                        naive_resolved.dt.replace_time_zone(
                            "UTC", ambiguous="earliest"
                        ).dt.convert_time_zone(tz)
                    )
            result = pl.coalesce(parts) if len(parts) > 1 else parts[0]

        if not keep_fractional:
            # Polars' ``%.f`` always pulls in the fractional segment. Truncate
            # back to whole-second precision when the caller explicitly opts
            # out — matches the pyarrow path's strip-don't-reattach branch.
            result = result.dt.truncate("1s")

        out = pl.DataFrame({"v": series}).select(result.alias("v")).to_series()
        return out.to_arrow()
    except Exception:
        # Anything polars surprises us with — fall through to the pyarrow
        # path rather than letting an internal hiccup nuke the whole parse.
        return None


def _pa_retimestamp(
    array: pa.Array, unit: str, tz: str | None, *, unsafe_tz: bool
) -> pa.Array:
    """Pure-pyarrow retimestamp — the fallback path, uses tz database."""
    src_tz = array.type.tz
    target = pa.timestamp(unit, tz)
    if src_tz is None and tz is not None and unsafe_tz:
        aligned = pc.cast(array, pa.timestamp(unit))
        stamped = pc.assume_timezone(aligned, timezone=tz)
        return stamped
    return pc.cast(array, target)


def retimestamp_prefer_polars(
    array: pa.Array,
    unit: str,
    tz: str | None,
    *,
    unsafe_tz: bool = True,
) -> pa.Array:
    """Cast a ``pa.timestamp`` array to ``pa.timestamp(unit, tz)``.

    Semantics match the pyarrow path:

    * naive → naive:  unit conversion only (no tz database lookup).
    * naive → tz-aware with ``unsafe_tz=True``: reinterpret the wall-clock
      time as living in ``tz`` (``"2023-01-02 03:04"`` Paris stays ``03:04``
      Paris).
    * naive → tz-aware with ``unsafe_tz=False``: treat the wall-clock as
      UTC, then shift to ``tz``.
    * tz-aware → tz-aware: convert to the target zone (same instant).
    * tz-aware → naive: shift to UTC and drop the zone tag.

    Prefers polars for the tz arithmetic because it bundles its own tz
    database; falls back to pyarrow's ``pc.cast`` /
    ``pc.assume_timezone`` when polars is not importable or when the
    requested unit is one polars does not support.
    """
    src_tz = array.type.tz

    # No tz on either side — pyarrow is fine, no tz database involved.
    if src_tz is None and tz is None:
        return pc.cast(array, pa.timestamp(unit))

    pl = _polars_or_none()
    if pl is None or unit not in _POLARS_UNITS:
        return _pa_retimestamp(array, unit, tz, unsafe_tz=unsafe_tz)

    try:
        s = pl.from_arrow(array)
        if s.dtype.time_unit != unit:
            s = s.cast(pl.Datetime(unit, s.dtype.time_zone))

        current_tz = s.dtype.time_zone  # type: ignore[attr-defined]

        if current_tz is None:
            assume = tz if unsafe_tz else "UTC"
            s = s.dt.replace_time_zone(assume, ambiguous="earliest")
            current_tz = assume

        if tz is None:
            # tz-aware → naive: convert to UTC, then drop the zone tag.
            s = s.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        elif current_tz != tz:
            s = s.dt.convert_time_zone(tz)

        return s.to_arrow()
    except Exception:
        # Any polars surprise (dtype mismatch, from_arrow quirk) — defer
        # to the pyarrow fallback rather than raising from the helper.
        return _pa_retimestamp(array, unit, tz, unsafe_tz=unsafe_tz)


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


_UNIT_DIGITS = {"s": 0, "ms": 3, "us": 6, "ns": 9}


def _strip_fractional_seconds(array: pa.Array) -> pa.Array:
    """Drop the fractional-second segment before strptime.

    Arrow's strptime is built on C ``strptime`` and doesn't understand ``%f``.
    The extracted fraction can be re-attached via
    :func:`attach_fractional_seconds` once the whole-second timestamp parses
    cleanly.
    """
    # Match a leading ``.`` plus up to 9 digits. Keep whatever follows (tz
    # offset, ``Z`` suffix, or end-of-string).
    return pc.replace_substring_regex(array, pattern=r"\.\d{1,9}", replacement="")


def _extract_fractional_as_duration(
    source_strs: pa.Array,
    unit: str,
) -> pa.Array:
    """Pull the ``.ddddd`` fragment out of a string column as a duration.

    Empty / missing fractions map to a zero-length duration so the caller can
    always ``pc.add`` the result to a parsed timestamp row-for-row.
    """
    digits = _UNIT_DIGITS.get(unit, 6)
    target = pa.duration(unit)

    if digits == 0:
        return pa.array([0] * len(source_strs), type=target)

    extracted = pc.extract_regex(source_strs, pattern=r"\.(?P<frac>\d+)")
    frac = extracted.field("frac")

    # Right-pad with zeros so ``.5`` becomes ``.500000`` (us) or ``.500000000``
    # (ns), then clip to the target precision — anything beyond is silently
    # truncated, matching Python's ``datetime`` behavior.
    padded = pc.utf8_rpad(frac, width=digits, padding="0")
    sliced = pc.utf8_slice_codeunits(padded, start=0, stop=digits)

    empty = pa.scalar("", type=pa.string())
    zero_str = pa.scalar("0", type=pa.string())
    non_empty = pc.if_else(pc.equal(sliced, empty), zero_str, sliced)
    as_int = pc.cast(non_empty, pa.int64(), safe=False)
    as_int = pc.fill_null(as_int, 0)
    return pc.cast(as_int, target)


def attach_fractional_seconds(
    timestamps: pa.Array,
    source_strs: pa.Array,
    unit: str,
) -> pa.Array:
    """Fold fractional-second precision from *source_strs* back onto *timestamps*.

    Timestamps that parsed to null stay null. Rows without a fractional
    segment get a zero offset, so this is always safe to chain after
    :func:`_coalesce_strptime`.
    """
    frac_dur = _extract_fractional_as_duration(source_strs, unit=unit)
    return pc.add(timestamps, frac_dur)


def _normalize_utc_suffix(array: pa.Array) -> pa.Array:
    """Rewrite trailing named-UTC tags so the ``%z`` strptime branch picks them up.

    Real-world columns love suffixes like ``" UTC"`` or ``"UTC"`` that Arrow's
    strptime can't map to a timezone. Normalizing to ``+0000`` lets the
    tz-aware format catalogue handle them identically to ``Z``.
    """
    # Trailing "UTC" with optional whitespace. Handles both " UTC" (space-
    # separated, common in ISO-ish exports) and stuck-on "UTC". We collapse
    # to the colon-less form so the next normalization step doesn't have to
    # touch this row again, and so Arrow's strptime ``%z`` accepts it on
    # every platform.
    return pc.replace_substring_regex(array, pattern=r"\s*UTC$", replacement="+0000")


def _normalize_utc_offset(array: pa.Array) -> pa.Array:
    """Strip the colon out of trailing ``+HH:MM`` / ``-HH:MM`` offsets.

    Arrow's strptime is built on the platform's C ``strptime``. glibc accepts
    both ``+0200`` and ``+02:00`` for ``%z``; Windows MSVCRT (and several BSD
    libc variants) only accept the colon-less form. Normalizing here keeps
    the cross-platform parse behavior identical regardless of the host.
    """
    return pc.replace_substring_regex(
        array,
        pattern=r"([+-])(\d{2}):(\d{2})$",
        replacement=r"\1\2\3",
    )


def arrow_str_to_timestamp(
    array: pa.Array | pa.ChunkedArray,
    unit: str = "us",
    tz: str | None = None,
    *,
    keep_fractional: bool = True,
    unsafe_tz: bool = True,
    formats: tuple[str, ...] | list[str] | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Best-effort parse a string array to ``timestamp[unit, tz]``.

    Parameters
    ----------
    keep_fractional:
        Polars' ``%.f`` parses sub-second precision inline, so the flag
        normally has no effect on the polars path. The pyarrow fallback —
        used when polars isn't installed or the unit is one polars doesn't
        support (``"s"``) — still strips and re-attaches the fraction
        explicitly; setting ``keep_fractional=False`` drops the re-attach.
    unsafe_tz:
        Controls what happens when *array* has no recorded timezone but *tz*
        is set. ``True`` (default, best-effort) reinterprets the wall-clock
        time as the target zone — ``"2023-01-02 03:04"`` in ``tz="Europe/Paris"``
        stays ``03:04`` Paris local. ``False`` (strict) assumes UTC and
        shifts the display to the target zone.
    formats:
        Optional override of the format catalogue. The polars path treats
        *formats* as tz-aware vs naive based on the presence of a ``%z`` /
        ``%#z`` / ``%:z`` directive (or a literal ``Z``). The pyarrow
        fallback tries the formats in the order supplied.
    """

    def _via_polars(chunk: pa.Array) -> pa.Array | None:
        if formats is not None:
            tz_fmts = tuple(
                f for f in formats if "%z" in f or "%:z" in f or "%#z" in f or f.endswith("Z")
            )
            naive_fmts = tuple(f for f in formats if f not in tz_fmts)
        else:
            tz_fmts = None
            naive_fmts = None
        return _polars_parse_str_to_timestamp(
            chunk,
            unit=unit,
            tz=tz,
            unsafe_tz=unsafe_tz,
            keep_fractional=keep_fractional,
            formats_tz=tz_fmts,
            formats_naive=naive_fmts,
        )

    def _via_arrow(chunk: pa.Array) -> pa.Array:
        chunk = _ensure_string(chunk)
        chunk = nullify_empty_strings(chunk)
        normalized = _normalize_utc_suffix(chunk)
        normalized = _normalize_utc_offset(normalized)
        stripped = _strip_fractional_seconds(normalized)

        chosen = (
            tuple(formats)
            if formats is not None
            else ARROW_DATETIME_FORMATS_TZ + ARROW_DATETIME_FORMATS_NAIVE
        )
        naive = _coalesce_strptime(stripped, chosen, unit=unit, strip_tz=True)

        if keep_fractional:
            naive = attach_fractional_seconds(naive, normalized, unit=unit)

        if tz:
            return retimestamp_prefer_polars(
                naive, unit=unit, tz=tz, unsafe_tz=unsafe_tz,
            )
        return naive

    def _one(chunk: pa.Array) -> pa.Array:
        chunk = _ensure_string(chunk)
        chunk = nullify_empty_strings(chunk)
        out = _via_polars(chunk)
        if out is not None:
            return out
        return _via_arrow(chunk)

    return _apply_chunked(array, _one)


def arrow_str_to_date(
    array: pa.Array | pa.ChunkedArray,
    *,
    formats: tuple[str, ...] | list[str] | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Best-effort parse a string array to ``date32``.

    Reuses :func:`arrow_str_to_timestamp` (polars-first) so date columns
    accept the same shapes — full ISO datetimes with offsets, ``Z`` suffix,
    ``" UTC"`` tail, naive minute precision — and the result is cast to
    ``date32``.
    """

    def _one(chunk: pa.Array) -> pa.Array:
        chunk = _ensure_string(chunk)
        chunk = nullify_empty_strings(chunk)

        ts = arrow_str_to_timestamp(
            chunk, unit="us", tz=None, formats=formats, keep_fractional=False
        )
        return pc.cast(ts, pa.date32())

    return _apply_chunked(array, _one)


def arrow_str_to_time(
    array: pa.Array | pa.ChunkedArray,
    unit: str = "us",
    *,
    keep_fractional: bool = True,
    formats: tuple[str, ...] | list[str] | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Best-effort parse a string array to ``time32/time64``.

    Arrow's strptime cannot parse a bare ``HH:MM:SS`` — it always wants a date
    component. We prepend a placeholder date before parsing, then cast the
    resulting timestamp back to ``time``. Sub-second precision is extracted
    separately and folded back on when ``keep_fractional=True``.
    """
    time_type = pa.time64(unit) if unit in {"us", "ns"} else pa.time32(unit)

    def _one(chunk: pa.Array) -> pa.Array:
        chunk = _ensure_string(chunk)
        chunk = nullify_empty_strings(chunk)
        stripped = _strip_fractional_seconds(chunk)

        # Prepend a fixed date so strptime treats the input as a full timestamp.
        prefix = pa.scalar("1970-01-01T", type=stripped.type)
        prefixed = pc.binary_join_element_wise(prefix, stripped, "")

        base_formats = tuple(formats) if formats is not None else ARROW_TIME_FORMATS
        full_formats = tuple(f"%Y-%m-%dT{fmt}" for fmt in base_formats)
        parsed = _coalesce_strptime(prefixed, full_formats, unit=unit)

        if keep_fractional:
            parsed = attach_fractional_seconds(parsed, chunk, unit=unit)

        return pc.cast(parsed, time_type)

    return _apply_chunked(array, _one)


def arrow_str_to_duration(
    array: pa.Array | pa.ChunkedArray,
    unit: str = "us",
) -> pa.Array | pa.ChunkedArray:
    """Parse a string array into ``duration[unit]``.

    Recognises three shapes per row:

    * Plain integer (``"60"``) — interpreted as a count in *unit*. This is
      the historical behaviour; pyarrow handles it vectorized when the whole
      column conforms.
    * Plain decimal (``"1.5"``) — interpreted as a float count in *unit*,
      rounded to the nearest integer.
    * ISO 8601 duration (``"PT15M"``, ``"P1D"``, ``"P1Y2M3DT4H5.5S"``,
      ``"-PT30S"``) — calendar fields collapse to whole days using the
      module-level ``ISO_DURATION_DAYS_PER_{YEAR,MONTH,WEEK}`` defaults.
      See :func:`parse_iso_duration_to_nanos` for the exact contract.

    Rows that match none of the above are nulled.
    """
    target = pa.duration(unit)
    nanos_per_unit = 1_000_000_000 // _TIME_UNIT_PER_SECOND[unit]

    def _one(chunk: pa.Array) -> pa.Array:
        chunk = _ensure_string(chunk)
        chunk = nullify_empty_strings(chunk)

        # Fast path: every non-null entry is a plain integer literal. This is
        # the common bulk-ingest shape and stays fully vectorized.
        try:
            as_int = pc.cast(chunk, pa.int64(), safe=True)
            return pc.cast(as_int, target)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            pass

        # Mixed / non-numeric column: fall back to per-row parsing. ISO 8601
        # durations are short and the regex is anchored, so this stays cheap
        # in practice — and it's the only way to honour the calendar-unit
        # defaults without a vectorized regex kernel.  Dictionary-encode so
        # each distinct string is parsed once even when the column has many
        # repeated values (common in schedules / bucketed durations).
        if len(chunk) == 0:
            return pa.array([], type=target)

        encoded = pc.dictionary_encode(chunk)
        unique_raw = encoded.dictionary.to_pylist()

        def _parse_one(raw: str | None) -> int | None:
            if raw is None:
                return None
            s = raw.strip()
            if not s:
                return None
            if _RE_INT_LITERAL.match(s):
                return int(s)
            iso_nanos = parse_iso_duration_to_nanos(s)
            if iso_nanos is not None:
                return iso_nanos // nanos_per_unit
            try:
                return int(round(float(s)))
            except ValueError:
                return None

        unique_units = [_parse_one(raw) for raw in unique_raw]
        unique_arr = pa.array(unique_units, type=pa.int64())
        as_int = pc.take(unique_arr, encoded.indices)
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
    *,
    keep_fractional: bool = True,
    unsafe_tz: bool = True,
) -> pa.Array | pa.ChunkedArray:
    """Dispatch any source Arrow type to a ``timestamp[unit, tz]`` array.

    See :func:`arrow_str_to_timestamp` for ``keep_fractional`` and
    ``unsafe_tz`` semantics. They only matter on the string branch and on
    naive→tz-aware transitions.
    """
    src = array.type
    target = pa.timestamp(unit, tz)

    if pa.types.is_timestamp(src):
        return retimestamp_prefer_polars(
            array, unit=unit, tz=tz, unsafe_tz=unsafe_tz,
        )

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
        return arrow_str_to_timestamp(
            array,
            unit=unit,
            tz=tz,
            keep_fractional=keep_fractional,
            unsafe_tz=unsafe_tz,
        )

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
    *,
    keep_fractional: bool = True,
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
        return arrow_str_to_time(array, unit=unit, keep_fractional=keep_fractional)

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


def _polars_format_temporal(
    array: pa.Array,
    fmt: str | None,
    kind: str,
) -> pa.Array | None:
    """Format a temporal arrow array as strings via polars chrono.

    Returns ``None`` when polars is unavailable or the source dtype isn't
    one polars stringifies cleanly — the caller falls back to pyarrow's
    ``pc.strftime`` in that case. polars' ``%.f`` only emits a fractional
    suffix when the value carries one, which matches the chrono parser on
    the way back in.
    """
    pl = _polars_or_none()
    if pl is None:
        return None

    if kind == "timestamp":
        src_tz = array.type.tz
        default = (
            "%Y-%m-%dT%H:%M:%S%.f%:z" if src_tz else "%Y-%m-%dT%H:%M:%S%.f"
        )
    elif kind == "date":
        default = "%Y-%m-%d"
    elif kind == "time":
        # Match the historical pyarrow ``cast(time, string)`` shape: always
        # emit six fractional digits so the output round-trips against
        # ``arrow_str_to_time`` regardless of source precision.
        default = "%H:%M:%S%.6f"
    else:
        return None

    try:
        series = pl.from_arrow(array)
        if not isinstance(series, pl.Series):
            series = pl.Series(values=series)
        out = series.dt.to_string(fmt or default)
        return out.to_arrow()
    except Exception:
        return None


def arrow_timestamp_to_string(
    array: pa.Array | pa.ChunkedArray,
    fmt: str | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Format a timestamp array as strings. Default is ISO-8601.

    Polars chrono is the primary path so timezone offsets (``%:z`` →
    ``+02:00``) and fractional seconds (``%.f``, only emitted when the
    value carries them) render the same way on every platform. pyarrow's
    ``pc.strftime`` is the fallback for environments without polars; it
    emits subseconds inline through ``%S`` so the visual shape stays close.
    """

    def _one(chunk: pa.Array) -> pa.Array:
        out = _polars_format_temporal(chunk, fmt, kind="timestamp")
        if out is not None:
            return out
        src = chunk.type
        default_fmt = "%Y-%m-%dT%H:%M:%S%z" if src.tz else "%Y-%m-%dT%H:%M:%S"
        return pc.strftime(chunk, format=fmt or default_fmt)

    return _apply_chunked(array, _one)


def arrow_date_to_string(
    array: pa.Array | pa.ChunkedArray,
    fmt: str | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Format a date array as strings. Default is ``YYYY-MM-DD``."""

    def _one(chunk: pa.Array) -> pa.Array:
        out = _polars_format_temporal(chunk, fmt, kind="date")
        if out is not None:
            return out
        return pc.strftime(chunk, format=fmt or "%Y-%m-%d")

    return _apply_chunked(array, _one)


def arrow_time_to_string(
    array: pa.Array | pa.ChunkedArray,
    fmt: str | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Format a time array as strings.

    Default emits ``HH:MM:SS.ffffff`` (six fractional digits, zero-padded)
    so the result round-trips through :func:`arrow_str_to_time` regardless
    of source precision. Polars handles ``time`` directly; the pyarrow
    fallback bounces through a 1970-01-01 timestamp because ``pc.strftime``
    refuses ``time`` inputs.
    """

    def _one(chunk: pa.Array) -> pa.Array:
        out = _polars_format_temporal(chunk, fmt, kind="time")
        if out is not None:
            return out
        src = chunk.type
        src_unit = src.unit
        as_int = pc.cast(chunk, pa.int64())
        as_dur = pc.cast(as_int, pa.duration(src_unit))
        epoch = pa.scalar(0, type=pa.timestamp(src_unit))
        ts = pc.add(epoch, as_dur)
        if fmt is None:
            return pc.strftime(ts, format="%H:%M:%S")
        return pc.strftime(ts, format=fmt)

    return _apply_chunked(array, _one)


def arrow_duration_to_string(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Render a duration as a plain integer string in the source unit.

    Round-trips through :func:`arrow_str_to_duration` without precision loss,
    which is what yggdrasil cares about more than human-friendly formatting.
    """

    def _one(chunk: pa.Array) -> pa.Array:
        return pc.cast(pc.cast(chunk, pa.int64()), pa.string())

    return _apply_chunked(array, _one)


def arrow_cast_to_string(
    array: pa.Array | pa.ChunkedArray,
    fmt: str | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Dispatch any source Arrow type to a ``string`` array.

    The default stringification for non-temporal types is ``pc.cast(arr,
    string())``. Temporal types get the custom helpers above so duration
    works at all and timestamp/date formats can be tuned.
    """
    src = array.type

    if pa.types.is_date(src):
        return arrow_date_to_string(array, fmt=fmt)

    if pa.types.is_time(src):
        return arrow_time_to_string(array, fmt=fmt)

    if pa.types.is_timestamp(src):
        return arrow_timestamp_to_string(array, fmt=fmt)

    if pa.types.is_duration(src):
        return arrow_duration_to_string(array)

    if pa.types.is_null(src):
        return pa.nulls(len(array), type=pa.string())

    return pc.cast(array, pa.string())


def arrow_temporal_to_string(
    array: pa.Array | pa.ChunkedArray,
    fmt: str | None = None,
) -> pa.Array | pa.ChunkedArray:
    """Back-compat alias for :func:`arrow_cast_to_string` on temporal inputs."""
    return arrow_cast_to_string(array, fmt=fmt)


# ---------------------------------------------------------------------------
# Polars helpers
# ---------------------------------------------------------------------------
#
# Polars uses chrono format strings (https://docs.rs/chrono/latest/chrono/format/strftime/),
# which overlap with C strptime but diverge for sub-second precision: ``%.f``
# matches an optional fractional-second segment, while ``%f`` is a fixed-width
# microseconds field. Polars' multi-format coalesce path leans on ``%.f`` so a
# single format covers both ``HH:MM:SS`` and ``HH:MM:SS.123456`` rows.

# Formats with an embedded UTC offset — strptime returns ``Datetime[tu, UTC]``.
# We parse those separately from naive formats because polars'
# ``pl.coalesce([naive, tz_aware])`` raises ``SchemaError`` when the branches
# disagree on tz state. The tz branch gets stripped to naive before merging.
POLARS_DATETIME_FORMATS_TZ: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S%.f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%.f%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M%z",
)

POLARS_DATETIME_FORMATS_NAIVE: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%.f",
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
    "%Y%m%d",
    "%d %b %Y %H:%M:%S",
    "%d %b %Y",
    "%b %d, %Y",
    "%Y-%m-%d",
)

POLARS_DATE_FORMATS: tuple[str, ...] = (
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

POLARS_TIME_FORMATS: tuple[str, ...] = (
    "%H:%M:%S%.f",
    "%H:%M:%S",
    "%H:%M",
    "%I:%M:%S %p",
    "%I:%M %p",
)


def _polars_numeric_classes() -> tuple:
    pl = get_polars()
    return (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    )


def _polars_dtype_class(dtype: Any) -> Any:
    """Return the polars dtype *class* whether *dtype* is a class or instance."""
    return dtype if isinstance(dtype, type) else dtype.__class__


def _polars_safe_strptime(expr: Any, dtype: Any, fmt: str) -> Any:
    """Single strptime attempt that yields null on mismatch (strict=False)."""
    return expr.str.strptime(dtype, fmt, strict=False, ambiguous="earliest")


def _polars_coalesce_strptime(
    expr: Any, dtype: Any, formats: tuple[str, ...] | list[str]
) -> Any:
    """Try each format in order; first non-null result wins (per row)."""
    pl = get_polars()
    return pl.coalesce([_polars_safe_strptime(expr, dtype, fmt) for fmt in formats])


def _polars_normalize_utc_suffix(expr: Any) -> Any:
    """Rewrite trailing ``UTC`` tags to ``+00:00`` so ``%z`` formats catch them.

    Mirrors :func:`_normalize_utc_suffix` on the arrow side so a single column
    can mix ``Z``, ``+0000``, and ``"UTC"`` suffixes and parse via the same
    tz-aware format catalogue.
    """
    return expr.str.replace(r"\s*UTC$", "+00:00")


def _polars_normalize_utc_offset(expr: Any) -> Any:
    """Rewrite ``+01:00`` style offsets to ``+0100`` so ``%z`` matches.

    Polars' chrono ``%z`` accepts ``+0100`` but not ``+01:00`` — strip the
    colon up front so both shapes parse identically.
    """
    return expr.str.replace(r"([+-])(\d{2}):(\d{2})$", r"${1}${2}${3}")


def _polars_string_to_datetime(expr: Any, target: Any) -> Any:
    """Best-effort string→Datetime parse using the polars format catalogues."""
    pl = get_polars()
    tu = getattr(target, "time_unit", "us") or "us"
    bare = pl.Datetime(tu)
    utc = pl.Datetime(tu, "UTC")

    normalized = _polars_normalize_utc_suffix(expr)
    normalized = _polars_normalize_utc_offset(normalized)

    tz_parsed = [
        _polars_safe_strptime(normalized, utc, fmt)
        .dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
        for fmt in POLARS_DATETIME_FORMATS_TZ
    ]
    naive_parsed = [
        _polars_safe_strptime(expr, bare, fmt) for fmt in POLARS_DATETIME_FORMATS_NAIVE
    ]
    return pl.coalesce(tz_parsed + naive_parsed)


def _polars_string_to_date(expr: Any) -> Any:
    pl = get_polars()
    return _polars_coalesce_strptime(expr, pl.Date(), POLARS_DATE_FORMATS)


def _polars_string_to_time(expr: Any) -> Any:
    pl = get_polars()
    return _polars_coalesce_strptime(expr, pl.Time(), POLARS_TIME_FORMATS)


def _polars_string_to_duration(expr: Any, target: Any) -> Any:
    """Interpret integer / decimal strings as a count in the target unit.

    Mirrors the arrow ``arrow_str_to_duration`` integer fast-path so polars
    columns of plain numeric strings ingest without falling through to the
    pure-Python ISO-8601 branch on every value.
    """
    pl = get_polars()
    tu = getattr(target, "time_unit", "us") or "us"
    return expr.cast(pl.Int64, strict=False).cast(pl.Duration(tu), strict=False)


def _polars_apply_tz(expr: Any, src_tz: str | None, tgt_tz: str | None) -> Any:
    """Transition a Datetime expression from *src_tz* to *tgt_tz*.

    Mirrors the pyarrow ``retimestamp_prefer_polars`` semantics on the polars
    side so naive ↔ aware transitions behave the same regardless of which
    engine the caller routes through.
    """
    if src_tz is None and tgt_tz is None:
        return expr
    if src_tz is None:
        return expr.dt.replace_time_zone(tgt_tz, ambiguous="earliest")
    if tgt_tz is None:
        return (
            expr.dt.replace_time_zone(src_tz, ambiguous="earliest")
            .dt.convert_time_zone("UTC")
            .dt.replace_time_zone(None)
        )
    if src_tz == tgt_tz:
        return expr.dt.replace_time_zone(src_tz, ambiguous="earliest")
    return (
        expr.dt.replace_time_zone(src_tz, ambiguous="earliest")
        .dt.convert_time_zone(tgt_tz)
    )


def _eval_expr_on_series(series: Any, expr: Any, *, col_name: str, out_name: str) -> Any:
    """Evaluate *expr* against *series*, preserving the original name."""
    out = series.to_frame(name=col_name).select(expr).to_series()
    return out.rename(out_name) if out_name else out


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
    """Cast *array* to a temporal Polars dtype with full timezone handling.

    Accepts a ``pl.Series`` or a ``pl.Expr`` and dispatches by the Polars
    dtype class of *target*. ``safe=False`` (the default) enables the
    multi-format string coalesce path; ``safe=True`` falls back to a single
    strict polars cast, leaving rows that don't match the canonical form as
    nulls / errors per polars' default behavior.
    """
    pl = get_polars()
    is_expr = isinstance(array, pl.Expr)
    series_name = "" if is_expr else (array.name or "")

    if parent_name and series_name:
        col_name = f"{parent_name}.{series_name}"
    else:
        col_name = series_name or parent_name or "__col__"

    working: Any = array if is_expr else pl.col(col_name)

    src_cls = _polars_dtype_class(source)
    tgt_cls = _polars_dtype_class(target)

    # Pull tz off either the dtype itself (when the caller passed a dtype
    # *instance*) or from the explicit source_tz / target_tz kwargs.
    src_tz = (
        getattr(source, "time_zone", None) if not isinstance(source, type) else None
    ) or source_tz
    tgt_tz = (
        getattr(target, "time_zone", None) if not isinstance(target, type) else None
    ) or target_tz

    string_classes = (pl.String, pl.Utf8)
    numeric_classes = _polars_numeric_classes()

    if tgt_cls is pl.Datetime:
        tu = getattr(target, "time_unit", "us") or "us"

        if src_cls is pl.Datetime:
            working = working.cast(pl.Datetime(tu), strict=safe)
            working = _polars_apply_tz(working, src_tz, tgt_tz)
        elif src_cls in string_classes:
            if safe:
                working = working.str.strptime(
                    pl.Datetime(tu), strict=True, ambiguous="earliest"
                )
            else:
                working = _polars_string_to_datetime(working, pl.Datetime(tu, tgt_tz))
            working = _polars_apply_tz(working, src_tz, tgt_tz)
        elif src_cls is pl.Date:
            working = working.cast(pl.Datetime(tu), strict=safe)
            working = _polars_apply_tz(working, None, tgt_tz)
        elif src_cls is pl.Time:
            epoch = pl.lit(0).cast(pl.Datetime(tu))
            working = epoch + working.cast(pl.Duration(tu), strict=safe)
            working = _polars_apply_tz(working, None, tgt_tz)
        elif src_cls is pl.Duration:
            epoch = pl.lit(0).cast(pl.Datetime(tu))
            working = epoch + working.cast(pl.Duration(tu), strict=safe)
            working = _polars_apply_tz(working, None, tgt_tz)
        elif src_cls in numeric_classes:
            working = working.cast(pl.Int64, strict=safe).cast(
                pl.Datetime(tu), strict=safe
            )
            working = _polars_apply_tz(working, src_tz, tgt_tz)
        else:
            working = working.cast(pl.Datetime(tu, tgt_tz), strict=safe)

    elif tgt_cls is pl.Date:
        if src_cls is pl.Date:
            pass
        elif src_cls in string_classes:
            if safe:
                working = working.str.strptime(
                    pl.Date(), strict=True, ambiguous="earliest"
                )
            else:
                # Cover full-datetime strings too — users routinely hand date
                # columns ISO timestamps and expect just the date portion.
                normalized = _polars_normalize_utc_suffix(working)
                normalized = _polars_normalize_utc_offset(normalized)
                date_branch = _polars_coalesce_strptime(
                    working, pl.Date(), POLARS_DATE_FORMATS
                )
                bare_dt = pl.Datetime("us")
                utc_dt = pl.Datetime("us", "UTC")
                tz_branches = [
                    _polars_safe_strptime(normalized, utc_dt, fmt)
                    .dt.convert_time_zone("UTC")
                    .dt.replace_time_zone(None)
                    .dt.date()
                    for fmt in POLARS_DATETIME_FORMATS_TZ
                ]
                naive_branches = [
                    _polars_safe_strptime(working, bare_dt, fmt).dt.date()
                    for fmt in POLARS_DATETIME_FORMATS_NAIVE
                ]
                working = pl.coalesce([date_branch, *tz_branches, *naive_branches])
        elif src_cls is pl.Datetime:
            if src_tz:
                working = working.dt.replace_time_zone(src_tz, ambiguous="earliest")
            if tgt_tz and tgt_tz != src_tz:
                working = working.dt.convert_time_zone(tgt_tz)
            working = working.dt.date()
        elif src_cls in numeric_classes:
            working = working.cast(pl.Int32, strict=safe).cast(pl.Date, strict=safe)
        else:
            working = working.cast(pl.Date, strict=safe)

    elif tgt_cls is pl.Time:
        if src_cls is pl.Time:
            pass
        elif src_cls in string_classes:
            if safe:
                working = working.str.strptime(
                    pl.Time(), strict=True, ambiguous="earliest"
                )
            else:
                working = _polars_string_to_time(working)
        elif src_cls is pl.Datetime:
            if src_tz:
                working = working.dt.replace_time_zone(src_tz, ambiguous="earliest")
            if tgt_tz and tgt_tz != src_tz:
                working = working.dt.convert_time_zone(tgt_tz)
            working = working.dt.time()
        elif src_cls is pl.Duration:
            working = working.cast(pl.Int64, strict=safe).cast(pl.Time, strict=safe)
        elif src_cls in numeric_classes:
            working = working.cast(pl.Int64, strict=safe).cast(pl.Time, strict=safe)
        else:
            working = working.cast(pl.Time, strict=safe)

    elif tgt_cls is pl.Duration:
        tu = getattr(target, "time_unit", "us") or "us"

        if src_cls is pl.Duration:
            working = working.cast(pl.Duration(tu), strict=safe)
        elif src_cls in string_classes:
            if safe:
                # Polars has no safe string→Duration kernel; fall back to the
                # integer-string fast path so callers still get something sane.
                working = working.cast(pl.Int64, strict=True).cast(
                    pl.Duration(tu), strict=True
                )
            else:
                working = _polars_string_to_duration(working, pl.Duration(tu))
        elif src_cls in numeric_classes:
            working = working.cast(pl.Int64, strict=safe).cast(
                pl.Duration(tu), strict=safe
            )
        elif src_cls is pl.Datetime:
            if src_tz:
                working = (
                    working.dt.replace_time_zone(src_tz, ambiguous="earliest")
                    .dt.convert_time_zone("UTC")
                    .dt.replace_time_zone(None)
                )
            epoch = pl.lit(0).cast(pl.Datetime(tu))
            working = working.cast(pl.Datetime(tu), strict=safe) - epoch
        elif src_cls is pl.Date:
            epoch = pl.lit(0).cast(pl.Date)
            working = (working - epoch).cast(pl.Duration(tu), strict=safe)
        else:
            working = working.cast(pl.Duration(tu), strict=safe)

    else:
        raise TypeError(f"Unsupported temporal target type: {target!r}")

    if series_name:
        working = working.alias(series_name)

    if is_expr or to_expr:
        return working

    return _eval_expr_on_series(
        array, working, col_name=col_name, out_name=series_name
    )


# ---------------------------------------------------------------------------
# Spark helpers
# ---------------------------------------------------------------------------


def _spark_when(F: Any) -> Any:
    return F


def spark_to_timestamp(
    column: "ps.Column",
    unit: str = "us",
    tz: str | None = None,
    *,
    unsafe_tz: bool = True,
) -> "ps.Column":
    """Best-effort cast a Spark column to a timestamp.

    ``unsafe_tz=True`` treats a tz-naive source's wall-clock as already
    belonging to *tz* — the opposite of Spark's default UTC assumption on
    ``to_timestamp``. Only matters when the source has no recorded zone.
    """
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

    if unsafe_tz and source_type_name in {None, "timestamp_ntz", "string"}:
        # Reinterpret naive/ambiguous wall-clock as *tz* instead of UTC.
        # Spark stores timestamps as instants; to_utc_timestamp treats the
        # input as *tz* local and returns the equivalent UTC instant —
        # exactly the "no display shift" behavior callers want here.
        return F.to_utc_timestamp(casted, tz)

    return casted


def spark_temporal_to_string(
    column: "ps.Column",
    fmt: str | None = None,
) -> "ps.Column":
    """Render a temporal Spark column as a string using Java ``date_format``.

    Source types without a schema (in-expression columns) fall back to a
    plain ``cast(string)``.
    """
    spark = get_spark_sql()
    F = spark.functions
    T = spark.types

    try:
        source_type_name = column._jc.expr().dataType().typeName()  # type: ignore[attr-defined]
    except Exception:
        source_type_name = None

    if source_type_name == "date":
        return F.date_format(column, fmt or "yyyy-MM-dd")

    if source_type_name in {"timestamp", "timestamp_ntz"}:
        return F.date_format(column, fmt or "yyyy-MM-dd'T'HH:mm:ss.SSS")

    return column.cast(T.StringType())


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
