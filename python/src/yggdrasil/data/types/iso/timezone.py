"""IANA timezone type.

Accepts IANA timezone identifiers (``UTC``, ``Europe/Paris``,
``America/New_York``, …), common deprecated aliases and legacy
abbreviation-only zones (``CET``, ``US/Eastern``, ``Asia/Calcutta``),
and fixed UTC offset strings (``+05:00``, ``-08:00``, ``UTC+5``,
``GMT-0530``, ``Z``).  Inputs are normalized to either the current
canonical IANA name or an ISO 8601 ``±HH:MM`` offset string.

The legacy abbreviation zones (``CET``, ``EET``, ``MET``, ``WET``,
``EST``, ``HST``, ``MST``, ``CST6CDT`` …) are intentionally **not**
treated as canonical — they get rewritten to a representative
Area/Location zone (``CET`` -> ``Europe/Paris``, ``MST`` ->
``America/Phoenix`` …).

Cross-type casting is registered with :mod:`yggdrasil.data.cast`: a
:class:`datetime.tzinfo` (including :class:`zoneinfo.ZoneInfo`) or a
:class:`datetime.timedelta` converts to the canonical timezone
string, and the string converts back to ``tzinfo``/``timedelta``
through the existing :mod:`yggdrasil.data.cast.datetime` helpers.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.cast.registry import register_converter

from .base import ISOType
from .data.timezones import TIMEZONES, TIMEZONE_ALIASES
from ..id import DataTypeId

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import polars
    from ..base import DataType
    from yggdrasil.data.cast.options import CastOptions


# Arrow type-id families used by the outgoing-cast hook.
_TZ_NUMERIC_TARGETS = frozenset({DataTypeId.INTEGER, DataTypeId.FLOAT, DataTypeId.DECIMAL})

__all__ = ["TimezoneType"]


# Accepts  "+5", "+05", "+05:00", "+0500", "UTC+5", "UTC+05:30", "GMT-08:00"  etc.
_OFFSET_RE = re.compile(
    r"^(?:UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?$",
    re.IGNORECASE,
)


def _normalize_key(value: str) -> str:
    """Uppercase and structure a timezone-like string for lookup.

    Preserves IANA separators (``/``, ``_``, ``-``, ``+``). Windows-style
    backslashes become forward slashes; interior whitespace is collapsed
    onto the IANA conventions (``/`` between area/location, ``_`` inside
    location segments).
    """
    text = value.replace("\\", "/").strip()
    if not text:
        return ""

    while " /" in text or "/ " in text:
        text = text.replace(" /", "/").replace("/ ", "/")

    segments = text.split("/")
    segments = ["_".join(seg.split()) for seg in segments]
    text = "/".join(segments)

    return text.upper()


def _parse_offset_token(token: str) -> str | None:
    """Parse a fixed UTC-offset string, returning the canonical ``±HH:MM`` form.

    Returns ``None`` if *token* doesn't look like an offset at all.
    """
    stripped = token.strip()
    if not stripped:
        return None

    # Bare Z is UTC.
    if stripped.upper() == "Z":
        return "+00:00"

    compact = stripped.replace(" ", "")
    match = _OFFSET_RE.match(compact)
    if match is None:
        return None

    sign, hh, mm = match.group(1), match.group(2), match.group(3)
    hours = int(hh)
    minutes = int(mm) if mm else 0

    if hours > 23 or minutes > 59:
        return None

    # Reject absurd offsets (IANA uses -12..+14; we accept up to ±23 for
    # robustness but bar anything beyond a legal day offset).
    if hours * 60 + minutes >= 24 * 60:
        return None

    return f"{sign}{hours:02d}:{minutes:02d}"


def _build_timezone_map() -> dict[str, str]:
    """NORMALIZED token -> canonical IANA name."""
    mapping: dict[str, str] = {}

    for name in TIMEZONES:
        mapping[_normalize_key(name)] = name

    for alias, target in TIMEZONE_ALIASES.items():
        mapping.setdefault(_normalize_key(alias), target)

    return mapping


_TIMEZONE_MAP: dict[str, str] = _build_timezone_map()
_VALID_NAMES: frozenset[str] = frozenset(TIMEZONES)


def _tzinfo_to_timezone_string(value: dt.tzinfo) -> str:
    """Canonical timezone string for a :class:`datetime.tzinfo`.

    - :class:`zoneinfo.ZoneInfo` → its ``key`` (resolved through aliases).
    - :class:`datetime.timezone` → ``±HH:MM`` (``+00:00`` for UTC).
    - Anything else — falls back to the current UTC offset if it can be
      computed, otherwise the tz's ``str()``.
    """
    if ZoneInfo is not None and isinstance(value, ZoneInfo):
        key = getattr(value, "key", None)
        if key:
            resolved = _TIMEZONE_MAP.get(_normalize_key(key))
            return resolved if resolved is not None else key

    if isinstance(value, dt.timezone):
        offset = value.utcoffset(None) or dt.timedelta(0)
        return _timedelta_to_offset_string(offset)

    # Generic tzinfo: use its current utcoffset at "now" as the best guess.
    try:
        offset = value.utcoffset(dt.datetime.now(tz=dt.timezone.utc).replace(tzinfo=None))
    except Exception:
        offset = None
    if offset is not None:
        return _timedelta_to_offset_string(offset)

    return str(value)


def _timedelta_to_offset_string(value: dt.timedelta) -> str:
    """Canonical ``±HH:MM`` form for a :class:`datetime.timedelta` offset."""
    total_minutes = int(round(value.total_seconds() / 60))
    sign = "-" if total_minutes < 0 else "+"
    total_minutes = abs(total_minutes)
    hours, minutes = divmod(total_minutes, 60)
    if hours >= 24:
        raise ValueError(
            f"Timedelta {value!r} is out of range for a UTC offset (±24h)."
        )
    return f"{sign}{hours:02d}:{minutes:02d}"


def _timezone_string_to_tzinfo(value: str) -> dt.tzinfo:
    """Canonical timezone string -> :class:`datetime.tzinfo`."""
    offset = _parse_offset_token(value)
    if offset is not None:
        sign = 1 if offset.startswith("+") else -1
        hh, mm = offset[1:].split(":")
        delta = dt.timedelta(hours=int(hh), minutes=int(mm)) * sign
        return dt.timezone(delta)

    if ZoneInfo is None:
        raise RuntimeError("zoneinfo is unavailable; cannot construct ZoneInfo.")
    return ZoneInfo(value)


@dataclass(frozen=True)
class TimezoneType(ISOType):
    """IANA timezone identifier (or ISO 8601 fixed offset).

    Accepts canonical IANA names (``UTC``, ``Europe/Paris``, …), the
    backward-compat Link aliases shipped in ``tzdata`` (``US/Eastern``,
    ``Asia/Calcutta``, ``GB`` …), the legacy abbreviation-only zones
    (``CET``, ``EST``, ``MST`` …), and fixed UTC offsets in any of
    ``Z`` / ``+05:00`` / ``+0530`` / ``UTC+5`` / ``GMT-08:00`` forms.

    Output is the current canonical ``Area/Location`` IANA name when
    known, or an ISO 8601 ``±HH:MM`` string for fixed offsets.

    The value also round-trips to :class:`datetime.tzinfo` via the
    registered converters: ``convert(tz, dt.tzinfo)`` ↔
    ``convert(tz_string, str)``.
    """

    iso_name: ClassVar[str] = "timezone"

    # ------------------------------------------------------------------
    # Python-object lookup
    # ------------------------------------------------------------------
    def _normalize(self, value: Any) -> str | None:
        if value is None:
            return None

        # Already a tzinfo — convert to a canonical string and feed it
        # through the string path so legacy aliases still get rewritten.
        if isinstance(value, dt.tzinfo):
            value = _tzinfo_to_timezone_string(value)
        elif isinstance(value, dt.timedelta):
            try:
                return _timedelta_to_offset_string(value)
            except ValueError:
                return None

        text = str(value)
        key = _normalize_key(text)
        return key or None

    def _resolve_token(self, token: str) -> str | None:
        # `token` is either a normalized uppercase IANA-style key or a
        # pre-canonicalized offset (e.g. "+05:00") that came from a
        # tzinfo/timedelta shortcut in `_normalize`.
        direct = _TIMEZONE_MAP.get(token)
        if direct is not None:
            return direct

        # Try to interpret as a fixed UTC offset.
        return _parse_offset_token(token)

    @classmethod
    def _build_lookup_map(cls) -> Mapping[str, str]:
        return _TIMEZONE_MAP

    # ------------------------------------------------------------------
    # Reverse conversions (string -> tzinfo / timedelta)
    # ------------------------------------------------------------------
    @staticmethod
    def to_tzinfo(value: str) -> dt.tzinfo:
        """Convert a canonical timezone string back into a :class:`tzinfo`."""
        return _timezone_string_to_tzinfo(value)

    @staticmethod
    def to_timedelta(value: str) -> dt.timedelta:
        """Convert a canonical offset string into a :class:`timedelta`."""
        offset = _parse_offset_token(value)
        if offset is None:
            raise ValueError(
                f"Cannot interpret {value!r} as a fixed UTC offset; use "
                "TimezoneType.to_tzinfo for IANA zones."
            )
        sign = 1 if offset.startswith("+") else -1
        hh, mm = offset[1:].split(":")
        return dt.timedelta(hours=int(hh), minutes=int(mm)) * sign

    # ------------------------------------------------------------------
    # Arrow vectorized normalization + lookup (with offset fallback).
    # ------------------------------------------------------------------
    def _normalize_arrow_string(self, array: pa.Array) -> pa.Array:
        # \ -> /   trim   collapse whitespace around /   collapse \s to _
        current = pc.replace_substring(array, pattern="\\", replacement="/")
        current = pc.utf8_trim_whitespace(current)
        current = pc.replace_substring_regex(current, pattern=r"\s*/\s*", replacement="/")
        current = pc.replace_substring_regex(current, pattern=r"\s+", replacement="_")
        return pc.utf8_upper(current)

    def _resolve_arrow_string(self, array: pa.Array) -> pa.Array:
        normalized = self._normalize_arrow_string(array)

        _, keys, values = self._lookup_arrays()
        indices = pc.index_in(normalized, value_set=keys)
        resolved = pc.take(values, indices)

        # Fallback: for still-null entries, try to parse a UTC offset from
        # the normalized string.  "+HH:MM"  "+HHMM"  "+H"  "UTC+..." "GMT+..."
        offset = self._resolve_offset_arrow(normalized)
        return pc.coalesce(resolved, offset)

    @staticmethod
    def _resolve_offset_arrow(normalized: pa.Array) -> pa.Array:
        # Canonicalize on plain utf8 so downstream kernels have consistent types.
        if pa.types.is_large_string(normalized.type):
            normalized = pc.cast(normalized, pa.string())
        # Strip leading UTC/GMT prefix if present (already uppercase).
        stripped = pc.replace_substring_regex(
            normalized, pattern=r"^(?:UTC|GMT)", replacement=""
        )
        # Standalone "Z" -> "+00:00"
        stripped = pc.if_else(pc.equal(stripped, pa.scalar("Z")), pa.scalar("+00:00"), stripped)

        # Extract sign, hours, minutes.  Minutes are optional.
        extracted = pc.extract_regex(
            stripped, pattern=r"^(?P<sign>[+-])(?P<hh>\d{1,2}):?(?P<mm>\d{2})?$"
        )

        sign = pc.struct_field(extracted, "sign")
        hh = pc.struct_field(extracted, "hh")
        mm = pc.struct_field(extracted, "mm")

        # extract_regex returns empty strings for missing groups; normalize to null.
        empty = pa.scalar("")
        mm_nullable = pc.if_else(pc.equal(mm, empty), pa.scalar(None, type=pa.string()), mm)
        # Pad hours to 2 digits; default missing minutes to "00".
        hh_padded = pc.utf8_lpad(hh, width=2, padding="0")
        mm_filled = pc.fill_null(mm_nullable, pa.scalar("00"))

        # Validate ranges (hours 0-23, minutes 0-59) — anything else nulls out.
        hh_int = pc.cast(hh_padded, pa.int32(), safe=False)
        mm_int = pc.cast(mm_filled, pa.int32(), safe=False)
        in_range = pc.and_(
            pc.and_(pc.greater_equal(hh_int, 0), pc.less(hh_int, 24)),
            pc.and_(pc.greater_equal(mm_int, 0), pc.less(mm_int, 60)),
        )

        combined = pc.binary_join_element_wise(sign, hh_padded, ":", mm_filled, "")
        return pc.if_else(in_range, combined, pa.scalar(None, type=pa.string()))

    # ------------------------------------------------------------------
    # Polars lazy expression
    # ------------------------------------------------------------------
    def _cast_polars_expr(
        self,
        expr: "polars.Expr",
        options,
    ):
        from yggdrasil.data.types.support import get_polars
        pl = get_polars()

        normalized = (
            expr.cast(pl.Utf8, strict=False)
            .str.replace_all("\\", "/", literal=True)
            .str.strip_chars()
            .str.replace_all(r"\s*/\s*", "/")
            .str.replace_all(r"\s+", "_")
            .str.to_uppercase()
        )

        named = normalized.replace_strict(
            _TIMEZONE_MAP, default=None, return_dtype=pl.Utf8
        )

        # Offset fallback: strip UTC/GMT prefix, normalize "Z", extract groups.
        stripped = (
            normalized
            .str.replace(r"^(?:UTC|GMT)", "", literal=False)
            .str.replace(r"^Z$", "+00:00", literal=False)
        )
        sign = stripped.str.extract(r"^([+-])\d{1,2}:?\d{0,2}$", 1)
        hh = stripped.str.extract(r"^[+-](\d{1,2}):?\d{0,2}$", 1)
        mm = stripped.str.extract(r"^[+-]\d{1,2}:?(\d{2})$", 1)

        hh_padded = hh.str.zfill(2)
        mm_filled = pl.when(mm.is_null()).then(pl.lit("00")).otherwise(mm)

        hh_int = hh_padded.cast(pl.Int32, strict=False)
        mm_int = mm_filled.cast(pl.Int32, strict=False)
        in_range = (
            (hh_int >= 0) & (hh_int < 24) & (mm_int >= 0) & (mm_int < 60)
        )

        offset = (
            pl.when(sign.is_not_null() & hh.is_not_null() & in_range)
            .then(sign + hh_padded + pl.lit(":") + mm_filled)
            .otherwise(pl.lit(None, dtype=pl.Utf8))
        )

        return pl.coalesce([named, offset])

    # ------------------------------------------------------------------
    # Spark lazy column
    # ------------------------------------------------------------------
    def _cast_spark_column(self, column, options):
        from yggdrasil.data.types.support import get_spark_sql
        spark = get_spark_sql()
        F = spark.functions
        options.check_source(column)

        current = column.cast(spark.types.StringType())
        current = F.regexp_replace(current, r"\\\\", "/")
        current = F.trim(current)
        current = F.regexp_replace(current, r"\s*/\s*", "/")
        current = F.regexp_replace(current, r"\s+", "_")
        normalized = F.upper(current)

        if _TIMEZONE_MAP:
            map_args: list[Any] = []
            for k, v in _TIMEZONE_MAP.items():
                map_args.append(F.lit(k))
                map_args.append(F.lit(v))
            lookup_map = F.create_map(*map_args)
            named = F.element_at(lookup_map, normalized)
        else:
            named = F.lit(None).cast(spark.types.StringType())

        # Offset fallback.
        stripped = F.regexp_replace(normalized, r"^(?:UTC|GMT)", "")
        stripped = F.when(stripped == F.lit("Z"), F.lit("+00:00")).otherwise(stripped)

        sign = F.regexp_extract(stripped, r"^([+-])\d{1,2}:?\d{0,2}$", 1)
        hh = F.regexp_extract(stripped, r"^[+-](\d{1,2}):?\d{0,2}$", 1)
        mm = F.regexp_extract(stripped, r"^[+-]\d{1,2}:?(\d{2})$", 1)

        hh_padded = F.lpad(hh, 2, "0")
        mm_filled = F.when(mm == F.lit(""), F.lit("00")).otherwise(mm)

        hh_int = hh_padded.cast(spark.types.IntegerType())
        mm_int = mm_filled.cast(spark.types.IntegerType())
        in_range = (
            (hh_int >= 0) & (hh_int < 24) & (mm_int >= 0) & (mm_int < 60)
        )

        has_match = (sign != F.lit("")) & (hh != F.lit("")) & in_range
        offset = F.when(
            has_match,
            F.concat(sign, hh_padded, F.lit(":"), mm_filled),
        ).otherwise(F.lit(None).cast(spark.types.StringType()))

        return F.coalesce(named, offset)

    # ------------------------------------------------------------------
    # Outgoing — timezone → UTC-offset duration / numeric.
    # Resolution per *unique* canonical value (dictionary-encode collapses
    # the typical ~hundreds of IANA zones down to a handful of offsets),
    # then pc.take broadcasts the result.
    # ------------------------------------------------------------------
    def _outgoing_cast_arrow_array(
        self,
        array: "pa.Array | pa.ChunkedArray",
        target: "DataType",
        options: "CastOptions",
    ) -> "pa.Array | pa.ChunkedArray | None":
        target_id = target.type_id

        if target_id == DataTypeId.DURATION:
            resolver = _timezone_resolver_duration(target.unit)
            return _apply_tz_resolver(array, resolver, target.to_arrow())

        if target_id in _TZ_NUMERIC_TARGETS:
            # Default unit: seconds; callers who want minutes/hours can
            # divide downstream — keeps the semantics unambiguous.
            resolver = _timezone_resolver_seconds()
            result = _apply_tz_resolver(array, resolver, pa.int64())
            if result.type != target.to_arrow():
                result = pc.cast(result, target.to_arrow(), safe=False)
            return result

        return None

    # ------------------------------------------------------------------
    # Dict round-trip
    # ------------------------------------------------------------------
    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        name = str(value.get("name", "")).upper()
        iso = str(value.get("iso", "")).lower()
        return name in {"TIMEZONETYPE", "TIMEZONE"} or iso == cls.iso_name


# ---------------------------------------------------------------------------
# Outgoing offset resolution (canonical timezone string → seconds).
# ---------------------------------------------------------------------------


_UNIT_TO_NANOS = {
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}


def _timezone_string_to_offset_seconds(value: str | None) -> int | None:
    """Canonical timezone string → current UTC offset in seconds.

    Fixed-offset strings are parsed directly.  IANA zones consult
    :class:`zoneinfo.ZoneInfo` using the current UTC wall clock (so
    DST-aware zones return whichever offset applies *now* — this is the
    conventional choice for point-in-time conversions).
    """
    if value is None:
        return None

    offset = _parse_offset_token(value)
    if offset is not None:
        sign = 1 if offset.startswith("+") else -1
        hh, mm = offset[1:].split(":")
        return sign * (int(hh) * 3600 + int(mm) * 60)

    if ZoneInfo is None:
        return None
    try:
        info = ZoneInfo(value)
    except Exception:
        return None
    delta = info.utcoffset(dt.datetime.now(tz=dt.timezone.utc))
    if delta is None:
        return None
    return int(delta.total_seconds())


def _timezone_resolver_seconds():
    """Return the canonical-string → int-seconds resolver (no per-call overhead)."""
    return _timezone_string_to_offset_seconds


def _timezone_resolver_duration(unit: str):
    """Return a resolver that produces duration-unit integers for *unit*."""
    nanos_per_unit = _UNIT_TO_NANOS.get(unit)
    if nanos_per_unit is None:
        raise ValueError(f"Unsupported duration unit: {unit!r}")

    def resolver(value: str | None) -> int | None:
        secs = _timezone_string_to_offset_seconds(value)
        if secs is None:
            return None
        # secs -> unit-count.  The divisions are exact for s/ms/us/ns.
        total_nanos = secs * 1_000_000_000
        return total_nanos // nanos_per_unit

    return resolver


def _apply_tz_resolver(
    array: "pa.Array | pa.ChunkedArray",
    resolver,
    target_type: pa.DataType,
) -> "pa.Array | pa.ChunkedArray":
    """Apply *resolver* per unique value, then ``pc.take`` back + cast.

    Preserves the chunked layout of the input so callers can pass tables
    through unchanged.  Delegates the per-unique-value loop to
    :func:`resolve_arrow_string_via_unique`-style dictionary encoding.
    """
    if isinstance(array, pa.ChunkedArray):
        chunks = [_apply_tz_resolver(c, resolver, target_type) for c in array.chunks]
        return pa.chunked_array(chunks, type=target_type)

    if pa.types.is_large_string(array.type) or pa.types.is_string_view(array.type):
        array = pc.cast(array, pa.string())
    elif not pa.types.is_string(array.type):
        array = pc.cast(array, pa.string())

    if len(array) == 0:
        return pa.array([], type=target_type)

    encoded = pc.dictionary_encode(array)
    unique_values = encoded.dictionary.to_pylist()
    resolved_unique = [resolver(v) if v is not None else None for v in unique_values]

    # Use int64 as the universal carrier; pc.cast narrows to the requested type.
    resolved_arr = pa.array(resolved_unique, type=pa.int64())
    broadcast = pc.take(resolved_arr, encoded.indices)
    if broadcast.type != target_type:
        broadcast = pc.cast(broadcast, target_type, safe=False)
    return broadcast


# ---------------------------------------------------------------------------
# Cross-type converters registered with yggdrasil.data.cast.
# ---------------------------------------------------------------------------


_GLOBAL_TZ = TimezoneType()


def _any_to_timezone_string(value: Any) -> str:
    """Coerce an arbitrary value into a canonical timezone string."""
    result = _GLOBAL_TZ._normalize(value)
    if result is not None:
        resolved = _GLOBAL_TZ._resolve_token(result)
        if resolved is not None:
            return resolved
    raise ValueError(f"Cannot interpret {value!r} as a timezone.")


@register_converter(dt.tzinfo, str)
def tzinfo_to_timezone_string(value: dt.tzinfo, opts: Any = None) -> str:
    return _any_to_timezone_string(value)


@register_converter(dt.timedelta, str)
def timedelta_to_timezone_string(value: dt.timedelta, opts: Any = None) -> str:
    return _timedelta_to_offset_string(value)


if ZoneInfo is not None:
    @register_converter(ZoneInfo, str)
    def zoneinfo_to_timezone_string(value: "ZoneInfo", opts: Any = None) -> str:
        return _any_to_timezone_string(value)

    @register_converter(str, ZoneInfo)
    def timezone_string_to_zoneinfo(value: str, opts: Any = None) -> "ZoneInfo":
        tz = _timezone_string_to_tzinfo(_any_to_timezone_string(value))
        if isinstance(tz, ZoneInfo):
            return tz
        raise ValueError(
            f"{value!r} resolved to a fixed UTC offset; use dt.tzinfo instead of ZoneInfo."
        )


@register_converter(TimezoneType, str)
def timezone_type_to_string(value: TimezoneType, opts: Any = None) -> str:
    return str(value)
