"""IANA timezone type.

Accepts IANA timezone identifiers (``UTC``, ``Europe/Paris``,
``America/New_York``, …) plus common deprecated aliases and legacy
abbreviation-only zones (``CET``, ``US/Eastern``, ``Asia/Calcutta``).
Inputs are normalized to the current canonical IANA name.

The legacy abbreviation zones (``CET``, ``EET``, ``MET``, ``WET``,
``EST``, ``HST``, ``MST``, ``CST6CDT`` …) are intentionally **not**
treated as canonical — they get rewritten to a representative
Area/Location zone (``CET`` -> ``Europe/Paris``, ``MST`` ->
``America/Phoenix`` …).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

import pyarrow as pa
import pyarrow.compute as pc

from .base import ISOType
from .data.timezones import TIMEZONES, TIMEZONE_ALIASES

if TYPE_CHECKING:
    import polars

__all__ = ["TimezoneType"]


def _normalize_key(value: str) -> str:
    """Uppercase a timezone string for case-insensitive lookup.

    Preserves the ``/``, ``_``, ``-``, ``+`` separators that are part
    of an IANA identifier.  Windows-style backslashes are rewritten
    to forward slashes; interior whitespace (``"Europe / Paris"`` or
    ``"America/New York"``) is collapsed onto the IANA conventions
    (``/`` between area/location, ``_`` inside location segments).
    """
    text = value.replace("\\", "/").strip()
    if not text:
        return ""

    # Trim whitespace around the area separator.
    while " /" in text or "/ " in text:
        text = text.replace(" /", "/").replace("/ ", "/")

    # Interior runs of whitespace map onto the IANA underscore convention
    # (e.g. "New York" -> "New_York").
    segments = text.split("/")
    segments = ["_".join(seg.split()) for seg in segments]
    text = "/".join(segments)

    return text.upper()


def _build_timezone_map() -> dict[str, str]:
    """Build the lookup map from NORMALIZED token -> canonical IANA name."""
    mapping: dict[str, str] = {}

    for name in TIMEZONES:
        mapping[_normalize_key(name)] = name

    for alias, target in TIMEZONE_ALIASES.items():
        mapping.setdefault(_normalize_key(alias), target)

    return mapping


_TIMEZONE_MAP: dict[str, str] = _build_timezone_map()
_VALID_NAMES: frozenset[str] = frozenset(TIMEZONES)


@dataclass(frozen=True)
class TimezoneType(ISOType):
    """IANA timezone identifier.

    Accepts canonical IANA names (``UTC``, ``Europe/Paris``,
    ``America/New_York`` …), the backward-compat Link aliases shipped
    in ``tzdata`` (``US/Eastern``, ``Asia/Calcutta``, ``GB`` …), and
    the legacy abbreviation-only zones (``CET``, ``EST``, ``MST`` …).
    Lookup is case-insensitive and tolerant of spaces around the
    area separator; output is always the current canonical
    ``Area/Location`` name.
    """

    iso_name: ClassVar[str] = "timezone"

    # ------------------------------------------------------------------
    # Python-object lookup
    # ------------------------------------------------------------------
    def _normalize(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        key = _normalize_key(text)
        return key or None

    def _resolve_token(self, token: str) -> str | None:
        # `token` is already the normalized uppercase key.
        return _TIMEZONE_MAP.get(token)

    @classmethod
    def _build_lookup_map(cls) -> Mapping[str, str]:
        return _TIMEZONE_MAP

    # ------------------------------------------------------------------
    # Arrow vectorized normalization — mirror _normalize_key.
    # ------------------------------------------------------------------
    def _normalize_arrow_string(self, array: pa.Array) -> pa.Array:
        # 1. \ -> /    2. trim    3. collapse whitespace around /
        # 4. collapse remaining whitespace runs to '_'    5. uppercase.
        current = pc.replace_substring(array, pattern="\\", replacement="/")
        current = pc.utf8_trim_whitespace(current)
        current = pc.replace_substring_regex(current, pattern=r"\s*/\s*", replacement="/")
        current = pc.replace_substring_regex(current, pattern=r"\s+", replacement="_")
        return pc.utf8_upper(current)

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
        return normalized.replace_strict(
            _TIMEZONE_MAP, default=None, return_dtype=pl.Utf8
        )

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

        if not _TIMEZONE_MAP:
            return F.lit(None).cast(spark.types.StringType())

        map_args: list[Any] = []
        for k, v in _TIMEZONE_MAP.items():
            map_args.append(F.lit(k))
            map_args.append(F.lit(v))
        lookup_map = F.create_map(*map_args)
        return F.element_at(lookup_map, normalized)

    # ------------------------------------------------------------------
    # Dict round-trip
    # ------------------------------------------------------------------
    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        name = str(value.get("name", "")).upper()
        iso = str(value.get("iso", "")).lower()
        return name in {"TIMEZONETYPE", "TIMEZONE"} or iso == cls.iso_name
