"""ISO-style city/town code type (UN/LOCODE-compatible) with flexible parsing.

The canonical form is ``<alpha-2>-<city>`` where ``city`` is up to 3
alphanumerics (UN/LOCODE uses exactly 3).  Parsing is tolerant of a
wide range of input shapes::

    FR-PAR              -> FR-PAR
    FR PAR              -> FR-PAR
    FRPAR               -> FR-PAR
    France Paris        -> FR-PAR   (first 3 alnum of "Paris")
    Germany - Berlin    -> DE-BER
    USA-NYC             -> US-NYC
    FR                  -> FR       (country only)

For country-only inputs the output is just the alpha-2 country code.
Unresolvable values become null when ``safe=False``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

import pyarrow as pa

from .base import ISOType, normalize_iso_token_keep_hyphen
from .country import _ALPHA2_MAP, _VALID_ALPHA2

if TYPE_CHECKING:
    import polars

__all__ = ["ISOCityType", "parse_city_token"]


_MAX_COUNTRY_TOKENS: int = 5

_WORD_RE = re.compile(r"[A-Z0-9]+")


def _resolve_country_alpha2(value: str) -> str | None:
    if value in _VALID_ALPHA2:
        return value
    return _ALPHA2_MAP.get(value)


def parse_city_token(token: str) -> str | None:
    """Flexible parser for UN/LOCODE-style inputs.

    ``token`` must already be upper-cased.  Returns the canonical
    ``CC-SSS`` form, or just ``CC`` when no city part is supplied, or
    None if the country prefix can't be matched.
    """
    words = _WORD_RE.findall(token)
    if not words:
        return None

    max_prefix = min(len(words), _MAX_COUNTRY_TOKENS)
    for prefix_len in range(max_prefix, 0, -1):
        country_text = " ".join(words[:prefix_len])
        country = _resolve_country_alpha2(country_text)
        if country is None:
            country = _resolve_country_alpha2(country_text.replace(" ", ""))
        if country is None:
            continue

        sub_words = words[prefix_len:]
        if not sub_words:
            return country

        sub_raw = "".join(sub_words)
        sub = sub_raw[:3]
        if not sub or not sub.isalnum():
            return country
        return f"{country}-{sub}"

    # Single-token fallback: "FRPAR", "USNYC" — chop the country prefix off.
    if len(words) == 1:
        return _chop_prefix_fallback(words[0])

    return None


def _chop_prefix_fallback(compact: str) -> str | None:
    """Fallback: treat the first 2 or 3 chars as a country code."""
    for split_at in (3, 2):
        if len(compact) <= split_at:
            continue
        country = _resolve_country_alpha2(compact[:split_at])
        if country is None:
            continue
        sub_raw = compact[split_at:]
        sub = sub_raw[:3]
        if not sub or not sub.isalnum():
            return country
        return f"{country}-{sub}"
    return None


@dataclass(frozen=True)
class ISOCityType(ISOType):
    """UN/LOCODE-style city code with flexible input parsing.

    Accepts country name/code + optional city identifier; the city
    part is truncated to the first 3 alphanumerics.  See
    :func:`parse_city_token` for the full grammar.
    """

    iso_name: ClassVar[str] = "iso_city"

    def _normalize(self, value: Any) -> str | None:
        return normalize_iso_token_keep_hyphen(value)

    def _resolve_token(self, token: str) -> str | None:
        return parse_city_token(token)

    # ------------------------------------------------------------------
    # Arrow — row-wise Python (flexible parsing).
    # ------------------------------------------------------------------
    def _resolve_arrow_string(self, array: pa.Array) -> pa.Array:
        out: list[str | None] = []
        for s in array.to_pylist():
            if s is None:
                out.append(None)
                continue
            token = self._normalize(s)
            if token is None:
                out.append(None)
                continue
            out.append(parse_city_token(token))
        return pa.array(out, type=pa.string())

    @classmethod
    def _build_lookup_map(cls) -> Mapping[str, str]:
        return {}

    # ------------------------------------------------------------------
    # Polars lazy path — map_elements with the flexible parser.
    # ------------------------------------------------------------------
    def _cast_polars_expr(
        self,
        expr: "polars.Expr",
        options,
    ):
        from yggdrasil.data.types.support import get_polars
        pl = get_polars()

        def _resolve(s: str | None) -> str | None:
            if s is None:
                return None
            token = self._normalize(s)
            if token is None:
                return None
            return parse_city_token(token)

        return expr.cast(pl.Utf8, strict=False).map_elements(
            _resolve, return_dtype=pl.Utf8
        )

    # ------------------------------------------------------------------
    # Spark lazy path — Python UDF.
    # ------------------------------------------------------------------
    def _cast_spark_column(self, column, options):
        from yggdrasil.data.types.support import get_spark_sql
        spark = get_spark_sql()
        F = spark.functions
        options.check_source(column)

        def _resolve(s):
            if s is None:
                return None
            token = self._normalize(s)
            if token is None:
                return None
            return parse_city_token(token)

        resolve_udf = F.udf(_resolve, spark.types.StringType())
        return resolve_udf(column.cast(spark.types.StringType()))

    # ------------------------------------------------------------------
    # Dict
    # ------------------------------------------------------------------
    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        name = str(value.get("name", "")).upper()
        iso = str(value.get("iso", "")).lower()
        return name in {"ISOCITYTYPE", "ISO_CITY"} or iso == cls.iso_name
