"""ISO 3166-2 subdivision type (country + subdivision code).

Canonical form: ``<alpha-2>-<subdivision>`` (e.g. ``US-CA``, ``FR-75``,
``GB-ENG``).  No embedded ISO 3166-2 catalog — validation and
extraction are structural with flexible country-prefix matching.

Parsing is tolerant of a wide range of input shapes::

    US-CA               -> US-CA
    USA-CA              -> US-CA
    US CA               -> US-CA
    France 75           -> FR-75
    Germany - Berlin    -> DE-BER   (subdivision truncated to 3 alnum)
    UNITED STATES CA    -> US-CA
    CH-                 -> CH       (country only)
    GB                  -> GB       (country only)

For country-only inputs (no trailing subdivision) the output is just
the alpha-2 country code.  Unresolvable values become null when
``safe=False``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

import pyarrow as pa

from .base import (
    ISOType,
    normalize_iso_token_keep_hyphen,
    resolve_arrow_string_via_unique,
)
from .country import _ALPHA2_MAP, _VALID_ALPHA2

if TYPE_CHECKING:
    import polars

__all__ = ["ISOSubdivisionType", "parse_subdivision_token"]


# Max number of leading words to test as a country name ("UNITED ARAB EMIRATES"
# is 3, "BONAIRE SINT EUSTATIUS AND SABA" is 5 but those keep the same
# alpha-2, so a cap of 5 covers every ISO 3166-1 short name we ship).
_MAX_COUNTRY_TOKENS: int = 5

_WORD_RE = re.compile(r"[A-Z0-9]+")


def _resolve_country_alpha2(value: str) -> str | None:
    if value in _VALID_ALPHA2:
        return value
    return _ALPHA2_MAP.get(value)


def parse_subdivision_token(token: str) -> str | None:
    """Flexible parser for ISO 3166-2-style inputs.

    ``token`` must already be upper-cased; this function handles
    whitespace/punctuation normalization internally.  Returns the
    canonical form or None when the country prefix can't be matched.
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
            # Country-only input like "CH-" or "GB".
            return country

        # Concatenate remaining alphanumerics and truncate to 3 chars
        # (ISO 3166-2 subdivision codes are 1-3 chars).
        sub_raw = "".join(sub_words)
        sub = sub_raw[:3]
        if not sub or not sub.isalnum():
            return country
        return f"{country}-{sub}"

    # Single-token fallback: "USCA", "FRPAR" — chop the country prefix off.
    if len(words) == 1:
        return _chop_prefix_fallback(words[0])

    return None


def _chop_prefix_fallback(compact: str) -> str | None:
    """Fallback: treat the first 2 or 3 chars as a country code."""
    # Try alpha-3 first (more specific), then alpha-2.
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
class ISOSubdivisionType(ISOType):
    """ISO 3166-2 subdivision code with flexible input parsing.

    Recognizes country names, alpha-2/alpha-3 codes, numeric codes, and
    common aliases at the country position; the trailing subdivision
    part is truncated to the first 3 alphanumerics.
    """

    iso_name: ClassVar[str] = "iso_subdivision"

    def _normalize(self, value: Any) -> str | None:
        return normalize_iso_token_keep_hyphen(value)

    def _resolve_token(self, token: str) -> str | None:
        return parse_subdivision_token(token)

    # ------------------------------------------------------------------
    # Vectorized Arrow — dictionary-encode to collapse duplicates so the
    # flexible Python parser runs once per unique raw input.
    # ------------------------------------------------------------------
    def _resolve_arrow_string(self, array: pa.Array) -> pa.Array:
        return resolve_arrow_string_via_unique(array, self._resolve_raw)

    def _resolve_raw(self, raw: str) -> str | None:
        token = self._normalize(raw)
        if token is None:
            return None
        return parse_subdivision_token(token)

    @classmethod
    def _build_lookup_map(cls) -> Mapping[str, str]:
        return {}

    # ------------------------------------------------------------------
    # Polars lazy — delegate to the Python parser via map_elements.
    # Lazy contract: never raises on unparseable values.
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
            return parse_subdivision_token(token)

        return expr.cast(pl.Utf8, strict=False).map_elements(
            _resolve, return_dtype=pl.Utf8
        )

    # ------------------------------------------------------------------
    # Spark lazy — Python UDF so the flexible parser is reused.
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
            return parse_subdivision_token(token)

        resolve_udf = F.udf(_resolve, spark.types.StringType())
        return resolve_udf(column.cast(spark.types.StringType()))

    # ------------------------------------------------------------------
    # Dict
    # ------------------------------------------------------------------
    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        name = str(value.get("name", "")).upper()
        iso = str(value.get("iso", "")).lower()
        return name in {"ISOSUBDIVISIONTYPE", "ISO_SUBDIVISION"} or iso == cls.iso_name
