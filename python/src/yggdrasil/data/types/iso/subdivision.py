"""ISO 3166-2 subdivision type (country + subdivision code).

The canonical form is ``<alpha-2>-<subdivision>`` (e.g. ``US-CA``,
``FR-75``, ``GB-ENG``).  No embedded lookup table — there are thousands
of subdivisions across countries and validation happens structurally:

1. Split on ``-`` (or space).
2. Resolve the country part through :class:`ISOCountryType` (alpha-2).
3. Require a subdivision part of 1-3 alphanumerics.

Values that don't match the structure become null when ``safe=False``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

import pyarrow as pa
import pyarrow.compute as pc

from .base import ISOType, normalize_iso_token_keep_hyphen
from .country import _ALPHA2_MAP, _VALID_ALPHA2

if TYPE_CHECKING:
    import polars

__all__ = ["ISOSubdivisionType"]


# ISO 3166-2 structural regex: alpha-2 country code + hyphen + 1-3 alphanumerics.
_ISO_3166_2_RE = re.compile(r"^([A-Z]{2})-([A-Z0-9]{1,3})$")


@dataclass(frozen=True)
class ISOSubdivisionType(ISOType):
    """ISO 3166-2 subdivision code (e.g. ``US-CA``, ``FR-75``).

    Validates structure only; no embedded catalog of all 5000+
    subdivision codes.  The country part is normalized via
    :class:`ISOCountryType` so inputs like ``USA-CA`` or ``UNITED
    STATES-CA`` resolve to ``US-CA``.
    """

    iso_name: ClassVar[str] = "iso_subdivision"

    def _normalize(self, value: Any) -> str | None:
        return normalize_iso_token_keep_hyphen(value)

    def _resolve_token(self, token: str) -> str | None:
        # token is upper-cased with non-alnum-except-hyphen collapsed to spaces.
        # Accept either "CC-SUB" or "CC SUB".
        compact = token.replace(" ", "")
        if not compact:
            return None

        # Try hyphen split first.
        if "-" in compact:
            country_part, _, sub_part = compact.partition("-")
        else:
            # No hyphen: try splitting at 2-char or 3-char prefix.
            country_part = compact[:2]
            sub_part = compact[2:]
            if not sub_part:
                return None

        country = _resolve_country_alpha2(country_part)
        if country is None:
            # Try alpha-3 prefix (first 3 chars).
            if len(compact) >= 4:
                country = _resolve_country_alpha2(compact[:3])
                if country is not None:
                    sub_part = compact[3:].lstrip("-")

        if country is None:
            return None

        sub = sub_part.strip("-")
        if not sub or not sub.isalnum():
            return None
        if not (1 <= len(sub) <= 3):
            return None

        canonical = f"{country}-{sub}"
        if _ISO_3166_2_RE.match(canonical):
            return canonical
        return None

    # ------------------------------------------------------------------
    # Vectorized Arrow — structural: normalize then regex-validate.
    # ------------------------------------------------------------------
    def _resolve_arrow_string(self, array: pa.Array) -> pa.Array:
        # Normalize: uppercase, collapse non-alnum-except-hyphen to single chars,
        # strip. We do upper, then replace non-[A-Z0-9-] with empty, then trim.
        upper = pc.utf8_upper(array)
        cleaned = pc.replace_substring_regex(upper, pattern=r"[^A-Z0-9-]+", replacement="")

        # Extract the ISO 3166-2 pattern: exactly 2 alpha + '-' + 1-3 alnum.
        extracted = pc.extract_regex(
            cleaned, pattern=r"^(?P<country>[A-Z]{2})-(?P<sub>[A-Z0-9]{1,3})$"
        )
        country = pc.struct_field(extracted, "country")
        sub = pc.struct_field(extracted, "sub")

        # Validate country part against alpha-2 set.
        valid_country_values = pa.array(sorted(_VALID_ALPHA2), type=pa.string())
        country_ok = pc.is_in(country, value_set=valid_country_values)

        combined = pc.binary_join_element_wise(country, sub, "-")
        return pc.if_else(country_ok, combined, pa.scalar(None, type=pa.string()))

    def _normalize_arrow_string(self, array: pa.Array) -> pa.Array:
        upper = pc.utf8_upper(array)
        return pc.replace_substring_regex(upper, pattern=r"[^A-Z0-9-]+", replacement="")

    @classmethod
    def _build_lookup_map(cls) -> Mapping[str, str]:
        return {}

    # ------------------------------------------------------------------
    # Polars lazy path — regex-validate structurally, unknown -> null.
    # ------------------------------------------------------------------
    def _cast_polars_expr(
        self,
        expr: "polars.Expr",
        options,
    ):
        from yggdrasil.data.types.support import get_polars
        pl = get_polars()

        cleaned = (
            expr.cast(pl.Utf8, strict=False)
            .str.to_uppercase()
            .str.replace_all(r"[^A-Z0-9-]+", "")
        )

        extracted = cleaned.str.extract_groups(
            r"^(?P<country>[A-Z]{2})-(?P<sub>[A-Z0-9]{1,3})$"
        )
        country = extracted.struct.field("country")
        sub = extracted.struct.field("sub")

        valid_country = pl.Series("_valid", sorted(_VALID_ALPHA2), dtype=pl.Utf8)
        country_ok = country.is_in(valid_country.implode())

        combined = country + pl.lit("-") + sub
        return pl.when(country_ok).then(combined).otherwise(pl.lit(None, dtype=pl.Utf8))

    # ------------------------------------------------------------------
    # Spark lazy path — regex-based structural validation.
    # ------------------------------------------------------------------
    def _cast_spark_column(self, column, options):
        from yggdrasil.data.types.support import get_spark_sql
        spark = get_spark_sql()
        F = spark.functions
        options.check_source(column)

        cleaned = F.regexp_replace(
            F.upper(column.cast(spark.types.StringType())),
            r"[^A-Z0-9-]+",
            "",
        )

        pattern = r"^([A-Z]{2})-([A-Z0-9]{1,3})$"
        country = F.regexp_extract(cleaned, pattern, 1)
        sub = F.regexp_extract(cleaned, pattern, 2)

        valid_list = sorted(_VALID_ALPHA2)
        # isin requires concrete python values; pass positional.
        country_ok = country.isin(*valid_list) & (sub != F.lit(""))

        combined = F.concat_ws("-", country, sub)
        return F.when(country_ok, combined).otherwise(F.lit(None).cast(spark.types.StringType()))

    # ------------------------------------------------------------------
    # Dict
    # ------------------------------------------------------------------
    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        name = str(value.get("name", "")).upper()
        iso = str(value.get("iso", "")).lower()
        return name in {"ISOSUBDIVISIONTYPE", "ISO_SUBDIVISION"} or iso == cls.iso_name


def _resolve_country_alpha2(value: str) -> str | None:
    if value in _VALID_ALPHA2:
        return value
    return _ALPHA2_MAP.get(value)
