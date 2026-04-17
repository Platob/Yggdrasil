"""ISO-style city/town code type (UN/LOCODE-compatible).

UN/LOCODE is ISO 3166-1 alpha-2 country code + 3-letter city code
(e.g. ``FR PAR`` for Paris).  The canonical form used here is
``<alpha-2>-<city>`` (hyphenated for consistency with subdivisions).

No embedded catalog — validation is structural only.  Unknown country
prefixes become null.
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

__all__ = ["ISOCityType"]


_LOCODE_RE = re.compile(r"^([A-Z]{2})-([A-Z0-9]{3})$")


@dataclass(frozen=True)
class ISOCityType(ISOType):
    """UN/LOCODE-style city code (``<alpha-2>-<3 alnum>``).

    Accepts inputs with or without a separator (``FR PAR`` / ``FR-PAR``
    / ``FRPAR``) and normalizes to the hyphenated form ``FR-PAR``.
    Validates the country prefix against ISO 3166-1 alpha-2; the city
    part must be exactly 3 alphanumerics.
    """

    iso_name: ClassVar[str] = "iso_city"

    def _normalize(self, value: Any) -> str | None:
        return normalize_iso_token_keep_hyphen(value)

    def _resolve_token(self, token: str) -> str | None:
        compact = token.replace(" ", "")
        if not compact:
            return None

        if "-" in compact:
            country_part, _, city_part = compact.partition("-")
        else:
            country_part = compact[:2]
            city_part = compact[2:]

        country = _resolve_country_alpha2(country_part)
        if country is None:
            # Try alpha-3 prefix.
            if len(compact) >= 5:
                country = _resolve_country_alpha2(compact[:3])
                if country is not None:
                    city_part = compact[3:].lstrip("-")

        if country is None:
            return None

        city = city_part.strip("-")
        if len(city) != 3 or not city.isalnum():
            return None

        canonical = f"{country}-{city}"
        if _LOCODE_RE.match(canonical):
            return canonical
        return None

    # ------------------------------------------------------------------
    # Arrow
    # ------------------------------------------------------------------
    def _normalize_arrow_string(self, array: pa.Array) -> pa.Array:
        upper = pc.utf8_upper(array)
        return pc.replace_substring_regex(upper, pattern=r"[^A-Z0-9-]+", replacement="")

    def _resolve_arrow_string(self, array: pa.Array) -> pa.Array:
        upper = pc.utf8_upper(array)
        cleaned = pc.replace_substring_regex(upper, pattern=r"[^A-Z0-9-]+", replacement="")

        extracted = pc.extract_regex(
            cleaned, pattern=r"^(?P<country>[A-Z]{2})-?(?P<city>[A-Z0-9]{3})$"
        )
        country = pc.struct_field(extracted, "country")
        city = pc.struct_field(extracted, "city")

        valid_country_values = pa.array(sorted(_VALID_ALPHA2), type=pa.string())
        country_ok = pc.is_in(country, value_set=valid_country_values)

        combined = pc.binary_join_element_wise(country, city, "-")
        return pc.if_else(country_ok, combined, pa.scalar(None, type=pa.string()))

    @classmethod
    def _build_lookup_map(cls) -> Mapping[str, str]:
        return {}

    # ------------------------------------------------------------------
    # Polars lazy path
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
            r"^(?P<country>[A-Z]{2})-?(?P<city>[A-Z0-9]{3})$"
        )
        country = extracted.struct.field("country")
        city = extracted.struct.field("city")

        valid_country = pl.Series("_valid", sorted(_VALID_ALPHA2), dtype=pl.Utf8)
        country_ok = country.is_in(valid_country.implode())

        combined = country + pl.lit("-") + city
        return pl.when(country_ok).then(combined).otherwise(pl.lit(None, dtype=pl.Utf8))

    # ------------------------------------------------------------------
    # Spark lazy path
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

        pattern = r"^([A-Z]{2})-?([A-Z0-9]{3})$"
        country = F.regexp_extract(cleaned, pattern, 1)
        city = F.regexp_extract(cleaned, pattern, 2)

        valid_list = sorted(_VALID_ALPHA2)
        country_ok = country.isin(*valid_list) & (city != F.lit(""))

        combined = F.concat_ws("-", country, city)
        return F.when(country_ok, combined).otherwise(F.lit(None).cast(spark.types.StringType()))

    # ------------------------------------------------------------------
    # Dict
    # ------------------------------------------------------------------
    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        name = str(value.get("name", "")).upper()
        iso = str(value.get("iso", "")).lower()
        return name in {"ISOCITYTYPE", "ISO_CITY"} or iso == cls.iso_name


def _resolve_country_alpha2(value: str) -> str | None:
    if value in _VALID_ALPHA2:
        return value
    return _ALPHA2_MAP.get(value)
