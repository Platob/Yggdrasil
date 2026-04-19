"""ISO 4217 currency type (alpha-3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

import pyarrow as pa
import pyarrow.compute as pc

from .base import ISOType, apply_arrow_lookup, normalize_iso_token
from .data import CURRENCIES, CURRENCY_ALIASES
from ..id import DataTypeId

if TYPE_CHECKING:
    import polars
    from ..base import DataType
    from yggdrasil.data.cast.options import CastOptions


__all__ = ["ISOCurrencyType"]


# Symbol -> alpha-3 substitutions applied before alphanumeric normalization,
# since the symbols themselves wouldn't survive the [^A-Z0-9]+ regex.
_SYMBOL_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("US$", "USD"),
    ("USD$", "USD"),
    ("AUD$", "AUD"),
    ("CAD$", "CAD"),
    ("NZD$", "NZD"),
    ("HK$", "HKD"),
    ("S$", "SGD"),
    ("\u20AC", "EUR"),
    ("\u00A3", "GBP"),
    ("\u00A5", "JPY"),
    ("\u20B9", "INR"),
    ("\u20BD", "RUB"),
    ("\u20A9", "KRW"),
    ("\u20BA", "TRY"),
    ("\u20B4", "UAH"),
    ("\u20AA", "ILS"),
    ("\u20AB", "VND"),
    ("$", "USD"),
)


def _spark_regex_escape(pattern: str) -> str:
    special = r".^$*+?()[]{}|\\"
    return "".join("\\" + ch if ch in special else ch for ch in pattern)


def _currency_name_aliases(name: str) -> tuple[str, ...]:
    upper = name.upper()
    squashed = upper.replace(" ", "")
    return (upper, squashed)


def _build_currency_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for alpha3, numeric, _minor, name in CURRENCIES:
        mapping[alpha3] = alpha3
        mapping[numeric] = alpha3
        lstripped = numeric.lstrip("0") or "0"
        mapping.setdefault(lstripped, alpha3)
        for token in _currency_name_aliases(name):
            mapping.setdefault(token, alpha3)

    for alias, code in CURRENCY_ALIASES.items():
        mapping.setdefault(alias, code)
        mapping.setdefault(alias.replace(" ", ""), code)
        alnum = "".join(ch for ch in alias if ch.isalnum())
        if alnum:
            mapping.setdefault(alnum, code)

    return mapping


_CURRENCY_MAP: dict[str, str] = _build_currency_map()
_VALID_CODES: frozenset[str] = frozenset(alpha3 for alpha3, *_ in CURRENCIES)

# Canonical alpha-3 → ISO 4217 numeric code / minor units.  Used by the
# outgoing-cast hook for numeric/decimal targets.
_CURRENCY_TO_NUMERIC: dict[str, int] = {alpha3: int(numeric) for alpha3, numeric, *_ in CURRENCIES}
_CURRENCY_TO_MINOR: dict[str, int] = {alpha3: int(minor) for alpha3, _, minor, _ in CURRENCIES}


@dataclass(frozen=True)
class ISOCurrencyType(ISOType):
    """ISO 4217 currency code type (alpha-3 only).

    Accepts alpha-3 codes, numeric codes, English names, common symbols
    (``$``, ``\u20AC``, ``\u00A3``, ``\u00A5``), and aliases (``DOLLAR``,
    ``EURO``, ``POUND``, ``YEN``, ``RMB`` …).  Always normalizes to the
    alpha-3 code.
    """

    iso_name: ClassVar[str] = "iso_currency"

    def _normalize(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().upper()
        if not text:
            return None
        for sym, code in _SYMBOL_SUBSTITUTIONS:
            text = text.replace(sym, f" {code} ")
        return normalize_iso_token(text)

    def _resolve_token(self, token: str) -> str | None:
        if token in _VALID_CODES:
            return token
        direct = _CURRENCY_MAP.get(token)
        if direct is not None:
            return direct
        squashed = token.replace(" ", "")
        return _CURRENCY_MAP.get(squashed)

    @classmethod
    def _build_lookup_map(cls) -> Mapping[str, str]:
        return _CURRENCY_MAP

    def _normalize_arrow_string(self, array: pa.Array) -> pa.Array:
        current = array
        for sym, code in _SYMBOL_SUBSTITUTIONS:
            current = pc.replace_substring(current, pattern=sym, replacement=f" {code} ")
        upper = pc.utf8_upper(current)
        collapsed = pc.replace_substring_regex(upper, pattern=r"[^A-Z0-9]+", replacement=" ")
        return pc.utf8_trim_whitespace(collapsed)

    def _cast_polars_expr(
        self,
        expr: "polars.Expr",
        options,
    ):
        from yggdrasil.data.types.support import get_polars
        pl = get_polars()

        current = expr.cast(pl.Utf8, strict=False)
        for sym, code in _SYMBOL_SUBSTITUTIONS:
            current = current.str.replace_all(sym, f" {code} ", literal=True)
        normalized = (
            current
            .str.to_uppercase()
            .str.replace_all(r"[^A-Z0-9]+", " ")
            .str.strip_chars()
        )
        return normalized.replace_strict(_CURRENCY_MAP, default=None, return_dtype=pl.Utf8)

    def _cast_spark_column(self, column, options):
        from yggdrasil.data.types.support import get_spark_sql
        spark = get_spark_sql()
        F = spark.functions
        options.check_source(column)

        current = column.cast(spark.types.StringType())
        for sym, code in _SYMBOL_SUBSTITUTIONS:
            current = F.regexp_replace(current, _spark_regex_escape(sym), f" {code} ")

        normalized = F.trim(
            F.regexp_replace(F.upper(current), r"[^A-Z0-9]+", " ")
        )

        if not _CURRENCY_MAP:
            return F.lit(None).cast(spark.types.StringType())

        map_args = []
        for k, v in _CURRENCY_MAP.items():
            map_args.append(F.lit(k))
            map_args.append(F.lit(v))
        lookup_map = F.create_map(*map_args)
        return F.element_at(lookup_map, normalized)

    # ------------------------------------------------------------------
    # Outgoing — currency → numeric ISO 4217 code.  Class-level cache so
    # the (keys, values) Arrow arrays are built once and reused.
    # ------------------------------------------------------------------
    _outgoing_numeric_arrays_cache: ClassVar[tuple[pa.Array, pa.Array] | None] = None

    @classmethod
    def _outgoing_numeric_arrays(cls) -> tuple[pa.Array, pa.Array]:
        cached = cls._outgoing_numeric_arrays_cache
        if cached is not None:
            return cached
        keys = pa.array(list(_CURRENCY_TO_NUMERIC.keys()), type=pa.string())
        values = pa.array(list(_CURRENCY_TO_NUMERIC.values()), type=pa.int32())
        cached = (keys, values)
        cls._outgoing_numeric_arrays_cache = cached
        return cached

    def _outgoing_cast_arrow_array(
        self,
        array: pa.Array | pa.ChunkedArray,
        target: "DataType",
        options: "CastOptions",
    ) -> pa.Array | pa.ChunkedArray | None:
        if target.type_id in {DataTypeId.INTEGER, DataTypeId.FLOAT, DataTypeId.DECIMAL}:
            keys, values = self._outgoing_numeric_arrays()
            return apply_arrow_lookup(
                array,
                keys,
                values,
                target.to_arrow(),
                memory_pool=options.arrow_memory_pool,
            )
        return None

    # ------------------------------------------------------------------
    # Dict
    # ------------------------------------------------------------------
    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        name = str(value.get("name", "")).upper()
        iso = str(value.get("iso", "")).lower()
        return name in {"ISOCURRENCYTYPE", "ISO_CURRENCY"} or iso == cls.iso_name
