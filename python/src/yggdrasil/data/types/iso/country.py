"""ISO 3166-1 country type."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

import pyarrow as pa
import pyarrow.compute as pc

from .base import ISOType, apply_arrow_lookup
from .data import COUNTRIES, COUNTRY_ALIASES
from ..id import DataTypeId

if TYPE_CHECKING:
    import polars
    from ..base import DataType
    from yggdrasil.data.cast.options import CastOptions

__all__ = ["ISOCountryType"]


def _country_name_aliases(name: str) -> tuple[str, ...]:
    """Collapsed and comma-swapped variants of a country name."""
    upper = name.upper()
    squashed = upper.replace(" ", "")
    tokens = (upper, squashed)
    if "," in upper:
        # "Korea, Republic of" -> "REPUBLIC OF KOREA"
        parts = [p.strip() for p in upper.split(",") if p.strip()]
        if len(parts) >= 2:
            swapped = " ".join(parts[::-1])
            tokens = (*tokens, swapped, swapped.replace(" ", ""))
    return tokens


def _build_country_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return (alpha2_map, alpha3_map) from normalized tokens -> canonical codes."""
    alpha2_map: dict[str, str] = {}
    alpha3_map: dict[str, str] = {}

    for a2, a3, numeric, name in COUNTRIES:
        for key in (a2, a3, numeric, numeric.lstrip("0") or "0", *_country_name_aliases(name)):
            alpha2_map.setdefault(key, a2)
            alpha3_map.setdefault(key, a3)

    for alias, a2 in COUNTRY_ALIASES.items():
        for a2c, a3c, _, _ in COUNTRIES:
            if a2c == a2:
                alpha2_map.setdefault(alias, a2c)
                alpha3_map.setdefault(alias, a3c)
                alpha2_map.setdefault(alias.replace(" ", ""), a2c)
                alpha3_map.setdefault(alias.replace(" ", ""), a3c)
                break

    return alpha2_map, alpha3_map


_ALPHA2_MAP, _ALPHA3_MAP = _build_country_maps()
_VALID_ALPHA2: frozenset[str] = frozenset(a2 for a2, *_ in COUNTRIES)
_VALID_ALPHA3: frozenset[str] = frozenset(a3 for _, a3, *_ in COUNTRIES)

# Canonical code → numeric ISO 3166-1 (as int).  Built once; used by the
# outgoing-cast hook to convert country arrays to numeric / string targets
# via pc.index_in + pc.take (no Python per-row loop).
_ALPHA2_TO_NUMERIC: dict[str, int] = {a2: int(numeric) for a2, _, numeric, _ in COUNTRIES}
_ALPHA3_TO_NUMERIC: dict[str, int] = {a3: int(numeric) for _, a3, numeric, _ in COUNTRIES}
_ALPHA2_TO_ALPHA3: dict[str, str] = {a2: a3 for a2, a3, _, _ in COUNTRIES}
_ALPHA3_TO_ALPHA2: dict[str, str] = {a3: a2 for a2, a3, _, _ in COUNTRIES}


@dataclass(frozen=True)
class ISOCountryType(ISOType):
    """ISO 3166-1 country type.

    Parameters
    ----------
    alpha : int
        Output code width: ``2`` (default) for alpha-2 (e.g. ``FR``)
        or ``3`` for alpha-3 (e.g. ``FRA``).

    Accepts alpha-2, alpha-3, numeric, English short names, and common
    aliases (``UK``, ``USA``, ``HOLLAND``, …) on input and normalizes to
    the configured output width.
    """

    alpha: int = 2

    iso_name: ClassVar[str] = "iso_country"

    def __post_init__(self) -> None:
        if self.alpha not in (2, 3):
            raise ValueError(
                f"ISOCountryType.alpha must be 2 or 3, got {self.alpha!r}."
            )

    # ------------------------------------------------------------------
    # Lookup tables
    # ------------------------------------------------------------------
    def _current_map(self) -> dict[str, str]:
        return _ALPHA3_MAP if self.alpha == 3 else _ALPHA2_MAP

    def _valid_codes(self) -> frozenset[str]:
        return _VALID_ALPHA3 if self.alpha == 3 else _VALID_ALPHA2

    @classmethod
    def _build_lookup_map(cls) -> Mapping[str, str]:
        # Base class caches per-subclass; for alpha-2 vs alpha-3 we use
        # instance-level _lookup_arrays override below.
        return _ALPHA2_MAP

    # Override: cache per (class, alpha) pair.
    _instance_lookup_cache: ClassVar[dict[int, tuple[dict[str, str], pa.Array, pa.Array]]] = {}

    def _lookup_arrays_instance(self) -> tuple[dict[str, str], pa.Array, pa.Array]:
        cached = ISOCountryType._instance_lookup_cache.get(self.alpha)
        if cached is not None:
            return cached
        mapping = self._current_map()
        keys = pa.array(list(mapping.keys()), type=pa.string())
        values = pa.array(list(mapping.values()), type=pa.string())
        cached = (mapping, keys, values)
        ISOCountryType._instance_lookup_cache[self.alpha] = cached
        return cached

    def _resolve_token(self, token: str) -> str | None:
        mapping = self._current_map()
        valid = self._valid_codes()
        if token in valid:
            return token
        direct = mapping.get(token)
        if direct is not None:
            return direct
        squashed = token.replace(" ", "")
        return mapping.get(squashed)

    # ------------------------------------------------------------------
    # Arrow override — uses the alpha-specific map.
    # ------------------------------------------------------------------
    def _resolve_arrow_string(self, array: pa.Array) -> pa.Array:
        normalized = self._normalize_arrow_string(array)
        _, keys, values = self._lookup_arrays_instance()
        indices = pc.index_in(normalized, value_set=keys)
        return pc.take(values, indices)

    # ------------------------------------------------------------------
    # Outgoing — country → numeric / string / other country-alpha.
    # Vectorized via pc.index_in + pc.take; no per-row Python.
    # ------------------------------------------------------------------
    _outgoing_numeric_cache: ClassVar[dict[int, tuple[pa.Array, pa.Array]]] = {}
    _outgoing_alpha_cache: ClassVar[dict[tuple[int, int], tuple[pa.Array, pa.Array]]] = {}

    @classmethod
    def _outgoing_numeric_arrays(cls, alpha: int) -> tuple[pa.Array, pa.Array]:
        cached = cls._outgoing_numeric_cache.get(alpha)
        if cached is not None:
            return cached
        mapping = _ALPHA3_TO_NUMERIC if alpha == 3 else _ALPHA2_TO_NUMERIC
        keys = pa.array(list(mapping.keys()), type=pa.string())
        values = pa.array(list(mapping.values()), type=pa.int32())
        cached = (keys, values)
        cls._outgoing_numeric_cache[alpha] = cached
        return cached

    @classmethod
    def _outgoing_alpha_arrays(cls, src_alpha: int, dst_alpha: int) -> tuple[pa.Array, pa.Array]:
        cached = cls._outgoing_alpha_cache.get((src_alpha, dst_alpha))
        if cached is not None:
            return cached
        if src_alpha == 2 and dst_alpha == 3:
            mapping = _ALPHA2_TO_ALPHA3
        elif src_alpha == 3 and dst_alpha == 2:
            mapping = _ALPHA3_TO_ALPHA2
        else:
            mapping = {k: k for k in (_VALID_ALPHA3 if src_alpha == 3 else _VALID_ALPHA2)}
        keys = pa.array(list(mapping.keys()), type=pa.string())
        values = pa.array(list(mapping.values()), type=pa.string())
        cached = (keys, values)
        cls._outgoing_alpha_cache[(src_alpha, dst_alpha)] = cached
        return cached

    def _outgoing_cast_arrow_array(
        self,
        array: pa.Array | pa.ChunkedArray,
        target: "DataType",
        options: "CastOptions",
    ) -> pa.Array | pa.ChunkedArray | None:
        target_id = target.type_id

        # Country → numeric ISO 3166-1 code.
        if target_id in {DataTypeId.INTEGER, DataTypeId.FLOAT, DataTypeId.DECIMAL}:
            keys, values = self._outgoing_numeric_arrays(self.alpha)
            return apply_arrow_lookup(
                array,
                keys,
                values,
                target.to_arrow(),
                memory_pool=options.arrow_memory_pool,
            )

        # Country → another ISOCountryType with different alpha width.
        if isinstance(target, ISOCountryType) and target.alpha != self.alpha:
            keys, values = self._outgoing_alpha_arrays(self.alpha, target.alpha)
            return apply_arrow_lookup(
                array,
                keys,
                values,
                target.to_arrow(),
                memory_pool=options.arrow_memory_pool,
            )

        return None

    # ------------------------------------------------------------------
    # Polars lazy override — alpha-specific map.
    # ------------------------------------------------------------------
    def _cast_polars_expr(
        self,
        expr: "polars.Expr",
        options,
    ):
        from yggdrasil.data.types.support import get_polars
        pl = get_polars()

        mapping = self._current_map()
        normalized = (
            expr.cast(pl.Utf8, strict=False)
            .str.to_uppercase()
            .str.replace_all(r"[^A-Z0-9]+", " ")
            .str.strip_chars()
        )
        return normalized.replace_strict(mapping, default=None, return_dtype=pl.Utf8)

    # ------------------------------------------------------------------
    # Spark lazy override — alpha-specific map.
    # ------------------------------------------------------------------
    def _cast_spark_column(self, column, options):
        from yggdrasil.data.types.support import get_spark_sql
        spark = get_spark_sql()
        F = spark.functions
        options.check_source(column)

        mapping = self._current_map()

        normalized = F.trim(
            F.regexp_replace(
                F.upper(column.cast(spark.types.StringType())),
                r"[^A-Z0-9]+",
                " ",
            )
        )

        if not mapping:
            return F.lit(None).cast(spark.types.StringType())

        map_args = []
        for k, v in mapping.items():
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
        return name in {"ISOCOUNTRYTYPE", "ISO_COUNTRY"} or iso == cls.iso_name

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ISOCountryType":
        alpha = int(value.get("alpha", 2))
        return cls(alpha=alpha)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["alpha"] = self.alpha
        return d

    def __repr__(self) -> str:
        return f"ISOCountryType(alpha={self.alpha})"

    def __str__(self) -> str:
        return f"iso_country({self.alpha})"
