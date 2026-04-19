"""Base class for ISO-coded string data types.

ISO types are string-backed extension types that normalize free-form
text into a canonical ISO code (or null when the value can't be
resolved).  Subclasses supply their own reference tables (codes +
aliases) and format constraints.

Casting contract
----------------
- ``safe=False`` (default): unparseable values become null.
- ``safe=True`` on eager inputs (Arrow arrays, Polars Series, Pandas
  Series): the first unparseable value raises ``ValueError``.
- ``safe=True`` on lazy inputs (``polars.Expr`` or ``pyspark.Column``):
  never raises because values aren't observable at call time — behaves
  like ``safe=False``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.support import get_pandas, get_polars, get_spark_sql
from yggdrasil.io import SaveMode

if TYPE_CHECKING:
    import pandas as pd
    import polars
    import pyspark.sql as ps
    import pyspark.sql.types as pst
    from yggdrasil.data.cast.options import CastOptions
    from yggdrasil.data.data_field import Field


__all__ = [
    "ISOType",
    "normalize_iso_token",
    "normalize_iso_token_keep_hyphen",
    "resolve_arrow_string_via_unique",
]


# Collapse internal whitespace to single spaces.
_WHITESPACE_RE = re.compile(r"\s+")
# Anything that isn't a letter or digit becomes a space.
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
# Keep hyphens (used by subdivisions/UN-LOCODE-like tokens).
_NON_ALNUM_KEEP_HYPHEN_RE = re.compile(r"[^A-Z0-9-]+")


def normalize_iso_token(value: Any) -> str | None:
    """Normalize a value for ISO lookup: strip, uppercase, collapse non-alnum to spaces.

    Returns None for values that can't become a usable token.
    """
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def normalize_iso_token_keep_hyphen(value: Any) -> str | None:
    """Like :func:`normalize_iso_token` but preserves hyphens.

    Used for subdivision/city codes where the hyphen is part of the
    canonical form (e.g. ``US-CA``).
    """
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    text = _NON_ALNUM_KEEP_HYPHEN_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def resolve_arrow_string_via_unique(
    array: pa.Array,
    resolver: Callable[[str], str | None],
) -> pa.Array:
    """Apply a Python resolver per *unique* value, then broadcast with ``pc.take``.

    This is the shared vectorized fallback for ISO types whose parsers
    don't fit the straight ``pc.index_in`` + ``pc.take`` mold (e.g.
    flexible city/subdivision tokenizers).  Dictionary-encoding collapses
    duplicates so the Python work scales with the cardinality of unique
    raw values instead of row count — typically 2-3 orders of magnitude
    smaller for real-world columns like country names.

    Nulls are preserved: ``pc.dictionary_encode`` puts them in the index
    array (not the dictionary), and ``pc.take`` propagates them through
    unchanged.
    """
    if len(array) == 0:
        return pa.array([], type=pa.string())

    encoded = pc.dictionary_encode(array)
    unique_tokens = encoded.dictionary.to_pylist()
    # ``dictionary_encode`` places non-null distinct values in the dictionary,
    # so the Python call count is bounded by the column's cardinality.
    resolved = [resolver(token) if token is not None else None for token in unique_tokens]
    resolved_arr = pa.array(resolved, type=pa.string())
    return pc.take(resolved_arr, encoded.indices)


@dataclass(frozen=True)
class ISOType(DataType):
    """Base class for ISO-coded string types.

    Concrete subclasses must implement :meth:`_resolve_token` (single-value
    lookup) and :attr:`_arrow_lookup_table` (bulk lookup for vectorized
    paths).  The class-level :attr:`iso_name` drives DDL, repr, and
    dict serialization.
    """

    # Subclasses override these:
    iso_name: ClassVar[str] = "iso"

    # ------------------------------------------------------------------
    # DataType protocol
    # ------------------------------------------------------------------
    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.STRING

    @property
    def children_fields(self) -> list["Field"]:
        return []

    # ------------------------------------------------------------------
    # Storage type — everyone is a plain utf-8 string under the hood.
    # ------------------------------------------------------------------
    def to_arrow(self) -> pa.DataType:
        return pa.string()

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.Utf8

    def to_spark(self) -> "pst.DataType":
        spark = get_spark_sql()
        return spark.types.StringType()

    def to_databricks_ddl(self) -> str:
        return "STRING"

    # ------------------------------------------------------------------
    # Nothing in Arrow/Polars/Spark identifies these uniquely.
    # They can only be obtained by explicit construction or from_str.
    # ------------------------------------------------------------------
    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return False

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        return False

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        return False

    # ------------------------------------------------------------------
    # Merge — two instances of the same class are interchangeable.
    # ------------------------------------------------------------------
    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "DataType":
        if type(self) is type(other):
            return self
        return self

    # ------------------------------------------------------------------
    # Single-value resolution — subclasses implement this.
    # ------------------------------------------------------------------
    def _resolve_token(self, token: str) -> str | None:
        """Return the canonical ISO code for *token* or None if unknown.

        *token* has already been uppercased and stripped of surrounding
        whitespace but preserves its interior structure.
        """
        raise NotImplementedError

    def _normalize(self, value: Any) -> str | None:
        """Pre-resolution normalization (stripped, upper, punctuation -> space)."""
        return normalize_iso_token(value)

    # ------------------------------------------------------------------
    # Bulk lookup map — subclasses supply this via _build_lookup_map.
    # Returns (keys_array, values_array) suitable for pc.index_in / pc.take.
    # ------------------------------------------------------------------
    @classmethod
    def _build_lookup_map(cls) -> Mapping[str, str]:
        """Return a mapping from normalized token -> canonical ISO code.

        Must be deterministic and include every accepted alias so that
        vectorized lookup matches the single-value path.
        """
        raise NotImplementedError

    _lookup_cache: ClassVar[dict[type, tuple[Mapping[str, str], pa.Array, pa.Array]]] = {}

    @classmethod
    def _lookup_arrays(cls) -> tuple[Mapping[str, str], pa.Array, pa.Array]:
        cached = cls._lookup_cache.get(cls)
        if cached is not None:
            return cached
        mapping = dict(cls._build_lookup_map())
        keys = pa.array(list(mapping.keys()), type=pa.string())
        values = pa.array(list(mapping.values()), type=pa.string())
        cls._lookup_cache[cls] = (mapping, keys, values)
        return cls._lookup_cache[cls]

    # ------------------------------------------------------------------
    # Vectorized Arrow cast
    # ------------------------------------------------------------------
    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.Array:
        safe = bool(getattr(options, "safe", False))

        # If the source isn't already a string, funnel it through cast
        # (safe=False so unconvertible scalars become null rather than
        # raising — matches the "lazy tolerant" policy).
        if not (pa.types.is_string(array.type) or pa.types.is_large_string(array.type)):
            try:
                array = pc.cast(array, pa.string(), safe=False)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                if safe:
                    raise
                return pa.nulls(len(array), type=pa.string())

        resolved = self._resolve_arrow_string(array)

        if safe:
            orig_valid = pc.is_valid(array)
            failed = pc.and_(orig_valid, pc.invert(pc.is_valid(resolved)))
            if pc.any(failed).as_py():
                failed_indices = pc.indices_nonzero(failed)
                idx = failed_indices[0].as_py()
                raw = array[idx].as_py()
                raise ValueError(
                    f"Cannot parse {type(self).__name__} from {raw!r} at index {idx}. "
                    f"Pass safe=False to null out unparseable values."
                )

        return resolved

    def _resolve_arrow_string(
        self,
        array: pa.Array,
    ) -> pa.Array:
        """Resolve an Arrow string array to canonical ISO codes.

        Nulls are preserved; unresolvable values become null.
        """
        normalized = self._normalize_arrow_string(array)
        _, keys, values = self._lookup_arrays()
        indices = pc.index_in(normalized, value_set=keys)
        return pc.take(values, indices)

    def _normalize_arrow_string(self, array: pa.Array) -> pa.Array:
        """Default vectorized normalization: upper-case, collapse non-alnum to space, trim.

        Subclasses can override for format-specific rules.
        """
        upper = pc.utf8_upper(array)
        # Replace every run of characters that aren't A-Z or 0-9 with a
        # single space, then trim surrounding whitespace.
        collapsed = pc.replace_substring_regex(upper, pattern=r"[^A-Z0-9]+", replacement=" ")
        trimmed = pc.utf8_trim_whitespace(collapsed)
        return trimmed

    # ------------------------------------------------------------------
    # Polars (eager Series path)
    # ------------------------------------------------------------------
    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ):
        pl = get_polars()
        options.check_source(series)
        safe = bool(getattr(options, "safe", False))

        # Normalize source to utf-8 by rounding through Arrow.
        try:
            arrow = series.to_arrow()
        except Exception:
            arrow = pa.array(series.to_list(), type=pa.string())

        resolved = self._cast_arrow_array(
            arrow,
            options.copy(safe=safe),
        )
        out = pl.Series(name=series.name, values=resolved, dtype=pl.Utf8)
        return out

    # ------------------------------------------------------------------
    # Polars (lazy Expr path) — never raises: unknowns silently become null.
    # ------------------------------------------------------------------
    def _cast_polars_expr(
        self,
        expr: "polars.Expr",
        options: "CastOptions",
    ):
        pl = get_polars()
        mapping, _, _ = self._lookup_arrays()

        normalized = (
            expr.cast(pl.Utf8, strict=False)
            .str.to_uppercase()
            .str.replace_all(r"[^A-Z0-9]+", " ")
            .str.strip_chars()
        )
        # replace_strict: unknown -> default (None here => null)
        return normalized.replace_strict(mapping, default=None, return_dtype=pl.Utf8)

    # ------------------------------------------------------------------
    # Pandas — round-trip through Arrow.
    # ------------------------------------------------------------------
    def _cast_pandas_series(
        self,
        series: "pd.Series",
        options: "CastOptions",
    ):
        pd = get_pandas()
        options.check_source(series)

        try:
            arrow = pa.Array.from_pandas(series)
        except Exception:
            arrow = pa.array(series.tolist(), from_pandas=True)

        resolved = self._cast_arrow_array(arrow, options)

        out = resolved.to_pandas(types_mapper=None)
        if not isinstance(out, pd.Series):
            out = pd.Series(out, index=series.index, name=series.name)
        else:
            out.index = series.index
            out.name = series.name
        return out

    # ------------------------------------------------------------------
    # Spark (lazy Column path) — never raises.
    # ------------------------------------------------------------------
    def _cast_spark_column(
        self,
        column: "ps.Column",
        options: "CastOptions",
    ):
        spark = get_spark_sql()
        F = spark.functions
        options.check_source(column)

        mapping, _, _ = self._lookup_arrays()

        # Normalize: cast to string, uppercase, replace non-alnum with space, trim.
        normalized = F.trim(
            F.regexp_replace(
                F.upper(column.cast(spark.types.StringType())),
                r"[^A-Z0-9]+",
                " ",
            )
        )

        # Build a literal map and look up the normalized token. Unknowns
        # become NULL via element_at semantics.
        if not mapping:
            return F.lit(None).cast(spark.types.StringType())

        map_args: list[Any] = []
        for k, v in mapping.items():
            map_args.append(F.lit(k))
            map_args.append(F.lit(v))
        lookup_map = F.create_map(*map_args)

        return F.element_at(lookup_map, normalized)

    # ------------------------------------------------------------------
    # Python-object conversion — delegates to _resolve_token.
    # ------------------------------------------------------------------
    def convert_pyobj(
        self,
        value: Any,
        nullable: bool,
        safe: bool = False,
    ) -> str | None:
        if value is None:
            if nullable:
                return None
            raise ValueError(
                f"Got None for a non-nullable {type(self).__name__} field."
            )

        token = self._normalize(value)
        if token is not None:
            resolved = self._resolve_token(token)
            if resolved is not None:
                return resolved

        if safe:
            raise ValueError(
                f"Cannot parse {type(self).__name__} from {value!r}."
            )
        if not nullable:
            raise ValueError(
                f"Cannot parse {type(self).__name__} from {value!r} and field "
                "is not nullable."
            )
        return None

    def _convert_pyobj(self, value: Any, safe: bool = False) -> str | None:
        token = self._normalize(value)
        if token is not None:
            resolved = self._resolve_token(token)
            if resolved is not None:
                return resolved
        if safe:
            raise ValueError(f"Cannot parse {type(self).__name__} from {value!r}.")
        return None

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    def default_pyobj(self, nullable: bool) -> str | None:
        if nullable:
            return None
        raise NotImplementedError(
            f"{type(self).__name__}.default_pyobj(nullable=False) is not supported."
        )

    def default_arrow_scalar(self, nullable: bool = True) -> pa.Scalar:
        if nullable:
            return pa.scalar(None, type=pa.string())
        raise NotImplementedError(
            f"{type(self).__name__}.default_arrow_scalar(nullable=False) is not supported."
        )

    # ------------------------------------------------------------------
    # Dict round-trip
    # ------------------------------------------------------------------
    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        if cls is ISOType:
            return False
        name = str(value.get("name", "")).upper()
        iso = str(value.get("iso", "")).lower()
        return name == cls.__name__.upper() or iso == cls.iso_name.lower()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ISOType":
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.type_id),
            "name": type(self).__name__,
            "iso": type(self).iso_name,
        }

    # ------------------------------------------------------------------
    # Repr / str
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __str__(self) -> str:
        return type(self).iso_name
