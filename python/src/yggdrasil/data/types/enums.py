"""DictionaryType — to be appended to ``yggdrasil/data/types/primitive.py``.

This snippet assumes it lives inside ``primitive.py`` itself; all imports
(``pa``, ``pc``, ``re``, ``decimal``, ``PrimitiveType``, ``StringType``,
``DataType``, ``DataTypeId``, ``Mode``, ``get_polars``, ``get_spark_sql``,
the ``CastOptions`` / ``polars`` / ``pst`` / ``Field`` TYPE_CHECKING aliases,
and the ``__all__`` list) are already in scope from the surrounding module.

When moving this into place:

* Append these lines to ``__all__`` at the top of ``primitive.py``:

      "DictionaryType",

* Paste the class below after ``DurationType`` (end of file).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data import DataType, DataTypeId
from yggdrasil.data.types import PrimitiveType, StringType
from yggdrasil.data.types.support import get_spark_sql, get_polars
from yggdrasil.data.enums import Mode


if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst
    from yggdrasil.data.options import CastOptions

# ---------------------------------------------------------------------------
# DictionaryType
# ---------------------------------------------------------------------------
#
# Dictionary-encoded categorical primitive. Strict enum-like semantics on the
# outside (fixed ``categories`` tuple, unknown values cast to null on ingest)
# and ``pa.dictionary(int32, value_type)`` on the Arrow side.
#
# Engine mapping:
#   * Arrow:  pa.dictionary(int32, value_type) with ``categories`` baked in.
#   * Polars: pl.Enum(categories) when value_type is StringType. Polars Enum
#             is string-only (``pl.Enum([1,2,3])`` raises TypeError), so for
#             non-string value types we degrade to ``value_type.to_polars()``
#             — the encoding is lost, the values survive.
#   * Spark:  no dictionary-encoded type exists → degrade to
#             ``value_type.to_spark()`` and gate unknowns to null via CASE WHEN.
#
# Index width is fixed at int32 to match Arrow's default and keep the wire
# format stable across small / large category sets.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DictionaryType(PrimitiveType):

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * indent * level
        return f"{pad}dictionary<{self.value_type.pretty_format(indent=indent, level=level + 1)}>"

    value_type: PrimitiveType = None  # defaulted to StringType() in __post_init__
    categories: tuple[Any, ...] = ()
    ordered: bool = False

    def __str__(self):
        n = len(self.categories)
        preview = ", ".join(repr(c) for c in self.categories[:3])
        more = f", …(+{n - 3})" if n > 3 else ""
        ord_tag = ", ordered" if self.ordered else ""
        return f"dictionary<{self.value_type}, [{preview}{more}]{ord_tag}>"

    def __repr__(self):
        return self.__str__()

    def __post_init__(self):
        # Default value_type. We can't use ``field(default_factory=StringType)``
        # without importing it at module load time in a way that hurts the
        # existing linear order — StringType is defined above us in the same
        # file, so resolving it here is fine and matches DecimalType's style.
        vt = self.value_type
        if vt is None:
            vt = StringType()
        elif not isinstance(vt, DataType):
            vt = DataType.from_any(vt)

        if not isinstance(vt, PrimitiveType):
            raise TypeError(
                f"DictionaryType.value_type must be a PrimitiveType, "
                f"got {type(vt).__name__}: {vt!r}"
            )
        if isinstance(vt, DictionaryType):
            raise TypeError("DictionaryType cannot be nested inside itself")

        # Coerce + dedup categories in first-seen order. Accept any iterable
        # on the way in; store a tuple so the dataclass stays hashable.
        raw = self.categories
        if raw is None:
            raise ValueError(
                "DictionaryType requires a non-empty `categories` iterable"
            )

        coerced: list[Any] = []
        seen: set[Any] = set()
        for item in raw:
            # Reject None on the raw input — letting it flow through
            # convert_pyobj(nullable=False) would substitute value_type's
            # non-null default, silently promoting None to "" / 0 / etc.
            if item is None:
                raise ValueError(
                    "DictionaryType categories cannot contain None — use a "
                    "nullable Field on the containing column to represent absence"
                )
            value = vt.convert_pyobj(item, nullable=False, safe=False)
            if value is None:
                # Coercion failed on a non-None input (lenient path returned
                # None). Surface it explicitly rather than silently dropping.
                raise ValueError(
                    f"DictionaryType category {item!r} could not be coerced "
                    f"to {type(vt).__name__}"
                )
            try:
                if value in seen:
                    continue
                seen.add(value)
            except TypeError:
                # Unhashable primitive value (e.g. bytearray) — equality scan.
                if any(existing == value for existing in coerced):
                    continue
            coerced.append(value)

        if not coerced:
            raise ValueError("DictionaryType categories cannot be empty")

        # Default byte_size: width of the packed {index + dictionary slot}
        # row, in line with how IntegerType / DecimalType expose byte_size.
        # int32 indices are always 4 bytes; per-category storage lives on
        # value_type.byte_size when that type carries one.
        object.__setattr__(self, "value_type", vt)
        object.__setattr__(self, "categories", tuple(coerced))
        object.__setattr__(self, "ordered", bool(self.ordered))
        if self.byte_size is None:
            object.__setattr__(self, "byte_size", 4)

    # ------------------------------------------------------------------ core

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.DICTIONARY

    @property
    def cardinality(self) -> int:
        return len(self.categories)

    # ---------------------------------------------------------------- merge

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: Mode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "DictionaryType":
        if not isinstance(other, DictionaryType):
            raise TypeError(
                f"Cannot merge DictionaryType with {other.__class__.__name__}"
            )
        if downcast == upcast:
            return self

        merged_value = self.value_type.merge_with(
            other.value_type, mode=mode, downcast=downcast, upcast=upcast
        )
        if not isinstance(merged_value, PrimitiveType):
            raise TypeError(
                f"DictionaryType value_type merge yielded non-primitive "
                f"{merged_value!r}; cannot construct a merged DictionaryType"
            )

        if downcast:
            # Intersection of categories (in self's order) — narrower enum.
            other_set = set(other.categories) if self._categories_hashable() else None
            if other_set is not None:
                cats = tuple(c for c in self.categories if c in other_set)
            else:
                cats = tuple(c for c in self.categories if c in other.categories)
            if not cats:
                # Degenerate intersection — keep self to preserve the schema
                # contract (non-empty categories).
                cats = self.categories
        else:
            # Union in first-seen order (self wins ties).
            seen: set[Any] = set()
            cats_list: list[Any] = []
            for cat in (*self.categories, *other.categories):
                try:
                    if cat in seen:
                        continue
                    seen.add(cat)
                except TypeError:
                    if any(existing == cat for existing in cats_list):
                        continue
                cats_list.append(cat)
            cats = tuple(cats_list)

        ordered = (
            self.ordered
            and other.ordered
            and self.categories == other.categories
        )

        return DictionaryType(
            value_type=merged_value,
            categories=cats,
            ordered=ordered,
        )

    def _categories_hashable(self) -> bool:
        try:
            hash(self.categories)
        except TypeError:
            return False
        return True

    # ------------------------------------------------------------------ arrow

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_dictionary(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "DictionaryType":
        # A bare pa.DataType doesn't carry the category values; a
        # DictionaryType without categories would be a lie. Callers that have
        # a real dictionary array should use from_arrow_array below.
        raise TypeError(
            f"Cannot construct DictionaryType from a pa.DataType alone "
            f"({dtype!r}) — the category values are not carried by the type. "
            "Use DictionaryType.from_arrow_array(arr) or build explicitly "
            "with categories=..."
        )

    @classmethod
    def from_arrow_array(
        cls, array: pa.Array | pa.ChunkedArray
    ) -> "DictionaryType":
        if isinstance(array, pa.ChunkedArray):
            if array.num_chunks == 0:
                raise ValueError(
                    "Cannot infer DictionaryType from empty ChunkedArray"
                )
            array = array.combine_chunks()
        if not pa.types.is_dictionary(array.type):
            raise TypeError(
                f"from_arrow_array expected a dictionary-typed array, "
                f"got {array.type!r}"
            )
        value_type = DataType.from_arrow_type(array.type.value_type)
        if not isinstance(value_type, PrimitiveType):
            raise TypeError(
                f"DictionaryType value_type must be primitive, got {value_type!r}"
            )
        return cls(
            value_type=value_type,
            categories=tuple(array.dictionary.to_pylist()),
            ordered=array.type.ordered,
        )

    def to_arrow(self) -> pa.DataType:
        return pa.dictionary(
            index_type=pa.int32(),
            value_type=self.value_type.to_arrow(),
            ordered=self.ordered,
        )

    def _categories_arrow(self) -> pa.Array:
        return pa.array(self.categories, type=self.value_type.to_arrow())

    # ----------------------------------------------------------------- polars

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        try:
            return isinstance(dtype, pl.Enum) or dtype == pl.Enum
        except Exception:
            return False

    @classmethod
    def from_polars_type(cls, dtype: "polars.Enum") -> "DictionaryType":
        pl = get_polars()
        if not isinstance(dtype, pl.Enum):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        return cls(
            value_type=StringType(),
            categories=tuple(dtype.categories.to_list()),
        )

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        if isinstance(self.value_type, StringType):
            return pl.Enum(list(self.categories))
        # Polars Enum is string-only — degrade to the value dtype.
        return self.value_type.to_polars()

    # ------------------------------------------------------------------ spark

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        # Spark has no dictionary-encoded type. Let value-type handlers claim
        # whatever the values look like natively.
        return False

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "DictionaryType":
        raise TypeError(
            f"Spark has no native dictionary-encoded type ({dtype!r}). "
            "Cast from the Spark value type and reconstruct DictionaryType "
            "with explicit categories."
        )

    def to_spark(self) -> Any:
        # Degrade to the value type — the encoding is lost, values survive.
        return self.value_type.to_spark()

    def to_spark_name(self) -> str:
        return self.value_type.to_spark_name()

    # ------------------------------------------------------------------ dict

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.DICTIONARY, "DICT", "ENCODED")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DictionaryType":
        value_type_payload = value.get("value_type") or value.get("valueType")
        if value_type_payload is None:
            raise ValueError(
                f"DictionaryType payload missing 'value_type': {value!r}"
            )
        categories = value.get("categories")
        if categories is None:
            raise ValueError(
                f"DictionaryType payload missing 'categories': {value!r}"
            )
        return cls(
            byte_size=value.get("byte_size"),
            value_type=DataType.from_any(value_type_payload),
            categories=tuple(categories),
            ordered=bool(value.get("ordered", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["value_type"] = self.value_type.to_dict()
        base["categories"] = list(self.categories)
        base["ordered"] = self.ordered
        return base

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        tags[b"encoding"] = b"dictionary"
        tags[b"value_kind"] = self.value_type.type_id.name.lower().encode("utf-8")
        tags[b"cardinality"] = str(len(self.categories)).encode("utf-8")
        tags[b"ordered"] = b"true" if self.ordered else b"false"
        return tags

    # --------------------------------------------------------------- pyobj

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        # First category is the deterministic non-null default, matching
        # strict enum semantics.
        return self.categories[0]

    def default_arrow_scalar(self, nullable: bool = True) -> pa.Scalar:
        if nullable:
            return pa.scalar(None, type=self.to_arrow())
        arr = pa.DictionaryArray.from_arrays(
            indices=pa.array([0], type=pa.int32()),
            dictionary=self._categories_arrow(),
            ordered=self.ordered,
        )
        return arr[0]

    def default_polars_scalar(self, nullable: bool = True) -> Any:
        return None if nullable else self.categories[0]

    def default_spark_scalar(self, nullable: bool = True) -> Any:
        return None if nullable else self.categories[0]

    def _convert_pyobj(self, value: Any, safe: bool = False) -> Any:
        coerced = self.value_type.convert_pyobj(value, nullable=True, safe=safe)
        if coerced is None:
            return None
        try:
            in_categories = coerced in self.categories
        except TypeError:
            # Unhashable — fall back to linear scan.
            in_categories = any(existing == coerced for existing in self.categories)
        if in_categories:
            return coerced
        if safe:
            raise ValueError(
                f"Value {value!r} is not a member of DictionaryType categories "
                f"{self.categories!r}"
            )
        return None  # lenient: unknown → null

    # ---------------------------------------------------------------- casts

    def _cast_arrow_array(
        self,
        array: pa.Array | pa.DictionaryArray,
        options: "CastOptions",
    ) -> pa.Array:
        """Encode raw values against our known categories.

        Unknowns → null (lenient). When the source is already a dictionary we
        re-encode its values so indices line up with *our* category order.
        """
        options = options.check_source(array)

        # Already-encoded under the same value type + category order + int32
        # indices: identity short-circuit.
        if pa.types.is_dictionary(array.type):
            src_dict = array.dictionary
            if (
                src_dict.type == self.value_type.to_arrow()
                and src_dict.to_pylist() == list(self.categories)
            ):
                if array.type.index_type == pa.int32():
                    return options.fill_arrow_nulls(array)
                return options.fill_arrow_nulls(
                    pa.DictionaryArray.from_arrays(
                        indices=pc.cast(array.indices, pa.int32()),
                        dictionary=src_dict,
                        ordered=self.ordered,
                    )
                )
            # Different dictionary — decode to values, then re-encode below.
            array = pc.cast(array, self.value_type.to_arrow())

        # Promote to value type first so pc.index_in compares like-for-like.
        if array.type != self.value_type.to_arrow():
            array = self.value_type._cast_arrow_array(array, options)

        categories_array = self._categories_arrow()
        # Member → position in categories, non-member → null (lenient).
        idx = pc.index_in(array, value_set=categories_array)
        idx = pc.cast(idx, pa.int32())

        encoded = pa.DictionaryArray.from_arrays(
            indices=idx,
            dictionary=categories_array,
            ordered=self.ordered,
        )
        return options.fill_arrow_nulls(encoded)

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ):
        pl = get_polars()
        target_dtype = self.to_polars()

        # strict=False on an Enum cast is exactly our lenient semantics:
        # unknown values map to null. For non-string value types we fall
        # through to the base pipeline (which runs on the raw value dtype).
        if isinstance(target_dtype, pl.Enum):
            casted = series.cast(target_dtype, strict=False)
            return self.fill_polars_array_nulls(
                casted, nullable=self._target_nullable(options)
            )

        return super()._cast_polars_series(series, options)

    def _cast_polars_expr(
        self,
        expr: "polars.Expr",
        options: "CastOptions",
    ):
        pl = get_polars()
        target_dtype = self.to_polars()

        if isinstance(target_dtype, pl.Enum):
            casted = expr.cast(target_dtype, strict=False)
            return self.fill_polars_array_nulls(
                casted, nullable=self._target_nullable(options)
            )

        return super()._cast_polars_expr(expr, options)

    def _cast_spark_column(
        self,
        column: Any,
        options: "CastOptions",
    ):
        spark = get_spark_sql()
        F = spark.functions
        options = options.check_source(column)

        # Cast to the value type first, then gate unknowns → null via
        # CASE WHEN value IN (...) THEN value ELSE NULL. Fine for realistic
        # category counts; very large category sets should use a broadcast
        # join rather than this type.
        value_spark = self.value_type.to_spark()
        casted = column.cast(value_spark)

        condition = None
        for cat in self.categories:
            eq = casted == F.lit(cat)
            condition = eq if condition is None else (condition | eq)

        gated = F.when(condition, casted).otherwise(F.lit(None).cast(value_spark))
        return self.fill_spark_column_nulls(
            gated, nullable=self._target_nullable(options)
        )