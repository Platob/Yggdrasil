"""DictionaryType — dictionary-encoded categorical primitive.

The Arrow side rides on ``pa.dictionary(int32, value_type)`` with the
``categories`` baked in. Polars maps to ``pl.Enum`` when the value
type is string (``pl.Enum`` is string-only — ``pl.Enum([1,2,3])``
raises ``TypeError``), and degrades to ``value_type.to_polars()`` for
non-string value types. Spark has no dictionary-encoded type at all,
so we degrade to ``value_type.to_spark()`` and gate unknowns to
null via ``CASE WHEN value IN (...) THEN value ELSE NULL``.

Index width is fixed at int32 to match Arrow's default and keep the
wire format stable across small / large category sets.

Strict enum-like semantics: an empty ``categories`` tuple means
"open dictionary" (any value of the underlying ``value_type`` is
accepted, useful for round-tripping a ``pa.dictionary`` whose values
aren't known at schema-declaration time); a non-empty ``categories``
tuple closes the set — anything outside the set casts to null on
ingest (lenient default) or raises (``safe=True``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from ..id import DataTypeId
from ..primitive.base import PrimitiveType
from ..primitive.string import StringType
from ..support import get_polars, get_spark_sql

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst
    from ...options import CastOptions
    from ..base import DataType


__all__ = ["DictionaryType"]


@dataclass(frozen=True, repr=False)
class DictionaryType(PrimitiveType):
    """Dictionary-encoded categorical primitive.

    ``value_type`` defaults to :class:`StringType` (the most common
    case). ``categories`` is an iterable of values that get coerced
    through ``value_type.convert_pyobj`` and de-duplicated in
    first-seen order; an empty tuple means "open dictionary".
    ``ordered`` flips the Arrow ``ordered`` flag (and is the only
    field whose merge requires a category-tuple match).
    """

    value_type: PrimitiveType = None  # defaulted to StringType() in __post_init__
    categories: tuple[Any, ...] = ()
    ordered: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __post_init__(self):
        # Resolve ``value_type`` first so category coercion below has
        # the right rules. ``DataType.from_any`` is imported lazily to
        # dodge the primitive-package bootstrap cycle.
        vt = self.value_type
        if vt is None:
            vt = StringType()
        elif not isinstance(vt, PrimitiveType):
            from ..base import DataType

            vt = DataType.from_any(vt)

        if not isinstance(vt, PrimitiveType):
            raise TypeError(
                f"DictionaryType.value_type must be a PrimitiveType, "
                f"got {type(vt).__name__}: {vt!r}"
            )
        if isinstance(vt, DictionaryType):
            raise TypeError("DictionaryType cannot be nested inside itself")

        coerced = self._coerce_categories(self.categories, vt)
        object.__setattr__(self, "value_type", vt)
        object.__setattr__(self, "categories", coerced)
        object.__setattr__(self, "ordered", bool(self.ordered))
        if self.byte_size is None:
            # Index width — int32 indices are always 4 bytes; per-
            # category storage lives on ``value_type.byte_size`` when
            # that type carries one.
            object.__setattr__(self, "byte_size", 4)

    @staticmethod
    def _coerce_categories(
        raw: Any, value_type: PrimitiveType
    ) -> tuple[Any, ...]:
        if raw is None:
            return ()

        coerced: list[Any] = []
        seen: set[Any] = set()
        for item in raw:
            if item is None:
                # Allowing ``None`` here would silently substitute the
                # value_type's non-null default; nullability belongs on
                # the containing :class:`Field`, not in the value set.
                raise ValueError(
                    "DictionaryType categories cannot contain None — use "
                    "a nullable Field on the containing column to "
                    "represent absence"
                )
            value = value_type.convert_pyobj(item, nullable=False, safe=False)
            if value is None:
                raise ValueError(
                    f"DictionaryType category {item!r} could not be "
                    f"coerced to {type(value_type).__name__}"
                )
            try:
                if value in seen:
                    continue
                seen.add(value)
            except TypeError:
                # Unhashable primitive value (e.g. bytearray) — fall
                # back to a linear equality scan.
                if any(existing == value for existing in coerced):
                    continue
            coerced.append(value)

        return tuple(coerced)

    # ------------------------------------------------------------------
    # Identity / formatting
    # ------------------------------------------------------------------

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * indent * level
        return (
            f"{pad}{self._head_name()}<"
            f"{self.value_type.pretty_format(indent=indent, level=0)}>"
        )

    def _head_name(self) -> str:
        return "dictionary"

    def __str__(self):
        n = len(self.categories)
        preview = ", ".join(repr(c) for c in self.categories[:3])
        more = f", …(+{n - 3})" if n > 3 else ""
        ord_tag = ", ordered" if self.ordered else ""
        cats = f"[{preview}{more}]" if n else "[]"
        return f"{self._head_name()}<{self.value_type}, {cats}{ord_tag}>"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.DICTIONARY

    @property
    def cardinality(self) -> int:
        return len(self.categories)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode=None,
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
            # Intersection in self's order — narrower enum.
            other_set = (
                set(other.categories) if self._categories_hashable() else None
            )
            if other_set is not None:
                cats = tuple(c for c in self.categories if c in other_set)
            else:
                cats = tuple(c for c in self.categories if c in other.categories)
            if not cats and (self.categories or other.categories):
                # Empty intersection between two non-empty sets keeps
                # self's order to preserve the schema contract; an
                # already-open dictionary just propagates as open.
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

        return type(self)(
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

    # ------------------------------------------------------------------
    # Arrow
    # ------------------------------------------------------------------

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_dictionary(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "DictionaryType":
        # A bare ``pa.DataType`` doesn't carry the category values — a
        # closed-set DictionaryType without ``categories`` would lie
        # about its own contract. Return an open dictionary (empty
        # ``categories``) so ``DataType.from_arrow_type`` round-trips
        # at the type level; callers that have an actual array should
        # use :meth:`from_arrow_array` to recover the values.
        if not pa.types.is_dictionary(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

        from ..base import DataType

        value_type = DataType.from_arrow_type(dtype.value_type)
        if not isinstance(value_type, PrimitiveType):
            raise TypeError(
                f"DictionaryType value_type must be primitive, got "
                f"{value_type!r}"
            )
        return cls(
            value_type=value_type,
            categories=(),
            ordered=bool(dtype.ordered),
        )

    @classmethod
    def from_arrow_array(
        cls, array: "pa.Array | pa.ChunkedArray"
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

        from ..base import DataType

        value_type = DataType.from_arrow_type(array.type.value_type)
        if not isinstance(value_type, PrimitiveType):
            raise TypeError(
                f"DictionaryType value_type must be primitive, got "
                f"{value_type!r}"
            )
        # ``categories`` is bounded type metadata (the enum's domain —
        # cardinality is the dictionary's category count, not the array's
        # length), so the to_pylist here is type-construction, not a
        # data-path materialisation.  Runs once per ``from_arrow_array``
        # call against a small (≤ thousands of entries) dictionary buffer.
        return cls(
            value_type=value_type,
            categories=tuple(array.dictionary.to_pylist()),
            ordered=bool(array.type.ordered),
        )

    def to_arrow(self) -> pa.DataType:
        return pa.dictionary(
            index_type=pa.int32(),
            value_type=self.value_type.to_arrow(),
            ordered=self.ordered,
        )

    def _categories_arrow(self) -> pa.Array:
        return pa.array(self.categories, type=self.value_type.to_arrow())

    # ------------------------------------------------------------------
    # Polars
    # ------------------------------------------------------------------

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        try:
            return isinstance(dtype, pl.Enum) or dtype == pl.Enum
        except Exception:
            return False

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "DictionaryType":
        pl = get_polars()
        if not (isinstance(dtype, pl.Enum) or dtype == pl.Enum):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")
        # ``pl.Enum`` is the dtype class itself when not parameterized
        # (no categories yet); fall through to an empty open dict.
        categories: tuple[Any, ...] = ()
        try:
            cats_series = dtype.categories
        except AttributeError:
            cats_series = None
        if cats_series is not None:
            # Same shape as ``from_arrow_array``: ``categories`` is
            # bounded enum metadata, not row-shaped data — the to_list
            # crossing is type-construction.
            try:
                categories = tuple(cats_series.to_list())
            except AttributeError:
                categories = tuple(cats_series)
        return cls(
            value_type=StringType(),
            categories=categories,
        )

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        if isinstance(self.value_type, StringType) and self.categories:
            return pl.Enum(list(self.categories))
        # Polars Enum is string-only and requires a non-empty category
        # set — degrade to the value dtype for everything else.
        return self.value_type.to_polars()

    # ------------------------------------------------------------------
    # Spark
    # ------------------------------------------------------------------

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        # Spark has no dictionary-encoded type. Let the value-type
        # handlers claim whatever the values look like natively.
        return False

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "DictionaryType":
        raise TypeError(
            f"Spark has no native dictionary-encoded type ({dtype!r}). "
            "Cast from the Spark value type and reconstruct DictionaryType "
            "with explicit categories."
        )

    def to_spark(self) -> Any:
        # Degrade to the value type — the encoding is lost, values
        # survive.
        return self.value_type.to_spark()

    def to_spark_name(self) -> str:
        return self.value_type.to_spark_name()

    # ------------------------------------------------------------------
    # Dict
    # ------------------------------------------------------------------

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(
            value, cls.class_type_id(), "DICT", "ENCODED", "CATEGORICAL"
        )

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], default: Any = ...
    ) -> "DictionaryType":
        try:
            from ..base import DataType

            value_type_payload = (
                value.get("value_type") or value.get("valueType")
            )
            if value_type_payload is not None:
                value_type = DataType.from_any(value_type_payload)
            else:
                value_type = StringType()

            categories = value.get("categories") or ()
            return cls(
                byte_size=value.get("byte_size"),
                value_type=value_type,
                categories=tuple(categories),
                ordered=bool(value.get("ordered", False)),
            )
        except Exception as e:
            if default is ...:
                raise ValueError(
                    f"Cannot construct {cls.__name__} from {value!r}"
                ) from e
            return default

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["value_type"] = self.value_type.to_dict()
        base["categories"] = list(self.categories)
        base["ordered"] = self.ordered
        return base

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        tags[b"encoding"] = b"dictionary"
        tags[b"value_kind"] = (
            self.value_type.type_id.name.lower().encode("utf-8")
        )
        tags[b"cardinality"] = str(len(self.categories)).encode("utf-8")
        tags[b"ordered"] = b"true" if self.ordered else b"false"
        return tags

    # ------------------------------------------------------------------
    # Pyobj
    # ------------------------------------------------------------------

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        if self.categories:
            # First category is the deterministic non-null default —
            # matches strict enum semantics.
            return self.categories[0]
        return self.value_type.default_pyobj(nullable=False)

    def default_arrow_scalar(self, nullable: bool = True) -> pa.Scalar:
        if nullable:
            return pa.scalar(None, type=self.to_arrow())
        if not self.categories:
            # Open dictionary — no canonical default value to encode.
            return pa.scalar(None, type=self.to_arrow())
        arr = pa.DictionaryArray.from_arrays(
            indices=pa.array([0], type=pa.int32()),
            dictionary=self._categories_arrow(),
            ordered=self.ordered,
        )
        return arr[0]

    def default_polars_scalar(self, nullable: bool = True) -> Any:
        if nullable or not self.categories:
            return None
        return self.categories[0]

    def default_spark_scalar(self, nullable: bool = True) -> Any:
        if nullable or not self.categories:
            return None
        return self.categories[0]

    def _convert_pyobj(self, value: Any, safe: bool = False) -> Any:
        coerced = self.value_type.convert_pyobj(value, nullable=True, safe=safe)
        if coerced is None:
            return None
        if not self.categories:
            # Open dictionary — accept anything the value type accepts.
            return coerced
        try:
            in_categories = coerced in self.categories
        except TypeError:
            in_categories = any(
                existing == coerced for existing in self.categories
            )
        if in_categories:
            return coerced
        if safe:
            raise ValueError(
                f"Value {value!r} is not a member of {type(self).__name__} "
                f"categories {self.categories!r}"
            )
        return None  # lenient: unknown → null

    # ------------------------------------------------------------------
    # Casts
    # ------------------------------------------------------------------

    def _cast_arrow_array(
        self,
        array: "pa.Array | pa.DictionaryArray",
        options: "CastOptions",
    ) -> pa.Array:
        """Encode raw values against our known categories.

        Unknowns → null (lenient). When the source is already a
        dictionary we re-encode its values so indices line up with
        *our* category order. With no categories declared, we
        round-trip through Arrow's plain cast (open dictionary).
        """
        options = options.check_source(array)
        nullable = self._target_nullable(options)

        # Open dictionary — no category set to encode against. Promote
        # values through the value_type cast and let pyarrow's own
        # cast wrap them in a dictionary array.
        if not self.categories:
            if pa.types.is_dictionary(array.type):
                if array.type == self.to_arrow():
                    return self.fill_arrow_array_nulls(array, nullable=nullable)
                # Different value_type or ordered flag — decode to
                # values and rebuild.
                array = pc.cast(array, self.value_type.to_arrow())
            else:
                array = self.value_type._cast_arrow_array(array, options)
            casted = pc.cast(array, self.to_arrow(), safe=options.safe)
            return self.fill_arrow_array_nulls(casted, nullable=nullable)

        # Already-encoded under the same value type + category order
        # + int32 indices: identity short-circuit.
        if pa.types.is_dictionary(array.type):
            src_dict = array.dictionary
            if (
                src_dict.type == self.value_type.to_arrow()
                and src_dict.equals(self._categories_arrow())
            ):
                if array.type.index_type == pa.int32():
                    return self.fill_arrow_array_nulls(
                        array, nullable=nullable
                    )
                return self.fill_arrow_array_nulls(
                    pa.DictionaryArray.from_arrays(
                        indices=pc.cast(array.indices, pa.int32()),
                        dictionary=src_dict,
                        ordered=self.ordered,
                    ),
                    nullable=nullable,
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
        return self.fill_arrow_array_nulls(encoded, nullable=nullable)

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ):
        pl = get_polars()
        target_dtype = self.to_polars()

        # ``strict=False`` on an Enum cast is exactly our lenient
        # semantics: unknown values map to null. For non-string value
        # types we fall through to the base pipeline (which runs on
        # the raw value dtype).
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
        # ``CASE WHEN value IN (...) THEN value ELSE NULL``. Fine for
        # realistic category counts; very large category sets should
        # use a broadcast join rather than this type.
        value_spark = self.value_type.to_spark()
        casted = column.cast(value_spark)

        if not self.categories:
            return self.fill_spark_column_nulls(
                casted, nullable=self._target_nullable(options)
            )

        condition = None
        for cat in self.categories:
            eq = casted == F.lit(cat)
            condition = eq if condition is None else (condition | eq)

        gated = F.when(condition, casted).otherwise(F.lit(None).cast(value_spark))
        return self.fill_spark_column_nulls(
            gated, nullable=self._target_nullable(options)
        )
