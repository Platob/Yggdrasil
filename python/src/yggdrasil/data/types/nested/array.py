from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.types as pat

from yggdrasil.data.types.id import DataTypeId
from yggdrasil.lazy_imports import pandas_module, polars_module, spark_sql_module
from yggdrasil.data.enums import Mode
from yggdrasil.lazy_imports import field_class
from ._cast_json import (
    cast_arrow_json_string_array,
    cast_polars_json_string_expr,
    cast_spark_json_string_column,
    is_json_string_source,
)
from .base import DataType, NestedType

if TYPE_CHECKING:
    import pandas as pd
    import polars
    import pyspark.sql.types as pst
    from yggdrasil.data.options import CastOptions
    from yggdrasil.data.data_field import Field
    from .map import MapType


__all__ = [
    "ArrayType",
    "cast_arrow_list_array",
    "cast_arrow_map_array_to_list",
    "cast_polars_list_expr",
    "cast_polars_list_series",
    "cast_pandas_list_series",
    "cast_spark_list_column",
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, repr=False)
class ArrayType(NestedType):
    item_field: "Field"
    list_size: int | None = None
    large: bool = False
    view: bool = False
    fixed_size: bool = False

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.ARRAY

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        inner = self.item_field.pretty_format(indent=indent, level=level + 1)
        return f"{pad}list<\n{inner}\n{pad}>"

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None

        if not self.list_size:
            return []
        elif self.list_size < 0:
            raise ValueError("list_size must be non-negative")

        return [self.item_field.default_value for _ in range(self.list_size)]

    def equals(
        self,
        other: "DataType",
        check_names: bool = True,
        check_dtypes: bool = True,
        check_metadata: bool = True,
    ) -> bool:
        if not isinstance(other, ArrayType):
            return super().equals(
                other, check_names=check_names, check_dtypes=check_dtypes,
                check_metadata=check_metadata
            )

        if not self.item_field.name:
            self.item_field.with_name(other.item_field.name, inplace=True)

        return (
            self.item_field.equals(other.item_field, check_metadata=check_metadata)
            and self.list_size == other.list_size
            and self.large == other.large
            and self.view == other.view
            and self.fixed_size == other.fixed_size
        )

    def _merge_with_same_id(
        self,
        other: "ArrayType",
        mode: "Mode" = Mode.AUTO,
        downcast: bool = False,
        upcast: bool = False,
    ):
        item_field = self.item_field.merge_with(
            other.item_field,
            mode=mode, downcast=downcast, upcast=upcast
        )

        if self.list_size and other.list_size:
            if downcast:
                list_size = min(self.list_size, other.list_size)
            else:
                list_size = max(self.list_size, other.list_size)
        else:
            list_size = self.list_size or other.list_size

        if self.large and other.large:
            large = True
        elif self.large or other.large:
            large = True
        else:
            large = False

        return self.__class__(
            item_field=item_field,
            list_size=list_size,
            large=large,
            view=self.view or other.view,
            fixed_size=self.fixed_size or other.fixed_size,
        )

    @property
    def children(self) -> list["Field"]:
        return [self.item_field]

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return (
            pa.types.is_list(dtype)
            or pa.types.is_large_list(dtype)
            or pa.types.is_list_view(dtype)
            or pa.types.is_large_list_view(dtype)
            or pa.types.is_fixed_size_list(dtype)
        )

    @classmethod
    def from_item(
        cls,
        item_field: "Field",
        list_size: int | None = None,
        large: bool = False,
        view: bool = False,
        fixed_size: bool = False,
    ):
        _f = field_class()
        item_field = _f.from_any(item_field)

        # Arrow's fixed-size list requires list_size >= 0; treat any negative
        # value (including the -1 placeholder some serializers emit) as
        # "variable length" so we don't construct an invalid Arrow type.
        if list_size is not None and list_size < 0:
            list_size = None

        # ``list_size`` already implies the fixed-size variant in Arrow; keep
        # the boolean in sync so callers that pass only one of the two get a
        # consistent type.
        if list_size is not None:
            fixed_size = True

        return cls(
            item_field=item_field,
            list_size=list_size,
            large=large,
            view=view,
            fixed_size=fixed_size,
        )

    @classmethod
    def from_arrow_type(
        cls,
        dtype: "pa.ListType | pa.ListViewType | pa.FixedSizeListType",
    ) -> "ArrayType":
        if not cls.handles_arrow_type(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

        _f = field_class()
        item_field = _f.from_arrow_field(dtype.value_field)

        if pa.types.is_list(dtype):
            return cls(
                item_field=item_field,
                list_size=None,
                large=False,
                view=False,
            )

        if pa.types.is_large_list(dtype):
            return cls(
                item_field=item_field,
                list_size=None,
                large=True,
                view=False,
            )

        if pa.types.is_list_view(dtype):
            return cls(
                item_field=item_field,
                list_size=None,
                large=False,
                view=True,
            )

        if pa.types.is_large_list_view(dtype):
            return cls(
                item_field=item_field,
                list_size=None,
                large=True,
                view=True,
            )

        if pa.types.is_fixed_size_list(dtype):
            return cls(
                item_field=item_field,
                list_size=dtype.list_size,
                large=False,
                view=False,
                fixed_size=True,
            )

        raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = polars_module()
        return isinstance(dtype, pl.List)

    @classmethod
    def from_polars_type(cls, dtype: "polars.List") -> "ArrayType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")

        _f = field_class()

        return cls(
            item_field=_f(
                name="item",
                dtype=DataType.from_polars_type(dtype.inner),
                nullable=True,
                metadata=None,
            ),
            list_size=None,
            large=False,
            view=False,
        )

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = spark_sql_module()
        return isinstance(dtype, spark.types.ArrayType)

    @classmethod
    def from_spark_type(cls, dtype: "pst.ArrayType") -> "ArrayType":
        if not cls.handles_spark_type(dtype):
            raise TypeError(f"Unsupported Spark data type: {dtype!r}")

        _f = field_class()

        return cls(
            item_field=_f(
                name="item",
                dtype=DataType.from_spark_type(dtype.elementType),
                nullable=dtype.containsNull,
                metadata=None,
            ),
            list_size=None,
            large=False,
            view=False,
        )

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.ARRAY)

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "ArrayType":
        _f = field_class()

        try:
            return cls(
                item_field=_f.from_any(value["item_field"]),
                list_size=value.get("list_size"),
                large=bool(value.get("large", False)),
                view=bool(value.get("view", False)),
                fixed_size=bool(value.get("fixed_size", value.get("list_size") is not None)),
            )
        except Exception as e:
            if default is ...:
                raise ValueError(f"Could not parse {cls.__name__} from dict: {value!r}") from e
            return default


    def to_arrow(self) -> pa.DataType:
        value_field = self.item_field.to_arrow_field()

        if self.list_size is not None:
            return pa.list_(value_field, self.list_size)

        if self.view:
            if self.large:
                return pa.large_list_view(value_field)
            return pa.list_view(value_field)

        if self.large:
            return pa.large_list(value_field)

        return pa.list_(value_field)

    def _cast_arrow_array(
        self,
        array: pa.Array | pa.ChunkedArray,
        options: "CastOptions",
    ) -> pa.Array | pa.ChunkedArray:
        options = options.check_source(array).check_target(self)

        if options.need_cast(source=array, target=self):
            source_type_id = options.source.dtype.type_id

            if source_type_id == DataTypeId.NULL or array.null_count == len(array):
                return options.target.default_arrow_array(
                    size=len(array),
                    memory_pool=options.arrow_memory_pool,
                )

            elif is_json_string_source(source_type_id):
                return cast_arrow_json_string_array(array, options=options)

            elif source_type_id == DataTypeId.ARRAY:
                return cast_arrow_list_array(
                    array,
                    options=options,
                )

            elif source_type_id == DataTypeId.MAP:
                return cast_arrow_map_array_to_list(
                    array,
                    options=options,
                )

            else:
                raise pa.ArrowInvalid(
                    f"Cannot cast {options.source} to {options.target}"
                )

        return array

    def to_polars(self) -> "polars.DataType":
        pl = polars_module()
        return pl.List(self.item_field.dtype.to_polars())

    def to_spark(self) -> Any:
        spark = spark_sql_module()
        return spark.types.ArrayType(
            self.item_field.dtype.to_spark(),
            containsNull=self.item_field.nullable,
        )

    def as_spark(self) -> "ArrayType":
        # Recurse via the field-level :meth:`Field.as_spark` so the
        # element's metadata + nullability survive alongside its
        # Spark-flavored dtype.
        spark_item = self.item_field.as_spark()
        if spark_item is self.item_field:
            return self
        return ArrayType.from_item(spark_item)

    def as_polars(self) -> "ArrayType":
        polars_item = self.item_field.as_polars()
        if polars_item is self.item_field:
            return self
        return ArrayType.from_item(polars_item)

    def to_spark_name(self) -> str:
        return f"ARRAY<{self.item_field.dtype.to_spark_name()}>"

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["item_field"] = self.item_field.to_dict()

        if self.list_size is not None and self.list_size >= 0:
            base["list_size"] = self.list_size

        if self.large:
            base["large"] = True
        if self.view:
            base["view"] = True
        if self.fixed_size:
            base["fixed_size"] = True

        return base

    def _convert_pyobj(self, value: Any, safe: bool = False) -> list | None:
        # Priority path: str/bytes → JSON-decode, then fall through.
        if isinstance(value, (bytes, bytearray, memoryview)):
            try:
                value = bytes(value).decode("utf-8")
            except UnicodeDecodeError:
                if safe:
                    raise ValueError(
                        f"Cannot decode bytes as UTF-8 for {type(self).__name__}: "
                        f"{value!r}"
                    )
                return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                if safe:
                    raise ValueError(
                        f"Cannot parse array from empty string for "
                        f"{type(self).__name__}."
                    )
                return None
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                if safe:
                    raise ValueError(
                        f"Cannot parse array from {value!r} for "
                        f"{type(self).__name__}."
                    )
                return None
            value = decoded

        if isinstance(value, (list, tuple, set, frozenset)):
            items = list(value)
        elif hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
            items = list(value.tolist())
        else:
            if safe:
                raise ValueError(
                    f"Cannot convert {type(value).__name__} to array "
                    f"for {type(self).__name__}: {value!r}."
                )
            return None

        item_dtype = self.item_field.dtype
        item_nullable = self.item_field.nullable
        return [
            item_dtype.convert_pyobj(item, nullable=item_nullable, safe=safe)
            for item in items
        ]

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ) -> "polars.Series":
        pl = polars_module()
        options = options.check_source(series).check_target(self)

        if options.source.dtype.type_id == DataTypeId.NULL or series.null_count() == len(series):
            return options.target.default_polars_series(size=len(series))

        expr = self._cast_polars_expr(
            pl.col(series.name),
            options=options,
        ).alias(options.target.name)
        return pl.DataFrame({series.name: series}).select(expr).to_series()

    def _cast_polars_expr(
        self,
        expr: Any,
        options: "CastOptions",
    ) -> Any:
        options = options.check_target(self)

        source_type_id = options.source.dtype.type_id

        if source_type_id == DataTypeId.NULL:
            return options.target.default_polars_expr(alias=options.target.name)

        elif is_json_string_source(source_type_id):
            return cast_polars_json_string_expr(expr, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_polars_list_expr(expr, options)

        else:
            raise TypeError(
                f"Cannot cast {options.source} to {options.target}"
            )

    def _cast_pandas_series(
        self,
        series: "pd.Series",
        options: "CastOptions",
    ) -> "pd.Series":
        pd = pandas_module()
        options = options.check_source(series).check_target(self)

        source_type_id = options.source.dtype.type_id

        if source_type_id == DataTypeId.NULL or series.isna().all():
            return options.target.default_pandas_series(size=len(series))

        elif is_json_string_source(source_type_id):
            return _cast_pandas_via_arrow(series, options, cast_arrow_json_string_array)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_pandas_list_series(series, options)

        elif source_type_id == DataTypeId.MAP:
            return _cast_pandas_via_arrow(series, options, cast_arrow_map_array_to_list)

        else:
            raise TypeError(
                f"Cannot cast {options.source} to {options.target}"
            )

    def _cast_spark_column(
        self,
        column: Any,
        options: "CastOptions",
    ) -> Any:
        options = options.check_source(column).check_target(self)

        source_type_id = options.source.dtype.type_id

        if source_type_id == DataTypeId.NULL:
            return options.target.default_spark_column()

        elif is_json_string_source(source_type_id):
            return cast_spark_json_string_column(column, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_spark_list_column(column, options)

        else:
            raise TypeError(
                f"Cannot cast {options.source} to {options.target}"
            )


def cast_arrow_list_array(
    array: pa.Array | pa.ChunkedArray,
    options: "CastOptions",
) -> pa.Array | pa.ChunkedArray:
    options = options.check_source(array)

    if options.target is None:
        return array
    elif options.source.dtype.type_id != DataTypeId.ARRAY:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source} to {options.target}"
        )

    source_field: Field = options.source
    target_field: Field = options.target.with_name(options.target.name or source_field.name, inplace=True)

    source_type: ArrayType = source_field.dtype
    target_type: ArrayType = target_field.dtype
    target_type.item_field.with_name(target_type.item_field.name or source_type.item_field.name, inplace=True)

    source_arrow_type = source_type.to_arrow()
    target_arrow_type = target_type.to_arrow()

    if array.type == target_arrow_type:
        return array

    if isinstance(array, pa.ChunkedArray):
        chunks = [
            cast_arrow_list_array(
                chunk,
                options=options,
            )
            for chunk in array.chunks
        ]
        return pa.chunked_array(
            chunks,
            type=options.target.dtype.to_arrow(),
        )

    # Slicing a ListArray keeps the offsets buffer pointing at the parent's
    # absolute positions and the null bitmap carrying a slice offset.
    # ``pa.ListArray.from_arrays(offsets=..., mask=...)`` rejects that
    # combination with "Null bitmap with offsets slice not supported."
    # Rebase the offsets to start at 0 (vectorised ``pc.subtract``) and
    # slice the flat values to match — all inside Arrow compute kernels,
    # no per-row work — before any item-cast or rebuild.
    src_offsets, src_values = _rebased_list_offsets_and_values(array)

    if source_arrow_type.value_type == target_arrow_type.value_type:
        target_values = src_values
    else:
        target_values = target_type.item_field.cast_arrow_array(
            src_values,
            options=options.copy(source=source_type.item_field),
        )

    if target_type.list_size is not None:
        return pa.FixedSizeListArray.from_arrays(
            values=target_values,
            list_size=target_type.list_size,
            mask=array.is_null(),
        )

    if target_type.view:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source} to {options.target}"
        )

    if target_type.large:
        return pa.LargeListArray.from_arrays(
            offsets=src_offsets,
            values=target_values,
            mask=array.is_null(),
        )

    return pa.ListArray.from_arrays(
        offsets=src_offsets,
        values=target_values,
        mask=array.is_null(),
    )


def _rebased_list_offsets_and_values(
    array: "pa.ListArray | pa.LargeListArray",
) -> tuple[pa.Array, pa.Array]:
    """Return offsets rebased to start at 0 plus values sliced to match.

    ``pa.ListArray.from_arrays`` (and the LargeList variant) refuses an
    offsets array whose first entry is non-zero when combined with a
    sliced null bitmap.  This helper rebases the offsets via
    ``pc.subtract`` (one C++ kernel call) and slices the flat values so
    the rebuilt list still points at the same logical rows.  Slice-free
    inputs short-circuit (no compute call, no copy).
    """
    import pyarrow.compute as pc

    offsets = array.offsets
    if len(offsets) == 0:
        return offsets, array.values

    first = offsets[0].as_py()
    last = offsets[-1].as_py()
    if first == 0:
        return offsets, array.values

    rebased = pc.subtract(offsets, first)
    sliced_values = array.values.slice(first, last - first)
    return rebased, sliced_values


def cast_arrow_map_array_to_list(
    array: pa.MapArray | pa.ChunkedArray,
    options: "CastOptions",
) -> pa.Array | pa.ChunkedArray:
    options = options.check_source(array)

    if options.target is None:
        return array
    elif options.source.dtype.type_id != DataTypeId.MAP:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source} to {options.target}"
        )

    if isinstance(array, pa.ChunkedArray):
        chunks = [
            cast_arrow_map_array_to_list(
                chunk,
                options=options,
            )
            for chunk in array.chunks
        ]
        return pa.chunked_array(
            chunks,
            type=options.target.dtype.to_arrow(),
        )

    source_field: Field = options.source
    target_field: Field = options.target

    source_type: "MapType" = source_field.dtype
    target_type: ArrayType = target_field.dtype

    target_item_type = target_type.item_field.dtype

    if target_item_type.type_id != DataTypeId.STRUCT:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source} to {options.target}"
        )

    if len(target_item_type.children) != 2:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source} to {options.target}"
        )

    target_key_field = target_item_type.field_at(0)
    target_value_field = target_item_type.field_at(1)

    target_key_array = target_key_field.cast_arrow_array(
        array.keys,
        options=options.copy(source=source_type.key_field),
    )

    target_value_array = target_value_field.cast_arrow_array(
        array.items,
        options=options.copy(source=source_type.value_field),
    )

    entry_values = pa.StructArray.from_arrays(
        [
            target_key_array,
            target_value_array,
        ],
        fields=[
            target_key_field.to_arrow_field(),
            target_value_field.to_arrow_field(),
        ],
        memory_pool=options.arrow_memory_pool,
    )

    if target_type.list_size is not None:
        return pa.FixedSizeListArray.from_arrays(
            values=entry_values,
            list_size=target_type.list_size,
            mask=array.is_null(),
        )

    if target_type.view:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source} to {options.target}"
        )

    if target_type.large:
        return pa.LargeListArray.from_arrays(
            offsets=array.offsets,
            values=entry_values,
            mask=array.is_null(),
        )

    return pa.ListArray.from_arrays(
        offsets=array.offsets,
        values=entry_values,
        mask=array.is_null(),
    )


# Polars

def cast_polars_list_expr(
    expr: Any,
    options: "CastOptions",
) -> Any:
    pl = polars_module()

    if options.target is None:
        return expr
    elif options.source.dtype.type_id != DataTypeId.ARRAY:
        raise TypeError(f"Cannot cast {options.source} to {options.target}")

    source_type: ArrayType = options.source.dtype
    target_type: ArrayType = options.target.dtype

    casted_element = target_type.item_field.cast_polars_expr(
        pl.element(),
        options=options.copy(
            source=source_type.item_field,
            target=target_type.item_field,
        ),
    )

    return expr.list.eval(casted_element)


def cast_polars_list_series(
    series: "polars.Series",
    options: "CastOptions",
) -> "polars.Series":
    pl = polars_module()
    expr = cast_polars_list_expr(pl.col(series.name), options).alias(options.target.name)
    return pl.DataFrame({series.name: series}).select(expr).to_series()


# Pandas

def _cast_pandas_via_arrow(
    series: "pd.Series",
    options: "CastOptions",
    caster,
) -> "pd.Series":
    # Round-trip pandas <-> Arrow without per-row Python materialisation:
    # ``pa.array(series, from_pandas=True, type=...)`` consumes the Series
    # via the pandas → Arrow C bridge, and ``Array.to_pandas()`` rebuilds
    # the Series on the way out — no ``.tolist()`` / ``.to_pylist()`` hop
    # in either direction.  Nested cells surface as numpy.ndarray (lists)
    # or dict (structs) — the natural pyarrow → pandas mapping.
    source_arrow_type = options.source.dtype.to_arrow()
    source_array = pa.array(series, type=source_arrow_type, from_pandas=True)
    casted = caster(source_array, options)

    if isinstance(casted, pa.ChunkedArray):
        casted = casted.combine_chunks()

    result = casted.to_pandas()
    result.index = series.index
    result.name = options.target.name
    return result


def cast_pandas_list_series(
    series: "pd.Series",
    options: "CastOptions",
) -> "pd.Series":
    return _cast_pandas_via_arrow(series, options, cast_arrow_list_array)


# PySpark

def cast_spark_list_column(
    column: Any,
    options: "CastOptions",
) -> Any:
    spark = spark_sql_module()
    F = spark.functions

    if options.target is None:
        return column
    elif options.source.dtype.type_id != DataTypeId.ARRAY:
        raise TypeError(f"Cannot cast {options.source} to {options.target}")

    source_type: ArrayType = options.source.dtype
    target_type: ArrayType = options.target.dtype

    return F.transform(
        column,
        lambda x: target_type.item_field.cast_spark_column(
            x,
            options=options.copy(
                source=source_type.item_field,
                target=target_type.item_field,
            ),
        ),
    )