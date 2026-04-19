from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.support import get_pandas, get_polars, get_spark_sql
from yggdrasil.io import SaveMode

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
    from yggdrasil.data.cast.options import CastOptions
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


@dataclass(frozen=True)
class ArrayType(NestedType):
    item_field: "Field"
    list_size: int | None = None
    large: bool = False
    view: bool = False

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.ARRAY

    def _merge_with_same_id(
        self,
        other: "ArrayType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ):
        item_field = self.item_field.merge_with(
            other.item_field,
            mode=mode, downcast=downcast, upcast=upcast
        )

        return self.__class__(
            item_field=item_field,
            list_size=self.list_size or other.list_size,
            large=self.large,
            view=self.view,
        )

    @property
    def children_fields(self) -> list["Field"]:
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
    def from_item_field(
        cls,
        item_field: "Field",
        list_size: int | None = None,
        large: bool = False,
        view: bool = False,
        safe: bool = False,
    ):
        if not safe:
            _f = cls.get_data_field_class()
            item_field = _f.from_any(item_field).with_name("item")

        # Arrow's fixed-size list requires list_size >= 0; treat any negative
        # value (including the -1 placeholder some serializers emit) as
        # "variable length" so we don't construct an invalid Arrow type.
        if list_size is not None and list_size < 0:
            list_size = None

        return cls(
            item_field=item_field,
            list_size=list_size,
            large=large,
            view=view,
        )

    @classmethod
    def from_arrow_type(
        cls,
        dtype: "pa.ListType | pa.ListViewType | pa.FixedSizeListType",
    ) -> "ArrayType":
        if not cls.handles_arrow_type(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

        _f = cls.get_data_field_class()
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
            )

        raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        return isinstance(dtype, pl.List)

    @classmethod
    def from_polars_type(cls, dtype: "polars.List") -> "ArrayType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")

        _f = cls.get_data_field_class()

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
        spark = get_spark_sql()
        return isinstance(dtype, spark.types.ArrayType)

    @classmethod
    def from_spark_type(cls, dtype: "pst.ArrayType") -> "ArrayType":
        if not cls.handles_spark_type(dtype):
            raise TypeError(f"Unsupported Spark data type: {dtype!r}")

        _f = cls.get_data_field_class()

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
    def from_dict(cls, value: dict[str, Any]) -> "ArrayType":
        _f = cls.get_data_field_class()

        return cls(
            item_field=_f.from_any(value["item_field"]),
            list_size=value.get("list_size"),
            large=bool(value.get("large", False)),
            view=bool(value.get("view", False)),
        )

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
        options.check_source(array)
        options.check_target(self)

        source_type_id = options.source_field.dtype.type_id

        if source_type_id == DataTypeId.NULL or array.null_count == len(array):
            return options.target_field.default_arrow_array(
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
                f"Cannot cast {options.source_field} to {options.target_field}"
            )

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.List(self.item_field.dtype.to_polars())

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        return spark.types.ArrayType(
            self.item_field.dtype.to_spark(),
            containsNull=self.item_field.nullable,
        )

    def to_databricks_ddl(self) -> str:
        return f"ARRAY<{self.item_field.dtype.to_databricks_ddl()}>"

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        tags[b"nested_kind"] = b"array"
        tags[b"element_type_id"] = self.item_field.dtype.type_id.name.encode("utf-8")
        if self.list_size is not None:
            tags[b"fixed_size"] = str(self.list_size).encode("utf-8")
        if self.large:
            tags[b"large"] = b"true"
        if self.view:
            tags[b"view"] = b"true"
        return tags

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["item_field"] = self.item_field.to_dict()

        if self.list_size is not None and self.list_size >= 0:
            base["list_size"] = self.list_size

        if self.large:
            base["large"] = True
        if self.view:
            base["view"] = True

        return base

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else []

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ) -> "polars.Series":
        pl = get_polars()
        options.check_source(series)
        options.check_target(self)

        if options.source_field.dtype.type_id == DataTypeId.NULL or series.null_count() == len(series):
            return options.target_field.default_polars_series(size=len(series))

        expr = self._cast_polars_expr(
            pl.col(series.name),
            options=options,
        ).alias(options.target_field.name)
        return pl.DataFrame({series.name: series}).select(expr).to_series()

    def _cast_polars_expr(
        self,
        expr: Any,
        options: "CastOptions",
    ) -> Any:
        options.check_target(self)

        source_type_id = options.source_field.dtype.type_id

        if source_type_id == DataTypeId.NULL:
            return options.target_field.default_polars_expr(alias=options.target_field.name)

        elif is_json_string_source(source_type_id):
            return cast_polars_json_string_expr(expr, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_polars_list_expr(expr, options)

        else:
            raise TypeError(
                f"Cannot cast {options.source_field} to {options.target_field}"
            )

    def _cast_pandas_series(
        self,
        series: "pd.Series",
        options: "CastOptions",
    ) -> "pd.Series":
        pd = get_pandas()
        options.check_source(series)
        options.check_target(self)

        source_type_id = options.source_field.dtype.type_id

        if source_type_id == DataTypeId.NULL or series.isna().all():
            return options.target_field.default_pandas_series(size=len(series))

        elif is_json_string_source(source_type_id):
            return _cast_pandas_via_arrow(series, options, cast_arrow_json_string_array)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_pandas_list_series(series, options)

        elif source_type_id == DataTypeId.MAP:
            return _cast_pandas_via_arrow(series, options, cast_arrow_map_array_to_list)

        else:
            raise TypeError(
                f"Cannot cast {options.source_field} to {options.target_field}"
            )

    def _cast_spark_column(
        self,
        column: Any,
        options: "CastOptions",
    ) -> Any:
        options.check_source(column)
        options.check_target(self)

        source_type_id = options.source_field.dtype.type_id

        if source_type_id == DataTypeId.NULL:
            return options.target_field.default_spark_column()

        elif is_json_string_source(source_type_id):
            return cast_spark_json_string_column(column, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_spark_list_column(column, options)

        else:
            raise TypeError(
                f"Cannot cast {options.source_field} to {options.target_field}"
            )


def cast_arrow_list_array(
    array: pa.Array | pa.ChunkedArray,
    options: "CastOptions",
) -> pa.Array | pa.ChunkedArray:
    options.check_source(array)

    if options.target_field is None:
        return array
    elif options.source_field.dtype.type_id != DataTypeId.ARRAY:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

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
            type=options.target_field.dtype.to_arrow(),
        )

    source_field: Field = options.source_field
    target_field: Field = options.target_field

    source_type: ArrayType = source_field.dtype
    target_type: ArrayType = target_field.dtype

    values = array.values

    target_values = target_type.item_field.cast_arrow_array(
        values,
        options=options.copy(source_field=source_type.item_field),
    )

    if target_type.list_size is not None:
        return pa.FixedSizeListArray.from_arrays(
            values=target_values,
            list_size=target_type.list_size,
            mask=array.is_null(),
        )

    if target_type.view:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    if target_type.large:
        return pa.LargeListArray.from_arrays(
            offsets=array.offsets,
            values=target_values,
            mask=array.is_null(),
        )

    return pa.ListArray.from_arrays(
        offsets=array.offsets,
        values=target_values,
        mask=array.is_null(),
    )


def cast_arrow_map_array_to_list(
    array: pa.MapArray | pa.ChunkedArray,
    options: "CastOptions",
) -> pa.Array | pa.ChunkedArray:
    options.check_source(array)

    if options.target_field is None:
        return array
    elif options.source_field.dtype.type_id != DataTypeId.MAP:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
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
            type=options.target_field.dtype.to_arrow(),
        )

    source_field: Field = options.source_field
    target_field: Field = options.target_field

    source_type: "MapType" = source_field.dtype
    target_type: ArrayType = target_field.dtype

    target_item_type = target_type.item_field.dtype

    if target_item_type.type_id != DataTypeId.STRUCT:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    if len(target_item_type.children_fields) != 2:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    target_key_field = target_item_type.field_at(0)
    target_value_field = target_item_type.field_at(1)

    target_key_array = target_key_field.cast_arrow_array(
        array.keys,
        options=options.copy(source_field=source_type.key_field),
    )

    target_value_array = target_value_field.cast_arrow_array(
        array.items,
        options=options.copy(source_field=source_type.value_field),
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
            f"Cannot cast {options.source_field} to {options.target_field}"
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
    pl = get_polars()

    if options.target_field is None:
        return expr
    elif options.source_field.dtype.type_id != DataTypeId.ARRAY:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_type: ArrayType = options.source_field.dtype
    target_type: ArrayType = options.target_field.dtype

    casted_element = target_type.item_field.cast_polars_expr(
        pl.element(),
        options=options.copy(
            source_field=source_type.item_field,
            target_field=target_type.item_field,
        ),
    )

    return expr.list.eval(casted_element)


def cast_polars_list_series(
    series: "polars.Series",
    options: "CastOptions",
) -> "polars.Series":
    pl = get_polars()
    expr = cast_polars_list_expr(pl.col(series.name), options).alias(options.target_field.name)
    return pl.DataFrame({series.name: series}).select(expr).to_series()


# Pandas

def _cast_pandas_via_arrow(
    series: "pd.Series",
    options: "CastOptions",
    caster,
) -> "pd.Series":
    pd = get_pandas()

    source_arrow_type = options.source_field.dtype.to_arrow()
    source_array = pa.array(
        series.tolist(),
        type=source_arrow_type,
        from_pandas=True,
    )
    casted = caster(source_array, options)

    if isinstance(casted, pa.ChunkedArray):
        values = casted.to_pylist()
    else:
        values = casted.to_pylist()

    return pd.Series(values, index=series.index, name=options.target_field.name, dtype="object")


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
    spark = get_spark_sql()
    F = spark.functions

    if options.target_field is None:
        return column
    elif options.source_field.dtype.type_id != DataTypeId.ARRAY:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_type: ArrayType = options.source_field.dtype
    target_type: ArrayType = options.target_field.dtype

    return F.transform(
        column,
        lambda x: target_type.item_field.cast_spark_column(
            x,
            options=options.copy(
                source_field=source_type.item_field,
                target_field=target_type.item_field,
            ),
        ),
    )