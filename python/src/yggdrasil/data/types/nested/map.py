from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested import NestedType
from yggdrasil.data.types.support import get_pandas, get_polars, get_spark_sql
from yggdrasil.environ.importlib import cached_from_import
from yggdrasil.io import SaveMode
from ._cast_json import (
    cast_arrow_json_string_array,
    cast_polars_json_string_expr,
    cast_spark_json_string_column,
    is_json_string_source,
)
from .array import ArrayType

if TYPE_CHECKING:
    import pandas as pd
    import polars
    import pyspark.sql as psql
    import pyspark.sql.types as pst
    from yggdrasil.data.cast.options import CastOptions
    from yggdrasil.data.data_field import Field
    from .struct import StructType

__all__ = [
    "MapType",
    "cast_arrow_map_array",
    "cast_arrow_list_array_to_map",
    "cast_arrow_struct_array_to_map",
    "cast_polars_map_series",
    "cast_polars_map_expr",
    "cast_polars_list_series_to_map",
    "cast_polars_list_expr_to_map",
    "cast_polars_struct_series_to_map",
    "cast_polars_struct_expr_to_map",
    "cast_pandas_map_series",
    "cast_pandas_list_series_to_map",
    "cast_pandas_struct_series_to_map",
    "cast_spark_map_column",
    "cast_spark_list_column_to_map",
    "cast_spark_struct_column_to_map",
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MapType(NestedType):
    item_field: "Field"
    keys_sorted: bool = False

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.MAP

    def _merge_with_same_id(
        self,
        other: "MapType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "MapType":
        key = self.key_field.merge_with(
            other.key_field,
            mode=mode,
            downcast=downcast,
            upcast=upcast,
        )
        value = self.value_field.merge_with(
            other.value_field,
            mode=mode,
            downcast=downcast,
            upcast=upcast,
        )

        return self.from_key_value(
            key,
            value,
            keys_sorted=self.keys_sorted or other.keys_sorted,
        )

    @property
    def children_fields(self) -> list["Field"]:
        return [self.item_field]

    @property
    def key_field(self) -> "Field":
        return self.item_field.field_at(0)

    @property
    def value_field(self) -> "Field":
        return self.item_field.field_at(1)

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_map(dtype)

    @classmethod
    def from_key_value(
        cls,
        key_field: "Field",
        value_field: "Field",
        keys_sorted: bool = False,
    ) -> "MapType":
        _f = cls.get_data_field_class()
        k = _f.from_any(key_field).with_name("key", inplace=True).with_nullable(
            nullable=False,
            inplace=True,
        )
        v = _f.from_any(value_field).with_name("value", inplace=True)
        entry_struct = cached_from_import(
            "yggdrasil.data.types.nested",
            "StructType",
        )(fields=[k, v])

        return cls(
            item_field=_f(
                name="entries",
                dtype=entry_struct,
                nullable=False,
            ),
            keys_sorted=keys_sorted,
        )

    @classmethod
    def from_arrow_type(cls, dtype: pa.MapType) -> "MapType":
        if not pa.types.is_map(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")

        _f = cached_from_import("yggdrasil.data.data_field", "Field")

        entry_struct = cached_from_import(
            "yggdrasil.data.types.nested",
            "StructType",
        )(fields=[
            _f.from_arrow_field(dtype.key_field),
            _f.from_arrow_field(dtype.item_field),
        ])

        return cls(
            item_field=_f(
                name="entries",
                dtype=entry_struct,
                nullable=False,
            ),
            keys_sorted=getattr(dtype, "keys_sorted", False),
        )

    @classmethod
    def handles_polars_type(cls, dtype: "polars.List") -> bool:
        pl = get_polars()
        return isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct)

    @classmethod
    def from_polars_type(cls, dtype: "polars.List") -> "MapType":
        _f = cached_from_import("yggdrasil.data.data_field", "Field")
        StructType = cached_from_import("yggdrasil.data.types.nested", "StructType")

        if not cls.handles_polars_type(dtype):
            raise TypeError(f"Unsupported Polars data type: {dtype!r}")

        fields = [_f.from_polars(f) for f in dtype.inner.fields]

        if len(fields) != 2:
            raise TypeError(f"Expected List[Struct[key, value]] for map type: {dtype!r}")

        return cls(
            item_field=_f(
                name="entries",
                dtype=StructType(fields=fields),
                nullable=False,
            ),
            keys_sorted=False,
        )

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = get_spark_sql()
        return isinstance(dtype, spark.types.MapType)

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "MapType":
        _f = cached_from_import("yggdrasil.data.data_field", "Field")
        StructType = cached_from_import("yggdrasil.data.types.nested", "StructType")
        spark = get_spark_sql()

        if not isinstance(dtype, spark.types.MapType):
            raise TypeError(f"Unsupported Spark data type: {dtype!r}")

        entry_struct = StructType(fields=[
            _f.from_spark(spark.types.StructField("key", dtype.keyType, nullable=False)),
            _f.from_spark(
                spark.types.StructField(
                    "value",
                    dtype.valueType,
                    nullable=dtype.valueContainsNull,
                )
            ),
        ])

        return cls(
            item_field=_f(
                name="entries",
                dtype=entry_struct,
                nullable=False,
            ),
            keys_sorted=False,
        )

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.MAP)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MapType":
        return cls(
            item_field=cached_from_import(
                "yggdrasil.data.data_field",
                "Field",
            ).from_dict(value["item_field"]),
            keys_sorted=bool(value.get("keys_sorted", False)),
        )

    def to_arrow(self) -> pa.DataType:
        return pa.map_(
            self.key_field.to_arrow_field(),
            self.value_field.to_arrow_field(),
            keys_sorted=self.keys_sorted,
        )

    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.MapArray | pa.ChunkedArray:
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

        elif source_type_id == DataTypeId.MAP:
            return cast_arrow_map_array(
                array,
                options=options,
            )

        elif source_type_id == DataTypeId.ARRAY:
            return cast_arrow_list_array_to_map(
                array,
                options=options,
            )

        elif source_type_id == DataTypeId.STRUCT:
            return cast_arrow_struct_array_to_map(
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

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ) -> "polars.Series":
        pl = get_polars()
        options.check_source(series)
        options.check_target(self)

        source_type_id = options.source_field.dtype.type_id

        if source_type_id == DataTypeId.NULL or series.null_count() == len(series):
            return options.target_field.default_polars_series(size=len(series))

        if is_json_string_source(source_type_id):
            # polars represents a Map as List<Struct<key, value>>, which
            # ``str.json_decode`` can only populate from a JSON *list* of
            # entries — not a plain JSON object.  Roundtrip via Arrow so
            # ``{"a":1,"b":2}`` shaped input still decodes correctly.
            arrow_input = series.to_arrow()
            arrow_output = cast_arrow_json_string_array(
                arrow_input, options=options
            )
            return pl.from_arrow(arrow_output).rename(
                options.target_field.name
            )

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
            # JSON object → Map needs materialisation (see _cast_polars_series)
            raise TypeError(
                f"Cannot cast {options.source_field} to {options.target_field} "
                "as an expression; use cast_polars_series for a JSON map source."
            )

        elif source_type_id == DataTypeId.MAP:
            return cast_polars_map_expr(expr, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_polars_list_expr_to_map(expr, options)

        elif source_type_id == DataTypeId.STRUCT:
            return cast_polars_struct_expr_to_map(expr, options)

        else:
            raise TypeError(
                f"Cannot cast {options.source_field} to {options.target_field}"
            )

    def _cast_pandas_series(
        self,
        series: "pd.Series",
        options: "CastOptions",
    ) -> "pd.Series":
        options.check_source(series)
        options.check_target(self)

        source_type_id = options.source_field.dtype.type_id

        if source_type_id == DataTypeId.NULL or series.isna().all():
            return options.target_field.default_pandas_series(size=len(series))

        elif is_json_string_source(source_type_id):
            return _cast_pandas_via_arrow(series, options, cast_arrow_json_string_array)

        elif source_type_id == DataTypeId.MAP:
            return cast_pandas_map_series(series, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_pandas_list_series_to_map(series, options)

        elif source_type_id == DataTypeId.STRUCT:
            return cast_pandas_struct_series_to_map(series, options)

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

        elif source_type_id == DataTypeId.MAP:
            return cast_spark_map_column(column, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_spark_list_column_to_map(column, options)

        elif source_type_id == DataTypeId.STRUCT:
            return cast_spark_struct_column_to_map(column, options)

        else:
            raise TypeError(
                f"Cannot cast {options.source_field} to {options.target_field}"
            )

    def _cast_spark_tabular(
        self,
        frame: "psql.DataFrame",
        options: "CastOptions",
    ) -> "psql.DataFrame":
        raise TypeError(
            f"Cannot cast tabular source {type(frame)!r} directly to {self.__class__.__name__}"
        )

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        return spark.types.MapType(
            keyType=self.key_field.dtype.to_spark(),
            valueType=self.value_field.dtype.to_spark(),
            valueContainsNull=self.value_field.nullable,
        )

    def to_databricks_ddl(self) -> str:
        return f"MAP<{self.key_field.dtype.to_databricks_ddl()}, {self.value_field.dtype.to_databricks_ddl()}>"

    def autotag(self) -> dict[bytes, bytes]:
        tags = super().autotag()
        tags[b"nested_kind"] = b"map"
        tags[b"key_type_id"] = self.key_field.dtype.type_id.name.encode("utf-8")
        tags[b"value_type_id"] = self.value_field.dtype.type_id.name.encode("utf-8")
        if self.keys_sorted:
            tags[b"keys_sorted"] = b"true"
        return tags

    def to_dict(self) -> dict[str, Any]:
        base = super(MapType, self).to_dict()
        base["item_field"] = self.item_field.to_dict()
        if self.keys_sorted:
            base["keys_sorted"] = self.keys_sorted
        return base

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else {}


_STRING_KEY_SOURCE_FIELD: "Field | None" = None


def _string_key_source_field() -> "Field":
    global _STRING_KEY_SOURCE_FIELD
    if _STRING_KEY_SOURCE_FIELD is None:
        Field = cached_from_import("yggdrasil.data.data_field", "Field")
        DataType = cached_from_import("yggdrasil.data.types", "DataType")
        _STRING_KEY_SOURCE_FIELD = Field(
            name="key",
            dtype=DataType.from_arrow_type(pa.string()),
            nullable=False,
        )
    return _STRING_KEY_SOURCE_FIELD


def _cast_pandas_via_arrow(
    series: "pd.Series",
    options: "CastOptions",
    caster: Callable[[pa.Array, "CastOptions"], pa.Array | pa.ChunkedArray],
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


def cast_arrow_map_array(
    array: pa.MapArray | pa.ChunkedArray,
    options: "CastOptions",
) -> pa.MapArray | pa.ChunkedArray:
    options.check_source(array)

    if options.target_field is None:
        return array
    elif options.source_field.dtype.type_id != DataTypeId.MAP:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    if isinstance(array, pa.ChunkedArray):
        chunks = [
            cast_arrow_map_array(
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

    source_type: MapType = source_field.dtype
    target_type: MapType = target_field.dtype

    target_key_array = target_type.key_field.cast_arrow_array(
        array.keys,
        options=options.copy(
            source_field=source_type.key_field,
            target_field=target_type.key_field,
        ),
    )

    target_value_array = target_type.value_field.cast_arrow_array(
        array.items,
        options=options.copy(
            source_field=source_type.value_field,
            target_field=target_type.value_field,
        ),
    )

    return pa.MapArray.from_arrays(
        offsets=array.offsets,
        keys=target_key_array,
        items=target_value_array,
        mask=array.is_null(),
        type=target_type.to_arrow(),
        pool=options.arrow_memory_pool,
    )


def cast_arrow_list_array_to_map(
    array: pa.ListArray | pa.ChunkedArray,
    options: "CastOptions",
) -> pa.MapArray | pa.ChunkedArray:
    options.check_source(array)

    if options.target_field is None:
        return array
    elif options.source_field.dtype.type_id != DataTypeId.ARRAY:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    if isinstance(array, pa.ChunkedArray):
        chunks = [
            cast_arrow_list_array_to_map(
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
    target_type: MapType = target_field.dtype

    source_item_field = source_type.item_field
    source_item_dtype = source_item_field.dtype

    if source_item_dtype.type_id != DataTypeId.STRUCT:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    if len(source_item_dtype.children_fields) != 2:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    source_key_field = source_item_dtype.field_at(0)
    source_value_field = source_item_dtype.field_at(1)

    values: pa.StructArray = array.values

    target_key_array = target_type.key_field.cast_arrow_array(
        values.field(source_key_field.name),
        options=options.copy(
            source_field=source_key_field,
            target_field=target_type.key_field,
        ),
    )

    target_value_array = target_type.value_field.cast_arrow_array(
        values.field(source_value_field.name),
        options=options.copy(
            source_field=source_value_field,
            target_field=target_type.value_field,
        ),
    )

    return pa.MapArray.from_arrays(
        offsets=array.offsets,
        keys=target_key_array,
        items=target_value_array,
        mask=array.is_null(),
        type=target_type.to_arrow(),
        pool=options.arrow_memory_pool,
    )


def cast_arrow_struct_array_to_map(
    array: pa.StructArray | pa.ChunkedArray,
    options: "CastOptions",
) -> pa.MapArray | pa.ChunkedArray:
    options.check_source(array)

    if options.target_field is None:
        return array
    elif options.source_field.dtype.type_id != DataTypeId.STRUCT:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    if isinstance(array, pa.ChunkedArray):
        chunks = [
            cast_arrow_struct_array_to_map(
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

    source_type: StructType = source_field.dtype
    target_type: MapType = target_field.dtype

    row_count = len(array)
    child_count = len(source_type.children_fields)

    if child_count == 0:
        offsets = pa.array([0] * (row_count + 1), type=pa.int32())
        empty_keys = pa.array([], type=target_type.key_field.dtype.to_arrow())
        empty_items = pa.array([], type=target_type.value_field.dtype.to_arrow())
        return pa.MapArray.from_arrays(
            offsets=offsets,
            keys=empty_keys,
            items=empty_items,
            mask=array.is_null(),
            type=target_type.to_arrow(),
            pool=options.arrow_memory_pool,
        )

    key_arrays: list[pa.Array] = []
    value_arrays: list[pa.Array] = []

    for child in source_type.children_fields:
        key_arr = pa.array(
            [child.name] * row_count,
            type=pa.string(),
            memory_pool=options.arrow_memory_pool,
        )
        casted_key_arr = target_type.key_field.cast_arrow_array(
            key_arr,
            options=options.copy(
                source_field=_string_key_source_field(),
                target_field=target_type.key_field,
            ),
        )
        casted_value_arr = target_type.value_field.cast_arrow_array(
            array.field(child.name),
            options=options.copy(
                source_field=child,
                target_field=target_type.value_field,
            ),
        )
        key_arrays.append(casted_key_arr)
        value_arrays.append(casted_value_arr)

    keys_child_major = pa.concat_arrays(key_arrays, memory_pool=options.arrow_memory_pool)
    values_child_major = pa.concat_arrays(value_arrays, memory_pool=options.arrow_memory_pool)

    row_mask = np.asarray(array.is_null().to_numpy(zero_copy_only=False), dtype=bool)
    valid_rows = np.flatnonzero(~row_mask)

    if len(valid_rows) == 0:
        take_indices = pa.array([], type=pa.int64())
    else:
        base = np.arange(child_count, dtype=np.int64)[:, None] * row_count
        reordered = (base + valid_rows[None, :]).T.reshape(-1)
        take_indices = pa.array(reordered, type=pa.int64())

    flat_keys = pc.take(keys_child_major, take_indices, memory_pool=options.arrow_memory_pool)
    flat_values = pc.take(values_child_major, take_indices, memory_pool=options.arrow_memory_pool)

    counts = np.where(row_mask, 0, child_count).astype(np.int32, copy=False)
    offsets_np = np.empty(row_count + 1, dtype=np.int32)
    offsets_np[0] = 0
    np.cumsum(counts, out=offsets_np[1:])
    offsets = pa.array(offsets_np, type=pa.int32())

    return pa.MapArray.from_arrays(
        offsets=offsets,
        keys=flat_keys,
        items=flat_values,
        mask=array.is_null(),
        type=target_type.to_arrow(),
        pool=options.arrow_memory_pool,
    )


def cast_polars_map_expr(
    expr: Any,
    options: "CastOptions",
) -> Any:
    pl = get_polars()

    if options.target_field is None:
        return expr
    elif options.source_field.dtype.type_id != DataTypeId.MAP:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_field: Field = options.source_field
    source_type: "MapType" = source_field.dtype
    target_type: MapType = options.target_field.dtype

    source_key_name = source_type.key_field.name
    source_value_name = source_type.value_field.name
    target_key_name = target_type.key_field.name
    target_value_name = target_type.value_field.name

    entry_expr = pl.struct([
        target_type.key_field.cast_polars_expr(
            pl.element().struct.field(source_key_name),
            options=options.copy(
                source_field=source_type.key_field,
                target_field=target_type.key_field,
            ),
        ).alias(target_key_name),
        target_type.value_field.cast_polars_expr(
            pl.element().struct.field(source_value_name),
            options=options.copy(
                source_field=source_type.value_field,
                target_field=target_type.value_field,
            ),
        ).alias(target_value_name),
    ])

    casted = expr.list.eval(entry_expr).cast(target_type.to_polars())
    return options.polars_alias(casted)


def cast_polars_list_expr_to_map(
    expr: Any,
    options: "CastOptions",
) -> Any:
    pl = get_polars()

    if options.target_field is None:
        return expr
    elif options.source_field.dtype.type_id != DataTypeId.ARRAY:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_field: Field = options.source_field
    target_field: Field = options.target_field

    source_type: ArrayType = source_field.dtype
    target_type: MapType = target_field.dtype

    source_item_dtype = source_type.item_field.dtype

    if source_item_dtype.type_id != DataTypeId.STRUCT or len(source_item_dtype.children_fields) != 2:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_key_field = source_item_dtype.field_at(0)
    source_value_field = source_item_dtype.field_at(1)

    target_key_name = target_type.key_field.name
    target_value_name = target_type.value_field.name

    entry_expr = pl.struct([
        target_type.key_field.cast_polars_expr(
            pl.element().struct.field(source_key_field.name),
            options=options.copy(
                source_field=source_key_field,
                target_field=target_type.key_field,
            ),
        ),
        target_type.value_field.cast_polars_expr(
            pl.element().struct.field(source_value_field.name),
            options=options.copy(
                source_field=source_value_field,
                target_field=target_type.value_field,
            ),
        ),
    ])

    casted = expr.list.eval(entry_expr).cast(target_type.to_polars())
    return options.polars_alias(casted)


def cast_polars_struct_expr_to_map(
    expr: Any,
    options: "CastOptions",
) -> Any:
    pl = get_polars()

    if options.target_field is None:
        return expr
    elif options.source_field.dtype.type_id != DataTypeId.STRUCT:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_field: Field = options.source_field
    target_field: Field = options.target_field

    source_type: StructType = source_field.dtype
    target_type: MapType = target_field.dtype

    target_key_name = target_type.key_field.name
    target_value_name = target_type.value_field.name

    entries = []
    for child in source_type.children_fields:
        key_expr = target_type.key_field.cast_polars_expr(
            pl.lit(child.name),
            options=options.copy(
                source_field=_string_key_source_field(),
                target_field=target_type.key_field,
            ),
        )

        value_expr = target_type.value_field.cast_polars_expr(
            expr.struct.field(child.name),
            options=options.copy(
                source_field=child,
                target_field=target_type.value_field,
            ),
        )

        entries.append(
            pl.struct([
                key_expr.alias(target_key_name),
                value_expr.alias(target_value_name),
            ])
        )

    list_expr = pl.concat_list(entries).cast(target_type.to_polars())

    casted = pl.when(expr.is_null()).then(pl.lit(None)).otherwise(list_expr)
    return options.polars_alias(casted)


def cast_polars_map_series(
    series: "polars.Series",
    options: "CastOptions",
) -> "polars.Series":
    pl = get_polars()
    expr = cast_polars_map_expr(pl.col(series.name), options)
    casted = pl.DataFrame({series.name: series}).select(expr).to_series()
    return options.polars_alias(casted)


def cast_polars_list_series_to_map(
    series: "polars.Series",
    options: "CastOptions",
) -> "polars.Series":
    pl = get_polars()
    expr = cast_polars_list_expr_to_map(pl.col(series.name), options)
    casted = pl.DataFrame({series.name: series}).select(expr).to_series()
    return options.polars_alias(casted)


def cast_polars_struct_series_to_map(
    series: "polars.Series",
    options: "CastOptions",
) -> "polars.Series":
    pl = get_polars()
    expr = cast_polars_struct_expr_to_map(pl.col(series.name), options)
    casted = pl.DataFrame({series.name: series}).select(expr).to_series()
    return options.polars_alias(casted)


def cast_pandas_map_series(
    series: "pd.Series",
    options: "CastOptions",
) -> "pd.Series":
    return _cast_pandas_via_arrow(series, options, cast_arrow_map_array)


def cast_pandas_list_series_to_map(
    series: "pd.Series",
    options: "CastOptions",
) -> "pd.Series":
    return _cast_pandas_via_arrow(series, options, cast_arrow_list_array_to_map)


def cast_pandas_struct_series_to_map(
    series: "pd.Series",
    options: "CastOptions",
) -> "pd.Series":
    return _cast_pandas_via_arrow(series, options, cast_arrow_struct_array_to_map)


def cast_spark_map_column(
    column: Any,
    options: "CastOptions",
) -> Any:
    spark = get_spark_sql()
    F = spark.functions

    options.check_source(column)

    if options.target_field is None:
        return column
    elif options.source_field.dtype.type_id != DataTypeId.MAP:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_field: Field = options.source_field
    target_field: Field = options.target_field

    source_type: MapType = source_field.dtype
    target_type: MapType = target_field.dtype

    entries = F.map_entries(column)

    key_array = F.transform(
        entries,
        lambda x: target_type.key_field.cast_spark_column(
            x[source_type.key_field.name],
            options=options.copy(
                source_field=source_type.key_field,
                target_field=target_type.key_field,
            ),
        ),
    )

    value_array = F.transform(
        entries,
        lambda x: target_type.value_field.cast_spark_column(
            x[source_type.value_field.name],
            options=options.copy(
                source_field=source_type.value_field,
                target_field=target_type.value_field,
            ),
        ),
    )

    casted = F.when(
        column.isNull(),
        F.lit(None),
    ).otherwise(F.map_from_arrays(key_array, value_array))
    return options.spark_alias(casted)


def cast_spark_list_column_to_map(
    column: Any,
    options: "CastOptions",
) -> Any:
    spark = get_spark_sql()
    F = spark.functions

    options.check_source(column)

    if options.target_field is None:
        return column
    elif options.source_field.dtype.type_id != DataTypeId.ARRAY:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_field: Field = options.source_field
    target_field: Field = options.target_field

    source_type: ArrayType = source_field.dtype
    target_type: MapType = target_field.dtype

    source_item_dtype = source_type.item_field.dtype

    if source_item_dtype.type_id != DataTypeId.STRUCT or len(source_item_dtype.children_fields) != 2:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_key_field = source_item_dtype.field_at(0)
    source_value_field = source_item_dtype.field_at(1)

    key_array = F.transform(
        column,
        lambda x: target_type.key_field.cast_spark_column(
            x[source_key_field.name],
            options=options.copy(
                source_field=source_key_field,
                target_field=target_type.key_field,
            ),
        ),
    )

    value_array = F.transform(
        column,
        lambda x: target_type.value_field.cast_spark_column(
            x[source_value_field.name],
            options=options.copy(
                source_field=source_value_field,
                target_field=target_type.value_field,
            ),
        ),
    )

    casted = F.when(
        column.isNull(),
        F.lit(None),
    ).otherwise(F.map_from_arrays(key_array, value_array))
    return options.spark_alias(casted)


def cast_spark_struct_column_to_map(
    column: Any,
    options: "CastOptions",
) -> Any:
    spark = get_spark_sql()
    F = spark.functions

    options.check_source(column)

    if options.target_field is None:
        return column
    elif options.source_field.dtype.type_id != DataTypeId.STRUCT:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_field: Field = options.source_field
    target_field: Field = options.target_field

    source_type: StructType = source_field.dtype
    target_type: MapType = target_field.dtype

    key_array = F.array(*[
        target_type.key_field.cast_spark_column(
            F.lit(child.name),
            options=options.copy(
                source_field=_string_key_source_field(),
                target_field=target_type.key_field,
            ),
        )
        for child in source_type.children_fields
    ])

    value_array = F.array(*[
        target_type.value_field.cast_spark_column(
            column[child.name],
            options=options.copy(
                source_field=child,
                target_field=target_type.value_field,
            ),
        )
        for child in source_type.children_fields
    ])

    casted = F.when(
        column.isNull(),
        F.lit(None),
    ).otherwise(F.map_from_arrays(key_array, value_array))
    return options.spark_alias(casted)
