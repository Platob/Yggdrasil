from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    import pandas as pd
    import polars
    import pyspark.sql as psql
    import pyspark.sql.types as pst
    from yggdrasil.data.cast.options import CastOptions
    from yggdrasil.data.data_field import Field
    from .array import ArrayType
    from .map import MapType


__all__ = [
    "NestedType",
    "StructType",
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructType(NestedType):
    fields: tuple["Field"] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))

    @property
    def children_fields(self) -> tuple["Field"]:
        return self.fields

    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.STRUCT

    def _merge_with_same_id(
        self,
        other: "NestedType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "StructType":
        if not isinstance(other, StructType):
            raise TypeError(
                f"Cannot merge {self.__class__.__name__} with {other.__class__.__name__}"
            )

        merged_fields: list[Field] = []
        seen: set[str] = set()

        for i, self_field in enumerate(self.fields):
            other_field = other.field_by(name=self_field.name, index=i, raise_error=False)

            if other_field is None:
                merged_fields.append(self_field)
            else:
                merged_fields.append(
                    self_field.merge_with(other_field, downcast=downcast, upcast=upcast)
                )
                seen.add(other_field.name)

        if mode is None or mode in (SaveMode.APPEND, SaveMode.UPSERT, SaveMode.AUTO):
            for other_field in other.fields:
                if other_field.name not in seen:
                    merged_fields.append(other_field)

        return self.__class__(fields=tuple(merged_fields))

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_struct(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "StructType":
        if not pa.types.is_struct(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")
        return cls(fields=[cls.get_data_field_class().from_arrow_field(f) for f in dtype])

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = get_polars()
        return isinstance(dtype, pl.Struct)

    @classmethod
    def from_polars_type(cls, dtype: "polars.Struct") -> "StructType":
        return cls(fields=[
            cached_from_import("yggdrasil.data.data_field", "Field").from_polars(f)
            for f in dtype.fields
        ])

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = get_spark_sql()
        return isinstance(dtype, spark.types.StructType)

    @classmethod
    def from_spark_type(cls, dtype: "pst.StructType") -> "StructType":
        return cls(fields=[
            cached_from_import("yggdrasil.data.data_field", "Field").from_spark(f)
            for f in dtype.fields
        ])

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.STRUCT)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StructType":
        return cls(fields=[
            cached_from_import("yggdrasil.data.data_field", "Field").from_any(f)
            for f in value.get("fields", [])
        ])

    def to_arrow(self) -> pa.DataType:
        return pa.struct([f.to_arrow_field() for f in self.fields])

    def to_polars(self) -> "polars.DataType":
        pl = get_polars()
        return pl.Struct([f.to_polars_field() for f in self.fields])

    def to_spark(self) -> Any:
        spark = get_spark_sql()
        return spark.types.StructType([f.to_pyspark_field() for f in self.fields])

    def to_databricks_ddl(self) -> str:
        fields_ddl = ", ".join(
            # Double any embedded backticks so Databricks/Spark parses the
            # backtick-quoted identifier correctly.
            f"`{f.name.replace('`', '``')}`: {f.dtype.to_databricks_ddl()}"
            for f in self.fields
        )
        return f"STRUCT<{fields_ddl}>"

    def to_dict(self) -> dict[str, Any]:
        base = super(StructType, self).to_dict()
        base["fields"] = [f.to_dict() for f in self.fields]
        return base

    def with_fields(
        self,
        fields: list[Field],
        safe: bool = False,
        inplace: bool = True,
    ) -> "StructType":
        if not safe:
            _f = self.get_data_field_class()
            fields = [_f.from_any(_) for _ in fields]

        if inplace:
            object.__setattr__(self, "fields", tuple(fields))
            return self
        return self.__class__(fields=tuple(fields))

    def default_pyobj(self, nullable: bool) -> Any:
        # Cascade through each child so non-nullable children get the dtype's
        # own default instead of silently degrading to ``None`` — that
        # guarantees ``pa.scalar(default, type=struct_type)`` round-trips and
        # recursively fills nested structs / arrays / maps.
        if nullable:
            return None
        return {f.name: f.default_pyobj for f in self.fields}

    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.StructArray | pa.ChunkedArray:
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

        elif source_type_id == DataTypeId.STRUCT:
            return cast_arrow_struct_array(array, options)

        elif source_type_id == DataTypeId.MAP:
            return cast_arrow_map_array(array, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_arrow_list_array(array, options)

        else:
            raise pa.ArrowInvalid(
                f"Cannot cast {options.source_field} to {options.target_field}"
            )

    def _cast_arrow_tabular(
        self,
        table: pa.Table | pa.RecordBatch,
        options: "CastOptions",
    ):
        options.check_source(table)
        options.check_target(self)
        return cast_arrow_tabular(table, options)

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

        elif source_type_id == DataTypeId.STRUCT:
            return cast_polars_struct_expr(expr, options)

        elif source_type_id == DataTypeId.MAP:
            return cast_polars_map_expr(expr, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_polars_list_expr(expr, options)

        else:
            raise TypeError(
                f"Cannot cast {options.source_field} to {options.target_field}"
            )

    def _cast_polars_tabular(
        self,
        table: "polars.DataFrame | polars.LazyFrame",
        options: "CastOptions",
    ):
        return cast_polars_tabular(
            table,
            options.check_source(table).check_target(self),
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
            return _cast_pandas_struct_via_arrow(
                series, options, cast_arrow_json_string_array
            )

        elif source_type_id == DataTypeId.STRUCT:
            return cast_pandas_struct_series(series, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_pandas_list_series(series, options)

        else:
            raise TypeError(
                f"Cannot cast {options.source_field} to {options.target_field}"
            )

    def _cast_pandas_tabular(
        self,
        frame: "pd.DataFrame",
        options: "CastOptions",
    ) -> "pd.DataFrame":
        return cast_pandas_tabular(frame, options.check_source(frame).check_target(self))

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

        elif source_type_id == DataTypeId.STRUCT:
            return cast_spark_struct_column(column, options)

        elif source_type_id == DataTypeId.MAP:
            return cast_spark_map_column(column, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_spark_list_column(column, options)

        else:
            raise TypeError(
                f"Cannot cast {options.source_field} to {options.target_field}"
            )

    def _cast_spark_tabular(
        self,
        frame: "psql.DataFrame",
        options: "CastOptions",
    ) -> "psql.DataFrame":
        return cast_spark_tabular(frame, options.check_source(frame).check_target(self))


def cast_arrow_struct_array(
    array: pa.StructArray,
    options: "CastOptions",
):
    options.check_source(array)

    if options.target_field is None:
        return array
    elif options.source_field.dtype.type_id != DataTypeId.STRUCT:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    source_field: Field = options.source_field
    source_type: StructType = source_field.dtype
    target_type: StructType = options.target_field.dtype

    children: list[pa.Array] = []

    for i, target_child in enumerate(target_type.children_fields):
        source_child = source_type.field_by(name=target_child.name, index=i, raise_error=False)

        if source_child is None:
            children.append(
                target_child.default_arrow_array(
                    size=len(array),
                    memory_pool=options.arrow_memory_pool,
                )
            )
        else:
            children.append(
                target_child.cast_arrow_array(
                    array.field(source_child.name),
                    options=options.copy(source_field=source_child, target_field=target_child),
                )
            )

    return pa.StructArray.from_arrays(
        children,
        fields=[f.to_arrow_field() for f in target_type.fields],
        mask=array.is_null(),
        memory_pool=options.arrow_memory_pool,
    )


def cast_arrow_map_array(
    array: pa.MapArray,
    options: "CastOptions",
):
    options.check_source(array)

    if options.target_field is None:
        return array
    elif options.source_field.dtype.type_id != DataTypeId.MAP:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    source_field: Field = options.source_field
    source_type: "MapType" = source_field.dtype
    target_type: StructType = options.target_field.dtype

    children: list[pa.Array] = []

    for target_child in target_type.children_fields:
        values = pc.map_lookup(array, target_child.name, occurrence="first")
        casted = target_child.cast_arrow_array(
            values,
            options=options.copy(
                source_field=source_type.value_field,
                target_field=target_child,
            ),
        )
        children.append(casted)

    return pa.StructArray.from_arrays(
        children,
        fields=[f.to_arrow_field() for f in target_type.fields],
        mask=array.is_null(),
        memory_pool=options.arrow_memory_pool,
    )


def cast_arrow_list_array(
    array: pa.Array | pa.ChunkedArray,
    options: "CastOptions",
):
    options.check_source(array)

    if options.target_field is None:
        return array
    elif options.source_field.dtype.type_id != DataTypeId.ARRAY:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    source_field: Field = options.source_field
    source_type: "ArrayType" = source_field.dtype
    target_type: StructType = options.target_field.dtype

    values_py = array.to_pylist()
    children: list[pa.Array] = []

    for i, target_child in enumerate(target_type.children_fields):
        extracted_py = [
            None if row is None or i >= len(row) else row[i]
            for row in values_py
        ]
        extracted = pa.array(
            extracted_py,
            memory_pool=options.arrow_memory_pool,
        )

        casted = target_child.cast_arrow_array(
            extracted,
            options=options.copy(
                source_field=source_type.item_field,
                target_field=target_child,
            ),
        )
        children.append(casted)

    return pa.StructArray.from_arrays(
        children,
        fields=[f.to_arrow_field() for f in target_type.fields],
        mask=array.is_null() if isinstance(array, pa.Array) else None,
        memory_pool=options.arrow_memory_pool,
    )


def cast_arrow_tabular(
    data: pa.Table | pa.RecordBatch,
    options: "CastOptions",
) -> pa.Table | pa.RecordBatch:
    if not isinstance(data, (pa.Table, pa.RecordBatch)):
        raise TypeError(f"Unsupported tabular type: {type(data)!r}")

    options.check_source(data)

    if options.target_field is None:
        return data

    source_schema = options.source_schema
    target_schema = options.target_schema

    target_arrays: list[pa.Array] = []
    num_rows = data.num_rows

    for i, target_field in enumerate(target_schema.children_fields):
        source_field = source_schema.field_by(
            name=target_field.name,
            index=i,
            raise_error=False,
        )

        if source_field is None:
            casted = target_field.default_arrow_array(
                size=num_rows,
                memory_pool=options.arrow_memory_pool,
            )
        else:
            source_array = data.column(source_field.name)
            casted = target_field.cast_arrow_array(
                source_array,
                options=options.copy(
                    source_field=source_field,
                    target_field=target_field,
                ),
            )

        target_arrays.append(casted)

    target_arrow_schema = target_schema.to_arrow_schema()

    if isinstance(data, pa.Table):
        return pa.Table.from_arrays(target_arrays, schema=target_arrow_schema)
    elif isinstance(data, pa.RecordBatch):
        return pa.RecordBatch.from_arrays(target_arrays, schema=target_arrow_schema)
    else:
        raise TypeError(f"Unsupported tabular type: {type(data)!r}")


# Polars

def cast_polars_struct_expr(
    expr: Any,
    options: "CastOptions",
) -> Any:
    pl = get_polars()

    if options.target_field is None:
        return expr
    elif options.source_field.dtype.type_id != DataTypeId.STRUCT:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_field: Field = options.source_field
    source_type: StructType = source_field.dtype
    target_type: StructType = options.target_field.dtype

    fields: list[Any] = []

    for i, target_child in enumerate(target_type.children_fields):
        source_child = source_type.field_by(name=target_child.name, index=i, raise_error=False)

        if source_child is None:
            child_expr = target_child.default_polars_expr(alias=target_child.name)
        else:
            child_expr = target_child.cast_polars_expr(
                expr.struct.field(source_child.name),
                options=options.copy(
                    source_field=source_child,
                    target_field=target_child,
                ),
            ).alias(target_child.name)

        fields.append(child_expr)

    struct_expr = pl.struct(fields)

    return pl.when(expr.is_null()).then(pl.lit(None)).otherwise(struct_expr)


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
    source_type: MapType = source_field.dtype
    target_type: MapType = options.target_field.dtype

    fields: list[Any] = []

    for target_child in target_type.children_fields:
        matched_values = expr.list.eval(
            pl.when(
                pl.element().struct.field(source_type.key_field.name) == pl.lit(target_child.name)
            )
            .then(pl.element().struct.field(source_type.value_field.name))
            .otherwise(None)
        )

        extracted = matched_values.list.drop_nulls().list.first()

        casted = target_child.cast_polars_expr(
            extracted,
            options=options.copy(
                source_field=source_type.value_field,
                target_field=target_child,
            ),
        ).alias(target_child.name)

        fields.append(casted)

    struct_expr = pl.struct(fields)
    casted = pl.when(expr.is_null()).then(pl.lit(None)).otherwise(struct_expr)
    return options.polars_alias(casted)


def cast_polars_list_expr(
    expr: Any,
    options: "CastOptions",
) -> Any:
    pl = get_polars()

    if options.target_field is None:
        return expr
    elif options.source_field.dtype.type_id != DataTypeId.ARRAY:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_field: Field = options.source_field
    source_type: "ArrayType" = source_field.dtype
    target_type: StructType = options.target_field.dtype

    fields: list[Any] = []

    for i, target_child in enumerate(target_type.children_fields):
        extracted = expr.list.get(i, null_on_oob=True)

        casted = target_child.cast_polars_expr(
            extracted,
            options=options.copy(
                source_field=source_type.item_field,
                target_field=target_child,
            ),
        ).alias(target_child.name)

        fields.append(casted)

    struct_expr = pl.struct(fields)
    return pl.when(expr.is_null()).then(pl.lit(None)).otherwise(struct_expr)


def cast_polars_struct_series(
    series: "polars.Series",
    options: "CastOptions",
) -> "polars.Series":
    pl = get_polars()
    expr = cast_polars_struct_expr(pl.col(series.name), options).alias(options.target_field.name)
    return pl.DataFrame({series.name: series}).select(expr).to_series()


def cast_polars_list_series(
    series: "polars.Series",
    options: "CastOptions",
) -> "polars.Series":
    pl = get_polars()
    expr = cast_polars_list_expr(pl.col(series.name), options).alias(options.target_field.name)
    return pl.DataFrame({series.name: series}).select(expr).to_series()


def cast_polars_tabular(
    data: "polars.DataFrame | polars.LazyFrame",
    options: "CastOptions",
) -> "polars.DataFrame | polars.LazyFrame":
    pl = get_polars()

    if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(f"Unsupported tabular type: {type(data)!r}")

    options = options.check_source(data)

    if options.target_schema is None:
        return data

    source_schema = options.source_schema
    target_schema = options.target_schema

    exprs: list[Any] = []

    for i, target_field in enumerate(target_schema.children_fields):
        source_field = source_schema.field_by(
            name=target_field.name,
            index=i,
            raise_error=False,
        )

        if source_field is None:
            expr = target_field.default_polars_expr(alias=target_field.name)
        else:
            expr = target_field.cast_polars_expr(
                pl.col(source_field.name),
                options=options.copy(
                    source_field=source_field,
                    target_field=target_field,
                ),
            )

        exprs.append(expr)

    return data.select(exprs)


# Pandas

def _cast_pandas_struct_via_arrow(
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

    return pd.Series(
        values,
        index=series.index,
        name=options.target_field.name,
        dtype="object",
    )


def cast_pandas_struct_series(
    series: "pd.Series",
    options: "CastOptions",
) -> "pd.Series":
    pd = get_pandas()

    options.check_source(series)

    if options.target_field is None:
        return series
    elif options.source_field.dtype.type_id != DataTypeId.STRUCT:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_field: Field = options.source_field
    source_type: StructType = source_field.dtype
    target_type: StructType = options.target_field.dtype

    def normalize_row(value: Any) -> dict[str, Any] | None:
        if value is None or (pd.isna(value) if not isinstance(value, dict) else False):
            return None
        if isinstance(value, dict):
            return value
        if hasattr(value, "asDict"):
            return value.asDict(recursive=True)
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        raise TypeError(f"Unsupported struct-like pandas value: {type(value)!r}")

    normalized = series.map(normalize_row)
    num_rows = len(series)
    out_cols: dict[str, pd.Series] = {}

    for i, target_child in enumerate(target_type.children_fields):
        source_child = source_type.field_by(
            name=target_child.name,
            index=i,
            raise_error=False,
        )

        if source_child is None:
            out_cols[target_child.name] = target_child.default_pandas_series(size=num_rows)
            continue

        extracted = normalized.map(
            lambda row, key=source_child.name: None if row is None else row.get(key)
        )

        out_cols[target_child.name] = target_child.cast_pandas_series(
            extracted,
            options=options.copy(
                source_field=source_child,
                target_field=target_child,
            ),
        )

    rows: list[dict[str, Any] | None] = []

    for row_idx in range(num_rows):
        src_row = normalized.iloc[row_idx]
        if src_row is None:
            rows.append(None)
            continue

        row: dict[str, Any] = {}
        for target_child in target_type.children_fields:
            row[target_child.name] = out_cols[target_child.name].iloc[row_idx]
        rows.append(row)

    return pd.Series(rows, index=series.index, name=options.target_field.name, dtype="object")


def cast_pandas_list_series(
    series: "pd.Series",
    options: "CastOptions",
) -> "pd.Series":
    pd = get_pandas()

    options.check_source(series)

    if options.target_field is None:
        return series
    elif options.source_field.dtype.type_id != DataTypeId.ARRAY:
        raise TypeError(f"Cannot cast {options.source_field} to {options.target_field}")

    source_field: Field = options.source_field
    source_type: "ArrayType" = source_field.dtype
    target_type: StructType = options.target_field.dtype

    def normalize_row(value: Any) -> list[Any] | None:
        if value is None:
            return None
        if pd.isna(value) if not isinstance(value, (list, tuple)) else False:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
            out = value.tolist()
            return out if isinstance(out, list) else list(out)
        raise TypeError(f"Unsupported list-like pandas value: {type(value)!r}")

    normalized = series.map(normalize_row)
    num_rows = len(series)
    out_cols: dict[str, pd.Series] = {}

    for i, target_child in enumerate(target_type.children_fields):
        extracted = normalized.map(
            lambda row, idx=i: None if row is None or idx >= len(row) else row[idx]
        )

        out_cols[target_child.name] = target_child.cast_pandas_series(
            extracted,
            options=options.copy(
                source_field=source_type.item_field,
                target_field=target_child,
            ),
        )

    rows: list[dict[str, Any] | None] = []

    for row_idx in range(num_rows):
        src_row = normalized.iloc[row_idx]
        if src_row is None:
            rows.append(None)
            continue

        row: dict[str, Any] = {}
        for target_child in target_type.children_fields:
            row[target_child.name] = out_cols[target_child.name].iloc[row_idx]
        rows.append(row)

    return pd.Series(rows, index=series.index, name=options.target_field.name, dtype="object")


def cast_pandas_tabular(
    data: "pd.DataFrame",
    options: "CastOptions",
) -> "pd.DataFrame":
    pd = get_pandas()

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Unsupported tabular type: {type(data)!r}")

    options.check_source(data)

    if options.target_schema is None:
        return data

    source_schema = options.source_schema
    target_schema = options.target_schema

    out: dict[str, pd.Series] = {}
    num_rows = len(data)

    for i, target_field in enumerate(target_schema.children_fields):
        source_field = source_schema.field_by(
            name=target_field.name,
            index=i,
            raise_error=False,
        )

        if source_field is None:
            casted = target_field.default_pandas_series(size=num_rows)
        else:
            casted = target_field.cast_pandas_series(
                data[source_field.name],
                options=options.copy(
                    source_field=source_field,
                    target_field=target_field,
                ),
            )

        out[target_field.name] = casted.rename(target_field.name)

    return pd.DataFrame(out, index=data.index)


# PySpark

def cast_spark_struct_column(
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
    source_type: StructType = source_field.dtype
    target_type: StructType = options.target_field.dtype

    child_columns: list[Any] = []

    for i, target_child in enumerate(target_type.children_fields):
        source_child = source_type.field_by(
            name=target_child.name,
            index=i,
            raise_error=False,
        )

        if source_child is None:
            child = target_child.default_spark_column(alias=target_child.name)
        else:
            child = target_child.cast_spark_column(
                column[source_child.name],
                options=options.copy(
                    source_field=source_child,
                    target_field=target_child,
                ),
            ).alias(target_child.name)

        child_columns.append(child)

    # Preserve null source rows: F.struct always returns a non-null struct,
    # so without this guard a null row becomes {child: None, ...}.
    return F.when(column.isNull(), F.lit(None)).otherwise(F.struct(*child_columns))


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
    source_type: "MapType" = source_field.dtype
    target_type: StructType = options.target_field.dtype

    child_columns: list[Any] = []

    for target_child in target_type.children_fields:
        extracted = F.element_at(column, F.lit(target_child.name))

        casted = target_child.cast_spark_column(
            extracted,
            options=options.copy(
                source_field=source_type.value_field,
                target_field=target_child,
            ),
        ).alias(target_child.name)

        child_columns.append(casted)

    return F.when(column.isNull(), F.lit(None)).otherwise(F.struct(*child_columns))


def cast_spark_list_column(
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
    source_type: "ArrayType" = source_field.dtype
    target_type: StructType = options.target_field.dtype

    child_columns: list[Any] = []

    for i, target_child in enumerate(target_type.children_fields):
        # F.get is the null-on-out-of-bounds accessor; plain column[i] /
        # column.getItem(i) raises an ArrayIndexOutOfBoundsException in
        # Spark 4.x when the source list is shorter than the target struct.
        extracted = F.get(column, F.lit(i))

        casted = target_child.cast_spark_column(
            extracted,
            options=options.copy(
                source_field=source_type.item_field,
                target_field=target_child,
            ),
        ).alias(target_child.name)

        child_columns.append(casted)

    return F.when(column.isNull(), F.lit(None)).otherwise(F.struct(*child_columns))


def cast_spark_tabular(
    data: "psql.DataFrame",
    options: "CastOptions",
) -> "psql.DataFrame":
    spark = get_spark_sql()

    if not isinstance(data, spark.DataFrame):
        raise TypeError(f"Unsupported tabular type: {type(data)!r}")

    options.check_source(data)

    if options.target_schema is None:
        return data

    source_schema = options.source_schema
    target_schema = options.target_schema

    cols: list[Any] = []

    for i, target_field in enumerate(target_schema.children_fields):
        source_field = source_schema.field_by(
            name=target_field.name,
            index=i,
            raise_error=False,
        )

        if source_field is None:
            col = target_field.default_spark_column(alias=target_field.name)
        else:
            col = target_field.cast_spark_column(
                spark.functions.col(source_field.name),
                options=options.copy(
                    source_field=source_field,
                    target_field=target_field,
                ),
            ).alias(target_field.name)

        cols.append(col)

    return data.select(*cols)