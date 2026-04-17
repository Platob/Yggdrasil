"""JSON-string cast helpers shared by nested types.

Parsing from string/binary to nested structures goes through vectorised
JSON decoding so we avoid per-row Python loops.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.support import get_polars, get_spark_sql

if TYPE_CHECKING:
    from yggdrasil.data.cast.options import CastOptions


__all__ = [
    "is_json_string_source",
    "cast_arrow_json_string_array",
    "cast_polars_json_string_expr",
    "cast_polars_json_string_series",
    "cast_spark_json_string_column",
]


_JSON_STRING_SOURCE_TYPES = frozenset({DataTypeId.STRING, DataTypeId.BINARY})


def is_json_string_source(source_type_id: DataTypeId) -> bool:
    return source_type_id in _JSON_STRING_SOURCE_TYPES


def _arrow_to_utf8(
    array: pa.Array | pa.ChunkedArray,
    memory_pool: pa.MemoryPool | None = None,
) -> pa.Array:
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()

    src_type = array.type
    if (
        pa.types.is_binary(src_type)
        or pa.types.is_large_binary(src_type)
        or pa.types.is_binary_view(src_type)
    ):
        return pc.cast(array, pa.string(), memory_pool=memory_pool)

    if (
        pa.types.is_string(src_type)
        or pa.types.is_large_string(src_type)
        or pa.types.is_string_view(src_type)
    ):
        if not pa.types.is_string(src_type):
            return pc.cast(array, pa.string(), memory_pool=memory_pool)
        return array

    raise pa.ArrowInvalid(
        f"JSON cast expects a string/binary source, got {src_type!r}"
    )


def cast_arrow_json_string_array(
    array: pa.Array | pa.ChunkedArray,
    options: "CastOptions",
) -> pa.Array:
    """Parse a string/binary Arrow array as JSON into the target arrow type.

    Uses one C-level ``json.loads`` on a comma-joined blob (so no per-row
    Python iteration) followed by ``pa.array`` with the target type.
    """
    target_field = options.target_field
    if target_field is None:
        return array

    target_arrow_type = target_field.dtype.to_arrow()
    memory_pool = options.arrow_memory_pool

    normalized = _arrow_to_utf8(array, memory_pool=memory_pool)

    if len(normalized) == 0:
        return pa.array([], type=target_arrow_type)

    normalized = DataType._nullify_empty_arrow_strings(normalized)

    null_literal = pa.scalar("null", type=pa.string())
    filled = pc.fill_null(normalized, null_literal)

    blob = "[" + ",".join(filled.to_pylist()) + "]"
    parsed = json.loads(blob)

    return pa.array(parsed, type=target_arrow_type, memory_pool=memory_pool)


def cast_polars_json_string_expr(
    expr: Any,
    options: "CastOptions",
) -> Any:
    pl = get_polars()

    target_field = options.target_field
    if target_field is None:
        return expr

    target_polars_type = target_field.dtype.to_polars()

    if options.source_field.dtype.type_id == DataTypeId.BINARY:
        expr = expr.cast(pl.String)

    return expr.str.json_decode(target_polars_type)


def cast_polars_json_string_series(
    series: Any,
    options: "CastOptions",
) -> Any:
    pl = get_polars()
    expr = cast_polars_json_string_expr(pl.col(series.name), options).alias(
        options.target_field.name
    )
    return pl.DataFrame({series.name: series}).select(expr).to_series()


def cast_spark_json_string_column(
    column: Any,
    options: "CastOptions",
) -> Any:
    spark = get_spark_sql()
    F = spark.functions

    target_field = options.target_field
    if target_field is None:
        return column

    target_ddl = target_field.dtype.to_databricks_ddl()

    if options.source_field.dtype.type_id == DataTypeId.BINARY:
        column = column.cast("string")

    return F.from_json(column, target_ddl)
