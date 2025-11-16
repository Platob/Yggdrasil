import decimal as dec
import datetime as dt
import pyarrow as pa
import polars as pl
import pandas as pd

from .py_utils import index_of

__all__ = [
    "PYTHON_TO_ARROW_TYPE_MAP",
    "ArrowTabular",
    "ArrowArrayLike",
    "safe_arrow_tabular",
    "arrow_default_scalar",
    "array_nulls",
    "array_length",
    "get_child_array"
]


PYTHON_TO_ARROW_TYPE_MAP = {
    bool: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    str: pa.utf8(),
    bytes: pa.binary(),
    memoryview: pa.binary(),
    bytearray: pa.binary(),
    dec.Decimal: pa.decimal128(38,18),
    dt.datetime: pa.timestamp("us"),
    dt.date: pa.date32(),
}


ARROW_DEFAULT_SCALARS: dict[pa.DataType, pa.Scalar] = {
    pa.string(): pa.scalar("", pa.string()),
    pa.bool_(): pa.scalar(False, pa.bool_()),
    pa.int8(): pa.scalar(0, pa.int8()),
    pa.int16(): pa.scalar(0, pa.int16()),
    pa.int32(): pa.scalar(0, pa.int32()),
    pa.int64(): pa.scalar(0, pa.int64()),
    pa.uint8(): pa.scalar(0, pa.uint8()),
    pa.uint16(): pa.scalar(0, pa.uint16()),
    pa.uint32(): pa.scalar(0, pa.uint32()),
    pa.uint64(): pa.scalar(0, pa.uint64()),
    pa.float16(): pa.scalar(0.0, pa.float16()),
    pa.float32(): pa.scalar(0.0, pa.float32()),
    pa.float64(): pa.scalar(0.0, pa.float64()),
}


ArrowArrayLike = pa.Array | pa.ChunkedArray
ArrowTabular = pa.Table | pa.RecordBatch


def safe_arrow_tabular(obj) -> ArrowTabular:
    if isinstance(obj, (pa.RecordBatch, pa.Table)):
        return obj
    if isinstance(obj, pl.DataFrame):
        return obj.to_arrow()
    if isinstance(obj, pd.DataFrame):
        return pa.table(obj)
    if isinstance(obj, pd.Series):
        return safe_arrow_tabular(obj.to_frame())

    raise TypeError(f"Cannot convert {type(obj)} to arrow Table or RecordBatch")


def get_child_array(
    field: pa.Field,
    arr: ArrowArrayLike | pa.StructArray,
    index: int | None = None,
    safe: bool | None = None,
    default: pa.Scalar | None = None,
    memory_pool: pa.MemoryPool | None = None
) -> ArrowArrayLike:
    assert pa.types.is_struct(field.type), f"Arrow field {field} is not struct"
    arrow_type: pa.StructType = field.type

    if not index or index < 0:
        index = index_of(collection=arrow_type.names, value=field.name, raise_error=False)

        if index < 0:
            if safe:
                raise pa.ArrowInvalid(f"Cannot find '{field.name}' in {arrow_type}")


            if isinstance(arr, pa.ChunkedArray):
                return pa.chunked_array([
                    array_nulls(
                        field=field,
                        size=array_length(chunk),
                        default=default,
                        memory_pool=memory_pool
                    )
                    for chunk in arr.chunks
                ])

            return array_nulls(
                field=field,
                size=len(arr),
                default=default,
                memory_pool=memory_pool
            )

    return arr.field(index=index)


def array_length(
    arr: ArrowArrayLike
) -> int:
    if isinstance(arr, pa.Array):
        return len(arr)
    return arr.length()


def array_nulls(
    field: pa.Field,
    size: int,
    default: pa.Scalar | None = None,
    memory_pool: pa.MemoryPool | None = None
) -> pa.Array:
    if field.nullable:
        return pa.nulls(size=size, type=field.type, memory_pool=memory_pool)

    default = default or arrow_default_scalar(arrow_type=field.type, nullable=field.nullable)

    return pa.repeat(default, size=size, memory_pool=memory_pool)


def arrow_default_scalar(arrow_type: pa.DataType, nullable: bool) -> pa.Scalar:
    if nullable:
        return pa.scalar(None, arrow_type)

    # check primitive default
    found = ARROW_DEFAULT_SCALARS.get(arrow_type)

    if found:
        return found

    # handle list types recursively
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
        value_field = arrow_type.value_field
        # recursively generate default for element
        element_scalar = arrow_default_scalar(value_field.type, nullable=value_field.nullable)
        # create an empty array and then a scalar from it
        arr = pa.array([element_scalar], type=value_field.type)
        list_arr = pa.array([arr], type=arrow_type)
        return pa.scalar(list_arr, arrow_type)

    # handle struct types recursively
    if pa.types.is_struct(arrow_type):
        fields = arrow_type
        struct_values = {}
        for f in fields:
            struct_values[f.name] = arrow_default_scalar(f.type, nullable=f.nullable)
        return pa.scalar(struct_values, arrow_type)

    # handle map types recursively
    if pa.types.is_map(arrow_type):
        key_scalar = arrow_default_scalar(arrow_type.key_type, nullable=False)
        item_scalar = arrow_default_scalar(arrow_type.item_type, nullable=arrow_type.item_nullable)
        # create single-entry map array and then scalar
        map_arr = pa.array([{key_scalar.as_py(): item_scalar.as_py()}], type=arrow_type)
        return pa.scalar(map_arr, arrow_type)

    raise pa.ArrowInvalid(f"Cannot generate non-nullable default scalar for arrow type {arrow_type}")