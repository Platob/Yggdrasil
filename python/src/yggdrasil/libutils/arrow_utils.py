import datetime as dt
import decimal as dec
import enum
import functools
import uuid
import zoneinfo
from datetime import timezone
from typing import Dict, List, Set, Tuple

import polars as pl
import pyarrow as pa

from .numpy_utils import numpy
from .pandas_utils import PandasDataFrame, PandasSeries
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
    # Basic Python types
    bool: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    str: pa.utf8(),
    bytes: pa.binary(),
    memoryview: pa.binary(),
    bytearray: pa.binary(),

    # Decimal and date/time types
    dec.Decimal: pa.decimal128(38,18),
    dt.datetime: pa.timestamp("us"),
    # Handle a timezone-aware datetime explicitly - use UTC by default
    timezone: pa.timestamp("us", tz="UTC"),
    zoneinfo.ZoneInfo: pa.timestamp("us", tz="UTC"),
    dt.date: pa.date32(),
    dt.time: pa.time64("us"),
    dt.timedelta: pa.duration("us"),

    # Additional types
    uuid.UUID: pa.string(),  # UUIDs represented as strings
    type(None): pa.null(),
    enum.Enum: pa.string(),  # Enums represented as strings

    # Container types - these are approximate mappings
    # and will be processed specially when encountered
    list: pa.list_(pa.null()),
    tuple: pa.list_(pa.null()),
    set: pa.list_(pa.null()),
    frozenset: pa.list_(pa.null()),
    dict: pa.map_(pa.string(), pa.null()),
    Dict: pa.map_(pa.string(), pa.null()),
    List: pa.list_(pa.null()),
    Set: pa.list_(pa.null()),
    Tuple: pa.list_(pa.null()),
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
    pa.float32(): pa.scalar(0.0, pa.float32()),
    pa.float64(): pa.scalar(0.0, pa.float64()),
    pa.timestamp('s'): pa.scalar(0, pa.timestamp('s')),
    pa.timestamp('ms'): pa.scalar(0, pa.timestamp('ms')),
    pa.timestamp('us'): pa.scalar(0, pa.timestamp('us')),
    pa.timestamp('ns'): pa.scalar(0, pa.timestamp('ns')),
    pa.date32(): pa.scalar(0, pa.date32()),
    pa.date64(): pa.scalar(0, pa.date64()),
    pa.time32('s'): pa.scalar(0, pa.time32('s')),
    pa.time32('ms'): pa.scalar(0, pa.time32('ms')),
    pa.time64('us'): pa.scalar(0, pa.time64('us')),
    pa.time64('ns'): pa.scalar(0, pa.time64('ns')),
    pa.duration('s'): pa.scalar(0, pa.duration('s')),
    pa.duration('ms'): pa.scalar(0, pa.duration('ms')),
    pa.duration('us'): pa.scalar(0, pa.duration('us')),
    pa.duration('ns'): pa.scalar(0, pa.duration('ns')),
    # Common timezone-aware timestamp types
    pa.timestamp('s', tz='UTC'): pa.scalar(0, pa.timestamp('s', tz='UTC')),
    pa.timestamp('ms', tz='UTC'): pa.scalar(0, pa.timestamp('ms', tz='UTC')),
    pa.timestamp('us', tz='UTC'): pa.scalar(0, pa.timestamp('us', tz='UTC')),
    pa.timestamp('ns', tz='UTC'): pa.scalar(0, pa.timestamp('ns', tz='UTC')),
}

try:
    ARROW_DEFAULT_SCALARS[pa.float16()] = pa.scalar(numpy.float16(0), pa.float16())
except ImportError:
    pass


ArrowArrayLike = pa.Array | pa.ChunkedArray
ArrowTabular = pa.Table | pa.RecordBatch


def safe_arrow_tabular(obj) -> ArrowTabular:
    if isinstance(obj, (pa.RecordBatch, pa.Table)):
        return obj
    if isinstance(obj, pl.DataFrame):
        return obj.to_arrow()
    if isinstance(obj, PandasDataFrame):
        return pa.table(obj)
    if isinstance(obj, PandasSeries):
        return safe_arrow_tabular(obj.to_frame())

    raise TypeError(f"Cannot convert {type(obj)} to arrow Table or RecordBatch")


def apply_arrow_array_like(func):
    """
    Decorator: if the first argument is a ChunkedArray, apply `func` to each chunk
    and return a ChunkedArray; otherwise apply directly to the array.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not args:
            raise ValueError("Function must have at least one positional argument")

        first_arg = args[0]

        if isinstance(first_arg, pa.ChunkedArray):
            # apply func to each chunk
            chunks = [
                func(chunk, *args[1:], **kwargs)
                for chunk in first_arg.chunks
            ]
            # reconstruct chunked array
            return pa.chunked_array(chunks, type=first_arg.type)
        elif isinstance(first_arg, pa.Array):
            return func(*args, **kwargs)
        else:
            raise TypeError(f"Expected pa.Array or pa.ChunkedArray, got {type(first_arg)}")

    return wrapper


@apply_arrow_array_like
def get_child_array(
    arr: ArrowArrayLike | pa.StructArray,
    field: pa.Field,
    index: int | None = None,
    strict_names: bool | None = None,
    default: pa.Scalar | None = None,
    memory_pool: pa.MemoryPool | None = None
) -> ArrowArrayLike:
    assert pa.types.is_struct(arr.type), f"Arrow array type {arr.type} is not struct"
    arrow_type: pa.StructType = arr.type

    if not index or index < 0:
        index = index_of(
            collection=arrow_type.names,
            value=field.name,
            strict_names=strict_names,
            raise_error=False
        )

        if index < 0:
            if strict_names:
                raise pa.ArrowInvalid(f"Cannot find '{field.name}' in {arrow_type}")

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

    if found is not None:
        return found

    # Special handling for timezone-aware timestamps not in our predefined list
    if pa.types.is_timestamp(arrow_type) and arrow_type.tz is not None:
        # Create a default timestamp with the specific timezone
        return pa.scalar(0, arrow_type)

    # handle list types recursively
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
        # create an empty array and then a scalar from it
        list_arr = pa.array([], type=arrow_type)
        return pa.scalar(list_arr, arrow_type)

    # handle struct types recursively
    if pa.types.is_struct(arrow_type):
        fields: pa.StructType = arrow_type
        struct_values = {}
        for f in fields:
            struct_values[f.name] = arrow_default_scalar(f.type, nullable=f.nullable)
        return pa.scalar(struct_values, arrow_type)

    # handle map types recursively
    if pa.types.is_map(arrow_type):
        # create single-entry map array and then scalar
        map_arr = pa.array([], type=arrow_type)
        return pa.scalar(map_arr, arrow_type)

    # handle dictionary types
    if pa.types.is_dictionary(arrow_type):
        # For dictionary types, create a default for the index type
        dict_type = arrow_type
        # Create an empty string scalar with the value type
        return pa.scalar("", dict_type.value_type)

    # handle union types (both dense and sparse)
    if pa.types.is_union(arrow_type):
        # For union types, use the default of the first type
        # Union types store field types differently in different PyArrow versions
        try:
            # Get the number of fields
            n_fields = arrow_type.num_fields
            if n_fields > 0:
                # Get the first field type
                first_field_type = arrow_type.field(0).type
                # Create a default value for the first field type
                return arrow_default_scalar(first_field_type, nullable=False)
            else:
                raise pa.ArrowInvalid(f"Cannot generate default scalar for empty union type {arrow_type}")
        except (AttributeError, pa.ArrowInvalid):
            # Fall back to using null
            return pa.scalar(None, arrow_type)

    # handle extension types
    if pa.types.is_extension_type(arrow_type):
        # Get the storage type and create a default scalar for it
        storage_type = arrow_type.storage_type
        storage_default = arrow_default_scalar(storage_type, nullable=False)
        # Convert the storage default to the extension type
        try:
            return pa.scalar(storage_default.as_py(), arrow_type)
        except:
            # If we can't convert directly, try creating an array first and then getting a scalar
            try:
                # Create an empty array with the extension type
                ext_array = pa.array([], type=arrow_type)
                if len(ext_array) == 0:
                    # We need at least one value, so create a zero-length array of the storage type
                    # and wrap it in the extension type
                    storage_array = pa.array([storage_default.as_py()], type=storage_type)
                    ext_array = pa.ExtensionArray.from_storage(arrow_type, storage_array)
                    # Return the first value as a scalar
                    return ext_array[0]
            except:
                # If all else fails, fall back to a null value
                raise pa.ArrowInvalid(f"Cannot generate non-nullable default scalar for extension type {arrow_type}")

    raise pa.ArrowInvalid(f"Cannot generate non-nullable default scalar for arrow type {arrow_type}")
