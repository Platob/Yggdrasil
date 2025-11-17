import datetime as dt
import decimal as dec
import enum
import functools
import uuid
import zoneinfo
from datetime import timezone
from typing import Dict, List, Set, Tuple

from .py_utils import (
    safe_str
)


def default_uuid() -> uuid.UUID:
    """Create a default UUID (all zeros/nil UUID)."""
    return uuid.UUID('00000000-0000-0000-0000-000000000000')


def uuid_to_bytes(uuid_val: uuid.UUID) -> bytes:
    """Convert a UUID to bytes for PyArrow compatibility."""
    return uuid_val.bytes

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
    "get_child_array",
    "refine_arrow_field",
    "dump_arrow_field_metadata",
    "get_base_type_name",
    "default_uuid",
    "uuid_to_bytes"
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
    uuid.UUID: pa.uuid(),  # UUIDs represented as native UUID type
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
    pa.uuid(): pa.scalar(uuid_to_bytes(default_uuid()), pa.uuid()),
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


def refine_arrow_field(field: pa.Field) -> pa.Field:
    """
    Apply type-specific refinements to an Arrow field based on its metadata.
    Adjusts the arrow type according to field type and metadata values and removes used metadata.

    Args:
        field: The PyArrow field to refine

    Returns:
        A new PyArrow field with refined type and cleaned metadata

    Example:
        >>> # A float field with metadata specifying precision and scale
        >>> field = pa.field('amount', pa.float64(), metadata={
        ...     b'precision': b'10',
        ...     b'scale': b'2',
        ...     b'description': b'Transaction amount'
        ... })
        >>> refined = refine_arrow_field(field)
        >>> print(refined)
        amount: decimal(10, 2) not null
        >>> print(refined.metadata)
        {b'description': b'Transaction amount'}
    """
    # Start with original field attributes
    arrow_type = field.type
    field_name = field.name
    nullable = field.nullable

    # Use original metadata directly or empty dict if None
    metadata = {} if field.metadata is None else field.metadata
    metadata_str = {}

    # Convert metadata binary keys/values to strings for easier handling
    for key, value in list(metadata.items()):
        if isinstance(key, bytes):
            key_str = safe_str(key)
        else:
            key_str = str(key)

        if isinstance(value, bytes):
            value_str = safe_str(value)
        else:
            value_str = str(value)

        metadata_str[key_str] = value_str

    # Decimal type conversion from data_type
    if pa.types.is_decimal(arrow_type) and 'precision' in metadata_str and 'scale' in metadata_str:
        try:
            precision = int(metadata_str['precision'])
            scale = int(metadata_str['scale'])
            arrow_type = pa.decimal128(precision, scale)
            # Pop used keys
            metadata.pop(b'precision' if b'precision' in metadata else 'precision', None)
            metadata.pop(b'scale' if b'scale' in metadata else 'scale', None)
        except (ValueError, TypeError):
            # If conversion fails, keep the original type
            pass

    # Handle time type
    elif pa.types.is_time(arrow_type) and 'unit' in metadata_str:
        unit = metadata_str['unit']
        if unit in ['s', 'ms']:
            arrow_type = pa.time32(unit)
            metadata.pop(b'unit' if b'unit' in metadata else 'unit', None)
        elif unit in ['us', 'ns']:
            arrow_type = pa.time64(unit)
            metadata.pop(b'unit' if b'unit' in metadata else 'unit', None)

    # Handle timestamp type
    elif pa.types.is_timestamp(arrow_type) and 'unit' in metadata_str:
        unit = metadata_str['unit']
        if unit in ['s', 'ms', 'us', 'ns']:
            if 'timezone' in metadata_str:
                timezone = metadata_str['timezone']
                arrow_type = pa.timestamp(unit, tz=timezone)
                metadata.pop(b'timezone' if b'timezone' in metadata else 'timezone', None)
            else:
                arrow_type = pa.timestamp(unit)
            metadata.pop(b'unit' if b'unit' in metadata else 'unit', None)

    # Handle duration type
    elif pa.types.is_duration(arrow_type) and 'unit' in metadata_str:
        unit = metadata_str['unit']
        if unit in ['s', 'ms', 'us', 'ns']:
            arrow_type = pa.duration(unit)
            metadata.pop(b'unit' if b'unit' in metadata else 'unit', None)

    # Process type refinements based on field type and metadata (not related to data_type)
    elif (pa.types.is_floating(arrow_type) or pa.types.is_decimal(arrow_type)) and 'precision' in metadata_str and 'scale' in metadata_str:
        try:
            precision = int(metadata_str['precision'])
            scale = int(metadata_str['scale'])
            arrow_type = pa.decimal128(precision, scale)
            # Pop used keys from metadata
            metadata.pop(b'precision' if b'precision' in metadata else 'precision', None)
            metadata.pop(b'scale' if b'scale' in metadata else 'scale', None)
        except (ValueError, TypeError):
            # If conversion fails, keep the original type
            pass

    # Handle integer with time units as timestamp/time/duration
    elif pa.types.is_integer(arrow_type) and 'unit' in metadata_str:
        unit = metadata_str['unit']
        if unit in ['s', 'ms', 'us', 'ns']:
            # Check for time-related type hints in other metadata
            if 'timezone' in metadata_str:
                # This is a timestamp with timezone
                timezone = metadata_str['timezone']
                arrow_type = pa.timestamp(unit, tz=timezone)
                # Pop used keys
                metadata.pop(b'unit' if b'unit' in metadata else 'unit', None)
                metadata.pop(b'timezone' if b'timezone' in metadata else 'timezone', None)
            else:
                # This could be a timestamp, time, or duration
                # Default to timestamp if no other indication
                arrow_type = pa.timestamp(unit)
                # Pop used key
                metadata.pop(b'unit' if b'unit' in metadata else 'unit', None)

    # Handle timestamp timezone addition
    elif pa.types.is_timestamp(arrow_type) and 'timezone' in metadata_str:
        timezone = metadata_str['timezone']
        arrow_type = pa.timestamp(arrow_type.unit, tz=timezone)
        # Pop used key
        metadata.pop(b'timezone' if b'timezone' in metadata else 'timezone', None)

    # Handle string with date format as date type
    elif pa.types.is_string(arrow_type) and 'format' in metadata_str:
        format_str = metadata_str['format'].lower()
        if any(date_hint in format_str for date_hint in ['date', 'yyyy', 'mm', 'dd']):
            arrow_type = pa.date32()
            # Pop used key
            metadata.pop(b'format' if b'format' in metadata else 'format', None)

    # Create a new field with refined type and remaining metadata
    return pa.field(field_name, arrow_type, nullable=nullable, metadata=metadata if metadata else None)


def dump_arrow_field_metadata(
    field: pa.Field,
    recursive: bool = False,
) -> dict:
    """
    Extract and format metadata from an Arrow field, optionally recursing through nested types.
    Returns a nested JSON structure with standardized key groups:
    - "type": Base type information (name, width, etc.)
    - "metadata": Field metadata from the metadata dict
    - "children": List of child fields (for nested types)
    - Type-specific keys (like "precision", "scale", etc.)

    Args:
        field: The PyArrow field to extract metadata from
        recursive: Whether to recursively extract metadata from nested fields

    Returns:
        A nested dictionary representing the field hierarchy and metadata

    Example:
        >>> schema = pa.schema([
        ...     pa.field('id', pa.int64()),
        ...     pa.field('details', pa.struct([
        ...         pa.field('name', pa.string(), metadata={b'description': b'Full name'}),
        ...         pa.field('value', pa.decimal128(10, 2), metadata={b'precision': b'10', b'scale': b'2'})
        ...     ]), metadata={b'description': b'Additional details'})
        ... ])
        >>> metadata = dump_arrow_field_metadata(schema.field('details'), recursive=True)
        >>> print(metadata)
        {
            "details": {
                "type": {
                    "name": "struct"
                },
                "metadata": {
                    "description": "Additional details"
                },
                "children": {
                    "name": {
                        "type": {
                            "name": "string",
                            "encoding": "UTF8"
                        },
                        "metadata": {
                            "description": "Full name"
                        }
                    },
                    "value": {
                        "type": {
                            "name": "decimal128"
                        },
                        "metadata": {
                            "precision": "10",
                            "scale": "2"
                        },
                        "precision": "10",
                        "scale": "2"
                    }
                }
            }
        }
    """
    def _create_field_data(field):
        """Helper function to create field metadata structure."""
        arrow_type = field.type
        field_data = {}

        # Add field metadata (convert from binary to string)
        metadata_dict = {}
        if field.metadata:
            for key, value in field.metadata.items():
                if isinstance(key, bytes):
                    key = safe_str(key)
                if isinstance(value, bytes):
                    value = safe_str(value)
                metadata_dict[key] = value

        if metadata_dict:
            field_data["metadata"] = metadata_dict

        # Add type as a simple string
        field_data["type"] = get_base_type_name(arrow_type)

        # Add type-specific attributes at the top level
        if (pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type) or
                pa.types.is_boolean(arrow_type)):
            try:
                field_data["bitwidth"] = safe_str(arrow_type.bit_width)
                field_data["bytewidth"] = safe_str(arrow_type.byte_width)
            except (AttributeError, ValueError):
                pass

        if pa.types.is_fixed_size_binary(arrow_type):
            field_data["byte_width"] = safe_str(arrow_type.byte_width)

        # Add type-specific information as top-level keys
        if pa.types.is_decimal(arrow_type):
            field_data["precision"] = safe_str(arrow_type.precision)
            field_data["scale"] = safe_str(arrow_type.scale)
        elif pa.types.is_date64(arrow_type):
            field_data["timeunit"] = "ms"
        elif pa.types.is_time(arrow_type):
            field_data["timeunit"] = safe_str(arrow_type.unit)
        elif pa.types.is_timestamp(arrow_type):
            field_data["timeunit"] = safe_str(arrow_type.unit)
            if arrow_type.tz:
                field_data["timezone"] = safe_str(arrow_type.tz)
        elif pa.types.is_string(arrow_type):
            field_data["encoding"] = "UTF8"

        return field_data

    # Create the result dictionary and add this field
    result = {}
    field_data = _create_field_data(field)
    result[field.name] = field_data

    # Handle nested types recursively if requested
    if recursive:
        arrow_type = field.type
        children = {}

        if pa.types.is_struct(arrow_type):
            for i in range(arrow_type.num_fields):
                child_field = arrow_type.field(i)
                # Recursively process this child's structure
                nested = dump_arrow_field_metadata(child_field, recursive=True)
                # Add to children dictionary
                children[child_field.name] = nested[child_field.name]

            if children:
                field_data["children"] = children

        elif pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
            child_field = arrow_type.value_field

            # Recursively process item field
            nested = dump_arrow_field_metadata(child_field, recursive=True)
            field_data["items"] = nested[child_field.name]

        elif pa.types.is_map(arrow_type):
            key_field = arrow_type.key_field
            item_field = arrow_type.item_field

            # Recursively process key and item fields
            key_data = dump_arrow_field_metadata(key_field, recursive=True)
            item_data = dump_arrow_field_metadata(item_field, recursive=True)

            field_data["keys"] = key_data[key_field.name]
            field_data["values"] = item_data[item_field.name]

        elif pa.types.is_union(arrow_type):
            variants = {}
            for i in range(arrow_type.num_fields):
                child_field = arrow_type.field(i)

                # Recursively process each variant field
                nested = dump_arrow_field_metadata(child_field, recursive=True)
                variants[child_field.name] = nested[child_field.name]

            if variants:
                field_data["variants"] = variants

    return result


def get_base_type_name(arrow_type: pa.DataType) -> str:
    """
    Get the base type name from an Arrow data type without parameters.

    Args:
        arrow_type: PyArrow data type

    Returns:
        String containing just the base type name (e.g., 'decimal', 'timestamp', 'list')
    """
    if pa.types.is_boolean(arrow_type):
        return 'bool'
    elif pa.types.is_integer(arrow_type):
        return 'integer'
    elif pa.types.is_floating(arrow_type):
        return 'float'
    elif pa.types.is_decimal(arrow_type):
        return 'decimal'
    elif pa.types.is_string(arrow_type):
        return 'string'
    elif pa.types.is_binary(arrow_type):
        return 'binary'
    elif pa.types.is_large_string(arrow_type):
        return 'string'
    elif pa.types.is_large_binary(arrow_type):
        return 'binary'
    elif pa.types.is_fixed_size_binary(arrow_type):
        return 'binary'
    elif pa.types.is_timestamp(arrow_type):
        return 'timestamp'
    elif pa.types.is_date(arrow_type):
        return 'date'
    elif pa.types.is_time(arrow_type):
        return 'time'
    elif pa.types.is_duration(arrow_type):
        return 'duration'
    elif pa.types.is_list(arrow_type):
        return 'list'
    elif pa.types.is_large_list(arrow_type):
        return 'list'
    elif pa.types.is_fixed_size_list(arrow_type):
        return 'list'
    elif pa.types.is_map(arrow_type):
        return 'map'
    elif pa.types.is_struct(arrow_type):
        return 'struct'
    elif pa.types.is_union(arrow_type):
        return 'union'
    elif pa.types.is_dictionary(arrow_type):
        return 'dictionary'
    elif hasattr(pa.types, 'is_uuid') and pa.types.is_uuid(arrow_type):
        return 'uuid'
    else:
        # Get the string representation and extract the base type name
        type_str = str(arrow_type)
        # Remove any parameters
        base_type = type_str.split('(')[0].split('[')[0].strip()
        return base_type


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

    # Special handling for UUID types not in our predefined list
    if hasattr(pa.types, 'is_uuid') and pa.types.is_uuid(arrow_type):
        # Create a nil UUID (all zeros) and convert to bytes for PyArrow
        return pa.scalar(uuid_to_bytes(default_uuid()), arrow_type)

    if pa.types.is_fixed_size_binary(arrow_type):
        dft = b"0" * arrow_type.byte_width
        return pa.scalar(dft, arrow_type)

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
