import datetime as dt
import decimal as dec
import enum
import functools
import uuid
import zoneinfo
from datetime import timezone
from typing import Dict, List, Set, Tuple

from .py_utils import (
    parse_decimal_metadata,
    parse_time_metadata,
    parse_timestamp_metadata
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
    "refine_decimal_type",
    "refine_time_type",
    "refine_timestamp_type",
    "refine_nested_type",
    "dump_arrow_field_metadata",
    "format_arrow_type",
    "get_base_type_name",
    "parse_arrow_type",
    "parse_arrow_field",
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


def refine_arrow_field(field: pa.Field, metadata: dict) -> pa.Field:
    """
    Refine an arrow field by adding or updating its metadata.

    Args:
        field: The PyArrow field to refine
        metadata: A dictionary of string key-value pairs to add as metadata

    Returns:
        A new PyArrow field with the updated metadata

    Note:
        This will merge any existing metadata with the provided metadata.
        If there are conflicting keys, the new metadata values will override existing ones.
    """
    # Convert metadata values to bytes as required by PyArrow
    metadata_bytes = {k: v.encode('utf-8') if isinstance(v, str) else v
                      for k, v in metadata.items()}

    # Get existing field metadata if any
    existing_metadata = field.metadata or {}

    # Merge with new metadata
    updated_metadata = {**existing_metadata, **metadata_bytes}

    # Create a new field with the updated metadata
    return pa.field(
        field.name,
        field.type,
        nullable=field.nullable,
        metadata=updated_metadata
    )


def refine_decimal_type(field: pa.Field, metadata: dict) -> pa.Field:
    """
    Refine a decimal field using precision and scale from metadata.

    Args:
        field: The PyArrow field to refine (should contain a decimal type)
        metadata: Dictionary containing 'precision' and 'scale' keys

    Returns:
        A new field with the decimal type refined according to metadata if different,
        otherwise returns the field with its metadata updated

    Note:
        - If the input field's type is not a decimal type, only the metadata will be updated
        - If metadata doesn't contain precision/scale info, default values will be used
        - The metadata is added to the resulting field
        - No new type will be created if precision and scale haven't changed
    """
    arrow_type = field.type

    # If not a decimal type, just add metadata to the field and return
    if not pa.types.is_decimal(arrow_type):
        return refine_arrow_field(field, metadata)

    # Parse precision and scale from metadata
    precision, scale = parse_decimal_metadata(metadata)

    # Get current precision and scale
    current_precision = arrow_type.precision
    current_scale = arrow_type.scale

    # Check if precision or scale actually changed
    if precision != current_precision or scale != current_scale:
        # Only create a new type if precision or scale changed
        new_decimal_type = pa.decimal128(precision, scale)

        # Create a new field with the updated type and metadata
        return pa.field(
            field.name,
            new_decimal_type,
            nullable=field.nullable,
            metadata=metadata
        )

    # If no change in precision/scale, just add metadata to the field
    return refine_arrow_field(field, metadata)


def refine_time_type(field: pa.Field, metadata: dict) -> pa.Field:
    """
    Refine a time field using unit from metadata.

    Args:
        field: The PyArrow field to refine (should contain a time type)
        metadata: Dictionary containing 'unit' key

    Returns:
        A new field with the time type refined according to metadata if different,
        otherwise returns the field with its metadata updated

    Note:
        - If the input field's type is not a time type, only the metadata will be updated
        - If metadata doesn't contain unit info, default value will be used
        - The metadata is added to the resulting field
        - No new type will be created if the unit hasn't changed
    """
    arrow_type = field.type

    # If not a time type, just add metadata to the field and return
    if not (pa.types.is_time32(arrow_type) or pa.types.is_time64(arrow_type)):
        return refine_arrow_field(field, metadata)

    # Parse unit from metadata
    unit = parse_time_metadata(metadata)

    # Get current unit
    current_unit = arrow_type.unit

    # Check if unit changed
    if unit != current_unit:
        # Determine the time type based on the unit
        # time32 supports 's' and 'ms'
        # time64 supports 'us' and 'ns'
        if unit in ['s', 'ms']:
            new_time_type = pa.time32(unit)
        else:  # unit in ['us', 'ns']
            new_time_type = pa.time64(unit)

        # Create a new field with the updated type and metadata
        return pa.field(
            field.name,
            new_time_type,
            nullable=field.nullable,
            metadata=metadata
        )

    # If no change in unit, just add metadata to the field
    return refine_arrow_field(field, metadata)


def refine_timestamp_type(field: pa.Field, metadata: dict) -> pa.Field:
    """
    Refine a timestamp field using unit and timezone from metadata.

    Args:
        field: The PyArrow field to refine (should contain a timestamp type)
        metadata: Dictionary containing 'unit' and optional 'tz' keys

    Returns:
        A new field with the timestamp type refined according to metadata if different,
        otherwise returns the field with its metadata updated

    Note:
        - If the input field's type is not a timestamp type, only the metadata will be updated
        - If metadata doesn't contain unit info, default value will be used
        - If metadata doesn't contain timezone info, no timezone is assumed (naive timestamp)
        - The metadata is added to the resulting field
        - No new type will be created if the unit and timezone haven't changed
    """
    arrow_type = field.type

    # If not a timestamp type, just add metadata to the field and return
    if not pa.types.is_timestamp(arrow_type):
        return refine_arrow_field(field, metadata)

    # Parse unit and timezone from metadata
    unit, timezone = parse_timestamp_metadata(metadata)

    # Get current unit and timezone
    current_unit = arrow_type.unit
    current_tz = arrow_type.tz

    # Check if unit or timezone changed
    if unit != current_unit or timezone != current_tz:
        # Create a new timestamp type with the specified unit and timezone
        new_timestamp_type = pa.timestamp(unit, tz=timezone)

        # Create a new field with the updated type and metadata
        return pa.field(
            field.name,
            new_timestamp_type,
            nullable=field.nullable,
            metadata=metadata
        )

    # If no change in unit or timezone, just add metadata to the field
    return refine_arrow_field(field, metadata)


def refine_nested_type(field: pa.Field, metadata: dict, field_metadata_map: dict = None) -> pa.Field:
    """
    Recursively refine a nested type (struct, list, map) and its children.

    Args:
        field: The PyArrow field to refine (may contain a nested type)
        metadata: Dictionary containing metadata for the field itself
        field_metadata_map: Dictionary mapping field paths to their metadata dictionaries.
                           Keys are field paths in the format "field.subfield.subsubfield"

    Returns:
        A new field with all nested types refined according to metadata if any changes,
        otherwise returns the field with its metadata updated

    Note:
        - Handles struct, list, fixed-size list, large list, and map types
        - If a type is not nested, its metadata will still be updated
        - field_metadata_map allows specifying metadata for nested fields
        - Specific type refinements (decimal, time, timestamp) will be applied to respective types
    """
    arrow_type = field.type
    field_path = field.name
    field_metadata_map = field_metadata_map or {}

    # Apply basic metadata to the current field
    refined_field = refine_arrow_field(field, metadata)

    # Handle struct types (fields within a struct)
    if pa.types.is_struct(arrow_type):
        struct_fields = []

        for child_field in arrow_type:
            child_path = f"{field_path}.{child_field.name}"
            child_metadata = field_metadata_map.get(child_path, {})

            # Recursively refine the child field
            refined_child = refine_nested_type(child_field, child_metadata, field_metadata_map)
            struct_fields.append(refined_child)

        # Create a new struct type with the refined fields
        new_struct_type = pa.struct(struct_fields)

        # Create a new field with the updated type and metadata
        refined_field = pa.field(
            field.name,
            new_struct_type,
            nullable=field.nullable,
            metadata=refined_field.metadata
        )

    # Handle list-like types (regular list, large list, fixed-size list)
    elif (pa.types.is_list(arrow_type) or
          pa.types.is_large_list(arrow_type) or
          pa.types.is_fixed_size_list(arrow_type)):

        value_field = arrow_type.value_field
        value_path = f"{field_path}.items"
        value_metadata = field_metadata_map.get(value_path, {})

        # Recursively refine the value field
        refined_value = refine_nested_type(value_field, value_metadata, field_metadata_map)

        # Create an appropriate list type based on the original type
        if pa.types.is_fixed_size_list(arrow_type):
            list_size = arrow_type.list_size
            new_list_type = pa.list_(refined_value.type, list_size)
        elif pa.types.is_large_list(arrow_type):
            new_list_type = pa.large_list(refined_value)
        else:
            new_list_type = pa.list_(refined_value)

        # Create a new field with the updated type and metadata
        refined_field = pa.field(
            field.name,
            new_list_type,
            nullable=field.nullable,
            metadata=refined_field.metadata
        )

    # Handle map types
    elif pa.types.is_map(arrow_type):
        key_field = arrow_type.key_field
        item_field = arrow_type.item_field

        key_path = f"{field_path}.keys"
        item_path = f"{field_path}.items"

        key_metadata = field_metadata_map.get(key_path, {})
        item_metadata = field_metadata_map.get(item_path, {})

        # Recursively refine key and item fields
        refined_key = refine_nested_type(key_field, key_metadata, field_metadata_map)
        refined_item = refine_nested_type(item_field, item_metadata, field_metadata_map)

        # Create a new map type with the refined key and item fields
        new_map_type = pa.map_(refined_key, refined_item)

        # Create a new field with the updated type and metadata
        refined_field = pa.field(
            field.name,
            new_map_type,
            nullable=field.nullable,
            metadata=refined_field.metadata
        )

    # Apply specific type refinements if applicable
    if pa.types.is_decimal(refined_field.type):
        refined_field = refine_decimal_type(refined_field, metadata)
    elif pa.types.is_time32(refined_field.type) or pa.types.is_time64(refined_field.type):
        refined_field = refine_time_type(refined_field, metadata)
    elif pa.types.is_timestamp(refined_field.type):
        refined_field = refine_timestamp_type(refined_field, metadata)

    return refined_field


def dump_arrow_field_metadata(field: pa.Field, recursive: bool = True, prefix: str = "", decode_values: bool = True, include_type: bool = True, exclude_keys: list = None) -> dict:
    """
    Extract and format metadata from an Arrow field, optionally recursing through nested types.

    Args:
        field: The PyArrow field to extract metadata from
        recursive: Whether to recursively extract metadata from nested fields
        prefix: String prefix to add to each field path (useful for nested calls)
        decode_values: Whether to decode byte values to strings when possible
        include_type: Whether to include the Arrow type information in the metadata
        exclude_keys: Optional list of keys to exclude from the metadata

    Returns:
        A dictionary mapping field paths to their metadata dictionaries

    Example:
        >>> schema = pa.schema([
        ...     pa.field('id', pa.int64()),
        ...     pa.field('details', pa.struct([
        ...         pa.field('name', pa.string(), metadata={b'description': b'Full name'}),
        ...         pa.field('value', pa.decimal128(10, 2), metadata={b'precision': b'10', b'scale': b'2'})
        ...     ]), metadata={b'description': b'Additional details'})
        ... ])
        >>> metadata = dump_arrow_field_metadata(schema.field('details'))
        >>> print(metadata)
        {
            'details': {'description': 'Additional details', 'type': 'struct<name: string, value: decimal(10,2)>'},
            'details.name': {'description': 'Full name', 'type': 'string'},
            'details.value': {'precision': '10', 'scale': '2', 'type': 'decimal(10,2)'}
        }
        >>> # Exclude 'type' key from metadata
        >>> metadata = dump_arrow_field_metadata(schema.field('details'), exclude_keys=['type'])
        >>> print(metadata)
        {
            'details': {'description': 'Additional details'},
            'details.name': {'description': 'Full name'},
            'details.value': {'precision': '10', 'scale': '2'}
        }
    """
    result = {}
    field_path = prefix + field.name if prefix else field.name
    arrow_type = field.type
    exclude_keys = exclude_keys or []

    # Initialize metadata dict for the current field
    metadata_dict = {}

    # Extract existing metadata from the current field
    if field.metadata:
        # Convert metadata from bytes to strings if requested
        if decode_values:
            metadata_dict = {
                k.decode('utf-8') if isinstance(k, bytes) else k:
                v.decode('utf-8') if isinstance(v, bytes) else v
                for k, v in field.metadata.items()
                if (k.decode('utf-8') if isinstance(k, bytes) else k) not in exclude_keys
            }
        else:
            metadata_dict = {
                k: v for k, v in field.metadata.items()
                if (k.decode('utf-8') if isinstance(k, bytes) else k) not in exclude_keys
            }

    # Add type information if requested and not excluded
    if include_type:
        # Add formatted type string to metadata if not excluded
        if 'type' not in exclude_keys:
            metadata_dict['type'] = format_arrow_type(arrow_type)
        # Add base type name (without parameters) if not excluded
        if 'base_type' not in exclude_keys:
            metadata_dict['base_type'] = get_base_type_name(arrow_type)
        # Add nullable information if not excluded
        if 'nullable' not in exclude_keys:
            metadata_dict['nullable'] = str(field.nullable).lower()

    # Only add to result if there's metadata or type info
    if metadata_dict:
        result[field_path] = metadata_dict

    # If not recursive, just return the current field's metadata
    if not recursive:
        return result

    # Handle struct types (fields within a struct)
    if pa.types.is_struct(arrow_type):
        for child_field in arrow_type:
            child_path_prefix = field_path + "."
            child_result = dump_arrow_field_metadata(
                child_field, recursive=recursive,
                prefix=child_path_prefix, decode_values=decode_values,
                include_type=include_type, exclude_keys=exclude_keys
            )
            result.update(child_result)

    # Handle list-like types
    elif (pa.types.is_list(arrow_type) or
          pa.types.is_large_list(arrow_type) or
          pa.types.is_fixed_size_list(arrow_type)):

        value_field = arrow_type.value_field
        item_path_prefix = field_path + ".items."

        # Recursively extract metadata from the list item type
        item_result = dump_arrow_field_metadata(
            value_field, recursive=recursive,
            prefix=field_path + ".items", decode_values=decode_values,
            include_type=include_type, exclude_keys=exclude_keys
        )
        result.update(item_result)

    # Handle map types
    elif pa.types.is_map(arrow_type):
        key_field = arrow_type.key_field
        item_field = arrow_type.item_field

        # Extract metadata from key and item types
        key_result = dump_arrow_field_metadata(
            key_field, recursive=recursive,
            prefix=field_path + ".keys", decode_values=decode_values,
            include_type=include_type, exclude_keys=exclude_keys
        )
        result.update(key_result)

        item_result = dump_arrow_field_metadata(
            item_field, recursive=recursive,
            prefix=field_path + ".items", decode_values=decode_values,
            include_type=include_type, exclude_keys=exclude_keys
        )
        result.update(item_result)

    return result


def parse_arrow_type(type_str: str) -> pa.DataType:
    """
    Parse a string representation of an Arrow type and create the actual Arrow type.

    Args:
        type_str: String representation of an Arrow type (e.g., 'decimal(10,2)', 'timestamp[ms,UTC]')

    Returns:
        A PyArrow DataType object

    Raises:
        ValueError: If the type string cannot be parsed

    Examples:
        >>> parse_arrow_type('int32')
        DataType(int32)
        >>> parse_arrow_type('decimal(10,2)')
        DataType(decimal(10, 2))
        >>> parse_arrow_type('timestamp[ms,UTC]')
        DataType(timestamp[ms, tz=UTC])
        >>> parse_arrow_type('list<string>')
        DataType(list<item: string>)
    """
    # Strip any whitespace
    type_str = type_str.strip()

    # Handle basic types
    if type_str == 'bool':
        return pa.bool_()
    elif type_str == 'int8':
        return pa.int8()
    elif type_str == 'int16':
        return pa.int16()
    elif type_str == 'int32':
        return pa.int32()
    elif type_str == 'int64':
        return pa.int64()
    elif type_str == 'uint8':
        return pa.uint8()
    elif type_str == 'uint16':
        return pa.uint16()
    elif type_str == 'uint32':
        return pa.uint32()
    elif type_str == 'uint64':
        return pa.uint64()
    elif type_str == 'float16':
        return pa.float16()
    elif type_str == 'float32':
        return pa.float32()
    elif type_str == 'float64':
        return pa.float64()
    elif type_str == 'string':
        return pa.string()
    elif type_str == 'binary':
        return pa.binary()
    elif type_str == 'large_string':
        return pa.large_string()
    elif type_str == 'large_binary':
        return pa.large_binary()
    elif type_str == 'null':
        return pa.null()

    # Handle decimal types: decimal(precision,scale)
    if type_str.startswith('decimal('):
        # Extract precision and scale from decimal(p,s)
        params = type_str[len('decimal('):-1]  # Remove 'decimal(' prefix and ')' suffix
        try:
            precision, scale = map(int, params.split(','))
            return pa.decimal128(precision, scale)
        except ValueError:
            raise ValueError(f"Invalid decimal parameters: {params}")

    # Handle timestamp types: timestamp[unit] or timestamp[unit,tz]
    if type_str.startswith('timestamp['):
        # Extract unit and optional timezone
        params = type_str[len('timestamp['):-1]  # Remove 'timestamp[' prefix and ']' suffix
        parts = params.split(',')
        unit = parts[0]
        tz = parts[1] if len(parts) > 1 else None
        return pa.timestamp(unit, tz=tz)

    # Handle time types: time32[unit] or time64[unit]
    if type_str.startswith('time32[') or type_str.startswith('time64['):
        # Extract unit
        unit = type_str[type_str.find('[')+1:type_str.find(']')]
        if type_str.startswith('time32['):
            return pa.time32(unit)
        else:  # time64
            return pa.time64(unit)

    # Handle date types
    if type_str == 'date32':
        return pa.date32()
    elif type_str == 'date64':
        return pa.date64()

    # Handle duration types: duration[unit]
    if type_str.startswith('duration['):
        unit = type_str[len('duration['):-1]
        return pa.duration(unit)

    # Handle list types: list<type>
    if type_str.startswith('list<'):
        # Extract inner type
        inner_type_str = type_str[len('list<'):-1]
        inner_type = parse_arrow_type(inner_type_str)
        return pa.list_(inner_type)

    # Handle large_list types: large_list<type>
    if type_str.startswith('large_list<'):
        # Extract inner type
        inner_type_str = type_str[len('large_list<'):-1]
        inner_type = parse_arrow_type(inner_type_str)
        return pa.large_list(inner_type)

    # Handle fixed size list types: fixed_size_list<type,size>
    if type_str.startswith('fixed_size_list<'):
        # Remove prefix and suffix
        params = type_str[len('fixed_size_list<'):-1]
        # Find the last comma (to handle nested types with commas)
        last_comma = params.rfind(',')
        if last_comma == -1:
            raise ValueError(f"Invalid fixed_size_list format: {type_str}")

        inner_type_str = params[:last_comma]
        size_str = params[last_comma+1:]

        try:
            size = int(size_str)
            inner_type = parse_arrow_type(inner_type_str)
            return pa.list_(inner_type, size)
        except ValueError:
            raise ValueError(f"Invalid fixed_size_list parameters: {params}")

    # Handle map types: map<key_type,item_type>
    if type_str.startswith('map<'):
        # Remove prefix and suffix
        params = type_str[len('map<'):-1]
        # Find the first comma (assuming key type doesn't have commas)
        first_comma = params.find(',')
        if first_comma == -1:
            raise ValueError(f"Invalid map format: {type_str}")

        key_type_str = params[:first_comma]
        item_type_str = params[first_comma+1:]

        key_type = parse_arrow_type(key_type_str)
        item_type = parse_arrow_type(item_type_str)
        return pa.map_(key_type, item_type)

    # Handle struct types: struct<name1: type1, name2: type2, ...>
    if type_str.startswith('struct<'):
        fields_str = type_str[len('struct<'):-1]
        if not fields_str:
            return pa.struct([])

        # This is a simplified parser that won't handle all edge cases
        # For a complete parser, we'd need a more sophisticated approach
        fields = []
        field_parts = []

        # Simple parser for struct fields that handles nesting
        # This won't handle all edge cases, but works for common cases
        depth = 0
        current = ""
        for char in fields_str:
            if char == ',' and depth == 0:
                field_parts.append(current.strip())
                current = ""
            else:
                if char == '<':
                    depth += 1
                elif char == '>':
                    depth -= 1
                current += char

        if current:
            field_parts.append(current.strip())

        # Parse each field
        for part in field_parts:
            if ':' not in part:
                raise ValueError(f"Invalid struct field format: {part}")

            name, type_def = part.split(':', 1)
            name = name.strip()
            type_def = type_def.strip()

            # Check if field is nullable (ends with !)
            nullable = True
            if type_def.endswith('!'):
                nullable = False
                type_def = type_def[:-1]

            field_type = parse_arrow_type(type_def)
            fields.append(pa.field(name, field_type, nullable=nullable))

        return pa.struct(fields)

    # If all else fails
    raise ValueError(f"Unsupported Arrow type format: {type_str}")


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
        if pa.types.is_int8(arrow_type): return 'int8'
        elif pa.types.is_int16(arrow_type): return 'int16'
        elif pa.types.is_int32(arrow_type): return 'int32'
        elif pa.types.is_int64(arrow_type): return 'int64'
        elif pa.types.is_uint8(arrow_type): return 'uint8'
        elif pa.types.is_uint16(arrow_type): return 'uint16'
        elif pa.types.is_uint32(arrow_type): return 'uint32'
        elif pa.types.is_uint64(arrow_type): return 'uint64'
        return 'integer'
    elif pa.types.is_floating(arrow_type):
        if pa.types.is_float16(arrow_type): return 'float16'
        elif pa.types.is_float32(arrow_type): return 'float32'
        elif pa.types.is_float64(arrow_type): return 'float64'
        return 'float'
    elif pa.types.is_decimal(arrow_type):
        return 'decimal'
    elif pa.types.is_string(arrow_type):
        return 'string'
    elif pa.types.is_binary(arrow_type):
        return 'binary'
    elif pa.types.is_large_string(arrow_type):
        return 'large_string'
    elif pa.types.is_large_binary(arrow_type):
        return 'large_binary'
    elif pa.types.is_fixed_size_binary(arrow_type):
        return 'fixed_size_binary'
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
        return 'large_list'
    elif pa.types.is_fixed_size_list(arrow_type):
        return 'fixed_size_list'
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


def parse_arrow_field(field_metadata: dict) -> pa.Field:
    """
    Parse a field from its metadata representation (as returned by dump_arrow_field_metadata).

    Args:
        field_metadata: Dictionary containing field metadata including 'type' key

    Returns:
        A PyArrow Field object

    Raises:
        ValueError: If required metadata is missing or invalid

    Examples:
        >>> metadata = {'type': 'decimal(10,2)', 'nullable': 'true', 'description': 'Amount field'}
        >>> field = parse_arrow_field(metadata)
        >>> field
        pyarrow.field('field_name', decimal(10, 2), nullable=True, metadata={b'description': b'Amount field'})
    """
    if 'type' not in field_metadata:
        raise ValueError("Field metadata must include 'type' key")

    # Parse the type
    arrow_type = parse_arrow_type(field_metadata['type'])

    # Get nullable status (default to True if not specified)
    nullable_str = field_metadata.get('nullable', 'true').lower()
    nullable = nullable_str not in ('false', 'f', '0', 'no')

    # Extract name (default to 'field')
    name = field_metadata.get('name', 'field')

    # Create metadata dictionary excluding special keys
    special_keys = {'type', 'base_type', 'nullable', 'name'}
    metadata = {
        k.encode('utf-8') if isinstance(k, str) else k:
        v.encode('utf-8') if isinstance(v, str) else v
        for k, v in field_metadata.items()
        if k not in special_keys
    }

    # Create and return the field
    return pa.field(name, arrow_type, nullable=nullable, metadata=metadata)


def format_arrow_type(arrow_type: pa.DataType) -> str:
    """
    Format an Arrow data type as a readable string.

    Args:
        arrow_type: PyArrow data type to format

    Returns:
        A formatted string representation of the type
    """
    # Handle special cases for common types
    if pa.types.is_decimal(arrow_type):
        return f"decimal({arrow_type.precision},{arrow_type.scale})"
    elif pa.types.is_timestamp(arrow_type):
        tz_str = f",{arrow_type.tz}" if arrow_type.tz else ""
        return f"timestamp[{arrow_type.unit}{tz_str}]"
    elif pa.types.is_time32(arrow_type) or pa.types.is_time64(arrow_type):
        return f"time[{arrow_type.unit}]"
    elif pa.types.is_list(arrow_type):
        return f"list<{format_arrow_type(arrow_type.value_type)}>"
    elif pa.types.is_large_list(arrow_type):
        return f"large_list<{format_arrow_type(arrow_type.value_type)}>"
    elif pa.types.is_fixed_size_list(arrow_type):
        return f"fixed_size_list<{format_arrow_type(arrow_type.value_type)},{arrow_type.list_size}>"
    elif pa.types.is_map(arrow_type):
        key_type = format_arrow_type(arrow_type.key_type)
        item_type = format_arrow_type(arrow_type.item_type)
        return f"map<{key_type},{item_type}>"
    elif pa.types.is_struct(arrow_type):
        field_strs = []
        for field in arrow_type:
            field_type = format_arrow_type(field.type)
            nullable_str = "" if field.nullable else "!"
            field_strs.append(f"{field.name}: {field_type}{nullable_str}")
        return f"struct<{', '.join(field_strs)}>"
    else:
        # For basic types, use the standard string representation
        return str(arrow_type)


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
