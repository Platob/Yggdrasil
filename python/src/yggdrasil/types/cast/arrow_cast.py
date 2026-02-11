"""Arrow casting helpers for arrays, tables, and schemas."""

import dataclasses
import enum
import logging
from dataclasses import is_dataclass
from typing import Optional, Union, List, Tuple, Any

import pyarrow as pa
import pyarrow.compute as pc

from .cast_options import CastOptions
from .registry import register_converter
from ..python_arrow import is_arrow_type_list_like, is_arrow_type_string_like, is_arrow_type_binary_like
from ..python_defaults import default_arrow_scalar, default_arrow_array
from ...dataclasses.dataclass import get_dataclass_arrow_field

__all__ = [
    "cast_arrow_array",
    "cast_arrow_tabular",
    "cast_arrow_record_batch_reader",
    "default_arrow_array",
    "to_spark_arrow_type",
    "to_polars_arrow_type",
    "arrow_field_to_schema",
    "is_arrow_type_binary_like",
    "is_arrow_type_string_like",
    "is_arrow_type_list_like",
    "record_batch_to_table",
    "arrow_schema_to_field",
]

logger = logging.getLogger(__name__)


def cast_to_struct_array(
    array: Union[pa.Array, pa.StructArray, pa.MapArray],
    options: Optional[CastOptions] = None,
) -> pa.StructArray:
    """Cast arrays to a struct Arrow array."""
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    if isinstance(array, pa.ChunkedArray):
        casted_chunks = [
            cast_to_struct_array(chunk, options)
            for chunk in array.chunks
        ]
        return pa.chunked_array(casted_chunks, type=options.target_arrow_field.type)

    source_type = options.source_arrow_field.type

    if pa.types.is_struct(source_type):
        return arrow_struct_to_struct_array(array, options)
    elif pa.types.is_map(source_type):
        return arrow_map_to_struct_array(array, options)
    else:
        logger.error(
            "Unsupported struct cast from %s to %s",
            options.source_field,
            options.target_field,
        )
        raise ValueError(f"Cannot cast {options.source_field} to {options.target_field}")


@register_converter(pa.MapArray, pa.StructArray)
def arrow_map_to_struct_array(
    array: pa.MapArray,
    options: Optional[CastOptions] = None,
) -> pa.StructArray:
    """Cast arrays to a struct Arrow array."""
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    source_field = options.source_field
    target_type: pa.StructType = options.target_field.type

    mask = array.is_null() if source_field.nullable and options.target_field.nullable else None
    children: List[pa.Array] = []

    map_type: pa.MapType = array.type

    for child_target_field in target_type:
        values = pc.map_lookup(array, child_target_field.name, "first")

        casted = cast_arrow_array(
            values,
            options.copy(
                source_arrow_field=map_type.item_field,
                target_arrow_field=child_target_field
            )
        )

        children.append(casted)

    return pa.StructArray.from_arrays(
        children,
        fields=list(target_type),
        mask=mask,
        memory_pool=options.arrow_memory_pool
    )


@register_converter(pa.StructArray, pa.StructArray)
def arrow_struct_to_struct_array(
    array: pa.StructArray,
    options: Optional[CastOptions] = None,
) -> pa.StructArray:
    """Cast arrays to a struct Arrow array."""
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    target_type: pa.StructType = options.target_field.type

    mask = array.is_null() if options.source_field.nullable and options.target_field.nullable else None
    children: List[pa.Array] = []

    name_to_index = {
        field.name: idx for idx, field in enumerate(array.type)
    }
    folded_to_index = {
        field.name.casefold(): idx for idx, field in enumerate(array.type)
    }

    for i, child_target_field in enumerate(target_type):
        child_target_field: pa.Field = child_target_field

        if child_target_field.name in name_to_index:
            child_idx = name_to_index[child_target_field.name]
            child_arr = array.field(child_idx)
            child_source_field = array.type[child_idx]
        elif (
            not options.strict_match_names
            and child_target_field.name.casefold() in folded_to_index
        ):
            child_idx = folded_to_index[child_target_field.name.casefold()]
            child_arr = array.field(child_idx)
            child_source_field = array.type[child_idx]
        elif not options.strict_match_names and i < array.type.num_fields:
            # Positional fallback
            child_idx = i
            child_arr = array.field(child_idx)
            child_source_field = array.type[child_idx]
        elif options.add_missing_columns:
            # Field missing -> create default-valued array
            child_arr = default_arrow_array(
                dtype=child_target_field.type,
                nullable=child_target_field.nullable,
                size=len(array),
                memory_pool=options.arrow_memory_pool
            )
            child_source_field = child_target_field
        else:
            raise pa.ArrowInvalid(
                "Missing struct field %s while casting; available fields: %s" % (
                    child_target_field.name, list(name_to_index.keys())
                )
            )

        children.append(
            cast_arrow_array(
                child_arr,
                options.copy(
                    source_arrow_field=child_source_field,
                    target_arrow_field=child_target_field
                )
            )
        )

    return pa.StructArray.from_arrays(
        children,
        fields=list(target_type),
        mask=mask,
        memory_pool=options.arrow_memory_pool
    )


def cast_to_list_arrow_array(
    array: Union[pa.Array, pa.ListArray, pa.LargeListArray],
    options: Optional[CastOptions] = None,
) -> Union[pa.ListArray, pa.LargeListArray]:
    """Cast arrays to a list or large list Arrow array."""
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    source_field = options.source_arrow_field
    target_field = options.target_arrow_field

    if isinstance(array, pa.ChunkedArray):
        casted_chunks = [
            cast_to_list_arrow_array(chunk, options)
            for chunk in array.chunks
        ]
        return pa.chunked_array(casted_chunks, type=target_field.type)

    target_type: Union[pa.ListType, pa.FixedSizeListType] = target_field.type

    if is_arrow_type_string_like(source_field.type):
        import polars

        try:
            array = (
                polars.from_arrow(array)
                .replace("", None)
                .str.json_decode()
                .to_arrow()
            )
        except Exception as e:
            raise pa.ArrowInvalid(
                "Failed to parse JSON strings in list cast from %s to %s: %s" % (
                    source_field,
                    target_field,
                    e
                )
            )

        if source_field:
            source_field = pa.field(
                name=source_field.name,
                type=array.type,
                nullable=source_field.nullable,
                metadata=source_field.metadata
            )

    mask = array.is_null() if source_field.nullable and target_field.nullable else None

    if is_arrow_type_list_like(source_field.type):
        list_source_field = array.type.value_field

        offsets = array.offsets
        values = cast_arrow_array(
            array.values,
            options.copy(
                source_arrow_field=list_source_field,
                target_arrow_field=target_type.value_field,
            )
        )
    else:
        raise pa.ArrowInvalid("Unsupported list cast from %s to %s" % (source_field, target_field))

    if pa.types.is_list(target_type):
        return pa.ListArray.from_arrays(
            offsets,
            values,
            type=target_type,
            mask=mask
        )
    if pa.types.is_large_list(target_type):
        return pa.LargeListArray.from_arrays(
            offsets,
            values,
            type=target_type,
            mask=mask
        )
    elif pa.types.is_fixed_size_list(target_type):
        return pa.FixedSizeListArray.from_arrays(
            values,
            list_size=target_type.list_size,
            type=target_type,
            mask=mask
        )
    else:
        raise pa.ArrowInvalid(f"Cannot build arrow array for target list type {target_type}")


def cast_to_map_array(
    array: Union[pa.Array, pa.MapArray, pa.StructArray],
    options: Optional[CastOptions] = None,
) -> pa.MapArray:
    """Cast arrays to a map Arrow array."""
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    source_field = options.source_arrow_field
    target_field = options.target_arrow_field

    if isinstance(array, pa.ChunkedArray):
        casted_chunks = [
            cast_to_map_array(chunk, options)
            for chunk in array.chunks
        ]
        return pa.chunked_array(casted_chunks, type=target_field.type)

    # Case 1: map -> map
    if pa.types.is_map(source_field.type):
        return arrow_map_to_map_array(array, options)
    elif pa.types.is_struct(source_field.type):
        return arrow_struct_to_map_array(array, options)
    else:
        raise pa.ArrowInvalid("Unsupported map cast from %s to %s" % (source_field, target_field))


@register_converter(pa.MapArray, pa.MapArray)
def arrow_map_to_map_array(
    array: pa.MapArray,
    options: Optional[CastOptions] = None,
) -> pa.MapArray:
    """Cast arrays to a map Arrow array."""
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    source_field = options.source_arrow_field
    target_field = options.target_arrow_field

    target_type: pa.MapType = target_field.type
    mask = array.is_null() if source_field.nullable and target_field.nullable else None

    keys = cast_arrow_array(
        array.keys,
        options.copy(
            source_arrow_field=array.type.key_field,
            target_arrow_field=target_type.key_field,
        )
    )
    items = cast_arrow_array(
        array.items,
        options.copy(
            source_arrow_field=array.type.item_field,
            target_arrow_field=target_type.item_field,
        )
    )

    return pa.MapArray.from_arrays(
        array.offsets,
        keys,
        items,
        mask=mask,
        type=target_type,
        pool=options.arrow_memory_pool
    )


@register_converter(pa.StructArray, pa.MapArray)
def arrow_struct_to_map_array(
    array: pa.StructArray,
    options: Optional[CastOptions] = None,
) -> pa.MapArray:
    """Cast arrays to a map Arrow array."""
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    source_field = options.source_arrow_field
    target_field = options.target_arrow_field
    target_type: pa.MapType = target_field.type

    num_rows = len(array)
    offsets = [0]
    keys: List[str] = []
    items: List[object] = []
    mask = array.is_null() if array.null_count else None

    # Pre-cast all children values
    casted_children = [
        cast_arrow_array(
            array.field(i),
            options.copy(
                source_arrow_field=array.type[i],
                target_arrow_field=target_type.item_field,
            )
        )
        for i in range(array.type.num_fields)
    ]

    for row_idx in range(num_rows):
        if mask is not None and mask[row_idx].as_py():
            # Null row -> no entries added for this row
            offsets.append(offsets[-1])
            continue

        for field_idx, field in enumerate(array.type):
            field_name = field.name
            keys.append(field_name)

            child_value = casted_children[field_idx][row_idx]
            items.append(child_value)

        offsets.append(len(keys))

    map_type = pa.map_(pa.string(), target_type.item_type, keys_sorted=False)

    return pa.MapArray.from_arrays(
        offsets,
        pa.array(keys, type=pa.string()),
        pa.array(items, type=target_type.item_type),
        mask=mask,
        type=map_type,
        pool=options.arrow_memory_pool
    )


def cast_primitive_array(
    array: Union[pa.ChunkedArray, pa.Array],
    options: CastOptions | None = None,
) -> pa.Array:
    """Cast simple scalar arrays via pyarrow.compute.cast."""
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    source_field = options.source_arrow_field
    target_field = options.target_arrow_field

    if is_arrow_type_string_like(source_field.type) and pa.types.is_timestamp(target_field.type):
        return arrow_strptime(array, options)
    else:
        casted = pc.cast(
            array,
            target_type=target_field.type,
            safe=options.safe,
            memory_pool=options.arrow_memory_pool,
        )

        return check_arrow_array_nullability(casted, options)


def check_arrow_array_nullability(
    array: Union[pa.Array, pa.ChunkedArray],
    options: Optional[CastOptions] = None,
) -> Union[pa.Array, pa.ChunkedArray]:
    """
    For non-nullable targets, replace nulls with default Python values.

    If the *source* is already non-nullable and has no nulls, we leave it alone.
    """
    options = CastOptions.check_arg(options)

    if not options.need_nullability_check(source_obj=array):
        return array

    if isinstance(array, pa.ChunkedArray):
        if array.null_count == 0:
            return array

        filled_chunks = [
            check_arrow_array_nullability(chunk, options)
            for chunk in array.chunks
        ]

        return pa.chunked_array(filled_chunks, type=array.type)

    if array.null_count == 0:
        # Source already guaranteed non-nullable and contains no nulls.
        return array
    else:
        default_arr = default_arrow_array(
            options.target_field.type,
            nullable=options.target_field.nullable,
            size=len(array)
        )

        return pc.if_else(pc.is_null(array), default_arr, array)


@register_converter(Any, pa.RecordBatch)
def any_to_arrow_scalar(
    scalar: Any,
    options: Optional[CastOptions] = None,
) -> pa.Scalar:
    """Convert a Python value to an Arrow scalar.

    Args:
        scalar: Input value.
        options: Optional cast options.

    Returns:
        Arrow scalar.
    """
    if isinstance(scalar, pa.Scalar):
        return cast_arrow_scalar(scalar, options)

    options = CastOptions.check_arg(options)
    target_field = options.target_field

    if isinstance(scalar, enum.Enum):
        scalar = scalar.value

    if is_dataclass(scalar):
        if not target_field:
            target_field = get_dataclass_arrow_field(scalar)
            options = options.copy(target_arrow_field=target_field)

        scalar = dataclasses.asdict(scalar)

    if target_field is None:
        if is_dataclass(scalar):
            scalar = pa.scalar(dataclasses.asdict(scalar), type=get_dataclass_arrow_field(scalar).type)
        else:
            scalar = pa.scalar(scalar)

        return scalar

    if scalar is None and not target_field.nullable:
        return default_arrow_scalar(target_field)

    try:
        scalar = pa.scalar(scalar, type=target_field.type)
    except:
        scalar = pa.scalar(scalar)

    return cast_arrow_scalar(scalar, options)


@register_converter(pa.Scalar, pa.Scalar)
def cast_arrow_scalar(
    scalar: pa.Scalar,
    options: Optional[CastOptions] = None,
) -> pa.Scalar:
    """Cast an Arrow scalar to the target Arrow field.

    Args:
        scalar: Arrow scalar to cast.
        options: Optional cast options.

    Returns:
        Cast Arrow scalar.
    """
    options = CastOptions.check_arg(options)
    target_field = options.target_field

    if target_field is None:
        return scalar

    arr = pa.array([scalar])
    casted = cast_arrow_array(arr, options)

    return casted[0]


@register_converter(pa.Array, pa.Array)
@register_converter(pa.ChunkedArray, pa.ChunkedArray)
def cast_arrow_array(
    array: Union[pa.ChunkedArray, pa.Array],
    options: Optional[CastOptions] = None,
) -> Union[pa.ChunkedArray, pa.Array]:
    """
    Cast an Arrow array or chunked array to the type described in options.target_field.

    This handles:
    - Scalars via pyarrow.compute.cast
    - Structs (with name/position-based matching, default values, map-to-struct)
    - Lists and large lists (recursive value casting)
    - Maps (map-to-map and struct-to-map conversions)

    Nullability is enforced using `default_from_arrow_hint` for non-nullable targets.
    """
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    source_type = options.source_field.type
    target_type = options.target_field.type

    if pa.types.is_null(source_type):
        return check_arrow_array_nullability(
            pc.cast(
                array, target_type,
                safe=False, memory_pool=options.arrow_memory_pool
            ),
            options=options
        )

    if pa.types.is_nested(target_type):
        if pa.types.is_struct(target_type):
            return cast_to_struct_array(array, options)
        elif is_arrow_type_list_like(target_type):
            return cast_to_list_arrow_array(array, options)
        elif pa.types.is_map(target_type):
            return cast_to_map_array(array, options)
        logger.error(
            "Unsupported nested target type %s for source %s",
            target_type,
            options.source_field,
        )
        raise ValueError(f"Unsupported nested target type {target_type}")
    else:
        return cast_primitive_array(array, options)


def arrow_strptime(
    arr: Union[pa.Array, pa.ChunkedArray, pa.StringArray],
    options: Optional[CastOptions] = None,
) -> Union[pa.TimestampArray, pa.ChunkedArray]:
    """
    Parse a string Arrow array into timestamps using a format string.

    Uses pyarrow.compute.strptime under the hood.

    Requires options.target_field to be a timestamp field, and
    options.format to be set to the desired strptime format string.
    """
    options = CastOptions.check_arg(options)
    target_field = options.target_field

    if not target_field:
        return arr

    if not pa.types.is_timestamp(target_field.type):
        logger.error(
            "arrow_strptime requires timestamp target; got %s",
            target_field,
        )
        raise ValueError("arrow_strptime requires target_field to be a timestamp type")

    if isinstance(arr, pa.ChunkedArray):
        casted_chunks = [
            arrow_strptime(chunk, options)
            for chunk in arr.chunks
        ]
        return pa.chunked_array(casted_chunks, type=target_field.type)

    patterns = options.datetime_patterns

    if not patterns:
        casted = pc.cast(
            arr,
            target_type=target_field.type,
            safe=options.safe,
            memory_pool=options.arrow_memory_pool,
        )
    else:
        last_error = None
        casted = None

        for pattern in patterns:
            try:
                casted = pc.strptime(
                    arr,
                    format=pattern,
                    unit=target_field.type.unit,
                    error_is_null=not options.safe,
                    memory_pool=options.arrow_memory_pool,
                )
                break
            except Exception as e:
                last_error = e

        if casted is None:
            logger.error(
                "Failed to parse timestamps with provided patterns %s for target %s; last error: %s",
                patterns,
                target_field,
                last_error,
            )
            raise last_error if last_error else ValueError("Failed to parse timestamps with provided patterns")

    return check_arrow_array_nullability(casted, options)

# ---------------------------------------------------------------------------
# Table / RecordBatch casting
# ---------------------------------------------------------------------------

@register_converter(pa.Table, pa.Table)
@register_converter(pa.RecordBatch, pa.RecordBatch)
def cast_arrow_tabular(
    data: Union[pa.Table, pa.RecordBatch],
    options: Optional[CastOptions] = None,
) -> Union[pa.Table, pa.RecordBatch]:
    """
    Cast a pyarrow.Table to `options.target_schema`.

    Handles:
    - Column name matching (exact, case-insensitive, positional)
    - Missing columns (optional default creation)
    - Column-wise casting via `cast_arrow_array`.
    """
    options = CastOptions.check_arg(options)
    target_arrow_schema = options.target_arrow_schema

    if target_arrow_schema is None:
        # No target schema -> return as-is
        return data

    if data.num_rows == 0:
        return data.__class__.from_arrays(
            arrays=[
                default_arrow_array(
                    dtype=field.type,
                    nullable=field.nullable,
                    size=0,
                    memory_pool=options.arrow_memory_pool,
                )
                for field in target_arrow_schema
            ],
            schema=target_arrow_schema
        )

    source_arrow_schema: pa.Schema = data.schema

    if source_arrow_schema == target_arrow_schema:
        return data

    source_name_to_index = {
        field.name: idx for idx, field in enumerate(source_arrow_schema)
    }

    if not options.strict_match_names:
        source_name_to_index.update({
            field.name.casefold(): idx for idx, field in enumerate(source_arrow_schema)
        })

    chunks = None
    if isinstance(data, pa.Table) and data.num_columns > 0:
        first_col = data.column(0)
        if isinstance(first_col, pa.ChunkedArray):
            chunks = [len(chunk) for chunk in first_col.chunks]

    casted_columns: List[Tuple[pa.Field, pa.ChunkedArray]] = []
    found_source_names = set()

    for target_field in target_arrow_schema:
        target_field: pa.Field = target_field

        source_index = source_name_to_index.get(target_field.name)

        if source_index is None:
            if not options.add_missing_columns:
                logger.error(
                    "Missing column %s while casting table; source columns: %s",
                    target_field.name,
                    list(source_arrow_schema.names),
                )
                raise pa.ArrowInvalid(f"Missing column {target_field.name!r} in source data {source_arrow_schema.names}")

            array = default_arrow_array(
                dtype=target_field.type,
                nullable=target_field.nullable,
                size=data.num_rows,
                memory_pool=options.arrow_memory_pool,
                chunks=chunks
            )
        else:
            source_field = source_arrow_schema.field(source_index)
            found_source_names.add(source_field.name)
            array = cast_arrow_array(
                data.column(source_index),
                options.copy(
                    source_arrow_field=source_field,
                    target_arrow_field=target_field,
                )
            )
        casted_columns.append((target_field, array))

    if options.allow_add_columns:
        extra_columns = [
            (src_field, data.column(idx))
            for idx, src_field in enumerate(source_arrow_schema)
            if src_field.name not in found_source_names
        ]

        if extra_columns:
            for src_field, array in extra_columns:
                casted_columns.append((src_field, array))

            target_arrow_schema = pa.schema(
                [field for field, _ in casted_columns],
                metadata=target_arrow_schema.metadata
            )

    all_arrays = [array for _, array in casted_columns]

    # Extra columns in `data` are ignored when building the new table
    return data.__class__.from_arrays(all_arrays, schema=target_arrow_schema)


@register_converter(pa.RecordBatchReader, pa.RecordBatchReader)
def cast_arrow_record_batch_reader(
    data: pa.RecordBatchReader,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """
    Wrap a RecordBatchReader and lazily cast each batch to `options.target_schema`.
    """
    options = CastOptions.check_arg(options)
    arrow_schema = options.target_arrow_schema

    if arrow_schema is None:
        # Nothing to cast, just return the original reader
        return data

    def casted_batches():
        """Yield casted batches from a RecordBatchReader.

        Yields:
            Casted RecordBatch instances.
        """
        for batch in data:
            yield cast_arrow_tabular(batch, options)

    return pa.RecordBatchReader.from_batches(arrow_schema, casted_batches())


# ---------------------------------------------------------------------------
# Type normalization helpers
# ---------------------------------------------------------------------------
def to_spark_arrow_type(
    dtype: Union[pa.DataType, pa.ListType, pa.MapType, pa.StructType]
) -> Union[pa.DataType, pa.ListType, pa.MapType, pa.StructType]:
    """
    Normalize an Arrow DataType to something Spark can handle:

    - large_string  -> string
    - large_binary  -> binary
    - large_list<T> -> list<T>
    - dictionary    -> value type
    - extension     -> storage type
    - recurse through struct/map/list fields
    """
    # Large scalar types
    if is_arrow_type_string_like(dtype):
        return pa.string()
    if is_arrow_type_binary_like(dtype):
        return pa.binary()

    # Large list -> normal list with normalized value type
    if is_arrow_type_list_like(dtype):
        return pa.list_(to_spark_arrow_type(dtype.value_type))

    # Dictionary-encoded types: Spark wants the value type, not the indices
    if pa.types.is_dictionary(dtype):
        return to_spark_arrow_type(dtype.value_type)

    # Extension types: unwrap to storage type
    if isinstance(dtype, pa.ExtensionType):
        return to_spark_arrow_type(dtype.storage_type)

    # Struct: normalize each child field
    if pa.types.is_struct(dtype):
        new_fields = [
            pa.field(
                f.name,
                to_spark_arrow_type(f.type),
                nullable=f.nullable,
                metadata=f.metadata,
            )
            for f in dtype
        ]
        return pa.struct(new_fields)

    # Map: normalize key/value types
    if pa.types.is_map(dtype):
        key_field = dtype.key_field
        item_field = dtype.item_field

        new_key = pa.field(
            key_field.name,
            to_spark_arrow_type(key_field.type),
            nullable=key_field.nullable,
            metadata=key_field.metadata,
        )
        new_item = pa.field(
            item_field.name,
            to_spark_arrow_type(item_field.type),
            nullable=item_field.nullable,
            metadata=item_field.metadata,
        )
        return pa.map_(new_key, new_item)

    # Everything else: leave as-is
    return dtype


def to_polars_arrow_type(dtype: pa.DataType) -> pa.DataType:
    """
    Normalize an Arrow DataType to something Polars can handle nicely.

    Special rule:
    - map<k,v> -> list<struct<key: K, value: V>>

    Also:
    - unwrap dictionary/extension/large_* similarly to Spark logic so we don't
      leak weird "view" types into Polars.
    """
    # First normalize "views" / large types using the Spark helper
    dtype = to_spark_arrow_type(dtype)

    # Map -> list<struct<key, value>>
    if pa.types.is_map(dtype):
        key_field = dtype.key_field
        item_field = dtype.item_field

        key_type = to_polars_arrow_type(key_field.type)
        value_type = to_polars_arrow_type(item_field.type)

        struct_type = pa.field(
            "entries",
            pa.struct(
                [
                    pa.field(
                        key_field.name,
                        key_type,
                        nullable=key_field.nullable,
                        metadata=key_field.metadata,
                    ),
                    pa.field(
                        item_field.name,
                        value_type,
                        nullable=item_field.nullable,
                        metadata=item_field.metadata,
                    ),
                ]
            ),
            nullable=True,
        )
        return pa.list_(struct_type)

    # Struct: recurse into children
    if pa.types.is_struct(dtype):
        new_fields = [
            pa.field(
                f.name,
                to_polars_arrow_type(f.type),
                nullable=f.nullable,
                metadata=f.metadata,
            )
            for f in dtype
        ]
        return pa.struct(new_fields)

    # List: recurse into element type
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return pa.list_(to_polars_arrow_type(dtype.value_type))

    return dtype


# ---------------------------------------------------------------------------
# Cross-container casting helpers
# ---------------------------------------------------------------------------
@register_converter(pa.Table, pa.RecordBatch)
def table_to_record_batch(
    data: pa.Table,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatch:
    """
    Cast a Table using `cast_arrow_table` and return a single RecordBatch.

    Handles the fact that Table columns are ChunkedArray, while
    RecordBatch expects plain Array.
    """
    casted: pa.Table = cast_arrow_tabular(data, options)

    # Empty table: build an empty batch with same schema
    if casted.num_rows == 0:
        arrays = [pa.array([], type=f.type) for f in casted.schema]
        return pa.RecordBatch.from_arrays(arrays, schema=casted.schema)

    # Merge multiple batches into one RecordBatch
    arrays = [
        chunked_array.combine_chunks()
        for chunked_array in casted.columns
    ]

    return pa.RecordBatch.from_arrays(arrays, schema=casted.schema)


@register_converter(pa.RecordBatch, pa.Table)
def record_batch_to_table(
    data: pa.RecordBatch,
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """
    Cast a RecordBatch using `cast_arrow_tabular` and wrap as a single-batch Table.
    """
    casted = cast_arrow_tabular(data, options)
    return pa.Table.from_batches(batches=[casted], schema=casted.schema)


@register_converter(pa.Table, pa.RecordBatchReader)
def table_to_record_batch_reader(
    data: pa.Table,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """
    Cast a Table and expose it as a RecordBatchReader.
    """
    casted = cast_arrow_tabular(data, options)
    return pa.RecordBatchReader.from_batches(
        casted.schema,
        casted.to_batches(),
    )


@register_converter(pa.RecordBatchReader, pa.Table)
def record_batch_reader_to_table(
    data: pa.RecordBatchReader,
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """
    Cast each batch in a RecordBatchReader and collect into a Table.
    """
    casted_reader: pa.RecordBatchReader = cast_arrow_record_batch_reader(data, options)
    return pa.Table.from_batches(batches=list(casted_reader), schema=casted_reader.schema)


@register_converter(pa.RecordBatch, pa.RecordBatchReader)
def record_batch_to_record_batch_reader(
    data: pa.RecordBatch,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """
    Cast a RecordBatch and wrap it into a single-batch RecordBatchReader.
    """
    casted = cast_arrow_tabular(data, options)
    return pa.RecordBatchReader.from_batches(schema=casted.schema, batches=[casted])


@register_converter(pa.RecordBatchReader, pa.RecordBatch)
def record_batch_reader_to_record_batch(
    data: pa.RecordBatchReader,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatch:
    """
    Cast a RecordBatchReader, collect to a Table, then to a single RecordBatch.

    Note: this will materialize all batches in memory.
    """
    table = record_batch_reader_to_table(data, options)
    return table_to_record_batch(table, options)


# ---------------------------------------------------------------------------
# Field / Schema converters
# ---------------------------------------------------------------------------
@register_converter(pa.Array, pa.Field)
@register_converter(pa.ChunkedArray, pa.Field)
def arrow_array_to_field(
    array: Union[pa.Array, pa.ChunkedArray],
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """
    Convert an Arrow array or chunked array to a Field describing its type.
    """
    options = CastOptions.check_arg(options=options)
    name = options.source_arrow_field.name if options.source_arrow_field else "root"
    metadata = options.source_arrow_field.metadata if options.source_arrow_field else None

    return pa.field(
        name,
        array.type,
        nullable=array.type == pa.null() or array.null_count > 0,
        metadata=metadata,
    )


@register_converter(pa.DataType, pa.Field)
def arrow_type_to_field(
    arrow_type: pa.DataType,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """
    Convert an Arrow type to a Field describing its type.
    """
    options = CastOptions.check_arg(options=options)
    name = options.source_arrow_field.name if options.source_arrow_field else "root"
    nullable = options.source_arrow_field.nullable if options.source_arrow_field else True
    metadata = options.source_arrow_field.metadata if options.source_arrow_field else None

    return pa.field(
        name,
        arrow_type,
        nullable=nullable,
        metadata=metadata,
    )


@register_converter(pa.Schema, pa.Field)
def arrow_schema_to_field(
    data: pa.Schema,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """Wrap an Arrow schema as a struct field.

    Args:
        data: Arrow schema.
        options: Optional cast options.

    Returns:
        Arrow field.
    """
    dtype = pa.struct(list(data))
    md = dict(data.metadata or {})
    name = md.setdefault(b"name", b"root")

    return pa.field(name.decode(), dtype, False, md)


@register_converter(pa.Field, pa.Schema)
def arrow_field_to_schema(
    data: pa.Field,
    options: Optional[CastOptions] = None,
) -> pa.Schema:
    """Return a schema view of an Arrow field.

    Args:
        data: Arrow field.
        options: Optional cast options.

    Returns:
        Arrow schema.
    """
    md = dict(data.metadata or {})
    md[b"name"] = data.name.encode()

    if pa.types.is_struct(data.type):
        return pa.schema(list(data.type), metadata=md)
    return pa.schema([data], metadata=md)


@register_converter(pa.Table, pa.Field)
@register_converter(pa.RecordBatch, pa.Field)
@register_converter(pa.RecordBatchReader, pa.Field)
def arrow_tabular_to_field(
    data: Union[pa.Table, pa.RecordBatch, pa.RecordBatchReader],
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """Return a field representing the schema of tabular data.

    Args:
        data: Arrow table/batch/reader.
        options: Optional cast options.

    Returns:
        Arrow field.
    """
    return arrow_schema_to_field(data.schema, options)
