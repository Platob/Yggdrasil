import dataclasses
import enum
from dataclasses import is_dataclass
from typing import Optional, Union, List

import pyarrow as pa
import pyarrow.compute as pc

from .cast_options import ArrowCastOptions
from .registry import register_converter
from ..python_defaults import default_from_arrow_hint
from ...dataclasses.dataclass import get_dataclass_arrow_field

__all__ = [
    "cast_arrow_array",
    "cast_arrow_table",
    "cast_arrow_batch",
    "cast_arrow_record_batch_reader",
    "default_arrow_array",
    "pylist_to_arrow_table",
    "pylist_to_record_batch",
    "pylist_to_record_batch_reader",
    "to_spark_arrow_type",
    "to_polars_arrow_type",
    "arrow_field_to_schema"
]


def cast_to_struct_array(
    arr: Union[pa.Array, pa.StructArray, pa.MapArray],
    options: Optional[ArrowCastOptions] = None,
) -> pa.StructArray:
    """Cast arrays to a struct Arrow array."""
    options = ArrowCastOptions.check_arg(options)
    target_field = options.target_field

    if target_field is None:
        return arr

    if isinstance(arr, pa.ChunkedArray):
        casted_chunks = [
            cast_to_struct_array(chunk, options)
            for chunk in arr.chunks
        ]
        return pa.chunked_array(casted_chunks, type=target_field.type)

    source_field = options.source_field or array_to_field(arr, options)
    target_type: pa.StructType = target_field.type
    mask = arr.is_null() if source_field.nullable and target_field.nullable else None
    children: List[pa.Array] = []

    # Case 1: struct -> struct
    if pa.types.is_struct(arr.type):
        name_to_index = {
            field.name: idx for idx, field in enumerate(arr.type)
        }
        folded_to_index = {
            field.name.casefold(): idx for idx, field in enumerate(arr.type)
        }

        for i, target_field in enumerate(target_type):
            if target_field.name in name_to_index:
                child_idx = name_to_index[target_field.name]
                child_arr = arr.field(child_idx)
                child_source_field = arr.type[child_idx]
            elif (
                not options.strict_match_names
                and target_field.name.casefold() in folded_to_index
            ):
                child_idx = folded_to_index[target_field.name.casefold()]
                child_arr = arr.field(child_idx)
                child_source_field = arr.type[child_idx]
            elif not options.strict_match_names and i < arr.type.num_fields:
                # Positional fallback
                child_idx = i
                child_arr = arr.field(child_idx)
                child_source_field = arr.type[child_idx]
            elif options.add_missing_columns:
                # Field missing -> create default-valued array
                child_arr = default_arrow_array(target_field, len(arr))
                child_source_field = array_to_field(child_arr, options)
            else:
                raise pa.ArrowInvalid(
                    f"Missing field {target_field.name} while casting struct"
                )

            children.append(
                cast_arrow_array(
                    child_arr,
                    options.copy(
                        source_field=child_source_field,
                        target_field=target_field
                    )
                )
            )

    # Case 2: map -> struct (e.g. map<string, value> with key-based lookup)
    else:
        map_arr = arr
        map_type: pa.MapType = arr.type

        for target_field in target_type:
            values = pc.map_lookup(map_arr, target_field.name, "first")

            casted = cast_arrow_array(
                values,
                options.copy(
                    source_field=map_type.item_field,
                    target_field=target_field
                )
            )

            children.append(casted)

    return pa.StructArray.from_arrays(
        children,
        fields=list(target_type),
        mask=mask,
        memory_pool=options.memory_pool
    )


def cast_to_list_array(
    arr: Union[pa.Array, pa.ListArray, pa.LargeListArray],
    options: Optional[ArrowCastOptions] = None,
) -> Union[pa.ListArray, pa.LargeListArray]:
    """Cast arrays to a list or large list Arrow array."""
    options = ArrowCastOptions.check_arg(options)
    target_field = options.target_field

    if target_field is None:
        return arr

    if isinstance(arr, pa.ChunkedArray):
        casted_chunks = [
            cast_to_list_array(chunk, options)
            for chunk in arr.chunks
        ]
        return pa.chunked_array(casted_chunks, type=target_field.type)

    target_type: Union[pa.ListType, pa.FixedSizeListType] = target_field.type
    source_field = options.source_field or array_to_field(arr, options)
    mask = arr.is_null() if source_field.nullable and target_field.nullable else None

    if is_type_list_like(source_field.type):
        list_source_field = arr.type.value_field

        offsets = arr.offsets
        values = cast_arrow_array(
            arr.values,
            options.copy(
                source_field=list_source_field,
                target_field=target_type.value_field,
            )
        )
    else:
        raise pa.ArrowInvalid(f"Unsupported list casting for type {arr.type}")

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
        raise ValueError(f"Cannot build arrow array {target_type}")


def cast_to_map_array(
    arr: Union[pa.Array, pa.MapArray, pa.StructArray],
    options: Optional[ArrowCastOptions] = None,
) -> pa.MapArray:
    """Cast arrays to a map Arrow array."""
    options = ArrowCastOptions.check_arg(options)
    target_field = options.target_field

    if target_field is None:
        return arr

    if isinstance(arr, pa.ChunkedArray):
        casted_chunks = [
            cast_to_map_array(chunk, options)
            for chunk in arr.chunks
        ]
        return pa.chunked_array(casted_chunks, type=target_field.type)

    source_field = options.source_field or array_to_field(arr, options)
    target_type: pa.MapType = options.target_field.type
    mask = arr.is_null() if source_field.nullable and target_field.nullable else None

    # Case 1: map -> map
    if pa.types.is_map(arr.type):
        keys = cast_arrow_array(
            arr.keys,
            options.copy(
                source_field=arr.type.key_field,
                target_field=target_type.key_field,
            )
        )
        items = cast_arrow_array(
            arr.items,
            options.copy(
                source_field=arr.type.item_field,
                target_field=target_type.item_field,
            )
        )
        return pa.MapArray.from_arrays(
            arr.offsets,
            keys,
            items,
            mask=mask,
            type=target_type,
            pool=options.memory_pool
        )

    # Case 2: struct -> map (field.name => value)
    if not pa.types.is_struct(arr.type):
        raise pa.ArrowInvalid(f"Cannot cast non-map array to map type {target_type}")

    num_rows = len(arr)
    offsets = [0]
    keys: List[str] = []
    items: List[object] = []
    mask = arr.is_null() if arr.null_count else None

    # Pre-cast all children values
    casted_children = [
        cast_arrow_array(
            arr.field(i),
            options.copy(
                source_field=arr.type[i],
                target_field=target_type.item_field,
            )
        )
        for i in range(arr.type.num_fields)
    ]

    for row_idx in range(num_rows):
        if mask is not None and mask[row_idx].as_py():
            # Null row -> no entries added for this row
            offsets.append(offsets[-1])
            continue

        for field_idx, field in enumerate(arr.type):
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
        pool=options.memory_pool
    )


def cast_primitive_array(
    arr: pa.Array,
    options: ArrowCastOptions | None = None,
) -> pa.Array:
    """Cast simple scalar arrays via pyarrow.compute.cast."""
    if not options:
        return arr

    target_field = options.target_field

    if target_field is None:
        return arr

    casted = pc.cast(
        arr,
        target_type=target_field.type,
        safe=options.safe,
        memory_pool=options.memory_pool,
    )

    return check_array_nullability(casted, options)


def check_array_nullability(
    arr: Union[pa.Array, pa.ChunkedArray],
    options: Optional[ArrowCastOptions] = None,
) -> Union[pa.Array, pa.ChunkedArray]:
    """
    For non-nullable targets, replace nulls with default Python values.

    If the *source* is already non-nullable and has no nulls, we leave it alone.
    """
    options = ArrowCastOptions.check_arg(options)
    target_field = options.target_field

    if target_field is None:
        return arr

    source_field = options.source_field or array_to_field(arr, options)

    if not source_field.nullable or target_field.nullable:
        return arr

    if isinstance(arr, pa.ChunkedArray):
        if arr.null_count == 0:
            return arr

        options = options.copy(
            source_field=source_field,
            target_field=target_field,
        )

        filled_chunks = [
            check_array_nullability(chunk, options)
            for chunk in arr.chunks
        ]
        return pa.chunked_array(filled_chunks, type=arr.type)

    if arr.null_count == 0:
        # Source already guaranteed non-nullable and contains no nulls.
        return arr

    if arr.null_count:
        fill_scalar = default_from_arrow_hint(target_field.type)
        default_arr = pa.array([fill_scalar] * len(arr), type=target_field.type)
        return pc.if_else(pc.is_null(arr), default_arr, arr)

    return arr


def is_type_list_like(arrow_type: pa.DataType) -> bool:
    """Check if an Arrow type is list-like."""
    return (
        pa.types.is_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
        or pa.types.is_list_view(arrow_type)
    )


def is_string_like(arrow_type: pa.DataType) -> bool:
    """Check if an Arrow type is string-like."""
    return (
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_string_view(arrow_type)
    )


@register_converter(object, pa.Scalar)
def any_to_arrow_scalar(
    scalar: object,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Scalar:
    if isinstance(scalar, pa.Scalar):
        return cast_arrow_scalar(scalar, options)

    options = ArrowCastOptions.check_arg(options)
    target_field = options.target_field

    if isinstance(scalar, enum.Enum):
        scalar = scalar.value

    if is_dataclass(scalar):
        if not target_field:
            target_field = get_dataclass_arrow_field(scalar)
            options = options.copy(target_field=target_field)

        scalar = dataclasses.asdict(scalar)

    if target_field is None:
        if is_dataclass(scalar):
            scalar = pa.scalar(dataclasses.asdict(scalar), type=get_dataclass_arrow_field(scalar).type)
        else:
            scalar = pa.scalar(scalar)

        return scalar

    if scalar is None and not target_field.nullable:
        return default_from_arrow_hint(target_field)

    try:
        scalar = pa.scalar(scalar, type=target_field.type)
    except:
        scalar = pa.scalar(scalar)

    return cast_arrow_scalar(scalar, options)


@register_converter(pa.Scalar, pa.Scalar)
def cast_arrow_scalar(
    scalar: pa.Scalar,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Scalar:
    options = ArrowCastOptions.check_arg(options)
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
    options: Optional[ArrowCastOptions] = None,
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
    options = ArrowCastOptions.check_arg(options)
    target_field = options.target_field

    # No target -> nothing to do
    if target_field is None:
        return array

    target_type = target_field.type

    if pa.types.is_nested(target_type):
        if pa.types.is_struct(target_type):
            return cast_to_struct_array(array, options)
        elif is_type_list_like(target_type):
            return cast_to_list_array(array, options)
        elif pa.types.is_map(target_type):
            return cast_to_map_array(array, options)
        raise ValueError(f"Unsupported nested target type {target_type}")
    else:
        return cast_primitive_array(array, options)

# ---------------------------------------------------------------------------
# Table / RecordBatch casting
# ---------------------------------------------------------------------------


@register_converter(pa.Table, pa.Table)
def cast_arrow_table(
    data: pa.Table,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Table:
    """
    Cast a pyarrow.Table to `options.target_schema`.

    Handles:
    - Column name matching (exact, case-insensitive, positional)
    - Missing columns (optional default creation)
    - Column-wise casting via `cast_arrow_array`.
    """
    options = ArrowCastOptions.check_arg(options)
    arrow_schema = options.target_schema

    if arrow_schema is None:
        # No target schema -> return as-is
        return data

    exact_name_to_index = {name: idx for idx, name in enumerate(data.column_names)}
    folded_name_to_index = {
        name.casefold(): idx for idx, name in enumerate(data.column_names)
    }

    columns: List[Union[pa.Array, pa.ChunkedArray]] = []

    for target_field in arrow_schema:
        # Exact match
        if target_field.name in exact_name_to_index:
            col_idx = exact_name_to_index[target_field.name]
            column = data.column(col_idx)
            source_field = data.schema.field(col_idx)

        # Case-insensitive match
        elif not options.strict_match_names and target_field.name.casefold() in folded_name_to_index:
            col_idx = folded_name_to_index[target_field.name.casefold()]
            column = data.column(col_idx)
            source_field = data.schema.field(col_idx)

        # Positional fallback
        elif not options.strict_match_names and data.num_columns > len(columns):
            col_idx = len(columns)
            column = data.column(col_idx)
            source_field = data.schema.field(col_idx)

        # Missing -> default column
        elif options.add_missing_columns:
            column = default_arrow_array(target_field, data.num_rows)
            source_field = None

        else:
            raise pa.ArrowInvalid(
                f"Missing column {target_field.name} while casting table"
            )

        casted_col = cast_arrow_array(
            column,
            options.copy(
                source_field=source_field,
                target_field=target_field,
            )
        )
        columns.append(casted_col)

    # Extra columns in `data` are ignored when building the new table
    return pa.Table.from_arrays(columns, schema=arrow_schema)


@register_converter(pa.RecordBatch, pa.RecordBatch)
def cast_arrow_batch(
    data: pa.RecordBatch,
    options: Optional[ArrowCastOptions] = None,
) -> pa.RecordBatch:
    """
    Cast a pyarrow.RecordBatch to `options.target_schema`.

    Same semantics as `cast_arrow_table`, but for a single RecordBatch.
    """
    if options is None:
        return data

    options = ArrowCastOptions.check_arg(options)
    arrow_schema = options.target_schema

    if arrow_schema is None:
        return data

    exact_name_to_index = {
        name: idx for idx, name in enumerate(data.schema.names)
    }

    folded_name_to_index = {
        name.casefold(): idx for idx, name in enumerate(data.schema.names)
    }

    columns: List[Union[pa.Array, pa.ChunkedArray]] = []

    for target_field in arrow_schema:
        # Exact match
        if target_field.name in exact_name_to_index:
            col_idx = exact_name_to_index[target_field.name]
            column = data.column(col_idx)
            source_field = data.schema.field(col_idx)

        # Case-insensitive match
        elif not options.strict_match_names and target_field.name.casefold() in folded_name_to_index:
            col_idx = folded_name_to_index[target_field.name.casefold()]
            column = data.column(col_idx)
            source_field = data.schema.field(col_idx)

        # Positional fallback
        elif not options.strict_match_names and data.num_columns > len(columns):
            col_idx = len(columns)
            column = data.column(col_idx)
            source_field = data.schema.field(col_idx)

        # Missing -> default column
        elif options.add_missing_columns:
            column = default_arrow_array(target_field, data.num_rows)
            source_field = None

        else:
            raise pa.ArrowInvalid(
                f"Missing column {target_field.name} while casting record batch"
            )

        casted_col = cast_arrow_array(
            column,
            options.copy(
                source_field=source_field,
                target_field=target_field,
            ),
        )
        columns.append(casted_col)

    # Extra columns are ignored in the constructed RecordBatch
    return pa.RecordBatch.from_arrays(columns, schema=arrow_schema)


@register_converter(pa.RecordBatchReader, pa.RecordBatchReader)
def cast_arrow_record_batch_reader(
    data: pa.RecordBatchReader,
    options: Optional[ArrowCastOptions] = None,
) -> pa.RecordBatchReader:
    """
    Wrap a RecordBatchReader and lazily cast each batch to `options.target_schema`.
    """
    options = ArrowCastOptions.check_arg(options)
    arrow_schema = options.target_schema

    if arrow_schema is None:
        # Nothing to cast, just return the original reader
        return data

    def casted_batches():
        for batch in data:
            yield cast_arrow_batch(batch, options)

    return pa.RecordBatchReader.from_batches(arrow_schema, casted_batches())


# ---------------------------------------------------------------------------
# Default-valued arrays
# ---------------------------------------------------------------------------


def default_arrow_array(
    field: Union[pa.Field, pa.DataType],
    length: int,
) -> pa.Array:
    """
    Build an Arrow array of length `length` filled with default values
    for the given field / dtype.

    - If field is nullable -> returns a null array of the given type.
    - Otherwise -> returns an array filled with default_from_arrow_hint(dtype).
    """
    value = default_from_arrow_hint(field)

    if value.as_py() is None:
        return pa.nulls(length, type=value.type)

    return pa.array([value] * length, type=value.type)


# ---------------------------------------------------------------------------
# Pylist -> Arrow
# ---------------------------------------------------------------------------
@register_converter(list, pa.Array)
def pylist_to_arrow_array(
    pylist: list,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Array:
    options = ArrowCastOptions.check_arg(options)
    target_field = options.target_field

    if not pylist:
        return pa.array([], type=target_field.type if target_field else pa.null())

    found_scalar = None
    null_count = 0

    for item in pylist:
        if item is not None:
            found_scalar = any_to_arrow_scalar(item, None)

            if target_field is None:
                target_field = arrow_type_to_field(found_scalar.type, None)
                options = options.copy(target_field=target_field)
            break
        else:
            null_count += 1

    if null_count == len(pylist):
        return default_arrow_array(field=target_field if target_field else pa.null(), length=len(pylist))

    if found_scalar:
        scalars = [
            any_to_arrow_scalar(item, options)
            for item in pylist
        ]
    else:
        return default_arrow_array(field=target_field if target_field else pa.null(), length=len(pylist))

    arr = pa.array(scalars, type=target_field.type if target_field else None)

    return cast_arrow_array(arr, options)


@register_converter(list, pa.RecordBatch)
def pylist_to_record_batch(
    data: list,
    options: Optional[ArrowCastOptions] = None,
) -> pa.RecordBatch:
    options = ArrowCastOptions.check_arg(options)

    array: Union[pa.Array, pa.StructArray] = pylist_to_arrow_array(data, options)

    target_field = options.target_field or arrow_array_to_field(array, None)
    target_type: Union[pa.DataType, pa.StructType] = target_field.type

    schema = arrow_field_to_schema(target_field, None)

    if isinstance(array, pa.StructArray):
        arrays = [
            array.field(i) for i in range(target_type.num_fields)
        ]
    else:
        arrays = [array]

    return pa.record_batch(
        arrays,
        schema=schema,
    )


@register_converter(list, pa.Table)
def pylist_to_arrow_table(
    data: list,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Table:
    batch = pylist_to_record_batch(data, options)
    return record_batch_to_table(batch, None)


@register_converter(list, pa.RecordBatchReader)
def pylist_to_record_batch_reader(
    data: list,
    options: Optional[ArrowCastOptions] = None,
) -> pa.RecordBatchReader:
    batch = pylist_to_record_batch(data, options)
    return record_batch_to_record_batch_reader(batch, None)


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
    if pa.types.is_large_string(dtype) or pa.types.is_string_view(dtype):
        return pa.string()
    if pa.types.is_large_binary(dtype) or pa.types.is_binary_view(dtype):
        return pa.binary()

    # Large list -> normal list with normalized value type
    if pa.types.is_large_list(dtype) or pa.types.is_list_view(dtype):
        return pa.list_(to_spark_arrow_type(dtype.value_type))

    # Normal list: still normalize value type
    if pa.types.is_list(dtype):
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
@register_converter(pa.Array, pa.Field)
@register_converter(pa.ChunkedArray, pa.Field)
def array_to_field(
    array: Union[pa.Array, pa.ChunkedArray],
    options: Optional[ArrowCastOptions] = None,
) -> pa.Field:
    """
    Convert an Arrow array or chunked array to a Field describing its type.
    """
    return pa.field(
        str(array.type),
        array.type,
        nullable=array.null_count > 0,
        metadata=None,
    )


@register_converter(pa.Table, pa.RecordBatch)
def table_to_record_batch(
    data: pa.Table,
    options: Optional[ArrowCastOptions] = None,
) -> pa.RecordBatch:
    """
    Cast a Table using `cast_arrow_table` and return a single RecordBatch.

    Handles the fact that Table columns are ChunkedArray, while
    RecordBatch expects plain Array.
    """
    casted = cast_arrow_table(data, options)

    # Empty table: build an empty batch with same schema
    if casted.num_rows == 0:
        arrays = [pa.array([], type=f.type) for f in casted.schema]
        return pa.RecordBatch.from_arrays(arrays, schema=casted.schema)

    # Convert table to batches (these have Array columns)
    batches = casted.to_batches()

    if len(batches) == 1:
        # Already a single RecordBatch with Array columns
        return batches[0]

    # Merge multiple batches into one RecordBatch
    merged_arrays = []
    for col_idx, field in enumerate(casted.schema):
        col_chunks = [b.column(col_idx) for b in batches]
        chunked = pa.chunked_array(col_chunks, type=field.type)
        merged_arrays.append(chunked.combine_chunks())

    return pa.RecordBatch.from_arrays(merged_arrays, schema=casted.schema)


@register_converter(pa.RecordBatch, pa.Table)
def record_batch_to_table(
    data: pa.RecordBatch,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Table:
    """
    Cast a RecordBatch using `cast_arrow_batch` and wrap as a single-batch Table.
    """
    casted = cast_arrow_batch(data, options)
    return pa.Table.from_batches([casted])


@register_converter(pa.Table, pa.RecordBatchReader)
def table_to_record_batch_reader(
    data: pa.Table,
    options: Optional[ArrowCastOptions] = None,
) -> pa.RecordBatchReader:
    """
    Cast a Table and expose it as a RecordBatchReader.
    """
    casted = cast_arrow_table(data, options)
    return pa.RecordBatchReader.from_batches(
        casted.schema,
        casted.to_batches(),
    )


@register_converter(pa.RecordBatchReader, pa.Table)
def record_batch_reader_to_table(
    data: pa.RecordBatchReader,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Table:
    """
    Cast each batch in a RecordBatchReader and collect into a Table.
    """
    casted_reader = cast_arrow_record_batch_reader(data, options)
    return pa.Table.from_batches(list(casted_reader))


@register_converter(pa.RecordBatch, pa.RecordBatchReader)
def record_batch_to_record_batch_reader(
    data: pa.RecordBatch,
    options: Optional[ArrowCastOptions] = None,
) -> pa.RecordBatchReader:
    """
    Cast a RecordBatch and wrap it into a single-batch RecordBatchReader.
    """
    casted = cast_arrow_batch(data, options)
    return pa.RecordBatchReader.from_batches(casted.schema, [casted])


@register_converter(pa.RecordBatchReader, pa.RecordBatch)
def record_batch_reader_to_record_batch(
    data: pa.RecordBatchReader,
    options: Optional[ArrowCastOptions] = None,
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


@register_converter(pa.DataType, pa.Field)
def arrow_type_to_field(
    data: pa.DataType,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Field:
    return pa.field(str(data), data, True, None)


@register_converter([pa.Array, pa.ChunkedArray], pa.Field)
def arrow_array_to_field(
    data: Union[pa.Array, pa.ChunkedArray],
    options: Optional[ArrowCastOptions] = None,
) -> pa.Field:
    return pa.field(str(data.type), data.type, data.null_count > 0, None)


@register_converter(pa.Schema, pa.Field)
def arrow_schema_to_field(
    data: pa.Schema,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Field:
    dtype = pa.struct(list(data))
    md = dict(data.metadata or {})
    name = md.setdefault(b"name", b"root")
    return pa.field(name.decode(), dtype, False, md)


@register_converter(pa.Field, pa.Schema)
def arrow_field_to_schema(
    data: pa.Field,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Schema:
    md = dict(data.metadata or {})
    md[b"name"] = data.name.encode()

    if pa.types.is_struct(data.type):
        return pa.schema(list(data.type), metadata=md)

    return pa.schema([data], metadata=md)


@register_converter([pa.Table, pa.RecordBatch, pa.RecordBatchReader], pa.Field)
def arrow_tabular_to_field(
    data: Union[pa.Table, pa.RecordBatch, pa.RecordBatchReader],
    options: Optional[ArrowCastOptions] = None,
) -> pa.Field:
    if isinstance(data, pa.RecordBatchReader):
        schema = data.schema
    else:
        schema = data.schema
    return arrow_schema_to_field(schema, options)
