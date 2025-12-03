from dataclasses import dataclass, replace as dc_replace, fields as dc_fields, is_dataclass, asdict
from functools import partial
from enum import Enum
from typing import Optional, Union, List

import pyarrow as pa
import pyarrow.compute as pc

from .registry import register_converter, convert
from ..python_arrow import arrow_field_from_hint
from ..python_defaults import default_from_arrow_hint

__all__ = [
    "DataFormat",
    "ArrowCastOptions",
    "cast_arrow_array",
    "cast_arrow_table",
    "cast_arrow_batch",
    "cast_arrow_record_batch_reader",
    "default_arrow_array",
    "pylist_to_arrow_table",
    "pylist_to_record_batch",
    "pylist_to_record_batch_reader",
    "to_spark_arrow_type",
    "to_polars_arrow_type"
]

from ...libs import polars


# ---------------------------------------------------------------------------
# DataFormat enum
# ---------------------------------------------------------------------------


class DataFormat(Enum):
    ARROW = "arrow"
    SPARK = "spark"
    POLARS = "polars"
    PANDAS = "pandas"


# ---------------------------------------------------------------------------
# ArrowCastOptions
# ---------------------------------------------------------------------------


@dataclass
class ArrowCastOptions:
    """
    Options controlling Arrow casting behavior.

    Attributes
    ----------
    safe:
        If True, only allow "safe" casts (delegated to pyarrow.compute.cast).
    add_missing_columns:
        If True, create default-valued columns/fields when target schema has
        fields that are missing in the source.
    strict_match_names:
        If True, only match fields/columns by exact name (case-sensitive).
        If False, allows case-insensitive and positional matching.
    allow_add_columns:
        If True, allow additional columns beyond the target schema to remain.
        If False, extra columns are effectively ignored.
    rename:
        Reserved / placeholder for rename behavior (currently unused).
    memory_pool:
        Optional Arrow memory pool passed down to compute kernels.
    source_field:
        Description of the source field/schema. Used to infer nullability behavior.
        Can be a pa.Field, pa.Schema, or pa.DataType (normalized elsewhere).
    target_field:
        Description of the target field/schema. Can be pa.Field, pa.Schema,
        or pa.DataType (normalized elsewhere).
    """

    safe: bool = False
    add_missing_columns: bool = True
    strict_match_names: bool = False
    allow_add_columns: bool = False
    rename: bool = True
    memory_pool: Optional[pa.MemoryPool] = None
    source_field: Optional[pa.Field] = None
    target_field: Optional[pa.Field] = None
    datetime_formats: Optional[List[str]] = None

    @classmethod
    def check_arg(
        cls,
        arg: Union[
            "ArrowCastOptions",
            dict,
            pa.DataType,
            pa.Field,
            pa.Schema,
            DataFormat,
            None,
        ] = None,
        kwargs: Optional[dict] = None,
    ) -> "ArrowCastOptions":
        """
        Normalize an argument into an ArrowCastOptions instance.

        - If `arg` is already ArrowCastOptions, return it.
        - Otherwise, treat `arg` as something convertible to pa.Field via
          the registry (`convert(arg, Optional[pa.Field])`) and apply it
          as `target_field` on top of DEFAULT_CAST_OPTIONS.
        - If arg is None, just use DEFAULT_CAST_OPTIONS.
        """
        if isinstance(arg, ArrowCastOptions):
            result = arg
        else:
            result = dc_replace(
                DEFAULT_CAST_OPTIONS,
                target_field=convert(arg, Optional[pa.Field]),
            )

        if kwargs:
            result = dc_replace(result, **kwargs)

        return result

    @property
    def target_schema(self) -> Optional[pa.Schema]:
        """
        Schema view of `target_field`.

        - If target_field is a struct, unwrap its children as schema fields.
        - Otherwise treat target_field as a single-field schema.
        """
        if self.target_field is not None:
            return arrow_field_to_schema(self.target_field, None)
        return None

    @target_schema.setter
    def target_schema(self, value: pa.Schema) -> None:
        """
        Set `target_field` from a `pa.Schema`, wrapping it as a root struct field.
        """
        self.target_field = pa.field(
            "root",
            pa.struct(list(value)),
            nullable=False,
            metadata=value.metadata,
        )


DEFAULT_CAST_OPTIONS = ArrowCastOptions()

# ---------------------------------------------------------------------------
# Core array casting
# ---------------------------------------------------------------------------


def cast_to_struct_array(
    arr: Union[pa.Array, pa.StructArray, pa.MapArray],
    target: pa.DataType,
    *,
    nullable: bool,
    source_field: Optional[pa.Field],
    options: ArrowCastOptions,
    _cast_array_fn,
) -> pa.StructArray:
    """Cast arrays to a struct Arrow array."""

    if not pa.types.is_struct(arr.type) and not pa.types.is_map(arr.type):
        raise pa.ArrowInvalid(f"Cannot cast non-struct array to struct type {target}")

    children: List[pa.Array] = []

    # Case 1: struct -> struct
    if pa.types.is_struct(arr.type):
        name_to_index = {field.name: idx for idx, field in enumerate(arr.type)}
        folded_to_index = {field.name.casefold(): idx for idx, field in enumerate(arr.type)}

        for i, field in enumerate(target):
            if field.name in name_to_index:
                child_idx = name_to_index[field.name]
                child_arr = arr.field(child_idx)
                child_source_field = arr.type[child_idx]
            elif (
                not options.strict_match_names
                and field.name.casefold() in folded_to_index
            ):
                child_idx = folded_to_index[field.name.casefold()]
                child_arr = arr.field(child_idx)
                child_source_field = arr.type[child_idx]
            elif not options.strict_match_names and i < arr.type.num_fields:
                # Positional fallback
                child_idx = i
                child_arr = arr.field(child_idx)
                child_source_field = arr.type[child_idx]
            elif options.add_missing_columns:
                # Field missing -> create default-valued array
                child_arr = default_arrow_array(field, len(arr))
                child_source_field = None
            else:
                raise pa.ArrowInvalid(
                    f"Missing field {field.name} while casting struct"
                )

            children.append(
                _cast_array_fn(
                    child_arr,
                    field.type,
                    nullable=field.nullable,
                    source_field=child_source_field,
                )
            )

    # Case 2: map -> struct (e.g. map<string, value> with key-based lookup)
    else:
        map_arr = arr

        # Optional case-insensitive map keys
        if not options.strict_match_names and pa.types.is_string(arr.type.key_type):
            lowered_keys = pc.utf8_lower(arr.keys)
            map_arr = pa.MapArray.from_arrays(
                arr.offsets,
                lowered_keys,
                arr.items,
                mask=arr.is_null() if arr.null_count else None,
                type=pa.map_(lowered_keys.type, arr.type.item_type),
            )

        for field in target:
            lookup_key = field.name if options.strict_match_names else field.name.casefold()

            values = pc.map_lookup(map_arr, lookup_key, "first")

            casted = _cast_array_fn(
                values,
                field.type,
                nullable=field.nullable,
                source_field=None,
            )

            # Enforce non-nullability with defaults if needed
            if not field.nullable:
                default_arr = default_arrow_array(field, len(arr))
                casted = pc.if_else(pc.is_null(casted), default_arr, casted)

            children.append(casted)

    mask = arr.is_null() if arr.null_count else None
    return pa.StructArray.from_arrays(
        children,
        fields=list(target),
        mask=mask,
    )


def cast_to_list_array(
    arr: Union[pa.Array, pa.ListArray, pa.LargeListArray],
    target: Union[pa.ListType, pa.LargeListType, pa.FixedSizeListType],
    *,
    nullable: bool,
    source_field: Optional[pa.Field],
    _cast_array_fn,
) -> Union[pa.ListArray, pa.LargeListArray]:
    """Cast arrays to a list or large list Arrow array."""
    mask = arr.is_null() if arr.null_count else None

    if (
        pa.types.is_list(arr.type)
        or pa.types.is_large_list(arr.type)
        or pa.types.is_fixed_size_list(arr.type)
        or pa.types.is_list_view(arr.type)
    ):
        list_source_field = arr.type.value_field

        offsets = arr.offsets
        values = _cast_array_fn(
            arr.values,
            target.value_type,
            nullable=True,
            source_field=list_source_field,
        )
    else:
        raise pa.ArrowInvalid(f"Unsupported list casting for type {arr.type}")

    if pa.types.is_list(target):
        return pa.ListArray.from_arrays(
            offsets,
            values,
            type=target,
            mask=mask
        )
    if pa.types.is_large_list(target):
        return pa.LargeListArray.from_arrays(
            offsets,
            values,
            type=target,
            mask=mask
        )
    elif pa.types.is_fixed_size_list(target):
        return pa.FixedSizeListArray.from_arrays(
            values,
            list_size=target.list_size,
            type=target,
            mask=mask
        )
    else:
        raise ValueError(f"Cannot build arrow array {target}")


def cast_to_map_array(
    arr: Union[pa.Array, pa.MapArray, pa.StructArray],
    target: pa.DataType,
    *,
    nullable: bool,
    source_field: Optional[pa.Field],
    options: ArrowCastOptions,
    _cast_array_fn,
) -> pa.MapArray:
    """Cast arrays to a map Arrow array."""

    # Case 1: map -> map
    if pa.types.is_map(arr.type):
        keys = _cast_array_fn(
            arr.keys,
            target.key_type,
            nullable=True,
            source_field=arr.type.key_field,
        )
        items = _cast_array_fn(
            arr.items,
            target.item_type,
            nullable=True,
            source_field=arr.type.item_field,
        )
        mask = arr.is_null() if arr.null_count else None
        return pa.MapArray.from_arrays(
            arr.offsets,
            keys,
            items,
            mask=mask,
            type=target,
        )

    # Case 2: struct -> map (field.name => value)
    if not pa.types.is_struct(arr.type):
        raise pa.ArrowInvalid(f"Cannot cast non-map array to map type {target}")

    num_rows = len(arr)
    offsets = [0]
    keys: List[str] = []
    items: List[object] = []
    mask = arr.is_null() if arr.null_count else None

    # Pre-cast all children values
    casted_children = [
        _cast_array_fn(
            arr.field(i),
            target.item_type,
            nullable=True,
            source_field=arr.type[i],
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

    map_type = pa.map_(pa.string(), target.item_type, keys_sorted=False)
    return pa.MapArray.from_arrays(
        offsets,
        pa.array(keys, type=pa.string()),
        pa.array(items, type=target.item_type),
        mask=mask,
        type=map_type,
    )


def cast_primitive_array(
    arr: pa.Array,
    target: pa.DataType,
    *,
    options: ArrowCastOptions,
) -> pa.Array:
    """Cast simple scalar arrays via pyarrow.compute.cast."""
    # Special handling: string -> timestamp with multi-format parsing via Polars

    if (
        pa.types.is_string(arr.type)
        or pa.types.is_large_string(arr.type)
        or pa.types.is_string_view(arr.type)
    ):
        if pa.types.is_timestamp(target):
            return cast_arrow_array_strptime(arr, target, options=options)

    try:
        return pc.cast(
            arr,
            target_type=target,
            safe=options.safe,
            memory_pool=options.memory_pool,
        )
    except:
        if not options.safe and polars is not None:
            from .polars_cast import arrow_type_to_polars_type
            pl_type = arrow_type_to_polars_type(target)
            try:
                pl_casted = (
                    polars.from_arrow(arr)
                    .cast(pl_type, strict=False)
                    .to_arrow()
                )
            except Exception as e:
                raise pa.ArrowInvalid(str(e))

            return pc.cast(
                pl_casted,
                target_type=target,
                safe=options.safe,
                memory_pool=options.memory_pool,
            )
        raise


def cast_arrow_array_strptime(
    arr: pa.Array,
    target: pa.TimestampType,
    *,
    options: "ArrowCastOptions",
) -> pa.Array:
    """
    Cast a string array to a timestamp array using Polars + regex parsing.

    Supported patterns:

      YYYY-MM-DD
      YYYY-MM-DD[ T]HH:MM
      YYYY-MM-DD[ T]HH:MM:SS
      ... optionally with .fraction (1–9 digits)
      ... optionally with timezone: Z, +HHMM, +HH:MM, -HHMM, -HH:MM
    """
    from .polars_cast import arrow_type_to_polars_type, cast_polars_series_strptime

    if options.target_field is None:
        return arr

    ticks_series = cast_polars_series_strptime(
        polars.from_arrow(arr),
        options=options,
        target=arrow_type_to_polars_type(options.target_field.type, options)
    )

    out_arr = pa.array(
        ticks_series.to_arrow(),
        type=pa.timestamp(target.unit, tz=target.tz),
    )

    if out_arr.type != target:
        out_arr = pc.cast(
            out_arr,
            target_type=target,
            safe=options.safe,
            memory_pool=options.memory_pool,
        )

    return out_arr


def _fill_non_nullable_defaults(
    arr: Union[pa.Array, pa.ChunkedArray],
    dtype: pa.DataType,
    *,
    nullable: bool,
    source_field: Optional[pa.Field],
) -> Union[pa.Array, pa.ChunkedArray]:
    """
    For non-nullable targets, replace nulls with default Python values.

    If the *source* is already non-nullable and has no nulls, we leave it alone.
    """
    if nullable:
        return arr

    if isinstance(arr, pa.ChunkedArray):
        if arr.null_count == 0:
            return arr

        filled_chunks = [
            _fill_non_nullable_defaults(
                chunk,
                dtype,
                nullable=nullable,
                source_field=source_field,
            )
            for chunk in arr.chunks
        ]
        return pa.chunked_array(filled_chunks, type=arr.type)

    if source_field is not None and source_field.nullable is False and arr.null_count == 0:
        # Source already guaranteed non-nullable and contains no nulls.
        return arr

    if arr.null_count:
        fill_scalar = default_from_arrow_hint(dtype)
        default_arr = pa.array([fill_scalar] * len(arr), type=dtype)
        return pc.if_else(pc.is_null(arr), default_arr, arr)

    return arr


def _cast_single(
    arr: pa.Array,
    target: pa.DataType,
    *,
    nullable: bool,
    source_field: Optional[pa.Field],
    options: ArrowCastOptions,
    _cast_array_fn,
) -> pa.Array:
    """
    Cast a single (non-chunked) array to the target type.
    """

    # ---------- Struct casting ----------
    if pa.types.is_struct(target):
        return cast_to_struct_array(
            arr,
            target,
            nullable=nullable,
            source_field=source_field,
            options=options,
            _cast_array_fn=_cast_array_fn,
        )

    # ---------- List / LargeList casting ----------
    if (
        pa.types.is_list(target)
        or pa.types.is_large_list(target)
        or pa.types.is_fixed_size_list(target)
        or pa.types.is_list_view(target)
    ):
        return cast_to_list_array(
            arr,
            target,
            nullable=nullable,
            source_field=source_field,
            _cast_array_fn=_cast_array_fn,
        )

    # ---------- Map casting ----------
    if pa.types.is_map(target):
        return cast_to_map_array(
            arr,
            target,
            nullable=nullable,
            source_field=source_field,
            options=options,
            _cast_array_fn=_cast_array_fn,
        )

    # ---------- Scalar / simple type casting ----------
    return cast_primitive_array(arr, target, options=options)


def _cast_array(
    arr: Union[pa.Array, pa.ChunkedArray],
    target: pa.DataType,
    *,
    nullable: bool,
    source_field: Optional[pa.Field],
    options: ArrowCastOptions,
) -> Union[pa.Array, pa.ChunkedArray]:
    """
    Recursive array casting with chunked-array support and default filling.
    """
    # Fast path: same type, only need to enforce nullability defaults
    if arr.type.equals(target):
        return _fill_non_nullable_defaults(
            arr,
            target,
            nullable=nullable,
            source_field=source_field,
        )

    # ChunkedArray: cast each chunk and then reassemble
    if isinstance(arr, pa.ChunkedArray):
        chunks = [
            _cast_array(
                chunk,
                target,
                nullable=nullable,
                source_field=source_field,
                options=options,
            )
            for chunk in arr.chunks
        ]
        casted = pa.chunked_array(chunks, type=target)
        return _fill_non_nullable_defaults(
            casted,
            target,
            nullable=nullable,
            source_field=source_field,
        )

    # Single array case
    casted = _cast_single(
        arr,
        target,
        nullable=nullable,
        source_field=source_field,
        options=options,
        _cast_array_fn=partial(_cast_array, options=options),
    )
    return _fill_non_nullable_defaults(
        casted,
        target,
        nullable=nullable,
        source_field=source_field,
    )



@register_converter(pa.Array, pa.Array)
@register_converter(pa.ChunkedArray, pa.ChunkedArray)
def cast_arrow_array(
    data: Union[pa.ChunkedArray, pa.Array],
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

    source_field = options.source_field
    target_field = options.target_field

    # No target -> nothing to do
    if target_field is None:
        return data

    target_type = target_field.type
    target_nullable = target_field.nullable

    return _cast_array(
        data,
        target_type,
        nullable=target_nullable,
        source_field=source_field,
        options=options,
    )

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

    for field in arrow_schema:
        # Exact match
        if field.name in exact_name_to_index:
            col_idx = exact_name_to_index[field.name]
            column = data.column(col_idx)
            source_field = data.schema.field(col_idx)

        # Case-insensitive match
        elif not options.strict_match_names and field.name.casefold() in folded_name_to_index:
            col_idx = folded_name_to_index[field.name.casefold()]
            column = data.column(col_idx)
            source_field = data.schema.field(col_idx)

        # Positional fallback
        elif not options.strict_match_names and data.num_columns > len(columns):
            col_idx = len(columns)
            column = data.column(col_idx)
            source_field = data.schema.field(col_idx)

        # Missing -> default column
        elif options.add_missing_columns:
            column = default_arrow_array(field, data.num_rows)
            source_field = None

        else:
            raise pa.ArrowInvalid(
                f"Missing column {field.name} while casting table"
            )

        casted_col = cast_arrow_array(
            column,
            dc_replace(
                options,
                source_field=source_field,
                target_field=field,
            ),
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
    options = ArrowCastOptions.check_arg(options)
    arrow_schema = options.target_schema

    if arrow_schema is None:
        return data

    exact_name_to_index = {name: idx for idx, name in enumerate(data.schema.names)}
    folded_name_to_index = {
        name.casefold(): idx for idx, name in enumerate(data.schema.names)
    }

    columns: List[Union[pa.Array, pa.ChunkedArray]] = []

    for field in arrow_schema:
        # Exact match
        if field.name in exact_name_to_index:
            col_idx = exact_name_to_index[field.name]
            column = data.column(col_idx)
            source_field = data.schema.field(col_idx)

        # Case-insensitive match
        elif not options.strict_match_names and field.name.casefold() in folded_name_to_index:
            col_idx = folded_name_to_index[field.name.casefold()]
            column = data.column(col_idx)
            source_field = data.schema.field(col_idx)

        # Positional fallback
        elif not options.strict_match_names and data.num_columns > len(columns):
            col_idx = len(columns)
            column = data.column(col_idx)
            source_field = data.schema.field(col_idx)

        # Missing -> default column
        elif options.add_missing_columns:
            column = default_arrow_array(field, data.num_rows)
            source_field = None

        else:
            raise pa.ArrowInvalid(
                f"Missing column {field.name} while casting record batch"
            )

        casted_col = cast_arrow_array(
            column,
            dc_replace(
                options,
                source_field=source_field,
                target_field=field,
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


def _normalize_pylist_value(value):
    if is_dataclass(value):
        return asdict(value)
    return value


def _schema_from_dataclass_instance(instance) -> pa.Schema:
    fields = []

    for field in dc_fields(type(instance)):
        if not field.init or field.name.startswith("_"):
            continue
        fields.append(arrow_field_from_hint(field.type, name=field.name))

    return pa.schema(fields)


def _table_from_pylist(
    data: list,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Table:
    options = ArrowCastOptions.check_arg(options)
    target_schema = options.target_schema
    schema = target_schema

    if schema is None and data:
        for item in data:
            if is_dataclass(item):
                schema = _schema_from_dataclass_instance(item)
                break

    requires_row_mapping = schema is not None or any(
        item is not None
        and (is_dataclass(item) or isinstance(item, dict))
        for item in data
    )

    normalized = []
    for item in data:
        if item is None and requires_row_mapping:
            normalized.append({})
            continue
        normalized.append(_normalize_pylist_value(item))

    if not normalized:
        if schema is None:
            raise pa.ArrowInvalid(
                "Cannot build Arrow table from empty list without target_field"
            )
        arrays = [pa.array([], type=field.type) for field in schema]
        return pa.Table.from_arrays(arrays, schema=schema)

    if schema is None:
        has_keys = any(isinstance(item, dict) and item for item in normalized)
        if not has_keys:
            raise pa.ArrowInvalid(
                "Cannot build Arrow table from list of None/empty rows without target_field"
            )

    return pa.Table.from_pylist(normalized, schema=schema)


@register_converter(list, pa.Table)
def pylist_to_arrow_table(
    data: list,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Table:
    table = _table_from_pylist(data, options)
    return cast_arrow_table(table, options)


@register_converter(list, pa.RecordBatch)
def pylist_to_record_batch(
    data: list,
    options: Optional[ArrowCastOptions] = None,
) -> pa.RecordBatch:
    table = _table_from_pylist(data, options)
    return table_to_record_batch(table, options)


@register_converter(list, pa.RecordBatchReader)
def pylist_to_record_batch_reader(
    data: list,
    options: Optional[ArrowCastOptions] = None,
) -> pa.RecordBatchReader:
    table = _table_from_pylist(data, options)
    return table_to_record_batch_reader(table, options)


# ---------------------------------------------------------------------------
# Type normalization helpers
# ---------------------------------------------------------------------------


def to_spark_arrow_type(
    dtype: Union[pa.DataType, pa.ListType, pa.MapType, pa.StructType]
) -> pa.DataType:
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
    md.setdefault(b"name", data.name.encode())
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
