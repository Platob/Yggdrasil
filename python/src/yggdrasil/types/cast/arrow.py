import decimal
from dataclasses import dataclass, replace
from typing import Union, Optional

import pyarrow as pa
import pyarrow.compute as pc

from .registry import register_converter

__all__ = [
    "ArrowCastOptions",
    "cast_arrow_array",
    "cast_arrow_table",
    "cast_arrow_batch",
    "cast_arrow_record_batch_reader",
    "default_arrow_array",
    "default_arrow_python_value",
]


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
        Can be a pa.Field, pa.Schema, or pa.DataType (normalized in __post_init__).
    target_field:
        Description of the target field/schema. Can be pa.Field, pa.Schema,
        or pa.DataType (normalized in __post_init__).
    """

    safe: bool = False
    add_missing_columns: bool = True
    strict_match_names: bool = False
    allow_add_columns: bool = False
    rename: bool = True
    memory_pool: Optional[pa.MemoryPool] = None
    default_value: object = None
    source_hint: object = None
    target_hint: object = None
    source_field: Optional[pa.Field] = None
    target_field: Optional[pa.Field] = None

    @classmethod
    def check_arg(
        cls,
        arg: Optional[Union["ArrowCastOptions", dict, pa.DataType, pa.Field, pa.Schema, object]],
    ) -> "ArrowCastOptions":
        """
        Normalize an argument into an ArrowCastOptions instance.

        - If `arg` is already ArrowCastOptions, return it.
        - If `arg` is a dict, delegate to `ArrowCastOptions.from_dict(arg)`.
          (Assumed to be implemented elsewhere.)
        - If `arg` is a DataType/Field/Schema, treat it as the target.
        - If None or anything else, return DEFAULT_CAST_OPTIONS.
        """
        if isinstance(arg, ArrowCastOptions):
            return arg

        if isinstance(arg, dict):
            # Assuming ArrowCastOptions.from_dict(...) exists in your codebase.
            return cls.from_dict(arg)

        if isinstance(arg, (pa.DataType, pa.Field, pa.Schema)):
            return replace(DEFAULT_CAST_OPTIONS, target_field=arg)

        if arg is not None:
            return replace(DEFAULT_CAST_OPTIONS, target_hint=arg)

        return DEFAULT_CAST_OPTIONS

    @property
    def target_schema(self) -> Optional[pa.Schema]:
        """
        Schema view of `target_field`.

        - If target_field is a struct, unwrap its children as schema fields.
        - Otherwise treat target_field as a single-field schema.
        """
        if self.target_field is not None:
            if pa.types.is_struct(self.target_field.type):
                return pa.schema(
                    list(self.target_field.type),
                    metadata=self.target_field.metadata,
                )
            return pa.schema([self.target_field], metadata=self.target_field.metadata)
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

    def __post_init__(self) -> None:
        """
        Normalize `source_field` and `target_field` to pa.Field instances.

        Supported input types:
        - pa.Schema  -> wrapped as struct("root", struct(schema_fields))
        - pa.DataType -> wrapped as field("root", dtype)
        - pa.Field     -> used as-is
        """
        # Normalize source_field
        if self.source_field is not None:
            if isinstance(self.source_field, pa.Schema):
                self.source_field = pa.field(
                    "root",
                    pa.struct(list(self.source_field)),
                    nullable=False,
                    metadata=self.source_field.metadata,
                )
            elif isinstance(self.source_field, pa.DataType):
                self.source_field = pa.field(
                    "root",
                    self.source_field,
                    nullable=True,
                    metadata=None,
                )

        # Normalize target_field
        if self.target_field is not None:
            if isinstance(self.target_field, pa.Schema):
                self.target_field = pa.field(
                    "root",
                    pa.struct(list(self.target_field)),
                    nullable=False,
                    metadata=self.target_field.metadata,
                )
            elif isinstance(self.target_field, pa.DataType):
                self.target_field = pa.field(
                    "root",
                    self.target_field,
                    nullable=True,
                    metadata=None,
                )

        if self.target_hint is None and self.target_field is not None:
            self.target_hint = self.target_field

        if self.source_hint is None and self.source_field is not None:
            self.source_hint = self.source_field


DEFAULT_CAST_OPTIONS = ArrowCastOptions()


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

    Nullability is enforced using `default_arrow_python_value` for non-nullable targets.
    """
    options = ArrowCastOptions.check_arg(options)
    default_value = options.default_value

    source_field = options.source_field
    target_field = options.target_field

    # No target -> nothing to do
    if target_field is None:
        return data

    target_type = target_field.type
    target_nullable = target_field.nullable

    def _cast_single(
        arr: pa.Array,
        target: pa.DataType,
        *,
        nullable: bool,
        source_field: Optional[pa.Field],
    ) -> pa.Array:
        """
        Cast a single (non-chunked) array to `target`, handling complex types.
        """

        # ---------- Struct casting ----------
        if pa.types.is_struct(target):
            if not pa.types.is_struct(arr.type) and not pa.types.is_map(arr.type):
                raise pa.ArrowInvalid(
                    f"Cannot cast non-struct array to struct type {target}"
                )

            children = []

            # Case 1: struct -> struct
            if pa.types.is_struct(arr.type):
                name_to_index = {field.name: idx for idx, field in enumerate(arr.type)}
                folded_to_index = {
                    field.name.casefold(): idx for idx, field in enumerate(arr.type)
                }

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
                    elif (
                        not options.strict_match_names
                        and i < arr.type.num_fields
                    ):
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
                        _cast_array(
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
                if (
                    not options.strict_match_names
                    and pa.types.is_string(arr.type.key_type)
                ):
                    lowered_keys = pc.utf8_lower(arr.keys)
                    map_arr = pa.MapArray.from_arrays(
                        arr.offsets,
                        lowered_keys,
                        arr.items,
                        mask=arr.is_null() if arr.null_count else None,
                        type=pa.map_(lowered_keys.type, arr.type.item_type),
                    )

                for field in target:
                    lookup_key = (
                        field.name
                        if options.strict_match_names
                        else field.name.casefold()
                    )

                    values = pc.map_lookup(map_arr, lookup_key, "first")

                    casted = _cast_array(
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

        # ---------- List / LargeList casting ----------
        if pa.types.is_list(target) or pa.types.is_large_list(target):
            if not pa.types.is_list(arr.type) and not pa.types.is_large_list(arr.type):
                raise pa.ArrowInvalid(
                    f"Cannot cast non-list array to list type {target}"
                )

            list_source_field: Optional[pa.Field] = None

            if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
                list_source_field = arr.type.value_field

                values = _cast_array(
                    arr.values,
                    target.value_type,
                    nullable=True,
                    source_field=list_source_field,
                )
                mask = arr.is_null() if arr.null_count else None
                return type(arr).from_arrays(
                    arr.offsets,
                    values,
                    mask=mask,
                    type=target,
                )

        # ---------- Map casting ----------
        if pa.types.is_map(target):
            # Case 1: map -> map
            if pa.types.is_map(arr.type):
                keys = _cast_array(
                    arr.keys,
                    target.key_type,
                    nullable=True,
                    source_field=arr.type.key_field,
                )
                items = _cast_array(
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
                raise pa.ArrowInvalid(
                    f"Cannot cast non-map array to map type {target}"
                )

            num_rows = len(arr)
            offsets = [0]
            keys = []
            items = []
            mask = arr.is_null() if arr.null_count else None

            # Pre-cast all children values
            casted_children = [
                _cast_array(
                    arr.field(i),
                    target.item_type,
                    nullable=True,
                    source_field=arr.type[i],
                )
                for i in range(arr.type.num_fields)
            ]

            for row_idx in range(num_rows):
                if mask is not None and mask[row_idx].as_py():
                    offsets.append(len(keys))
                    continue

                for child_idx, field in enumerate(arr.type):
                    keys.append(field.name)
                    items.append(casted_children[child_idx][row_idx].as_py())

                offsets.append(len(keys))

            return pa.MapArray.from_arrays(
                pa.array(offsets, type=pa.int64()),
                pa.array(keys, type=target.key_type),
                pa.array(items, type=target.item_type),
                mask=mask,
                type=target,
            )

        # ---------- Scalar / simple type casting ----------
        return pc.cast(
            arr,
            target_type=target,
            safe=options.safe,
            memory_pool=options.memory_pool,
        )

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

        if source_field is not None and source_field.nullable is False:
            # Source already guaranteed non-nullable.
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

        if arr.null_count:
            fill_value = default_value
            if fill_value is None:
                fill_value = default_arrow_python_value(dtype)

            default_arr = pa.array([fill_value] * len(arr), type=dtype)
            return pc.if_else(pc.is_null(arr), default_arr, arr)

        return arr

    def _cast_array(
        arr: Union[pa.Array, pa.ChunkedArray],
        target: pa.DataType,
        *,
        nullable: bool,
        source_field: Optional[pa.Field],
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
        )
        return _fill_non_nullable_defaults(
            casted,
            target,
            nullable=nullable,
            source_field=source_field,
        )

    # Entry point
    return _cast_array(
        data,
        target_type,
        nullable=target_nullable,
        source_field=source_field,
    )


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

    columns = []

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
            replace(
                options,
                source_field=source_field,
                target_field=field,
            ),
        )
        columns.append(casted_col)

    # Extra columns in `data` are ignored when building the new table
    return pa.Table.from_arrays(columns, schema=arrow_schema)


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

    columns = []

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
            replace(
                options,
                source_field=source_field,
                target_field=field,
            ),
        )
        columns.append(casted_col)

    # Extra columns are ignored in the constructed RecordBatch
    return pa.RecordBatch.from_arrays(columns, schema=arrow_schema)


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


def default_arrow_array(
    field: Union[pa.Field, pa.DataType],
    length: int,
) -> pa.Array:
    """
    Build an Arrow array of length `length` filled with default values
    for the given field / dtype.

    - If field is nullable -> returns a null array of the given type.
    - Otherwise -> returns an array filled with `default_arrow_python_value(dtype)`.
    """
    if isinstance(field, pa.Field):
        nullable = field.nullable
        dtype = field.type
    else:
        nullable = True
        dtype = field

    if nullable:
        return pa.nulls(length, type=dtype)

    value = default_arrow_python_value(dtype)
    return pa.array([value] * length, type=dtype)


def default_arrow_python_value(dtype: pa.DataType):
    """
    Return a Python default value for a given Arrow dtype.

    Used when we need to fill non-nullable fields that contain nulls.
    """
    if pa.types.is_struct(dtype):
        # For non-nullable struct fields, recurse on nested types.
        return {
            field.name: (
                default_arrow_python_value(field.type)
                if not field.nullable
                else None
            )
            for field in dtype
        }

    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return []

    if pa.types.is_map(dtype):
        return {}

    if pa.types.is_integer(dtype) or pa.types.is_unsigned_integer(dtype):
        return 0

    if pa.types.is_floating(dtype) or pa.types.is_decimal(dtype):
        return decimal.Decimal(0) if pa.types.is_decimal(dtype) else 0.0

    if pa.types.is_boolean(dtype):
        return False

    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return ""

    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return b""

    if pa.types.is_fixed_size_binary(dtype):
        return b"\x00" * dtype.byte_width

    if (
        pa.types.is_timestamp(dtype)
        or pa.types.is_time(dtype)
        or pa.types.is_duration(dtype)
        or pa.types.is_interval(dtype)
    ):
        # Represent temporal zero as 0 (epoch, zero duration, etc.)
        return 0

    # Fallback: nothing better to do
    return None


# ---------------------------------------------------------------------------
# Cross-container casting helpers
# ---------------------------------------------------------------------------

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
        # Column per batch -> list[Array]
        col_chunks = [b.column(col_idx) for b in batches]
        chunked = pa.chunked_array(col_chunks, type=field.type)
        merged_arrays.append(chunked.combine_chunks())

    return pa.RecordBatch.from_arrays(merged_arrays, schema=casted.schema)


def record_batch_to_table(
    data: pa.RecordBatch,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Table:
    """
    Cast a RecordBatch using `cast_arrow_batch` and wrap as a single-batch Table.
    """
    casted = cast_arrow_batch(data, options)
    return pa.Table.from_batches([casted])


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


def record_batch_reader_to_table(
    data: pa.RecordBatchReader,
    options: Optional[ArrowCastOptions] = None,
) -> pa.Table:
    """
    Cast each batch in a RecordBatchReader and collect into a Table.
    """
    casted_reader = cast_arrow_record_batch_reader(data, options)
    return pa.Table.from_batches(list(casted_reader))


def record_batch_to_record_batch_reader(
    data: pa.RecordBatch,
    options: Optional[ArrowCastOptions] = None,
) -> pa.RecordBatchReader:
    """
    Cast a RecordBatch and wrap it into a single-batch RecordBatchReader.
    """
    casted = cast_arrow_batch(data, options)
    return pa.RecordBatchReader.from_batches(casted.schema, [casted])


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


# Inner / column-level
register_converter(pa.Array, pa.Array)(cast_arrow_array)
register_converter(pa.ChunkedArray, pa.ChunkedArray)(cast_arrow_array)

# Same-type tabular / batch
register_converter(pa.Table, pa.Table)(cast_arrow_table)
register_converter(pa.RecordBatch, pa.RecordBatch)(cast_arrow_batch)
register_converter(pa.RecordBatchReader, pa.RecordBatchReader)(
    cast_arrow_record_batch_reader
)

# Table <-> RecordBatch
register_converter(pa.Table, pa.RecordBatch)(table_to_record_batch)
register_converter(pa.RecordBatch, pa.Table)(record_batch_to_table)

# Table <-> RecordBatchReader
register_converter(pa.Table, pa.RecordBatchReader)(table_to_record_batch_reader)
register_converter(pa.RecordBatchReader, pa.Table)(
    record_batch_reader_to_table
)

# RecordBatch <-> RecordBatchReader
register_converter(pa.RecordBatch, pa.RecordBatchReader)(
    record_batch_to_record_batch_reader
)
register_converter(pa.RecordBatchReader, pa.RecordBatch)(
    record_batch_reader_to_record_batch
)
