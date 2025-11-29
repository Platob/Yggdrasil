from dataclasses import dataclass, field, replace, InitVar
from typing import Union, Optional

import decimal

import pyarrow as pa
import pyarrow.compute as pc

__all__ = [
    "ArrowCastOptions",
    "cast_arrow_array",
    "cast_arrow_table",
    "cast_arrow_batch",
    "cast_arrow_record_batch_reader",
]


@dataclass(init=False)
class ArrowCastOptions:
    safe: bool = False
    add_missing_columns: bool = True
    strict_match_names: bool = False
    allow_add_columns: bool =  False
    rename: bool = True
    memory_pool: Optional[pa.MemoryPool] = None
    source_field: Optional[pa.Field] = None
    target_field: Optional[pa.Field] = None

    _target_schema: Optional[pa.Schema] = field(default=None, repr=False)

    def __init__(
        self,
        safe: Optional[bool] = False,
        add_missing_columns: Optional[bool] = True,
        strict_match_names: Optional[bool] = False,
        allow_add_columns: Optional[bool] = False,
        rename: Optional[bool] = True,
        memory_pool: Optional[pa.MemoryPool] = None,
        source_field: Optional[pa.Field] = None,
        target_field: Optional[pa.Field] = None,
        target_schema: Optional[pa.Schema] = None,
        _target_schema: Optional[pa.Schema] = None,
    ):
        self.safe = False if safe is None else safe
        self.add_missing_columns = True if add_missing_columns is None else add_missing_columns
        self.strict_match_names = False if strict_match_names is None else strict_match_names
        self.allow_add_columns = False if allow_add_columns is None else allow_add_columns
        self.rename = True if rename is None else rename
        self.memory_pool = memory_pool
        self.source_field = source_field
        self.target_field = target_field

        cached_schema = target_schema if _target_schema is None else _target_schema
        if cached_schema is None and target_field is not None:
            cached_schema = pa.schema([target_field])

        self._target_schema = cached_schema

    def __setattr__(self, name, value):
        if name == "target_field":
            object.__setattr__(self, "_target_schema", None)
        object.__setattr__(self, name, value)

    @property
    def target_schema(self) -> Optional[pa.Schema]:
        if self._target_schema is None and self.target_field is not None:
            self._target_schema = pa.schema([self.target_field])
        return self._target_schema

    @target_schema.setter
    def target_schema(self, value: Optional[pa.Schema]):
        self._target_schema = value


DEFAULT_CAST_OPTIONS = ArrowCastOptions()


def cast_arrow_array(
    data: Union[pa.ChunkedArray, pa.Array],
    arrow_type: Union[pa.DataType, pa.Field],
    options: Optional[ArrowCastOptions] = None
):
    options = options or DEFAULT_CAST_OPTIONS

    source_field = options.source_field
    target_field_option = options.target_field

    if isinstance(arrow_type, pa.Field):
        target_field = arrow_type
        target_type = arrow_type.type
        target_nullable = arrow_type.nullable
    elif target_field_option is not None:
        target_field = target_field_option
        target_type = target_field_option.type
        target_nullable = target_field_option.nullable
    else:
        target_field = None
        target_type = arrow_type
        target_nullable = True

    def _cast_single(
        arr: pa.Array,
        target: pa.DataType,
        *,
        nullable: bool,
        source_field: Optional[pa.Field],
    ):
        if pa.types.is_struct(target):
            if not pa.types.is_struct(arr.type) and not pa.types.is_map(arr.type):
                raise pa.ArrowInvalid(f"Cannot cast non-struct array to struct type {target}")

            children = []
            if pa.types.is_struct(arr.type):
                name_to_index = {field.name: idx for idx, field in enumerate(arr.type)}
                folded_to_index = {
                    field.name.casefold(): idx for idx, field in enumerate(arr.type)
                }

                for i, field in enumerate(target):
                    if field.name in name_to_index:
                        child_arr = arr.field(name_to_index[field.name])
                        child_source_field = arr.type[name_to_index[field.name]]
                    elif not options.strict_match_names and field.name.casefold() in folded_to_index:
                        index = folded_to_index[field.name.casefold()]
                        child_arr = arr.field(index)
                        child_source_field = arr.type[index]
                    elif not options.strict_match_names and i < arr.type.num_fields:
                        child_arr = arr.field(i)
                        child_source_field = arr.type[i]
                    elif options.add_missing_columns:
                        child_arr = _default_array(field, len(arr))
                        child_source_field = None
                    else:
                        raise pa.ArrowInvalid(f"Missing field {field.name} while casting struct")

                    children.append(
                        _cast_array(
                            child_arr,
                            field.type,
                            nullable=field.nullable,
                            source_field=child_source_field,
                        )
                    )
            else:
                map_arr = arr
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
                        field.name if options.strict_match_names else field.name.casefold()
                    )
                    values = pc.map_lookup(map_arr, lookup_key, "first")
                    casted = _cast_array(
                        values,
                        field.type,
                        nullable=field.nullable,
                        source_field=None,
                    )

                    if not field.nullable:
                        default_arr = _default_array(field, len(arr))
                        casted = pc.if_else(pc.is_null(casted), default_arr, casted)

                    children.append(casted)

            mask = arr.is_null() if arr.null_count else None
            return pa.StructArray.from_arrays(
                children,
                fields=list(target),
                mask=mask
            )

        if pa.types.is_list(target) or pa.types.is_large_list(target):
            if not pa.types.is_list(arr.type) and not pa.types.is_large_list(arr.type):
                raise pa.ArrowInvalid(f"Cannot cast non-list array to list type {target}")

            list_source_field = None
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
                type=target
            )

        if pa.types.is_map(target):
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
                    type=target
                )

            if not pa.types.is_struct(arr.type):
                raise pa.ArrowInvalid(f"Cannot cast non-map array to map type {target}")

            num_rows = len(arr)
            offsets = [0]
            keys = []
            items = []

            mask = arr.is_null() if arr.null_count else None

            casted_children = [
                    _cast_array(
                        arr.field(i),
                        target.item_type,
                        nullable=True,
                        source_field=arr.type[i],
                    )
                    for i in range(arr.type.num_fields)
                ]

            for row in range(num_rows):
                if mask is not None and mask[row].as_py():
                    offsets.append(len(keys))
                    continue

                for idx, field in enumerate(arr.type):
                    keys.append(field.name)
                    items.append(casted_children[idx][row].as_py())

                offsets.append(len(keys))
            return pa.MapArray.from_arrays(
                pa.array(offsets, type=pa.int64()),
                pa.array(keys, type=target.key_type),
                pa.array(items, type=target.item_type),
                mask=mask,
                type=target,
            )

        return pc.cast(
            arr,
            target_type=target,
            safe=options.safe,
            memory_pool=options.memory_pool
        )

    def _fill_non_nullable_defaults(
        arr: Union[pa.Array, pa.ChunkedArray],
        dtype: pa.DataType,
        *,
        nullable: bool,
        source_field: Optional[pa.Field],
    ):
        if nullable:
            return arr

        if source_field is not None and source_field.nullable is False:
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
            default_value = _default_python_value(dtype)
            default_arr = pa.array([default_value] * len(arr), type=dtype)
            return pc.if_else(pc.is_null(arr), default_arr, arr)

        return arr

    def _cast_array(
        arr: Union[pa.Array, pa.ChunkedArray],
        target: pa.DataType,
        *,
        nullable: bool,
        source_field: Optional[pa.Field],
    ):
        if arr.type.equals(target):
            return _fill_non_nullable_defaults(
                arr,
                target,
                nullable=nullable,
                source_field=source_field,
            )

        if isinstance(arr, pa.ChunkedArray):
            chunks = [
                _cast_array(chunk, target, nullable=nullable, source_field=source_field)
                for chunk in arr.chunks
            ]
            casted = pa.chunked_array(chunks, type=target)
            return _fill_non_nullable_defaults(
                casted, target, nullable=nullable, source_field=source_field
            )

        casted = _cast_single(
            arr,
            target,
            nullable=nullable,
            source_field=source_field,
        )
        return _fill_non_nullable_defaults(
            casted, target, nullable=nullable, source_field=source_field
        )

    return _cast_array(
        data,
        target_type,
        nullable=target_nullable,
        source_field=source_field,
    )


def _resolve_target_schema(
    options: ArrowCastOptions,
    *,
    context: str,
):
    schema = options.target_schema

    if schema is None:
        raise pa.ArrowInvalid(f"Target schema is required for {context}")

    if isinstance(schema, pa.Field):
        options = replace(options, target_field=schema, _target_schema=None)
        schema = pa.schema([schema])
    else:
        options = replace(options, _target_schema=schema)

    return schema, options


def cast_arrow_table(
    data: pa.Table,
    options: Optional[ArrowCastOptions] = None
):
    options = options or DEFAULT_CAST_OPTIONS

    arrow_schema, options = _resolve_target_schema(options, context="casting table")

    exact_name_to_index = {name: idx for idx, name in enumerate(data.column_names)}
    folded_name_to_index = {
        name.casefold(): idx for idx, name in enumerate(data.column_names)
    }

    columns = []
    for field in arrow_schema:
        if field.name in exact_name_to_index:
            column_index = exact_name_to_index[field.name]
            column = data.column(column_index)
            source_field = data.schema.field(column_index)
        elif not options.strict_match_names and field.name.casefold() in folded_name_to_index:
            column_index = folded_name_to_index[field.name.casefold()]
            column = data.column(column_index)
            source_field = data.schema.field(column_index)
        elif not options.strict_match_names and data.num_columns > len(columns):
            column_index = len(columns)
            column = data.column(column_index)
            source_field = data.schema.field(column_index)
        elif options.add_missing_columns:
            column = _default_array(field, data.num_rows)
            source_field = None
        else:
            raise pa.ArrowInvalid(f"Missing column {field.name} while casting table")

        columns.append(
            cast_arrow_array(
                column,
                field,
                replace(
                    options,
                    source_field=source_field,
                    target_field=field,
                ),
            )
        )

    if not options.allow_add_columns and data.num_columns > len(columns):
        data = data.select(range(len(columns)))

    return pa.Table.from_arrays(columns, schema=arrow_schema)


def cast_arrow_batch(
    data: pa.RecordBatch,
    options: Optional[ArrowCastOptions] = None
):
    options = options or DEFAULT_CAST_OPTIONS

    arrow_schema, options = _resolve_target_schema(
        options, context="casting record batch"
    )

    exact_name_to_index = {name: idx for idx, name in enumerate(data.schema.names)}
    folded_name_to_index = {
        name.casefold(): idx for idx, name in enumerate(data.schema.names)
    }

    columns = []
    for field in arrow_schema:
        if field.name in exact_name_to_index:
            column_index = exact_name_to_index[field.name]
            column = data.column(column_index)
            source_field = data.schema.field(column_index)
        elif not options.strict_match_names and field.name.casefold() in folded_name_to_index:
            column_index = folded_name_to_index[field.name.casefold()]
            column = data.column(column_index)
            source_field = data.schema.field(column_index)
        elif not options.strict_match_names and data.num_columns > len(columns):
            column_index = len(columns)
            column = data.column(column_index)
            source_field = data.schema.field(column_index)
        elif options.add_missing_columns:
            column = _default_array(field, data.num_rows)
            source_field = None
        else:
            raise pa.ArrowInvalid(f"Missing column {field.name} while casting record batch")

        columns.append(
            cast_arrow_array(
                column,
                field,
                replace(
                    options,
                    source_field=source_field,
                    target_field=field,
                ),
            )
        )

    if not options.allow_add_columns and data.num_columns > len(columns):
        data = data.slice(0, data.num_rows)

    return pa.RecordBatch.from_arrays(columns, schema=arrow_schema)


def cast_arrow_record_batch_reader(
    data: pa.RecordBatchReader,
    options: Optional[ArrowCastOptions] = None,
):
    options = options or DEFAULT_CAST_OPTIONS

    arrow_schema, options = _resolve_target_schema(
        options, context="casting record batch reader"
    )

    casted_batches = [cast_arrow_batch(batch, options) for batch in data]

    return pa.RecordBatchReader.from_batches(arrow_schema, casted_batches)


def _default_array(field: Union[pa.Field, pa.DataType], length: int):
    if isinstance(field, pa.Field):
        nullable = field.nullable
        dtype = field.type
    else:
        nullable = True
        dtype = field

    if nullable:
        return pa.nulls(length, type=dtype)

    value = _default_python_value(dtype)
    return pa.array([value] * length, type=dtype)


def _default_python_value(dtype: pa.DataType):
    if pa.types.is_struct(dtype):
        return {
            field.name: _default_python_value(field.type)
            if not field.nullable
            else None
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

    if pa.types.is_timestamp(dtype) or pa.types.is_time(dtype) or pa.types.is_duration(dtype) or pa.types.is_interval(dtype):
        return 0

    return None
