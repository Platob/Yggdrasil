from dataclasses import dataclass
from typing import Union, Optional

import decimal

import pyarrow as pa
import pyarrow.compute as pc

__all__ = [
    "ArrowCastOptions",
    "cast_arrow_array",
    "cast_arrow_table",
    "cast_arrow_batch"
]


@dataclass
class ArrowCastOptions:
    safe: bool = False
    add_missing_columns: bool = True
    strict_match_names: bool = False
    allow_add_columns: bool =  False
    rename: bool = True
    memory_pool: Optional[pa.MemoryPool] = None

    def __post_init__(self):
        if self.safe is None:
            self.safe = False

        if self.add_missing_columns is None:
            self.add_missing_columns = True

        if self.strict_match_names is None:
            self.strict_match_names = False

        if self.allow_add_columns is None:
            self.allow_add_columns = False

        if self.rename is None:
            self.rename = True


DEFAULT_CAST_OPTIONS = ArrowCastOptions()


def cast_arrow_array(
    data: Union[pa.ChunkedArray, pa.Array],
    arrow_type: pa.DataType,
    options: Optional[ArrowCastOptions] = None
):
    options = options or DEFAULT_CAST_OPTIONS

    def _cast_single(arr: pa.Array, target: pa.DataType):
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
                    elif not options.strict_match_names and field.name.casefold() in folded_to_index:
                        child_arr = arr.field(folded_to_index[field.name.casefold()])
                    elif not options.strict_match_names and i < arr.type.num_fields:
                        child_arr = arr.field(i)
                    elif options.add_missing_columns:
                        child_arr = _default_array(field, len(arr))
                    else:
                        raise pa.ArrowInvalid(f"Missing field {field.name} while casting struct")

                    children.append(_cast_array(child_arr, field.type))
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
                    casted = _cast_array(values, field.type)

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

            values = _cast_array(arr.values, target.value_type)
            mask = arr.is_null() if arr.null_count else None
            return type(arr).from_arrays(
                arr.offsets,
                values,
                mask=mask,
                type=target
            )

        if pa.types.is_map(target):
            if pa.types.is_map(arr.type):
                keys = _cast_array(arr.keys, target.key_type)
                items = _cast_array(arr.items, target.item_type)
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
                _cast_array(arr.field(i), target.item_type)
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

    def _cast_array(arr: Union[pa.Array, pa.ChunkedArray], target: pa.DataType):
        if arr.type.equals(target):
            return arr

        if isinstance(arr, pa.ChunkedArray):
            chunks = [_cast_array(chunk, target) for chunk in arr.chunks]
            return pa.chunked_array(chunks, type=target)

        return _cast_single(arr, target)

    return _cast_array(data, arrow_type)


def cast_arrow_table(
    data: pa.Table,
    arrow_schema: pa.Schema,
    options: Optional[ArrowCastOptions] = None
):
    options = options or DEFAULT_CAST_OPTIONS

    exact_name_to_index = {name: idx for idx, name in enumerate(data.column_names)}
    folded_name_to_index = {
        name.casefold(): idx for idx, name in enumerate(data.column_names)
    }

    columns = []
    for field in arrow_schema:
        if field.name in exact_name_to_index:
            column = data.column(exact_name_to_index[field.name])
        elif not options.strict_match_names and field.name.casefold() in folded_name_to_index:
            column = data.column(folded_name_to_index[field.name.casefold()])
        elif not options.strict_match_names and data.num_columns > len(columns):
            column = data.column(len(columns))
        elif options.add_missing_columns:
            column = _default_array(field, data.num_rows)
        else:
            raise pa.ArrowInvalid(f"Missing column {field.name} while casting table")

        columns.append(cast_arrow_array(column, field.type, options))

    if not options.allow_add_columns and data.num_columns > len(columns):
        data = data.select(range(len(columns)))

    return pa.Table.from_arrays(columns, schema=arrow_schema)


def cast_arrow_batch(
    data: pa.RecordBatch,
    arrow_schema: pa.Schema,
    options: Optional[ArrowCastOptions] = None
):
    options = options or DEFAULT_CAST_OPTIONS

    exact_name_to_index = {name: idx for idx, name in enumerate(data.schema.names)}
    folded_name_to_index = {
        name.casefold(): idx for idx, name in enumerate(data.schema.names)
    }

    columns = []
    for field in arrow_schema:
        if field.name in exact_name_to_index:
            column = data.column(exact_name_to_index[field.name])
        elif not options.strict_match_names and field.name.casefold() in folded_name_to_index:
            column = data.column(folded_name_to_index[field.name.casefold()])
        elif not options.strict_match_names and data.num_columns > len(columns):
            column = data.column(len(columns))
        elif options.add_missing_columns:
            column = _default_array(field, data.num_rows)
        else:
            raise pa.ArrowInvalid(f"Missing column {field.name} while casting record batch")

        columns.append(cast_arrow_array(column, field.type, options))

    if not options.allow_add_columns and data.num_columns > len(columns):
        data = data.slice(0, data.num_rows)

    return pa.RecordBatch.from_arrays(columns, schema=arrow_schema)


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
