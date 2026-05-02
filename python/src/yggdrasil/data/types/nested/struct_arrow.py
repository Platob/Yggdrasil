"""Arrow-side cast helpers for :class:`StructType` targets.

Four entry points, one per source shape:

* :func:`cast_arrow_struct_array` — struct → struct (per-child rebuild,
  missing children defaulted, nested casts threaded through
  ``options.copy(source_field=, target_field=)``).
* :func:`cast_arrow_map_array` — map → struct via ``pc.map_lookup``;
  one lookup per target child.
* :func:`cast_arrow_list_array` — list → struct by positional index;
  out-of-bounds is null.
* :func:`cast_arrow_tabular` — Table/RecordBatch column rebuild against
  the merged schema; missing source columns get defaults.

All four short-circuit on ``options.need_cast`` and rely on the parent
:meth:`StructType._cast_arrow_array` for engine-level dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.types.id import DataTypeId

if TYPE_CHECKING:
    from yggdrasil.data.cast.options import CastOptions
    from yggdrasil.data.data_field import Field
    from .array import ArrayType
    from .map import MapType
    from .struct import StructType


__all__ = [
    "cast_arrow_struct_array",
    "cast_arrow_map_array",
    "cast_arrow_list_array",
    "cast_arrow_tabular",
]


def cast_arrow_struct_array(
    array: pa.StructArray,
    options: "CastOptions",
):
    if not options.need_cast(array):
        return array

    if options.source_field.dtype.type_id != DataTypeId.STRUCT:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    source_field: "Field" = options.source_field
    source_type: "StructType" = source_field.dtype
    target_type: "StructType" = options.target_field.dtype

    children: list[pa.Array] = []

    for i, target_child in enumerate(target_type.children_fields):
        source_child = source_type.field_by(name=target_child.name, index=i, raise_error=False)

        if source_child is None:
            children.append(
                target_child.default_arrow_array(
                    size=len(array),
                    memory_pool=options.arrow_memory_pool,
                )
            )
        else:
            children.append(
                target_child.cast_arrow_array(
                    array.field(source_child.name),
                    options=options.copy(source_field=source_child, target_field=target_child),
                )
            )

    return pa.StructArray.from_arrays(
        children,
        fields=[f.to_arrow_field() for f in target_type.fields],
        mask=array.is_null(),
        memory_pool=options.arrow_memory_pool,
    )


def cast_arrow_map_array(
    array: pa.MapArray,
    options: "CastOptions",
):
    if not options.need_cast(array):
        return array

    if options.source_field.dtype.type_id != DataTypeId.MAP:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    source_field: "Field" = options.source_field
    source_type: "MapType" = source_field.dtype
    target_type: "StructType" = options.target_field.dtype

    children: list[pa.Array] = []

    for target_child in target_type.children_fields:
        values = pc.map_lookup(array, target_child.name, occurrence="first")
        casted = target_child.cast_arrow_array(
            values,
            options=options.copy(
                source_field=source_type.value_field,
                target_field=target_child,
            ),
        )
        children.append(casted)

    return pa.StructArray.from_arrays(
        children,
        fields=[f.to_arrow_field() for f in target_type.fields],
        mask=array.is_null(),
        memory_pool=options.arrow_memory_pool,
    )


def cast_arrow_list_array(
    array: pa.Array | pa.ChunkedArray,
    options: "CastOptions",
):
    if not options.need_cast(array):
        return array

    if options.source_field.dtype.type_id != DataTypeId.ARRAY:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )

    source_field: "Field" = options.source_field
    source_type: "ArrayType" = source_field.dtype
    target_type: "StructType" = options.target_field.dtype

    values_py = array.to_pylist()
    children: list[pa.Array] = []

    for i, target_child in enumerate(target_type.children_fields):
        extracted_py = [
            None if row is None or i >= len(row) else row[i]
            for row in values_py
        ]
        extracted = pa.array(
            extracted_py,
            memory_pool=options.arrow_memory_pool,
        )

        casted = target_child.cast_arrow_array(
            extracted,
            options=options.copy(
                source_field=source_type.item_field,
                target_field=target_child,
            ),
        )
        children.append(casted)

    return pa.StructArray.from_arrays(
        children,
        fields=[f.to_arrow_field() for f in target_type.fields],
        mask=array.is_null() if isinstance(array, pa.Array) else None,
        memory_pool=options.arrow_memory_pool,
    )


def cast_arrow_tabular(
    data: pa.Table | pa.RecordBatch,
    options: "CastOptions",
) -> pa.Table | pa.RecordBatch:
    if not isinstance(data, (pa.Table, pa.RecordBatch)):
        raise TypeError(f"Unsupported tabular type: {type(data)!r}")

    if not options.need_cast(data, check_names=True):
        return data

    source_schema = options.source_schema
    target_schema = options.merged_schema

    target_arrays: list[pa.Array] = []
    num_rows = data.num_rows

    for i, target_field in enumerate(target_schema.children_fields):
        source_field = source_schema.field_by(
            name=target_field.name,
            index=i,
            raise_error=False,
        )

        if source_field is None:
            casted = target_field.default_arrow_array(
                size=num_rows,
                memory_pool=options.arrow_memory_pool,
            )
        else:
            source_array = data.column(source_field.name)
            casted = target_field.cast_arrow_array(
                source_array,
                options=options.copy(
                    source_field=source_field,
                    target_field=target_field,
                ),
            )

        target_arrays.append(casted)

    target_arrow_schema = target_schema.to_arrow_schema()

    if isinstance(data, pa.Table):
        return pa.Table.from_arrays(target_arrays, schema=target_arrow_schema)
    return pa.RecordBatch.from_arrays(target_arrays, schema=target_arrow_schema)
