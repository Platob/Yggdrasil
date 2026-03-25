"""
Arrow casting helpers for arrays, tables, and schemas.

This module provides a comprehensive casting layer on top of PyArrow, designed for
commodity trading and data pipeline workloads where schemas evolve across systems
(Spark, Polars, Pandas) and strict type enforcement is required.

Architecture
------------
The module is organized around a converter registry pattern (see :mod:`.registry`).
Every public function is decorated with ``@register_converter(source_type, target_type)``
so the registry can dispatch dynamically at runtime.  Direct calls work too — the
decorators are purely additive.

Casting hierarchy
~~~~~~~~~~~~~~~~~
::

    cast_arrow_array          <- main entry point for column-level casting
    ├── cast_to_struct_array  <- struct targets
    │   ├── arrow_struct_to_struct_array  (struct  → struct)
    │   └── arrow_map_to_struct_array     (map     → struct)
    ├── cast_to_list_arrow_array          (list / large_list targets)
    ├── cast_to_map_array                 (map targets)
    │   ├── arrow_map_to_map_array        (map     → map)
    │   └── arrow_struct_to_map_array     (struct  → map)
    └── cast_primitive_array              (scalars, timestamps via strptime)

    cast_arrow_tabular        <- table / record-batch level casting
    cast_arrow_record_batch_reader  <- lazy streaming cast

Nullability enforcement
~~~~~~~~~~~~~~~~~~~~~~~
``check_arrow_array_nullability`` is called after every cast.  For non-nullable
target fields it replaces any remaining nulls with type-appropriate defaults
produced by :func:`~.python_defaults.default_arrow_array`.

Type normalisation helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~
``to_spark_arrow_type`` and ``to_polars_arrow_type`` strip or transform types that
are not supported by the respective engine (e.g. ``large_string`` → ``string`` for
Spark; ``map<k,v>`` → ``list<struct<key,value>>`` for Polars).

``merge_arrow_fields`` / ``merge_arrow_types`` (defined in
:mod:`~.python_arrow` and re-exported here) implement a non-destructive merge
strategy: widen types rather than narrow them, preserve nullability, and recurse
into nested containers.

Public API
----------
.. autofunction:: cast_arrow_array
.. autofunction:: cast_arrow_tabular
.. autofunction:: cast_arrow_record_batch_reader
.. autofunction:: any_to_arrow_table
.. autofunction:: any_to_arrow_record_batch
.. autofunction:: any_to_arrow_scalar
.. autofunction:: any_to_arrow_field
.. autofunction:: any_to_arrow_schema
.. autofunction:: merge_arrow_fields
.. autofunction:: merge_arrow_types
.. autofunction:: to_spark_arrow_type
.. autofunction:: to_polars_arrow_type
"""

import dataclasses
import enum
import logging
from dataclasses import is_dataclass
from typing import Optional, Union, List, Tuple, Any, Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types as pat

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.cast.registry import register_converter
from yggdrasil.dataclasses.dataclass import dataclass_to_arrow_field
from yggdrasil.pickle.serde import ObjectSerde
from .python_arrow import (
    is_arrow_type_list_like,
    is_arrow_type_string_like,
    is_arrow_type_binary_like,
    merge_arrow_fields,
    merge_arrow_types,
)
from .python_defaults import default_arrow_scalar, default_arrow_array

__all__ = [
    "ArrowDataType",
    "cast_arrow_array",
    "cast_arrow_tabular",
    "cast_arrow_record_batch_reader",
    "default_arrow_array",
    "to_spark_arrow_type",
    "to_polars_arrow_type",
    "arrow_field_to_schema",
    "arrow_type_to_field",
    "is_arrow_type_binary_like",
    "is_arrow_type_string_like",
    "is_arrow_type_list_like",
    "record_batch_to_table",
    "arrow_schema_to_field",
    "arrow_field_to_field",
    "arrow_schema_to_schema",
    "any_to_arrow_scalar",
    "any_to_arrow_field",
    "any_to_arrow_table",
    "any_to_arrow_record_batch",
    "any_to_arrow_schema",
    "arrow_field_to_dict",
    "arrow_type_to_dict",
    "dict_to_arrow_type",
    "dict_to_arrow_field",
    "default_arrow_scalar",
    "merge_arrow_fields",
    "merge_arrow_types",
]

logger = logging.getLogger(__name__)

#: Union alias covering all Arrow DataType variants used throughout this module.
ArrowDataType = Union[
    pa.DataType,
    pa.Decimal128Type,
    pa.TimestampType,
    pa.DictionaryType,
    pa.MapType,
    pa.StructType,
    pa.FixedSizeListType,
]


# ---------------------------------------------------------------------------
# Struct casting
# ---------------------------------------------------------------------------

def cast_to_struct_array(
    array: Union[pa.Array, pa.StructArray, pa.MapArray],
    options: Optional[CastOptions] = None,
) -> pa.StructArray:
    """Dispatch cast of *array* to a struct Arrow array.

    Acts as a router: delegates to the appropriate specialised function based on
    the source type (struct-to-struct or map-to-struct).  Handles
    :class:`~pyarrow.ChunkedArray` by casting each chunk individually and
    reassembling.

    Args:
        array: Source array.  Accepted types: ``StructArray``, ``MapArray``,
            or a ``ChunkedArray`` containing either.
        options: Cast configuration including source/target field descriptors,
            nullability settings, and memory pool.  Defaults are inferred when
            ``None``.

    Returns:
        A ``pa.StructArray`` matching the target field described in *options*.

    Raises:
        ValueError: If the source type is not struct-like or map-like.
    """
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
        raise ValueError(
            f"Cannot cast {options.source_field} to {options.target_field}"
        )


@register_converter(pa.MapArray, pa.StructArray)
def arrow_map_to_struct_array(
    array: pa.MapArray,
    options: Optional[CastOptions] = None,
) -> pa.StructArray:
    """Cast a ``MapArray`` to a ``StructArray``.

    Looks up each target struct field by name via
    :func:`pyarrow.compute.map_lookup` (returning the first matching value per
    row).  Missing map keys produce null values for the corresponding struct
    field — nullability rules are then enforced by
    :func:`cast_arrow_array`.

    Args:
        array: Source ``MapArray``.
        options: Cast options.  ``options.target_field.type`` must be a struct type.

    Returns:
        A ``pa.StructArray`` with one child per target struct field.

    Note:
        Chunked arrays are handled by the caller (:func:`cast_to_struct_array`).
    """
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    source_field = options.source_field
    target_type: pa.StructType = options.target_field.type

    # Propagate null mask from the map array to the resulting struct.
    mask = (
        array.is_null()
        if source_field.nullable and options.target_field.nullable
        else None
    )
    children: List[pa.Array] = []

    map_type: pa.MapType = array.type

    for child_target_field in target_type:
        # Extract all values associated with this key across all rows.
        values = pc.map_lookup(array, child_target_field.name, "first")  # type: ignore

        casted = cast_arrow_array(
            values,
            options.copy(
                source_field=map_type.item_field,
                target_field=child_target_field,
            ),
        )
        children.append(casted)

    return pa.StructArray.from_arrays(
        children,
        fields=list(target_type),
        mask=mask,
        memory_pool=options.arrow_memory_pool,
    )  # type: ignore


@register_converter(pa.StructArray, pa.StructArray)
def arrow_struct_to_struct_array(
    array: pa.StructArray,
    options: Optional[CastOptions] = None,
) -> pa.StructArray:
    """Cast a ``StructArray`` to a (potentially different-schema) ``StructArray``.

    Field matching strategy (in priority order):

    1. **Exact name** match.
    2. **Case-insensitive** name match (when ``options.strict_match_names`` is
       ``False``).
    3. **Positional** fallback — source field at the same index (when
       ``strict_match_names`` is ``False``).
    4. **Default value** column — synthesised via
       :func:`~.python_defaults.default_arrow_array` (when
       ``options.add_missing_columns`` is ``True``).

    Args:
        array: Source ``StructArray``.
        options: Cast options.

    Returns:
        A ``pa.StructArray`` conforming to the target field schema.

    Raises:
        pa.ArrowInvalid: If a required target field is absent and
            ``options.add_missing_columns`` is ``False``.
    """
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    target_type: pa.StructType = options.target_field.type

    mask = (
        array.is_null()
        if options.source_field.nullable and options.target_field.nullable
        else None
    )
    children: List[pa.Array] = []

    # Build lookup maps for O(1) field resolution.
    name_to_index = {field.name: idx for idx, field in enumerate(array.type)}
    folded_to_index = {
        field.name.casefold(): idx for idx, field in enumerate(array.type)
    }

    for i, child_target_field in enumerate(target_type):
        child_target_field: pa.Field = child_target_field

        if child_target_field.name in name_to_index:
            # Strategy 1: exact match.
            child_idx = name_to_index[child_target_field.name]
            child_arr = array.field(child_idx)
            child_source_field = array.type[child_idx]

        elif (
            not options.strict_match_names
            and child_target_field.name.casefold() in folded_to_index
        ):
            # Strategy 2: case-insensitive match.
            child_idx = folded_to_index[child_target_field.name.casefold()]
            child_arr = array.field(child_idx)
            child_source_field = array.type[child_idx]

        elif not options.strict_match_names and i < array.type.num_fields:
            # Strategy 3: positional fallback.
            child_idx = i
            child_arr = array.field(child_idx)
            child_source_field = array.type[child_idx]

        elif options.add_missing_columns:
            # Strategy 4: synthesise a default-valued column.
            child_arr = default_arrow_array(
                dtype=child_target_field.type,
                nullable=child_target_field.nullable,
                size=len(array),
                memory_pool=options.arrow_memory_pool,
            )
            child_source_field = child_target_field

        else:
            raise pa.ArrowInvalid(
                "Missing struct field %s while casting; available fields: %s"
                % (child_target_field.name, list(name_to_index.keys()))
            )

        children.append(
            cast_arrow_array(
                child_arr,
                options.copy(
                    source_field=child_source_field,
                    target_field=child_target_field,
                ),
            )
        )

    return pa.StructArray.from_arrays(
        children,
        fields=list(target_type),
        mask=mask,
        memory_pool=options.arrow_memory_pool,
    )  # type: ignore


# ---------------------------------------------------------------------------
# List casting
# ---------------------------------------------------------------------------

def cast_to_list_arrow_array(
    array: Union[pa.Array, pa.ListArray, pa.LargeListArray],
    options: Optional[CastOptions] = None,
) -> Union[pa.ListArray, pa.LargeListArray]:
    """Cast *array* to a list, large-list, or fixed-size-list Arrow array.

    Supports the following source-to-target combinations:

    * ``list`` / ``large_list`` → ``list``, ``large_list``, or
      ``fixed_size_list`` (offsets are preserved / reused).
    * ``string`` / ``large_string`` → any list type: the string values are
      JSON-decoded via Polars before further casting.

    The value array is recursively cast using :func:`cast_arrow_array`.

    Args:
        array: Source array.
        options: Cast options.  ``options.target_field.type`` must be a list-like
            Arrow type.

    Returns:
        A ``ListArray``, ``LargeListArray``, or ``FixedSizeListArray`` depending
        on the target type.

    Raises:
        pa.ArrowInvalid: If the source type is neither list-like nor string-like,
            or if JSON decoding of string values fails.
        pa.ArrowInvalid: If the target type is an unrecognised list variant.
    """
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

    # --- JSON string decoding path ---
    if is_arrow_type_string_like(source_field.type):
        import polars

        try:
            # Replace empty strings with null before JSON decode so they
            # become null list entries rather than parse errors.
            array = (
                polars.from_arrow(array)
                .replace("", None)
                .str.json_decode()
                .to_arrow()
            )
        except Exception as e:
            raise pa.ArrowInvalid(
                "Failed to parse JSON strings in list cast from %s to %s: %s"
                % (source_field, target_field, e)
            )

        if source_field:
            source_field = options.source_field = pa.field(
                name=source_field.name,
                type=array.type,
                nullable=source_field.nullable,
                metadata=source_field.metadata,
            )

    mask = (
        array.is_null()
        if source_field.nullable and target_field.nullable
        else None
    )

    if is_arrow_type_list_like(source_field.type):
        list_source_field = array.type.value_field
        offsets = array.offsets

        # Recursively cast the flat values buffer.
        values = cast_arrow_array(
            array.values,
            options.copy(
                source_field=list_source_field,
                target_field=target_type.value_field,
            ),
        )
    else:
        raise pa.ArrowInvalid(
            "Unsupported list cast from %s to %s" % (source_field, target_field)
        )

    # Reconstruct the correct list container type.
    if pa.types.is_list(target_type):
        return pa.ListArray.from_arrays(
            offsets, values, type=target_type, mask=mask  # type: ignore
        )  # type: ignore
    if pa.types.is_large_list(target_type):
        return pa.LargeListArray.from_arrays(
            offsets, values, type=target_type, mask=mask  # type: ignore
        )  # type: ignore
    elif pa.types.is_fixed_size_list(target_type):
        return pa.FixedSizeListArray.from_arrays(
            values,
            list_size=target_type.list_size,
            type=target_type,  # type: ignore
            mask=mask,
        )  # type: ignore
    else:
        raise pa.ArrowInvalid(
            f"Cannot build arrow array for target list type {target_type}"
        )


# ---------------------------------------------------------------------------
# Map casting
# ---------------------------------------------------------------------------

def cast_to_map_array(
    array: Union[pa.Array, pa.MapArray, pa.StructArray],
    options: Optional[CastOptions] = None,
) -> pa.MapArray:
    """Dispatch cast of *array* to a ``MapArray``.

    Routes to :func:`arrow_map_to_map_array` (map → map) or
    :func:`arrow_struct_to_map_array` (struct → map) based on the source type.
    Handles ``ChunkedArray`` inputs by casting each chunk individually.

    Args:
        array: Source array (``MapArray``, ``StructArray``, or a chunked version
            of either).
        options: Cast options.

    Returns:
        A ``pa.MapArray`` conforming to the target field type.

    Raises:
        pa.ArrowInvalid: For unsupported source types.
    """
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

    if pa.types.is_map(source_field.type):
        return arrow_map_to_map_array(array, options)
    elif pa.types.is_struct(source_field.type):
        return arrow_struct_to_map_array(array, options)
    else:
        raise pa.ArrowInvalid(
            "Unsupported map cast from %s to %s" % (source_field, target_field)
        )


@register_converter(pa.MapArray, pa.MapArray)
def arrow_map_to_map_array(
    array: pa.MapArray,
    options: Optional[CastOptions] = None,
) -> pa.MapArray:
    """Cast a ``MapArray`` to a (possibly different-typed) ``MapArray``.

    Recursively casts the flat *keys* and *items* buffers while preserving the
    original row offsets.

    Args:
        array: Source ``MapArray``.
        options: Cast options.

    Returns:
        A ``pa.MapArray`` with keys and values cast to the target key/item types.
    """
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    source_field = options.source_arrow_field
    target_field = options.target_arrow_field
    target_type: pa.MapType = target_field.type

    mask = (
        array.is_null()
        if source_field.nullable and target_field.nullable
        else None
    )

    keys = cast_arrow_array(
        array.keys,
        options.copy(
            source_field=array.type.key_field,
            target_field=target_type.key_field,
        ),
    )
    items = cast_arrow_array(
        array.items,
        options.copy(
            source_field=array.type.item_field,
            target_field=target_type.item_field,
        ),
    )

    return pa.MapArray.from_arrays(
        array.offsets,
        keys,
        items,
        mask=mask,
        type=target_type,  # type: ignore
        pool=options.arrow_memory_pool,  # type: ignore
    )  # type: ignore


@register_converter(pa.StructArray, pa.MapArray)
def arrow_struct_to_map_array(
    array: pa.StructArray,
    options: Optional[CastOptions] = None,
) -> pa.MapArray:
    """Cast a ``StructArray`` to a ``MapArray``.

    Each struct *field name* becomes a map *key* (always ``pa.string()``), and
    each field *value* is cast to the target map item type.  Null struct rows
    produce null map entries.

    This is an O(n_rows × n_fields) operation because offsets are built in pure
    Python.  Prefer pre-aggregation before calling this on large arrays.

    Args:
        array: Source ``StructArray``.
        options: Cast options.  ``options.target_field.type`` must be a map type
            whose item type is compatible with all struct field values.

    Returns:
        A ``pa.MapArray`` with ``string`` keys and item values cast to the target
        map item type.
    """
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

    # Pre-cast all children to the target item type in bulk (vectorised).
    casted_children = [
        cast_arrow_array(
            array.field(i),
            options.copy(
                source_field=array.type[i],
                target_field=target_type.item_field,
            ),
        )
        for i in range(array.type.num_fields)
    ]

    for row_idx in range(num_rows):
        if mask is not None and mask[row_idx].as_py():
            # Null row: advance offset without adding any entries.
            offsets.append(offsets[-1])
            continue

        for field_idx, field in enumerate(array.type):
            keys.append(field.name)
            items.append(casted_children[field_idx][row_idx])

        offsets.append(len(keys))

    map_type = pa.map_(pa.string(), target_type.item_type, keys_sorted=False)

    return pa.MapArray.from_arrays(
        offsets,
        pa.array(keys, type=pa.string()),
        pa.array(items, type=target_type.item_type),
        mask=mask,
        type=map_type,  # type: ignore
        pool=options.arrow_memory_pool,  # type: ignore
    )  # type: ignore


# ---------------------------------------------------------------------------
# Primitive casting
# ---------------------------------------------------------------------------

def cast_primitive_array(
    array: Union[pa.ChunkedArray, pa.Array],
    options: CastOptions | None = None,
) -> pa.Array:
    """Cast a primitive (non-nested) Arrow array to the target type.

    For most types this delegates directly to :func:`pyarrow.compute.cast`.
    The special case of ``string → timestamp`` is routed through
    :func:`arrow_strptime` which supports multiple format patterns defined in
    ``options.datetime_patterns``.

    Args:
        array: Source array of a primitive (scalar) Arrow type.
        options: Cast options.

    Returns:
        Cast array with nullability enforced by
        :func:`check_arrow_array_nullability`.
    """
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    source_field = options.source_arrow_field
    target_field = options.target_arrow_field

    if is_arrow_type_string_like(source_field.type) and pa.types.is_timestamp(
        target_field.type
    ):
        return arrow_strptime(array, options)
    else:
        try:
            casted = pc.cast(
                array,
                target_type=target_field.type,
                safe=options.safe,
                memory_pool=options.arrow_memory_pool,
            )
        except pa.ArrowNotImplementedError as e:
            raise pa.ArrowInvalid(
                f"Unsupported cast from {source_field} to {target_field}"
            ) from e
        return check_arrow_array_nullability(casted, options)


# ---------------------------------------------------------------------------
# Nullability enforcement
# ---------------------------------------------------------------------------

def check_arrow_array_nullability(
    array: Union[pa.Array, pa.ChunkedArray],
    options: Optional[CastOptions] = None,
) -> Union[pa.Array, pa.ChunkedArray]:
    """Fill nulls for non-nullable target fields.

    If the target field is non-nullable, any remaining null values in *array*
    are replaced with type-appropriate defaults from
    :func:`~.python_defaults.default_arrow_array` using
    :func:`pyarrow.compute.if_else`.

    Short-circuits early (returning *array* unchanged) when:

    * The cast options indicate no nullability check is needed
      (``options.need_nullability_check`` returns ``False``).
    * The array has zero nulls.

    ``ChunkedArray`` inputs are handled chunk-by-chunk so no data is
    materialised into a single buffer unless necessary.

    Args:
        array: Array potentially containing nulls.
        options: Cast options carrying target field metadata.

    Returns:
        The original array (if no nulls / not needed) or a null-filled copy.
    """
    options = CastOptions.check_arg(options)

    if not options.need_nullability_fill(source_obj=array):
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
        return array
    else:
        default_arr = default_arrow_array(
            options.target_field.type,
            nullable=options.target_field.nullable,
            size=len(array),
        )
        return pc.if_else(pc.is_null(array), default_arr, array)  # type: ignore


# ---------------------------------------------------------------------------
# Any-to-Arrow table / record batch
# ---------------------------------------------------------------------------

@register_converter(Any, pa.Table)
def any_to_arrow_table(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """Convert *any* supported object to a ``pa.Table``, then cast to the target schema.

    Supported input types (detected via namespace inspection):

    * ``pa.Table`` — passed through to :func:`cast_arrow_tabular`.
    * ``pa.RecordBatch`` — wrapped in a single-batch table first.
    * ``pandas.DataFrame`` — via
      :func:`~.pandas.cast.pandas_dataframe_to_arrow_table`.
    * ``pyspark.sql.DataFrame`` — converted via Spark's ``toArrow()`` method.
    * Everything else (Polars, dicts, dataclasses, …) — routed through
      :func:`~.polars.cast.any_to_polars_dataframe` then
      :func:`~.polars.cast.polars_dataframe_to_arrow_table`.

    Args:
        obj: Any object that can be converted to an Arrow table.
        options: Cast options including the target schema.

    Returns:
        A ``pa.Table`` cast to ``options.target_arrow_schema`` (if set).
    """
    if not isinstance(obj, pa.Table):
        if isinstance(obj, pa.RecordBatch):
            obj = pa.Table.from_batches([obj])  # type: ignore
        else:
            namespace = ObjectSerde.full_namespace(obj)

            if namespace.startswith("pandas."):
                from yggdrasil.pandas.cast import pandas_dataframe_to_arrow_table
                obj = pandas_dataframe_to_arrow_table(obj, options)

            if namespace.startswith("pyspark."):
                from yggdrasil.spark.lib import pyspark_sql
                from yggdrasil.spark.cast import any_to_spark_dataframe

                obj: pyspark_sql.DataFrame = any_to_spark_dataframe(obj, options)
                obj = obj.toArrow()
            else:
                from yggdrasil.polars.cast import any_to_polars_dataframe, polars_dataframe_to_arrow_table
                obj = any_to_polars_dataframe(obj, options)
                obj = polars_dataframe_to_arrow_table(obj, options)

    return cast_arrow_tabular(obj, options)


@register_converter(Any, pa.RecordBatch)
def any_to_arrow_record_batch(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatch:
    """Convert *any* supported object to a single ``pa.RecordBatch``.

    Delegates to :func:`any_to_arrow_table` and then extracts the first batch.

    Args:
        obj: Any object convertible to an Arrow table.
        options: Cast options.

    Returns:
        The first ``pa.RecordBatch`` from the resulting table.
    """
    if not isinstance(obj, pa.RecordBatch):
        obj: pa.Table = any_to_arrow_table(obj, options)
        return obj.to_batches()[0]

    return cast_arrow_tabular(obj, options)


# ---------------------------------------------------------------------------
# Scalar casting
# ---------------------------------------------------------------------------

@register_converter(Any, pa.Scalar)
def any_to_arrow_scalar(
    scalar: Any,
    options: Optional[CastOptions] = None,
) -> pa.Scalar:
    """Convert a Python value to an Arrow scalar, then cast to the target field type.

    Conversion rules:

    * ``None`` → type-appropriate default scalar (respects nullability).
    * ``enum.Enum`` → the enum's ``.value`` attribute is used.
    * dataclass → converted to a ``dict`` via ``dataclasses.asdict``, then to a
      struct scalar.  If no target field is set, the field is inferred from the
      dataclass definition via :func:`~.dataclasses.dataclass.dataclass_to_arrow_field`.
    * All other values → ``pa.scalar(value, type=target_field.type)`` with a
      fallback to untyped ``pa.scalar(value)`` on ``ArrowInvalid``.

    Args:
        scalar: Input Python value.
        options: Cast options.  ``options.target_field`` determines the output type.

    Returns:
        An Arrow scalar cast to the target field type.
    """
    if not isinstance(scalar, pa.Scalar):
        options = CastOptions.check_arg(options)
        target_field = options.target_field

        if scalar is None:
            return default_arrow_scalar(
                target_field,
                nullable=True if target_field is None else target_field.nullable,
            )

        if isinstance(scalar, enum.Enum):
            scalar = scalar.value

        if is_dataclass(scalar):
            if not target_field:
                target_field = dataclass_to_arrow_field(scalar)
                options = options.copy(target_field=target_field)
            scalar = dataclasses.asdict(scalar)

        if target_field is None:
            if is_dataclass(scalar):
                scalar = pa.scalar(
                    dataclasses.asdict(scalar),
                    type=dataclass_to_arrow_field(scalar).type,
                )
            else:
                scalar = pa.scalar(scalar)
            return scalar

        try:
            scalar = pa.scalar(scalar, type=target_field.type)
        except pa.ArrowInvalid:
            # Fall back to untyped scalar; cast_arrow_scalar will handle the type.
            scalar = pa.scalar(scalar)

    return cast_arrow_scalar(scalar, options)


@register_converter(pa.Scalar, pa.Scalar)
def cast_arrow_scalar(
    scalar: pa.Scalar,
    options: Optional[CastOptions] = None,
) -> pa.Scalar:
    """Cast an existing Arrow scalar to the type described in *options*.

    Wraps the scalar in a single-element array, delegates to
    :func:`cast_arrow_array`, and returns the first element.  This ensures
    all casting logic is centralised in the array path.

    Args:
        scalar: Arrow scalar to cast.
        options: Cast options.  If ``options.target_field`` is ``None`` the
            scalar is returned unchanged.

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


# ---------------------------------------------------------------------------
# Main array cast dispatcher
# ---------------------------------------------------------------------------

@register_converter(pa.Array, pa.Array)
@register_converter(pa.ChunkedArray, pa.ChunkedArray)
def cast_arrow_array(
    array: Union[pa.ChunkedArray, pa.Array],
    options: Optional[CastOptions] = None,
) -> Union[pa.ChunkedArray, pa.Array]:
    """Cast an Arrow array or chunked array to the type described in *options*.

    This is the primary column-level entry point.  It inspects the target type
    and routes to the appropriate specialist:

    * **null source** — direct ``pc.cast`` then nullability check.
    * **struct target** — :func:`cast_to_struct_array`.
    * **list / large_list / fixed_size_list target** — :func:`cast_to_list_arrow_array`.
    * **map target** — :func:`cast_to_map_array`.
    * **all other (primitive) targets** — :func:`cast_primitive_array`.

    If ``options.need_arrow_type_cast`` returns ``False`` (types already match),
    only nullability is enforced via :func:`check_arrow_array_nullability`.

    Args:
        array: Column to cast.  May be a plain ``pa.Array`` or a
            ``pa.ChunkedArray``; nested cast functions handle chunking
            internally.
        options: Cast options carrying source/target field descriptors,
            safety flags, and memory pool reference.

    Returns:
        Cast array (same container type as input where possible).

    Raises:
        ValueError: For unsupported nested target types.
    """
    options = CastOptions.check_arg(options)

    if not options.need_arrow_type_cast(source_obj=array):
        return check_arrow_array_nullability(array, options)

    source_type = options.source_field.type
    target_type = options.target_field.type

    if pa.types.is_null(source_type):
        # Source is all-null: cast and let nullability check fill in defaults.
        return check_arrow_array_nullability(
            pc.cast(
                array,
                target_type,
                safe=False,
                memory_pool=options.arrow_memory_pool,
            ),
            options=options,
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


# ---------------------------------------------------------------------------
# Timestamp string parsing
# ---------------------------------------------------------------------------

def arrow_strptime(
    arr: Union[pa.Array, pa.ChunkedArray, pa.StringArray],
    options: Optional[CastOptions] = None,
) -> Union[pa.TimestampArray, pa.ChunkedArray]:
    """Parse a string Arrow array into timestamps.

    Tries each pattern in ``options.datetime_patterns`` in order, stopping at
    the first that succeeds.  If no patterns are set, falls back to a plain
    ``pc.cast`` (which defers to Arrow's ISO-8601 parser).

    Chunked arrays are processed chunk-by-chunk to avoid materialisation.

    Args:
        arr: String array whose values represent timestamps.
        options: Cast options.  Must satisfy:

            * ``options.target_field`` is a timestamp field.
            * ``options.datetime_patterns`` is a list of ``strptime``-compatible
              format strings (e.g. ``["%Y-%m-%d", "%d/%m/%Y"]``).

    Returns:
        A ``TimestampArray`` (or ``ChunkedArray[timestamp]`` for chunked input).

    Raises:
        ValueError: If ``options.target_field`` is not a timestamp type.
        Exception: The last parsing error encountered if all patterns fail.
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
        raise ValueError(
            "arrow_strptime requires target_field to be a timestamp type"
        )

    if isinstance(arr, pa.ChunkedArray):
        casted_chunks = [arrow_strptime(chunk, options) for chunk in arr.chunks]
        return pa.chunked_array(casted_chunks, type=target_field.type)

    patterns = options.datetime_patterns

    if not patterns:
        # No explicit patterns: rely on Arrow's ISO-8601 / RFC-3339 parser.
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
                casted = pc.strptime(  # type: ignore
                    arr,
                    format=pattern,
                    unit=target_field.type.unit,
                    error_is_null=not options.safe,
                    memory_pool=options.arrow_memory_pool,
                )
                break  # First successful pattern wins.
            except Exception as e:
                last_error = e

        if casted is None:
            logger.error(
                "Failed to parse timestamps with provided patterns %s for target %s; last error: %s",
                patterns,
                target_field,
                last_error,
            )
            raise last_error if last_error else ValueError(
                "Failed to parse timestamps with provided patterns"
            )

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
    """Cast a ``pa.Table`` or ``pa.RecordBatch`` to a target schema.

    Column matching uses the same priority order as
    :func:`arrow_struct_to_struct_array`:

    1. Exact name match.
    2. Case-insensitive name match (when ``options.strict_match_names`` is
       ``False``).
    3. Synthesised default column (when ``options.add_missing_columns`` is
       ``True``).

    Extra source columns (not referenced by the target schema) are silently
    dropped *unless* ``options.allow_add_columns`` is ``True``, in which case
    they are appended to the output.

    Empty tables are handled by returning a zero-row table with the correct
    target schema immediately (no column-level casting needed).

    Args:
        data: Source table or record batch.
        options: Cast options.  If ``options.target_arrow_schema`` is ``None``
            the input is returned unchanged.

    Returns:
        Table / RecordBatch with columns cast and reordered to match the target
        schema.

    Raises:
        pa.ArrowInvalid: If a required column is absent and
            ``options.add_missing_columns`` is ``False``.
    """
    options = CastOptions.check_arg(options)
    target_arrow_schema = options.target_arrow_schema

    if target_arrow_schema is None:
        return data

    if data.num_rows == 0:
        # Fast path: return correctly-typed empty container.
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
            schema=target_arrow_schema,
        )

    source_arrow_schema: pa.Schema | Iterable[pa.Field] = data.schema

    if source_arrow_schema == target_arrow_schema:
        return data

    # Build a combined lookup: exact names + case-folded aliases.
    source_name_to_index = {
        field.name: idx for idx, field in enumerate(source_arrow_schema)
    }
    if not options.strict_match_names:
        source_name_to_index.update(
            {
                field.name.casefold(): idx
                for idx, field in enumerate(source_arrow_schema)
            }
        )

    # Detect existing chunk boundaries to produce aligned output chunks.
    chunks = None
    if isinstance(data, pa.Table) and data.num_columns > 0:
        first_col = data.column(0)
        if isinstance(first_col, pa.ChunkedArray):
            chunks = [len(chunk) for chunk in first_col.chunks]

    casted_columns: List[Tuple[pa.Field, pa.ChunkedArray]] = []
    found_source_names: set = set()

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
                raise pa.ArrowInvalid(
                    f"Missing column {target_field.name!r} in source data "
                    f"{source_arrow_schema.names}"
                )

            # Synthesise a default-valued column.
            array = default_arrow_array(
                dtype=target_field.type,
                nullable=target_field.nullable,
                size=data.num_rows,
                memory_pool=options.arrow_memory_pool,
                chunks=chunks,
            )
        else:
            source_field = source_arrow_schema.field(source_index)
            found_source_names.add(source_field.name)
            array = cast_arrow_array(
                data.column(source_index),
                options.copy(
                    source_field=source_field,
                    target_field=target_field,
                ),
            )

        casted_columns.append((target_field, array))

    if options.allow_add_columns:
        # Append source columns not consumed by the target schema.
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
                metadata=target_arrow_schema.metadata,
            )

    all_arrays = [array for _, array in casted_columns]
    return data.__class__.from_arrays(all_arrays, schema=target_arrow_schema)


@register_converter(pa.RecordBatchReader, pa.RecordBatchReader)
def cast_arrow_record_batch_reader(
    data: pa.RecordBatchReader,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """Lazily wrap a ``RecordBatchReader`` so each emitted batch is cast on-the-fly.

    No data is read or materialised until the returned reader is iterated.
    This is the preferred entry point for streaming workloads (e.g. reading
    Parquet files in Arrow IPC streams) where full materialisation is
    undesirable.

    Args:
        data: Source reader.
        options: Cast options including the target schema.  If
            ``options.target_arrow_schema`` is ``None`` the original reader is
            returned as-is.

    Returns:
        A ``pa.RecordBatchReader`` whose schema matches the target and whose
        batches are lazily cast.
    """
    options = CastOptions.check_arg(options)
    arrow_schema = options.target_arrow_schema

    if arrow_schema is None:
        return data

    def casted_batches(opt=options):
        """Lazily yield cast batches from the upstream reader."""
        for batch in data:
            yield cast_arrow_tabular(batch, opt)

    return pa.RecordBatchReader.from_batches(arrow_schema, casted_batches())  # type: ignore


# ---------------------------------------------------------------------------
# Type normalisation helpers
# ---------------------------------------------------------------------------

def to_spark_arrow_type(dtype: ArrowDataType) -> ArrowDataType:
    """Normalise an Arrow ``DataType`` to a Spark-compatible equivalent.

    Spark does not support several Arrow type variants.  This function
    recursively replaces them:

    * ``large_string`` / ``large_binary`` → ``string`` / ``binary``
    * ``large_list<T>`` → ``list<T>`` (value type also normalised)
    * ``dictionary<index, value>`` → ``value`` type (Spark resolves categories)
    * ``ExtensionType`` → ``storage_type`` (unwrap custom extensions)
    * ``struct`` → ``struct`` with each child field normalised
    * ``map`` → ``map`` with key and item types normalised

    Args:
        dtype: Arrow data type to normalise.

    Returns:
        A Spark-compatible Arrow data type.
    """
    if is_arrow_type_string_like(dtype):
        return pa.string()
    if is_arrow_type_binary_like(dtype):
        return pa.binary()
    if is_arrow_type_list_like(dtype):
        return pa.list_(to_spark_arrow_type(dtype.value_type))
    if pa.types.is_dictionary(dtype):
        return to_spark_arrow_type(dtype.value_type)
    if isinstance(dtype, pa.ExtensionType):
        return to_spark_arrow_type(dtype.storage_type)
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

    return dtype


def to_polars_arrow_type(dtype: ArrowDataType) -> ArrowDataType:
    """Normalise an Arrow ``DataType`` to a Polars-compatible equivalent.

    Extends :func:`to_spark_arrow_type` with one additional rule required by
    Polars: ``map<k,v>`` is not a first-class type in Polars' data model and
    must be represented as ``list<struct<key: K, value: V>>``.

    Transformation summary:

    * All Spark normalisation rules apply first.
    * ``map<K, V>`` → ``list<struct<key: K, value: V>>``
    * ``struct`` children are recursively normalised.
    * ``list`` element types are recursively normalised.

    Args:
        dtype: Arrow data type to normalise.

    Returns:
        A Polars-compatible Arrow data type.
    """
    # Apply Spark normalisation first (handles large_*, dictionary, extensions).
    dtype = to_spark_arrow_type(dtype)

    if pa.types.is_map(dtype):
        key_field = dtype.key_field
        item_field = dtype.item_field

        key_type = to_polars_arrow_type(key_field.type)
        value_type = to_polars_arrow_type(item_field.type)

        # Represent as list<struct<key, value>> — Polars' canonical map form.
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
    """Cast a ``Table`` and return a single ``RecordBatch``.

    All ``ChunkedArray`` columns are merged via
    :meth:`~pyarrow.ChunkedArray.combine_chunks` before constructing the batch.
    For empty tables an empty batch with the correct schema is returned.

    Args:
        data: Source table.
        options: Cast options.

    Returns:
        A single ``pa.RecordBatch``.
    """
    casted: pa.Table = cast_arrow_tabular(data, options)

    if casted.num_rows == 0:
        arrays = [pa.array([], type=f.type) for f in casted.schema]
        return pa.RecordBatch.from_arrays(arrays, schema=casted.schema)  # type: ignore

    arrays = [chunked_array.combine_chunks() for chunked_array in casted.columns]
    return pa.RecordBatch.from_arrays(arrays, schema=casted.schema)  # type: ignore


@register_converter(pa.RecordBatch, pa.Table)
def record_batch_to_table(
    data: pa.RecordBatch,
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """Cast a ``RecordBatch`` and wrap the result as a single-batch ``Table``.

    Args:
        data: Source record batch.
        options: Cast options.

    Returns:
        A ``pa.Table`` containing one batch.
    """
    casted = cast_arrow_tabular(data, options)
    return pa.Table.from_batches(batches=[casted], schema=casted.schema)  # type: ignore


@register_converter(pa.Table, pa.RecordBatchReader)
def table_to_record_batch_reader(
    data: pa.Table,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """Cast a ``Table`` and expose the result as a ``RecordBatchReader``.

    Args:
        data: Source table.
        options: Cast options.

    Returns:
        A ``pa.RecordBatchReader`` iterating over the cast table's batches.
    """
    casted = cast_arrow_tabular(data, options)
    return pa.RecordBatchReader.from_batches(casted.schema, casted.to_batches())  # type: ignore


@register_converter(pa.RecordBatchReader, pa.Table)
def record_batch_reader_to_table(
    data: pa.RecordBatchReader,
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """Cast each batch in a ``RecordBatchReader`` and collect into a ``Table``.

    Args:
        data: Source reader.  All batches are materialised.
        options: Cast options.

    Returns:
        A ``pa.Table`` containing all cast batches.
    """
    casted_reader: pa.RecordBatchReader = cast_arrow_record_batch_reader(data, options)
    return pa.Table.from_batches(batches=list(casted_reader), schema=casted_reader.schema)  # type: ignore


@register_converter(pa.RecordBatch, pa.RecordBatchReader)
def record_batch_to_record_batch_reader(
    data: pa.RecordBatch,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """Cast a ``RecordBatch`` and wrap it in a single-batch ``RecordBatchReader``.

    Args:
        data: Source record batch.
        options: Cast options.

    Returns:
        A ``pa.RecordBatchReader`` with one batch.
    """
    casted = cast_arrow_tabular(data, options)
    return pa.RecordBatchReader.from_batches(schema=casted.schema, batches=[casted])  # type: ignore


@register_converter(pa.RecordBatchReader, pa.RecordBatch)
def record_batch_reader_to_record_batch(
    data: pa.RecordBatchReader,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatch:
    """Materialise a ``RecordBatchReader`` into a single ``RecordBatch``.

    .. warning::
        All batches are read into memory before merging.  Use
        :func:`cast_arrow_record_batch_reader` for streaming workloads.

    Args:
        data: Source reader.
        options: Cast options.

    Returns:
        A single merged ``pa.RecordBatch``.
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
    """Derive a ``pa.Field`` descriptor from an Arrow array.

    The field name is taken from ``options.source_arrow_field.name`` (or
    ``"root"`` as fallback).  Nullability is set to ``True`` if the array type
    is ``null`` or the array contains any nulls.

    Args:
        array: Array to introspect.
        options: Cast options.

    Returns:
        A ``pa.Field`` describing the array's type and nullability.
    """
    options = CastOptions.check_arg(options=options)
    name = options.source_arrow_field.name if options.source_arrow_field else "root"
    metadata = options.source_arrow_field.metadata if options.source_arrow_field else None

    arrow_field = pa.field(
        name,
        array.type,
        nullable=array.type == pa.null() or array.null_count > 0,
        metadata=metadata,
    )
    return arrow_field_to_field(arrow_field, options)


@register_converter(pa.DataType, pa.Field)
def arrow_type_to_field(
    arrow_type: ArrowDataType,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """Wrap an Arrow ``DataType`` into a ``pa.Field``.

    Args:
        arrow_type: Arrow type to wrap.
        options: Cast options.  Used to derive the field name and nullability.

    Returns:
        A ``pa.Field`` with the given type.
    """
    options = CastOptions.check_arg(options=options)
    name = options.source_arrow_field.name if options.source_arrow_field else "root"
    nullable = (
        options.source_arrow_field.nullable if options.source_arrow_field else True
    )
    metadata = (
        options.source_arrow_field.metadata if options.source_arrow_field else None
    )

    arrow_field = pa.field(name, arrow_type, nullable=nullable, metadata=metadata)
    return arrow_field_to_field(arrow_field, options)


@register_converter(pa.Field, pa.Field)
def arrow_field_to_field(
    arrow_field: pa.Field,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """Optionally merge *arrow_field* with a target field from *options*.

    When ``options.merge`` is ``True`` and ``options.target_arrow_field`` is
    set, the two fields are merged via :func:`merge_arrow_fields`.  Otherwise
    *arrow_field* is returned as-is.

    Args:
        arrow_field: Source field.
        options: Cast options.

    Returns:
        Merged or original ``pa.Field``.
    """
    if options is None:
        return arrow_field

    options = CastOptions.check_arg(options)

    if options.merge and options.target_arrow_field is not None:
        return merge_arrow_fields(arrow_field, options.target_arrow_field)
    return arrow_field


@register_converter(pa.Schema, pa.Schema)
def arrow_schema_to_schema(
    arrow_schema: pa.Schema,
    options: Optional[CastOptions] = None,
) -> pa.Schema:
    """Optionally merge *arrow_schema* with a target field from *options*.

    When ``options.merge`` is ``True`` and ``options.target_arrow_field`` is
    set, converts the schema to a struct field, merges it, then converts back
    to a schema.

    Args:
        arrow_schema: Source schema.
        options: Cast options.

    Returns:
        Merged or original ``pa.Schema``.
    """
    if options is None:
        return arrow_schema

    options = CastOptions.check_arg(options)

    if options.merge and options.target_arrow_field is not None:
        base_field = arrow_schema_to_field(arrow_schema, None)
        casted = merge_arrow_fields(base_field, options.target_arrow_field)
        return arrow_field_to_schema(casted, None)

    return arrow_schema


@register_converter(pa.Schema, pa.Field)
def arrow_schema_to_field(
    data: pa.Schema,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """Wrap an Arrow schema as a struct ``pa.Field``.

    The schema's top-level fields become children of a ``pa.struct`` type.
    The field name is stored in the schema metadata under the ``b"name"`` key
    (defaulting to ``"root"``).

    Args:
        data: Arrow schema to wrap.
        options: Cast options.

    Returns:
        A struct ``pa.Field`` encapsulating the schema.
    """
    dtype = pa.struct(list(data))
    md = dict(data.metadata or {})
    name = md.setdefault(b"name", b"root")

    arrow_field = pa.field(name.decode(), dtype, False, md)
    return arrow_field_to_field(arrow_field, options)


@register_converter(pa.Field, pa.Schema)
def arrow_field_to_schema(
    arrow_field: pa.Field,
    options: Optional[CastOptions] = None,
) -> pa.Schema:
    """Return a schema view of an Arrow field.

    For struct fields, the schema lists the struct's children directly.
    For non-struct fields, the schema contains a single field.

    The field name is preserved in the schema metadata under the ``b"name"``
    key.

    Args:
        arrow_field: Arrow field to convert.
        options: Cast options.

    Returns:
        ``pa.Schema`` representation of the field.
    """
    arrow_field = arrow_field_to_field(arrow_field, options)
    md = dict(arrow_field.metadata or {})
    md[b"name"] = arrow_field.name.encode()

    if pa.types.is_struct(arrow_field.type):
        return pa.schema(list(arrow_field.type), metadata=md)
    return pa.schema([arrow_field], metadata=md)


@register_converter(Any, pa.Field)
def any_to_arrow_field(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """Derive a ``pa.Field`` from *any* supported object.

    Dispatch table:

    * ``pa.Field`` — passed through / merged with target via
      :func:`arrow_field_to_field`.
    * ``pa.Schema`` — wrapped via :func:`arrow_schema_to_field`.
    * Object with ``.schema`` or ``.arrow_schema`` attribute — the attribute
      (callable or plain) is resolved and the schema is wrapped.
    * ``None`` — returns ``options.target_arrow_field`` if set.
    * dataclass instance — converted via
      :func:`~.dataclasses.dataclass.dataclass_to_arrow_field`.
    * PySpark objects — via :func:`~.spark.cast.any_spark_to_arrow_field`.
    * Polars objects — via :func:`~.polars.cast.any_polars_to_arrow_field`.
    * Everything else — converted to a Polars ``DataFrame`` first, then
      introspected.

    Args:
        obj: Object to introspect.
        options: Cast options.

    Returns:
        Arrow field description of the object.

    Raises:
        ValueError: If *obj* is ``None`` and no target field is set in *options*.
    """
    if not isinstance(obj, pa.Field):
        if isinstance(obj, pa.Schema):
            return arrow_schema_to_field(obj, options)
        elif hasattr(obj, "schema") or hasattr(obj, "arrow_schema"):
            attr = getattr(obj, "schema", None) or getattr(obj, "arrow_schema", None)

            if callable(attr):
                try:
                    attr = attr()
                except Exception:
                    pass

            if isinstance(attr, pa.Schema):
                return arrow_schema_to_field(attr, options)
        elif isinstance(obj, pa.DataType):
            return arrow_type_to_field(obj, options)
        elif isinstance(obj, dict):
            return dict_to_arrow_field(obj, options)
        elif callable(obj):
            obj = obj()

        options = CastOptions.check_arg(options)

        if obj is None:
            if not options.target_arrow_field:
                raise ValueError("Cannot convert None to pyarrow.Field")
            return options.target_arrow_field

        if is_dataclass(obj):
            from yggdrasil.dataclasses.dataclass import dataclass_to_arrow_field
            return dataclass_to_arrow_field(obj)

        namespace = ObjectSerde.full_namespace(obj)

        if namespace.startswith("pyspark"):
            from yggdrasil.spark.cast import any_spark_to_arrow_field
            return any_spark_to_arrow_field(obj, options)
        elif namespace.startswith("polars"):
            from yggdrasil.polars.cast import any_polars_to_arrow_field
            obj = any_polars_to_arrow_field(obj, options)

            if options.source_field:
                obj = merge_arrow_fields(options.source_field, obj)
        else:
            from yggdrasil.polars.lib import polars
            from yggdrasil.polars.cast import any_to_polars_dataframe, any_polars_to_arrow_field

            df: polars.DataFrame = any_to_polars_dataframe(obj, options)
            obj = any_polars_to_arrow_field(df, options)

            if options.source_field:
                obj = merge_arrow_fields(options.source_field, obj)

    return arrow_field_to_field(obj, options)


@register_converter(Any, pa.Schema)
def any_to_arrow_schema(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.Schema:
    """Derive a ``pa.Schema`` from *any* supported object.

    Converts *obj* to an Arrow field via :func:`any_to_arrow_field`, then
    expands the field to a schema via :func:`arrow_field_to_schema`.

    For objects that are already ``pa.Schema``, delegates to
    :func:`arrow_schema_to_schema` (which can optionally merge with a target).

    Args:
        obj: Object to introspect.
        options: Cast options.

    Returns:
        Arrow schema description of the object.
    """
    if not isinstance(obj, pa.Schema):
        options = CastOptions.check_arg(options)
        obj = any_to_arrow_field(obj, options)
        return arrow_field_to_schema(obj, options)

    return arrow_schema_to_schema(obj, options)


# ---------------------------------------------------------------------------
# Dict serialization helpers
# ---------------------------------------------------------------------------


@register_converter(pa.Field, dict)
def arrow_field_to_dict(field: pa.Field, options=None) -> dict[str, Any]:
    """Convert a pyarrow.Field to a JSON-serializable dict."""
    return {
        "name": field.name,
        "type": arrow_type_to_dict(field.type),
        "nullable": field.nullable,
        "metadata": (
            {k.decode(): v.decode() for k, v in field.metadata.items()}
            if field.metadata
            else None
        ),
    }

@register_converter(dict, pa.Field)
def dict_to_arrow_field(d: dict[str, Any], options=None) -> pa.Field:
    """Reconstruct a pyarrow.Field from a dict produced by field_to_dict."""
    metadata = d.get("metadata")
    if metadata:
        metadata = {k.encode(): v.encode() for k, v in metadata.items()}

    return pa.field(
        name=d["name"],
        type=dict_to_arrow_type(d["type"]),
        nullable=d.get("nullable", True),
        metadata=metadata,
    )


# ── Type serialization ────────────────────────────────────────────────────────
@register_converter(pa.DataType, dict)
def arrow_type_to_dict(t: ArrowDataType, options=None) -> dict[str, Any]:
    """Recursively serialize a pyarrow DataType."""
    # Nested: struct
    if pa.types.is_struct(t):
        return {
            "name": "struct",
            "fields": [arrow_field_to_dict(t.field(i)) for i in range(t.num_fields)],
        }
    # Nested: list / large_list
    if pa.types.is_list(t):
        return {"name": "list", "value_type": arrow_type_to_dict(t.value_type)}
    if pa.types.is_large_list(t):
        return {"name": "large_list", "value_type": arrow_type_to_dict(t.value_type)}
    if pa.types.is_fixed_size_list(t):
        return {
            "name": "fixed_size_list",
            "value_type": arrow_type_to_dict(t.value_type),
            "list_size": t.list_size,
        }
    # Nested: map
    if pa.types.is_map(t):
        return {
            "name": "map",
            "key_type": arrow_type_to_dict(t.key_type),
            "item_type": arrow_type_to_dict(t.item_type),
        }
    # Nested: dictionary (categorical)
    if pa.types.is_dictionary(t):
        return {
            "name": "dictionary",
            "index_type": arrow_type_to_dict(t.index_type),
            "value_type": arrow_type_to_dict(t.value_type),
            "ordered": t.ordered,
        }
    # Parameterized: decimal
    if pa.types.is_decimal(t):
        return {"name": "decimal128", "precision": t.precision, "scale": t.scale}
    # Parameterized: timestamp
    if pa.types.is_timestamp(t):
        return {"name": "timestamp", "unit": t.unit, "tz": t.tz}
    # Parameterized: time32 / time64
    if pa.types.is_time32(t):
        return {"name": "time32", "unit": t.unit}
    if pa.types.is_time64(t):
        return {"name": "time64", "unit": t.unit}
    # Parameterized: duration
    if pa.types.is_duration(t):
        return {"name": "duration", "unit": t.unit}
    # Parameterized: fixed_size_binary
    if pa.types.is_fixed_size_binary(t):
        return {"name": "fixed_size_binary", "byte_width": t.byte_width}
    # Primitives — str(t) gives "int64", "float32", "bool", "utf8", "date32", etc.
    return {"name": str(t)}


STR_PRIMITIVE_TYPES: dict[str, pa.DataType] = {
    "null": pa.null(),
    "bool": pa.bool_(),
    "int8": pa.int8(), "int16": pa.int16(), "int32": pa.int32(), "int64": pa.int64(),
    "uint8": pa.uint8(), "uint16": pa.uint16(), "uint32": pa.uint32(), "uint64": pa.uint64(),
    "float": pa.float16(),   # pa calls it float16 but str() → "halffloat"; kept for round-trip
    "halffloat": pa.float16(),
    "float16": pa.float16(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "double": pa.float64(),
    "string": pa.string(), "utf8": pa.utf8(),
    "large_string": pa.large_utf8(), "large_utf8": pa.large_utf8(),
    "binary": pa.binary(), "large_binary": pa.large_binary(),
    "date32": pa.date32(), "date64": pa.date64(),
}

@register_converter(dict, pa.DataType)
def dict_to_arrow_type(d: dict[str, Any], options=None) -> ArrowDataType:
    """Recursively deserialize a pyarrow DataType from a dict."""
    name = d["name"]

    match name:
        case "struct":
            return pa.struct([dict_to_arrow_type(f) for f in d["fields"]])
        case "list":
            return pa.list_(dict_to_arrow_type(d["value_type"]))
        case "large_list":
            return pa.large_list(dict_to_arrow_type(d["value_type"]))
        case "fixed_size_list":
            return pa.list_(dict_to_arrow_type(d["value_type"]), d["list_size"])
        case "map":
            return pa.map_(dict_to_arrow_type(d["key_type"]), dict_to_arrow_type(d["item_type"]))
        case "dictionary":
            return pa.dictionary(
                dict_to_arrow_type(d["index_type"]),
                dict_to_arrow_type(d["value_type"]),
                d.get("ordered", False),
            )
        case "decimal128":
            return pa.decimal128(d["precision"], d["scale"])
        case "timestamp":
            return pa.timestamp(d["unit"], tz=d.get("tz"))
        case "time32":
            return pa.time32(d["unit"])
        case "time64":
            return pa.time64(d["unit"])
        case "duration":
            return pa.duration(d["unit"])
        case "fixed_size_binary":
            return pa.binary(d["byte_width"])
        case _:
            if name in STR_PRIMITIVE_TYPES:
                return STR_PRIMITIVE_TYPES[name]
            raise ValueError(f"Unknown Arrow type name: {name!r}")
