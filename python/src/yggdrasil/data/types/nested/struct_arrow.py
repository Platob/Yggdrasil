"""Arrow-side cast helpers for :class:`StructType` targets.

Materialized entry points, one per source shape:

* :func:`cast_arrow_struct_array` — struct → struct (per-child rebuild,
  missing children defaulted, nested casts threaded through
  ``options.copy(source_field=, target_field=)``).
* :func:`cast_arrow_map_array` — map → struct via ``pc.map_lookup``;
  one lookup per target child.
* :func:`cast_arrow_list_array` — list → struct by positional index;
  out-of-bounds is null.
* :func:`cast_arrow_tabular` — Table/RecordBatch column rebuild against
  the merged schema; missing source columns get defaults.

Streaming entry points (span batch boundaries):

* :func:`cast_arrow_batch_iterator` — per-batch tabular cast over an
  iterable of ``pa.RecordBatch``, with optional streamed rechunking
  keyed on :attr:`CastOptions.byte_size`.
* :func:`rechunk_arrow_batches` — re-exported from
  :mod:`yggdrasil.arrow.cast` (lives there because it's a pure pyarrow
  util with no struct-cast coupling); kept here for back-compat with
  existing import paths.

All materialized helpers short-circuit on ``options.need_cast`` and
rely on the parent :meth:`StructType._cast_arrow_array` for engine-
level dispatch. The iterator helper threads through
:func:`cast_arrow_tabular` per batch, so the same short-circuit holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Iterable

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.types.id import DataTypeId

if TYPE_CHECKING:
    from yggdrasil.data.options import CastOptions
    from yggdrasil.data.data_field import Field
    from .array import ArrayType
    from .map import MapType
    from .struct import StructType


__all__ = [
    "cast_arrow_struct_array",
    "cast_arrow_map_array",
    "cast_arrow_list_array",
    "cast_arrow_tabular",
    "cast_arrow_batch_iterator",
]


def cast_arrow_struct_array(
    array: pa.StructArray,
    options: "CastOptions",
):
    # Struct casts honour child nullability — a ``STRING NOT NULL``
    # child is structurally different from a nullable one, and Spark /
    # Delta refuse the implicit cast on the wire even when no value is
    # null. The rebuild emits ``pa.StructArray.from_arrays`` with the
    # target's field shapes, so firing on a nullability-only mismatch
    # is essentially free (no per-row work, just a metadata rebind).
    if not options.need_cast(array, check_nullable=True):
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
        # Lookup precedence mirrors the tabular struct cast:
        # target.name (with name-or-positional shortcut) →
        # target.alias. Lets target schemas declare an alias to pull
        # a differently-named source column without a manual rename
        # pass.
        source_child = source_type.field_by(
            name=target_child.name, index=i, raise_error=False,
        )
        if source_child is None and target_child.has_alias:
            source_child = source_type.field_by(
                name=target_child.alias, raise_error=False,
            )

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

    # Tabular bypass: identity-return only when every target child
    # matches the source one for name + dtype + nullability. The
    # default ``need_cast`` ignores nullability (correct for scalar
    # bypass — flipping a flag is value-free) but a tabular cast also
    # has to honour child nullability: a column declared
    # ``STRUCT<x: STRING NOT NULL>`` is structurally different from a
    # nullable variant, and Spark / Delta refuse the implicit cast
    # downstream even when no value is null. Firing on a
    # nullability-only mismatch is cheap — the rebuild collapses to a
    # ``pa.Table.from_arrays(arrays, schema=target_schema)`` metadata
    # rebind rather than a per-row copy.
    src = options.source_field
    tgt = options.target_field
    if src is None or tgt is None:
        return data
    src_children = src.children_fields
    tgt_children = tgt.children_fields
    if len(src_children) == len(tgt_children) and all(
        s.equals(
            t,
            check_names=True,
            check_dtypes=True,
            check_nullable=True,
            check_metadata=False,
        )
        for s, t in zip(src_children, tgt_children)
    ):
        return data

    source_schema = options.source_schema
    target_schema = options.merged_schema

    target_arrays: list[pa.Array] = []
    num_rows = data.num_rows

    for i, target_field in enumerate(target_schema.children_fields):
        # Lookup precedence: target.name (with the legacy
        # name-or-positional shortcut from ``field_by(name=, index=)``)
        # → target.alias. The alias step lets a target schema rename
        # source columns without forcing the caller to pre-project
        # the frame.
        source_field = source_schema.field_by(
            name=target_field.name, index=i, raise_error=False,
        )
        if source_field is None and target_field.has_alias:
            source_field = source_schema.field_by(
                name=target_field.alias, raise_error=False,
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


# ---------------------------------------------------------------------------
# Streaming — iterator of pa.RecordBatch with byte_size-driven rechunking
# ---------------------------------------------------------------------------


def cast_arrow_batch_iterator(
    batches: Iterable[pa.RecordBatch],
    options: "CastOptions",
) -> Iterator[pa.RecordBatch]:
    """Cast an iterable of ``pa.RecordBatch`` and stream-rechunk to target size.

    Per-batch cast goes through :func:`cast_arrow_tabular` (which
    already short-circuits on schema match — also when
    ``options.target_field`` is unbound). Source-field binding is
    deferred to the first batch so callers don't have to peek upstream;
    the bound options are reused for every subsequent batch.

    Sizing knobs ``options.row_size`` and ``options.byte_size`` drive
    output rechunking via :func:`rechunk_arrow_batches`.
    When neither is set, casted batches pass through unchanged.

    :raises TypeError: if any item is not a :class:`pa.RecordBatch`.
    """
    from yggdrasil.arrow.cast import rechunk_arrow_batches

    iterator = iter(batches)

    try:
        first = next(iterator)
    except StopIteration:
        return

    if not isinstance(first, pa.RecordBatch):
        raise TypeError(
            f"cast_arrow_batch_iterator expected pa.RecordBatch items, "
            f"got {type(first).__name__}."
        )

    bound = options.check_source(first)

    def _cast_stream() -> Iterator[pa.RecordBatch]:
        yield bound.cast_arrow_tabular(first)
        for batch in iterator:
            if not isinstance(batch, pa.RecordBatch):
                raise TypeError(
                    f"cast_arrow_batch_iterator expected pa.RecordBatch items, "
                    f"got {type(batch).__name__}."
                )
            yield bound.cast_arrow_tabular(batch)

    if not bound.byte_size and not bound.row_size:
        yield from _cast_stream()
        return

    yield from rechunk_arrow_batches(
        _cast_stream(),
        byte_size=bound.byte_size,
        row_size=bound.row_size,
        memory_pool=bound.arrow_memory_pool,
    )


# ``rechunk_arrow_batches`` lives in
# :mod:`yggdrasil.arrow.cast` (pure-pyarrow util with no struct-cast
# coupling). It's imported lazily inside
# :func:`cast_arrow_batch_iterator` to break the circular chain
# ``arrow.cast -> data.schema -> data.data_field -> data.types ->
# data.types.nested -> struct_arrow``; callers that need it directly
# should import from ``yggdrasil.arrow.cast``.
