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
* :func:`rechunk_arrow_batches_by_byte_size` — the rechunker, exposed
  for callers that already have a casted stream and only need sizing.

All materialized helpers short-circuit on ``options.need_cast`` and
rely on the parent :meth:`StructType._cast_arrow_array` for engine-
level dispatch. The iterator helper threads through
:func:`cast_arrow_tabular` per batch, so the same short-circuit holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Iterator

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
    "rechunk_arrow_batches_by_byte_size",
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
    output rechunking via :func:`rechunk_arrow_batches_by_byte_size`.
    When neither is set, casted batches pass through unchanged.

    :raises TypeError: if any item is not a :class:`pa.RecordBatch`.
    """
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

    yield from rechunk_arrow_batches_by_byte_size(
        _cast_stream(),
        byte_size=bound.byte_size,
        row_size=bound.row_size,
        memory_pool=bound.arrow_memory_pool,
    )


def rechunk_arrow_batches_by_byte_size(
    batches: Iterable[pa.RecordBatch],
    *,
    byte_size: int | None = None,
    row_size: int | None = None,
    memory_pool: pa.MemoryPool | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream-coalesce/slice batches to ~``byte_size`` bytes / ``row_size`` rows.

    Both knobs are optional:

    * Neither set → passthrough.
    * ``row_size`` only → emit fixed-size chunks of at most
      ``row_size`` rows; no buffering, zero-copy slices.
    * ``byte_size`` only → emit ~``byte_size``-byte chunks using the
      per-segment bytes/row ratio to derive a row target.
    * Both set → ``byte_size`` drives the row target; ``row_size``
      caps it (final ``target_rows = min(row_size, derived)``).

    Algorithm (byte_size path):

    * Empty incoming batch → drop (no schema gymnastics on zero-row
      flushes — the consumer already saw a schema in an earlier batch
      or will get one from the upstream reader).
    * Buffer empty + incoming batch already at/over target → slice it
      directly into target-sized chunks (zero-copy).
    * Otherwise accumulate; once buffered ``nbytes`` crosses the
      target, concat + slice the buffer to target-sized chunks. Yield
      everything except a possibly-undersized tail; carry the tail
      forward.
    * On exhaustion → flush whatever is left as a single concat'd
      batch (may be smaller than ``byte_size``).
    """
    has_byte = bool(byte_size and byte_size > 0)
    has_row = bool(row_size and row_size > 0)

    if not has_byte and not has_row:
        yield from batches
        return

    if not has_byte:
        for batch in batches:
            if batch.num_rows == 0:
                continue
            if batch.num_rows <= row_size:
                yield batch
                continue
            for offset in range(0, batch.num_rows, row_size):
                yield batch.slice(offset, row_size)
        return

    def _target_rows(batch: pa.RecordBatch) -> int:
        bytes_per_row = max(1, batch.nbytes // max(1, batch.num_rows))
        derived = max(1, byte_size // bytes_per_row)
        return min(row_size, derived) if has_row else derived

    def _slice_to_target(batch: pa.RecordBatch) -> Iterator[pa.RecordBatch]:
        target = _target_rows(batch)
        if batch.num_rows <= target:
            yield batch
            return
        for offset in range(0, batch.num_rows, target):
            yield batch.slice(offset, target)

    buffer: list[pa.RecordBatch] = []
    buffered_bytes = 0

    for batch in batches:
        if batch.num_rows == 0:
            continue

        if not buffer and batch.nbytes >= byte_size:
            yield from _slice_to_target(batch)
            continue

        buffer.append(batch)
        buffered_bytes += batch.nbytes

        if buffered_bytes < byte_size:
            continue

        combined = pa.concat_batches(buffer, memory_pool=memory_pool)
        target = _target_rows(combined)

        if combined.num_rows <= target:
            # Estimator pushed the row-cap above the combined batch
            # (under-estimate of bytes/row from skewed inputs). Emit as
            # one batch and reset.
            yield combined
            buffer = []
            buffered_bytes = 0
            continue

        sliced = list(_slice_to_target(combined))
        for chunk in sliced[:-1]:
            yield chunk

        tail = sliced[-1]
        if tail.nbytes >= byte_size:
            yield tail
            buffer = []
            buffered_bytes = 0
        else:
            buffer = [tail]
            buffered_bytes = tail.nbytes

    if buffer:
        combined = pa.concat_batches(buffer, memory_pool=memory_pool)
        if has_row and combined.num_rows > row_size:
            yield from _slice_to_target(combined)
        else:
            yield combined
