"""Arrow-side cast helpers for :class:`StructType` targets.

Materialized entry points, one per source shape:

* :func:`cast_arrow_struct_array` — struct → struct (per-child rebuild,
  missing children defaulted, nested casts threaded through
  ``options.copy(source=, target=)``).
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
from yggdrasil.exceptions import CastError

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

    if options.source.dtype.type_id != DataTypeId.STRUCT:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source} to {options.target}"
        )

    source_field: "Field" = options.source
    source_type: "StructType" = source_field.dtype
    target_type: "StructType" = options.target.dtype

    children: list[pa.Array] = []
    target_fields = [f.to_arrow_field() for f in target_type.fields]

    for i, target_child in enumerate(target_type.children):
        # See the tabular variant for the rationale — single-method
        # name-then-alias lookup against the source struct's
        # children.
        source_child = source_type.field(name=target_child.name, index=i, raise_error=False)

        if source_child is None:
            children.append(
                target_child.default_arrow_array(
                    size=len(array),
                    memory_pool=options.arrow_memory_pool,
                )
            )
            continue

        source_array = array.field(source_child.name)
        target_arrow_field = target_fields[i]
        # Per-child fast path: when the source child already carries
        # the exact target arrow type and the target is nullable (no
        # fill needed), the recursive cast + finalize cycle is a no-op
        # — skip the ``options.copy`` + ``Field.cast_arrow_array``
        # wrap entirely and reuse the source array. Wide structs
        # (32+ fields where only one widens) pay 30+ pointless
        # per-child option clones otherwise. The strict equality
        # check covers field-level nullability too: ``pa.Field.type``
        # comparison ignores nullability, so we test the full
        # ``pa.Field`` to make sure a ``NOT NULL`` target still
        # routes through the rebuild that re-emits the target field
        # shape.
        if (
            target_child.nullable
            and not target_child.has_default
            and source_array.type == target_arrow_field.type
        ):
            children.append(source_array)
            continue

        children.append(
            target_child.cast_arrow_array(
                source_array,
                options=options.copy(source=source_child, target=target_child),
            )
        )
    return pa.StructArray.from_arrays(
        children,
        fields=target_fields,
        mask=array.is_null(),
        memory_pool=options.arrow_memory_pool,
    )


def cast_arrow_map_array(
    array: pa.MapArray,
    options: "CastOptions",
):
    if not options.need_cast(array):
        return array

    if options.source.dtype.type_id != DataTypeId.MAP:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source} to {options.target}"
        )

    source_field: "Field" = options.source
    source_type: "MapType" = source_field.dtype
    target_type: "StructType" = options.target.dtype

    children: list[pa.Array] = []

    for target_child in target_type.children:
        values = pc.map_lookup(array, target_child.name, occurrence="first")
        casted = target_child.cast_arrow_array(
            values,
            options=options.copy(
                source=source_type.value_field,
                target=target_child,
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

    if options.source.dtype.type_id != DataTypeId.ARRAY:
        raise pa.ArrowInvalid(
            f"Cannot cast {options.source} to {options.target}"
        )

    source_field: "Field" = options.source
    source_type: "ArrayType" = source_field.dtype
    target_type: "StructType" = options.target.dtype

    target_children = target_type.children
    memory_pool = options.arrow_memory_pool

    children: list[pa.Array] = _extract_list_positions(
        array, len(target_children), memory_pool,
    )

    casted_children: list[pa.Array] = []
    for target_child, extracted in zip(target_children, children):
        casted_children.append(
            target_child.cast_arrow_array(
                extracted,
                options=options.copy(
                    source=source_type.item_field,
                    target=target_child,
                ),
            )
        )

    return pa.StructArray.from_arrays(
        casted_children,
        fields=[f.to_arrow_field() for f in target_type.fields],
        mask=array.is_null() if isinstance(array, pa.Array) else None,
        memory_pool=memory_pool,
    )


def _extract_list_positions(
    array: pa.Array,
    num_positions: int,
    memory_pool: "pa.MemoryPool | None",
) -> list[pa.Array]:
    """Extract positions ``0..num_positions-1`` from every row of *array*.

    Vectorised path: for ``ListArray`` / ``LargeListArray`` we use the
    flat ``values`` buffer + ``offsets`` to compute the absolute index
    of each ``(row, position)`` cell, mask out positions that overflow
    a row's actual length (or fall inside a null parent), and run a
    single ``pc.take`` per position. No per-row Python crossings, no
    ``to_pylist`` materialisation — see CLAUDE.md "Never loop over data
    rows in Python."

    ``FixedSizeListArray`` is the rare uniform-shape case; we use
    ``pc.list_element`` for positions inside the fixed size and emit a
    zero-or-null slot for positions beyond it.

    Anything exotic (union-typed list, future Arrow list variant we
    don't recognise) routes through Polars: ``pl.from_arrow`` lifts the
    array, ``list.get(i, null_on_oob=True)`` extracts position ``i``
    per row, and ``to_arrow()`` hands the result back.  Still
    vectorised — no ``to_pylist`` walk.
    """
    n = len(array)
    if num_positions == 0:
        return []

    if isinstance(array, (pa.ListArray, pa.LargeListArray)):
        offsets = array.offsets
        # ``lengths`` is null where the parent row is null, so the
        # ``pc.greater(lengths, i)`` mask below already encodes both
        # "parent null" and "list shorter than i+1 items" in one pass.
        lengths = pc.list_value_length(array)
        # Absolute start of each row inside the flat values buffer.
        # ``offsets`` has length n+1; we want the first n entries
        # (start-of-row, not start-of-next-row).
        starts = offsets.slice(0, n)
        values = array.values

        out: list[pa.Array] = []
        for i in range(num_positions):
            has_element = pc.greater(lengths, i)
            flat_idx = pc.add(starts, i)
            safe_idx = pc.if_else(has_element, flat_idx, None)
            out.append(pc.take(values, safe_idx))
        return out

    if isinstance(array, pa.FixedSizeListArray):
        list_size = array.type.list_size
        out = []
        for i in range(num_positions):
            if i < list_size:
                extracted = pc.list_element(array, i)
                # ``list_element`` doesn't propagate parent nulls on
                # FixedSizeListArray — apply the parent mask explicitly
                # so a null parent stays null in the child.
                if array.null_count > 0:
                    extracted = pc.if_else(
                        pc.is_valid(array), extracted, None,
                    )
                out.append(extracted)
            else:
                out.append(
                    pa.nulls(n, type=array.type.value_type, memory_pool=memory_pool)
                )
        return out

    # Long-tail fallback: unfamiliar list shape (custom subclass,
    # future Arrow variant). Route through Polars so the per-position
    # extraction stays vectorised — ``list.get(i, null_on_oob=True)``
    # propagates parent nulls and out-of-bounds positions as null
    # without a Python crossing.
    import polars as pl

    series_pl = pl.from_arrow(array)
    return [
        series_pl.list.get(i, null_on_oob=True).to_arrow()
        for i in range(num_positions)
    ]


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
    src = options.source
    tgt = options.target
    if src is None or tgt is None:
        return data
    src_children = src.children
    tgt_children = tgt.children
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

    source_schema = options.source
    target_schema = options.target.to_struct()

    target_arrow_schema = target_schema.to_arrow_schema()
    if data.schema.equals(target_arrow_schema, check_metadata=False):
        return data
    if (
        len(data.schema) == len(target_arrow_schema)
        and all(
            data.schema.field(i).name == target_arrow_schema.field(i).name
            and data.schema.field(i).nullable == target_arrow_schema.field(i).nullable
            and data.schema.field(i).type.equals(
                target_arrow_schema.field(i).type,
            )
            for i in range(len(data.schema))
        )
    ):
        return data

    target_arrays: list[pa.Array] = []
    num_rows = data.num_rows

    for i, target_field in enumerate(target_schema.children):
        source_field = source_schema.field(name=target_field.name, index=i, raise_error=False)

        if source_field is None:
            casted = target_field.default_arrow_array(
                size=num_rows,
                memory_pool=options.arrow_memory_pool,
            )
        else:
            source_array = data.column(source_field.name)
            # CastError wrapping lives on ``Field.cast_arrow_array``
            # now — it fires here too (per leaf), and recursively on
            # any nested rebuild beneath it. No need to wrap again.
            casted = target_field.cast_arrow_array(
                source_array,
                options=options.copy(
                    source=source_field,
                    target=target_field,
                ),
            )

        target_arrays.append(casted)

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
    ``options.target`` is unbound). Source-field binding is
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
