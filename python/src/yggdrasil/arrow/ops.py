"""Set-style operations on Arrow tabular data.

Currently exposes :func:`upsert_arrow_tabular`, which combines two
``pa.Table`` / ``pa.RecordBatch`` operands keyed by one or more shared
columns, and :func:`upsert_arrow_batches`, the streaming counterpart
that accepts two ``pa.RecordBatch`` iterables and yields the merged
batches without materializing both sides as Tables. The match
semantics are governed by :class:`Mode`:

- :attr:`Mode.APPEND` — keep ``left`` intact for matching keys; only
  append rows from ``right`` whose keys are absent from ``left``.
- anything else (e.g. :attr:`Mode.UPSERT`, :attr:`Mode.MERGE`,
  :attr:`Mode.OVERWRITE`) — drop ``left`` rows whose keys are present
  in ``right``, then append all of ``right`` so its values win on
  conflicts.

The return shape mirrors ``left`` (``pa.Table`` in → ``pa.Table`` out;
``pa.RecordBatch`` in → ``pa.RecordBatch`` out) unless ``row_size`` /
``byte_size`` are set: explicit rechunking always materializes a
``pa.Table`` since a ``pa.RecordBatch`` is by definition a single
chunk.
"""
from __future__ import annotations

from typing import Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.enums.jointype import JoinType
from yggdrasil.data.enums.mode import Mode, ModeLike

from ._typing import ArrowTabular
from .cast import rechunk_arrow_batches, rechunk_arrow_table

__all__ = ["upsert_arrow_batches", "upsert_arrow_tabular"]


def upsert_arrow_tabular(
    left: ArrowTabular,
    right: ArrowTabular,
    match_by_names: list[str],
    mode: ModeLike,
    *,
    row_size: int | None = None,
    byte_size: int | None = None,
    memory_pool: pa.MemoryPool | None = None,
) -> ArrowTabular:
    """Upsert ``right`` into ``left``, matching rows by ``match_by_names``.

    Parameters
    ----------
    left, right
        ``pa.Table`` or ``pa.RecordBatch``. ``right`` is projected and
        cast onto ``left``'s schema before concatenation, so any extra
        columns on ``right`` are dropped and shared columns are
        coerced to ``left``'s dtypes.
    match_by_names
        One or more column names — present on both operands — that
        identify a row. Nested key types (``struct`` / ``list`` /
        ``map`` / union) are supported via a Python-set fallback; flat
        keys take a vectorized left-anti join.
    mode
        :attr:`Mode.APPEND` keeps ``left`` for matching keys and only
        appends rows from ``right`` whose keys are not in ``left``. Any
        other mode replaces matching rows in ``left`` with ``right``'s
        values and appends the rest of ``right``. Accepts the full
        :class:`ModeLike` grammar (``"upsert"``, ``"append"``,
        :class:`Mode` member, integer code, …).
    row_size, byte_size
        Optional output chunking caps. When either is set the result is
        streamed through :func:`yggdrasil.arrow.cast.rechunk_arrow_batches`
        and emitted as a ``pa.Table`` whose chunks honour the requested
        size — overriding the "same kind as ``left``" rule, since a
        ``pa.RecordBatch`` cannot represent multiple chunks.
    memory_pool
        Forwarded to the rechunker for ``pa.concat_batches`` when
        coalescing buffered batches under ``byte_size``.

    Returns
    -------
    pa.Table | pa.RecordBatch
        Same kind as ``left`` when neither chunking knob is set;
        otherwise a ``pa.Table`` carrying the rechunked batches.
    """
    if not isinstance(left, (pa.Table, pa.RecordBatch)):
        raise TypeError(
            "upsert_arrow_tabular: 'left' must be a pyarrow Table or "
            f"RecordBatch, got {type(left).__name__}."
        )
    if not isinstance(right, (pa.Table, pa.RecordBatch)):
        raise TypeError(
            "upsert_arrow_tabular: 'right' must be a pyarrow Table or "
            f"RecordBatch, got {type(right).__name__}."
        )
    if not match_by_names:
        raise ValueError(
            "upsert_arrow_tabular: 'match_by_names' must contain at "
            "least one column name to match on."
        )

    resolved_mode = Mode.from_(mode)
    keys = list(match_by_names)

    left_names = list(left.schema.names)
    right_names = list(right.schema.names)
    missing_left = [n for n in keys if n not in left_names]
    missing_right = [n for n in keys if n not in right_names]
    if missing_left or missing_right:
        raise ValueError(
            "upsert_arrow_tabular: match_by_names not found — missing "
            f"in left: {missing_left}; missing in right: {missing_right}. "
            f"Available left: {left_names}, right: {right_names}."
        )

    out_is_table = isinstance(left, pa.Table)

    left_t = left if out_is_table else pa.Table.from_batches([left])
    right_t = (
        right if isinstance(right, pa.Table) else pa.Table.from_batches([right])
    )

    # Align right onto left's schema so concat_tables doesn't trip on
    # column order or compatible-but-unequal dtypes. Missing columns
    # surface here as a clear KeyError from select().
    if right_t.schema.names != left_t.schema.names:
        right_t = right_t.select(left_t.schema.names)
    if not right_t.schema.equals(left_t.schema):
        right_t = right_t.cast(left_t.schema)

    # pyarrow's join kernel rejects nested key types (struct / list /
    # map / union); detect that up front and pick the right path. Flat
    # keys go through the vectorized left-anti join; nested keys fall
    # back to a Python-set membership test that handles arbitrarily
    # deep nesting via row-wise tuple/dict materialization.
    if _has_nested_key(left_t.schema, keys):
        result = _upsert_via_python_set(left_t, right_t, keys, resolved_mode)
    else:
        result = _upsert_via_join(left_t, right_t, keys, resolved_mode)

    if (row_size and row_size > 0) or (byte_size and byte_size > 0):
        # Explicit chunking forces a Table return — a RecordBatch can't
        # carry multiple chunks. Hand off to the cast-side rechunker so
        # row/byte caps stay consistent across the codebase.
        return rechunk_arrow_table(
            result,
            row_size=row_size,
            byte_size=byte_size,
            memory_pool=memory_pool,
        )

    if out_is_table:
        return result
    return _table_to_record_batch(result)


def upsert_arrow_batches(
    left: Iterable[pa.RecordBatch],
    right: Iterable[pa.RecordBatch],
    match_by_names: list[str] | None,
    mode: ModeLike,
    *,
    schema: pa.Schema | None = None,
    row_size: int | None = None,
    byte_size: int | None = None,
    memory_pool: pa.MemoryPool | None = None,
) -> Iterator[pa.RecordBatch]:
    """Streaming upsert over two ``pa.RecordBatch`` iterables.

    Tuned to avoid concatenating either side into a single Table:

    - :attr:`Mode.APPEND` — only ``left``'s key tuples are buffered (a
      Python set of hashable rows). ``left`` batches stream through
      untouched while their keys accumulate; ``right`` is then streamed
      batch-by-batch and filtered against the seen keys.
    - upsert-like modes (anything other than ``APPEND``) — ``right`` is
      drained first and held as a list of batches (kept separate, not
      concatenated) so its keys can drive the ``left`` filter; ``left``
      is streamed batch-by-batch, then the buffered ``right`` is emitted
      so its values win on conflicts.

    Parameters
    ----------
    left, right
        Iterables of ``pa.RecordBatch`` (e.g. a generator, a
        ``pa.RecordBatchReader``, or ``pa.Table.to_batches()``).
        ``right``'s batches are projected onto ``left``'s schema before
        emission, so extra columns are dropped and shared columns are
        coerced.
    match_by_names
        One or more shared column names that identify a row. Nested key
        types work transparently — keys are materialized to hashable
        Python values for set membership. ``None`` / empty degrades to
        a plain key-less concatenation (``left`` first, then ``right``)
        so callers can wire the same dispatch through whether or not
        they have keys to dedup on; alignment to ``left``'s schema is
        still applied.
    mode
        Same grammar as :func:`upsert_arrow_tabular`. Ignored when
        ``match_by_names`` is empty — without keys, every mode collapses
        to "concat ``left`` then ``right``".
    schema
        Optional output schema override. When omitted, the schema is
        taken from the first ``left`` batch (or the first ``right``
        batch when ``left`` is empty).
    row_size, byte_size
        Optional output chunking caps. When set, the result is piped
        through :func:`rechunk_arrow_batches` before yielding.
    memory_pool
        Forwarded to :func:`rechunk_arrow_batches` for buffered
        ``pa.concat_batches`` calls.

    Yields
    ------
    pa.RecordBatch
        The merged stream. Empty (zero-row) batches are dropped.
    """
    keys = list(match_by_names or ())

    if not keys:
        merged = _concat_batches(left, right, schema)
    else:
        resolved_mode = Mode.from_(mode)
        if resolved_mode is Mode.APPEND:
            merged = _upsert_batches_append(left, right, keys, schema)
        else:
            merged = _upsert_batches_overwrite(left, right, keys, schema)

    if (row_size and row_size > 0) or (byte_size and byte_size > 0):
        yield from rechunk_arrow_batches(
            merged,
            row_size=row_size,
            byte_size=byte_size,
            memory_pool=memory_pool,
        )
        return
    yield from merged


def _concat_batches(
    left: Iterable[pa.RecordBatch],
    right: Iterable[pa.RecordBatch],
    schema: pa.Schema | None,
) -> Iterator[pa.RecordBatch]:
    """Key-less concatenation: yield ``left`` then ``right``.

    Used when ``match_by_names`` is empty — there's nothing to dedup
    against, so the merge collapses to "stream both sides through".
    ``right`` is still aligned to the output schema (anchored on
    ``left``'s first batch when no explicit ``schema`` is given) so
    shape mismatches surface as cast errors here rather than at the
    write boundary.
    """
    out_schema = schema
    for batch in left:
        if out_schema is None:
            out_schema = batch.schema
        if batch.num_rows:
            yield batch
    for batch in right:
        if out_schema is None:
            out_schema = batch.schema
        batch = _align_batch_to_schema(batch, out_schema)
        if batch.num_rows:
            yield batch


def _upsert_batches_append(
    left: Iterable[pa.RecordBatch],
    right: Iterable[pa.RecordBatch],
    keys: list[str],
    schema: pa.Schema | None,
) -> Iterator[pa.RecordBatch]:
    out_schema = schema
    seen: set = set()
    for batch in left:
        _ensure_keys(batch.schema, keys, "left")
        if out_schema is None:
            out_schema = batch.schema
        seen.update(_hashable_key_rows(batch, keys))
        if batch.num_rows:
            yield batch
    for batch in right:
        _ensure_keys(batch.schema, keys, "right")
        if out_schema is None:
            # left was empty — anchor on right's first batch schema.
            out_schema = batch.schema
        batch = _align_batch_to_schema(batch, out_schema)
        if not batch.num_rows:
            continue
        rows = _hashable_key_rows(batch, keys)
        mask = pa.array([r not in seen for r in rows], type=pa.bool_())
        filtered = batch.filter(mask)
        if filtered.num_rows:
            yield filtered


def _upsert_batches_overwrite(
    left: Iterable[pa.RecordBatch],
    right: Iterable[pa.RecordBatch],
    keys: list[str],
    schema: pa.Schema | None,
) -> Iterator[pa.RecordBatch]:
    # Drain right first so its keys can drive left's filter. Batches
    # stay separate; we never concatenate the buffer into a Table.
    right_buffer: list[pa.RecordBatch] = []
    for batch in right:
        _ensure_keys(batch.schema, keys, "right")
        right_buffer.append(batch)

    out_schema = schema
    aligned_right: list[pa.RecordBatch] | None = None
    seen: set = set()

    def _materialize_right() -> list[pa.RecordBatch]:
        nonlocal aligned_right
        if aligned_right is not None:
            return aligned_right
        target = out_schema
        result: list[pa.RecordBatch] = []
        for rb in right_buffer:
            ar = _align_batch_to_schema(rb, target) if target is not None else rb
            if ar.num_rows:
                seen.update(_hashable_key_rows(ar, keys))
            result.append(ar)
        aligned_right = result
        return aligned_right

    for batch in left:
        _ensure_keys(batch.schema, keys, "left")
        if out_schema is None:
            out_schema = batch.schema
        if aligned_right is None:
            _materialize_right()
        if not batch.num_rows:
            continue
        rows = _hashable_key_rows(batch, keys)
        mask = pa.array([r not in seen for r in rows], type=pa.bool_())
        filtered = batch.filter(mask)
        if filtered.num_rows:
            yield filtered

    if aligned_right is None:
        # left was empty — anchor schema on right's first batch (if any).
        if out_schema is None and right_buffer:
            out_schema = right_buffer[0].schema
        _materialize_right()

    for rb in aligned_right:
        if rb.num_rows:
            yield rb


def _ensure_keys(schema: pa.Schema, keys: list[str], side: str) -> None:
    missing = [n for n in keys if n not in schema.names]
    if missing:
        raise ValueError(
            f"upsert_arrow_batches: match_by_names not found on {side} "
            f"batch — missing: {missing}. Available: {list(schema.names)}."
        )


def _align_batch_to_schema(
    batch: pa.RecordBatch, target: pa.Schema
) -> pa.RecordBatch:
    """Project ``batch`` onto ``target`` — reorder, drop extras, cast types."""
    if batch.schema.names != target.names:
        batch = batch.select(target.names)
    if batch.schema.equals(target):
        return batch
    arrays = [
        batch.column(i).cast(target.field(i).type)
        for i in range(target.num_fields)
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=target)


def _upsert_via_join(
    left_t: pa.Table,
    right_t: pa.Table,
    keys: list[str],
    mode: Mode,
) -> pa.Table:
    """Vectorized path — used when every key column is a flat type."""
    if mode is Mode.APPEND:
        new_rows = right_t.join(
            left_t.select(keys), keys=keys, join_type=JoinType.LEFT_ANTI.arrow,
        ).select(left_t.schema.names)
        return pa.concat_tables([left_t, new_rows])
    keep_rows = left_t.join(
        right_t.select(keys), keys=keys, join_type=JoinType.LEFT_ANTI.arrow,
    ).select(left_t.schema.names)
    return pa.concat_tables([keep_rows, right_t])


def _upsert_via_python_set(
    left_t: pa.Table,
    right_t: pa.Table,
    keys: list[str],
    mode: Mode,
) -> pa.Table:
    """Fallback for nested key columns.

    Materializes each key tuple into a hashable Python value and uses a
    set membership test to drive the filter mask. ``right`` was already
    cast onto ``left``'s schema upstream, so the per-row representation
    of any given key is comparable across the two operands.
    """
    left_rows = _hashable_key_rows(left_t, keys)
    right_rows = _hashable_key_rows(right_t, keys)
    if mode is Mode.APPEND:
        seen = set(left_rows)
        new_mask = pa.array([r not in seen for r in right_rows], type=pa.bool_())
        return pa.concat_tables([left_t, right_t.filter(new_mask)])
    seen = set(right_rows)
    keep_mask = pa.array([r not in seen for r in left_rows], type=pa.bool_())
    return pa.concat_tables([left_t.filter(keep_mask), right_t])


def _has_nested_key(schema: pa.Schema, keys: list[str]) -> bool:
    """True if any of the named key columns has a nested Arrow type."""
    return any(pa.types.is_nested(schema.field(n).type) for n in keys)


def _hashable_key_rows(table: pa.Table, keys: list[str]) -> list:
    """Return one hashable Python value per row identifying its key tuple.

    Single-key inputs collapse to the scalar value; multi-key inputs
    return a tuple per row. Nested values (dicts from struct, lists
    from list/large_list/fixed_size_list, dict-of-list from map) are
    walked recursively so the result is hashable end-to-end.
    """
    cols = [table.column(n).to_pylist() for n in keys]
    if len(cols) == 1:
        return [_to_hashable(v) for v in cols[0]]
    return [tuple(_to_hashable(v) for v in row) for row in zip(*cols)]


def _to_hashable(value):
    """Recursively coerce an Arrow ``to_pylist`` value into a hashable form.

    Arrow surfaces struct as ``dict``, list/large_list/fixed_size_list as
    ``list``, map as ``list[tuple]`` (already hashable after recursion),
    and primitives as their native Python types. We freeze dicts to a
    tuple of items (Arrow preserves field order, and ``right`` was cast
    to ``left``'s schema, so item order is consistent across operands).
    """
    if isinstance(value, dict):
        return tuple((k, _to_hashable(v)) for k, v in value.items())
    if isinstance(value, list):
        return tuple(_to_hashable(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_to_hashable(v) for v in value)
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    return value


def _consolidate(col: pa.ChunkedArray | pa.Array) -> pa.Array:
    """Return *col* as a single non-chunked ``pa.Array``."""
    if isinstance(col, pa.ChunkedArray):
        if col.num_chunks == 0:
            return pa.array([], type=col.type)
        if col.num_chunks == 1:
            return col.chunk(0)
        return pa.concat_arrays(col.chunks)
    return col


def _table_to_record_batch(table: pa.Table) -> pa.RecordBatch:
    """Collapse *table* into a single ``pa.RecordBatch``.

    Preserves the schema even when the result is empty, so callers that
    handed in a ``pa.RecordBatch`` always get one back.
    """
    arrays = [_consolidate(table.column(i)) for i in range(table.num_columns)]
    return pa.RecordBatch.from_arrays(arrays, schema=table.schema)
