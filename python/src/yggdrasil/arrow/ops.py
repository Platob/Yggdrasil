"""Set-style operations on Arrow tabular data.

Currently exposes :func:`upsert_arrow_tabular`, which combines two
``pa.Table`` / ``pa.RecordBatch`` operands keyed by one or more shared
columns. The match semantics are governed by :class:`Mode`:

- :attr:`Mode.APPEND` — keep ``left`` intact for matching keys; only
  append rows from ``right`` whose keys are absent from ``left``.
- anything else (e.g. :attr:`Mode.UPSERT`, :attr:`Mode.MERGE`,
  :attr:`Mode.OVERWRITE`) — drop ``left`` rows whose keys are present
  in ``right``, then append all of ``right`` so its values win on
  conflicts.

The return shape mirrors ``left``: ``pa.Table`` in → ``pa.Table`` out;
``pa.RecordBatch`` in → ``pa.RecordBatch`` out.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.enums.mode import Mode, ModeLike

from ._typing import ArrowTabular

__all__ = ["upsert_arrow_tabular"]


def upsert_arrow_tabular(
    left: ArrowTabular,
    right: ArrowTabular,
    match_by_names: list[str],
    mode: ModeLike,
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
        identify a row.
    mode
        :attr:`Mode.APPEND` keeps ``left`` for matching keys and only
        appends rows from ``right`` whose keys are not in ``left``. Any
        other mode replaces matching rows in ``left`` with ``right``'s
        values and appends the rest of ``right``. Accepts the full
        :class:`ModeLike` grammar (``"upsert"``, ``"append"``,
        :class:`Mode` member, integer code, …).

    Returns
    -------
    pa.Table | pa.RecordBatch
        Same kind as ``left``.
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

    if resolved_mode is Mode.APPEND:
        # left wins on conflicts: append only right rows whose key is
        # not already in left.
        new_rows = right_t.join(
            left_t.select(keys), keys=keys, join_type="left anti",
        )
        # join may reorder columns by key-first; restore left's column order.
        new_rows = new_rows.select(left_t.schema.names)
        result = pa.concat_tables([left_t, new_rows])
    else:
        # right wins on conflicts: drop left rows whose keys appear in
        # right, then concat all of right.
        keep_rows = left_t.join(
            right_t.select(keys), keys=keys, join_type="left anti",
        )
        keep_rows = keep_rows.select(left_t.schema.names)
        result = pa.concat_tables([keep_rows, right_t])

    if out_is_table:
        return result
    return _table_to_record_batch(result)


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
