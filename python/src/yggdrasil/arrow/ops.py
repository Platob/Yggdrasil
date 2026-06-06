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

from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
except ImportError:
    # numpy / pyarrow are heavy optional deps this module needs — auto-install
    # them into the running interpreter on first import (the project's standard
    # import-or-install guard) so Arrow ops just work out of the box.
    from yggdrasil.lazy_imports import _lazy_import

    np = _lazy_import("numpy", install=True)
    pa = _lazy_import("pyarrow", install=True)
    pc = _lazy_import("pyarrow.compute", "pyarrow", install=True)


def _row_index_array(n: int) -> pa.Array:
    """Allocate an int64 ``[0, n)`` array via numpy.

    :func:`pa.array(range(n))` walks every element through Python's
    int → arrow conversion (~135 ns / row); :func:`pa.array(np.arange(n))`
    hands pyarrow a contiguous int64 numpy buffer it can zero-copy
    wrap in a single C-side allocation. Benched on a 10k-row table
    in :func:`yggdrasil.arrow.ops.dedup_arrow_table`:

    * ``pa.array(range(10000))``: 1350 us
    * ``pa.array(np.arange(10000))``: 7 us  (184x faster)

    That gap drops the per-call cost of every dedup / resample by
    ~70% on a typical 10k-row scan since the row-index allocation
    dominated the Python-side preamble.
    """
    return pa.array(np.arange(n, dtype=np.int64))

from yggdrasil.enums.jointype import JoinType
from yggdrasil.enums.mode import Mode, ModeLike

from ._typing import ArrowTabular
from .cast import rechunk_arrow_batches, rechunk_arrow_table

if TYPE_CHECKING:
    from yggdrasil.data.data_field import Field

__all__ = [
    "dedup_arrow_batches",
    "dedup_arrow_table",
    "fill_arrow_table",
    "resample_arrow_batches",
    "resample_arrow_table",
    "upsert_arrow_batches",
    "upsert_arrow_tabular",
]


# Accepted ``fill_strategy`` tokens. ``"ffill"`` / ``"bfill"`` mirror the
# pandas / polars vocabulary; ``"none"`` / ``""`` / ``None`` disable the
# pass. Anything else raises so a typo doesn't silently no-op.
_FILL_STRATEGIES = frozenset({"ffill", "bfill", "none", ""})


def _normalize_fill_strategy(fill_strategy: "str | None") -> str:
    """Validate ``fill_strategy`` and return its canonical lowercase form.

    ``None`` collapses to ``"none"`` so callers can pass ``None`` to opt
    out without a separate branch. Empty string is also treated as
    "no fill". Anything outside :data:`_FILL_STRATEGIES` raises
    ``ValueError`` with the accepted vocabulary.
    """
    if fill_strategy is None:
        return "none"
    token = fill_strategy.lower()
    if token not in _FILL_STRATEGIES:
        raise ValueError(
            f"fill_strategy={fill_strategy!r} is not supported. "
            f"Pass one of {sorted(t for t in _FILL_STRATEGIES if t)} "
            f"or ``None`` / ``\"none\"`` to disable the fill."
        )
    return token


def fill_arrow_table(
    table: pa.Table,
    *,
    sort_by: "str | None" = None,
    partition_by: "Sequence[str] | None" = None,
    fill_strategy: "str | None" = "ffill",
    fill_columns: "Sequence[str] | None" = None,
) -> pa.Table:
    """Forward / backward fill nulls per partition.

    Parameters
    ----------
    table
        Input table. The fill is applied in-place on a copy — the
        input is returned by identity when ``fill_strategy`` disables
        the pass.
    sort_by
        Column to sort by *within* each partition before filling.
        When given, the table is first sorted by
        ``(*partition_by, sort_by)`` so ``ffill`` / ``bfill`` runs on
        the time-ordered axis. When ``None``, the table is assumed to
        be in the correct order already (the resample path already
        emits sorted output).
    partition_by
        Columns that bound the fill — nulls don't carry across
        partition boundaries. Each partition's fill is independent.
        ``None`` / empty runs a flat global fill.
    fill_strategy
        ``"ffill"`` (default) propagates the last non-null value
        forward into subsequent nulls; ``"bfill"`` propagates the
        next non-null value backward. ``None`` / ``"none"`` / ``""``
        is a no-op (returns the input by identity).
    fill_columns
        Restrict the fill to these columns. ``None`` runs the fill on
        every non-partition / non-sort column. Nested types
        (struct / list / map) are always skipped — pyarrow's
        ``fill_null_forward`` kernel doesn't accept them.

    Returns
    -------
    pa.Table
        The filled table. The input is returned unchanged when the
        strategy disables the pass, the table is empty, or no
        fillable column remains after filtering.
    """
    strategy = _normalize_fill_strategy(fill_strategy)
    if strategy in {"none", ""}:
        return table
    if table.num_rows == 0:
        return table

    part_cols = list(partition_by or ())
    skip = set(part_cols)
    if sort_by is not None:
        skip.add(sort_by)

    if fill_columns is None:
        candidates = [c for c in table.schema.names if c not in skip]
    else:
        candidates = [c for c in fill_columns if c in table.schema.names and c not in skip]

    # Drop nested types up front — ``pc.fill_null_forward`` rejects
    # struct / list / map / union arrays. Filling those is ill-defined
    # anyway (would a null inside a list element get filled? from
    # where?), and the caller wants graceful degradation, not a
    # surprise crash on a wide schema with one struct column.
    fillable = [
        c for c in candidates
        if not pa.types.is_nested(table.schema.field(c).type)
    ]
    if not fillable:
        return table

    if sort_by is not None and sort_by in table.schema.names:
        sort_keys = [(c, "ascending") for c in (part_cols + [sort_by])]
        sort_idx = pc.sort_indices(table, sort_keys=sort_keys)
        table = table.take(sort_idx)

    partition_id = _partition_ids_for_sorted(table, part_cols)

    fill_fn = pc.fill_null_forward if strategy == "ffill" else pc.fill_null_backward

    for name in fillable:
        col = table.column(name)
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        # When ``partition_by`` is empty, the whole table is one
        # partition and the fill_null_forward result is already
        # correct — skip the partition-leak masking.
        if not part_cols:
            filled = fill_fn(col)
            table = table.set_column(table.schema.get_field_index(name), name, filled)
            continue

        nonnull_mask = pc.is_valid(col)
        pid_marker = pc.if_else(
            nonnull_mask, partition_id, pa.scalar(None, type=pa.int64()),
        )
        last_pid = fill_fn(pid_marker)
        filled = fill_fn(col)
        # ``last_pid`` is null only when no non-null value exists in
        # the fill direction within the partition; mark those rows
        # null. When ``last_pid`` is set, the row's partition_id must
        # match — otherwise the fill leaked across a boundary.
        same_partition = pc.equal(last_pid, partition_id)
        same_partition = pc.fill_null(same_partition, False)
        final = pc.if_else(
            same_partition, filled, pa.scalar(None, type=col.type),
        )
        table = table.set_column(table.schema.get_field_index(name), name, final)
    return table


def _partition_ids_for_sorted(
    table: pa.Table, partition_by: "Sequence[str]",
) -> pa.Array:
    """Return an int64 partition_id per row.

    Assumes the table is already sorted by ``partition_by`` — rows
    sharing a partition tuple are contiguous. Computes the running
    sum of "first row of a new partition" indicators so each unique
    partition tuple gets a 0-based identifier.

    Pure pyarrow compute (no Python row walk):

    * For each partition column, build a "previous value" view by
      slicing one element off the head and prepending a null.
    * The row starts a new partition when *any* column differs from
      its previous row (treating ``null`` and a non-null as a
      change, and ``null == null`` as no change so the boundary
      detection survives nullable partition keys).
    * Row 0 is unconditionally a partition boundary.
    * Cumulative sum of the bool mask (cast to int64) gives a 1-based
      counter; subtracting 1 makes it 0-based.
    """
    n = table.num_rows
    if not partition_by or n == 0:
        return pa.array(np.zeros(n, dtype=np.int64))

    is_boundary: "pa.Array | None" = None
    for c in partition_by:
        col = table.column(c)
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        if n == 1:
            changed = pa.array([True])
        else:
            prev = pa.concat_arrays([pa.nulls(1, type=col.type), col.slice(0, n - 1)])
            eq = pc.equal(col, prev)
            both_null = pc.and_(pc.is_null(col), pc.is_null(prev))
            # ``equal`` returns null when either side is null;
            # collapse that to False ("not equal") then OR with
            # both_null so two consecutive nulls count as equal.
            eq_norm = pc.or_(pc.fill_null(eq, False), both_null)
            changed = pc.invert(eq_norm)
        is_boundary = changed if is_boundary is None else pc.or_(is_boundary, changed)

    # Row 0 is always a boundary — overwrite whatever the synthetic
    # prev-null comparison decided so single-partition tables get a
    # boundary at index 0 even when row 0's partition values are
    # themselves null.
    head_mask = pa.array(np.arange(n) == 0, type=pa.bool_())
    is_boundary = pc.or_(is_boundary, head_mask)

    pid_one_based = pc.cumulative_sum(pc.cast(is_boundary, pa.int64()))
    return pc.subtract(pid_one_based, pa.scalar(1, type=pa.int64()))


# Timestamp unit → microseconds-per-tick multiplier. Used by
# :func:`resample_arrow_table` to convert a ``pa.timestamp`` column to
# an int64 microsecond axis we can floor against the sampling grid.
_TS_UNIT_TO_US: dict[str, int] = {
    "s": 1_000_000,
    "ms": 1_000,
    "us": 1,
    "ns": 1,  # ns is widened to int64 us via integer division; see below.
}


def resample_arrow_table(
    table: pa.Table,
    *,
    time_column: str,
    sampling_seconds: int,
    partition_by: "Sequence[str] | None" = None,
    fill_strategy: "str | None" = "ffill",
) -> pa.Table:
    """Align *table* to a fixed sampling grid on *time_column*.

    Every timestamp is floored to the largest multiple of
    ``sampling_seconds`` that's <= the original — the column ends up
    on the ``sampling_seconds`` grid and rows that landed in the same
    bucket collapse via ``"first"`` aggregation (matches the
    :func:`dedup_arrow_table` semantics: pick the first occurrence per
    group). Pure pyarrow compute, no Python row walk.

    ``partition_by`` carries the entity columns the resample is
    independent on — passing ``["symbol"]`` on a multi-instrument
    price feed groups by ``(symbol, bucket)`` so each instrument's
    rows bucket on their own timeline instead of getting collapsed
    across instruments. Default ``None`` runs a flat resample (one
    global timeline). The caller / :meth:`CastOptions.resample_on_read`
    auto-derives this list from the target schema's
    :attr:`Field.primary_key` set, minus ``time_column`` itself.

    The contract is **aggregate (downsample) or identity**. When the
    source already lives on a coarser grid than the target, every
    bucket has one row and the result equals the input modulo the
    optional timestamp snap. When the source is finer, the dense
    rows collapse into the coarser bucket. Expanding (upsample) a
    coarse source to a finer grid is *not* in scope — gap-filling is
    application-specific and best done explicitly.

    ``fill_strategy`` runs on the resampled output before return —
    ``"ffill"`` (default) carries the last non-null value forward
    into subsequent nulls *within the same partition*, ``"bfill"``
    propagates the next non-null backward, ``None`` / ``"none"``
    skips the pass. Buckets where the first row had a null column
    inherit from neighbouring buckets on the same partition's
    timeline; bucket "0" of a partition that has no prior non-null
    in that column stays null (no cross-partition leak). The fill
    sorts the resampled output by ``(*partition_by, time_column)``
    so the result is canonically ordered on return.

    Short-circuits in three cases (returns input by identity):

    * ``time_column`` missing from the schema,
    * the column isn't a timestamp,
    * ``sampling_seconds <= 0`` (caller's "no resample requested" knob).
    """
    if sampling_seconds <= 0:
        return table
    if time_column not in table.schema.names:
        return table

    ts_col = table.column(time_column)
    ts_type = ts_col.type
    if not pa.types.is_timestamp(ts_type):
        return table
    if table.num_rows == 0:
        return table

    # Filter partition_by to the columns that actually exist in the
    # batch; a stale primary-key reference (column dropped on the
    # source side) shouldn't crash the resample, just degrade to a
    # flatter grouping. Strip the time column out too — grouping by
    # ``(time_column, bucket)`` is a no-op since both move in lock-step.
    schema_names = set(table.schema.names)
    part_cols: list[str] = []
    if partition_by:
        for c in partition_by:
            if c == time_column:
                continue
            if c in schema_names and c not in part_cols:
                part_cols.append(c)

    # Convert the timestamp to int64 microseconds. ``us`` is the
    # canonical internal unit yggdrasil leans on (it's the default
    # :class:`yggdrasil.data.types.primitive.TimestampType` unit and
    # the precision the cache layer's :data:`RESPONSE_SCHEMA` uses).
    # Nanosecond columns are floored to microseconds here — the
    # ``sampling_seconds`` budget is already coarser than a us, so
    # the precision loss is benign for this op.
    unit = ts_type.unit
    if unit == "ns":
        ts_us = pc.divide_checked(pc.cast(ts_col, pa.int64()), 1000)
    else:
        ts_us = pc.cast(ts_col, pa.int64())
        scale = _TS_UNIT_TO_US.get(unit, 1)
        if scale != 1:
            ts_us = pc.multiply(ts_us, scale)

    bucket_us = sampling_seconds * 1_000_000
    # Floor-divide then re-multiply to snap to the bucket boundary.
    # pyarrow's ``divide_checked`` on int64 is truncating-toward-zero
    # for non-negative timestamps (which is all we care about — the
    # epoch is post-1970 for every real workload), so the floor
    # collapses to one C++ kernel call per direction.
    bucket = pc.multiply(pc.divide_checked(ts_us, bucket_us), bucket_us)

    idx_col = "__ygg_idx__"
    bucket_col = "__ygg_bucket__"
    indexed = table.append_column(bucket_col, bucket).append_column(
        idx_col, _row_index_array(table.num_rows),
    )
    group_keys = [*part_cols, bucket_col]
    grouped = indexed.group_by(group_keys, use_threads=False).aggregate(
        [(idx_col, "first")],
    )
    keep = grouped.column(f"{idx_col}_first").sort()
    selected = table.take(keep)

    # Snap the time column itself to the bucket boundary — without
    # this the rows carry their original timestamps, which is wrong
    # for a "resampled to grid" output. Project the bucket value
    # (which lives in microseconds in the indexed table) back into
    # the original timestamp type, then swap it in.
    bucket_keep = pa.compute.take(bucket, keep)
    if unit == "ns":
        bucket_native = pc.cast(pc.multiply(bucket_keep, 1000), ts_type)
    else:
        scale = _TS_UNIT_TO_US.get(unit, 1)
        if scale != 1:
            bucket_native = pc.cast(
                pc.divide_checked(bucket_keep, scale), ts_type,
            )
        else:
            bucket_native = pc.cast(bucket_keep, ts_type)

    resampled = selected.set_column(
        selected.schema.get_field_index(time_column),
        time_column,
        bucket_native,
    )

    if _normalize_fill_strategy(fill_strategy) in {"none", ""}:
        return resampled
    return fill_arrow_table(
        resampled,
        sort_by=time_column,
        partition_by=part_cols or None,
        fill_strategy=fill_strategy,
    )


def resample_arrow_batches(
    batches: "Iterable[pa.RecordBatch]",
    *,
    time_column: str,
    sampling_seconds: int,
    partition_by: "Sequence[str] | None" = None,
    fill_strategy: "str | None" = "ffill",
) -> "Iterator[pa.RecordBatch]":
    """Iterator-shaped wrapper around :func:`resample_arrow_table`.

    Materialises the stream into a single :class:`pa.Table` (a
    duplicate's bucket-mates can straddle any chunk boundary, just
    like dedup), runs the resample, and re-batches on pyarrow's
    natural chunk boundaries.

    ``partition_by`` carries the entity columns the resample is
    independent on — see :func:`resample_arrow_table` for the
    semantics.

    Empty / zero-budget short-circuits to the input iterator
    unchanged so the caller can route every read through this
    without paying when there's nothing to resample.
    """
    if sampling_seconds <= 0:
        yield from batches
        return

    materialised = [b for b in batches if b is not None]
    if not materialised:
        return

    table = pa.Table.from_batches(materialised)
    if table.num_rows == 0:
        yield from materialised
        return

    resampled = resample_arrow_table(
        table,
        time_column=time_column,
        sampling_seconds=sampling_seconds,
        partition_by=partition_by,
        fill_strategy=fill_strategy,
    )
    yield from resampled.to_batches()


def dedup_arrow_table(
    table: pa.Table,
    keys: Sequence[str],
) -> pa.Table:
    """Drop duplicate rows on the *keys* columns, keep the first occurrence.

    Implementation runs entirely in pyarrow's C++ kernels:

    1. Append a synthetic ``__ygg_idx__`` column carrying the row
       index (one ``pa.array`` allocation, no row walk).
    2. ``group_by(keys, use_threads=False).aggregate([(__ygg_idx__,
       "first")])`` collapses the table to one row per key tuple,
       picking the first occurrence's row index. ``"first"`` is an
       ordered aggregator (pyarrow requires ``use_threads=False``);
       ``"min"`` would coincide on monotonic row indices but
       benched *slower* at every table size from 100 to 10 000 rows
       (multi-threading overhead exceeds its benefit on dedup-shaped
       work). ``"first"`` is also the semantic answer — pick the
       first row, not the smallest synthetic index.
    3. Sort those indices so the output preserves the input order
       (``group_by`` makes no ordering promise on its output rows),
       then ``Table.take`` rebuilds the deduped table.

    Empty input / empty key list short-circuits to the input
    unchanged so the caller can call this unconditionally on every
    read pass.
    """
    if not keys or table.num_rows == 0:
        return table

    idx_col = "__ygg_idx__"
    indexed = table.append_column(idx_col, _row_index_array(table.num_rows))
    grouped = indexed.group_by(list(keys), use_threads=False).aggregate(
        [(idx_col, "first")],
    )
    keep = grouped.column(f"{idx_col}_first").sort()
    return table.take(keep)


def dedup_arrow_batches(
    batches: "Iterable[pa.RecordBatch]",
    keys: Sequence[str],
) -> "Iterator[pa.RecordBatch]":
    """Iterator-shaped wrapper around :func:`dedup_arrow_table`.

    An iterator dedup is fundamentally a stop-the-world op: a
    duplicate's first occurrence can straddle any chunk boundary, so
    we have to materialise every batch before deciding which rows
    survive. Pre-materialise into a :class:`pa.Table`, hand that to
    :func:`dedup_arrow_table`, and re-batch the result with pyarrow's
    natural chunk boundaries on the way out.

    Empty key list short-circuits to the input iterator unchanged so
    the read pipeline can call this unconditionally — the common
    case (no ``unique`` columns in the target schema) stays
    zero-cost.
    """
    if not keys:
        yield from batches
        return

    materialised = [b for b in batches if b is not None]
    if not materialised:
        return

    table = pa.Table.from_batches(materialised)
    if table.num_rows == 0:
        yield from materialised
        return

    deduped = dedup_arrow_table(table, keys)
    yield from deduped.to_batches()


def _resolve_match_by_keys(
    match_by: "Sequence[str | Field] | None",
) -> list[str]:
    """Normalize a ``match_by`` argument to a flat ``list[str]`` of column names.

    Accepts the historic plain-string list, the new
    :class:`yggdrasil.data.Field` list, or a mix. Field entries
    contribute :attr:`Field.name`; the alias / position fallbacks
    are handled by the per-frame ``select_in_*`` consumers, not the
    raw key-list used by the dedup hash.
    """
    if not match_by:
        return []
    keys: list[str] = []
    for item in match_by:
        if isinstance(item, str):
            keys.append(item)
        elif hasattr(item, "name"):
            keys.append(item.name)
        else:
            raise TypeError(
                f"match_by entry must be a column name or a Field, "
                f"got {type(item).__name__}: {item!r}"
            )
    return keys


def upsert_arrow_tabular(
    left: ArrowTabular,
    right: ArrowTabular,
    match_by: "Sequence[str | Field]",
    mode: ModeLike,
    *,
    row_size: int | None = None,
    byte_size: int | None = None,
    memory_pool: pa.MemoryPool | None = None,
) -> ArrowTabular:
    """Upsert ``right`` into ``left``, matching rows by ``match_by``.

    Parameters
    ----------
    left, right
        ``pa.Table`` or ``pa.RecordBatch``. ``right`` is projected and
        cast onto ``left``'s schema before concatenation, so any extra
        columns on ``right`` are dropped and shared columns are
        coerced to ``left``'s dtypes.
    match_by
        One or more column references — names (``str``) or
        :class:`yggdrasil.data.Field` instances — present on both
        operands and identifying a row. Field entries contribute
        their :attr:`Field.name`; per-frame alias / position
        fallbacks live on the ``select_in_*`` side. Nested key
        types (``struct`` / ``list`` / ``map`` / union) are
        supported via a Python-set fallback; flat keys take a
        vectorized left-anti join.
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
    keys = _resolve_match_by_keys(match_by)
    if not keys:
        raise ValueError(
            "upsert_arrow_tabular: 'match_by' must contain at least "
            "one column name (or Field) to match on."
        )

    resolved_mode = Mode.from_(mode)

    left_names = list(left.schema.names)
    right_names = list(right.schema.names)
    missing_left = [n for n in keys if n not in left_names]
    missing_right = [n for n in keys if n not in right_names]
    if missing_left or missing_right:
        raise ValueError(
            "upsert_arrow_tabular: match_by columns not found — missing "
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
    match_by: "Sequence[str | Field] | None",
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
    match_by
        Column references — names (``str``) or
        :class:`yggdrasil.data.Field` instances — present on both
        operands and identifying a row. Field entries contribute
        their :attr:`Field.name`; alias / position fallbacks live
        on the ``select_in_*`` side and don't bleed into the dedup
        hash. Nested key types work transparently — keys are
        materialized to hashable Python values for set membership.
        ``None`` / empty degrades to a plain key-less concatenation
        (``left`` first, then ``right``) so callers can wire the
        same dispatch through whether or not they have keys to
        dedup on; alignment to ``left``'s schema is still applied.
    mode
        Same grammar as :func:`upsert_arrow_tabular`. Ignored when
        ``match_by`` is empty — without keys, every mode collapses to
        "concat ``left`` then ``right``".
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
    keys = _resolve_match_by_keys(match_by)

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

    Used when ``match_by`` is empty — there's nothing to dedup
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
            f"upsert_arrow_batches: match_by columns not found on {side} "
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
    # ``pa.Schema`` exposes its column count via ``len(...)``;
    # ``num_fields`` only exists on the nested-type variant
    # (``pa.StructType``).
    arrays = [
        batch.column(i).cast(target.field(i).type)
        for i in range(len(target))
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
