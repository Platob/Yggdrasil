"""Set-style operations on Spark DataFrames — mirror of :mod:`yggdrasil.arrow.ops`.

Exposes :func:`dedup_spark_dataframe`, :func:`resample_spark_dataframe`,
and :func:`fill_spark_dataframe`. The signatures match their arrow
counterparts (same parameter names, same defaults, same vocabulary
for ``fill_strategy``) so :class:`yggdrasil.data.options.CastOptions`
can route a read through whichever engine it ends up holding without
branching at the call site.

Distribution model
------------------

Two execution paths, picked by what the caller passes:

* **partition-by present** — the op routes through
  ``df.groupBy(*partition_by).applyInArrow(...)``, delegating each
  group's work to :mod:`yggdrasil.arrow.ops` running inside the
  executor. Spark owns the shuffle; pyarrow owns the per-group
  algorithm. This is the canonical path for the entity-keyed cases
  the resample / dedup contract is designed for (one timeline per
  symbol, one row per tenant, …).

* **partition-by absent** — Spark SQL window functions over an empty
  ``Window.partitionBy()`` (one global partition). Cheaper to plan
  than the grouped-arrow path on small inputs but funnels everything
  to one executor — fine when the caller really wants a single
  global timeline and the frame is bounded.

PySpark 4.0+ is assumed (the runtime pin in ``pyproject.toml``);
``applyInArrow`` was added in 4.0 and is the natural arrow-typed
group map.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import pyarrow as pa

from yggdrasil.arrow.ops import (
    _normalize_fill_strategy,
    resample_arrow_table,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame


__all__ = [
    "dedup_spark_dataframe",
    "fill_spark_dataframe",
    "resample_spark_dataframe",
]


logger = logging.getLogger(__name__)


# Synthetic column names used inside the SQL window paths. The
# ``__ygg_*__`` namespace is reserved by yggdrasil's spark ops; the
# wrappers always strip them on the way out so the output schema
# matches the input schema exactly.
_ORDER_COL = "__ygg_order__"
_RN_COL = "__ygg_rn__"
_BUCKET_COL = "__ygg_bucket__"


def dedup_spark_dataframe(
    df: "SparkDataFrame",
    keys: Sequence[str],
) -> "SparkDataFrame":
    """Drop duplicate rows on *keys*, keeping the first occurrence per group.

    Mirrors :func:`yggdrasil.arrow.ops.dedup_arrow_table`. When
    ``keys`` is empty the input is returned unchanged so callers
    can call this unconditionally on every read pass — matching the
    arrow op's "zero-cost when the target schema has no unique
    columns" short-circuit.

    "First occurrence" is anchored on
    :func:`pyspark.sql.functions.monotonically_increasing_id`, which
    preserves Spark's internal per-partition ordering. That order
    isn't a contract across shuffles, but it's the strongest
    "first row in input order" a distributed engine can offer
    without a user-supplied sort key — and it's stable across plan
    evaluations of the same job, which is what callers actually
    rely on.

    Spark's ``dropDuplicates`` would be a one-liner but it picks an
    arbitrary row per duplicate group; the
    ``row_number() OVER (...) = 1`` form keeps the contract aligned
    with the arrow path's "first non-null, first occurrence" answer.
    """
    if not keys:
        return df
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    keys = list(keys)
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(
            f"dedup_spark_dataframe: keys not found on frame — "
            f"missing: {missing}. Available: {df.columns}."
        )

    df_ordered = df.withColumn(_ORDER_COL, F.monotonically_increasing_id())
    window = Window.partitionBy(*keys).orderBy(F.col(_ORDER_COL))
    return (
        df_ordered.withColumn(_RN_COL, F.row_number().over(window))
        .where(F.col(_RN_COL) == 1)
        .drop(_RN_COL, _ORDER_COL)
    )


def resample_spark_dataframe(
    df: "SparkDataFrame",
    *,
    time_column: str,
    sampling_seconds: int,
    partition_by: "Sequence[str] | None" = None,
    fill_strategy: "str | None" = "ffill",
) -> "SparkDataFrame":
    """Align *df* to a fixed sampling grid on *time_column*.

    Mirrors :func:`yggdrasil.arrow.ops.resample_arrow_table` —
    timestamps are floored to the largest multiple of
    ``sampling_seconds`` <= the original, rows sharing a bucket
    collapse via "first occurrence", and ``fill_strategy`` runs the
    same ffill / bfill pass on the resampled output (per partition
    when ``partition_by`` is set, globally otherwise).

    Distribution:

    * ``partition_by`` is non-empty → ``groupBy(*partition_by)
      .applyInArrow(...)`` delegates the bucket collapse + ffill to
      :func:`yggdrasil.arrow.ops.resample_arrow_table` running on
      each group's :class:`pa.Table` inside the executor. Spark
      owns the shuffle; pyarrow owns the per-group algorithm. This
      is the canonical path.
    * ``partition_by`` is empty → SQL window functions on a single
      global partition. Funnels the whole frame to one executor;
      use only when "one global timeline" is genuinely what the
      caller wants.

    Short-circuits the same way as the arrow op:

    * ``sampling_seconds <= 0`` → input returned unchanged.
    * ``time_column`` missing from the schema → input returned unchanged.
    * the column isn't a timestamp → input returned unchanged.
    """
    if sampling_seconds <= 0:
        return df
    if time_column not in df.columns:
        return df

    from pyspark.sql import types as T

    field = df.schema[time_column]
    if not isinstance(field.dataType, (T.TimestampType, T.TimestampNTZType)):
        return df

    part_cols = [c for c in (partition_by or ()) if c != time_column and c in df.columns]

    if part_cols:
        return _resample_grouped_arrow(
            df,
            time_column=time_column,
            sampling_seconds=sampling_seconds,
            partition_by=part_cols,
            fill_strategy=fill_strategy,
        )
    return _resample_window(
        df,
        time_column=time_column,
        sampling_seconds=sampling_seconds,
        fill_strategy=fill_strategy,
    )


def fill_spark_dataframe(
    df: "SparkDataFrame",
    *,
    sort_by: "str | None" = None,
    partition_by: "Sequence[str] | None" = None,
    fill_strategy: "str | None" = "ffill",
    fill_columns: "Sequence[str] | None" = None,
) -> "SparkDataFrame":
    """Forward / backward fill nulls per partition on a Spark frame.

    Mirrors :func:`yggdrasil.arrow.ops.fill_arrow_table` — same
    semantics, same vocabulary for ``fill_strategy``, same
    "nested types are skipped, partition boundaries are honored"
    rules. Uses :func:`pyspark.sql.functions.last` / ``first`` with
    ``ignorenulls=True`` over a partition-and-time window so the
    fill is fully push-down inside Spark's Catalyst optimiser — no
    arrow round-trip needed for the flat null-propagation case.
    """
    strategy = _normalize_fill_strategy(fill_strategy)
    if strategy in {"none", ""}:
        return df

    from pyspark.sql import functions as F, types as T
    from pyspark.sql.window import Window

    part_cols = list(partition_by or ())
    skip = set(part_cols)
    if sort_by is not None:
        skip.add(sort_by)

    if fill_columns is None:
        candidates = [c for c in df.columns if c not in skip]
    else:
        candidates = [c for c in fill_columns if c in df.columns and c not in skip]

    fillable: list[str] = []
    for name in candidates:
        dt = df.schema[name].dataType
        if isinstance(dt, (T.StructType, T.ArrayType, T.MapType)):
            continue
        fillable.append(name)
    if not fillable:
        return df

    if sort_by is None or sort_by not in df.columns:
        # No sort key — assume the frame is already in the right
        # order. Without an ``orderBy`` Spark forbids the
        # rowsBetween-anchored window, so fall through to a sort by
        # ``monotonically_increasing_id`` (per-partition stable
        # order, same anchor the dedup uses).
        df = df.withColumn(_ORDER_COL, F.monotonically_increasing_id())
        order_col: str = _ORDER_COL
    else:
        order_col = sort_by

    if strategy == "ffill":
        bounds = (Window.unboundedPreceding, 0)
        agg = F.last
    else:
        bounds = (0, Window.unboundedFollowing)
        agg = F.first
    window = (
        Window.partitionBy(*part_cols).orderBy(F.col(order_col)).rowsBetween(*bounds)
    )

    select_cols = []
    for name in df.columns:
        if name == _ORDER_COL:
            continue
        if name in fillable:
            select_cols.append(agg(F.col(name), ignorenulls=True).over(window).alias(name))
        else:
            select_cols.append(F.col(name))
    return df.select(*select_cols)


# ---------------------------------------------------------------------------
# Resample paths
# ---------------------------------------------------------------------------


def _resample_grouped_arrow(
    df: "SparkDataFrame",
    *,
    time_column: str,
    sampling_seconds: int,
    partition_by: list[str],
    fill_strategy: "str | None",
) -> "SparkDataFrame":
    """``groupBy(partition_by).applyInArrow`` delegating to arrow.ops."""

    # Capture parameters for the executor closure. The function
    # is shipped to the executor by PySpark's Arrow UDF machinery, so
    # the closure can't reference module-level state — pin everything
    # to local names.
    _time_column = time_column
    _sampling_seconds = int(sampling_seconds)
    _fill_strategy = fill_strategy
    # Pre-compute the Arrow schema PySpark's output verifier expects.
    # PySpark builds it from ``df.schema`` via ``to_arrow_schema`` and
    # checks per-column. Pyarrow's internal type cascade and Spark's
    # ``StructType -> arrow`` mapping disagree on a few canonical
    # spellings (``timestamp[us, tz=UTC]`` vs ``Etc/UTC``); casting
    # the resample result onto this schema explicitly absorbs that
    # drift — without it, the verifier rejects a perfectly-shaped
    # table with a ``RESULT_COLUMN_TYPES_MISMATCH``.
    from pyspark.sql.pandas.types import to_arrow_schema
    _expected_schema = to_arrow_schema(df.schema)

    def _per_group(table: pa.Table) -> pa.Table:
        # ``partition_by=None`` inside the group: this group already
        # holds a single (symbol / tenant / …) value across every row,
        # so the bucket collapse and the per-partition ffill collapse
        # to a flat resample over the rows handed in.
        out = resample_arrow_table(
            table,
            time_column=_time_column,
            sampling_seconds=_sampling_seconds,
            partition_by=None,
            fill_strategy=_fill_strategy,
        )
        if out.schema != _expected_schema:
            out = out.cast(_expected_schema)
        return out

    return df.groupBy(*partition_by).applyInArrow(_per_group, schema=df.schema)


def _resample_window(
    df: "SparkDataFrame",
    *,
    time_column: str,
    sampling_seconds: int,
    fill_strategy: "str | None",
) -> "SparkDataFrame":
    """Flat (no partition_by) resample via SQL window functions.

    Funnels the whole frame to a single executor through
    ``Window.partitionBy()``. Acceptable for small / bounded inputs
    where "one global timeline" is the caller's contract; the
    grouped-arrow path is preferred whenever the data has a natural
    partition key.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    bucket_us = int(sampling_seconds) * 1_000_000
    ts_us = F.unix_micros(F.col(time_column))
    bucket_col = F.timestamp_micros(
        (ts_us / F.lit(bucket_us)).cast("long") * F.lit(bucket_us),
    )
    df_bucketed = df.withColumn(_BUCKET_COL, bucket_col)
    window = Window.partitionBy(_BUCKET_COL).orderBy(F.col(time_column))
    first_per_bucket = (
        df_bucketed
        .withColumn(_RN_COL, F.row_number().over(window))
        .where(F.col(_RN_COL) == 1)
        .drop(_RN_COL)
    )
    # Swap the original time column for the bucketed one (cast back
    # to the original timestamp dtype so downstream consumers don't
    # see a sudden type drift).
    out_cols = []
    for name in first_per_bucket.columns:
        if name == time_column:
            out_cols.append(F.col(_BUCKET_COL).cast(df.schema[time_column].dataType).alias(time_column))
        elif name == _BUCKET_COL:
            continue
        else:
            out_cols.append(F.col(name))
    resampled = first_per_bucket.select(*out_cols)

    if _normalize_fill_strategy(fill_strategy) in {"none", ""}:
        return resampled
    return fill_spark_dataframe(
        resampled,
        sort_by=time_column,
        partition_by=None,
        fill_strategy=fill_strategy,
    )
