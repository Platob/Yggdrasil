"""Benchmark Delta read paths — yggdrasil ``DeltaIO`` vs ``deltalake``.

What this measures
------------------

A Delta read is two stages: **file pruning** (which files to open) and
**row scan** (read+filter rows from the surviving files). The yggdrasil
predicate AST + :func:`simplify` + :func:`extract_partition_filters`
combo lets a single :class:`Predicate` drive both — partition columns
get pinned to a value set for the file prune, and the residual row-
level predicate gets a pyarrow.compute filter on every batch. The
comparison points:

1. **yggdrasil**: ``prune_values`` only, ``predicate`` only, and both
   together. The predicate-only shape is the new path — it walks the
   AST once via :func:`extract_partition_filters`, intersects with any
   explicit prune dict, and feeds the row-level filter the residual.
2. **deltalake** (rust crate): ``partitions=[(col, op, val), ...]``
   for the partition prune and ``filters=[...]`` for row-level
   pushdown. The library handles both in one call.
3. **No filter** baselines on each side so the speedup vs full scan
   is visible.

Scenarios target real workloads:

- *single value*: ``region == "us"`` — the canonical partition-prune
  shape; one of N files survives.
- *OR collapse*: ``region == "us" | region == "eu" | region == "uk"``
  — the simplify pass collapses to an InList; should match the same
  cost as the explicit ``is_in([...])``.
- *mixed*: ``region IN (...) AND id > 100`` — partition prune narrows
  the file set, row-level filter prunes inside surviving files.
- *non-partition only*: ``id > 1000`` — no partition prune, every
  file opens, row filter runs.

Usage::

    PYTHONPATH=src python benchmarks/io/nested/bench_delta.py
    PYTHONPATH=src python benchmarks/io/nested/bench_delta.py \\
        --rows 100000 --partitions 32 --repeat 5
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable

import pyarrow as pa

from yggdrasil.data.data_field import Field
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.primitive import Int64Type, StringType
from yggdrasil.io.nested.delta import DeltaIO, DeltaOptions
from yggdrasil.io.tabular.execution.expr import col as expr_col


try:
    from deltalake import DeltaTable, write_deltalake
    HAS_DELTALAKE = True
except ImportError:  # pragma: no cover - bench is informational
    HAS_DELTALAKE = False


# ---------------------------------------------------------------------------
# Fixture — partitioned delta table built once per run.
# ---------------------------------------------------------------------------


PARTITION_KEYS = [f"p{i:02d}" for i in range(64)]


def _partition_schema() -> Schema:
    """Schema partitioned by ``region`` — drives the file split."""
    s = Schema()
    s.with_field(Field(name="id", dtype=Int64Type()))
    s.with_field(
        Field(name="region", dtype=StringType()).with_partition_by(True)
    )
    s.with_field(Field(name="val", dtype=StringType()))
    return s


def _arrow_table(rows: int, *, partitions: int) -> pa.Table:
    """Build the sample table: rows evenly distributed across N partitions."""
    keys = PARTITION_KEYS[:partitions]
    return pa.table({
        "id": pa.array(range(rows), type=pa.int64()),
        "region": pa.array([keys[i % partitions] for i in range(rows)]),
        "val": pa.array([f"row-{i}" for i in range(rows)]),
    })


# ---------------------------------------------------------------------------
# Timing — same shape as the rest of benchmarks/.
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 3)):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
    }


def _fmt(r: dict) -> str:
    scale = 1e6
    unit = "us"
    if r["best"] < 1e-6:
        scale, unit = 1e9, "ns"
    elif r["best"] >= 1e-3:
        scale, unit = 1e3, "ms"
    return (
        f"{r['label']:<70s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _write_scenarios(
    table: pa.Table,
    rows: int,
    partitions: int,
    repeat: int,
) -> list[dict]:
    """Write-throughput comparison: yggdrasil + (when available) deltalake.

    The write side is the partner of the read benchmarks — every read
    is meaningless without confirming the writer keeps up too. Each
    scenario writes to a fresh directory per inner repeat so we
    measure cold-target throughput.
    """
    out: list[dict] = []

    def _yggdrasil_write():
        tmp = tempfile.mkdtemp(prefix="ygg-delta-")
        try:
            d = DeltaIO(path=tmp + "/t")
            d.write_arrow_table(table, options=DeltaOptions(target=_partition_schema()))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    out.append(_time_one(
        f"write: yggdrasil DeltaIO rows={rows} parts={partitions}",
        _yggdrasil_write,
        repeat=repeat, inner=1,
    ))

    if HAS_DELTALAKE:
        def _deltalake_write():
            tmp = tempfile.mkdtemp(prefix="ygg-deltalake-")
            try:
                write_deltalake(tmp + "/t", table, partition_by=["region"])
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        out.append(_time_one(
            f"write: deltalake rows={rows} parts={partitions}",
            _deltalake_write,
            repeat=repeat, inner=1,
        ))
    return out


def _read_scenarios(
    table_path: Path,
    rows: int,
    partitions: int,
    repeat: int,
) -> list[dict]:
    """Read-throughput comparison across pruning strategies."""
    out: list[dict] = []
    keys = PARTITION_KEYS[:partitions]
    target_key = keys[0]
    or_keys = keys[: min(4, partitions)]  # 4-way OR-collapse target

    # yggdrasil holders — cache the snapshot so the first scenario's
    # snapshot resolution doesn't dominate.
    d = DeltaIO(path=str(table_path))
    d.snapshot(fresh=True)  # warm the cache

    dt = DeltaTable(str(table_path)) if HAS_DELTALAKE else None

    # ----- yggdrasil: full scan -----
    def _ygg_full():
        d.refresh()
        return d.read_arrow_table()

    out.append(_time_one(
        f"read: yggdrasil full-scan rows={rows} parts={partitions}",
        _ygg_full, repeat=repeat, inner=2,
    ))

    # ----- yggdrasil: prune_values (legacy) -----
    out.append(_time_one(
        f"read: yggdrasil prune_values=[{target_key}] (1 of {partitions} files)",
        lambda: d.read_arrow_table(
            options=DeltaOptions(prune_values={"region": (target_key,)}),
        ),
        repeat=repeat, inner=10,
    ))

    # ----- yggdrasil: predicate (new path — drives partition prune) -----
    eq_pred = expr_col("region") == target_key
    out.append(_time_one(
        f"read: yggdrasil predicate region == {target_key}",
        lambda: d.read_arrow_table(options=DeltaOptions(predicate=eq_pred)),
        repeat=repeat, inner=10,
    ))

    # ----- yggdrasil: OR-of-EQ → InList via simplify -----
    or_pred = expr_col("region") == or_keys[0]
    for k in or_keys[1:]:
        or_pred = or_pred | (expr_col("region") == k)
    out.append(_time_one(
        f"read: yggdrasil predicate region == ... ({len(or_keys)} OR'd)",
        lambda: d.read_arrow_table(options=DeltaOptions(predicate=or_pred)),
        repeat=repeat, inner=10,
    ))

    # ----- yggdrasil: predicate touching both partition + non-partition -----
    mixed_pred = (expr_col("region") == target_key) & (expr_col("id") > rows // 2)
    out.append(_time_one(
        f"read: yggdrasil predicate region == X AND id > N/2",
        lambda: d.read_arrow_table(options=DeltaOptions(predicate=mixed_pred)),
        repeat=repeat, inner=10,
    ))

    # ----- yggdrasil: row-level only (no partition pruning possible) -----
    non_part_pred = expr_col("id") > rows - 100
    out.append(_time_one(
        f"read: yggdrasil predicate id > N-100 (row-only, all files)",
        lambda: d.read_arrow_table(options=DeltaOptions(predicate=non_part_pred)),
        repeat=repeat, inner=5,
    ))

    if dt is not None:
        # ----- deltalake: full scan -----
        out.append(_time_one(
            f"read: deltalake full-scan rows={rows} parts={partitions}",
            lambda: dt.to_pyarrow_table(),
            repeat=repeat, inner=2,
        ))

        # ----- deltalake: partition filter -----
        out.append(_time_one(
            f"read: deltalake partitions=region={target_key}",
            lambda: dt.to_pyarrow_table(partitions=[("region", "=", target_key)]),
            repeat=repeat, inner=10,
        ))

        # ----- deltalake: IN-style partition filter -----
        out.append(_time_one(
            f"read: deltalake partitions=region IN ... ({len(or_keys)} values)",
            lambda: dt.to_pyarrow_table(partitions=[("region", "in", list(or_keys))]),
            repeat=repeat, inner=10,
        ))

        # ----- deltalake: combined partition + row filter -----
        out.append(_time_one(
            f"read: deltalake partitions+filters region+id",
            lambda: dt.to_pyarrow_table(
                partitions=[("region", "=", target_key)],
                filters=[("id", ">", rows // 2)],
            ),
            repeat=repeat, inner=10,
        ))

        # ----- deltalake: row-level only -----
        out.append(_time_one(
            f"read: deltalake filters id > N-100 (row-only)",
            lambda: dt.to_pyarrow_table(filters=[("id", ">", rows - 100)]),
            repeat=repeat, inner=5,
        ))

    return out


def _extract_scenarios(repeat: int) -> list[dict]:
    """Standalone cost of the partition-filter extractor.

    These are the pure-AST building blocks the read path runs once
    per :meth:`DeltaIO._read_arrow_batches` invocation. The whole
    pipeline budget for "prepare partition prune" is the sum of
    the simplify + extract numbers — typically tens of µs.
    """
    from yggdrasil.io.tabular.execution.expr import extract_partition_filters
    out: list[dict] = []
    partition_cols = ("region", "date")

    # Equality on partition column.
    eq = expr_col("region") == "us"
    out.append(_time_one(
        "extract: eq on partition col",
        lambda: extract_partition_filters(eq, partition_cols),
        repeat=repeat, inner=5_000,
    ))

    # Long InList — the canonical "many distinct partitions" shape.
    big_in = expr_col("region").is_in(PARTITION_KEYS[:32])
    out.append(_time_one(
        f"extract: InList 32 values on partition col",
        lambda: extract_partition_filters(big_in, partition_cols),
        repeat=repeat, inner=5_000,
    ))

    # OR-of-EQ — simplify collapses to InList first, then extract.
    or_chain = expr_col("region") == PARTITION_KEYS[0]
    for k in PARTITION_KEYS[1:16]:
        or_chain = or_chain | (expr_col("region") == k)
    out.append(_time_one(
        "extract: OR-of-EQ 16 values (collapsed then extracted)",
        lambda: extract_partition_filters(or_chain, partition_cols),
        repeat=repeat, inner=2_000,
    ))

    # Mixed: partition + non-partition column.
    mixed = (expr_col("region").is_in(["us", "eu"])) & (expr_col("id") > 100)
    out.append(_time_one(
        "extract: mixed (partition + non-partition AND)",
        lambda: extract_partition_filters(mixed, partition_cols),
        repeat=repeat, inner=5_000,
    ))

    # No constraint extractable.
    range_pred = expr_col("id") > 100
    out.append(_time_one(
        "extract: non-extractable (range on non-partition)",
        lambda: extract_partition_filters(range_pred, partition_cols),
        repeat=repeat, inner=5_000,
    ))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000,
                    help="Row count for the in-memory fixture table.")
    ap.add_argument("--partitions", type=int, default=16,
                    help="Distinct partition values (must be ≤ "
                         f"{len(PARTITION_KEYS)}).")
    ap.add_argument("--repeat", type=int, default=3,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    if args.partitions > len(PARTITION_KEYS):
        ap.error(f"--partitions must be ≤ {len(PARTITION_KEYS)}")

    print(f"# rows={args.rows}  partitions={args.partitions}  repeat={args.repeat}")
    if not HAS_DELTALAKE:
        print("# (deltalake not installed — yggdrasil-only timings)")
    print(f"# {'label':<70s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")

    table = _arrow_table(args.rows, partitions=args.partitions)

    # Build the read fixture once — write benchmarks build their own.
    tmp_root = Path(tempfile.mkdtemp(prefix="ygg-delta-bench-"))
    try:
        read_path = tmp_root / "read_t"
        d = DeltaIO(path=str(read_path))
        d.write_arrow_table(table, options=DeltaOptions(target=_partition_schema()))

        for row in _extract_scenarios(args.repeat):
            print(_fmt(row))
        for row in _write_scenarios(
            table, args.rows, args.partitions, args.repeat,
        ):
            print(_fmt(row))
        for row in _read_scenarios(
            read_path, args.rows, args.partitions, args.repeat,
        ):
            print(_fmt(row))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
