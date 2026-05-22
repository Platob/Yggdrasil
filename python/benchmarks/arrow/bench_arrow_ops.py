"""Benchmark :mod:`yggdrasil.arrow.ops` — the pure-Arrow set / time ops.

The dedup / resample / upsert helpers here are the hot path for the
local-cache read pipeline (every batched cache hit funnels through
:meth:`CastOptions.dedup_arrow_batches` and
:meth:`CastOptions.resample_arrow_batches`) and for the partitioned
folder write path. The arrow.ops surface is pure pyarrow C++ kernels
— no python row walk — so the cost scales with the kernel's own
group-by + take rather than with row count, but the Python-side
overhead (column appends, schema rebinds) is the bench's target.

Coverage:

* :func:`dedup_arrow_table` / :func:`dedup_arrow_batches` —
  single-key, multi-key, no-dup short-circuit, large-N.
* :func:`resample_arrow_table` / :func:`resample_arrow_batches` —
  flat resample, partition-by resample, varying entity / row mix.
* Reference shapes — group_by directly, table.filter, table.take —
  so the bench surfaces whether arrow.ops adds material overhead
  above the underlying kernel.

Usage::

    PYTHONPATH=src python benchmarks/arrow/bench_arrow_ops.py
    PYTHONPATH=src python benchmarks/arrow/bench_arrow_ops.py --rows 10000 --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.arrow.ops import (
    dedup_arrow_batches,
    dedup_arrow_table,
    resample_arrow_batches,
    resample_arrow_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _dedup_table(rows: int, *, dup_ratio: float = 0.5) -> pa.Table:
    """Build a table whose ``id`` column has ``dup_ratio`` duplicates."""
    distinct = max(1, int(rows * (1.0 - dup_ratio)))
    ids = [i % distinct for i in range(rows)]
    return pa.table({
        "id": pa.array(ids),
        "a": pa.array(list(range(rows))),
        "b": pa.array(["x"] * rows),
    })


def _time_series_table(
    rows: int, *,
    entities: int = 1,
    step_sec: int = 60,
    epoch: dt.datetime = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
) -> pa.Table:
    """Build a time-series table for resample benches.

    Total rows: ``entities * (rows // entities)``. With ``entities=1`` it's
    a single timeline; with ``entities>1`` rows are interleaved on disk
    so the resample's partition_by branch can actually demonstrate its
    grouping cost.
    """
    per = max(1, rows // entities)
    all_rows: list[tuple[str, dt.datetime, int]] = []
    for sym_i in range(entities):
        sym = f"s{sym_i:04d}"
        for i in range(per):
            all_rows.append((sym, epoch + dt.timedelta(seconds=i * step_sec), i))
    syms = pa.array([r[0] for r in all_rows])
    ts = pa.array([r[1] for r in all_rows], type=pa.timestamp("us", "UTC"))
    vals = pa.array([r[2] for r in all_rows])
    return pa.table({"symbol": syms, "ts": ts, "v": vals})


def _to_batches(table: pa.Table) -> "list[pa.RecordBatch]":
    return table.to_batches()


# ---------------------------------------------------------------------------
# Time / format helpers (same shape as bench_arrow_cast.py)
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 50)):
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


def _dedup_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []

    # Single-key dedup, varying dup ratio. The Python overhead is
    # essentially constant; the kernel cost scales with rows.
    for ratio in (0.0, 0.1, 0.5, 0.9):
        table = _dedup_table(rows, dup_ratio=ratio)
        out.append(_time_one(
            f"dedup_arrow_table single-key rows={rows} dup={ratio:.1f}",
            lambda t=table: dedup_arrow_table(t, ["id"]),
            repeat=repeat, inner=20,
        ))

    # Multi-key dedup — the group_by cost scales with the cardinality
    # of the key tuple, not just the number of columns.
    multi = pa.table({
        "a": pa.array([i % 100 for i in range(rows)]),
        "b": pa.array([i % 7 for i in range(rows)]),
        "v": pa.array(list(range(rows))),
    })
    out.append(_time_one(
        f"dedup_arrow_table multi-key (a,b) rows={rows}",
        lambda t=multi: dedup_arrow_table(t, ["a", "b"]),
        repeat=repeat, inner=20,
    ))

    # No-dedup short-circuit (empty key list) — pays one isinstance check.
    no_dup = _dedup_table(rows, dup_ratio=0.0)
    out.append(_time_one(
        f"dedup_arrow_table empty keys (short-circuit) rows={rows}",
        lambda t=no_dup: dedup_arrow_table(t, []),
        repeat=repeat, inner=5_000,
    ))

    # Iterator wrapper — measures the batch materialise + re-batch
    # cost on top of the table op.
    dup = _dedup_table(rows, dup_ratio=0.5)
    batches = _to_batches(dup)
    out.append(_time_one(
        f"dedup_arrow_batches single-key rows={rows} (4 batches in)",
        lambda b=batches: list(dedup_arrow_batches(iter(b), ["id"])),
        repeat=repeat, inner=20,
    ))

    # Reference: bare ``group_by + take`` — same shape arrow.ops uses
    # internally. Surfaces the python-side overhead arrow.ops adds.
    def _bare_dedup(t: pa.Table) -> pa.Table:
        idx = t.append_column("__i", pa.array(range(t.num_rows)))
        g = idx.group_by(["id"], use_threads=False).aggregate([("__i", "first")])
        return t.take(g.column("__i_first").sort())

    out.append(_time_one(
        f"REF group_by+take rows={rows} dup=0.5",
        lambda t=dup: _bare_dedup(t),
        repeat=repeat, inner=20,
    ))

    return out


def _resample_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []

    # Flat resample, single timeline.
    flat = _time_series_table(rows, entities=1, step_sec=60)
    for sec in (60, 3600):  # PT1M (no-op grid) and PT1H (collapse 60×)
        out.append(_time_one(
            f"resample_arrow_table flat rows={rows} sampling={sec}s",
            lambda t=flat, s=sec: resample_arrow_table(
                t, time_column="ts", sampling_seconds=s,
            ),
            repeat=repeat, inner=20,
        ))

    # Partitioned resample at varying entity counts. The bucket
    # group-by gains a ``partition_by`` column prefix — the cost
    # is dominated by the wider group key, not by entity count
    # since the group_by is fully Arrow-side.
    for entities in (10, 100):
        per = _time_series_table(rows, entities=entities, step_sec=60)
        out.append(_time_one(
            f"resample_arrow_table partition_by=[symbol] rows={rows} entities={entities}",
            lambda t=per: resample_arrow_table(
                t, time_column="ts", sampling_seconds=3600,
                partition_by=["symbol"],
            ),
            repeat=repeat, inner=20,
        ))

    # Reference: bare ``group_by`` on the same shape — surfaces the
    # arrow.ops Python overhead.
    def _bare_resample(t: pa.Table, sec: int) -> pa.Table:
        ts_us = pa.compute.cast(t.column("ts"), pa.int64())
        bucket = pa.compute.multiply(
            pa.compute.divide_checked(ts_us, sec * 1_000_000),
            sec * 1_000_000,
        )
        idx = (t.append_column("__b", bucket)
                .append_column("__i", pa.array(range(t.num_rows))))
        g = idx.group_by(["__b"], use_threads=False).aggregate([("__i", "first")])
        return t.take(g.column("__i_first").sort())

    out.append(_time_one(
        f"REF bare bucket+group_by rows={rows} sampling=3600s",
        lambda t=flat: _bare_resample(t, 3600),
        repeat=repeat, inner=20,
    ))

    # Iterator wrapper at 4 batches in.
    flat_batches = _to_batches(flat)
    out.append(_time_one(
        f"resample_arrow_batches flat rows={rows} (4 batches in)",
        lambda b=flat_batches: list(resample_arrow_batches(
            iter(b), time_column="ts", sampling_seconds=3600,
        )),
        repeat=repeat, inner=20,
    ))

    # Zero-budget short-circuit (sampling_seconds<=0).
    out.append(_time_one(
        f"resample_arrow_table sampling=0 (short-circuit) rows={rows}",
        lambda t=flat: resample_arrow_table(t, time_column="ts", sampling_seconds=0),
        repeat=repeat, inner=10_000,
    ))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def scenarios(rows: int, repeat: int) -> list[dict]:
    return [
        *_dedup_scenarios(rows, repeat),
        *_resample_scenarios(rows, repeat),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000,
                    help="Row count for the in-memory fixture tables.")
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# rows={args.rows}  repeat={args.repeat}")
    print(f"# {'label':<70s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    for row in scenarios(args.rows, args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
