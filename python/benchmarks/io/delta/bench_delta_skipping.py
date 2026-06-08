"""Benchmark Delta data-skipping — per-file stats pruning on read.

Builds a many-file local Delta table (one AddFile per commit, each with
a disjoint ``id`` range so the per-file min/max stats are non-overlapping)
and measures a selective-predicate scan with stats data-skipping ON vs
OFF. With skipping ON, files whose min/max can't satisfy the predicate
are dropped before any parquet is opened; with it OFF, every file is read
and the rows are filtered afterwards.

The win scales with selectivity: a predicate that lands in one file out
of N reads ~1/N of the bytes.

Before / after (64 files x 25k rows = 1.6M rows, predicate selects
~1 file, median of 5 ``--repeat``)::

    full scan (no predicate)     :  83.08 ms   (reads all 64 files)
    selective scan, skipping OFF : 121.70 ms   (reads all 64 files + filters)
    selective scan, skipping ON  :   7.33 ms   (reads 1 file)
                                   ----------
                                   ~16.6x faster on a 1/64-selective predicate

Numbers vary by machine; run it locally to capture your own. Skipping
never changes the result set — only how many files are touched (the
no-regression + correctness guards live in
``tests/test_yggdrasil/test_io/test_delta/test_delta_databricks_compat.py``).

Usage::

    PYTHONPATH=src python benchmarks/io/delta/bench_delta_skipping.py
    PYTHONPATH=src python benchmarks/io/delta/bench_delta_skipping.py \\
        --files 64 --rows-per-file 25000 --repeat 5
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from typing import Callable

import pyarrow as pa

import yggdrasil.io.delta.delta_folder as delta_folder
from yggdrasil.saga.expr import col
from yggdrasil.io.delta import DeltaFolder, DeltaOptions


def _build(root: str, files: int, rows_per_file: int) -> DeltaFolder:
    d = DeltaFolder(path=root)
    for f in range(files):
        base = f * rows_per_file * 10  # gap so ranges never overlap
        ids = pa.array(range(base, base + rows_per_file), pa.int64())
        d.write_arrow_table(pa.table({
            "id": ids,
            "name": pa.array([f"row_{i % 997}" for i in range(rows_per_file)],
                             pa.string()),
            "value": pa.array([float(i) * 1.5 for i in range(rows_per_file)],
                              pa.float64()),
        }))
    return d


def _time(label: str, fn: Callable[[], int], *, repeat: int) -> dict:
    fn()  # warm
    samples = []
    rows = 0
    for _ in range(repeat):
        t0 = time.perf_counter()
        rows = fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"label": label, "rows": rows,
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.fmean(samples)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=64)
    ap.add_argument("--rows-per-file", type=int, default=25_000)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="ygg_delta_skip_")
    try:
        d = _build(tmp, args.files, args.rows_per_file)
        # Selective predicate: lands inside the last file's id range only.
        last_base = (args.files - 1) * args.rows_per_file * 10
        pred = (col("id") >= last_base) & (col("id") < last_base + args.rows_per_file)

        def _scan() -> int:
            return d.refresh().read_arrow_table(
                options=DeltaOptions(predicate=pred)).num_rows

        # Skipping ON (default behavior).
        on = _time("selective scan, skipping ON", _scan, repeat=args.repeat)

        # Skipping OFF — neutralize the stats filter so every file is read.
        orig = delta_folder._data_skip_adds
        delta_folder._data_skip_adds = lambda snap, adds, predicate: iter(adds)
        try:
            off = _time("selective scan, skipping OFF", _scan, repeat=args.repeat)
        finally:
            delta_folder._data_skip_adds = orig

        full = _time(
            "full scan (no predicate)",
            lambda: d.refresh().read_arrow_table().num_rows,
            repeat=args.repeat,
        )

        print(f"\nDelta data-skipping bench: {args.files} files x "
              f"{args.rows_per_file} rows ({args.files * args.rows_per_file} total)")
        print("-" * 64)
        for r in (full, off, on):
            print(f"{r['label']:<34} {r['median_ms']:8.2f} ms  "
                  f"(rows={r['rows']})")
        if on["median_ms"] > 0:
            print("-" * 64)
            print(f"speedup (OFF/ON): {off['median_ms'] / on['median_ms']:.1f}x")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
