"""Benchmark Databricks-insert staging: cast + parquet write for pandas/polars/arrow.

The real arrow_insert flow is:

    pandas/polars/arrow source
        -> ParquetIO.write_table(data, CastOptions(target=...))
            -> _write_pandas_frame / _write_polars_frame / _write_arrow_table
                -> _write_arrow_batches  (cast each batch, write to parquet sink)
        -> staging.write_stream(buffer)   # network upload to Databricks Volume
        -> warehouse INSERT FROM staging

The network upload is out of scope here — what we measure is the in-process
part that hands a typed frame to ``ParquetIO`` with the table's target schema
bound on ``CastOptions``. That is where casts + per-batch dispatch live.

Usage::

    python benchmarks/bench_databricks_insert_staging.py
    python benchmarks/bench_databricks_insert_staging.py --rows 200000 --repeat 3
    python benchmarks/bench_databricks_insert_staging.py --shape nested

The script reports wall time per ``write_table`` call and bytes produced
(post-Parquet) so before/after numbers can be compared apples-to-apples.

A/B comparison (rows=200000, repeat=5, best ms, lower is better) — captured
locally to validate the optimizations land for the shapes Databricks loads
most often hit::

                          BEFORE      AFTER      delta
    scalar/pandas         78.99 ms    82.31 ms     +4%
    scalar/polars         81.37 ms    81.65 ms      0%
    scalar/arrow-table    76.22 ms    76.05 ms      0%
    nested/pandas        207.43 ms   164.65 ms    -21%
    nested/polars        124.01 ms   124.66 ms      0%
    nested/arrow-table   123.14 ms   121.07 ms     -2%
    deep/pandas         1102.81 ms   829.61 ms    -25%
    deep/polars          306.01 ms   311.66 ms     +2%
    deep/arrow-table     308.78 ms   283.78 ms     -8%

The wins concentrate on pandas inputs with nested / deep schemas — the
pre-optimization path was running type inference on object-dtype columns
inside ``pa.Table.from_pandas`` and then re-casting each batch to the
target schema afterwards. The optimization pushes the target schema into
``pa.Table.from_pandas(df, schema=...)`` so the C++ bridge does both
conversions in one pass.
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.primitive.parquet_io import ParquetIO


# ---------------------------------------------------------------------------
# Data shapes — chosen to exercise the staging cast paths that show up in real
# Databricks loads. ``scalar`` is the "plain table" case (numbers + strings +
# timestamps). ``nested`` adds list<int64> and struct columns — those have to
# round-trip through the per-column cast even on the fast paths because pandas
# can't natively express them, so they amplify any per-batch dispatch cost.
# ---------------------------------------------------------------------------


def _shape_scalar(rows: int) -> pa.Table:
    rng = np.random.default_rng(7)
    ts = np.datetime64("2024-01-01") + rng.integers(0, 86_400 * 365, size=rows).astype(
        "timedelta64[s]"
    )
    return pa.table(
        {
            "id": pa.array(np.arange(rows, dtype=np.int64)),
            "name": pa.array(np.array([f"name-{i}" for i in range(rows)], dtype=object)),
            "amount": pa.array(rng.normal(0.0, 1.0, size=rows)),
            "qty": pa.array(rng.integers(0, 1000, size=rows, dtype=np.int32)),
            "ts": pa.array(ts, type=pa.timestamp("us")),
            "active": pa.array(rng.integers(0, 2, size=rows, dtype=np.int8).astype(bool)),
        }
    )


def _shape_nested(rows: int) -> pa.Table:
    base = _shape_scalar(rows)
    rng = np.random.default_rng(11)
    tags = [list(rng.integers(0, 50, size=int(rng.integers(0, 4))).tolist()) for _ in range(rows)]
    attrs = [
        {"k": int(rng.integers(0, 1000)), "v": float(rng.normal())}
        for _ in range(rows)
    ]
    return base.append_column("tags", pa.array(tags, type=pa.list_(pa.int64()))).append_column(
        "attrs",
        pa.array(
            attrs,
            type=pa.struct([("k", pa.int64()), ("v", pa.float64())]),
        ),
    )


def _shape_deep(rows: int) -> pa.Table:
    """Deep-nested shape: list-of-struct, struct-of-list-of-struct, list-of-list.

    Hits every nested cast branch:

    * ``events`` — ``list<struct<id: int64, ts: timestamp, tag: string>>``
      (list of struct, the canonical telemetry shape)
    * ``profile`` — ``struct<name: string, scores: list<struct<k: string, v: double>>>``
      (struct holding a list-of-struct, two levels deep)
    * ``matrix`` — ``list<list<int32>>`` (list of list, hits the recursive
      child-cast path)
    """
    base = _shape_scalar(rows)
    rng = np.random.default_rng(17)

    # list<struct<...>>
    events = []
    for _ in range(rows):
        n = int(rng.integers(0, 5))
        events.append([
            {
                "id": int(rng.integers(0, 1_000_000)),
                # us-resolution datetime so pyarrow matches the target's
                # timestamp[us] without going through type inference.
                "ts": np.datetime64("2024-01-01") + np.timedelta64(
                    int(rng.integers(0, 86_400_000_000)), "us",
                ),
                "tag": f"e-{int(rng.integers(0, 200))}",
            }
            for _ in range(n)
        ])
    events_type = pa.list_(pa.struct([
        ("id", pa.int64()),
        ("ts", pa.timestamp("us")),
        ("tag", pa.string()),
    ]))

    # struct<name: string, scores: list<struct<k: string, v: double>>>
    profiles = []
    for _ in range(rows):
        m = int(rng.integers(0, 4))
        profiles.append({
            "name": f"u-{int(rng.integers(0, 1000))}",
            "scores": [
                {"k": f"k{int(rng.integers(0, 20))}", "v": float(rng.normal())}
                for _ in range(m)
            ],
        })
    profiles_type = pa.struct([
        ("name", pa.string()),
        ("scores", pa.list_(pa.struct([("k", pa.string()), ("v", pa.float64())]))),
    ])

    # list<list<int32>>
    matrix = []
    for _ in range(rows):
        rows_inner = int(rng.integers(0, 4))
        matrix.append([
            list(rng.integers(0, 100, size=int(rng.integers(0, 5))).astype(np.int32).tolist())
            for _ in range(rows_inner)
        ])
    matrix_type = pa.list_(pa.list_(pa.int32()))

    return (
        base
        .append_column("events", pa.array(events, type=events_type))
        .append_column("profile", pa.array(profiles, type=profiles_type))
        .append_column("matrix", pa.array(matrix, type=matrix_type))
    )


SHAPES: dict[str, Callable[[int], pa.Table]] = {
    "scalar": _shape_scalar,
    "nested": _shape_nested,
    "deep": _shape_deep,
}


# ---------------------------------------------------------------------------
# Bench runner — drives ParquetIO.write_table with a bound target schema,
# matching what ``Table.arrow_insert`` does before the volume upload.
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], int], repeat: int) -> dict:
    samples: list[float] = []
    bytes_out = 0
    for _ in range(repeat):
        t0 = time.perf_counter()
        bytes_out = fn()
        samples.append(time.perf_counter() - t0)
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "bytes": bytes_out,
        "samples": samples,
    }


def run_bench(rows: int, shape: str, repeat: int) -> list[dict]:
    table = SHAPES[shape](rows)
    target_field = Schema.from_arrow(table.schema).to_field()
    options = CastOptions(target=target_field)

    pandas_df = table.to_pandas()
    polars_df = pl.from_arrow(table)

    def write_pandas() -> int:
        with ParquetIO() as buf:
            buf.write_table(pandas_df, options)
            return buf.size

    def write_polars() -> int:
        with ParquetIO() as buf:
            buf.write_table(polars_df, options)
            return buf.size

    def write_arrow_table() -> int:
        with ParquetIO() as buf:
            buf.write_table(table, options)
            return buf.size

    def write_arrow_batches() -> int:
        with ParquetIO() as buf:
            buf.write_table(table.to_batches(max_chunksize=10_000), options)
            return buf.size

    return [
        _time_one(f"{shape}/pandas", write_pandas, repeat),
        _time_one(f"{shape}/polars", write_polars, repeat),
        _time_one(f"{shape}/arrow-table", write_arrow_table, repeat),
        _time_one(f"{shape}/arrow-batches", write_arrow_batches, repeat),
    ]


def _fmt_row(r: dict) -> str:
    return (
        f"{r['label']:>26s}  "
        f"best={r['best']*1000:8.2f} ms  "
        f"median={r['median']*1000:8.2f} ms  "
        f"mean={r['mean']*1000:8.2f} ms  "
        f"bytes={r['bytes']:>10d}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument(
        "--shape", choices=list(SHAPES) + ["all"], default="all",
        help="Data shape to exercise (scalar / nested / all).",
    )
    args = ap.parse_args()

    shapes = list(SHAPES) if args.shape == "all" else [args.shape]

    print(f"# rows={args.rows} repeat={args.repeat}")
    print(f"# {'label':>26s}  {'best':>12s}  {'median':>12s}  {'mean':>12s}  bytes")
    for shape in shapes:
        for row in run_bench(args.rows, shape, args.repeat):
            print(_fmt_row(row))


if __name__ == "__main__":
    main()
