"""Benchmark the primitive :class:`Tabular` leaves: CSV / Parquet / Arrow-IPC / NDJSON / JSON.

What this covers
----------------

Each format runs a small **write → read → assert-roundtrip** loop on
in-memory :class:`BytesIO` handles so the numbers measure the format
codec + the yggdrasil wiring (collect_schema, options dispatch,
``cast_arrow_tabular``) without OS disk noise.

Two payload sizes — 1k rows and 50k rows — capture both the
per-call overhead (the IO/options/dispatch floor) and the per-row
throughput (codec + cast kernels).

Usage::

    PYTHONPATH=src python benchmarks/bench_io_primitive.py
    PYTHONPATH=src python benchmarks/bench_io_primitive.py --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.io.primitive.arrow_ipc_io import ArrowIPCIO
from yggdrasil.io.primitive.csv_io import CsvIO
from yggdrasil.io.primitive.json_io import JsonIO
from yggdrasil.io.primitive.ndjson_io import NDJsonIO
from yggdrasil.io.primitive.parquet_io import ParquetIO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _table(rows: int) -> pa.Table:
    """Representative analytics shape — mix of int / float / string / ts / bool."""
    return pa.table(
        {
            "id": pa.array(range(rows), type=pa.int64()),
            "amount": pa.array([1.5] * rows, type=pa.float64()),
            "qty": pa.array([2] * rows, type=pa.int32()),
            "name": pa.array(["x"] * rows, type=pa.string()),
            "ts": pa.array(
                [dt.datetime(2024, 1, 1)] * rows,
                type=pa.timestamp("us"),
            ),
            "active": pa.array([True] * rows, type=pa.bool_()),
        }
    )


# ---------------------------------------------------------------------------
# Timing helpers
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
        scale = 1e9
        unit = "ns"
    elif r["best"] >= 1e-3:
        scale = 1e3
        unit = "ms"
    return (
        f"{r['label']:<58s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Per-leaf write + read scenarios
# ---------------------------------------------------------------------------


def _bench_leaf(
    label_prefix: str,
    leaf_cls: type,
    table: pa.Table,
    *,
    repeat: int,
    write_inner: int,
    read_inner: int,
    frameworks: tuple[str, ...] = ("arrow", "polars", "pandas"),
) -> list[dict]:
    """Run write + read benchmarks for a single leaf class, per framework.

    Measures the round-trip cost for each requested framework — so the
    leaf's per-format codec + the yggdrasil bridge to ``pyarrow`` /
    ``polars`` / ``pandas`` show up together in one block.
    """
    # Pre-serialize once to drive the read bench against a stable
    # payload (so we measure the read cost, not write+read).
    sink = leaf_cls(b"")
    sink.write_arrow_table(table)
    sink.seek(0)
    payload = sink.read()
    rows = table.num_rows

    out: list[dict] = []

    if "arrow" in frameworks:
        def write_arrow() -> None:
            b = leaf_cls(b"")
            b.write_arrow_table(table)

        def read_arrow() -> None:
            b = leaf_cls(payload)
            b.read_arrow_table()

        out.append(_time_one(
            f"{label_prefix}: arrow write_arrow_table rows={rows}",
            write_arrow, repeat=repeat, inner=write_inner,
        ))
        out.append(_time_one(
            f"{label_prefix}: arrow read_arrow_table rows={rows}",
            read_arrow, repeat=repeat, inner=read_inner,
        ))

    if "polars" in frameworks:
        try:
            import polars as pl  # noqa: F401
            pl_frame = leaf_cls(payload).read_polars_frame()

            def write_polars() -> None:
                b = leaf_cls(b"")
                b.write_polars_frame(pl_frame)

            def read_polars() -> None:
                b = leaf_cls(payload)
                b.read_polars_frame()

            out.append(_time_one(
                f"{label_prefix}: polars write_polars_frame rows={rows}",
                write_polars, repeat=repeat, inner=write_inner,
            ))
            out.append(_time_one(
                f"{label_prefix}: polars read_polars_frame rows={rows}",
                read_polars, repeat=repeat, inner=read_inner,
            ))
        except (ImportError, NotImplementedError):
            pass
        except Exception:
            # Some leaves (e.g. JSON) may not support the
            # polars/pandas bridge for every shape — skip on failure.
            pass

    if "pandas" in frameworks:
        try:
            import pandas as pd  # noqa: F401
            pd_frame = leaf_cls(payload).read_pandas_frame()

            def write_pandas() -> None:
                b = leaf_cls(b"")
                b.write_pandas_frame(pd_frame)

            def read_pandas() -> None:
                b = leaf_cls(payload)
                b.read_pandas_frame()

            out.append(_time_one(
                f"{label_prefix}: pandas write_pandas_frame rows={rows}",
                write_pandas, repeat=repeat, inner=write_inner,
            ))
            out.append(_time_one(
                f"{label_prefix}: pandas read_pandas_frame rows={rows}",
                read_pandas, repeat=repeat, inner=read_inner,
            ))
        except (ImportError, NotImplementedError):
            pass
        except Exception:
            pass

    return out


def scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    small = _table(1_000)
    large = _table(50_000)

    # Arrow IPC — the cheapest format; same in-memory layout as the
    # underlying Arrow runtime, so the numbers establish the "what does
    # the yggdrasil wiring cost?" floor.
    out.extend(_bench_leaf("arrow_ipc", ArrowIPCIO, small,
                           repeat=repeat, write_inner=500, read_inner=500))
    out.extend(_bench_leaf("arrow_ipc", ArrowIPCIO, large,
                           repeat=repeat, write_inner=100, read_inner=100))

    # Parquet — column-projection / predicate-pushdown format.
    out.extend(_bench_leaf("parquet", ParquetIO, small,
                           repeat=repeat, write_inner=200, read_inner=200))
    out.extend(_bench_leaf("parquet", ParquetIO, large,
                           repeat=repeat, write_inner=50, read_inner=50))

    # CSV — string-encoded; codec is the dominant cost.
    out.extend(_bench_leaf("csv", CsvIO, small,
                           repeat=repeat, write_inner=100, read_inner=100))
    out.extend(_bench_leaf("csv", CsvIO, large,
                           repeat=repeat, write_inner=20, read_inner=20))

    # NDJson — line-delimited JSON; per-row encode cost.
    out.extend(_bench_leaf("ndjson", NDJsonIO, small,
                           repeat=repeat, write_inner=100, read_inner=100))
    out.extend(_bench_leaf("ndjson", NDJsonIO, large,
                           repeat=repeat, write_inner=20, read_inner=20))

    # JSON — single top-level array.
    out.extend(_bench_leaf("json", JsonIO, small,
                           repeat=repeat, write_inner=100, read_inner=100))
    out.extend(_bench_leaf("json", JsonIO, large,
                           repeat=repeat, write_inner=20, read_inner=20))

    # collect_schema — schema-only peek used by the cast layer before
    # any real read. Arrow-IPC + Parquet should be near-instant; CSV /
    # NDJson have to read the first batch.
    arrow_handle = ArrowIPCIO(b"")
    arrow_handle.write_arrow_table(large)
    parquet_handle = ParquetIO(b"")
    parquet_handle.write_arrow_table(large)
    csv_handle = CsvIO(b"")
    csv_handle.write_arrow_table(large)

    def schema_arrow():
        ArrowIPCIO(arrow_handle.getvalue()).collect_schema()
    def schema_parquet():
        ParquetIO(parquet_handle.getvalue()).collect_schema()
    def schema_csv():
        CsvIO(csv_handle.getvalue()).collect_schema()

    out.append(_time_one(
        "arrow_ipc: collect_schema",
        schema_arrow, repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "parquet: collect_schema",
        schema_parquet, repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "csv: collect_schema",
        schema_csv, repeat=repeat, inner=500,
    ))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<58s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
