"""Benchmark :class:`LocalPath` write + read across frameworks.

Measures the per-format codec + the yggdrasil bridge to
``pyarrow`` / ``polars`` / ``pandas`` when the holder is a real local
file. The Parquet / Arrow-IPC / CSV leaves all expose ``_local_path_str``
so the native readers can mmap the file directly.

Row count is kept small (``rows = 10_000``) so the bench finishes in
under a minute even at ``--repeat 5``. The companion
``bench_io_pushdown.py`` covers projection / filter pushdown on the
larger shapes where those features matter.

Usage::

    PYTHONPATH=src python benchmarks/bench_io_local.py
    PYTHONPATH=src python benchmarks/bench_io_local.py --repeat 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable

import pyarrow as pa

from yggdrasil.data.enums import Mode
from yggdrasil.data.options import CastOptions
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile
from yggdrasil.io.primitive.csv_file import CSVFile
from yggdrasil.io.primitive.parquet_file import ParquetFile


# Writes share the same target path across iterations, so use OVERWRITE
# to keep each write independent. Without this, AUTO mode appends +
# does a read-modify-rewrite and per-iteration cost grows linearly
# with iteration count — the bench would measure snowballing upsert
# work rather than per-write throughput.
_OVERWRITE = CastOptions(mode=Mode.OVERWRITE)


ROWS = 5_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _table(rows: int) -> pa.Table:
    return pa.table(
        {
            "id": pa.array(range(rows), type=pa.int64()),
            "amount": pa.array([1.5] * rows, type=pa.float64()),
            "qty": pa.array([2] * rows, type=pa.int32()),
            "name": pa.array(["x"] * rows, type=pa.string()),
            "ts": pa.array([dt.datetime(2024, 1, 1)] * rows,
                           type=pa.timestamp("us")),
            "active": pa.array([True] * rows, type=pa.bool_()),
        }
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    # Single warm-up call — full ``min(inner, ...)`` warm-ups multiply
    # the wall-clock cost on slow local-I/O benches.
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
        f"{r['label']:<62s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Per-format runner
# ---------------------------------------------------------------------------


def _bench_format(
    label: str,
    leaf_cls: type,
    filename: str,
    table: pa.Table,
    *,
    repeat: int,
    tmp: Path,
    inner: int,
) -> list[dict]:
    out: list[dict] = []
    path = tmp / filename

    # Seed the file so the read bench has stable bytes to load.
    leaf_cls(path=LocalPath(str(path))).write_arrow_table(table, _OVERWRITE)
    rows = table.num_rows

    def write_arrow():
        leaf_cls(path=LocalPath(str(path))).write_arrow_table(table, _OVERWRITE)
    def read_arrow():
        leaf_cls(path=LocalPath(str(path))).read_arrow_table()

    out.append(_time_one(
        f"local: {label} write_arrow_table rows={rows}",
        write_arrow, repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"local: {label} read_arrow_table rows={rows}",
        read_arrow, repeat=repeat, inner=inner,
    ))

    try:
        import polars  # noqa: F401
        pl_frame = leaf_cls(path=LocalPath(str(path))).read_polars_frame()

        def write_polars():
            leaf_cls(path=LocalPath(str(path))).write_polars_frame(pl_frame, _OVERWRITE)
        def read_polars():
            leaf_cls(path=LocalPath(str(path))).read_polars_frame()

        out.append(_time_one(
            f"local: {label} write_polars_frame rows={rows}",
            write_polars, repeat=repeat, inner=inner,
        ))
        out.append(_time_one(
            f"local: {label} read_polars_frame rows={rows}",
            read_polars, repeat=repeat, inner=inner,
        ))
    except ImportError:
        pass
    except Exception as e:
        out.append({
            "label": f"local: {label} polars SKIPPED ({type(e).__name__}: {e})",
            "best": 0.0, "median": 0.0, "mean": 0.0,
        })

    try:
        import pandas  # noqa: F401
        pd_frame = leaf_cls(path=LocalPath(str(path))).read_pandas_frame()

        def write_pandas():
            leaf_cls(path=LocalPath(str(path))).write_pandas_frame(pd_frame, _OVERWRITE)
        def read_pandas():
            leaf_cls(path=LocalPath(str(path))).read_pandas_frame()

        out.append(_time_one(
            f"local: {label} write_pandas_frame rows={rows}",
            write_pandas, repeat=repeat, inner=inner,
        ))
        out.append(_time_one(
            f"local: {label} read_pandas_frame rows={rows}",
            read_pandas, repeat=repeat, inner=inner,
        ))
    except ImportError:
        pass
    except Exception as e:
        out.append({
            "label": f"local: {label} pandas SKIPPED ({type(e).__name__}: {e})",
            "best": 0.0, "median": 0.0, "mean": 0.0,
        })

    return out


def scenarios(repeat: int, tmp: Path) -> list[dict]:
    out: list[dict] = []
    table = _table(ROWS)
    out.extend(_bench_format("parquet", ParquetFile, "data.parquet",
                             table, repeat=repeat, tmp=tmp, inner=5))
    out.extend(_bench_format("arrow-ipc", ArrowIPCFile, "data.arrow",
                             table, repeat=repeat, tmp=tmp, inner=5))
    out.extend(_bench_format("csv", CSVFile, "data.csv",
                             table, repeat=repeat, tmp=tmp, inner=3))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}  rows={ROWS}")
    print(f"# {'label':<62s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    with tempfile.TemporaryDirectory() as tmp:
        for row in scenarios(args.repeat, Path(tmp)):
            print(_fmt(row))


if __name__ == "__main__":
    main()
