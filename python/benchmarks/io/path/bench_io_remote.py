"""Benchmark :class:`RemotePath` round-trips.

Remote backends need credentials so the bench is skipped by default.
Pass ``--remote-url`` pointing at a backend already configured in the
environment (S3 with AWS creds, Databricks Volumes with PAT, etc.)
to run it.

Measures the same round-trip shape as ``bench_io_local.py`` —
parquet write + read, across arrow / polars / pandas — so the
two benches give a clean apples-to-apples comparison of local vs
remote.

Usage::

    PYTHONPATH=src python benchmarks/io/path/bench_io_remote.py \\
        --remote-url s3://my-bucket/ygg-bench

    PYTHONPATH=src python benchmarks/io/path/bench_io_remote.py \\
        --remote-url dbfs:/Volumes/main/default/scratch/ygg-bench
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.io.path.path import Path
from yggdrasil.io.parquet_file import ParquetFile


ROWS = 5_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _table(rows: int) -> pa.Table:
    return pa.table(
        {
            "id": pa.array(range(rows), type=pa.int64()),
            "amount": pa.array([1.5] * rows, type=pa.float64()),
            "name": pa.array(["x"] * rows, type=pa.string()),
            "ts": pa.array([dt.datetime(2024, 1, 1)] * rows,
                           type=pa.timestamp("us")),
        }
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    # Single warm-up — remote ops are expensive; full warm-ups cost
    # real money on metered backends.
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
# Scenarios
# ---------------------------------------------------------------------------


def scenarios(repeat: int, remote_url: str) -> list[dict]:
    out: list[dict] = []
    table = _table(ROWS)

    base = Path.from_(remote_url)
    target = base.joinpath("ygg-bench-remote.parquet")

    def write_arrow():
        ParquetFile(path=target).write_arrow_table(table)
    def read_arrow():
        ParquetFile(path=target).read_arrow_table()

    out.append(_time_one(
        f"remote ({base.url.scheme}): parquet write_arrow rows={ROWS}",
        write_arrow, repeat=repeat, inner=3,
    ))
    out.append(_time_one(
        f"remote ({base.url.scheme}): parquet read_arrow rows={ROWS}",
        read_arrow, repeat=repeat, inner=3,
    ))

    try:
        import polars  # noqa: F401
        pl_frame = ParquetFile(path=target).read_polars_frame()
        def write_polars():
            ParquetFile(path=target).write_polars_frame(pl_frame)
        def read_polars():
            ParquetFile(path=target).read_polars_frame()
        out.append(_time_one(
            f"remote ({base.url.scheme}): parquet write_polars rows={ROWS}",
            write_polars, repeat=repeat, inner=3,
        ))
        out.append(_time_one(
            f"remote ({base.url.scheme}): parquet read_polars rows={ROWS}",
            read_polars, repeat=repeat, inner=3,
        ))
    except ImportError:
        pass
    except Exception as e:
        out.append({
            "label": f"remote: polars SKIPPED ({type(e).__name__}: {e})",
            "best": 0.0, "median": 0.0, "mean": 0.0,
        })

    try:
        import pandas  # noqa: F401
        pd_frame = ParquetFile(path=target).read_pandas_frame()
        def write_pandas():
            ParquetFile(path=target).write_pandas_frame(pd_frame)
        def read_pandas():
            ParquetFile(path=target).read_pandas_frame()
        out.append(_time_one(
            f"remote ({base.url.scheme}): parquet write_pandas rows={ROWS}",
            write_pandas, repeat=repeat, inner=3,
        ))
        out.append(_time_one(
            f"remote ({base.url.scheme}): parquet read_pandas rows={ROWS}",
            read_pandas, repeat=repeat, inner=3,
        ))
    except ImportError:
        pass
    except Exception as e:
        out.append({
            "label": f"remote: pandas SKIPPED ({type(e).__name__}: {e})",
            "best": 0.0, "median": 0.0, "mean": 0.0,
        })

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-url", required=True,
                    help=("Remote URL to bench against (e.g. "
                          "s3://bucket/prefix, dbfs:/Volumes/…). "
                          "Backend must already be configured in the env."))
    ap.add_argument("--repeat", type=int, default=3,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}  rows={ROWS}  remote_url={args.remote_url}")
    print(f"# {'label':<62s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    try:
        rows = scenarios(args.repeat, args.remote_url)
    except Exception as e:
        print(f"# bench aborted: {type(e).__name__}: {e}")
        return
    for row in rows:
        print(_fmt(row))


if __name__ == "__main__":
    main()
