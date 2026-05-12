"""Benchmark the nested :class:`Tabular` leaves: FolderIO + ZipIO.

What this covers
----------------

Both leaves aggregate many tabular children behind a single Tabular
surface, so the interesting paths are:

* ``list_entries`` / ``iter_children`` — directory walk overhead.
* ``collect_schema`` — schema peek (should hit the first child only).
* ``read_arrow_table`` — concatenated read across every child.
* Per-framework variants (arrow / polars / pandas) — measures the
  bridge from the aggregated Arrow read through to the engine frame.

ZipIO uses a self-built in-memory archive (no disk). FolderIO uses a
temp directory of parquet files.

Usage::

    PYTHONPATH=src python benchmarks/bench_io_nested.py
    PYTHONPATH=src python benchmarks/bench_io_nested.py --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import statistics
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.io.nested.folder_io import FolderIO
from yggdrasil.io.nested.zip_io import ZipIO
from yggdrasil.io.primitive.parquet_io import ParquetIO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _table(rows: int, seed: int = 0) -> pa.Table:
    return pa.table(
        {
            "id": pa.array(range(seed, seed + rows), type=pa.int64()),
            "amount": pa.array([1.5] * rows, type=pa.float64()),
            "name": pa.array(["x"] * rows, type=pa.string()),
            "ts": pa.array(
                [dt.datetime(2024, 1, 1)] * rows,
                type=pa.timestamp("us"),
            ),
        }
    )


def _build_zip_payload(files: int, rows_per_file: int) -> bytes:
    """Build an in-memory zip with N parquet entries, returned as bytes."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_STORED) as zf:
        for i in range(files):
            inner = io.BytesIO()
            pq.write_table(_table(rows_per_file, seed=i * rows_per_file), inner)
            zf.writestr(f"part-{i:03d}.parquet", inner.getvalue())
    return bio.getvalue()


def _build_folder(folder: Path, files: int, rows_per_file: int) -> None:
    """Write N parquet files into *folder*."""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(files):
        pq.write_table(
            _table(rows_per_file, seed=i * rows_per_file),
            str(folder / f"part-{i:03d}.parquet"),
        )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 20)):
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
        f"{r['label']:<58s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Zip scenarios
# ---------------------------------------------------------------------------


def _zip_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    payload_8x1k = _build_zip_payload(files=8, rows_per_file=1_000)
    payload_8x50k = _build_zip_payload(files=8, rows_per_file=50_000)

    out.append(_time_one(
        "zip: list_entries 8 files",
        lambda: ZipIO(payload_8x1k).list_entries(),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "zip: collect_schema 8 files",
        lambda: ZipIO(payload_8x1k).collect_schema(),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        "zip: read_arrow_table 8x1k",
        lambda: ZipIO(payload_8x1k).read_arrow_table(),
        repeat=repeat, inner=100,
    ))
    out.append(_time_one(
        "zip: read_arrow_table 8x50k",
        lambda: ZipIO(payload_8x50k).read_arrow_table(),
        repeat=repeat, inner=10,
    ))

    try:
        import polars  # noqa: F401
        out.append(_time_one(
            "zip: read_polars_frame 8x1k",
            lambda: ZipIO(payload_8x1k).read_polars_frame(),
            repeat=repeat, inner=100,
        ))
        out.append(_time_one(
            "zip: read_polars_frame 8x50k",
            lambda: ZipIO(payload_8x50k).read_polars_frame(),
            repeat=repeat, inner=10,
        ))
    except ImportError:
        pass

    try:
        import pandas  # noqa: F401
        out.append(_time_one(
            "zip: read_pandas_frame 8x1k",
            lambda: ZipIO(payload_8x1k).read_pandas_frame(),
            repeat=repeat, inner=100,
        ))
        out.append(_time_one(
            "zip: read_pandas_frame 8x50k",
            lambda: ZipIO(payload_8x50k).read_pandas_frame(),
            repeat=repeat, inner=10,
        ))
    except ImportError:
        pass

    return out


# ---------------------------------------------------------------------------
# Folder scenarios
# ---------------------------------------------------------------------------


def _folder_scenarios(repeat: int, tmp_root: str) -> list[dict]:
    out: list[dict] = []
    small_dir = Path(tmp_root) / "small"
    large_dir = Path(tmp_root) / "large"
    _build_folder(small_dir, files=8, rows_per_file=1_000)
    _build_folder(large_dir, files=8, rows_per_file=50_000)

    out.append(_time_one(
        "folder: iter_children 8 files",
        lambda: list(FolderIO(path=small_dir).iter_children()),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        "folder: collect_schema 8 files",
        lambda: FolderIO(path=small_dir).collect_schema(),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        "folder: read_arrow_table 8x1k",
        lambda: FolderIO(path=small_dir).read_arrow_table(),
        repeat=repeat, inner=100,
    ))
    out.append(_time_one(
        "folder: read_arrow_table 8x50k",
        lambda: FolderIO(path=large_dir).read_arrow_table(),
        repeat=repeat, inner=20,
    ))

    try:
        import polars  # noqa: F401
        out.append(_time_one(
            "folder: read_polars_frame 8x1k",
            lambda: FolderIO(path=small_dir).read_polars_frame(),
            repeat=repeat, inner=100,
        ))
        out.append(_time_one(
            "folder: read_polars_frame 8x50k",
            lambda: FolderIO(path=large_dir).read_polars_frame(),
            repeat=repeat, inner=20,
        ))
    except ImportError:
        pass

    try:
        import pandas  # noqa: F401
        out.append(_time_one(
            "folder: read_pandas_frame 8x1k",
            lambda: FolderIO(path=small_dir).read_pandas_frame(),
            repeat=repeat, inner=100,
        ))
        out.append(_time_one(
            "folder: read_pandas_frame 8x50k",
            lambda: FolderIO(path=large_dir).read_pandas_frame(),
            repeat=repeat, inner=20,
        ))
    except ImportError:
        pass

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def scenarios(repeat: int, tmp_root: str) -> list[dict]:
    return [
        *_zip_scenarios(repeat),
        *_folder_scenarios(repeat, tmp_root),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<58s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    with tempfile.TemporaryDirectory() as tmp:
        for row in scenarios(args.repeat, tmp):
            print(_fmt(row))


if __name__ == "__main__":
    main()
