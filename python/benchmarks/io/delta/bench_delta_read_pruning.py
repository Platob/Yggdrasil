"""Benchmark DeltaFolder column pushdown — peak RSS + time, full vs 1 column.

Reads a **wide** local-filesystem Delta table (many numeric columns plus one
fat string column) two ways and compares:

* ``full``  — read every column (the pre-pushdown behaviour: the leaf parquet
  was opened with all columns, then the projection dropped them *after* the
  decode).
* ``one``   — read a single narrow column. After Task A, the projection is
  pushed into each parquet leaf (``columns=``), so only that column's chunks
  are decoded off disk.

Four surfaces, each funnelling through the pruned Arrow read:

    to_arrow         DeltaFolder.read_arrow_table(columns=…)
    to_polars        DeltaFolder.read_polars_frame(columns=…)
    scan_polars      DeltaFolder.scan_polars_frame().select(…).collect()
    to_pandas        DeltaFolder.read_pandas_frame(columns=…)

Memory is the **net RSS increase while the result is held** (subprocess per
op, like ``benchmarks/io/bench_tabular_conversions.py``) — robust to a pooling
allocator. The ``one`` column read should both return only that column *and*
cut peak RSS + bytes decoded sharply vs ``full``.

BEFORE = ``full`` read (every column off disk, then project).
AFTER  = ``one``  read (column pushdown into the parquet leaf).

Measured (rows=1_000_000, cols=12 incl. one fat string, repeat=3,
best time / median net RSS), local fs, no creds::

    surface        BEFORE (full)        AFTER (one col)      win
    to_arrow      297.4 MB / 0.37s     18.0 MB / 0.05s      -94% mem, -88% t
    to_polars     337.5 MB / 0.47s     43.0 MB / 0.15s      -87% mem, -68% t
    scan_polars   338.7 MB / 0.46s     50.3 MB / 0.16s      -85% mem, -64% t
    to_pandas     362.9 MB / 0.40s     26.5 MB / 0.05s      -93% mem, -87% t

The fat string column dominates the on-disk + decoded footprint, so reading
one narrow numeric column collapses both peak memory and time to the cost of
that single column's chunks. ``to_arrow`` / ``to_pandas`` land lowest (the
pandas read self-destructs the Arrow table as it converts); the polars
surfaces pay a little more for per-batch frame materialization but still cut
peak by ~6×.

Usage::

    python benchmarks/io/delta/bench_delta_read_pruning.py
    python benchmarks/io/delta/bench_delta_read_pruning.py --rows 1000000 --repeat 7
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import pyarrow as pa

from yggdrasil.enums import Mode
from yggdrasil.io.delta.delta_folder import DeltaFolder

# The narrow column we project to in the ``one`` read.
_PROJECT = "n0"

_SURFACES = {
    "to_arrow": lambda d, cols: d.read_arrow_table(columns=cols),
    "to_polars": lambda d, cols: d.read_polars_frame(columns=cols),
    "scan_polars": lambda d, cols: (
        d.scan_polars_frame().select(*cols).collect() if cols
        else d.scan_polars_frame().collect()
    ),
    "to_pandas": lambda d, cols: d.read_pandas_frame(columns=cols),
}


def _rss_bytes() -> int:
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * 4096


def _wide_table(rows: int, n_numeric: int = 11):
    data = {f"n{i}": pa.array(range(i, rows + i)) for i in range(n_numeric)}
    # One fat string column — dominates the decoded + on-disk footprint, so
    # projecting it away is where the memory win shows up.
    data["fat"] = pa.array(["payload-" + "z" * 96 + f"-{i}" for i in range(rows)])
    return pa.table(data)


def _build_table(path: str, rows: int) -> DeltaFolder:
    d = DeltaFolder(path=path)
    base = _wide_table(rows)
    # A few row groups → realistic multi-file/multi-chunk read-back.
    step = max(1, rows // 8)
    for off in range(0, rows, step):
        d.write_arrow_table(base.slice(off, min(step, rows - off)), mode=Mode.APPEND)
    return d


def _worker(surface: str, mode: str, rows: int, table_dir: str) -> None:
    """Run one read; print ``{time_s, net_mb}`` JSON.

    Net memory = RSS held with the result alive minus the pre-read baseline —
    the read output's marginal footprint, robust to a pooling allocator.
    """
    d = DeltaFolder(path=table_dir)
    cols = None if mode == "full" else [_PROJECT]
    op = _SURFACES[surface]

    gc.collect()
    before = _rss_bytes()
    start = time.perf_counter()
    out = op(d, cols)
    elapsed = time.perf_counter() - start
    gc.collect()
    after = _rss_bytes()

    # Keep the result alive across the measurement.
    assert out is not None
    n_cols = (out.num_columns if hasattr(out, "num_columns")
              else (out.shape[1] if hasattr(out, "shape") else len(out.columns)))
    print(json.dumps({
        "time_s": elapsed,
        "net_mb": max(0.0, (after - before) / 1e6),
        "n_cols": n_cols,
    }))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--surface", choices=list(_SURFACES), help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("full", "one"), help=argparse.SUPPRESS)
    parser.add_argument("--table-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.surface:  # worker mode
        _worker(args.surface, args.mode, args.rows, args.table_dir)
        return 0

    work = tempfile.mkdtemp(prefix="ygg-delta-prune-bench-")
    table_dir = os.path.join(work, "wide")
    try:
        _build_table(table_dir, args.rows)
        print(
            f"\nDeltaFolder read pruning (rows={args.rows:,}, repeat={args.repeat}, "
            f"best time / median net RSS)\n"
        )
        header = f"{'surface':>14} {'mode':>6} {'best_s':>9} {'net_MB':>9} {'cols':>5}"
        print(header)
        print("-" * len(header))
        for surface in _SURFACES:
            for mode in ("full", "one"):
                times: list[float] = []
                mems: list[float] = []
                cols_seen = 0
                for _ in range(args.repeat):
                    proc = subprocess.run(
                        [sys.executable, os.path.abspath(__file__),
                         "--surface", surface, "--mode", mode,
                         "--rows", str(args.rows), "--table-dir", table_dir],
                        capture_output=True, text=True, check=True,
                    )
                    rec = json.loads(proc.stdout.strip().splitlines()[-1])
                    times.append(rec["time_s"])
                    mems.append(rec["net_mb"])
                    cols_seen = rec["n_cols"]
                tag = "BEFORE" if mode == "full" else "AFTER"
                print(f"{surface:>14} {tag:>6} {min(times):>9.3f} "
                      f"{statistics.median(mems):>9.1f} {cols_seen:>5}")
            print()
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
