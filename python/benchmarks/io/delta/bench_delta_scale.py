"""Benchmark DeltaFolder scaling — does it stay cheap as the table grows?

Appends ``--commits`` batches of ``--rows`` rows each to one **local** Delta
table (no network), then probes the operations whose cost decides whether the
folder scales to huge volumes:

- **snapshot(fresh=True)** at growing commit counts. A naive log reader is
  O(commits) (replay every commit JSON from version 0); DeltaFolder writes a
  checkpoint every ``checkpoint_interval`` (10) commits, so a snapshot replays
  one checkpoint + the few commits after it — cost tracks *active files*
  (the checkpoint size), **not** the commit count. The sweep shows the
  snapshot time flattening relative to commits.
- **full read** throughput (rows/s) over all the part files.
- **pruned read** — a selective predicate on a per-file-stat'd column. Delta
  data-skipping drops non-overlapping files before opening any parquet, so a
  1/N-selective predicate reads ~1 file regardless of N.
- **incremental advance** — re-reading a cached snapshot after K new commits
  replays only those K commits (Snapshot.advanced), vs a full rebuild.

Example (commits=120, rows=5000 → 600k rows, dev box, local fs)::

    commits  cum_write_ms  snapshot_ms  active_files  log_files
          1             5          0.5             1          2
         10            39          9.6            10         12   <- first checkpoint
         40           250          4.2            40         45
        120           979          7.2           120        133
    full read   600000 rows  ~100 ms   (~5.8M rows/s)
    pruned read    5000 rows    ~9 ms   (read 1 of 120 files)
    incremental advance (last 20 commits)  ~2 ms  vs full snapshot ~7 ms

Snapshot stays in the single-digit-ms range while commits grow 100x — it's
O(active files via the checkpoint), the inherent Delta cost, not O(commits).

Usage::

    python benchmarks/io/delta/bench_delta_scale.py
    python benchmarks/io/delta/bench_delta_scale.py --commits 500 --rows 10000
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time

import pyarrow as pa

from yggdrasil.enums import Mode
from yggdrasil.saga.expr import Expression
from yggdrasil.io.delta import DeltaFolder, DeltaOptions


def _batch(i: int, rows: int) -> pa.Table:
    return pa.table({
        "id": pa.array(range(i * rows, (i + 1) * rows), pa.int64()),
        "g": pa.array([i] * rows, pa.int32()),       # one distinct value per file
        "v": pa.array([float(i)] * rows, pa.float64()),
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commits", type=int, default=120)
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--repeat", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()

    root = tempfile.mkdtemp(prefix="ygg_delta_scale_")
    try:
        d = DeltaFolder(path=root)
        # Sweep points (log-ish) capped at --commits.
        marks = sorted({1, 5, 10, 20, 40, 80, args.commits} & set(range(1, args.commits + 1)))

        print(f"\nDeltaFolder scale — {args.commits} commits x {args.rows} rows "
              f"({args.commits * args.rows:,} rows)\n")
        hdr = f"{'commits':>8} {'cum_write_ms':>13} {'snapshot_ms':>12} {'active_files':>13} {'log_files':>10}"
        print(hdr)
        print("-" * len(hdr))

        cum_write = 0.0
        for n in range(1, args.commits + 1):
            t = time.perf_counter()
            d.write_arrow_table(_batch(n, args.rows), mode=Mode.APPEND)
            cum_write += time.perf_counter() - t
            if n in marks:
                t = time.perf_counter()
                snap = d.snapshot(fresh=True)
                snap_ms = (time.perf_counter() - t) * 1000
                log_files = len(list((d.path / "_delta_log").iterdir()))
                print(f"{n:>8} {cum_write * 1000:>13.0f} {snap_ms:>12.1f} "
                      f"{snap.num_active_files():>13} {log_files:>10}")

        total_rows = args.commits * args.rows
        print()
        print(f"write throughput : {total_rows / cum_write:>12,.0f} rows/s")

        t = time.perf_counter()
        full = d.read_arrow_table()
        full_ms = (time.perf_counter() - t) * 1000
        print(f"full read        : {full.num_rows:>12,} rows  {full_ms:8.0f} ms  "
              f"({full.num_rows / (full_ms / 1000):,.0f} rows/s)")

        pred = Expression.from_sql(f"g = {args.commits // 2}")
        t = time.perf_counter()
        pruned = d.read_arrow_table(options=DeltaOptions(predicate=pred))
        pruned_ms = (time.perf_counter() - t) * 1000
        print(f"pruned read (1/N): {pruned.num_rows:>12,} rows  {pruned_ms:8.0f} ms  "
              f"(predicate g={args.commits // 2}, {d.snapshot(fresh=True).num_active_files()} files total)")

        # Incremental advance: base snapshot at an earlier version, advance by
        # the remaining commits — only those commits are replayed.
        snap = d.snapshot(fresh=True)
        base = Snapshot_from(d, max(0, args.commits - 21))
        t = time.perf_counter()
        adv = base.advanced(d.log, d.log.commits_after(base.version), snap.version)
        adv_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        d.refresh(); d.snapshot(fresh=True)
        full_snap_ms = (time.perf_counter() - t) * 1000
        print(f"incremental adv  : +{snap.version - base.version} commits  {adv_ms:8.1f} ms  "
              f"vs full snapshot {full_snap_ms:.1f} ms  (active={adv.num_active_files()})")
        print()
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def Snapshot_from(folder, version):
    from yggdrasil.io.delta.snapshot import Snapshot
    folder.log.invalidate()
    return Snapshot.from_log(folder.log, version)


if __name__ == "__main__":
    raise SystemExit(main())
