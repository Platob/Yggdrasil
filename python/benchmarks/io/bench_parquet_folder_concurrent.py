"""Benchmark concurrent appends to one :class:`ParquetFolder`.

A :class:`~yggdrasil.parquet.ParquetFolder` write mints a fresh
``part-{epoch_ms}-{seed}.parquet`` leaf per call (see
:meth:`Folder.make_child`), so concurrent appends to the same folder are
**lock-free**: each writer lands its own part file and there is no shared
commit log / version counter to contend for — unlike
:class:`~yggdrasil.io.delta.DeltaFolder`, where every append races for the
next free commit version (see ``bench_delta_concurrency.py``). This bench
pins that property end to end.

``N`` threads each append ``M`` batches of ``R`` rows to **one** local
folder, in two layouts:

- **flat** — every part lands directly under the folder root. ``N*M``
  independent part files, zero cross-thread coordination.
- **hive** — the batch schema tags a low-cardinality ``region`` column
  ``partition_by``, so the writer fans each batch into ``region=<v>/``
  sub-folders. Threads now write part files **into shared partition
  directories** concurrently — the partition sub-folder is auto-created
  on first landing, so this stresses concurrent ``mkdir`` + part mint on
  the same paths.

Each layout is run **serial** (1 thread does all ``N*M`` appends) and
**concurrent** (``N`` threads do ``M`` each) for the same total work, so
the speedup column is the wall-clock win from running the GIL-releasing
parquet encode in parallel with no lock in the write path.

Correctness is asserted, not assumed: after every run the folder is read
back and

- ``rows == N*M*R`` — no append was lost or double-counted, and
- ``parts == N*M`` (flat) — every append produced exactly one part file,
  i.e. no two threads minted the same ``part-*`` name and clobbered each
  other. (Hive fans each append across the ``region`` partitions, so it
  lands ``N*M*_N_REGIONS`` parts — only the flat count is pinned exactly.)

A regression that loses a write (filename collision, dropped batch) fails
the run loudly instead of reporting a misleading throughput number.

Indicative (4 threads x 16 appends = 64 appends, 25k rows/append, local
tmpfs, median of 3 ``--repeat``, dev box)::

    flat   serial      wall 0.32s  cpu 0.32s  202.3 appends/s    64 parts
    flat   concurrent  wall 0.26s  cpu 0.38s  250.1 appends/s    64 parts  1.24x
    hive   serial      wall 0.61s  cpu 0.61s  105.0 appends/s   256 parts
    hive   concurrent  wall 0.46s  cpu 0.76s  139.5 appends/s   256 parts  1.33x

The headline is that concurrent appends **scale** (no lock-step on a
commit version) and stay **correct** (every part lands, every row reads
back) — the absolute wall clock is dominated by the local-FS part write
and the per-batch parquet encode, and the GIL-bound Python pre-amble per
part caps the parallel speedup short of linear.

Usage::

    PYTHONPATH=src python benchmarks/io/bench_parquet_folder_concurrent.py
    PYTHONPATH=src python benchmarks/io/bench_parquet_folder_concurrent.py \\
        --threads 8 --appends-per-thread 32 --rows 50000 --repeat 5
"""
from __future__ import annotations

import argparse
import os
import shutil
import statistics
import tempfile
import threading
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.enums import Mode
from yggdrasil.parquet import ParquetFolder, ParquetFolderOptions


#: Cardinality of the Hive ``region`` partition column. Kept well below
#: the thread count so multiple threads provably fan into the **same**
#: ``region=<v>/`` sub-folder concurrently — that overlap is the point of
#: the partitioned layout.
_N_REGIONS = 4
_REGIONS = [f"r{i}" for i in range(_N_REGIONS)]

#: Arrow schema whose ``region`` column carries the ``partition_by`` tag
#: the folder write path reads off the incoming batch (mirrors the
#: ``b"t:partition_by"`` field-metadata convention the folder tests use).
_HIVE_SCHEMA = pa.schema([
    pa.field("region", pa.utf8(), metadata={b"t:partition_by": b"True"}),
    pa.field("id", pa.int64()),
    pa.field("tid", pa.int64()),
])


def _flat_batch(base: int, rows: int, tid: int) -> pa.RecordBatch:
    """One ``rows``-row flat batch with globally unique ``id`` values."""
    return pa.record_batch(
        [
            pa.array(range(base, base + rows), pa.int64()),
            pa.array([tid] * rows, pa.int64()),
        ],
        names=["id", "tid"],
    )


def _hive_batch(base: int, rows: int, tid: int) -> pa.RecordBatch:
    """One ``rows``-row batch whose ``region`` column fans across
    ``_N_REGIONS`` partitions (round-robin on ``id``)."""
    ids = list(range(base, base + rows))
    regions = [_REGIONS[i % _N_REGIONS] for i in ids]
    return pa.record_batch(
        [
            pa.array(regions, pa.utf8()),
            pa.array(ids, pa.int64()),
            pa.array([tid] * rows, pa.int64()),
        ],
        schema=_HIVE_SCHEMA,
    )


def _count_parts(root: str) -> int:
    """Number of ``.parquet`` part files under *root* (skips ``.ygg``)."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        total += sum(1 for f in filenames if f.endswith(".parquet"))
    return total


def _run(*, threads: int, appends: int, rows: int, hive: bool) -> dict:
    """Fire ``threads`` writers, each doing ``appends`` APPENDs to one folder.

    ``threads=1`` is the serial baseline; the row id space is partitioned
    per (thread, append) so every batch is globally unique and the
    read-back row count is an exact lost-write check.
    """
    tmp = tempfile.mkdtemp(prefix="ygg_pqfolder_conc_")
    # Don't let a stale singleton (same tmp URL reused across repeats) hand
    # back a folder with a warm schema cache.
    ParquetFolder._INSTANCES.clear()
    make_batch = _hive_batch if hive else _flat_batch
    errors: list[BaseException] = []

    def _worker(tid: int) -> None:
        folder = ParquetFolder(path=tmp)
        opts = ParquetFolderOptions(mode=Mode.APPEND)
        for i in range(appends):
            base = (tid * appends + i) * rows
            try:
                folder.write_arrow_batches([make_batch(base, rows, tid)], options=opts)
            except BaseException as exc:  # noqa: BLE001 — surface in summary
                errors.append(exc)
                return

    try:
        t0 = time.perf_counter()
        c0 = time.process_time()
        if threads == 1:
            _worker(0)
        else:
            ts = [threading.Thread(target=_worker, args=(k,)) for k in range(threads)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
        if errors:
            raise errors[0]

        ParquetFolder._INSTANCES.clear()
        read_rows = ParquetFolder(path=tmp).read_arrow_table().num_rows
        parts = _count_parts(tmp)
        return {
            "wall": wall,
            "cpu": cpu,
            "appends": threads * appends,
            "rows": read_rows,
            "parts": parts,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _median(fn: Callable[[], dict], *, repeat: int) -> dict:
    samples = [fn() for _ in range(repeat)]
    return {
        "wall": statistics.median(s["wall"] for s in samples),
        "cpu": statistics.median(s["cpu"] for s in samples),
        "appends": samples[0]["appends"],
        "rows": samples[0]["rows"],
        "parts": int(statistics.median(s["parts"] for s in samples)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--appends-per-thread", type=int, default=16)
    ap.add_argument("--rows", type=int, default=25_000)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    n, m, r = args.threads, args.appends_per_thread, args.rows
    total_appends = n * m
    total_rows = total_appends * r

    print(
        f"\nParquetFolder concurrent-append bench: {n} threads x {m} appends "
        f"= {total_appends} appends x {r} rows = {total_rows} rows to one folder"
    )
    print("-" * 78)
    print(f"{'layout':<7}{'mode':<12}{'wall':>8}{'cpu':>8}{'appends/s':>11}"
          f"{'parts':>7}{'speedup':>9}")
    print("-" * 78)

    for hive in (False, True):
        layout = "hive" if hive else "flat"
        serial = _median(lambda: _run(threads=1, appends=total_appends,
                                       rows=r, hive=hive), repeat=args.repeat)
        concurrent = _median(lambda: _run(threads=n, appends=m,
                                          rows=r, hive=hive), repeat=args.repeat)

        # Correctness guard: every row must read back, every flat append must
        # have minted exactly one part (no clobbered filenames / lost writes).
        for label, res in (("serial", serial), ("concurrent", concurrent)):
            assert res["rows"] == total_rows, (
                f"{layout} {label}: read back {res['rows']} rows != {total_rows} "
                "— a concurrent append was lost or double-counted")
            if not hive:
                assert res["parts"] == total_appends, (
                    f"{layout} {label}: {res['parts']} part files != "
                    f"{total_appends} appends — a part-* name collided")

        for label, res in (("serial", serial), ("concurrent", concurrent)):
            tput = res["appends"] / res["wall"] if res["wall"] else 0.0
            speed = (serial["wall"] / res["wall"]
                     if label == "concurrent" and res["wall"] else None)
            speed_s = f"{speed:7.2f}x" if speed is not None else ""
            print(f"{layout:<7}{label:<12}{res['wall']:7.2f}s{res['cpu']:7.2f}s"
                  f"{tput:10.1f} {res['parts']:6d}{speed_s:>9}")
    print("-" * 78)
    print("correctness: every run read back all rows; every flat append minted "
          "one part.")


if __name__ == "__main__":
    main()
