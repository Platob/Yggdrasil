"""Memory / zero-copy profile of the Saga tabular paths.

Measures peak *Python* allocations (``tracemalloc``) and peak process RSS
(``ru_maxrss``) for the ways tabular bytes move through the result paths, to
surface where a full copy happens vs where Arrow keeps it zero-copy:

  * encode a result to Arrow IPC in one buffer  (a full encoded copy in RAM)
  * vs. the streaming encoder                    (bounded — one chunk at a time)
  * vs. spilling to disk then streaming off it    (the heavy path)
  * read a spilled file with ``pa.memory_map``    (zero-copy mmap)
  * vs. reading the bytes then ``open_stream``    (one full copy)
  * a bounded preview with CastOptions.row_limit  (reads only N rows)

Usage::  PYTHONPATH=src python benchmarks/saga/bench_saga_memory.py --rows 1000000
"""
from __future__ import annotations

import argparse
import resource
import tracemalloc
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from yggdrasil.node import transport


def _maxrss_mb() -> float:
    # ru_maxrss is KiB on Linux, bytes on macOS.
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 if kb > 1 << 20 else kb / 1024


def profile(label: str, fn) -> None:
    tracemalloc.start()
    rss0 = _maxrss_mb()
    out = fn()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss1 = _maxrss_mb()
    extra = f" → {out}" if out else ""
    print(f"  {label:46s}: py-peak {peak/1024/1024:7.1f} MB · rss+{max(0, rss1-rss0):6.1f} MB{extra}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    args = ap.parse_args()
    n = args.rows

    print(f"building a {n:,}-row Arrow table (4 cols)…")
    table = pa.table({
        "id": pa.array(range(n), type=pa.int64()),
        "px": pa.array([float(i % 1000) for i in range(n)], type=pa.float64()),
        "sym": pa.array([f"S{i % 500}" for i in range(n)], type=pa.string()),
        "qty": pa.array([i % 100 for i in range(n)], type=pa.int32()),
    })
    nbytes = table.nbytes
    print(f"in-memory table footprint: {nbytes/1024/1024:.1f} MB\n")

    tmp = Path(__file__).resolve().parent / "_mem_tmp.arrows"

    print("=== encode result → Arrow IPC ===")
    # One contiguous buffer: the whole encoded result lands in RAM at once.
    profile("write_arrow_stream_bytes (one buffer)",
            lambda: f"{len(transport.write_arrow_stream_bytes(table))/1024/1024:.0f} MB out")
    # Streaming encoder: drains per batch, so peak ≈ one chunk, not the whole result.
    def _stream():
        total = 0
        for chunk in transport.iter_arrow_ipc_stream(iter(table.to_batches(max_chunksize=65536)), table.schema):
            total += len(chunk)
        return f"{total/1024/1024:.0f} MB streamed"
    profile("iter_arrow_ipc_stream (streamed)", _stream)

    print("\n=== spill to disk, then stream off it ===")
    profile("write_arrow_ipc_file (batch-by-batch → disk)",
            lambda: f"{transport.write_arrow_ipc_file(str(tmp), iter(table.to_batches(max_chunksize=65536)), table.schema):,} rows")
    def _stream_file():
        total = 0
        for chunk in transport.iter_file_chunks(str(tmp)):
            total += len(chunk)
        return f"{total/1024/1024:.0f} MB off disk"
    profile("iter_file_chunks (stream off disk)", _stream_file)

    print("\n=== read a spilled file back ===")
    # Zero-copy: memory-map the file; pyarrow reads over the mapping, no Python copy.
    def _mmap():
        t = ipc.open_stream(pa.memory_map(str(tmp), "r")).read_all()
        return f"{t.num_rows:,} rows (mmap)"
    profile("pa.memory_map + open_stream (zero-copy)", _mmap)
    # Full copy: slurp the bytes into Python, then parse.
    def _bytes():
        data = Path(tmp).read_bytes()
        t = ipc.open_stream(data).read_all()
        return f"{t.num_rows:,} rows ({len(data)/1024/1024:.0f} MB copied)"
    profile("read_bytes + open_stream (full copy)", _bytes)

    print("\n=== bounded preview (CastOptions.row_limit pushdown) ===")
    from yggdrasil.arrow.tabular import ArrowTabular
    from yggdrasil.data.options import CastOptions
    src = ArrowTabular(table)
    profile("read 1k rows of a 1M-row result",
            lambda: f"{src.read_arrow_table(options=CastOptions(row_limit=1000)).num_rows} rows")

    tmp.unlink(missing_ok=True)
    print("\nTakeaway: the streaming encoder + disk spill + mmap read keep peak")
    print("memory near a single chunk, not the whole result — the heavy/remote path.")


if __name__ == "__main__":
    main()
