"""Benchmark the eager ``to_*`` conversions on a :class:`Tabular`.

Measures wall time **and marginal memory** for the three conversion exits
that every backend (parquet, csv, warehouse, …) funnels through after the
Arrow table is assembled:

    tabular.to_arrow_table()   ->  pa.Table
    tabular.to_polars()        ->  pl.DataFrame
    tabular.to_pandas()        ->  pandas.DataFrame

The source is an in-memory :class:`ArrowTabular` (no network), so the
numbers isolate the conversion cost from any read / transport.

Two memory levers under test:

* ``to_polars`` ingests with ``rechunk=False`` — polars no longer memcpy's
  every column into one contiguous chunk, so numeric columns stay
  zero-copy views over the Arrow buffers.
* ``to_pandas`` converts with ``split_blocks=True`` — pandas skips the
  same-type block consolidation copy; numeric, null-free columns alias the
  Arrow buffers instead of being copied into a fresh 2-D block.

Memory is reported as the **net RSS increase while the source tabular and
the conversion result are both alive** — i.e. the marginal footprint of the
conversion's output. This sidesteps the pooling-allocator trap (freed pages
stay mapped, so an in-process transient-peak sampler reads ~0): with the
source held, a copying conversion *must* grow RSS by ~the result size,
while a zero-copy conversion grows ~0. Each op runs in a fresh subprocess so
the baseline is clean.

A/B (rows=3_000_000, repeat=5, best time / median net RSS). Each win shows
on the backend that exercises it — ``to_polars`` (rechunk) on the multi-chunk
in-memory holder, ``to_pandas`` (self-destruct) on the owned parquet read::

    backend   op            BEFORE              AFTER             win
    arrow     to_polars   203.7 MB / 0.156s   70.7 MB / 0.120s  -65% mem, -23% t
    arrow     to_pandas    96.8 MB / 0.026s   96.9 MB / 0.027s  unchanged (cached:
                                                                self-destruct off)
    parquet   to_pandas   136.1 MB / 0.102s   56.7 MB / 0.083s  -58% mem, -19% t

``flat_table`` is numeric-heavy with one low-cardinality string column, so
the optimized conversions collapse toward roughly just the columns that
can't alias the Arrow buffers, while the originals copy every column.

Usage::

    python benchmarks/io/bench_tabular_conversions.py
    python benchmarks/io/bench_tabular_conversions.py --rows 1000000 --repeat 7
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time

# Sibling-module import that works both when run directly and via
# ``run_all.py`` (which spawns the file by absolute path).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.io.parquet_file import ParquetFile

from _common import flat_table  # type: ignore[import-not-found]


_OPS = {
    "to_arrow_table": lambda t: t.to_arrow_table(),
    "to_polars": lambda t: t.to_polars(),
    "to_pandas": lambda t: t.to_pandas(),
}

# ``arrow``  — in-memory holder, returns a cached table by reference
#              (``_READ_TABLE_OWNED = False``): the to_pandas self-destruct
#              path is *off*, so it isolates the to_polars rechunk win.
# ``parquet`` — decodes a fresh, solely-owned table per read
#              (``_READ_TABLE_OWNED = True``): exercises both the to_polars
#              rechunk win and the to_pandas self-destruct win.
_BACKENDS = ("arrow", "parquet")


def _rss_bytes() -> int:
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * 4096


def _make_tabular(backend: str, rows: int):
    # Multi-chunk source — the realistic shape (warehouse: one chunk per
    # fetch flush; parquet: one per row group; spilled holder: one per
    # part). ``rechunk=False`` only earns its keep with >1 chunk.
    base = flat_table(rows)
    n_chunks = 16
    step = max(1, rows // n_chunks)
    if backend == "arrow":
        batches = [
            b
            for off in range(0, rows, step)
            for b in base.slice(off, min(step, rows - off)).to_batches()
        ]
        return ArrowTabular(batches, spill_bytes=0)
    sink = ParquetFile(b"")
    # Small row groups → many chunks on read-back.
    sink.write_arrow_table(base, row_group_size=step)
    sink.seek(0)
    return ParquetFile(sink.read())


def _worker(op_name: str, rows: int, backend: str) -> None:
    """Run one conversion; print ``{time_s, net_mb}`` JSON.

    Net memory = RSS held with *both* the source tabular and the result
    alive, minus RSS with just the source — the conversion output's
    marginal footprint, robust to a pooling allocator.
    """
    op = _OPS[op_name]
    tab = _make_tabular(backend, rows)
    tab.to_arrow_table()  # warm any cached Arrow table
    gc.collect()

    before = _rss_bytes()
    start = time.perf_counter()
    out = op(tab)
    elapsed = time.perf_counter() - start
    gc.collect()
    after = _rss_bytes()

    # Keep both alive across the measurement so a copy can't reuse the
    # source's pages.
    assert out is not None and tab is not None
    print(json.dumps({"time_s": elapsed, "net_mb": max(0.0, (after - before) / 1e6)}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3_000_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--op", choices=list(_OPS), help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=_BACKENDS, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.op:  # worker mode
        _worker(args.op, args.rows, args.backend or "arrow")
        return 0

    print(
        f"\nTabular conversions (rows={args.rows:,}, repeat={args.repeat}, "
        f"best time / median net RSS)\n"
    )
    for backend in _BACKENDS:
        owned = "owned" if backend == "parquet" else "cached/shared"
        print(f"  [{backend}]  ({owned})")
        header = f"{'op':>16} {'best_s':>9} {'net_MB':>9}"
        print(header)
        print("-" * len(header))
        for name in _OPS:
            times: list[float] = []
            mems: list[float] = []
            for _ in range(args.repeat):
                proc = subprocess.run(
                    [sys.executable, os.path.abspath(__file__),
                     "--op", name, "--rows", str(args.rows), "--backend", backend],
                    capture_output=True, text=True, check=True,
                )
                rec = json.loads(proc.stdout.strip().splitlines()[-1])
                times.append(rec["time_s"])
                mems.append(rec["net_mb"])
            print(f"{name:>16} {min(times):>9.3f} {statistics.median(mems):>9.1f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
