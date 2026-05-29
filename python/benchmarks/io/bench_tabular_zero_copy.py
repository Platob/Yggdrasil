"""Tabular read perf + memory across formats, over an in-memory holder.

Reads the same table back through each :class:`Tabular` leaf
(Parquet / Arrow IPC / CSV / NDJSON) from an in-memory :class:`Memory`
holder, reporting read wall time AND peak Python memory
(``tracemalloc``). The point is the *memory* axis: every leaf reads
through :meth:`IO.arrow_input_stream`, which now wraps the buffer's
``read_mv`` memoryview in a pyarrow Buffer instead of snapshotting via
``to_bytes()`` — so a read no longer makes a full-payload intermediate
copy. Compare ``--mode current`` (zero-copy) vs ``--mode legacy``
(forces the old ``to_bytes`` wrap) to see the delta.

Note: tracemalloc traces Python allocations; the decoded Arrow table
lands in C++ and is invisible, so the column reports the *intermediate*
buffering — exactly what the zero-copy wrap removes.

Usage::

    python benchmarks/io/bench_tabular_zero_copy.py
    python benchmarks/io/bench_tabular_zero_copy.py --rows 400000 --repeat 5
    python benchmarks/io/bench_tabular_zero_copy.py --mode legacy   # A/B
"""
from __future__ import annotations

import argparse
import io
import statistics
import time
import tracemalloc

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile
from yggdrasil.io.primitive.csv_file import CSVFile
from yggdrasil.io.primitive.ndjson_file import NDJSONFile
from yggdrasil.io.primitive.parquet_file import ParquetFile
from yggdrasil.path.memory import Memory


def _table(rows: int) -> pa.Table:
    return pa.table({
        "id": pa.array(range(rows), type=pa.int64()),
        "x": pa.array((i * 1.5 for i in range(rows)), type=pa.float64()),
        "y": pa.array((i * 2 for i in range(rows)), type=pa.int64()),
        "label": pa.array([f"row-{i}" for i in range(rows)]),
    })


def _encode(table: pa.Table) -> dict[str, bytes]:
    pbuf = io.BytesIO()
    pq.write_table(table, pbuf)
    abuf = pa.BufferOutputStream()
    with ipc.new_file(abuf, table.schema) as w:
        w.write_table(table)
    cbuf = io.BytesIO()
    pacsv.write_csv(table, cbuf)
    return {
        "parquet": pbuf.getvalue(),
        "arrow-ipc": abuf.getvalue().to_pybytes(),
        "csv": cbuf.getvalue(),
    }


_LEAVES = {
    "parquet": ParquetFile,
    "arrow-ipc": ArrowIPCFile,
    "csv": CSVFile,
}


def _legacy_read(name: str, holder: Memory) -> pa.Table:
    """The old path: snapshot the buffer with ``to_bytes()`` (a full
    copy) and hand the bytes to a fresh ``pa.BufferReader`` — what
    ``arrow_input_stream`` used to do before the zero-copy wrap."""
    reader = pa.BufferReader(holder.to_bytes())
    if name == "parquet":
        return pq.read_table(reader)
    if name == "arrow-ipc":
        return ipc.RecordBatchFileReader(reader).read_all()
    return pacsv.read_csv(reader)


def _bench_read(name, leaf_cls, holder, repeat: int, legacy: bool):
    # Holder is pre-created outside the measurement window, so ``peak``
    # captures only the read's *intermediate* allocations — exactly what
    # the zero-copy wrap removes (the decoded table is C++/untraced).
    times, peak, rows = [], 0, 0
    for _ in range(repeat):
        tracemalloc.start()
        t0 = time.perf_counter()
        if legacy:
            tbl = _legacy_read(name, holder)
        else:
            tbl = leaf_cls(holder=holder, owns_holder=False).read_arrow_table()
        times.append(time.perf_counter() - t0)
        peak = max(peak, tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        rows = tbl.num_rows
    return min(times), statistics.median(times), peak, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--mode", choices=["current", "legacy"], default="current")
    args = ap.parse_args()

    table = _table(args.rows)
    payloads = _encode(table)
    legacy = args.mode == "legacy"

    print(
        f"\nTabular read perf + memory ({args.mode}) — rows={args.rows}, "
        f"repeat={args.repeat}\n"
    )
    hdr = f"{'format':<12} {'MiB on disk':>12} {'best ms':>9} {'median ms':>10} {'peak MiB':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, leaf_cls in _LEAVES.items():
        payload = payloads[name]
        holder = Memory(binary=payload)  # pre-created — outside the window
        best, med, peak, rows = _bench_read(
            name, leaf_cls, holder, args.repeat, legacy,
        )
        assert rows == args.rows
        print(
            f"{name:<12} {len(payload) / 1048576:>12.2f} {best * 1000:>9.2f} "
            f"{med * 1000:>10.2f} {peak / 1048576:>9.2f}"
        )
    print(
        "\nPeak MiB = intermediate Python buffering during the read (the "
        "decoded Arrow table is C++/untraced). The headline is MEMORY: "
        "'current' wraps the buffer zero-copy (peak ~0); 'legacy' is the "
        "old to_bytes() snapshot (peak ~ payload — a full extra copy per "
        "read). Time: 'current' is the real per-leaf read; arrow-ipc is "
        "notably faster (true zero-copy IPC read). The 'legacy' time is "
        "NOT comparable for csv/json — that path is raw pyarrow without "
        "the leaf's cast/options, so only its memory column is a fair A/B."
    )


if __name__ == "__main__":
    main()
