"""Benchmark :class:`ArrowIPCFile` — IPC stream / file format.

Arrow-IPC is the cheapest format (same in-memory layout as the
underlying Arrow runtime), so it establishes the "what does the
yggdrasil wiring cost?" floor. Production shapes match the other
primitive benches so the formats are directly comparable.

Usage::

    PYTHONPATH=src python benchmarks/io/primitive/bench_arrow_ipc.py
    PYTHONPATH=src python benchmarks/io/primitive/bench_arrow_ipc.py --repeat 5
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from yggdrasil.data.options import CastOptions
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile

from _common import (  # type: ignore[import-not-found]
    bench_roundtrip,
    flat_table,
    make_cli,
    nested_table,
    time_one,
    wide_table,
)


def scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    out.extend(bench_roundtrip("arrow_ipc flat 1k", ArrowIPCFile, flat_table(1_000),
                               repeat=repeat, inner=500))
    out.extend(bench_roundtrip("arrow_ipc flat 50k", ArrowIPCFile, flat_table(50_000),
                               repeat=repeat, inner=100))
    out.extend(bench_roundtrip("arrow_ipc nested 10k", ArrowIPCFile, nested_table(10_000),
                               repeat=repeat, inner=100))
    out.extend(bench_roundtrip("arrow_ipc wide 32x10k", ArrowIPCFile, wide_table(10_000),
                               repeat=repeat, inner=100))

    sink = ArrowIPCFile(b"")
    sink.write_arrow_table(flat_table(50_000))
    sink.seek(0)
    payload = sink.read()
    out.append(time_one(
        "arrow_ipc: collect_schema flat 50k",
        lambda: ArrowIPCFile(payload).collect_schema(),
        repeat=repeat, inner=2_000,
    ))
    out.append(time_one(
        "arrow_ipc: read_arrow_batches flat 50k @ batch=8k",
        lambda: list(ArrowIPCFile(payload).read_arrow_batches(
            CastOptions(row_size=8_000)
        )),
        repeat=repeat, inner=100,
    ))
    return out


main = make_cli(scenarios)


if __name__ == "__main__":
    main()
