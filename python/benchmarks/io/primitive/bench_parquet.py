"""Benchmark :class:`ParquetIO` — production read/write scenarios.

Production shapes covered:

* **flat 1k / 50k** — typical wire/batch sizes for analytics tables.
* **nested 10k** — list / map / struct columns alongside scalars.
* **wide 32-col 10k** — the schema shape projection pushdown unlocks.
* **collect_schema** — schema-only read against the file footer.
* **batched 50k @ 8k batch_size** — measures ``iter_batches`` /
  ``cast_arrow_batch_iterator`` per-batch overhead the streaming
  callers (folder readers, statement results) pay.

Each shape runs write + read across **arrow / polars / pandas** so the
bridge cost shows up alongside the codec.

Usage::

    PYTHONPATH=src python benchmarks/io/primitive/bench_parquet.py
    PYTHONPATH=src python benchmarks/io/primitive/bench_parquet.py --repeat 5
"""
from __future__ import annotations

import os
import sys

# Sibling-module import that works both when this file is run
# directly (``python bench_parquet.py``) and from ``run_all.py``
# (which spawns it via ``python <abs path>``). In both cases the
# script's directory needs to be on ``sys.path``.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from yggdrasil.data.options import CastOptions
from yggdrasil.io.primitive.parquet_io import ParquetIO

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
    out.extend(bench_roundtrip("parquet flat 1k", ParquetIO, flat_table(1_000),
                               repeat=repeat, inner=200))
    out.extend(bench_roundtrip("parquet flat 50k", ParquetIO, flat_table(50_000),
                               repeat=repeat, inner=50))
    out.extend(bench_roundtrip("parquet nested 10k", ParquetIO, nested_table(10_000),
                               repeat=repeat, inner=50))
    out.extend(bench_roundtrip("parquet wide 32x10k", ParquetIO, wide_table(10_000),
                               repeat=repeat, inner=50))

    # Schema-only — file-footer read.
    sink = ParquetIO(b"")
    sink.write_arrow_table(flat_table(50_000))
    sink.seek(0)
    payload = sink.read()
    out.append(time_one(
        "parquet: collect_schema flat 50k",
        lambda: ParquetIO(payload).collect_schema(),
        repeat=repeat, inner=2_000,
    ))

    # Batched stream — small batches simulate the streaming readers.
    out.append(time_one(
        "parquet: read_arrow_batches flat 50k @ batch=8k",
        lambda: list(ParquetIO(payload).read_arrow_batches(
            CastOptions(row_size=8_000)
        )),
        repeat=repeat, inner=50,
    ))

    return out


main = make_cli(scenarios)


if __name__ == "__main__":
    main()
