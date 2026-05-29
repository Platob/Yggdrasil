"""Benchmark /api/v2/tabular/inspect — hit on every file open in the UI.

Inspect needs the schema, an exact-ish row count, and an editable flag. For
parquet that all lives in the footer (O(1)); the old path pulled cap+1 rows
just to size the file. This shows the footer path against a row-read of the
same large parquet.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_tabular_inspect.py
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.data.options import CastOptions
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.api.services.tabular import TabularService
from yggdrasil.node.config import Settings
from yggdrasil.path import Path as YggPath


def main() -> None:
    rows, cols = 500_000, 20
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        path = home / "big.parquet"
        pq.write_table(pa.table({f"c{j}": pa.array(range(rows)) for j in range(cols)}), str(path))
        mb = path.stat().st_size // 1024 // 1024
        print(f"\n  parquet {mb} MB, {rows} rows x {cols} cols\n")

        settings = Settings(node_id="bench", node_home=home, front_home=home)
        svc = TabularService(settings, fs=FsService(settings))
        cap = settings.tabular_preview_max_rows

        # old behaviour: read cap+1 rows to size the file
        t0 = time.perf_counter()
        for _ in range(10):
            with YggPath.from_(str(path)).open("rb") as bio:
                t = bio.read_arrow_table(options=CastOptions(row_limit=cap + 1))
                _ = t.num_rows, t.schema
        old_ms = (time.perf_counter() - t0) / 10 * 1000

        # new inspect: footer schema + metadata row count
        t0 = time.perf_counter()
        for _ in range(10):
            info = asyncio.run(svc.inspect("big.parquet"))
        new_ms = (time.perf_counter() - t0) / 10 * 1000

        print(f"  read {cap}+1 rows to size:   {old_ms:8.2f} ms")
        print(f"  footer-metadata inspect:    {new_ms:8.2f} ms   {old_ms / new_ms:5.1f}x faster")
        print(f"  (now reports exact row_count={info.row_count}, editable={info.editable})\n")


if __name__ == "__main__":
    main()
