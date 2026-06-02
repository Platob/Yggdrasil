"""Benchmark the analysis row-count cache.

The AnalysisService caches ``len(scan)`` results per file (keyed by path +
mtime). A repeated call to aggregate/series/ohlc on the same file avoids the
extra ``select(pl.len()).collect()`` scan. This benchmark measures the wall-clock
speedup for the repeated-call case on a 1M-row parquet.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_analysis_cache.py
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.node.api.schemas.analysis import AggMeasure, AggregateRequest, OhlcRequest, SeriesRequest
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def main() -> None:
    n = 1_000_000
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        tbl = pa.table({
            "sector": [["Tech", "Energy", "Finance"][i % 3] for i in range(n)],
            "price":  [100.0 + (i % 500) * 0.2 for i in range(n)],
            "ts":     list(range(n)),
        })
        pq.write_table(tbl, str(home / "data.parquet"))
        mb = (home / "data.parquet").stat().st_size // 1024 // 1024

        settings = Settings(node_id="bench", node_home=home, front_home=home)
        svc = AnalysisService(settings, fs=FsService(settings))

        print(f"\n  data.parquet: {n:,} rows ({mb} MB)\n")

        req_agg = AggregateRequest(path="data.parquet", group_by=["sector"],
                                   measures=[AggMeasure(column="price", agg="mean")])
        req_ser = SeriesRequest(path="data.parquet", column="price", x="ts", points=800)
        req_ohlc = OhlcRequest(path="data.parquet", column="price", buckets=200)

        for label, coro in [
            ("aggregate (cold)", lambda: asyncio.run(svc.aggregate(req_agg))),
            ("aggregate (warm cache)", lambda: asyncio.run(svc.aggregate(req_agg))),
            ("series   (cold)", lambda: asyncio.run(svc.series(req_ser))),
            ("series   (warm cache)", lambda: asyncio.run(svc.series(req_ser))),
            ("ohlc     (cold)", lambda: asyncio.run(svc.ohlc(req_ohlc))),
            ("ohlc     (warm cache)", lambda: asyncio.run(svc.ohlc(req_ohlc))),
        ]:
            t0 = time.perf_counter()
            for _ in range(3):
                coro()
            ms = (time.perf_counter() - t0) / 3 * 1000
            print(f"  {label:<32}  {ms:8.1f} ms")

        print()
        print("  Note: 'warm cache' avoids a second streaming len() scan.")
        print("  The speedup is most visible on large parquet files.\n")


if __name__ == "__main__":
    main()
