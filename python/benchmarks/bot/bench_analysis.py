"""Benchmark /analysis/aggregate — polars group-by over Arrow vs naive Python.

The pivot endpoint hands the Arrow table to polars zero-copy and aggregates
with its vectorized group-by. This compares that against a dict-accumulation
loop in pure Python over the same rows.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_analysis.py
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.node.api.schemas.analysis import AggMeasure, AggregateRequest
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def main() -> None:
    n = 1_000_000
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        sectors = ["Tech", "Energy", "Finance", "Health", "Industrials"]
        pq.write_table(pa.table({
            "sector": [sectors[i % 5] for i in range(n)],
            "price": [100.0 + (i % 1000) * 0.1 for i in range(n)],
            "volume": [(i % 500) + 1 for i in range(n)],
        }), str(home / "trades.parquet"))
        settings = Settings(node_id="bench", node_home=home, front_home=home, analysis_max_rows=n)
        svc = AnalysisService(settings, fs=FsService(settings))
        print(f"\n  trades.parquet: {n:,} rows, group by sector (5 groups)\n")

        req = AggregateRequest(
            path="trades.parquet", group_by=["sector"],
            measures=[AggMeasure(column="price", agg="mean"), AggMeasure(column="volume", agg="sum")],
        )
        t0 = time.perf_counter()
        for _ in range(5):
            res = asyncio.run(svc.aggregate(req))
        pl_ms = (time.perf_counter() - t0) / 5 * 1000

        # Naive Python: read columns, accumulate per group.
        table = pq.read_table(str(home / "trades.parquet"))
        secs, prices, vols = table["sector"].to_pylist(), table["price"].to_pylist(), table["volume"].to_pylist()
        t0 = time.perf_counter()
        sums: dict = {}
        for s, p, v in zip(secs, prices, vols):
            a = sums.setdefault(s, [0.0, 0, 0])
            a[0] += p
            a[1] += 1
            a[2] += v
        _ = {s: (a[0] / a[1], a[2]) for s, a in sums.items()}
        naive_ms = (time.perf_counter() - t0) * 1000

        print(f"  polars aggregate (read+group):  {pl_ms:8.1f} ms   ({res.group_count} groups)")
        print(f"  naive python loop (no read):    {naive_ms:8.1f} ms")
        print(f"  ==> {naive_ms / pl_ms:5.1f}x  (polars also pays the parquet read)\n")


if __name__ == "__main__":
    main()
