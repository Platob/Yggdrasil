"""Benchmark the new technical indicators endpoint.

Usage::
    PYTHONPATH=src uv run --project . --extra dev --extra node python/benchmarks/bot/bench_indicators.py
"""
from __future__ import annotations
import asyncio
import math
import tempfile
import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.node.api.schemas.analysis import IndicatorsRequest, CompareRequest, CompareSeries
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def main() -> None:
    n = 100_000
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        # Synthetic OHLCV price series
        prices = [100.0 * (1 + 0.001 * math.sin(i * 0.05) + 0.0005 * (i % 7)) for i in range(n)]
        high = [p * 1.005 for p in prices]
        low = [p * 0.995 for p in prices]
        vol = [float(1000 + i % 500) for i in range(n)]
        pq.write_table(pa.table({"close": prices, "high": high, "low": low, "volume": vol}),
                       str(home / "ohlcv.parquet"))

        settings = Settings(node_id="bench", node_home=home, front_home=home)
        svc = AnalysisService(settings, fs=FsService(settings))

        print(f"\n  ohlcv.parquet: {n:,} rows\n")

        # Indicators benchmark
        req = IndicatorsRequest(path="ohlcv.parquet", column="close",
                                atr_high="high", atr_low="low", limit=n)
        t0 = time.perf_counter()
        for _ in range(5):
            res = asyncio.run(svc.indicators(req))
        ind_ms = (time.perf_counter() - t0) / 5 * 1000
        print(f"  indicators (RSI+MACD+BB+ATR) {n:,} rows: {ind_ms:.1f} ms  "
              f"rsi range=[{min(v for v in res.rsi if v):.1f},{max(v for v in res.rsi if v):.1f}]")

        # Compare (2 series, normalize=True)
        pq.write_table(pa.table({"close": [p * 1.01 for p in prices]}),
                       str(home / "alt.parquet"))
        creq = CompareRequest(
            series=[
                CompareSeries(path="ohlcv.parquet", column="close", label="main"),
                CompareSeries(path="alt.parquet", column="close", label="alt"),
            ],
            normalize=True,
        )
        t0 = time.perf_counter()
        for _ in range(5):
            cres = asyncio.run(svc.compare(creq))
        cmp_ms = (time.perf_counter() - t0) / 5 * 1000
        corr = cres.correlation[0][1] if cres.correlation else None
        print(f"  compare 2×{n:,} rows (normalized): {cmp_ms:.1f} ms  corr={corr}")
        print()


if __name__ == "__main__":
    main()
