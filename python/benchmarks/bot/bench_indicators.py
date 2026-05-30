"""Benchmark technical indicators: RSI, MACD, EMA, Bollinger Bands.

Each indicator is a chain of polars expressions (ewm/rolling) over the ordered
price series, sorted by the time column and capped at 5000 points. This times
the per-indicator cost and a combined RSI+MACD+BB request on a synthetic price
series.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_indicators.py
"""
from __future__ import annotations

import asyncio
import math
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.node.api.schemas.analysis import IndicatorRequest, IndicatorSpec
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def main() -> None:
    n = 100_000
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        price = [100 + 10 * math.sin(i / 50) + (i % 17) * 0.1 for i in range(n)]
        ts = [f"2024-01-{(i // 1000) % 28 + 1:02d}T{(i % 1000) // 60:02d}:{i % 60:02d}:00" for i in range(n)]
        pq.write_table(pa.table({"price": price, "ts": ts}), str(home / "prices.parquet"))
        settings = Settings(node_id="bench", node_home=home, front_home=home)
        svc = AnalysisService(settings, fs=FsService(settings))
        print(f"\n  prices.parquet: {n:,} rows\n")

        for inds in [
            [IndicatorSpec(type="rsi")],
            [IndicatorSpec(type="ema", params={"period": 20})],
            [IndicatorSpec(type="macd")],
            [IndicatorSpec(type="bb")],
            [IndicatorSpec(type="rsi"), IndicatorSpec(type="macd"), IndicatorSpec(type="bb")],
        ]:
            req = IndicatorRequest(path="prices.parquet", column="price", x="ts", indicators=inds)
            t0 = time.perf_counter()
            res = asyncio.run(svc.indicators(req))
            ms = (time.perf_counter() - t0) * 1000
            names = [i.type for i in inds]
            print(f"  {'+'.join(names):35s}  {ms:7.1f} ms  ({res.source_rows:,} rows)")
    print()


if __name__ == "__main__":
    main()
