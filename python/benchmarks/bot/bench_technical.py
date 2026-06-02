"""Benchmark the technical analysis engine.

Generates a synthetic OHLCV parquet with 500k bars and times each indicator
type individually, then a "full stack" run (RSI + MACD + BB + ATR + VWAP + Stoch).
"""
from __future__ import annotations
import asyncio, tempfile, time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.node.api.schemas.technical import TechnicalRequest, IndicatorSpec
from yggdrasil.node.api.services.technical import TechnicalService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def main() -> None:
    n = 500_000
    rng = np.random.default_rng(42)
    price = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    high = price * (1 + rng.uniform(0, 0.02, n))
    low = price * (1 - rng.uniform(0, 0.02, n))
    volume = rng.integers(100_000, 10_000_000, n).astype(float)

    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        tbl = pa.table({"close": price, "high": high, "low": low, "volume": volume})
        pq.write_table(tbl, str(home / "ohlcv.parquet"))
        mb = (home / "ohlcv.parquet").stat().st_size / 1024 / 1024

        settings = Settings(node_id="bench", node_home=home, front_home=home)
        svc = TechnicalService(settings, fs=FsService(settings))
        print(f"\n  ohlcv.parquet: {n:,} bars ({mb:.1f} MB)\n")

        indicators_to_bench = [
            ("RSI(14)", [IndicatorSpec(type="rsi", period=14)]),
            ("SMA(20)", [IndicatorSpec(type="sma", period=20)]),
            ("EMA(20)", [IndicatorSpec(type="ema", period=20)]),
            ("MACD(12,26,9)", [IndicatorSpec(type="macd", fast=12, slow=26, signal=9)]),
            ("BB(20,2)", [IndicatorSpec(type="bb", period=20, std_dev=2.0)]),
            ("ATR(14)", [IndicatorSpec(type="atr", period=14)]),
            ("VWAP", [IndicatorSpec(type="vwap")]),
            ("OBV", [IndicatorSpec(type="obv")]),
            ("Stoch(14,3)", [IndicatorSpec(type="stoch", period=14, d_period=3)]),
            ("Full stack (all)", [
                IndicatorSpec(type="rsi", period=14),
                IndicatorSpec(type="macd", fast=12, slow=26, signal=9),
                IndicatorSpec(type="bb", period=20, std_dev=2.0),
                IndicatorSpec(type="atr", period=14),
                IndicatorSpec(type="vwap"),
                IndicatorSpec(type="stoch", period=14, d_period=3),
            ]),
        ]

        for label, inds in indicators_to_bench:
            req = TechnicalRequest(
                path="ohlcv.parquet", close="close",
                high="high", low="low", volume="volume",
                indicators=inds,
            )
            # warmup
            asyncio.run(svc.compute(req))
            t0 = time.perf_counter()
            for _ in range(3):
                asyncio.run(svc.compute(req))
            ms = (time.perf_counter() - t0) / 3 * 1000
            print(f"  {label:<30} {ms:8.1f} ms")

        print()


if __name__ == "__main__":
    main()
