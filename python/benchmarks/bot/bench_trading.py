"""Benchmark the trading analysis endpoints: technical indicators + correlation.

Technical indicators (RSI, MACD, Bollinger Bands, ATR) run purely in Python
over a collected polars frame; correlation uses polars pearson_corr per pair.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_trading.py
    PYTHONPATH=src python benchmarks/bot/bench_trading.py --repeat 7
"""
from __future__ import annotations

import argparse
import asyncio
import math
import tempfile
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.node.api.schemas.analysis import (
    CorrelationRequest,
    TechnicalRequest,
)
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def main(repeat: int = 5) -> None:
    n = 10_000
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)

        # -- OHLCV-like synthetic data with 10 factor columns ----------------
        t = list(range(n))
        price = [100.0 + 10.0 * math.sin(2 * math.pi * i / 252) + 0.05 * i for i in range(n)]
        high = [p * 1.005 for p in price]
        low = [p * 0.995 for p in price]
        volume = [float(1_000_000 + (i % 100) * 10_000) for i in range(n)]
        cols: dict = {
            "ts": t, "close": price, "high": high, "low": low, "volume": volume,
            **{f"factor_{j}": [float((i + j) % 50) / 50 for i in range(n)] for j in range(10)},
        }
        pq.write_table(pa.table(cols), str(home / "ohlcv.parquet"))
        settings = Settings(node_id="bench", node_home=home, front_home=home)
        svc = AnalysisService(settings, fs=FsService(settings))

        print(f"\n  ohlcv.parquet: {n:,} rows (close, high, low, volume + 10 factors)\n")

        # -- Technical indicators (RSI + MACD + Bollinger + ATR) -------------
        req_tech = TechnicalRequest(
            path="ohlcv.parquet", column="close", x="ts",
            high="high", low="low", rsi_period=14, macd_fast=12,
            macd_slow=26, bb_period=20,
        )
        t0 = time.perf_counter()
        for _ in range(repeat):
            res = asyncio.run(svc.technical(req_tech))
        ms = (time.perf_counter() - t0) / repeat * 1000
        sigs = len(res.signals)
        print(f"  RSI+MACD+BB+ATR ({n:,} rows):              {ms:8.1f} ms   {sigs} signals")

        # -- Technical indicators (no ATR, no x) -----------------------------
        req_noatr = TechnicalRequest(path="ohlcv.parquet", column="close")
        t0 = time.perf_counter()
        for _ in range(repeat):
            res2 = asyncio.run(svc.technical(req_noatr))
        ms2 = (time.perf_counter() - t0) / repeat * 1000
        print(f"  RSI+MACD+BB only ({n:,} rows):             {ms2:8.1f} ms   {len(res2.signals)} signals")

        # -- Pearson correlation matrix (all 14 numeric columns) -------------
        req_corr = CorrelationRequest(path="ohlcv.parquet")
        t0 = time.perf_counter()
        for _ in range(repeat):
            cr = asyncio.run(svc.correlation(req_corr))
        ms_corr = (time.perf_counter() - t0) / repeat * 1000
        nc = len(cr.columns)
        print(f"  Pearson correlation ({nc}×{nc}, {n:,} rows):    {ms_corr:8.1f} ms")

        # -- Spearman correlation (4 columns) --------------------------------
        req_sp = CorrelationRequest(
            path="ohlcv.parquet", method="spearman",
            columns=["close", "high", "low", "volume"],
        )
        t0 = time.perf_counter()
        for _ in range(repeat):
            cr_sp = asyncio.run(svc.correlation(req_sp))
        ms_sp = (time.perf_counter() - t0) / repeat * 1000
        print(f"  Spearman correlation (4×4, {n:,} rows):      {ms_sp:8.1f} ms")

        # -- Large series (100k rows, capped at limit=2000) ------------------
        n2 = 100_000
        price2 = [100.0 + 0.001 * i + 5.0 * math.sin(2 * math.pi * i / 252) for i in range(n2)]
        pq.write_table(
            pa.table({"ts": list(range(n2)), "close": price2}),
            str(home / "large.parquet"),
        )
        req_large = TechnicalRequest(path="large.parquet", column="close", x="ts", limit=2000)
        t0 = time.perf_counter()
        res_large = asyncio.run(svc.technical(req_large))
        ms_large = (time.perf_counter() - t0) * 1000
        print(
            f"  RSI+MACD+BB ({n2:,} rows, capped@2000):    {ms_large:8.1f} ms   "
            f"{len(res_large.signals)} signals  truncated={res_large.truncated}"
        )

        print()
        print("  Signal breakdown (10k-row series):")
        kinds = Counter(s.kind for s in res.signals)
        for kind, count in sorted(kinds.items()):
            print(f"    {kind:30s}  {count}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    main(repeat=args.repeat)
