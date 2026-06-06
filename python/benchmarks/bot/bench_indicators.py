"""Benchmark the technical-indicator + trading-signal analytics.

Both run on a single streaming polars pass over the price series (RSI, MACD,
Bollinger bands, SMA/EMA, ATR proxy computed in one ``with_columns``). Signals
reuse that same computation, then scan the bounded tail for crossings and
threshold violations — so this also measures that the shared ``_indicators_df``
isn't re-read per endpoint.

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

from yggdrasil.node.api.schemas.analysis import IndicatorRequest, SignalRequest
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def main() -> None:
    n = 100_000
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        # A noisy trending walk so RSI/MACD/BB actually fire signals.
        ts = list(range(n))
        price, p = [], 100.0
        for i in range(n):
            p += 0.0008 * i / n + 6.0 * math.sin(2 * math.pi * i / 250) * 0.01 + ((i * 2654435761) % 1000 - 500) * 0.002
            price.append(round(max(1.0, p), 4))
        vol = [float(1000 + (i % 500)) for i in range(n)]
        pq.write_table(pa.table({"ts": ts, "close": price, "volume": vol}), str(home / "prices.parquet"))
        mb = (home / "prices.parquet").stat().st_size / 1024 / 1024
        settings = Settings(node_id="bench", node_home=home, front_home=home)
        svc = AnalysisService(settings, fs=FsService(settings))
        print(f"\n  prices.parquet: {n:,} rows ({mb:.1f} MB) — close series, ts ordered\n")

        ind_req = IndicatorRequest(
            path="prices.parquet", column="close", x="ts",
            indicators=["rsi", "macd", "bb", "sma", "ema", "atr"],
        )
        # warm + time
        res = asyncio.run(svc.indicators(ind_req))
        t0 = time.perf_counter()
        for _ in range(5):
            res = asyncio.run(svc.indicators(ind_req))
        ind_ms = (time.perf_counter() - t0) / 5 * 1000
        print(f"  indicators RSI+MACD+BB+SMA+EMA+ATR:  {ind_ms:8.1f} ms   "
              f"({len(res.x):,} pts, truncated={res.truncated})")

        sig_req = SignalRequest(path="prices.parquet", column="close", x="ts", last_n=2000)
        sres = asyncio.run(svc.signals(sig_req))
        t0 = time.perf_counter()
        for _ in range(5):
            sres = asyncio.run(svc.signals(sig_req))
        sig_ms = (time.perf_counter() - t0) / 5 * 1000
        rsi_txt = f"{sres.current_rsi:.1f}" if sres.current_rsi is not None else "n/a"
        print(f"  signals (indicators + tail scan):    {sig_ms:8.1f} ms   "
              f"({len(sres.signals)} signals, bias={sres.bias}, rsi={rsi_txt})")
        print()


if __name__ == "__main__":
    main()
