"""Benchmark the analysis engine: lazy projection pushdown + streaming.

Everything runs on polars lazy scans. On a WIDE file an aggregate that touches
2 of 30 columns should read only those 2 (projection pushdown) and stream the
group-by — versus eagerly loading the whole table first. Also times the
adaptive downsample and OHLC resample.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_analysis.py
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.node.api.schemas.analysis import (
    AggMeasure, AggregateRequest, CorrelationRequest, ForecastRequest, IndicatorsRequest,
    OhlcRequest, RiskRequest, SeriesRequest,
)
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def main() -> None:
    n, ncols = 1_000_000, 30
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        cols = {"sector": [["Tech", "Energy", "Finance", "Health", "Ind"][i % 5] for i in range(n)],
                "price": [100.0 + (i % 1000) * 0.1 for i in range(n)]}
        for j in range(ncols - 2):
            cols[f"pad{j}"] = [float(i % 97) for i in range(n)]   # noise columns
        pq.write_table(pa.table(cols), str(home / "wide.parquet"))
        mb = (home / "wide.parquet").stat().st_size // 1024 // 1024
        settings = Settings(node_id="bench", node_home=home, front_home=home)
        svc = AnalysisService(settings, fs=FsService(settings))
        print(f"\n  wide.parquet: {n:,} rows x {ncols} cols ({mb} MB), aggregate touches 2 cols\n")

        req = AggregateRequest(path="wide.parquet", group_by=["sector"],
                               measures=[AggMeasure(column="price", agg="mean")])
        t0 = time.perf_counter()
        for _ in range(5):
            res = asyncio.run(svc.aggregate(req))
        lazy_ms = (time.perf_counter() - t0) / 5 * 1000

        # eager: load the whole table, then aggregate (reads all 30 columns)
        t0 = time.perf_counter()
        for _ in range(5):
            df = pl.from_arrow(pq.read_table(str(home / "wide.parquet")))
            _ = df.group_by("sector").agg(pl.col("price").mean())
        eager_ms = (time.perf_counter() - t0) / 5 * 1000

        print(f"  lazy scan + projection pushdown (2 cols):  {lazy_ms:8.1f} ms   ({res.group_count} groups)")
        print(f"  eager full read (30 cols) + aggregate:     {eager_ms:8.1f} ms")
        print(f"  ==> {eager_ms / lazy_ms:5.1f}x  (pushdown skips 28 unused columns)\n")

        # downsample + ohlc
        t0 = time.perf_counter()
        s = asyncio.run(svc.series(SeriesRequest(path="wide.parquet", column="price", points=800)))
        ds_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        o = asyncio.run(svc.ohlc(OhlcRequest(path="wide.parquet", column="price", buckets=120)))
        ohlc_ms = (time.perf_counter() - t0) * 1000
        print(f"  downsample {n:,} -> {len(s.x)} pts:  {ds_ms:8.1f} ms")
        print(f"  ohlc {n:,} -> {o.bars} bars:      {ohlc_ms:8.1f} ms\n")

    # -- forecasting (xgboost→gbr→ridge over engineered features) -----------
    import math
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        m = 50_000
        ts = list(range(m))
        grp = [["a", "b", "c", "d"][i % 4] for i in range(m)]
        val = [100.0 + 0.01 * i + 12.0 * math.sin(2 * math.pi * i / 24) for i in range(m)]
        pq.write_table(pa.table({"ts": ts, "grp": grp, "value": val}), str(home / "ts.parquet"))
        settings = Settings(node_id="bench", node_home=home, front_home=home)
        svc = AnalysisService(settings, fs=FsService(settings))
        print(f"  ts.parquet: {m:,} rows, 4 groups — forecast value~ts\n")
        for model in ("ridge", "gbr", "xgboost"):
            t0 = time.perf_counter()
            try:
                r = asyncio.run(svc.forecast(ForecastRequest(
                    path="ts.parquet", column="value", x="ts", group="grp",
                    horizon=48, model=model, period=24)))
            except Exception as exc:                       # backend not installed
                print(f"  forecast {model:8s}: skipped ({type(exc).__name__})")
                continue
            ms = (time.perf_counter() - t0) * 1000
            rmse = r.series[0].rmse
            print(f"  forecast {model:8s} ({r.model_used:8s}): {ms:8.1f} ms  "
                  f"{len(r.series)} series x 48h  rmse≈{rmse}")
        print()

    # -- risk metrics ---------------------------------------------------------
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        import random, math as _math
        rng = random.Random(42)
        m = 5_000
        # Simulate GBM price series
        price = [100.0]
        for _ in range(m - 1):
            price.append(price[-1] * (1 + rng.gauss(0.0003, 0.015)))
        pq.write_table(pa.table({"price": price}), str(home / "prices.parquet"))
        settings = Settings(node_id="bench", node_home=home, front_home=home)
        svc = AnalysisService(settings, fs=FsService(settings))
        t0 = time.perf_counter()
        for _ in range(10):
            risk = asyncio.run(svc.risk(RiskRequest(path="prices.parquet", column="price", periods_per_year=252)))
        risk_ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"  risk metrics ({m:,} price rows, 10x avg): {risk_ms:6.1f} ms")
        print(f"    sharpe={risk.sharpe_ratio}  max_dd={risk.max_drawdown:.3f}  var95={risk.var_95:.4f}\n")

    # -- technical indicators -------------------------------------------------
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        rng = random.Random(99)
        m = 2_000
        price2 = [100.0]
        for _ in range(m - 1):
            price2.append(price2[-1] * (1 + rng.gauss(0.0002, 0.012)))
        high2 = [p * (1 + abs(rng.gauss(0, 0.005))) for p in price2]
        low2 = [p * (1 - abs(rng.gauss(0, 0.005))) for p in price2]
        vol2 = [rng.randint(100_000, 5_000_000) for _ in range(m)]
        pq.write_table(pa.table({"close": price2, "high": high2, "low": low2, "volume": vol2}), str(home / "ohlcv.parquet"))
        settings = Settings(node_id="bench", node_home=home, front_home=home)
        svc = AnalysisService(settings, fs=FsService(settings))
        t0 = time.perf_counter()
        for _ in range(10):
            ind = asyncio.run(svc.indicators(IndicatorsRequest(
                path="ohlcv.parquet", column="close", high="high", low="low", volume="volume",
                sma=[20, 50], ema=[12, 26], rsi=14, macd=True, bollinger=20, atr=14, stoch=14, obv=True)))
        ind_ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"  indicators ({m:,} OHLCV rows, 10x avg): {ind_ms:6.1f} ms  ({len(ind.indicators)} series)")
        print(f"    computed: {', '.join(ind.indicators.keys())}\n")

    # -- correlation matrix ---------------------------------------------------
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        rng = random.Random(42)
        m = 3_000
        assets = {f"asset_{i}": [100.0 * (1 + rng.gauss(0.0003, 0.015)) ** j for j in range(m)] for i in range(10)}
        pq.write_table(pa.table(assets), str(home / "assets.parquet"))
        settings = Settings(node_id="bench", node_home=home, front_home=home)
        svc = AnalysisService(settings, fs=FsService(settings))
        cols = list(assets.keys())
        t0 = time.perf_counter()
        for _ in range(10):
            cr = asyncio.run(svc.correlate(CorrelationRequest(path="assets.parquet", columns=cols, method="pearson")))
        corr_ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"  correlation ({m:,} rows x {len(cols)} assets, 10x avg): {corr_ms:6.1f} ms")


if __name__ == "__main__":
    main()
