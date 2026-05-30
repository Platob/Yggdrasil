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
    AggMeasure, AggregateRequest, ForecastRequest, OhlcRequest, SeriesRequest,
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


if __name__ == "__main__":
    main()
