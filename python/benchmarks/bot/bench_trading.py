"""Benchmark the trading analysis service: indicators, signals, backtest.

Measures the hot paths in-process (no HTTP): indicator computation via polars
window expressions, vectorized signal generation, and the per-bar backtest loop.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_trading.py
    PYTHONPATH=src python benchmarks/bot/bench_trading.py --rows 500000
"""
from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _gen_ohlcv(path: Path, rows: int) -> None:
    import math, random
    price = 100.0
    opens, highs, lows, closes, vols = [], [], [], [], []
    for i in range(rows):
        change = random.gauss(0, 1.5)
        open_ = price
        close = max(1.0, price + change)
        high = max(open_, close) + abs(random.gauss(0, 0.5))
        low = min(open_, close) - abs(random.gauss(0, 0.5))
        vol = random.randint(10_000, 500_000)
        opens.append(round(open_, 4))
        highs.append(round(high, 4))
        lows.append(round(low, 4))
        closes.append(round(close, 4))
        vols.append(vol)
        price = close
    pq.write_table(pa.table({
        "ts": list(range(rows)),
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols,
    }), str(path))


def _timeit(label: str, fn, n: int = 5) -> float:
    # Warmup
    for _ in range(2):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    ms = (time.perf_counter() - t0) / n * 1000.0
    print(f"  {label:<55} {ms:>8.1f} ms")
    return ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000)
    args = ap.parse_args()
    run = asyncio.run

    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        src = home / "ohlcv.parquet"
        print(f"\ngenerating {args.rows:,}-row OHLCV parquet…")
        _gen_ohlcv(src, args.rows)
        mb = src.stat().st_size / 1024 / 1024

        from yggdrasil.node.config import Settings
        from yggdrasil.node.api.services.fs import FsService
        from yggdrasil.node.api.services.trading import TradingService

        settings = Settings(node_id="bench", node_home=home, front_home=home)
        fs = FsService(settings)
        svc = TradingService(settings, fs)

        print(f"  source: {mb:.1f} MB parquet, {args.rows:,} rows × 6 cols\n")
        print(f"  {'scenario':<55} {'ms/op':>8}")
        print(f"  {'-'*65}")

        # Indicators (full TA suite: EMA, SMA, RSI, MACD, Bollinger, ATR, VWAP)
        ind = [None]
        def _ind():
            ind[0] = run(svc.indicators("ohlcv.parquet", "close", "ts"))

        _timeit("indicators (EMA/SMA/RSI/MACD/BB/ATR/VWAP)", _ind)

        # Signals (EMA cross, RSI, MACD crossover)
        _timeit("signals (ema_cross + rsi + macd)", lambda: run(svc.signals("ohlcv.parquet", "close")))

        # Backtest scenarios
        for strategy in ("ema_cross", "rsi_mean_reversion", "macd", "buy_and_hold"):
            def _bt(s=strategy):
                return run(svc.backtest("ohlcv.parquet", "close", strategy=s))
            res = _bt()
            ms = None
            # Quick timing
            t0 = time.perf_counter()
            for _ in range(3):
                _bt()
            ms = (time.perf_counter() - t0) / 3 * 1000
            print(f"  {'backtest:' + strategy:<55} {ms:>8.1f} ms  "
                  f"[return={res['total_return']:+.3f} trades={res['n_trades']}]")

        # Correlation matrix (2 assets)
        _timeit("correlation (2 assets)", lambda: run(svc.correlation(["ohlcv.parquet", "ohlcv.parquet"])))

        print(f"\n  indicators result: "
              f"{len(ind[0]['price'])} rows, "
              f"ema_9 non-null={sum(1 for v in ind[0]['ema_9'] if v is not None)}, "
              f"rsi non-null={sum(1 for v in ind[0]['rsi_14'] if v is not None)}")

        # HTTP endpoint timing
        print(f"\n  --- HTTP endpoint overhead (in-process FastAPI) ---")
        print(f"  {'endpoint':<55} {'p50 ms':>8}")
        print(f"  {'-'*65}")
        import asyncio as _asyncio, statistics, json

        import httpx
        from yggdrasil.node.api.app import create_api

        api_app = create_api(settings)
        transport = httpx.ASGITransport(app=api_app)

        async def _bench_http(n: int = 50) -> None:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                # Warm up
                for _ in range(3):
                    await client.post("/api/v2/trading/indicators",
                                      json={"path": "ohlcv.parquet", "column": "close"})

                for label, method, path, payload in [
                    ("GET /api/v2/trading/strategies", "get",
                     "/api/v2/trading/strategies", None),
                    ("POST /api/v2/trading/indicators", "post",
                     "/api/v2/trading/indicators",
                     {"path": "ohlcv.parquet", "column": "close"}),
                    ("POST /api/v2/trading/signals", "post",
                     "/api/v2/trading/signals",
                     {"path": "ohlcv.parquet", "column": "close"}),
                    ("POST /api/v2/trading/backtest", "post",
                     "/api/v2/trading/backtest",
                     {"path": "ohlcv.parquet", "column": "close", "strategy": "ema_cross"}),
                ]:
                    samples = []
                    for _ in range(n):
                        t0 = time.perf_counter()
                        if method == "get":
                            await client.get(path)
                        else:
                            await client.post(path, json=payload)
                        samples.append((time.perf_counter() - t0) * 1000)
                    samples.sort()
                    p50 = samples[len(samples) // 2]
                    print(f"  {label:<55} {p50:>8.1f}")

        _asyncio.run(_bench_http())
        print()


if __name__ == "__main__":
    main()
