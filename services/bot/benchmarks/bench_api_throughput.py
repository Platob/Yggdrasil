"""Benchmark FastAPI endpoint throughput using TestClient (in-process, no network).

Measures:
- Health endpoint (baseline overhead)
- /market/quote (with cache warm/cold)
- /signals/{symbol} (signal computation)
- Portfolio PnL
- OHLCV JSON vs Arrow response

Usage::

    PYTHONPATH=../../.. python bench_api_throughput.py
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient
from services.bot.api.main import app
from services.bot.api.core import market as market_core
from services.bot.api.models.market import OHLCV, Quote


def _make_bars(n: int = 252) -> list[OHLCV]:
    random.seed(99)
    price, base = 150.0, datetime(2024, 1, 1)
    bars = []
    for i in range(n):
        price = max(1.0, price + random.gauss(0, 2))
        bars.append(OHLCV(symbol="AAPL", timestamp=base + timedelta(days=i),
                          open=price, high=price+1, low=price-1, close=price,
                          volume=random.randint(10_000_000, 80_000_000)))
    return bars


def _seed_cache() -> None:
    """Inject mock data directly into the in-memory caches to avoid network."""
    bars = _make_bars()
    q = Quote(symbol="AAPL", price=187.50, change=1.2, change_pct=0.64, volume=55_000_000)
    # seed OHLCV cache
    market_core._ohlcv_cache[("AAPL", "3mo", "1d")] = (bars, time.monotonic())
    market_core._ohlcv_cache[("AAPL", "1mo", "1d")] = (bars[-30:], time.monotonic())
    market_core._quote_cache["AAPL"] = (q, time.monotonic())


def _bench(name: str, fn, repeat: int) -> None:
    times = []
    for _ in range(repeat * 10):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    best = min(times) * 1e3
    med  = statistics.median(times) * 1e3
    rps  = 1 / statistics.mean(times)
    print(f"{name:<55} best={best:>6.2f} ms  median={med:>6.2f} ms  ~{rps:,.0f} rps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    _seed_cache()
    client = TestClient(app, raise_server_exceptions=True)

    print(f"{'Endpoint':<55} {'Best':>14}  {'Median':>14}  {'~RPS':>12}")
    print("-" * 105)

    _bench("GET /health",               lambda: client.get("/health"),            args.repeat)
    _bench("GET /market/quote/AAPL (warm cache)",
                                         lambda: client.get("/market/quote/AAPL"), args.repeat)
    _bench("GET /signals/AAPL (warm cache)",
                                         lambda: client.get("/signals/AAPL"),      args.repeat)
    _bench("GET /market/ohlcv/AAPL json",
                                         lambda: client.get("/market/ohlcv/AAPL"), args.repeat)
    _bench("GET /market/ohlcv/AAPL arrow",
                                         lambda: client.get("/market/ohlcv/AAPL?fmt=arrow"), args.repeat)
    _bench("GET /portfolio/1/pnl",       lambda: client.get("/portfolio/1/pnl"),   args.repeat)
    _bench("GET /signals/batch/scan",    lambda: client.get("/signals/batch/scan?symbols=AAPL&symbols=MSFT"), args.repeat)


if __name__ == "__main__":
    main()
