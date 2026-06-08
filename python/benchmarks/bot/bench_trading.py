"""Benchmark the trading signal computation path.

Tests Polars-based signal computation (SMA, RSI, MACD, Bollinger Bands)
on synthetic OHLCV data, and the market-scan endpoint via ASGI transport.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_trading.py
    PYTHONPATH=src python benchmarks/bot/bench_trading.py --repeat 5 --inner 50
"""
from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import time
from pathlib import Path

REPEAT = 5
INNER = 50

_HDR = f"{'scenario':<48}  {'best µs':>10}  {'median µs':>10}"
_SEP = "-" * len(_HDR)


def _make_ohlcv(n: int) -> dict:
    """Synthetic sine-wave OHLCV for *n* trading days."""
    rows = []
    base = 100.0
    ts = 1_700_000_000_000  # arbitrary epoch ms
    for i in range(n):
        noise = math.sin(i * 0.3) * 5 + math.sin(i * 0.07) * 10
        close = base + noise + i * 0.05
        rows.append({
            "ts": ts + i * 86_400_000,
            "open": round(close - 1.0 + noise * 0.1, 4),
            "high": round(close + 1.5, 4),
            "low": round(close - 1.5, 4),
            "close": round(close, 4),
            "volume": 1_000_000 + i * 1000,
        })
    return {"data": rows, "symbol": "SYNTH", "interval": "1d"}


def _bench_signal_compute(repeat: int, inner: int) -> None:
    from yggdrasil.node.api.trading import _compute_signals

    print()
    print(_HDR)
    print(_SEP)

    for n_days in (60, 130, 260, 520):
        ohlcv = _make_ohlcv(n_days)
        # warm-up
        for _ in range(5):
            _compute_signals(ohlcv)

        samples: list[float] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            for _ in range(inner):
                _compute_signals(ohlcv)
            samples.append((time.perf_counter() - t0) / inner)

        best = min(samples) * 1e6
        med = statistics.median(samples) * 1e6
        print(f"{'signal compute — ' + str(n_days) + ' days':<48}  {best:>10.1f}  {med:>10.1f}")


async def _bench_endpoints(repeat: int, inner: int) -> None:
    import httpx
    from yggdrasil.node import create_app, Settings

    app = create_app(Settings())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        print()
        print(f"{'endpoint scenario':<48}  {'best µs':>10}  {'median µs':>10}")
        print(_SEP)

        for label, method, path, body in [
            ("/api/v2/trading/scan (default watchlist)", "GET", "/api/v2/trading/scan", None),
        ]:
            # We mock market data so the scan doesn't make real HTTP calls.
            # This just measures ASGI dispatch + Polars overhead on cached data.
            print(f"  (skipping live-network endpoints in benchmark mode)")
            break

        # Batch-quotes endpoint with pre-primed cache
        from yggdrasil.node.api.market import _CHART_CACHE
        from unittest.mock import MagicMock
        # Prime cache with synthetic data so no real HTTP calls happen.
        _fake_chart = {
            "meta": {"regularMarketPrice": 150.0, "chartPreviousClose": 148.0,
                     "currency": "USD", "exchangeName": "NASDAQ", "symbol": "AAPL",
                     "regularMarketVolume": 50_000_000, "regularMarketTime": 1_700_000_000},
            "timestamp": [1_700_000_000 + i * 86_400 for i in range(130)],
            "indicators": {"quote": [{
                "open":   [150.0 + math.sin(i*0.3) for i in range(130)],
                "high":   [152.0 + math.sin(i*0.3) for i in range(130)],
                "low":    [148.0 + math.sin(i*0.3) for i in range(130)],
                "close":  [150.0 + math.sin(i*0.3) + i*0.05 for i in range(130)],
                "volume": [50_000_000 for _ in range(130)],
            }]},
        }
        for sym in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "BRK-B"]:
            _CHART_CACHE.set(f"{sym}|1d|1d", _fake_chart)
            _CHART_CACHE.set(f"{sym}|1d|6mo", _fake_chart)

        for label, path in [
            ("/api/v2/market/batch (10 symbols, cached)", "/api/v2/market/batch?symbols=AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,JPM,V,BRK-B"),
            ("/api/v2/trading/scan (10 symbols, cached)", "/api/v2/trading/scan?symbols=AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,JPM,V,BRK-B"),
        ]:
            # warm-up
            for _ in range(3):
                await client.get(path)

            samples: list[float] = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                for _ in range(max(inner // 5, 3)):
                    await client.get(path)
                samples.append((time.perf_counter() - t0) / max(inner // 5, 3))

            best = min(samples) * 1e6
            med = statistics.median(samples) * 1e6
            print(f"{label:<48}  {best:>10.1f}  {med:>10.1f}")


def run(repeat: int = REPEAT, inner: int = INNER) -> None:
    print()
    print("=" * 82)
    print(f"  yggdrasil.node trading benchmark  (repeat={repeat}, inner={inner})")
    print("=" * 82)
    print()
    print("--- Polars signal computation (in-process, no HTTP) ---")
    _bench_signal_compute(repeat, inner)
    print()
    print("--- API endpoint overhead (ASGI transport, cached market data) ---")
    asyncio.run(_bench_endpoints(repeat, inner))
    print()
    print("=" * 82)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    parser.add_argument("--inner", type=int, default=INNER)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    run(args.repeat, args.inner)
