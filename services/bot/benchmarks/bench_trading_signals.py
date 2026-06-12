"""Benchmark signal computation speed.

Tests: RSI, MACD, Bollinger Bands, full signal generation
across different data sizes.

Usage::

    PYTHONPATH=../../.. python bench_trading_signals.py
    PYTHONPATH=../../.. python bench_trading_signals.py --repeat 5
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

# Adjust path so api is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from services.bot.api.core.signals import (
    compute_rsi,
    compute_macd,
    compute_bollinger,
    generate_signals,
)
from services.bot.api.models.market import OHLCV
from datetime import datetime, timedelta


def _make_bars(n: int) -> list[OHLCV]:
    random.seed(42)
    price = 150.0
    bars = []
    base = datetime(2024, 1, 1)
    for i in range(n):
        change = random.gauss(0, 2.0)
        price = max(1.0, price + change)
        bars.append(OHLCV(
            symbol="TEST",
            timestamp=base + timedelta(days=i),
            open=price - abs(random.gauss(0, 0.5)),
            high=price + abs(random.gauss(0, 1.0)),
            low=price - abs(random.gauss(0, 1.0)),
            close=price,
            volume=random.randint(1_000_000, 50_000_000),
        ))
    return bars


def _bench(name: str, fn, repeat: int) -> None:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    best = min(times) * 1e6
    med  = statistics.median(times) * 1e6
    print(f"{name:<50} best={best:>8.2f} us  median={med:>8.2f} us")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    bars_sm = _make_bars(60)
    bars_md = _make_bars(252)
    bars_lg = _make_bars(1000)
    closes_sm = [b.close for b in bars_sm]
    closes_md = [b.close for b in bars_md]
    closes_lg = [b.close for b in bars_lg]

    print(f"{'Benchmark':<50} {'Best':>16}  {'Median':>16}")
    print("-" * 88)

    _bench("RSI(14) — 60 bars",        lambda: compute_rsi(closes_sm), args.repeat)
    _bench("RSI(14) — 252 bars",       lambda: compute_rsi(closes_md), args.repeat)
    _bench("RSI(14) — 1000 bars",      lambda: compute_rsi(closes_lg), args.repeat)
    _bench("MACD(12,26,9) — 252 bars", lambda: compute_macd(closes_md), args.repeat)
    _bench("MACD(12,26,9) — 1000 bars",lambda: compute_macd(closes_lg), args.repeat)
    _bench("Bollinger(20) — 252 bars", lambda: compute_bollinger(closes_md), args.repeat)
    _bench("generate_signals — 60 bars", lambda: generate_signals("TEST", bars_sm), args.repeat)
    _bench("generate_signals — 252 bars",lambda: generate_signals("TEST", bars_md), args.repeat)
    _bench("generate_signals — 1000 bars",lambda: generate_signals("TEST", bars_lg), args.repeat)


if __name__ == "__main__":
    main()
