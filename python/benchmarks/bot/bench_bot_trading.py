"""Benchmark the trading service.

Covers price generation throughput, order processing, signal computation,
and the full HTTP stack.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_bot_trading.py
    PYTHONPATH=src python benchmarks/bot/bench_bot_trading.py --repeat 5
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Callable


INNER = 200


def _time_fn(fn: Callable[[], Any], *, repeat: int, inner: int) -> list[float]:
    for _ in range(min(inner, 10)):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return samples


def _fmt(label: str, samples: list[float]) -> str:
    best = min(samples) * 1e6
    med = statistics.median(samples) * 1e6
    ops = 1.0 / min(samples) if min(samples) > 0 else 0
    return f"{label:<45}  {best:>10.1f}  {med:>10.1f}  {ops:>12,.0f}"


def _bench_service(repeat: int) -> None:
    from yggdrasil.node.config import Settings
    from yggdrasil.node.services.trading import TradingService
    from yggdrasil.node.schemas.trading import OrderCreate, PriceAlertCreate

    settings = Settings(allow_remote=True)
    svc = TradingService(settings)

    print("\n--- trading service direct ---")
    print(f"{'scenario':<45}  {'best us':>10}  {'median us':>10}  {'ops/sec':>12}")
    print("-" * 82)

    samples = _time_fn(lambda: svc.get_price("AAPL"), repeat=repeat, inner=INNER)
    print(_fmt("get_price (AAPL)", samples))

    samples = _time_fn(svc.get_all_prices, repeat=repeat, inner=INNER)
    print(_fmt("get_all_prices (9 default symbols)", samples))

    samples = _time_fn(lambda: svc.get_signal("NVDA"), repeat=repeat, inner=INNER)
    print(_fmt("get_signal (NVDA, MA+RSI on 100 bars)", samples))

    samples = _time_fn(svc.get_all_signals, repeat=repeat, inner=INNER // 2)
    print(_fmt("get_all_signals (9 symbols)", samples))

    samples = _time_fn(svc.get_portfolio, repeat=repeat, inner=INNER)
    print(_fmt("get_portfolio (empty)", samples))

    # Place an order then benchmark portfolio with a position.
    svc.place_order(OrderCreate(symbol="AAPL", side="buy", qty=10))
    samples = _time_fn(svc.get_portfolio, repeat=repeat, inner=INNER)
    print(_fmt("get_portfolio (1 position)", samples))

    # Order throughput is interesting — but each order mutates state, so
    # we measure throughput end-to-end with a fresh service each loop.
    print()
    print("--- order throughput ---")
    for batch in (50, 200):
        s2 = TradingService(settings)
        s2._cash = 1_000_000_000.0  # type: ignore[attr-defined]
        orders = [OrderCreate(symbol="AAPL", side="buy", qty=1) for _ in range(batch)]
        t0 = time.perf_counter()
        for o in orders:
            s2.place_order(o)
        elapsed = time.perf_counter() - t0
        print(f"  {batch:>4} market orders in {elapsed*1e3:>7.2f} ms  ({batch/elapsed:>10,.0f} orders/s)")


def _bench_endpoint(repeat: int) -> None:
    from fastapi.testclient import TestClient
    from yggdrasil.node.app import create_app
    from yggdrasil.node.config import Settings

    settings = Settings(allow_remote=True)
    app = create_app(settings)
    client = TestClient(app)

    print("\n--- trading endpoint (full HTTP stack) ---")
    print(f"{'scenario':<45}  {'best ms':>10}  {'median ms':>10}")
    print("-" * 68)

    samples = _time_fn(lambda: client.get("/api/trading/prices"), repeat=repeat, inner=INNER)
    print(f"{'GET /trading/prices':<45}  {min(samples)*1e3:>10.2f}  {statistics.median(samples)*1e3:>10.2f}")

    samples = _time_fn(lambda: client.get("/api/trading/portfolio"), repeat=repeat, inner=INNER)
    print(f"{'GET /trading/portfolio':<45}  {min(samples)*1e3:>10.2f}  {statistics.median(samples)*1e3:>10.2f}")

    samples = _time_fn(lambda: client.get("/api/trading/signals"), repeat=repeat, inner=INNER // 2)
    print(f"{'GET /trading/signals':<45}  {min(samples)*1e3:>10.2f}  {statistics.median(samples)*1e3:>10.2f}")


def run(repeat: int) -> None:
    print()
    print("=" * 82)
    print(f"  yggdrasil.node trading benchmark  (repeat={repeat})")
    print("=" * 82)
    _bench_service(repeat)
    _bench_endpoint(repeat)
    print()
    print("=" * 82)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3, help="Outer timing loops")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    run(repeat=args.repeat)
