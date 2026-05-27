"""Benchmarks for MarketService (cache layer) and AIService (code analysis)."""
from __future__ import annotations

import asyncio
import time

from yggdrasil.node.config import Settings

SAMPLE_CODE = '''
import os
import json

def process(data: list) -> dict:
    result = {}
    for item in data:
        if item > 0:
            result[item] = item * 2
        elif item < 0:
            result[item] = abs(item)
        else:
            result[item] = 0
    return result

def main():
    import sys
    data = json.loads(os.environ.get("__ygg_inputs__", "{}")).get("data", [1, -2, 3])
    return process(data)
'''


async def bench_ai_analyze(n: int = 200) -> None:
    from yggdrasil.node.services.ai import AIService
    from yggdrasil.node.schemas.ai import CodeAnalysisRequest

    service = AIService(Settings())
    req = CodeAnalysisRequest(code=SAMPLE_CODE, language="python")

    start = time.perf_counter()
    for _ in range(n):
        service.analyze_code(req)
    elapsed = time.perf_counter() - start
    print(f"AI code analysis (n={n}):")
    print(f"  total: {elapsed:.4f}s ({n / elapsed:.0f} ops/s)")


async def bench_market_watchlist(n: int = 10000) -> None:
    from yggdrasil.node.services.market import MarketService

    service = MarketService(Settings())

    start = time.perf_counter()
    for _ in range(n):
        service.get_watchlist()
    elapsed = time.perf_counter() - start
    print(f"\nMarket watchlist get (n={n}):")
    print(f"  total: {elapsed:.4f}s ({n / elapsed:.0f} ops/s)")

    # Add/remove cycles
    cycles = n // 100
    start = time.perf_counter()
    for i in range(cycles):
        service.add_to_watchlist("USD/CHF")
        service.remove_from_watchlist("USD/CHF")
    elapsed = time.perf_counter() - start
    print(f"Market watchlist add/remove (n={cycles} cycles):")
    print(f"  total: {elapsed:.4f}s ({cycles / elapsed:.0f} ops/s)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- Run {i + 1}/{args.repeat} ---")
        asyncio.run(bench_ai_analyze())
        asyncio.run(bench_market_watchlist())
