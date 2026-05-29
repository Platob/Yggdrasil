"""Benchmark the new trading + AI status endpoints.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_trading_ai.py
"""
from __future__ import annotations

import asyncio
import statistics
import time

import httpx

from yggdrasil.node.api.app import create_api


ENDPOINTS = [
    ("/api/v2/trading/signals",   "signals/list  "),
    ("/api/v2/trading/portfolio", "portfolio     "),
    ("/api/v2/ai/status",         "ai/status     "),
    ("/api/v2/stats",             "stats(gather) "),
    ("/api/v2/health",            "health(gather)"),
]


async def time_endpoint(client: httpx.AsyncClient, path: str, n: int) -> tuple[list[float], int]:
    samples: list[float] = []
    status_seen = 0
    for _ in range(n):
        t0 = time.perf_counter()
        r = await client.get(path)
        elapsed = (time.perf_counter() - t0) * 1_000_000
        samples.append(elapsed)
        status_seen = r.status_code
    return samples, status_seen


async def main(iterations: int = 500) -> None:
    app = create_api()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for path, _ in ENDPOINTS:
            for _ in range(5):
                await client.get(path)

        print(f"\n  endpoint        n     p50us    p99us    avgus    req/s    status")
        print(f"  {'-' * 72}")
        for path, label in ENDPOINTS:
            samples, status = await time_endpoint(client, path, iterations)
            samples.sort()
            p50 = samples[len(samples) // 2]
            p99 = samples[int(len(samples) * 0.99)]
            avg = statistics.mean(samples)
            rps = 1_000_000 / avg if avg > 0 else 0
            print(f"  {label}  {iterations:>4d}  {p50:>7.0f}  {p99:>7.0f}  {avg:>7.0f}  {rps:>7.0f}    {status}")


if __name__ == "__main__":
    asyncio.run(main())
