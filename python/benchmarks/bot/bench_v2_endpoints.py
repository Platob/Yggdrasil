"""Benchmark the v2 API hot endpoints of the trading node.

Spins up the in-process FastAPI app (no uvicorn round-trip cost) and
hits each endpoint N times with httpx.AsyncClient. Reports p50/p99 in
microseconds plus throughput in req/s. Designed as the canonical
before/after probe for backend perf work.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_v2_endpoints.py
"""
from __future__ import annotations

import asyncio
import statistics
import time

import httpx

from yggdrasil.node.app import create_api


ENDPOINTS = [
    ("/api/ping",                              "ping       "),
    ("/api/v2/health",                         "health     "),
    ("/api/v2/stats",                          "stats      "),
    ("/api/v2/market/assets",                  "assets     "),
    ("/api/v2/market/candles?symbol=BTC/USD&interval=1h&limit=200", "candles    "),
    ("/api/v2/market/tick?symbol=BTC/USD",     "tick       "),
    ("/api/v2/market/book?symbol=BTC/USD",     "book       "),
]


async def time_endpoint(client: httpx.AsyncClient, path: str, n: int) -> tuple[list[float], int]:
    samples: list[float] = []
    status_seen = 0
    for _ in range(n):
        t0 = time.perf_counter()
        r = await client.get(path)
        elapsed = (time.perf_counter() - t0) * 1_000_000  # microseconds
        samples.append(elapsed)
        status_seen = r.status_code
    return samples, status_seen


async def main(iterations: int = 500) -> None:
    app = create_api()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Warm-up: hit each endpoint a few times to fill caches.
        for path, _ in ENDPOINTS:
            for _ in range(5):
                await client.get(path)

        print(f"\n  endpoint     n     p50us    p99us    avgus    req/s    status")
        print(f"  {'-' * 70}")
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
