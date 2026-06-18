"""Benchmark the YGG Bot v2 API hot endpoints.

Spins the in-process ASGI app (no uvicorn round-trip) and hits each
endpoint N times via httpx.AsyncClient.  Reports p50 / p99 in
microseconds and throughput in req/s — the canonical before/after probe
for bot-backend perf work.

Usage::

    PYTHONPATH=src python benchmarks/bot_v2/bench_bot_v2_endpoints.py
    PYTHONPATH=src python benchmarks/bot_v2/bench_bot_v2_endpoints.py --n 1000
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx


ENDPOINTS = [
    ("/api/ping",                  "ping       "),
    ("/api/v2/health",             "health     "),
    ("/api/v2/stats",              "stats      "),
    ("/api/v2/market/prices",      "prices     "),
    ("/api/v2/market/fx",          "fx         "),
    ("/api/v2/signals",            "signals    "),
]


async def _time_endpoint(
    client: httpx.AsyncClient, path: str, n: int
) -> tuple[list[float], int]:
    samples: list[float] = []
    status = 0
    for _ in range(n):
        t0 = time.perf_counter()
        r = await client.get(path)
        samples.append((time.perf_counter() - t0) * 1_000_000)
        status = r.status_code
    return samples, status


async def main(n: int = 500) -> None:
    from yggdrasil.bot import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Warm-up pass
        for path, _ in ENDPOINTS:
            for _ in range(5):
                await client.get(path)

        print(f"\n  {'endpoint':<14} {'n':>5}  {'p50µs':>8}  {'p99µs':>8}  {'avgµs':>8}  {'req/s':>8}  status")
        print(f"  {'-' * 68}")
        for path, label in ENDPOINTS:
            samples, status = await _time_endpoint(client, path, n)
            samples.sort()
            p50 = samples[len(samples) // 2]
            p99 = samples[int(len(samples) * 0.99)]
            avg = statistics.mean(samples)
            rps = 1_000_000 / avg if avg > 0 else 0
            flag = " " if status == 200 else f" [{status}]"
            print(f"  {label}  {n:>5}  {p50:>8.0f}  {p99:>8.0f}  {avg:>8.0f}  {rps:>8.0f}  {status}{flag}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500, help="Requests per endpoint")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    asyncio.run(main(args.n))
