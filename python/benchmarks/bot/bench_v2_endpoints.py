"""Benchmark the v2 API hot endpoints: /ping, /health, /stats, /backend.

Spins up the in-process FastAPI app (no uvicorn round-trip cost) and hits
each endpoint N times with httpx's ASGI transport. Reports p50/p99 in
microseconds plus throughput in req/s. Designed as the canonical before/after
probe for backend perf work.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_v2_endpoints.py
    PYTHONPATH=src python benchmarks/bot/bench_v2_endpoints.py --repeat 7 --inner 200
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx


REPEAT = 5
INNER = 100

ENDPOINTS = [
    ("/api/ping",       "ping        "),
    ("/api/v2/health",  "health      "),
    ("/api/v2/stats",   "stats       "),
    ("/api/v2/backend", "backend     "),
]

_HDR = f"{'endpoint':<40}  {'p50 µs':>10}  {'p99 µs':>10}  {'req/s':>10}"
_SEP = "-" * len(_HDR)


def _fmt(label: str, samples: list[float]) -> str:
    p50 = statistics.median(samples) * 1e6
    p99 = statistics.quantiles(samples, n=100)[98] * 1e6
    rps = 1.0 / statistics.median(samples) if samples else 0
    return f"{label:<40}  {p50:>10.1f}  {p99:>10.1f}  {rps:>10.0f}"


async def _bench(repeat: int, inner: int) -> None:
    from yggdrasil.node import create_app, Settings

    app = create_app(Settings())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # warm-up
        for path, _ in ENDPOINTS:
            for _ in range(min(inner, 20)):
                await client.get(path)

        print()
        print("=" * 80)
        print(f"  yggdrasil.node v2 endpoint benchmark  (repeat={repeat}, inner={inner})")
        print("=" * 80)
        print()
        print(_HDR)
        print(_SEP)

        for path, label in ENDPOINTS:
            all_samples: list[float] = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                for _ in range(inner):
                    await client.get(path)
                elapsed = time.perf_counter() - t0
                all_samples.extend([elapsed / inner] * inner)
            print(_fmt(label.strip() or path, all_samples))

        # Parallel burst: 10 concurrent /api/ping
        print()
        print("--- concurrent burst (10 parallel /api/ping) ---")
        burst_samples: list[float] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            for _ in range(inner // 10):
                await asyncio.gather(*[client.get("/api/ping") for _ in range(10)])
            elapsed = time.perf_counter() - t0
            burst_samples.extend([elapsed / inner] * inner)
        print(_fmt("10× parallel /api/ping", burst_samples))

        print()
        print("=" * 80)
        print()


def run(repeat: int = REPEAT, inner: int = INNER) -> None:
    asyncio.run(_bench(repeat, inner))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    parser.add_argument("--inner", type=int, default=INNER)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    run(args.repeat, args.inner)
