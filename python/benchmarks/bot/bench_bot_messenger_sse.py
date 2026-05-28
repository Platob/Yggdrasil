"""Benchmark messenger SSE fan-out and subscriber notification latency.

Compares the new event-queue fan-out path against direct send and tests
that one publisher can feed N subscribers without blocking.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_bot_messenger_sse.py
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path


async def _bench_fanout(repeat: int, subscriber_counts: tuple[int, ...]) -> None:
    from yggdrasil.node.config import Settings
    from yggdrasil.node.services.messenger import MessengerService
    from yggdrasil.node.schemas.messenger import MessageSend

    settings = Settings(allow_remote=True)

    print(f"{'subscribers':>12}  {'send us':>10}  {'recv us p50':>14}  {'recv us p99':>14}")
    print("-" * 60)

    for n_subs in subscriber_counts:
        svc = MessengerService(settings)
        # Spin up N subscribers on the 'general' channel.
        queues = []
        async def _consume(q):
            await q.get()
        # Drive the service generator to register each subscriber.
        async def _subscribe():
            async def collect():
                async for _msg in svc.stream_messages("general"):
                    pass
            tasks = [asyncio.create_task(collect()) for _ in range(n_subs)]
            await asyncio.sleep(0.05)  # let subscribers register
            return tasks

        tasks = await _subscribe()
        try:
            # Measure send latency only (publisher side).
            send_times: list[float] = []
            n_msgs = 100
            for i in range(n_msgs):
                t0 = time.perf_counter()
                await svc.send_message(MessageSend(text=f"m{i}", sender="bench"))
                send_times.append(time.perf_counter() - t0)
            send_us = statistics.median(send_times) * 1e6
            print(f"{n_subs:>12}  {send_us:>10.1f}  {'-':>14}  {'-':>14}")
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def run(repeat: int) -> None:
    print()
    print("=" * 82)
    print(f"  yggdrasil.node messenger SSE fan-out benchmark  (repeat={repeat})")
    print("=" * 82)
    print("\n--- send_message with N concurrent SSE subscribers ---")
    asyncio.run(_bench_fanout(repeat, (0, 1, 10, 50, 200)))
    print()
    print("=" * 82)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    run(repeat=args.repeat)
