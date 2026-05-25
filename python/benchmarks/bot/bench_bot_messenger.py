"""Benchmark yggdrasil.node messenger service.

Covers in-memory message throughput, channel operations, and the
HTTP endpoint overhead for the chat system.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_bot_messenger.py
    PYTHONPATH=src python benchmarks/bot/bench_bot_messenger.py --repeat 5
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Callable


INNER = 100


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


def _bench_service_direct(repeat: int) -> None:
    """Benchmark the MessengerService directly (no HTTP overhead)."""
    import asyncio
    from yggdrasil.node.config import Settings
    from yggdrasil.node.services.messenger import MessengerService
    from yggdrasil.node.schemas.messenger import MessageSend

    settings = Settings(allow_remote=True)
    service = MessengerService(settings)

    loop = asyncio.new_event_loop()

    print("\n--- messenger service direct (no HTTP) ---")
    print(f"{'scenario':<45}  {'best us':>10}  {'median us':>10}  {'ops/sec':>12}")
    print("-" * 82)

    msg = MessageSend(text="hello world", sender="bench-user", channel="general")

    samples = _time_fn(
        lambda: loop.run_until_complete(service.send_message(msg)),
        repeat=repeat,
        inner=INNER * 5,
    )
    print(_fmt("send_message (general)", samples))

    loop.run_until_complete(service.create_channel("bench-chan"))
    msg2 = MessageSend(text="bench msg", sender="bench-user", channel="bench-chan")
    samples = _time_fn(
        lambda: loop.run_until_complete(service.send_message(msg2)),
        repeat=repeat,
        inner=INNER * 5,
    )
    print(_fmt("send_message (custom channel)", samples))

    samples = _time_fn(
        lambda: loop.run_until_complete(service.list_channels()),
        repeat=repeat,
        inner=INNER * 5,
    )
    print(_fmt("list_channels", samples))

    samples = _time_fn(
        lambda: loop.run_until_complete(service.get_messages("general", limit=50)),
        repeat=repeat,
        inner=INNER * 5,
    )
    print(_fmt("get_messages (50 limit)", samples))

    samples = _time_fn(
        lambda: loop.run_until_complete(service.get_channel("general")),
        repeat=repeat,
        inner=INNER * 5,
    )
    print(_fmt("get_channel", samples))

    loop.close()


def _bench_endpoint(repeat: int) -> None:
    """Benchmark messenger via TestClient (full HTTP stack)."""
    from fastapi.testclient import TestClient
    from yggdrasil.node.app import create_app
    from yggdrasil.node.config import Settings

    settings = Settings(allow_remote=True)
    app = create_app(settings)
    client = TestClient(app)

    print("\n--- messenger endpoint (full HTTP stack) ---")
    print(f"{'scenario':<45}  {'best ms':>10}  {'median ms':>10}")
    print("-" * 68)

    def _send():
        return client.post(
            "/api/messenger",
            json={"text": "bench message", "sender": "bench-user"},
        )

    samples = _time_fn(_send, repeat=repeat, inner=INNER)
    best_ms = min(samples) * 1e3
    med_ms = statistics.median(samples) * 1e3
    print(f"{'POST /messenger (send)':<45}  {best_ms:>10.2f}  {med_ms:>10.2f}")

    def _list_channels():
        return client.get("/api/messenger/channels")

    samples = _time_fn(_list_channels, repeat=repeat, inner=INNER)
    best_ms = min(samples) * 1e3
    med_ms = statistics.median(samples) * 1e3
    print(f"{'GET /messenger/channels':<45}  {best_ms:>10.2f}  {med_ms:>10.2f}")

    def _get_messages():
        return client.get("/api/messenger/channels/general/messages?limit=50")

    samples = _time_fn(_get_messages, repeat=repeat, inner=INNER)
    best_ms = min(samples) * 1e3
    med_ms = statistics.median(samples) * 1e3
    print(f"{'GET /messenger/.../messages (50)':<45}  {best_ms:>10.2f}  {med_ms:>10.2f}")


def _bench_throughput(repeat: int) -> None:
    """Measure raw message throughput."""
    import asyncio
    from yggdrasil.node.config import Settings
    from yggdrasil.node.services.messenger import MessengerService
    from yggdrasil.node.schemas.messenger import MessageSend

    settings = Settings(allow_remote=True)
    service = MessengerService(settings)
    loop = asyncio.new_event_loop()

    print("\n--- throughput (burst send) ---")

    for burst_size in (100, 1000, 5000):
        msgs = [
            MessageSend(text=f"msg-{i}", sender=f"user-{i % 10}", channel="general")
            for i in range(burst_size)
        ]

        async def _burst():
            for m in msgs:
                await service.send_message(m)

        t0 = time.perf_counter()
        loop.run_until_complete(_burst())
        elapsed = time.perf_counter() - t0
        rate = burst_size / elapsed
        print(f"  {burst_size:>5} messages in {elapsed*1e3:>8.1f} ms  ({rate:>10,.0f} msg/s)")

    loop.close()


def run(repeat: int) -> None:
    print()
    print("=" * 82)
    print(f"  yggdrasil.node messenger benchmark  (repeat={repeat})")
    print("=" * 82)

    _bench_service_direct(repeat)
    _bench_endpoint(repeat)
    _bench_throughput(repeat)

    print()
    print("=" * 82)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5, help="Outer timing loops")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    run(repeat=args.repeat)
