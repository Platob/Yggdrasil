"""Benchmarks for function CRUD and execution throughput."""
from __future__ import annotations

import asyncio
import time

from yggdrasil.node.config import Settings


async def bench_function_crud(n: int = 100) -> None:
    """Benchmark function create/read/delete cycle."""
    from yggdrasil.node.services.function import FunctionService
    from yggdrasil.node.schemas.function import FunctionCreate

    service = FunctionService(Settings())
    start = time.perf_counter()

    ids = []
    for i in range(n):
        response = await service.create(FunctionCreate(
            name=f"bench-func-{i}",
            code=f"print({i})",
            language="python",
        ))
        ids.append(response.function.id)

    create_time = time.perf_counter() - start

    start = time.perf_counter()
    for fid in ids:
        await service.get(fid)
    read_time = time.perf_counter() - start

    start = time.perf_counter()
    for fid in ids:
        await service.delete(fid)
    delete_time = time.perf_counter() - start

    print(f"Function CRUD (n={n}):")
    print(f"  create: {create_time:.4f}s ({n / create_time:.0f} ops/s)")
    print(f"  read:   {read_time:.4f}s ({n / read_time:.0f} ops/s)")
    print(f"  delete: {delete_time:.4f}s ({n / delete_time:.0f} ops/s)")


async def bench_monitor_snapshot(n: int = 1000) -> None:
    """Benchmark monitor snapshot collection.

    Two paths:
    - cached: debounce hit (< 0.5s since last collect) — just returns in-memory snapshot
    - uncached: full psutil collection including process tracking
    """
    from yggdrasil.node.services.monitor import MonitorService

    service = MonitorService(Settings(), history_size=n)

    # Seed first snapshot
    service.snapshot()

    # Cached path: do NOT reset _last_collect
    start = time.perf_counter()
    for _ in range(n):
        service.snapshot()
    cached_elapsed = time.perf_counter() - start

    # Uncached path: force re-collect each time
    uncached_n = min(n, 20)
    start = time.perf_counter()
    for _ in range(uncached_n):
        service.snapshot()
        service._last_collect = 0
    uncached_elapsed = time.perf_counter() - start

    print(f"\nMonitor snapshots:")
    print(f"  cached   (n={n}): {cached_elapsed:.4f}s ({n / cached_elapsed:.0f} ops/s)")
    print(f"  uncached (n={uncached_n}): {uncached_elapsed:.4f}s ({uncached_n / uncached_elapsed:.0f} ops/s) [psutil full scan]")


if __name__ == "__main__":
    asyncio.run(bench_function_crud())
    asyncio.run(bench_monitor_snapshot())
