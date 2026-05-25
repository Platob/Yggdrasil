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
        entry = await service.create(FunctionCreate(
            name=f"bench-func-{i}",
            code=f"print({i})",
            language="python",
        ))
        ids.append(entry.id)

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
    """Benchmark monitor snapshot collection."""
    from yggdrasil.node.services.monitor import MonitorService

    service = MonitorService(Settings(), history_size=n)
    start = time.perf_counter()

    for _ in range(n):
        service.snapshot()
        service._last_collect = 0  # force re-collect

    elapsed = time.perf_counter() - start
    print(f"\nMonitor snapshots (n={n}):")
    print(f"  total: {elapsed:.4f}s ({n / elapsed:.0f} ops/s)")


if __name__ == "__main__":
    asyncio.run(bench_function_crud())
    asyncio.run(bench_monitor_snapshot())
