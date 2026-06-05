"""Benchmark paged /fs/ls — a page now builds only its own FsEntry models.

The lazy tree + paged listing fetch a window at a time. ``ls`` scandirs the
directory once (unavoidable without an index) but builds pydantic FsEntry
models for ONLY the requested page, so a 50k-entry directory pages cheaply
instead of constructing 50k models + a giant JSON payload per request.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_fs_paging.py
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def main() -> None:
    n = 50_000
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        for i in range(n):
            (home / f"f{i:06d}.bin").write_bytes(b"x")
        svc = FsService(Settings(node_id="bench", node_home=home, front_home=home))
        print(f"\n  directory: {n} files\n")

        t0 = time.perf_counter()
        for _ in range(10):
            full = asyncio.run(svc.ls(""))
        full_ms = (time.perf_counter() - t0) / 10 * 1000

        t0 = time.perf_counter()
        for _ in range(10):
            page = asyncio.run(svc.ls("", offset=0, limit=100))
        page_ms = (time.perf_counter() - t0) / 10 * 1000

        print(f"  ls() full      ({len(full.entries)} entries built):  {full_ms:8.2f} ms")
        print(f"  ls(limit=100)  ({len(page.entries)} entries built):     {page_ms:8.2f} ms   {full_ms / page_ms:5.1f}x faster")
        print(f"  (both scandir all {page.total} entries; the page skips 49,900 model builds)\n")


if __name__ == "__main__":
    main()
