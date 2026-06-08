"""Benchmark the bounded /fs/read preview path.

Before the read cap, ``FsService.read`` did ``Path.read_text()`` on the whole
file — so previewing a 1 GB log pulled 1 GB into memory and latency scaled
with file size. The bounded read pulls at most ``max_read_bytes`` no matter
how big the file is. This bench shows latency and bytes-returned stay flat as
the file grows, which is the whole point of the guard.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_fs_read.py
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings

CAP = 4 * 1024 * 1024  # 4 MB, the default max_read_bytes


def main() -> None:
    sizes_mb = [1, 16, 64, 256]
    with tempfile.TemporaryDirectory() as td:
        home = Path(td)
        svc = FsService(Settings(node_id="bench", node_home=home, front_home=home, max_read_bytes=CAP))
        print(f"\n  bounded /fs/read  (cap = {CAP // (1024*1024)} MB)\n")
        print(f"  {'file size':>10}  {'read ms':>9}  {'bytes back':>12}  {'truncated':>9}")
        for mb in sizes_mb:
            path = home / f"f{mb}.txt"
            with open(path, "wb") as fh:
                fh.write(b"a" * (mb * 1024 * 1024))
            t0 = time.perf_counter()
            res = asyncio.run(svc.read(path.name))
            elapsed = (time.perf_counter() - t0) * 1000
            back = len(res.content)
            print(f"  {mb:>8d}MB  {elapsed:>9.2f}  {back:>12d}  {str(res.truncated):>9}")
            os.unlink(path)
        print("\n  latency and bytes-returned stay flat as the file grows — the")
        print("  node never loads the whole file just to preview it.\n")


if __name__ == "__main__":
    main()
