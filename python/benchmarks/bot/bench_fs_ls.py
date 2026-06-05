"""Benchmark /fs/ls directory listing — the hot path behind the lazy tree.

The old listing used ``iterdir()`` + a per-child ``_entry`` that stat'd each
child 3-4× (sort key is_dir, then stat, is_dir, is_dir again). The new path
uses ``os.scandir`` which caches the dirent type and stats once. This bench
times the live service against directories of growing width and prints both
the new path and a re-creation of the old iterdir+stat path for contrast.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_fs_ls.py
"""
from __future__ import annotations

import asyncio
import datetime as dt
import tempfile
import time
from pathlib import Path

from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings


def _old_listing(root: Path) -> int:
    # Mirrors the pre-scandir cost: stat-heavy iterdir + per-entry stats.
    n = 0
    for child in sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        st = child.stat()
        _ = child.is_dir()
        _ = (st.st_size if not child.is_dir() else 0)
        _ = dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).isoformat()
        n += 1
    return n


def main() -> None:
    widths = [200, 1000, 5000]
    print(f"\n  /fs/ls listing  ({'  '.join(str(w) for w in widths)} entries)\n")
    print(f"  {'entries':>8}  {'old iterdir ms':>15}  {'scandir ls ms':>14}  {'speedup':>8}")
    for w in widths:
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            for i in range(w):
                (home / f"f{i:05d}.bin").write_bytes(b"x")
            for i in range(w // 10):
                (home / f"dir{i:04d}").mkdir()
            svc = FsService(Settings(node_id="bench", node_home=home, front_home=home))

            t0 = time.perf_counter()
            for _ in range(5):
                _old_listing(home)
            old_ms = (time.perf_counter() - t0) / 5 * 1000

            t0 = time.perf_counter()
            for _ in range(5):
                res = asyncio.run(svc.ls(""))
            new_ms = (time.perf_counter() - t0) / 5 * 1000

            total = w + w // 10
            print(f"  {total:>8}  {old_ms:>15.2f}  {new_ms:>14.2f}  {old_ms / new_ms:>7.2f}x  ({len(res.entries)} entries)")


if __name__ == "__main__":
    main()
