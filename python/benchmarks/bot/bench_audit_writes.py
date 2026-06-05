"""Benchmark audit log write throughput.

The audit log used to open audit.jsonl per-entry which dominated the cost
of any mutation burst. This bench measures append throughput.

Usage::

    PYTHONPATH=src python benchmarks/bot/bench_audit_writes.py
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from yggdrasil.node.api.services.audit import AuditLog


class _Settings:
    def __init__(self, root: Path) -> None:
        self.logs_root = root


def main(n: int = 50_000) -> None:
    with tempfile.TemporaryDirectory() as td:
        s = _Settings(Path(td))
        audit = AuditLog(s, max_entries=10_000)
        t0 = time.perf_counter()
        for i in range(n):
            audit.log("create", "pyfunc", i, detail=f"name=fn_{i}")
        elapsed = time.perf_counter() - t0
        rate = n / elapsed
        print(f"\n  audit append:  {n:>6d} entries  {elapsed*1000:>7.1f} ms")
        print(f"                 {rate:>8.0f} entries/sec  ({elapsed/n*1e6:.1f} us each)")


if __name__ == "__main__":
    main()
