"""Benchmark the in-memory :class:`Memory` holder.

Targets the per-write / per-read / per-view hot path that backs
``BytesIO`` (and therefore every primitive-IO leaf when there's no
local file). Construction, write-grow, read, ``view`` slicing, and
``memoryview`` access all show up under request/response bodies and
intermediate buffer chains.

Spill-to-disk is out of scope here — covered separately by the
local-path bench.

Usage::

    PYTHONPATH=src python benchmarks/io/bench_io_memory.py
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

from yggdrasil.io.memory import Memory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SMALL = b"hello world\n" * 8                           # 96 B
MEDIUM = b"x" * 64_000                                  # 64 KB
LARGE = b"x" * 1_000_000                                # ~1 MB


def _seed(payload: bytes) -> Memory:
    """Build a :class:`Memory` already containing *payload*.

    Routes through :meth:`Memory.write_bytes` (rather than the
    internal ``_write_mv``) so the visible :attr:`size` stays in
    sync with the buffer contents — :meth:`read_bytes` and
    :meth:`pread` enforce ``size`` bounds and would otherwise
    refuse a read on a holder that only had bytes spliced in via
    the private primitive.
    """
    m = Memory()
    m.write_bytes(payload)
    return m


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 200)):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
    }


def _fmt(r: dict) -> str:
    scale = 1e6
    unit = "us"
    if r["best"] < 1e-6:
        scale, unit = 1e9, "ns"
    elif r["best"] >= 1e-3:
        scale, unit = 1e3, "ms"
    return (
        f"{r['label']:<58s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    out.append(_time_one(
        "Memory() empty construct",
        lambda: Memory(),
        repeat=repeat, inner=200_000,
    ))

    # Write — capacity grow + write, the per-buffer-build hot path.
    def write_small():
        m = Memory()
        m.reserve(len(SMALL))
        m._write_mv(memoryview(SMALL), 0)
    def write_medium():
        m = Memory()
        m.reserve(len(MEDIUM))
        m._write_mv(memoryview(MEDIUM), 0)
    def write_large():
        m = Memory()
        m.reserve(len(LARGE))
        m._write_mv(memoryview(LARGE), 0)

    out.append(_time_one(
        "Memory _write_mv 96 B",
        write_small, repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "Memory _write_mv 64 KB",
        write_medium, repeat=repeat, inner=5_000,
    ))
    out.append(_time_one(
        "Memory _write_mv 1 MB",
        write_large, repeat=repeat, inner=500,
    ))

    # Read / view — already-seeded holder.
    m_small = _seed(SMALL)
    m_medium = _seed(MEDIUM)
    m_large = _seed(LARGE)

    out.append(_time_one(
        "Memory _read_mv 96 B",
        lambda: m_small._read_mv(len(SMALL), 0),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "Memory _read_mv 64 KB (zero-copy slice)",
        lambda: m_medium._read_mv(len(MEDIUM), 0),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "Memory _read_mv 1 MB (zero-copy slice)",
        lambda: m_large._read_mv(len(LARGE), 0),
        repeat=repeat, inner=200_000,
    ))

    out.append(_time_one(
        "Memory.memoryview() 96 B",
        lambda: m_small.memoryview(),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "Memory.memoryview() 1 MB",
        lambda: m_large.memoryview(),
        repeat=repeat, inner=200_000,
    ))

    out.append(_time_one(
        "Memory.to_bytes() 96 B",
        lambda: m_small.to_bytes(),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "Memory.to_bytes() 64 KB (copy)",
        lambda: m_medium.to_bytes(),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "Memory.to_bytes() 1 MB (copy)",
        lambda: m_large.to_bytes(),
        repeat=repeat, inner=2_000,
    ))

    # Reserve + truncate — the in-place capacity knobs that buffer
    # pre-allocations and rewrites pay for.
    out.append(_time_one(
        "Memory.reserve(64 KB) on empty",
        lambda: Memory().reserve(64_000),
        repeat=repeat, inner=20_000,
    ))

    def truncate_small():
        m = Memory()
        m.reserve(len(MEDIUM))
        m._write_mv(memoryview(MEDIUM), 0)
        m.truncate(1_000)
    out.append(_time_one(
        "Memory.truncate(1000) after 64 KB write",
        truncate_small, repeat=repeat, inner=5_000,
    ))

    # Stat — read by every Tabular hook for size / mtime / media-type.
    out.append(_time_one(
        "Memory._stat (live IOStats)",
        lambda: m_medium._stat(),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "Memory.size",
        lambda: m_medium.size,
        repeat=repeat, inner=500_000,
    ))

    # pread / pwrite / read_bytes / write_bytes — the cursorless surface
    # every IO leaf and HTTP body chain reaches before opening a cursor.
    out.append(_time_one(
        "Memory.pread 96 B",
        lambda: m_small.pread(len(SMALL), 0),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "Memory.pread 1 MB",
        lambda: m_large.pread(len(LARGE), 0),
        repeat=repeat, inner=2_000,
    ))

    def pwrite_small():
        m = Memory()
        m.pwrite(SMALL, 0)
    def pwrite_medium():
        m = Memory()
        m.pwrite(MEDIUM, 0)
    out.append(_time_one(
        "Memory.pwrite 96 B (fresh)",
        pwrite_small, repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "Memory.pwrite 64 KB (fresh)",
        pwrite_medium, repeat=repeat, inner=2_000,
    ))

    out.append(_time_one(
        "Memory.read_bytes() 96 B",
        lambda: m_small.read_bytes(),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "Memory.read_bytes() 64 KB",
        lambda: m_medium.read_bytes(),
        repeat=repeat, inner=20_000,
    ))

    # Text round-trip — utf-8 encode/decode on the buffer surface.
    text = "hello world\n" * 64  # 768 B utf-8
    m_text = _seed(text.encode("utf-8"))
    out.append(_time_one(
        "Memory.read_text() 768 B",
        lambda: m_text.read_text(),
        repeat=repeat, inner=50_000,
    ))

    def write_text_cycle():
        m = Memory()
        m.write_text(text)
    out.append(_time_one(
        "Memory.write_text() 768 B (fresh)",
        write_text_cycle, repeat=repeat, inner=10_000,
    ))

    # xxh3_int64 — cached against (size, mtime); first call walks, repeats
    # hit the memoized digest. Both shapes matter: the cold digest cost
    # bounds hash-aware paths (cache keys, content equality); the warm
    # cost bounds repeat calls (digest property re-reads).
    def cold_digest():
        m = _seed(MEDIUM)
        m.xxh3_int64()
    out.append(_time_one(
        "Memory.xxh3_int64 64 KB (cold)",
        cold_digest, repeat=repeat, inner=5_000,
    ))
    # Pre-warm the cache.
    m_medium.xxh3_int64()
    out.append(_time_one(
        "Memory.xxh3_int64 64 KB (warm cached)",
        lambda: m_medium.xxh3_int64(),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "Memory.xxh3_64_digest 64 KB (warm cached)",
        lambda: m_medium.xxh3_64_digest,
        repeat=repeat, inner=200_000,
    ))

    # media_type — lazy resolve on first read; warm reads hit the slot.
    m_typed = _seed(SMALL)
    m_typed.url = "mem://anonymous.parquet"
    _ = m_typed.media_type  # warm
    out.append(_time_one(
        "Memory.media_type (warm)",
        lambda: m_typed.media_type,
        repeat=repeat, inner=500_000,
    ))

    # Equality / hashing — Holders compare by payload bytes.
    m_eq_a = _seed(SMALL)
    m_eq_b = _seed(SMALL)
    out.append(_time_one(
        "Memory == bytes (96 B equal)",
        lambda: m_eq_a == SMALL,
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "Memory == Memory (96 B equal)",
        lambda: m_eq_a == m_eq_b,
        repeat=repeat, inner=200_000,
    ))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<58s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
