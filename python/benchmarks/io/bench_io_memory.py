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
    """Build a :class:`Memory` already containing *payload*."""
    m = Memory()
    m.reserve(len(payload))
    m._write_mv(memoryview(payload), 0)  # type: ignore[attr-defined]
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
