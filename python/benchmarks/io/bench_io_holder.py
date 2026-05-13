"""Benchmark the :class:`Holder` lifecycle + cursor surface.

What this covers
----------------

Every Tabular leaf opens a cursor on a holder, writes / reads through
it, and closes. The per-call overhead of that cycle bounds how cheap
the format-level reads / writes can ever be — especially on small
buffers where the format codec itself isn't the bottleneck.

Two backends in scope:

- :class:`yggdrasil.io.memory.Memory` — pure in-process, no fd /
  syscall on the open path.
- :class:`yggdrasil.io.path.local_path.LocalPath` — fd-backed, one
  ``os.open`` per acquire and one ``os.close`` per release.

Spill-to-disk + remote path holders are out of scope; the local-path
+ memory split already captures the "cheap" / "syscall" ends of the
spectrum.

Usage::

    PYTHONPATH=src python benchmarks/io/bench_io_holder.py
    PYTHONPATH=src python benchmarks/io/bench_io_holder.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path as _PyPath
from typing import Callable

from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SMALL = b"hello world\n" * 8                            # 96 B
MEDIUM = b"x" * 64_000                                   # 64 KB
LARGE = b"x" * 1_000_000                                 # ~1 MB


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 100)):
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
        f"{r['label']:<62s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Memory-backed scenarios — the no-syscall ceiling for the cycle.
# ---------------------------------------------------------------------------


def _memory_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    seeded = Memory(SMALL)

    def acquire_release():
        m = Memory()
        m.acquire()
        m.close()

    def open_cursor_cycle():
        m = Memory()
        with m.open() as bio:
            bio.write(SMALL)

    def open_view():
        with seeded.open(mode="rb") as bio:
            bio.read()

    def write_bytes_payload():
        m = Memory()
        m.write_bytes(SMALL)

    out.append(_time_one(
        "Memory acquire+close (no payload)",
        acquire_release, repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "Memory open() write+close (96 B)",
        open_cursor_cycle, repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "Memory open(rb) read+close (96 B)",
        open_view, repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "Memory.write_bytes(96 B) (fresh)",
        write_bytes_payload, repeat=repeat, inner=20_000,
    ))

    seeded_medium = Memory(MEDIUM)

    def open_read_medium():
        with seeded_medium.open(mode="rb") as bio:
            bio.read()

    out.append(_time_one(
        "Memory open(rb) read+close (64 KB)",
        open_read_medium, repeat=repeat, inner=5_000,
    ))

    return out


# ---------------------------------------------------------------------------
# LocalPath-backed scenarios — measures the fd open/close overhead.
# ---------------------------------------------------------------------------


def _localpath_scenarios(repeat: int, tmp: _PyPath) -> list[dict]:
    out: list[dict] = []

    # Pre-staged file we re-open repeatedly — captures the "warm path"
    # cost: known-good file, no mkdir retry, just open / close.
    payload_path = tmp / "warm.bin"
    payload_path.write_bytes(SMALL)

    p_warm = LocalPath(str(payload_path))

    def acquire_release_warm():
        p = LocalPath(str(payload_path))
        p.acquire()
        p.close()

    def open_read_warm():
        with p_warm.open(mode="rb") as bio:
            bio.read()

    def write_overwrite_warm():
        with p_warm.open(mode="wb") as bio:
            bio.write(SMALL)

    out.append(_time_one(
        "LocalPath acquire+close (warm path)",
        acquire_release_warm, repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "LocalPath open(rb) read+close (96 B)",
        open_read_warm, repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "LocalPath open(wb) write+close (96 B)",
        write_overwrite_warm, repeat=repeat, inner=2_000,
    ))

    # The fd-direct path: pread / pwrite without opening a cursor.
    out.append(_time_one(
        "LocalPath.read_bytes() (closed, 96 B)",
        lambda: p_warm.read_bytes(),
        repeat=repeat, inner=2_000,
    ))

    # Repeated reads while the fd stays open: amortizes the open cost
    # over many reads — what every IO leaf does inside one cursor.
    def repeated_pread():
        p = LocalPath(str(payload_path))
        p.acquire()
        try:
            for _ in range(8):
                p.read_bytes()
        finally:
            p.close()

    out.append(_time_one(
        "LocalPath open + 8x read_bytes + close (96 B)",
        repeated_pread, repeat=repeat, inner=1_000,
    ))

    # Medium-size payload: the open/close overhead amortizes faster,
    # but the per-read syscall + bytes copy starts to show.
    payload_path_med = tmp / "warm-medium.bin"
    payload_path_med.write_bytes(MEDIUM)
    p_med = LocalPath(str(payload_path_med))

    def read_medium_open():
        with p_med.open(mode="rb") as bio:
            bio.read()

    out.append(_time_one(
        "LocalPath open(rb) read+close (64 KB)",
        read_medium_open, repeat=repeat, inner=1_000,
    ))

    # Anonymous staging path: minted under the staging dir; the round
    # trip captures the per-call cost of "give me a scratch buffer."
    def staging_round_trip():
        p = LocalPath.staging_path()
        with p.open(mode="wb") as bio:
            bio.write(SMALL)
        p.close()

    out.append(_time_one(
        "LocalPath.staging_path write+close (96 B)",
        staging_round_trip, repeat=repeat, inner=500,
    ))

    return out


# ---------------------------------------------------------------------------
# Aggregator + CLI
# ---------------------------------------------------------------------------


def scenarios(repeat: int, tmp: _PyPath) -> list[dict]:
    return [
        *_memory_scenarios(repeat),
        *_localpath_scenarios(repeat, tmp),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<62s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    with tempfile.TemporaryDirectory() as tmp:
        for row in scenarios(args.repeat, _PyPath(tmp)):
            print(_fmt(row))


if __name__ == "__main__":
    main()
