"""Benchmark :class:`LocalPath` filesystem-shape operations.

The companion ``bench_io_path.py`` covers the pure-URL surface; this
file targets the syscalls — :meth:`exists`, :meth:`stat`, :meth:`is_file`,
:meth:`iterdir`, :meth:`mkdir`, :meth:`unlink`, :meth:`touch`, the
``__fspath__`` round-trip — that filesystem callers actually pay for.

Each scenario operates against a pre-seeded :class:`tempfile.TemporaryDirectory`
so we don't measure ``tmpdir`` setup costs in the loop.

Usage::

    PYTHONPATH=src python benchmarks/io/path/bench_io_local_fs.py
"""
from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time
from pathlib import Path as _PyPath
from typing import Callable

from yggdrasil.io.path.local_path import LocalPath


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
# Scenarios
# ---------------------------------------------------------------------------


def scenarios(repeat: int, tmp: _PyPath) -> list[dict]:
    out: list[dict] = []

    # Seed: a file with a few bytes, a directory with a handful of children.
    payload_path = tmp / "payload.bin"
    payload_path.write_bytes(b"x" * 4096)
    dir_path = tmp / "children"
    dir_path.mkdir()
    for i in range(8):
        (dir_path / f"part-{i:03d}.bin").write_bytes(b"y" * 16)

    missing_path = tmp / "does-not-exist.bin"

    p_file = LocalPath(str(payload_path))
    p_dir = LocalPath(str(dir_path))
    p_missing = LocalPath(str(missing_path))

    # exists / is_file / is_dir — each is one os.stat. The "missing"
    # branch is the common cold-cache fast-no.
    out.append(_time_one(
        "LocalPath.exists() (file present)",
        lambda: p_file.exists(),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "LocalPath.exists() (missing)",
        lambda: p_missing.exists(),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "LocalPath.is_file()",
        lambda: p_file.is_file(),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "LocalPath.is_dir()",
        lambda: p_dir.is_dir(),
        repeat=repeat, inner=20_000,
    ))

    # stat / size / mtime — Holder reads stat per call (no caching across
    # acquires), so each is one ``os.stat``.
    out.append(_time_one(
        "LocalPath.stat() (closed)",
        lambda: p_file.stat(),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "LocalPath.size (closed)",
        lambda: p_file.size,
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "LocalPath.mtime (closed)",
        lambda: p_file.mtime,
        repeat=repeat, inner=20_000,
    ))

    # iterdir — bulk scandir + per-entry Path materialization.
    out.append(_time_one(
        "LocalPath.iterdir() drain (8 children)",
        lambda: [p for p in p_dir.iterdir()],
        repeat=repeat, inner=2_000,
    ))

    # full_path / __fspath__ / os_path — every native-engine dispatch
    # in the IO leaves goes through these.
    out.append(_time_one(
        "LocalPath.full_path()",
        lambda: p_file.full_path(),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "LocalPath.os_path",
        lambda: p_file.os_path,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "os.fspath(LocalPath)",
        lambda: os.fspath(p_file),
        repeat=repeat, inner=500_000,
    ))

    # mkdir / touch / unlink — capture only the syscall overhead; the
    # bench rebuilds the same target each iteration so we measure one
    # round-trip rather than ``parents=True`` chain cost.
    def cycle_mkdir() -> None:
        target = tmp / "mk-and-rm"
        LocalPath(str(target)).mkdir(parents=False, exist_ok=True)
        os.rmdir(str(target))

    def cycle_touch_unlink() -> None:
        target = tmp / "touch-and-rm.bin"
        LocalPath(str(target)).touch().unlink(missing_ok=True)

    out.append(_time_one(
        "LocalPath.mkdir + rmdir round-trip",
        cycle_mkdir, repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "LocalPath.touch + unlink round-trip",
        cycle_touch_unlink, repeat=repeat, inner=2_000,
    ))

    # Construction shapes.
    out.append(_time_one(
        "LocalPath(str path) construct",
        lambda: LocalPath(str(payload_path)),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "LocalPath.staging_path() (anonymous)",
        lambda: LocalPath.staging_path(),
        repeat=repeat, inner=10_000,
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
    print(f"# {'label':<62s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    with tempfile.TemporaryDirectory() as tmp:
        for row in scenarios(args.repeat, _PyPath(tmp)):
            print(_fmt(row))


if __name__ == "__main__":
    main()
