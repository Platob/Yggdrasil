"""Benchmark the :class:`Path` base API — parse, derived properties, traversal.

Local + remote subclasses have their own benches; this one measures
the construction / coercion / property surface that every path
type inherits.

Usage::

    PYTHONPATH=src python benchmarks/io/path/bench_io_path.py
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

from yggdrasil.io.path.path import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


LOCAL_STR = "/home/data/2024/q1/orders.parquet"
S3_STR = "s3://bucket/datasets/raw/year=2024/month=01/part-00000.parquet"
HTTPS_STR = "https://api.example.com/v1/orders?from=2024-01-01"

LOCAL_PATH = Path.from_(LOCAL_STR)
S3_PATH = Path.from_(S3_STR)
HTTPS_PATH = Path.from_(HTTPS_STR)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 1000)):
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

    # from_ — dispatch by scheme / shape.
    out.append(_time_one(
        "Path.from_('/abs/path.parquet')",
        lambda: Path.from_(LOCAL_STR),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "Path.from_('s3://…')",
        lambda: Path.from_(S3_STR),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "Path.from_('https://…')",
        lambda: Path.from_(HTTPS_STR),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "Path.from_(existing Path) identity",
        lambda: Path.from_(LOCAL_PATH),
        repeat=repeat, inner=500_000,
    ))

    # Property accessors — hot on every IO routing decision.
    out.append(_time_one(
        "path.url",
        lambda: LOCAL_PATH.url,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "path.parts",
        lambda: LOCAL_PATH.parts,
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "path.name",
        lambda: LOCAL_PATH.name,
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "path.parent",
        lambda: LOCAL_PATH.parent,
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "path.joinpath('subdir', 'file.csv')",
        lambda: LOCAL_PATH.joinpath("subdir", "file.csv"),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "str(path)",
        lambda: str(LOCAL_PATH),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "repr(path)",
        lambda: repr(LOCAL_PATH),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "hash(path)",
        lambda: hash(LOCAL_PATH),
        repeat=repeat, inner=500_000,
    ))

    # Pure-path derivations — every URL routing decision touches these.
    out.append(_time_one(
        "path.stem",
        lambda: LOCAL_PATH.stem,
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "path.suffix",
        lambda: LOCAL_PATH.suffix,
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "path.suffixes",
        lambda: LOCAL_PATH.suffixes,
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "path.parents",
        lambda: LOCAL_PATH.parents,
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "path.is_absolute",
        lambda: LOCAL_PATH.is_absolute,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "path / 'leaf.csv'",
        lambda: LOCAL_PATH / "leaf.csv",
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "path.with_suffix('.csv')",
        lambda: LOCAL_PATH.with_suffix(".csv"),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "path.with_stem('renamed')",
        lambda: LOCAL_PATH.with_stem("renamed"),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "path.with_name('renamed.csv')",
        lambda: LOCAL_PATH.with_name("renamed.csv"),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "path == path (distinct, equal)",
        lambda: LOCAL_PATH == Path.from_(LOCAL_STR),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "path.__fspath__()",
        lambda: LOCAL_PATH.__fspath__(),
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
