"""Benchmark the LRU + dict caches added for data-cast hot paths.

Optimizations under test
------------------------
1. ``_coerce_interval`` LRU cache (maxsize=256)
   ``truncate_datetime`` / ``iter_datetime_ranges`` parse the same
   ISO interval string on every call.  The LRU cache makes repeated
   calls to ``_coerce_interval("PT15M")`` a single dict lookup.

2. ``_DATACLASS_HINTS_CACHE`` in ``convert_to_python_dataclass``
   ``get_type_hints()`` does module-dict + forward-ref resolution on
   every call; caching the result per class makes the second and
   subsequent conversions pay only a single dict lookup.

3. ``str_to_tzinfo`` uppercase normalisation fix
   The fast-path set membership check is now ``{"ETC/UTC", "UTC", "Z"}``
   (all uppercase, matching ``u = s.upper()``).  The previous set
   contained the mixed-case ``"Etc/UTC"`` which never matched because
   ``u`` is always uppercase — forcing every "Etc/UTC" input to fall
   through to ``ZoneInfo(s)``.

Usage::

    PYTHONPATH=src python benchmarks/data/bench_cast_caches.py
    PYTHONPATH=src python benchmarks/data/bench_cast_caches.py --repeat 7
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from yggdrasil.data.cast.datetime import _coerce_interval, str_to_tzinfo
from yggdrasil.data.cast.registry import (
    _DATACLASS_HINTS_CACHE,
    convert,
    convert_to_python_dataclass,
)


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
        scale = 1e9
        unit = "ns"
    return (
        f"{r['label']:<68s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# 1. _coerce_interval — cold vs warm
# ---------------------------------------------------------------------------


def _bench_coerce_interval(repeat: int) -> list[dict]:
    INTERVALS = ["PT15M", "P1D", "P1M", "P1Y", "PT1H30M"]

    def cold():
        _coerce_interval.cache_clear()
        for s in INTERVALS:
            _coerce_interval(s)

    def warm():
        for s in INTERVALS:
            _coerce_interval(s)

    # Prime the cache before warm bench.
    for s in INTERVALS:
        _coerce_interval(s)

    return [
        _time_one(
            "_coerce_interval x5 intervals — cold (cache cleared each time)",
            cold, repeat=repeat, inner=10_000,
        ),
        _time_one(
            "_coerce_interval x5 intervals — warm (LRU hit)",
            warm, repeat=repeat, inner=100_000,
        ),
    ]


# ---------------------------------------------------------------------------
# 2. convert_to_python_dataclass — cold vs warm hints cache
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    host: str = "localhost"
    port: int = 5432
    debug: bool = False
    timeout: float = 30.0


_INPUT = {"host": "db.example.com", "port": "5433", "debug": "true", "timeout": "60"}


def _bench_dataclass_hints(repeat: int) -> list[dict]:
    def cold():
        _DATACLASS_HINTS_CACHE.pop(_Config, None)
        convert_to_python_dataclass(_INPUT, _Config)

    def warm():
        convert_to_python_dataclass(_INPUT, _Config)

    # Prime the cache.
    convert_to_python_dataclass(_INPUT, _Config)

    return [
        _time_one(
            "convert_to_python_dataclass — cold (hints cache evicted each time)",
            cold, repeat=repeat, inner=10_000,
        ),
        _time_one(
            "convert_to_python_dataclass — warm (hints cache hit)",
            warm, repeat=repeat, inner=50_000,
        ),
    ]


# ---------------------------------------------------------------------------
# 3. str_to_tzinfo fast-path correctness + speed
# ---------------------------------------------------------------------------


def _bench_tzinfo(repeat: int) -> list[dict]:
    return [
        _time_one(
            "str_to_tzinfo: 'UTC' (fast-path set hit)",
            lambda: str_to_tzinfo("UTC"),
            repeat=repeat, inner=200_000,
        ),
        _time_one(
            "str_to_tzinfo: 'Z' (fast-path set hit)",
            lambda: str_to_tzinfo("Z"),
            repeat=repeat, inner=200_000,
        ),
        _time_one(
            "str_to_tzinfo: 'Etc/UTC' (now fast-path — was ZoneInfo fallback)",
            lambda: str_to_tzinfo("Etc/UTC"),
            repeat=repeat, inner=200_000,
        ),
        _time_one(
            "str_to_tzinfo: 'ETC/UTC' (uppercase variant — same fast path)",
            lambda: str_to_tzinfo("ETC/UTC"),
            repeat=repeat, inner=200_000,
        ),
        _time_one(
            "str_to_tzinfo: '+05:30' (offset — regex path)",
            lambda: str_to_tzinfo("+05:30"),
            repeat=repeat, inner=100_000,
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    r = args.repeat

    sections = [
        ("_coerce_interval — LRU cache cold vs warm", _bench_coerce_interval(r)),
        ("convert_to_python_dataclass — hints cache cold vs warm", _bench_dataclass_hints(r)),
        ("str_to_tzinfo — fast-path fix + variants", _bench_tzinfo(r)),
    ]

    for title, results in sections:
        print(f"\n{'─'*80}")
        print(f"  {title}")
        print(f"{'─'*80}")
        for row in results:
            print(_fmt(row))

    info = _coerce_interval.cache_info()
    print(f"\n_coerce_interval cache: hits={info.hits} misses={info.misses} "
          f"maxsize={info.maxsize} currsize={info.currsize}")


if __name__ == "__main__":
    main()
