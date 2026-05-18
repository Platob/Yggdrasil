"""Benchmark the pickle serde hot paths: dumps / loads roundtrip.

Covers the four tiers that matter most in practice:

* Small primitives — str, int, float, bool, None
* Medium Python objects — dicts with mixed-type values
* Large bytes payload — above and below the compression threshold
* Arrow table — the dominant production use case

Usage::

    PYTHONPATH=src python benchmarks/pickle/bench_serde.py
    PYTHONPATH=src python benchmarks/pickle/bench_serde.py --repeat 7
"""
from __future__ import annotations

import argparse
import dataclasses
import statistics
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import pyarrow as pa

from yggdrasil.pickle import dumps, loads
from yggdrasil.pickle.ser.constants import (
    CODEC_NONE,
    CODEC_ZLIB,
    CODEC_ZSTD,
    COMPRESS_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_STR_SMALL = "hello, world"
_STR_LARGE = "x" * (COMPRESS_THRESHOLD + 1)  # above threshold
_INT_SMALL = 42
_FLOAT_VAL = 3.141592653589793
_BYTES_SMALL = b"\x00\xff" * 64
_BYTES_LARGE = b"\xab\xcd" * (COMPRESS_THRESHOLD // 2 + 1)  # above threshold


@dataclasses.dataclass
class _Point:
    x: float
    y: float
    label: str


_DICT_MEDIUM = {
    "id": 99,
    "name": "Alice",
    "score": 98.6,
    "active": True,
    "tags": ["a", "b", "c"],
    "meta": {"created": "2024-01-01", "version": 3},
}

_LIST_MEDIUM = list(range(200))

_DATACLASS_OBJ = _Point(x=1.0, y=2.5, label="origin")

_ARROW_TABLE_SMALL = pa.table(
    {
        "id": pa.array(range(1_000), type=pa.int64()),
        "value": pa.array([1.5] * 1_000, type=pa.float64()),
        "name": pa.array(["item"] * 1_000, type=pa.string()),
    }
)

_ARROW_TABLE_LARGE = pa.table(
    {
        "id": pa.array(range(50_000), type=pa.int64()),
        "amount": pa.array([9.99] * 50_000, type=pa.float64()),
        "qty": pa.array([1] * 50_000, type=pa.int32()),
        "name": pa.array(["product"] * 50_000, type=pa.string()),
        "active": pa.array([True] * 50_000, type=pa.bool_()),
    }
)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _time_one(
    label: str,
    fn: Callable[[], Any],
    *,
    repeat: int,
    inner: int,
) -> dict:
    # Warm up — enough to trigger JIT (none here) and lazy imports.
    for _ in range(min(inner, 50)):
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


def _fmt(r: dict, *, extra: str = "") -> str:
    best = r["best"]
    if best < 1e-6:
        scale, unit = 1e9, "ns"
    elif best < 1e-3:
        scale, unit = 1e6, "us"
    else:
        scale, unit = 1e3, "ms"
    line = (
        f"{r['label']:<58s}  "
        f"best={r['best'] * scale:9.2f} {unit}  "
        f"median={r['median'] * scale:9.2f} {unit}"
    )
    if extra:
        line += f"  {extra}"
    return line


# ---------------------------------------------------------------------------
# Scenario groups
# ---------------------------------------------------------------------------

def _size_kib(blob: bytes) -> str:
    n = len(blob)
    if n < 1024:
        return f"{n}B"
    return f"{n / 1024:.1f}KiB"


def _primitive_scenarios(repeat: int) -> list[dict]:
    out = []
    cases: list[tuple[str, Any, int]] = [
        ("None", None, 10_000),
        ("bool True", True, 10_000),
        ("int small (42)", 42, 10_000),
        ("float (pi)", 3.141592653589793, 10_000),
        ("str small (12 chars)", _STR_SMALL, 10_000),
        ("bytes small (128 B)", _BYTES_SMALL, 10_000),
        ("UUID", uuid.UUID("12345678-1234-5678-1234-567812345678"), 5_000),
        ("datetime (tz-aware)", datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc), 5_000),
    ]
    for name, obj, inner in cases:
        blob = dumps(obj)
        extra = _size_kib(blob)
        out.append(_time_one(
            f"dumps primitive: {name}",
            lambda o=obj: dumps(o),
            repeat=repeat, inner=inner,
        ) | {"_extra": extra})
        out.append(_time_one(
            f"loads primitive: {name}",
            lambda b=blob: loads(b),
            repeat=repeat, inner=inner,
        ) | {"_extra": extra})
    return out


def _medium_scenarios(repeat: int) -> list[dict]:
    out = []
    cases: list[tuple[str, Any, int]] = [
        ("dict medium (6 keys, nested)", _DICT_MEDIUM, 1_000),
        ("list[int] 200", _LIST_MEDIUM, 2_000),
        ("dataclass _Point", _DATACLASS_OBJ, 5_000),
    ]
    for name, obj, inner in cases:
        blob = dumps(obj)
        extra = _size_kib(blob)
        out.append(_time_one(
            f"dumps medium: {name}",
            lambda o=obj: dumps(o),
            repeat=repeat, inner=inner,
        ) | {"_extra": extra})
        out.append(_time_one(
            f"loads medium: {name}",
            lambda b=blob: loads(b),
            repeat=repeat, inner=inner,
        ) | {"_extra": extra})
    return out


def _large_bytes_scenarios(repeat: int) -> list[dict]:
    out = []
    # Large bytes — exercise the compression path.
    blob_zstd = dumps(_BYTES_LARGE, codec=CODEC_ZSTD)
    blob_zlib = dumps(_BYTES_LARGE, codec=CODEC_ZLIB)
    blob_none = dumps(_BYTES_LARGE, codec=CODEC_NONE)

    raw_size = len(_BYTES_LARGE)
    for codec_name, blob, inner in [
        ("zstd", blob_zstd, 50),
        ("zlib", blob_zlib, 50),
        ("none", blob_none, 100),
    ]:
        ratio = f"raw={_size_kib(_BYTES_LARGE)} wire={_size_kib(blob)}"
        out.append(_time_one(
            f"dumps large bytes ({codec_name})",
            lambda o=_BYTES_LARGE, c={"zstd": CODEC_ZSTD, "zlib": CODEC_ZLIB, "none": CODEC_NONE}[codec_name]: dumps(o, codec=c),
            repeat=repeat, inner=inner,
        ) | {"_extra": ratio})
        out.append(_time_one(
            f"loads large bytes ({codec_name})",
            lambda b=blob: loads(b),
            repeat=repeat, inner=inner,
        ) | {"_extra": ratio})

    # Large string — typically compressible text.
    blob_str = dumps(_STR_LARGE)
    out.append(_time_one(
        "dumps large str (above threshold, auto-codec)",
        lambda: dumps(_STR_LARGE),
        repeat=repeat, inner=50,
    ) | {"_extra": f"raw={_size_kib(_STR_LARGE.encode())} wire={_size_kib(blob_str)}"})
    out.append(_time_one(
        "loads large str (above threshold, auto-codec)",
        lambda b=blob_str: loads(b),
        repeat=repeat, inner=50,
    ) | {"_extra": f"wire={_size_kib(blob_str)}"})
    return out


def _arrow_scenarios(repeat: int) -> list[dict]:
    out = []
    for name, tbl, inner in [
        ("Arrow table (1K rows × 3 cols)", _ARROW_TABLE_SMALL, 100),
        ("Arrow table (50K rows × 5 cols)", _ARROW_TABLE_LARGE, 20),
    ]:
        blob = dumps(tbl)
        extra = f"wire={_size_kib(blob)}"
        out.append(_time_one(
            f"dumps {name}",
            lambda t=tbl: dumps(t),
            repeat=repeat, inner=inner,
        ) | {"_extra": extra})
        out.append(_time_one(
            f"loads {name}",
            lambda b=blob: loads(b),
            repeat=repeat, inner=inner,
        ) | {"_extra": extra})
    return out


def _roundtrip_scenarios(repeat: int) -> list[dict]:
    """Combined dumps+loads — matches the most common call pattern."""
    out = []
    cases: list[tuple[str, Any, int]] = [
        ("roundtrip str small", _STR_SMALL, 5_000),
        ("roundtrip int", 12345, 5_000),
        ("roundtrip dict medium", _DICT_MEDIUM, 500),
        ("roundtrip Arrow small table", _ARROW_TABLE_SMALL, 50),
    ]
    for name, obj, inner in cases:
        out.append(_time_one(
            name,
            lambda o=obj: loads(dumps(o)),
            repeat=repeat, inner=inner,
        ))
    return out


def scenarios(repeat: int) -> list[dict]:
    return [
        *_primitive_scenarios(repeat),
        *_medium_scenarios(repeat),
        *_large_bytes_scenarios(repeat),
        *_arrow_scenarios(repeat),
        *_roundtrip_scenarios(repeat),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    print(f"# pickle serde benchmark  repeat={args.repeat}")
    print(f"# compression threshold = {COMPRESS_THRESHOLD // 1024} KiB")
    print(f"# {'label':<58s}  {'best':>14s}  {'median':>17s}")
    prev_group = ""
    for row in scenarios(args.repeat):
        label = row["label"]
        group = label.split()[0]
        if group != prev_group:
            print()
            prev_group = group
        print(_fmt(row, extra=row.get("_extra", "")))


if __name__ == "__main__":
    main()
