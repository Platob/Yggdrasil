"""Benchmark yggdrasil wire-format pickle serialization.

Covers the full round-trip (dumps / loads) for common Python object shapes
and compares against stdlib ``pickle`` and ``cloudpickle`` as references.

Shapes exercised
----------------
- Primitives: None, bool, int (small/large), float, str (short/long), bytes
- Collections: list[int] (10 / 1 000), dict[str, int] (10 / 100), nested
- Dataclass with typed fields
- Arrow Table  (100 rows / 10 000 rows) — uses IPC, skipped if pyarrow absent

Metrics reported per scenario
------------------------------
- best / median / worst round-trip time across ``--repeat`` outer loops,
  each outer loop running ``INNER`` inner iterations (to amortise GC jitter)
- wire size (bytes)
- throughput (MB/s for bulk data scenarios)

Usage::

    PYTHONPATH=src python benchmarks/pickle/bench_pickle_ser.py
    PYTHONPATH=src python benchmarks/pickle/bench_pickle_ser.py --repeat 7
"""
from __future__ import annotations

import argparse
import dataclasses
import pickle
import statistics
import struct
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# optional deps — graceful skip
# ---------------------------------------------------------------------------

try:
    import cloudpickle as _cloudpickle
    _HAS_CLOUDPICKLE = True
except ImportError:
    _cloudpickle = None  # type: ignore[assignment]
    _HAS_CLOUDPICKLE = False

try:
    import pyarrow as pa
    _HAS_ARROW = True
except ImportError:
    pa = None  # type: ignore[assignment]
    _HAS_ARROW = False

from yggdrasil.pickle.ser.serde import dumps, loads

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

INNER = 200  # inner iterations per outer sample

_SHORT_STR = "hello world"
_LONG_STR = "x" * 4_096
_SMALL_BYTES = b"\x00" * 64
_LARGE_BYTES = b"\xff" * 32_768

_LIST_SMALL = list(range(10))
_LIST_LARGE = list(range(1_000))
_DICT_SMALL = {f"key_{i}": i for i in range(10)}
_DICT_LARGE = {f"key_{i}": i for i in range(100)}
_NESTED = {"a": list(range(20)), "b": {"c": list(range(20))}, "d": [{"x": i} for i in range(5)]}


@dataclasses.dataclass
class _Point:
    x: float
    y: float
    z: float
    label: str
    ts: datetime


_DATACLASS_OBJ = _Point(
    x=1.234,
    y=5.678,
    z=9.012,
    label="sensor-0042",
    ts=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
)


def _arrow_table(n: int) -> "pa.Table":
    return pa.table(
        {
            "id": pa.array(range(n), type=pa.int64()),
            "value": pa.array([float(i) * 1.1 for i in range(n)], type=pa.float64()),
            "label": pa.array([f"row_{i}" for i in range(n)], type=pa.string()),
        }
    )


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------

def _time_fn(fn: Callable[[], Any], *, repeat: int, inner: int) -> list[float]:
    """Return *repeat* samples; each sample times *inner* calls."""
    for _ in range(min(inner, 50)):  # warm-up
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return samples


def _wire_size(obj: Any) -> int:
    return len(dumps(obj))


def _pickle_size(obj: Any) -> int:
    return len(pickle.dumps(obj, protocol=5))


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

_HDR = f"{'scenario':<34}  {'lib':<10}  {'best µs':>9}  {'median µs':>10}  {'wire B':>8}"
_SEP = "-" * len(_HDR)


def _fmt(label: str, lib: str, samples: list[float], size: int) -> str:
    best_us = min(samples) * 1e6
    med_us = statistics.median(samples) * 1e6
    return f"{label:<34}  {lib:<10}  {best_us:>9.2f}  {med_us:>10.2f}  {size:>8,}"


# ---------------------------------------------------------------------------
# benchmark cases
# ---------------------------------------------------------------------------

def _bench_primitive(label: str, obj: Any, *, repeat: int) -> None:
    ygg_size = _wire_size(obj)
    pkl_size = _pickle_size(obj)

    ygg = _time_fn(lambda: loads(dumps(obj)), repeat=repeat, inner=INNER)
    pkl = _time_fn(lambda: pickle.loads(pickle.dumps(obj, 5)), repeat=repeat, inner=INNER)

    print(_fmt(label, "ygg", ygg, ygg_size))
    print(_fmt(label, "pickle", pkl, pkl_size))
    if _HAS_CLOUDPICKLE:
        cp = _time_fn(lambda: _cloudpickle.loads(_cloudpickle.dumps(obj)), repeat=repeat, inner=INNER)
        cp_size = len(_cloudpickle.dumps(obj))
        print(_fmt(label, "cloudpickle", cp, cp_size))


def _bench_bulk(label: str, obj: Any, *, repeat: int) -> None:
    """Like _bench_primitive but also prints MB/s throughput."""
    ygg_raw = dumps(obj)
    pkl_raw = pickle.dumps(obj, 5)
    ygg_size = len(ygg_raw)
    pkl_size = len(pkl_raw)

    ygg = _time_fn(lambda: loads(dumps(obj)), repeat=repeat, inner=max(INNER // 10, 5))
    pkl = _time_fn(lambda: pickle.loads(pickle.dumps(obj, 5)), repeat=repeat, inner=max(INNER // 10, 5))

    data_mb = len(pickle.dumps(obj, 5)) / 1e6  # rough payload size ref

    def _throughput(samples: list[float]) -> str:
        best_s = min(samples)
        return f"{data_mb / best_s:>6.1f} MB/s" if best_s > 0 else "    n/a   "

    print(_fmt(label, "ygg", ygg, ygg_size) + f"  {_throughput(ygg)}")
    print(_fmt(label, "pickle", pkl, pkl_size) + f"  {_throughput(pkl)}")
    if _HAS_CLOUDPICKLE:
        cp = _time_fn(lambda: _cloudpickle.loads(_cloudpickle.dumps(obj)), repeat=repeat, inner=max(INNER // 10, 5))
        cp_size = len(_cloudpickle.dumps(obj))
        print(_fmt(label, "cloudpickle", cp, cp_size) + f"  {_throughput(cp)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(repeat: int) -> None:
    print()
    print("=" * 90)
    print(f"  yggdrasil pickle benchmark  (repeat={repeat}, inner={INNER})")
    print("=" * 90)
    print(_HDR)
    print(_SEP)

    # --- primitives ---
    _bench_primitive("None", None, repeat=repeat)
    _bench_primitive("bool True", True, repeat=repeat)
    _bench_primitive("int (small: 42)", 42, repeat=repeat)
    _bench_primitive("int (large: 2^60)", 2**60, repeat=repeat)
    _bench_primitive("float (3.14)", 3.14, repeat=repeat)
    _bench_primitive("str (11 chars)", _SHORT_STR, repeat=repeat)
    _bench_primitive("str (4 096 chars)", _LONG_STR, repeat=repeat)
    _bench_primitive("bytes (64 B)", _SMALL_BYTES, repeat=repeat)
    _bench_primitive("bytes (32 KiB)", _LARGE_BYTES, repeat=repeat)
    _bench_primitive("Decimal", Decimal("123.456"), repeat=repeat)
    _bench_primitive("datetime (tz-aware)", _DATACLASS_OBJ.ts, repeat=repeat)

    print(_SEP)

    # --- collections ---
    _bench_primitive("list[int] (10)", _LIST_SMALL, repeat=repeat)
    _bench_primitive("list[int] (1 000)", _LIST_LARGE, repeat=repeat)
    _bench_primitive("dict[str,int] (10)", _DICT_SMALL, repeat=repeat)
    _bench_primitive("dict[str,int] (100)", _DICT_LARGE, repeat=repeat)
    _bench_primitive("nested dict/list", _NESTED, repeat=repeat)

    print(_SEP)

    # --- dataclass ---
    _bench_primitive("dataclass (5 fields)", _DATACLASS_OBJ, repeat=repeat)

    print(_SEP)

    # --- Arrow (bulk) ---
    if _HAS_ARROW:
        tbl_small = _arrow_table(100)
        tbl_large = _arrow_table(10_000)
        print(_HDR + "  throughput")
        print(_SEP + "-" * 14)
        _bench_bulk("Arrow Table (100 rows)", tbl_small, repeat=repeat)
        _bench_bulk("Arrow Table (10 000 rows)", tbl_large, repeat=repeat)
    else:
        print("  [pyarrow not installed — Arrow benchmarks skipped]")

    print(_SEP)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5, help="Outer timing loops")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    run(repeat=args.repeat)
