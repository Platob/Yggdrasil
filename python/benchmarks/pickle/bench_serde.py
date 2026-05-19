"""Benchmark yggdrasil pickle serialization / deserialization.

Scenarios
---------
Primitives     — None, bool, int (small/large), str (short/100 KB), bytes (small/1 MB)
Collections    — list[int] (1K), dict[str, int] (1K)
Arrow          — pa.Table with 100 K rows × 4 columns
Polars         — pl.DataFrame with 100 K rows × 4 columns (serialized via Arrow)
Round-trip     — dumps() → loads() combined latency

Inner counts are calibrated so each scenario takes < 3 s per repeat:
  primitives ~1 µs → inner=50_000, collections ~50 ms → inner=10, Arrow ~12 ms → inner=20.

Usage::

    PYTHONPATH=src python benchmarks/pickle/bench_serde.py
    PYTHONPATH=src python benchmarks/pickle/bench_serde.py --repeat 7
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import statistics
import time
import uuid
from typing import Callable

import pyarrow as pa

from yggdrasil.pickle.ser.serde import dumps, loads


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NONE_OBJ = None
BOOL_OBJ = True
INT_SMALL = 42
INT_LARGE = 2**60
STR_SHORT = "hello world"
STR_MEDIUM = "x" * 1_000
STR_LARGE = "x" * 100_000
BYTES_SMALL = b"hello world"
BYTES_LARGE = b"x" * 1_000_000
LIST_INT = list(range(1_000))
DICT_STR_INT = {str(i): i for i in range(1_000)}
LIST_STR = [f"item-{i:04d}" for i in range(100)]
TUPLE_INT = tuple(range(100))

DT_OBJ = datetime.datetime(2024, 1, 15, 10, 30, 0, tzinfo=datetime.timezone.utc)
UUID_OBJ = uuid.UUID("12345678-1234-5678-1234-567812345678")


@dataclasses.dataclass
class Point:
    x: float
    y: float
    label: str


DATACLASS_OBJ = Point(x=1.5, y=2.5, label="origin")

# Arrow — 100K rows; created once, reused across all timing loops.
_PA_TABLE = pa.table({
    "id": pa.array(range(100_000), type=pa.int64()),
    "value": pa.array([i * 0.1 for i in range(100_000)], type=pa.float64()),
    "label": pa.array([f"row-{i:06d}" for i in range(100_000)], type=pa.large_utf8()),
    "flag": pa.array([i % 2 == 0 for i in range(100_000)], type=pa.bool_()),
})

try:
    import polars as pl
    _PL_DF = pl.from_arrow(_PA_TABLE)
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

# Pre-serialized bytes for load-only scenarios (avoids re-serialization cost
# inside the timed loop).
_NONE_BYTES = dumps(NONE_OBJ)
_BOOL_BYTES = dumps(BOOL_OBJ)
_INT_SMALL_BYTES = dumps(INT_SMALL)
_INT_LARGE_BYTES = dumps(INT_LARGE)
_STR_SHORT_BYTES = dumps(STR_SHORT)
_STR_MEDIUM_BYTES = dumps(STR_MEDIUM)
_STR_LARGE_BYTES = dumps(STR_LARGE)
_BYTES_SMALL_BYTES = dumps(BYTES_SMALL)
_BYTES_LARGE_BYTES = dumps(BYTES_LARGE)
_DT_BYTES = dumps(DT_OBJ)
_UUID_BYTES = dumps(UUID_OBJ)
_DC_BYTES = dumps(DATACLASS_OBJ)
_LIST_INT_BYTES = dumps(LIST_INT)
_DICT_BYTES = dumps(DICT_STR_INT)
_LIST_STR_BYTES = dumps(LIST_STR)
_TUPLE_BYTES = dumps(TUPLE_INT)
_ARROW_BYTES = dumps(_PA_TABLE)
_POLARS_BYTES = dumps(_PL_DF) if _HAS_POLARS else None


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_one(
    label: str,
    fn: Callable[[], object],
    *,
    repeat: int,
    inner: int,
    warmup: int | None = None,
) -> dict:
    n_warmup = warmup if warmup is not None else max(3, min(inner, 10))
    for _ in range(n_warmup):
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
        f"{r['label']:<62s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Serialize (dumps) scenarios
# ---------------------------------------------------------------------------

def _serialize_scenarios(repeat: int):  # type: ignore[return]  # yields, not list
    yield _time_one("dumps(None)", lambda: dumps(NONE_OBJ), repeat=repeat, inner=50_000)
    yield _time_one("dumps(True)", lambda: dumps(BOOL_OBJ), repeat=repeat, inner=50_000)
    yield _time_one("dumps(42)", lambda: dumps(INT_SMALL), repeat=repeat, inner=50_000)
    yield _time_one("dumps(2**60)", lambda: dumps(INT_LARGE), repeat=repeat, inner=50_000)
    yield _time_one("dumps(str 11B)", lambda: dumps(STR_SHORT), repeat=repeat, inner=50_000)
    yield _time_one("dumps(str 1KB)", lambda: dumps(STR_MEDIUM), repeat=repeat, inner=5_000)
    yield _time_one("dumps(str 100KB)", lambda: dumps(STR_LARGE), repeat=repeat, inner=200)
    yield _time_one("dumps(bytes 11B)", lambda: dumps(BYTES_SMALL), repeat=repeat, inner=50_000)
    yield _time_one("dumps(bytes 1MB)", lambda: dumps(BYTES_LARGE), repeat=repeat, inner=50)
    yield _time_one("dumps(datetime utc)", lambda: dumps(DT_OBJ), repeat=repeat, inner=20_000)
    yield _time_one("dumps(uuid)", lambda: dumps(UUID_OBJ), repeat=repeat, inner=20_000)
    yield _time_one("dumps(dataclass Point)", lambda: dumps(DATACLASS_OBJ), repeat=repeat, inner=2_000)
    yield _time_one("dumps(list[int] 1K)", lambda: dumps(LIST_INT), repeat=repeat, inner=10)
    yield _time_one("dumps(dict[str,int] 1K)", lambda: dumps(DICT_STR_INT), repeat=repeat, inner=5)
    yield _time_one("dumps(list[str] 100)", lambda: dumps(LIST_STR), repeat=repeat, inner=100)
    yield _time_one("dumps(tuple[int] 100)", lambda: dumps(TUPLE_INT), repeat=repeat, inner=500)
    yield _time_one("dumps(pa.Table 100K rows)", lambda: dumps(_PA_TABLE), repeat=repeat, inner=10)
    if _HAS_POLARS:
        yield _time_one("dumps(pl.DataFrame 100K rows)", lambda: dumps(_PL_DF), repeat=repeat, inner=10)


# ---------------------------------------------------------------------------
# Deserialize (loads) scenarios
# ---------------------------------------------------------------------------

def _deserialize_scenarios(repeat: int):
    yield _time_one("loads(None)", lambda: loads(_NONE_BYTES), repeat=repeat, inner=50_000)
    yield _time_one("loads(True)", lambda: loads(_BOOL_BYTES), repeat=repeat, inner=50_000)
    yield _time_one("loads(42)", lambda: loads(_INT_SMALL_BYTES), repeat=repeat, inner=50_000)
    yield _time_one("loads(2**60)", lambda: loads(_INT_LARGE_BYTES), repeat=repeat, inner=50_000)
    yield _time_one("loads(str 11B)", lambda: loads(_STR_SHORT_BYTES), repeat=repeat, inner=50_000)
    yield _time_one("loads(str 1KB)", lambda: loads(_STR_MEDIUM_BYTES), repeat=repeat, inner=10_000)
    yield _time_one("loads(str 100KB)", lambda: loads(_STR_LARGE_BYTES), repeat=repeat, inner=500)
    yield _time_one("loads(bytes 11B)", lambda: loads(_BYTES_SMALL_BYTES), repeat=repeat, inner=50_000)
    yield _time_one("loads(bytes 1MB)", lambda: loads(_BYTES_LARGE_BYTES), repeat=repeat, inner=50)
    yield _time_one("loads(datetime utc)", lambda: loads(_DT_BYTES), repeat=repeat, inner=20_000)
    yield _time_one("loads(uuid)", lambda: loads(_UUID_BYTES), repeat=repeat, inner=20_000)
    yield _time_one("loads(dataclass Point)", lambda: loads(_DC_BYTES), repeat=repeat, inner=2_000)
    yield _time_one("loads(list[int] 1K)", lambda: loads(_LIST_INT_BYTES), repeat=repeat, inner=10)
    yield _time_one("loads(dict[str,int] 1K)", lambda: loads(_DICT_BYTES), repeat=repeat, inner=5)
    yield _time_one("loads(list[str] 100)", lambda: loads(_LIST_STR_BYTES), repeat=repeat, inner=100)
    yield _time_one("loads(tuple[int] 100)", lambda: loads(_TUPLE_BYTES), repeat=repeat, inner=500)
    yield _time_one("loads(pa.Table 100K rows)", lambda: loads(_ARROW_BYTES), repeat=repeat, inner=20)
    if _HAS_POLARS:
        yield _time_one("loads(pl.DataFrame 100K rows)", lambda: loads(_POLARS_BYTES), repeat=repeat, inner=20)


# ---------------------------------------------------------------------------
# Round-trip scenarios
# ---------------------------------------------------------------------------

def _roundtrip_scenarios(repeat: int):
    yield _time_one("rt None", lambda: loads(dumps(NONE_OBJ)), repeat=repeat, inner=30_000)
    yield _time_one("rt int small", lambda: loads(dumps(INT_SMALL)), repeat=repeat, inner=30_000)
    yield _time_one("rt str short", lambda: loads(dumps(STR_SHORT)), repeat=repeat, inner=30_000)
    yield _time_one("rt bytes 1MB", lambda: loads(dumps(BYTES_LARGE)), repeat=repeat, inner=30)
    yield _time_one("rt list[int] 1K", lambda: loads(dumps(LIST_INT)), repeat=repeat, inner=5)
    yield _time_one("rt dict[str,int] 1K", lambda: loads(dumps(DICT_STR_INT)), repeat=repeat, inner=3)
    yield _time_one("rt pa.Table 100K", lambda: loads(dumps(_PA_TABLE)), repeat=repeat, inner=5)
    if _HAS_POLARS:
        yield _time_one("rt pl.DataFrame 100K", lambda: loads(dumps(_PL_DF)), repeat=repeat, inner=5)


# ---------------------------------------------------------------------------
# Aggregator + CLI
# ---------------------------------------------------------------------------

def scenarios(repeat: int):
    yield from _serialize_scenarios(repeat)
    yield from _deserialize_scenarios(repeat)
    yield from _roundtrip_scenarios(repeat)


def main() -> None:
    import sys
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<62s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    sys.stdout.flush()
    # Run each group individually and flush after each row so progress is
    # visible without waiting for all scenarios to finish.
    for group_fn in (_serialize_scenarios, _deserialize_scenarios, _roundtrip_scenarios):
        for row in group_fn(args.repeat):
            print(_fmt(row))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
