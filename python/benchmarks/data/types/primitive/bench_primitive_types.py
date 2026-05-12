"""Benchmark :mod:`yggdrasil.data.types.primitive` — construction,
identity, projections, dict / string round-trips.

Mirrors the source tree: every primitive ``DataType`` subclass
(``IntegerType``, ``FloatingPointType``, ``BooleanType``,
``StringType``, ``DateType``, ``TimestampType``, ...) lives under
``yggdrasil.data.types.primitive``, so its bench lives next to it
here.

Construction, ``type_id``, hash, equality, ``to_arrow`` /
``to_polars`` (warm + cold), and the ``from_*`` dispatch path are
hit on every cast site / cross-engine handoff.

Usage::

    PYTHONPATH=src python benchmarks/data/types/primitive/bench_primitive_types.py
    PYTHONPATH=src python benchmarks/data/types/primitive/bench_primitive_types.py --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.data import DataType, DataTypeId
from yggdrasil.data.types.primitive import (
    BooleanType,
    DateType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
)


# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------


PRIMITIVE_DT = IntegerType()
STRING_DT = StringType()
TS_DT = TimestampType(unit="us", tz="UTC")

PA_INT = pa.int64()
PA_STRING = pa.string()
PA_TS = pa.timestamp("us", tz="UTC")

INT_DICT = {"id": int(DataTypeId.INT64), "name": "INT64"}


# ---------------------------------------------------------------------------
# Timing helpers.
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
# Scenarios.
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []

    # Construction — singleton fast path on default args.
    results.append(_time_one(
        "construct: IntegerType() singleton",
        lambda: IntegerType(),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "construct: StringType() singleton",
        lambda: StringType(),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "construct: BooleanType() singleton",
        lambda: BooleanType(),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "construct: DateType() singleton",
        lambda: DateType(),
        repeat=repeat, inner=200_000,
    ))
    # Non-default construction — singleton miss, fresh dataclass allocation.
    results.append(_time_one(
        "construct: IntegerType(byte_size=4) specialized",
        lambda: IntegerType(byte_size=4, signed=True),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "construct: TimestampType(unit='ns', tz='UTC')",
        lambda: TimestampType(unit="ns", tz="UTC"),
        repeat=repeat, inner=50_000,
    ))

    # Identity / equality / hash.
    results.append(_time_one(
        "id: dtype.type_id",
        lambda: PRIMITIVE_DT.type_id,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "eq: dtype == dtype (same)",
        lambda: PRIMITIVE_DT == PRIMITIVE_DT,
        repeat=repeat, inner=500_000,
    ))
    other_int = IntegerType(byte_size=8, signed=True)
    results.append(_time_one(
        "eq: dtype == dtype (equal, different instances)",
        lambda: PRIMITIVE_DT == other_int,
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "hash: hash(IntegerType())",
        lambda: hash(PRIMITIVE_DT),
        repeat=repeat, inner=500_000,
    ))

    # Engine projections — warm cache + the cold first call.
    results.append(_time_one(
        "to_arrow: warm cached IntegerType",
        lambda: PRIMITIVE_DT.to_arrow(),
        repeat=repeat, inner=500_000,
    ))

    # Dispatch — from_arrow_type / from_str / from_pytype / from_any.
    results.append(_time_one(
        "from_arrow_type: pa.int64()",
        lambda: DataType.from_arrow_type(PA_INT),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_one(
        "from_arrow_type: pa.timestamp('us', tz='UTC')",
        lambda: DataType.from_arrow_type(PA_TS),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "from_str: 'int64'",
        lambda: DataType.from_str("int64"),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "from_str: 'timestamp(us, UTC)'",
        lambda: DataType.from_str("timestamp(us, UTC)"),
        repeat=repeat, inner=5_000,
    ))
    results.append(_time_one(
        "from_pytype: int",
        lambda: DataType.from_pytype(int),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_one(
        "from_pytype: dt.datetime",
        lambda: DataType.from_pytype(dt.datetime),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_one(
        "from_any: existing DataType (identity)",
        lambda: DataType.from_any(PRIMITIVE_DT),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "from_any: pa.DataType",
        lambda: DataType.from_any(PA_INT),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "from_any: str",
        lambda: DataType.from_any("int64"),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "from_any: int builtin",
        lambda: DataType.from_any(int),
        repeat=repeat, inner=20_000,
    ))

    # Dict round-trip — used by JSON metadata persistence + cross-engine handoff.
    results.append(_time_one(
        "to_dict: IntegerType",
        lambda: PRIMITIVE_DT.to_dict(),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "from_dict: IntegerType payload",
        lambda: DataType.from_dict(INT_DICT),
        repeat=repeat, inner=20_000,
    ))

    # Autotag — emitted as Databricks-friendly schema metadata.
    results.append(_time_one(
        "autotag: IntegerType",
        lambda: PRIMITIVE_DT.autotag(),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "autotag: TimestampType",
        lambda: TS_DT.autotag(),
        repeat=repeat, inner=50_000,
    ))

    return results


# ---------------------------------------------------------------------------
# CLI.
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
