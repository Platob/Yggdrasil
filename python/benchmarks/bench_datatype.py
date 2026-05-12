"""Benchmark the :class:`DataType` hot path.

Why this exists
---------------

:class:`DataType` (and its concrete subclasses) sit underneath every
:class:`Field`, every :class:`Schema`, every :class:`CastOptions`
target, and every engine-side projection (Arrow / Polars / Spark).
Construction, type-id lookup, dict / arrow projection, and string
parsing all show up in cast-heavy workloads: a folder-of-folders
read with N files and M columns pays each of these per-(file × column)
even before any actual data moves.

This benchmark targets the cheap-looking accessors and constructors
that fire repeatedly in cast pipelines:

* :meth:`DataType.__new__` singleton fast path
  (``Int64Type()`` / ``StringType()`` / ...);
* :attr:`type_id` — read on every cast site, ``equals``, and routing
  branch in ``from_arrow_type`` / ``from_str``;
* :meth:`to_arrow` / :meth:`to_polars` (lazy-cached on first call;
  the second-call cost is what production pays);
* :meth:`from_arrow_type` / :meth:`from_str` / :meth:`from_pytype` /
  :meth:`from_any` — every public API entry that takes a "type-ish"
  argument funnels through one of these;
* :meth:`to_dict` / :meth:`to_json` round-trips used for cross-engine
  metadata payloads.

Usage::

    PYTHONPATH=src python benchmarks/bench_datatype.py
    PYTHONPATH=src python benchmarks/bench_datatype.py --repeat 7
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
from yggdrasil.data.types.nested import ArrayType, MapType, StructType
from yggdrasil.data import Field


# ---------------------------------------------------------------------------
# Shared fixtures — built once so the timing loop doesn't pay for them.
# ---------------------------------------------------------------------------


PRIMITIVE_DT = IntegerType()
STRING_DT = StringType()
TS_DT = TimestampType(unit="us", tz="UTC")
ARRAY_DT = ArrayType.from_item(IntegerType())
STRUCT_DT = StructType(
    fields=(
        Field("id", IntegerType(), nullable=False),
        Field("name", StringType()),
        Field("amount", FloatingPointType()),
        Field("ts", TS_DT),
        Field("active", BooleanType()),
    )
)
MAP_DT = MapType.from_key_value(
    key_field=Field("k", StringType(), nullable=False),
    value_field=Field("v", IntegerType()),
)

PA_INT = pa.int64()
PA_STRING = pa.string()
PA_TS = pa.timestamp("us", tz="UTC")
PA_LIST = pa.list_(pa.int64())
PA_STRUCT = pa.struct(
    [
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("amount", pa.float64()),
        pa.field("ts", pa.timestamp("us")),
        pa.field("active", pa.bool_()),
    ]
)

INT_DICT = {"id": int(DataTypeId.INT64), "name": "INT64"}
STRUCT_DICT = STRUCT_DT.to_dict()


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
        scale = 1e9
        unit = "ns"
    return (
        f"{r['label']:<52s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
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

    # Identity / equality / hash — used by registry dispatch and CastOptions.
    results.append(_time_one(
        "id: dtype.type_id",
        lambda: PRIMITIVE_DT.type_id,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "id: STRUCT_DT.type_id",
        lambda: STRUCT_DT.type_id,
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
    results.append(_time_one(
        "hash: hash(STRUCT_DT)",
        lambda: hash(STRUCT_DT),
        repeat=repeat, inner=200_000,
    ))

    # Engine projections — cached after first call. Cold = first call cost,
    # warm = the lazy-cache hit cost (what the pipeline actually pays).
    results.append(_time_one(
        "to_arrow: warm cached IntegerType",
        lambda: PRIMITIVE_DT.to_arrow(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "to_arrow: warm cached StructType",
        lambda: STRUCT_DT.to_arrow(),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "to_arrow: cold StructType (new each call)",
        lambda: StructType(
            fields=(
                Field("id", IntegerType(), nullable=False),
                Field("name", StringType()),
            )
        ).to_arrow(),
        repeat=repeat, inner=2_000,
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
        "from_arrow_type: pa.struct(...)",
        lambda: DataType.from_arrow_type(PA_STRUCT),
        repeat=repeat, inner=2_000,
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
        "from_str: 'struct<id:int64,name:string>'",
        lambda: DataType.from_str("struct<id:int64,name:string>"),
        repeat=repeat, inner=2_000,
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

    # Dict round-trip — used by JSON metadata persistence, cross-engine handoff.
    results.append(_time_one(
        "to_dict: IntegerType",
        lambda: PRIMITIVE_DT.to_dict(),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "to_dict: STRUCT_DT",
        lambda: STRUCT_DT.to_dict(),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "from_dict: IntegerType payload",
        lambda: DataType.from_dict(INT_DICT),
        repeat=repeat, inner=20_000,
    ))

    # Autotag — emitted as part of Databricks-friendly schema metadata.
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
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<52s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
