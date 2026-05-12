"""Benchmark :mod:`yggdrasil.data.types.nested` — struct / array / map.

Mirrors the source tree: ``yggdrasil.data.types.nested`` houses
``StructType``, ``ArrayType``, ``MapType``. Cast-heavy pipelines
hit hash / equality / projection on these per batch; deep nested
shapes (struct-of-struct, list-of-struct, map<str, list<int>>)
are the analytics-warehouse / JSON-ingest workloads.

Usage::

    PYTHONPATH=src python benchmarks/data/types/nested/bench_nested_types.py
    PYTHONPATH=src python benchmarks/data/types/nested/bench_nested_types.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.data import DataType, Field
from yggdrasil.data.types.nested import ArrayType, MapType, StructType
from yggdrasil.data.types.primitive import (
    BooleanType,
    DecimalType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
)


# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------


TS_DT = TimestampType(unit="us", tz="UTC")

ARRAY_DT = ArrayType.from_item(IntegerType())
MAP_DT = MapType.from_key_value(
    key_field=Field("k", StringType(), nullable=False),
    value_field=Field("v", IntegerType()),
)
STRUCT_DT = StructType(fields=(
    Field("id", IntegerType(), nullable=False),
    Field("name", StringType()),
    Field("amount", FloatingPointType()),
    Field("ts", TS_DT),
    Field("active", BooleanType()),
))

DEEP_STRUCT_DT = StructType(fields=(
    Field("id", IntegerType(), nullable=False),
    Field("name", StringType()),
    Field("amount", DecimalType(precision=18, scale=2)),
    Field("address", StructType(fields=(
        Field("street", StringType()),
        Field("city", StringType()),
        Field("zip", StringType()),
    ))),
    Field("tags", ArrayType.from_item(Field("item", StringType()))),
    Field("attributes", MapType.from_key_value(
        key_field=Field("k", StringType(), nullable=False),
        value_field=Field("v", StringType()),
    )),
))
LIST_OF_STRUCT_DT = ArrayType.from_item(Field(
    "item",
    StructType(fields=(
        Field("k", StringType()),
        Field("v", IntegerType()),
    )),
))
MAP_STR_LIST_DT = MapType.from_key_value(
    key_field=Field("k", StringType(), nullable=False),
    value_field=Field("v", ArrayType.from_item(Field("item", IntegerType()))),
)

PA_LIST = pa.list_(pa.int64())
PA_STRUCT = STRUCT_DT.to_arrow()
PA_DEEP = DEEP_STRUCT_DT.to_arrow()
PA_LIST_OF_STRUCT = LIST_OF_STRUCT_DT.to_arrow()
PA_MAP_STR_LIST = MAP_STR_LIST_DT.to_arrow()


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

    # type_id — read on every cast site / routing branch.
    results.append(_time_one(
        "id: StructType.type_id",
        lambda: STRUCT_DT.type_id,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "id: ArrayType.type_id",
        lambda: ARRAY_DT.type_id,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "id: MapType.type_id",
        lambda: MAP_DT.type_id,
        repeat=repeat, inner=500_000,
    ))

    # Hash — dict / set membership in schema diff caches.
    results.append(_time_one(
        "hash: hash(StructType) flat",
        lambda: hash(STRUCT_DT),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "hash: hash(ArrayType)",
        lambda: hash(ARRAY_DT),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "hash: hash(MapType)",
        lambda: hash(MAP_DT),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "hash: hash(DEEP_STRUCT) struct-of-struct+list+map",
        lambda: hash(DEEP_STRUCT_DT),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "hash: hash(LIST_OF_STRUCT)",
        lambda: hash(LIST_OF_STRUCT_DT),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "hash: hash(MAP_STR_LIST) map<str,list<int>>",
        lambda: hash(MAP_STR_LIST_DT),
        repeat=repeat, inner=100_000,
    ))

    # Equality — registry dispatch + CastOptions.need_cast.
    results.append(_time_one(
        "eq: DEEP_STRUCT == DEEP_STRUCT (identity)",
        lambda: DEEP_STRUCT_DT == DEEP_STRUCT_DT,
        repeat=repeat, inner=500_000,
    ))

    # Engine projections — warm-cache reads.
    results.append(_time_one(
        "to_arrow: StructType warm cache",
        lambda: STRUCT_DT.to_arrow(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "to_arrow: ArrayType warm cache",
        lambda: ARRAY_DT.to_arrow(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "to_arrow: MapType warm cache",
        lambda: MAP_DT.to_arrow(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "to_arrow: DEEP_STRUCT warm cache",
        lambda: DEEP_STRUCT_DT.to_arrow(),
        repeat=repeat, inner=500_000,
    ))

    # Cold projection — common in pipelines that build fresh dtypes per batch.
    results.append(_time_one(
        "to_arrow: cold StructType (new each call)",
        lambda: StructType(fields=(
            Field("id", IntegerType(), nullable=False),
            Field("name", StringType()),
        )).to_arrow(),
        repeat=repeat, inner=2_000,
    ))

    # Dict round-trip.
    results.append(_time_one(
        "to_dict: StructType",
        lambda: STRUCT_DT.to_dict(),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "to_dict: ArrayType",
        lambda: ARRAY_DT.to_dict(),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "to_dict: MapType",
        lambda: MAP_DT.to_dict(),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "to_dict: DEEP_STRUCT",
        lambda: DEEP_STRUCT_DT.to_dict(),
        repeat=repeat, inner=5_000,
    ))
    results.append(_time_one(
        "to_dict: LIST_OF_STRUCT",
        lambda: LIST_OF_STRUCT_DT.to_dict(),
        repeat=repeat, inner=20_000,
    ))

    # from_arrow_type — parquet / arrow readers feed these per schema.
    results.append(_time_one(
        "from_arrow_type: pa.struct(...)",
        lambda: DataType.from_arrow_type(PA_STRUCT),
        repeat=repeat, inner=2_000,
    ))
    results.append(_time_one(
        "from_arrow_type: pa.list_(pa.int64())",
        lambda: DataType.from_arrow_type(PA_LIST),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "from_arrow_type: pa.struct(...deep...)",
        lambda: DataType.from_arrow_type(PA_DEEP),
        repeat=repeat, inner=500,
    ))
    results.append(_time_one(
        "from_arrow_type: pa.list_(pa.struct(...))",
        lambda: DataType.from_arrow_type(PA_LIST_OF_STRUCT),
        repeat=repeat, inner=2_000,
    ))
    results.append(_time_one(
        "from_arrow_type: pa.map_(str, list<int>)",
        lambda: DataType.from_arrow_type(PA_MAP_STR_LIST),
        repeat=repeat, inner=2_000,
    ))

    # from_str — DDL-style string parser for nested shapes.
    results.append(_time_one(
        "from_str: 'struct<id:int64,name:string>'",
        lambda: DataType.from_str("struct<id:int64,name:string>"),
        repeat=repeat, inner=2_000,
    ))
    results.append(_time_one(
        "from_str: 'array<int64>'",
        lambda: DataType.from_str("array<int64>"),
        repeat=repeat, inner=5_000,
    ))
    results.append(_time_one(
        "from_str: 'map<string,int64>'",
        lambda: DataType.from_str("map<string,int64>"),
        repeat=repeat, inner=5_000,
    ))
    results.append(_time_one(
        "from_str: 'struct<id:int64,addr:struct<city:string>>'",
        lambda: DataType.from_str(
            "struct<id:int64,addr:struct<city:string>>"
        ),
        repeat=repeat, inner=2_000,
    ))

    # autotag — Unity-Catalog metadata recursion.
    results.append(_time_one(
        "autotag: StructType",
        lambda: STRUCT_DT.autotag(),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "autotag: ArrayType",
        lambda: ARRAY_DT.autotag(),
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
