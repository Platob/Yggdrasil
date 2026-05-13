"""Benchmark the Spark dtype / field dispatch path.

Why this exists
---------------

Every Spark hand-off — reading a ``pyspark.sql.DataFrame`` schema,
ingesting a Databricks Delta table descriptor, mapping a yggdrasil
:class:`Field` onto a Spark ``StructField`` — walks through
:meth:`DataType.from_spark_type` /
:meth:`Field.from_spark_field` / :meth:`Field.to_pyspark_field`.

A wide schema (50+ columns) and a deep one (struct of struct of
list of map) both bottleneck on the recursive subclass-walk
dispatch — same shape as the Arrow / Polars paths benched
elsewhere, but with the extra cost of every ``handles_spark_type``
checking against a fresh :mod:`pyspark.sql` import per call.

Skips gracefully when :mod:`pyspark` isn't installed so the bench
can run in a base install.

Usage::

    PYTHONPATH=src python benchmarks/data/bench_spark.py
    PYTHONPATH=src python benchmarks/data/bench_spark.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

try:
    import pyspark.sql.types as pst
    HAS_PYSPARK = True
except ImportError:  # pragma: no cover - bench-only optional path
    HAS_PYSPARK = False

import pyarrow as pa  # noqa: F401  (DataType import indirectly imports pa.compute)

from yggdrasil.data import Field
from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.primitive import (
    BooleanType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
)
from yggdrasil.data.types.nested import ArrayType, MapType, StructType


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
        f"{r['label']:<60s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios.
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    if not HAS_PYSPARK:
        return []

    results: list[dict] = []

    sp_long = pst.LongType()
    sp_int = pst.IntegerType()
    sp_string = pst.StringType()
    sp_double = pst.DoubleType()
    sp_bool = pst.BooleanType()
    sp_ts = pst.TimestampType()
    sp_date = pst.DateType()
    sp_decimal = pst.DecimalType(10, 2)
    sp_binary = pst.BinaryType()

    sp_array = pst.ArrayType(pst.LongType())
    sp_map = pst.MapType(pst.StringType(), pst.LongType())
    sp_struct = pst.StructType([
        pst.StructField("id", pst.LongType(), nullable=False),
        pst.StructField("name", pst.StringType()),
        pst.StructField("amount", pst.DoubleType()),
    ])
    sp_deep = pst.StructType([
        pst.StructField("id", pst.LongType(), nullable=False),
        pst.StructField("addr", pst.StructType([
            pst.StructField("city", pst.StringType()),
            pst.StructField("zip", pst.LongType()),
        ])),
        pst.StructField("tags", pst.ArrayType(pst.StringType())),
        pst.StructField("scores", pst.MapType(
            pst.StringType(),
            pst.ArrayType(pst.DoubleType()),
        )),
    ])
    sp_wide = pst.StructType([
        pst.StructField(f"c{i}", pst.LongType() if i % 2 == 0 else pst.StringType())
        for i in range(50)
    ])

    sf = pst.StructField("id", pst.LongType(), nullable=False)
    sf_with_meta = pst.StructField(
        "id", pst.LongType(), nullable=False, metadata={"comment": "primary id"}
    )
    sf_struct = pst.StructField("row", sp_struct, nullable=False)
    sf_array = pst.StructField("tags", sp_array)
    sf_map = pst.StructField("lookup", sp_map)

    # DataType.from_spark_type — cache-hit hot path. Every per-column
    # Spark schema ingest hits this; the cache makes the steady-state
    # cost a dict lookup.
    results.append(_time_one(
        "from_spark_type: LongType",
        lambda: DataType.from_spark_type(sp_long),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "from_spark_type: IntegerType",
        lambda: DataType.from_spark_type(sp_int),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "from_spark_type: StringType",
        lambda: DataType.from_spark_type(sp_string),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "from_spark_type: DoubleType",
        lambda: DataType.from_spark_type(sp_double),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "from_spark_type: BooleanType",
        lambda: DataType.from_spark_type(sp_bool),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "from_spark_type: TimestampType",
        lambda: DataType.from_spark_type(sp_ts),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "from_spark_type: DateType",
        lambda: DataType.from_spark_type(sp_date),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "from_spark_type: DecimalType(10,2)",
        lambda: DataType.from_spark_type(sp_decimal),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "from_spark_type: BinaryType",
        lambda: DataType.from_spark_type(sp_binary),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "from_spark_type: ArrayType<long>",
        lambda: DataType.from_spark_type(sp_array),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "from_spark_type: MapType<str,long>",
        lambda: DataType.from_spark_type(sp_map),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "from_spark_type: StructType[3 flat]",
        lambda: DataType.from_spark_type(sp_struct),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "from_spark_type: deep struct (struct+list+map)",
        lambda: DataType.from_spark_type(sp_deep),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_one(
        "from_spark_type: wide StructType[50 columns]",
        lambda: DataType.from_spark_type(sp_wide),
        repeat=repeat, inner=20_000,
    ))

    # Field.from_spark_field — per-column ingest.
    results.append(_time_one(
        "Field.from_spark_field: flat (no metadata)",
        lambda: Field.from_spark_field(sf),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "Field.from_spark_field: flat (with metadata)",
        lambda: Field.from_spark_field(sf_with_meta),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "Field.from_spark_field: struct child",
        lambda: Field.from_spark_field(sf_struct),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_one(
        "Field.from_spark_field: array child",
        lambda: Field.from_spark_field(sf_array),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_one(
        "Field.from_spark_field: map child",
        lambda: Field.from_spark_field(sf_map),
        repeat=repeat, inner=20_000,
    ))

    # Field.from_spark — polymorphic entry (DataFrame / StructField /
    # DataType / Column).
    results.append(_time_one(
        "Field.from_spark: pst.LongType",
        lambda: Field.from_spark(sp_long),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "Field.from_spark: pst.StructField",
        lambda: Field.from_spark(sf),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "Field.from_spark: pst.StructType[wide]",
        lambda: Field.from_spark(sp_wide),
        repeat=repeat, inner=10_000,
    ))

    # to_pyspark_* — warm (cached) + cold (drop cache) paths.
    F_FLAT = Field("id", IntegerType(), nullable=False)
    F_STRUCT = Field(
        "row",
        StructType(fields=(
            Field("id", IntegerType(), nullable=False),
            Field("name", StringType()),
            Field("amount", FloatingPointType()),
            Field("ts", TimestampType(unit="us", tz="UTC")),
            Field("active", BooleanType()),
        )),
    )
    F_WIDE = Field(
        "row",
        StructType(fields=tuple(
            Field(f"c{i}", IntegerType() if i % 2 == 0 else StringType())
            for i in range(50)
        )),
    )
    F_ARRAY = Field("tags", ArrayType(item_field=Field("item", StringType())))
    F_MAP = Field(
        "lookup",
        MapType(item_field=Field("entry", StructType(fields=[
            Field("key", StringType()),
            Field("value", IntegerType()),
        ]))),
    )

    results.append(_time_one(
        "Field.to_pyspark_field: flat (warm cache)",
        lambda: F_FLAT.to_pyspark_field(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "Field.to_pyspark_field: struct (warm cache)",
        lambda: F_STRUCT.to_pyspark_field(),
        repeat=repeat, inner=500_000,
    ))

    def _cold_flat():
        object.__setattr__(F_FLAT, "_spark_field", None)
        return F_FLAT.to_pyspark_field()

    def _cold_struct():
        object.__setattr__(F_STRUCT, "_spark_field", None)
        return F_STRUCT.to_pyspark_field()

    def _cold_array():
        object.__setattr__(F_ARRAY, "_spark_field", None)
        return F_ARRAY.to_pyspark_field()

    def _cold_map():
        object.__setattr__(F_MAP, "_spark_field", None)
        return F_MAP.to_pyspark_field()

    results.append(_time_one(
        "Field.to_pyspark_field: flat (cold)",
        _cold_flat,
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_one(
        "Field.to_pyspark_field: struct (cold)",
        _cold_struct,
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "Field.to_pyspark_field: array (cold, type_json dumped)",
        _cold_array,
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "Field.to_pyspark_field: map (cold, type_json dumped)",
        _cold_map,
        repeat=repeat, inner=10_000,
    ))

    # Field.to_spark_schema — folder-of-folders write builds this once
    # per write.
    def _cold_schema():
        object.__setattr__(F_WIDE, "_spark_schema", None)
        return F_WIDE.to_spark_schema()

    results.append(_time_one(
        "Field.to_spark_schema: wide (cold)",
        _cold_schema,
        repeat=repeat, inner=5_000,
    ))
    results.append(_time_one(
        "Field.to_spark_schema: wide (warm cache)",
        lambda: F_WIDE.to_spark_schema(),
        repeat=repeat, inner=500_000,
    ))

    # DataType.to_spark — warm cached engine method.
    INT_DT = IntegerType()
    TS_DT = TimestampType(unit="us", tz="UTC")
    results.append(_time_one(
        "to_spark: IntegerType (warm cache)",
        lambda: INT_DT.to_spark(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "to_spark: TimestampType (warm cache)",
        lambda: TS_DT.to_spark(),
        repeat=repeat, inner=500_000,
    ))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5,
                        help="Number of timing repeats (default: 5)")
    args = parser.parse_args()

    if not HAS_PYSPARK:
        print("# pyspark not installed — skipping bench_spark.py")
        return 0

    results = scenarios(args.repeat)
    print(f"# repeat={args.repeat}")
    print(f"# {'label':<60s}  {'best':>15}  {'median':>15}  {'mean':>15}")
    for r in results:
        print(_fmt(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
