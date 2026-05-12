"""Benchmark the ``__eq__`` / ``equals`` / ``need_cast`` fast paths.

Why this exists
---------------

Type-equality checks are everywhere on the cast hot path:

* ``CastOptions.need_cast`` runs a full ``Field.equals`` per cast site
  to decide whether the engine kernel can be skipped.
* The :class:`Field` ``__eq__`` is what backs dict / set membership for
  schema diffs, projection pruning, and cache keys.
* :class:`DataType` ``__eq__`` powers the per-column type-equal short
  circuit inside Arrow / Polars cast kernels.
* :class:`Schema` ``equals`` recurses into every child Field.

A folder-of-folders persist with N batches × M columns runs hundreds
of equality / hash / need_cast calls per batch before any real data
moves. This benchmark times those exact shapes so type-check
regressions are visible to the per-batch overhead.

Usage::

    PYTHONPATH=src python benchmarks/bench_equals.py
    PYTHONPATH=src python benchmarks/bench_equals.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.data import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.primitive import (
    BooleanType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
)
from yggdrasil.data.types.nested import StructType


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


INT_DT = IntegerType()
STR_DT = StringType()
TS_DT = TimestampType(unit="us", tz="UTC")

F_INT = Field("id", INT_DT, nullable=False)
F_INT_OTHER = Field("id", INT_DT, nullable=False)            # equal, distinct
F_INT_DIFF_NAME = Field("price", INT_DT, nullable=False)     # not equal
F_INT_NULL = Field("id", INT_DT, nullable=True)              # not equal
F_STR = Field("name", STR_DT)

F_STRUCT = Field(
    "row",
    StructType(
        fields=(
            Field("id", INT_DT, nullable=False),
            Field("name", STR_DT),
            Field("amount", FloatingPointType()),
            Field("ts", TS_DT),
            Field("active", BooleanType()),
        )
    ),
)
F_STRUCT_OTHER = Field(
    "row",
    StructType(
        fields=(
            Field("id", INT_DT, nullable=False),
            Field("name", STR_DT),
            Field("amount", FloatingPointType()),
            Field("ts", TS_DT),
            Field("active", BooleanType()),
        )
    ),
)

SCHEMA = Schema.from_fields(
    [
        Field("id", "int64", nullable=False),
        Field("amount", "float64"),
        Field("qty", "int32"),
        Field("name", "string"),
        Field("ts", "timestamp(us)"),
        Field("active", "bool"),
    ]
)
SCHEMA_OTHER = Schema.from_fields(
    [
        Field("id", "int64", nullable=False),
        Field("amount", "float64"),
        Field("qty", "int32"),
        Field("name", "string"),
        Field("ts", "timestamp(us)"),
        Field("active", "bool"),
    ]
)


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
        scale = 1e9
        unit = "ns"
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

    # DataType-level equality.
    INT_OTHER = IntegerType()
    INT_NE = IntegerType(byte_size=4, signed=True)
    STRUCT_DT = F_STRUCT.dtype
    STRUCT_DT_OTHER = F_STRUCT_OTHER.dtype

    results.append(_time_one(
        "dtype: INT_DT == INT_DT (identity)",
        lambda: INT_DT == INT_DT,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "dtype: INT_DT == INT_OTHER (equal)",
        lambda: INT_DT == INT_OTHER,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "dtype: INT_DT == INT_NE (different width)",
        lambda: INT_DT == INT_NE,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "dtype: STRUCT_DT == STRUCT_DT (identity)",
        lambda: STRUCT_DT == STRUCT_DT,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "dtype: STRUCT_DT == STRUCT_DT_OTHER (equal)",
        lambda: STRUCT_DT == STRUCT_DT_OTHER,
        repeat=repeat, inner=200_000,
    ))

    # Field-level equality.
    results.append(_time_one(
        "field: F_INT == F_INT (identity)",
        lambda: F_INT == F_INT,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "field: F_INT == F_INT_OTHER (equal, distinct)",
        lambda: F_INT == F_INT_OTHER,
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "field: F_INT == F_INT_DIFF_NAME (name differs)",
        lambda: F_INT == F_INT_DIFF_NAME,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "field: F_INT == F_INT_NULL (nullable differs)",
        lambda: F_INT == F_INT_NULL,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "field: F_INT == F_STR (different dtype)",
        lambda: F_INT == F_STR,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "field: F_STRUCT == F_STRUCT (identity)",
        lambda: F_STRUCT == F_STRUCT,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "field: F_STRUCT == F_STRUCT_OTHER (equal, distinct)",
        lambda: F_STRUCT == F_STRUCT_OTHER,
        repeat=repeat, inner=50_000,
    ))

    # Field.equals — the schema-aware path used by need_cast.
    results.append(_time_one(
        "field: F_INT.equals(F_INT_OTHER)",
        lambda: F_INT.equals(F_INT_OTHER),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "field: F_STRUCT.equals(F_STRUCT_OTHER)",
        lambda: F_STRUCT.equals(F_STRUCT_OTHER),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_one(
        "field: SCHEMA.equals(SCHEMA_OTHER)",
        lambda: SCHEMA.equals(SCHEMA_OTHER),
        repeat=repeat, inner=20_000,
    ))

    # need_cast — the cast hot path entry.
    opts_eq = CastOptions(target=F_INT, source=F_INT_OTHER)
    opts_ne = CastOptions(target=F_INT, source=F_STR)
    opts_struct_eq = CastOptions(target=F_STRUCT, source=F_STRUCT_OTHER)
    opts_schema_eq = CastOptions(target=SCHEMA, source=SCHEMA_OTHER)
    results.append(_time_one(
        "opts: need_cast() equal field (no cast)",
        lambda: opts_eq.need_cast(),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "opts: need_cast() different field (cast needed)",
        lambda: opts_ne.need_cast(),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "opts: need_cast() equal struct (no cast)",
        lambda: opts_struct_eq.need_cast(),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_one(
        "opts: need_cast() equal schema (no cast)",
        lambda: opts_schema_eq.need_cast(),
        repeat=repeat, inner=20_000,
    ))

    # Engine-type equality — what _cast_arrow_array bypasses on.
    pa_int = pa.int64()
    pa_int_other = pa.int64()
    pa_struct = F_STRUCT.to_arrow_type()
    pa_struct_other = F_STRUCT_OTHER.to_arrow_type()
    results.append(_time_one(
        "arrow: pa.int64() == pa.int64() (equal)",
        lambda: pa_int == pa_int_other,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "arrow: pa.struct(...) == pa.struct(...) (equal)",
        lambda: pa_struct == pa_struct_other,
        repeat=repeat, inner=200_000,
    ))

    # Dict / set membership — the path schema-diff code pays.
    field_set = {F_INT, F_STR, F_STRUCT}
    results.append(_time_one(
        "set: F_INT in {F_INT, F_STR, F_STRUCT}",
        lambda: F_INT in field_set,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "set: F_INT_OTHER in {F_INT, F_STR, F_STRUCT}",
        lambda: F_INT_OTHER in field_set,
        repeat=repeat, inner=500_000,
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
