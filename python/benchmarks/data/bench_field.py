"""Benchmark the :class:`Field` hot path.

Why this exists
---------------

:class:`Field` is the canonical column descriptor — every Schema is
a tree of Fields, every :class:`CastOptions` target is a Field (or
coerces to one in ``__post_init__``), and every cross-engine
projection (Arrow / Polars / Spark) is built by walking Fields.

This benchmark targets the construction, equality, hash, projection,
and from-* coercion paths Fields pay on the cast / IO hot path.
A folder-of-folders persist with N files and M columns runs the
``__init__`` → ``DataType.from_any`` → ``_normalize_metadata`` →
``_adopt_children`` chain N*M times per logical read.

Usage::

    PYTHONPATH=src python benchmarks/bench_field.py
    PYTHONPATH=src python benchmarks/bench_field.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.data import Field, field
from yggdrasil.data.types.primitive import (
    BooleanType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
)
from yggdrasil.data.types.nested import StructType

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:  # pragma: no cover - bench-only optional path
    HAS_POLARS = False


# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------


INT_DT = IntegerType()
STRING_DT = StringType()
TS_DT = TimestampType(unit="us", tz="UTC")

F_FLAT = Field("id", INT_DT, nullable=False)
F_WITH_META = Field("id", INT_DT, nullable=False,
                    metadata={"comment": "primary id"})

STRUCT_DT = StructType(
    fields=(
        Field("id", INT_DT, nullable=False),
        Field("name", STRING_DT),
        Field("amount", FloatingPointType()),
        Field("ts", TS_DT),
        Field("active", BooleanType()),
    )
)
F_STRUCT = Field("row", STRUCT_DT)

PA_FIELD_FLAT = pa.field("id", pa.int64(), nullable=False)
PA_FIELD_STRUCT = pa.field(
    "row",
    pa.struct(
        [
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
            pa.field("amount", pa.float64()),
            pa.field("ts", pa.timestamp("us")),
            pa.field("active", pa.bool_()),
        ]
    ),
)
PA_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("amount", pa.float64()),
        pa.field("qty", pa.int32()),
        pa.field("name", pa.string()),
        pa.field("ts", pa.timestamp("us")),
        pa.field("active", pa.bool_()),
    ]
)

F_LARGE_STRUCT = Field(
    "row",
    StructType(
        fields=tuple(
            Field(f"c{i}", INT_DT if i % 2 == 0 else STRING_DT)
            for i in range(20)
        )
    ),
)


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

    # Construction — Field(...) direct vs. ``field(...)`` helper vs.
    # passing pa.DataType / str hint.
    results.append(_time_one(
        "construct: Field('id', INT_DT, nullable=False)",
        lambda: Field("id", INT_DT, nullable=False),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "construct: field('id', 'int64', nullable=False)",
        lambda: field("id", "int64", nullable=False),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_one(
        "construct: Field('id', pa.int64())",
        lambda: Field("id", pa.int64()),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_one(
        "construct: Field('row', STRUCT_DT) -> Schema redirect",
        lambda: Field("row", STRUCT_DT),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "construct: Field with metadata={'comment': ...}",
        lambda: Field("id", INT_DT, nullable=False,
                      metadata={"comment": "x"}),
        repeat=repeat, inner=20_000,
    ))

    # Identity / equality / hash.
    results.append(_time_one(
        "eq: F_FLAT == F_FLAT (identity)",
        lambda: F_FLAT == F_FLAT,
        repeat=repeat, inner=500_000,
    ))
    other_flat = Field("id", INT_DT, nullable=False)
    results.append(_time_one(
        "eq: F_FLAT == other_flat (equal but distinct)",
        lambda: F_FLAT == other_flat,
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "eq: F_STRUCT == F_STRUCT (struct identity)",
        lambda: F_STRUCT == F_STRUCT,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "hash: hash(F_FLAT)",
        lambda: hash(F_FLAT),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "hash: hash(F_WITH_META)",
        lambda: hash(F_WITH_META),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "hash: hash(F_STRUCT)",
        lambda: hash(F_STRUCT),
        repeat=repeat, inner=50_000,
    ))

    # Properties — type_id, arrow_type, children, fields, default_value.
    results.append(_time_one(
        "prop: F_FLAT.type_id",
        lambda: F_FLAT.type_id,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "prop: F_FLAT.arrow_type",
        lambda: F_FLAT.arrow_type,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "prop: F_STRUCT.children",
        lambda: F_STRUCT.children,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "prop: F_STRUCT.fields (filter constraint_key)",
        lambda: F_STRUCT.fields,
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "prop: F_FLAT.has_default",
        lambda: F_FLAT.has_default,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "prop: F_FLAT.primary_key (tag flag, no metadata)",
        lambda: F_FLAT.primary_key,
        repeat=repeat, inner=500_000,
    ))

    # Lookup — field_by name / index on a struct.
    results.append(_time_one(
        "lookup: F_LARGE_STRUCT.field_by('c10')",
        lambda: F_LARGE_STRUCT.field_by("c10"),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "lookup: F_LARGE_STRUCT.field_by(5)",
        lambda: F_LARGE_STRUCT.field_by(5),
        repeat=repeat, inner=100_000,
    ))

    # Engine projections — warm-cache hits (production cost after first call).
    F_FLAT.to_arrow_field()  # warm
    F_STRUCT.to_arrow_field()
    F_STRUCT.to_arrow_schema()
    results.append(_time_one(
        "to_arrow_field: F_FLAT (warm cache)",
        lambda: F_FLAT.to_arrow_field(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "to_arrow_field: F_STRUCT (warm cache)",
        lambda: F_STRUCT.to_arrow_field(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "to_arrow_schema: F_STRUCT (warm cache)",
        lambda: F_STRUCT.to_arrow_schema(),
        repeat=repeat, inner=500_000,
    ))

    # Cold projection — common in pipelines that build fields per-batch.
    results.append(_time_one(
        "to_arrow_field: fresh Field (cold)",
        lambda: Field("id", INT_DT, nullable=False).to_arrow_field(),
        repeat=repeat, inner=10_000,
    ))

    # from_* — Arrow / pa.Field / pa.Schema → Field.
    results.append(_time_one(
        "from_arrow_field: pa.Field (flat)",
        lambda: Field.from_arrow_field(PA_FIELD_FLAT),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_one(
        "from_arrow_field: pa.Field (struct)",
        lambda: Field.from_arrow_field(PA_FIELD_STRUCT),
        repeat=repeat, inner=2_000,
    ))
    results.append(_time_one(
        "from_arrow_schema: pa.Schema",
        lambda: Field.from_arrow_schema(PA_SCHEMA),
        repeat=repeat, inner=2_000,
    ))
    results.append(_time_one(
        "from_any: pa.Field (flat)",
        lambda: Field.from_any(PA_FIELD_FLAT),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "from_any: pa.Schema",
        lambda: Field.from_any(PA_SCHEMA),
        repeat=repeat, inner=2_000,
    ))
    results.append(_time_one(
        "from_any: existing Field (identity)",
        lambda: Field.from_any(F_FLAT),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "from_pytype: int",
        lambda: Field.from_pytype(int, name="x"),
        repeat=repeat, inner=10_000,
    ))

    # Mutators — with_*, copy.
    results.append(_time_one(
        "mutate: F_FLAT.with_name('id') no-op",
        lambda: F_FLAT.with_name("id"),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "mutate: F_FLAT.with_nullable(False) no-op",
        lambda: F_FLAT.with_nullable(False, inplace=False),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "mutate: F_FLAT.copy()",
        lambda: F_FLAT.copy(),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "mutate: F_STRUCT.copy() (deep-copy children)",
        lambda: F_STRUCT.copy(),
        repeat=repeat, inner=2_000,
    ))

    # Polars projections — optional, skipped without polars installed.
    # Mirrors the Arrow scenarios above so the construction / from-* /
    # to-* / round-trip costs stay visible per engine.
    if HAS_POLARS:
        # Warm caches once before the timing loop — to_polars_field /
        # to_polars_schema cache the polars output on the Field, so the
        # production cost after the first call is what we measure.
        F_FLAT.to_polars_field()
        F_STRUCT.to_polars_field()
        F_STRUCT.to_polars_schema()
        results.append(_time_one(
            "to_polars_field: F_FLAT (warm cache)",
            lambda: F_FLAT.to_polars_field(),
            repeat=repeat, inner=500_000,
        ))
        results.append(_time_one(
            "to_polars_field: F_STRUCT (warm cache)",
            lambda: F_STRUCT.to_polars_field(),
            repeat=repeat, inner=500_000,
        ))
        results.append(_time_one(
            "to_polars_schema: F_STRUCT (warm cache)",
            lambda: F_STRUCT.to_polars_schema(),
            repeat=repeat, inner=500_000,
        ))
        # Cold projection — common in pipelines that build fields
        # per-batch (e.g. polars LazyFrame schema collection).
        results.append(_time_one(
            "to_polars_field: fresh Field (cold)",
            lambda: Field(
                "id", INT_DT, nullable=False
            ).to_polars_field(),
            repeat=repeat, inner=10_000,
        ))
        results.append(_time_one(
            "to_polars_schema: fresh struct Field (cold)",
            lambda: Field("row", STRUCT_DT).to_polars_schema(),
            repeat=repeat, inner=2_000,
        ))

        # from_polars_* — Polars Field / Schema / Series → Field.
        PL_SCHEMA = F_STRUCT.to_polars_schema()
        PL_FIELD_FLAT = pl.Field("id", pl.Int64())
        PL_SERIES_INT = pl.Series("id", [1, 2, 3], dtype=pl.Int64)
        PL_SERIES_STR = pl.Series("name", ["a", "b", None], dtype=pl.Utf8)

        results.append(_time_one(
            "from_polars_field: pl.Field (flat)",
            lambda: Field.from_polars_field(PL_FIELD_FLAT),
            repeat=repeat, inner=10_000,
        ))
        results.append(_time_one(
            "from_polars_schema: pl.Schema (struct)",
            lambda: Field.from_polars_schema(PL_SCHEMA),
            repeat=repeat, inner=2_000,
        ))
        results.append(_time_one(
            "from_polars_series: pl.Series (int)",
            lambda: Field.from_polars_series(PL_SERIES_INT),
            repeat=repeat, inner=20_000,
        ))
        results.append(_time_one(
            "from_polars_series: pl.Series (str w/ nulls)",
            lambda: Field.from_polars_series(PL_SERIES_STR),
            repeat=repeat, inner=20_000,
        ))
        results.append(_time_one(
            "from_polars: pl.Schema (dispatched)",
            lambda: Field.from_polars(PL_SCHEMA),
            repeat=repeat, inner=2_000,
        ))
        results.append(_time_one(
            "from_polars: pl.Series (dispatched)",
            lambda: Field.from_polars(PL_SERIES_INT),
            repeat=repeat, inner=10_000,
        ))
        results.append(_time_one(
            "from_any: pl.Schema (cross-engine dispatch)",
            lambda: Field.from_any(PL_SCHEMA),
            repeat=repeat, inner=2_000,
        ))

        # as_polars — same-shape no-op when dtype is already
        # polars-compatible (the common case for primitive fields).
        results.append(_time_one(
            "as_polars: F_FLAT (already polars-compatible)",
            lambda: F_FLAT.as_polars(),
            repeat=repeat, inner=200_000,
        ))

    # Dict / repr / pretty_format.
    results.append(_time_one(
        "to_dict: F_FLAT",
        lambda: F_FLAT.to_dict(),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "to_dict: F_STRUCT",
        lambda: F_STRUCT.to_dict(),
        repeat=repeat, inner=5_000,
    ))
    results.append(_time_one(
        "repr: repr(F_FLAT)",
        lambda: repr(F_FLAT),
        repeat=repeat, inner=20_000,
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
