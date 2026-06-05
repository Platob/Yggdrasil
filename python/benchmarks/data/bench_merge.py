"""Benchmark the schema-merge hot paths.

Why this exists
---------------

Schema merging powers:

* ``Field.merge_with`` — the per-column merge that ``CastOptions``
  / ``Schema.merge_with`` walk on a column-by-column basis.
* :class:`StructType.merge_with` — the dtype-level recursion that
  handles struct child reconciliation.
* The set operators (``+``, ``-``, ``&``, ``|`` and in-place
  variants) on :class:`Field` / :class:`Schema`.

These run every time a schema is reconciled across batches
(Spark catalog merge, Delta schema evolution, Tabular union,
config-vs-discovered comparison). Heavy callers walk every field
in every batch, so the per-merge cost adds up fast.

Usage::

    PYTHONPATH=src python benchmarks/bench_merge.py
    PYTHONPATH=src python benchmarks/bench_merge.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

from yggdrasil.data import Field
from yggdrasil.enums import Mode
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

F_INT = Field("id", INT_DT, nullable=False)
F_INT_OTHER = Field("id", INT_DT, nullable=True)  # same name, different nullability
F_STR = Field("name", STR_DT)
F_PRICE = Field("price", FloatingPointType())
F_QTY = Field("qty", "int32")
F_TS = Field("ts", TimestampType(unit="us", tz="UTC"))
F_ACT = Field("active", BooleanType())

SCHEMA_A = Schema.from_fields([F_INT, F_STR, F_PRICE])
SCHEMA_B = Schema.from_fields([F_QTY, F_TS, F_ACT])
SCHEMA_OVERLAP = Schema.from_fields([F_INT_OTHER, F_QTY, F_PRICE])

STRUCT_A = SCHEMA_A.dtype
STRUCT_B = SCHEMA_B.dtype
STRUCT_OVERLAP = SCHEMA_OVERLAP.dtype


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

    # Field.merge_with — single-column merge.
    results.append(_time_one(
        "field: F_INT.merge_with(F_INT) no-op",
        lambda: F_INT.merge_with(F_INT),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "field: F_INT.merge_with(F_INT_OTHER) nullable diff",
        lambda: F_INT.merge_with(F_INT_OTHER),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "field: F_INT.merge_with(F_STR) dtype diff",
        lambda: F_INT.merge_with(F_STR),
        repeat=repeat, inner=50_000,
    ))

    # DataType.merge_with — same shape, recurses into children for struct.
    results.append(_time_one(
        "dtype: INT_DT.merge_with(INT_DT) same dtype",
        lambda: INT_DT.merge_with(INT_DT),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "dtype: STRUCT_A.merge_with(STRUCT_A) same struct",
        lambda: STRUCT_A.merge_with(STRUCT_A),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_one(
        "dtype: STRUCT_A.merge_with(STRUCT_B) disjoint",
        lambda: STRUCT_A.merge_with(STRUCT_B),
        repeat=repeat, inner=5_000,
    ))
    results.append(_time_one(
        "dtype: STRUCT_A.merge_with(STRUCT_OVERLAP) overlap",
        lambda: STRUCT_A.merge_with(STRUCT_OVERLAP),
        repeat=repeat, inner=5_000,
    ))

    # Schema operators — the public + / - / & / | surface.
    results.append(_time_one(
        "schema: SCHEMA_A + SCHEMA_B (disjoint union)",
        lambda: SCHEMA_A + SCHEMA_B,
        repeat=repeat, inner=2_000,
    ))
    results.append(_time_one(
        "schema: SCHEMA_A + SCHEMA_OVERLAP (overlap union)",
        lambda: SCHEMA_A + SCHEMA_OVERLAP,
        repeat=repeat, inner=2_000,
    ))
    results.append(_time_one(
        "schema: SCHEMA_A | SCHEMA_B (or-union)",
        lambda: SCHEMA_A | SCHEMA_B,
        repeat=repeat, inner=2_000,
    ))
    results.append(_time_one(
        "schema: SCHEMA_A & SCHEMA_OVERLAP (intersection)",
        lambda: SCHEMA_A & SCHEMA_OVERLAP,
        repeat=repeat, inner=2_000,
    ))
    results.append(_time_one(
        "schema: SCHEMA_A - SCHEMA_OVERLAP (difference)",
        lambda: SCHEMA_A - SCHEMA_OVERLAP,
        repeat=repeat, inner=2_000,
    ))

    # Mode-driven merges — APPEND / UPSERT / OVERWRITE.
    results.append(_time_one(
        "field: F_INT.merge_with(F_STR, mode=UPSERT)",
        lambda: F_INT.merge_with(F_STR, mode=Mode.UPSERT),
        repeat=repeat, inner=50_000,
    ))
    results.append(_time_one(
        "field: F_INT.merge_with(F_STR, mode=OVERWRITE)",
        lambda: F_INT.merge_with(F_STR, mode=Mode.OVERWRITE),
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
