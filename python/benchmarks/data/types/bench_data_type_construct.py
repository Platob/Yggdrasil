"""Benchmark DataType construction / hash / equality overhead.

Why this exists
---------------

``DataType`` is implemented as a frozen ``@dataclass`` hierarchy
(``DataType`` → ``PrimitiveType`` / ``NestedType`` → leaf classes).
Two costs flow from that choice:

1. **Construction** — frozen dataclass ``__init__`` calls
   ``object.__setattr__`` per field. Primitives are singleton-cached
   in ``DataType.__new__`` so the cost only fires on a miss; nested
   types (``ArrayType`` / ``MapType`` / ``StructType``) allocate
   fresh on every call.
2. **Hash + equality** — the generated dunders walk every dataclass
   field. ``DataType`` instances land in registry dispatch keys,
   ``CastOptions`` MRO lookups, and frame-level ``equals`` checks, so
   the per-instance cost compounds across a column-wise cast.

The point of this bench is to give a single load-bearing number per
shape: "primitive default construction", "primitive with kwargs"
(forces a non-singleton allocation), "list / map / struct
construction from inner fields", and the corresponding ``hash`` and
``==`` calls. Compare against a tiny ``slots=True`` variant of the
same dataclass shape so the dataclass overhead is bounded against a
realistic alternative.

Usage::

    PYTHONPATH=src python benchmarks/data/types/bench_data_type_construct.py
    PYTHONPATH=src python benchmarks/data/types/bench_data_type_construct.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable

from yggdrasil.data import field as make_field
from yggdrasil.data.types import (
    ArrayType,
    Int32Type,
    Int64Type,
    MapType,
    StringType,
    StructType,
)


# ---------------------------------------------------------------------------
# Reference shapes used as the "what does the dataclass cost look like
# without the singleton + engine-cache machinery" baseline.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PlainFrozenInt:
    byte_size: int = 8


@dataclass(frozen=True, slots=True)
class _SlotsFrozenInt:
    byte_size: int = 8


class _ManualInt:
    __slots__ = ("byte_size",)

    def __init__(self, byte_size: int = 8) -> None:
        self.byte_size = byte_size

    def __hash__(self) -> int:
        return hash((type(self), self.byte_size))

    def __eq__(self, other: object) -> bool:
        return type(other) is _ManualInt and other.byte_size == self.byte_size


# ---------------------------------------------------------------------------
# Timing helpers — copied from ``bench_parser.py`` so the report shape
# matches the rest of the suite.
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
    scale, unit = 1e9, "ns"
    if r["best"] >= 1e-6:
        scale, unit = 1e6, "us"
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

    # ------------------------------------------------------------------
    # Construction — primitives.
    #
    # Default-arg ``Int64Type()`` hits the per-class singleton in
    # ``DataType.__new__`` and pays only the cache LOAD_ATTR. Passing
    # a kwarg routes around the singleton and forces the full frozen
    # dataclass ``__init__`` so this is the upper bound on primitive
    # construction cost.
    # ------------------------------------------------------------------

    results.append(_time_one(
        "construct: Int64Type()   [singleton hit]",
        lambda: Int64Type(),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "construct: StringType()  [singleton hit]",
        lambda: StringType(),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "construct: Int64Type(byte_size=8)  [no singleton]",
        lambda: Int64Type(byte_size=8),
        repeat=repeat, inner=100_000,
    ))

    # ------------------------------------------------------------------
    # Construction — nested types.
    #
    # ``ArrayType.from_item`` / ``StructType.from_fields`` /
    # ``MapType.from_key_value`` are the canonical entry points used by
    # the parser, the engine ``from_arrow_type`` paths, and the
    # ``from_dict`` round-trip. They always allocate (no singleton),
    # which makes them the realistic upper bound on dataclass cost
    # under a schema-ingest workload.
    # ------------------------------------------------------------------

    item_field = make_field("item", Int64Type())
    fields = (
        make_field("a", Int64Type()),
        make_field("b", StringType()),
        make_field("c", Int32Type()),
    )
    key_field = make_field("key", StringType())
    value_field = make_field("value", Int64Type())

    results.append(_time_one(
        "construct: ArrayType.from_item(int64)",
        lambda: ArrayType.from_item(item_field),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "construct: StructType.from_fields(3)",
        lambda: StructType.from_fields(fields),
        repeat=repeat, inner=100_000,
    ))
    results.append(_time_one(
        "construct: MapType.from_key_value(str→int64)",
        lambda: MapType.from_key_value(key_field, value_field),
        repeat=repeat, inner=100_000,
    ))

    # ------------------------------------------------------------------
    # Hash + equality on the produced instances.
    #
    # Every cast dispatch hashes the source / target DataType, and the
    # registry MRO walk compares them. A primitive singleton is a
    # ``hash((cls, byte_size))`` walk; a nested instance walks the
    # whole frozen dataclass tuple per call.
    # ------------------------------------------------------------------

    int64 = Int64Type()
    int64_other = Int64Type(byte_size=8)
    string = StringType()
    array = ArrayType.from_item(item_field)
    array_other = ArrayType.from_item(item_field)
    struct = StructType.from_fields(fields)
    struct_other = StructType.from_fields(fields)

    results.append(_time_one(
        "hash:      hash(Int64Type())",
        lambda: hash(int64),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "hash:      hash(StringType())",
        lambda: hash(string),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "hash:      hash(ArrayType(int64))",
        lambda: hash(array),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "hash:      hash(StructType(3 fields))",
        lambda: hash(struct),
        repeat=repeat, inner=100_000,
    ))

    results.append(_time_one(
        "eq:        Int64Type() == Int64Type()  [singleton, is-check]",
        lambda: int64 == int64,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "eq:        Int64Type() == Int64Type(byte_size=8)",
        lambda: int64 == int64_other,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "eq:        ArrayType == ArrayType (same inner)",
        lambda: array == array_other,
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "eq:        StructType == StructType (3 fields)",
        lambda: struct == struct_other,
        repeat=repeat, inner=100_000,
    ))

    # ------------------------------------------------------------------
    # Reference baseline — plain frozen dataclass / slots dataclass /
    # manual ``__slots__`` class with the same single-field shape.
    # Anchors the absolute numbers above against "the cheapest
    # equivalent dataclass" so the next reader can judge whether the
    # DataType overhead is the dataclass tax itself or the extra
    # machinery (singleton, init_subclass, engine-cache, MRO equals).
    # ------------------------------------------------------------------

    results.append(_time_one(
        "ref:       @dataclass(frozen=True)()",
        lambda: _PlainFrozenInt(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "ref:       @dataclass(frozen=True, slots=True)()",
        lambda: _SlotsFrozenInt(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "ref:       plain class with __slots__()",
        lambda: _ManualInt(),
        repeat=repeat, inner=500_000,
    ))

    return results


# ---------------------------------------------------------------------------
# Memory footprint — single one-shot snapshot, not a timed sample.
# ---------------------------------------------------------------------------


def _print_footprint() -> None:
    item_field = make_field("item", Int64Type())
    fields = (
        make_field("a", Int64Type()),
        make_field("b", StringType()),
    )

    samples = [
        ("Int64Type()",          Int64Type()),
        ("StringType()",         StringType()),
        ("ArrayType.from_item",  ArrayType.from_item(item_field)),
        ("StructType.from_fields", StructType.from_fields(fields)),
        ("_PlainFrozenInt()",    _PlainFrozenInt()),
        ("_SlotsFrozenInt()",    _SlotsFrozenInt()),
        ("_ManualInt()",         _ManualInt()),
    ]

    print()
    print("# instance footprint (sys.getsizeof, no children)")
    for label, obj in samples:
        size = sys.getsizeof(obj)
        # __dict__ contribution for non-slots classes — important for
        # the comparison; ``sys.getsizeof(obj)`` doesn't follow the
        # dict pointer on its own.
        dict_size = sys.getsizeof(obj.__dict__) if hasattr(obj, "__dict__") else 0
        total = size + dict_size
        extra = f" + dict={dict_size}" if dict_size else ""
        print(f"  {label:<32s}  {size:>4d} B{extra}  (total={total} B)")


# ---------------------------------------------------------------------------
# CLI
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

    _print_footprint()


if __name__ == "__main__":
    main()
