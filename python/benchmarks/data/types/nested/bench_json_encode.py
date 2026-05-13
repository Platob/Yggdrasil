"""Benchmark JSON encoding: nested Arrow arrays → string/binary.

Covers ``cast_arrow_json_encode_array`` (the Arrow encode path used by
SJsonType / BJsonType when writing nested → JSON string columns).

Hot path uses ``orjson.dumps(...).decode()`` for common types (int / float /
str / bool / list / dict) and falls back to stdlib ``json.dumps`` + a
``_json_default`` hook for edge-case types (Decimal, raw bytes, timedelta).

Three source shapes per scenario:

* **struct<id: int64, name: str, amount: float64, active: bool>** — the most
  common warehouse shape; all fields are orjson-native.
* **list<int64>** — flat numeric list; orjson-native.
* **map<str, int64>** — dict-shaped map; orjson-native.
* **struct with Decimal** — triggers the orjson → stdlib fallback so the
  fallback path has a representative cost to compare.

Row counts: 1 000 / 10 000 / 100 000.

Usage::

    PYTHONPATH=src python benchmarks/data/types/nested/bench_json_encode.py
    PYTHONPATH=src python benchmarks/data/types/nested/bench_json_encode.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from decimal import Decimal
from typing import Callable

import pyarrow as pa

from yggdrasil.data import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types import (
    ArrayType,
    BJsonType,
    DecimalType,
    FloatingPointType,
    IntegerType,
    MapType,
    SJsonType,
    StringType,
    StructType,
)
from yggdrasil.data.types.primitive import BooleanType

# ---------------------------------------------------------------------------
# Target field definitions.
# ---------------------------------------------------------------------------

_INT64 = IntegerType(byte_size=8, signed=True)
_F64 = FloatingPointType(byte_size=8)
_STR = StringType()
_BOOL = BooleanType()

_STRUCT_FIELD = Field(
    "row",
    StructType(fields=(
        Field("id", _INT64, nullable=False),
        Field("name", _STR),
        Field("amount", _F64),
        Field("active", _BOOL),
    )),
)
_LIST_FIELD = Field(
    "arr",
    ArrayType.from_item(Field("item", _INT64)),
)
_MAP_FIELD = Field(
    "m",
    MapType.from_key_value(
        key_field=Field("k", _STR, nullable=False),
        value_field=Field("v", _INT64),
    ),
)
_STRUCT_DECIMAL_FIELD = Field(
    "row",
    StructType(fields=(
        Field("id", _INT64, nullable=False),
        Field("amount", DecimalType(precision=18, scale=2)),
    )),
)

# ---------------------------------------------------------------------------
# Source array builders.
# ---------------------------------------------------------------------------


def _build_struct(n: int) -> pa.Array:
    return pa.array(
        [{"id": i, "name": f"x{i}", "amount": float(i) * 0.5, "active": i % 2 == 0}
         for i in range(n)],
        type=pa.struct([
            ("id", pa.int64()), ("name", pa.string()),
            ("amount", pa.float64()), ("active", pa.bool_()),
        ]),
    )


def _build_list(n: int) -> pa.Array:
    return pa.array([[i, i + 1] for i in range(n)], type=pa.list_(pa.int64()))


def _build_map(n: int) -> pa.Array:
    return pa.array([[("k", i)] for i in range(n)],
                    type=pa.map_(pa.string(), pa.int64()))


def _build_struct_decimal(n: int) -> pa.Array:
    return pa.array(
        [{"id": i, "amount": Decimal(str(i) + ".50")} for i in range(n)],
        type=pa.struct([
            ("id", pa.int64()),
            ("amount", pa.decimal128(18, 2)),
        ]),
    )


# ---------------------------------------------------------------------------
# Timing helpers.
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 50)):
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
    v = r["best"]
    if v < 1e-6:
        scale, unit = 1e9, "ns"
    elif v < 1e-3:
        scale, unit = 1e6, "us"
    else:
        scale, unit = 1e3, "ms"
    return (
        f"{r['label']:<72s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios.
# ---------------------------------------------------------------------------


def _encode_block(
    label: str,
    src: pa.Array,
    target_field: Field,
    sjson: bool,
    *,
    repeat: int,
    inner: int,
) -> dict:
    json_dtype = SJsonType() if sjson else BJsonType()
    opts = CastOptions(target=Field(target_field.name, json_dtype))

    return _time_one(
        label,
        lambda: json_dtype.cast_arrow_array(
            src, source=target_field, target=Field(target_field.name, json_dtype)
        ),
        repeat=repeat, inner=inner,
    )


def scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    for n in (1_000, 10_000, 100_000):
        inner = max(1, 500 // (n // 1_000))

        struct_arr = _build_struct(n)
        list_arr = _build_list(n)
        map_arr = _build_map(n)
        decimal_arr = _build_struct_decimal(n)

        out.append(_encode_block(
            f"struct{{id,name,amount,active}} → sjson  rows={n:>7,}",
            struct_arr, _STRUCT_FIELD, sjson=True,
            repeat=repeat, inner=inner,
        ))
        out.append(_encode_block(
            f"list<int64>                    → bjson  rows={n:>7,}",
            list_arr, _LIST_FIELD, sjson=False,
            repeat=repeat, inner=inner,
        ))
        out.append(_encode_block(
            f"map<str,int64>                 → sjson  rows={n:>7,}",
            map_arr, _MAP_FIELD, sjson=True,
            repeat=repeat, inner=inner,
        ))
        out.append(_encode_block(
            f"struct{{id,decimal}} (fallback) → sjson  rows={n:>7,}",
            decimal_arr, _STRUCT_DECIMAL_FIELD, sjson=True,
            repeat=repeat, inner=inner,
        ))
        out.append({"label": "", "best": 0.0, "median": 0.0, "mean": 0.0})

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<72s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        if not row["label"]:
            print()
        else:
            print(_fmt(row))


if __name__ == "__main__":
    main()
