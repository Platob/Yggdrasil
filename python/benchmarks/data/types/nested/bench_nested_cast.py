"""Benchmark nested :class:`DataType` cast kernels — arrow / polars / pandas.

Mirrors the source tree: lives next to
``yggdrasil/data/types/nested``. Covers ``ArrayType``, ``MapType``,
and ``StructType`` casts plus the deeply-nested combinations that
real warehouse / JSON-ingest shapes carry:

* ``list<int>``, ``map<str, int>``, flat ``struct``
* ``list<struct>``, ``map<str, list<int>>``
* deep ``struct{address, tags, attrs, decimal}``

Three shapes per type:

* **MATCH** — source matches target; engine-type bypass fires.
* **CAST** — source diverges by one inner type (e.g. inner ``int32`` →
  ``int64``); the per-element cast kernel fires.

Spark is intentionally omitted — the per-call SparkSession overhead
dominates the kernel measurement. The tabular bench covers Spark
through its frame layer.

Usage::

    PYTHONPATH=src python benchmarks/data/types/nested/bench_nested_cast.py
    PYTHONPATH=src python benchmarks/data/types/nested/bench_nested_cast.py --repeat 7
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
from yggdrasil.data.types.nested import ArrayType, MapType, StructType
from yggdrasil.data.types.primitive import (
    BooleanType,
    DecimalType,
    FloatingPointType,
    IntegerType,
    StringType,
)


ROWS = 10_000


# ---------------------------------------------------------------------------
# Field targets.
# ---------------------------------------------------------------------------


F_LIST_INT = Field(
    "arr",
    ArrayType.from_item(Field("item", IntegerType(byte_size=8, signed=True))),
)
F_MAP_STR_INT = Field(
    "m",
    MapType.from_key_value(
        key_field=Field("k", StringType(), nullable=False),
        value_field=Field("v", IntegerType(byte_size=8, signed=True)),
    ),
)
F_STRUCT = Field(
    "row",
    StructType(fields=(
        Field("id", IntegerType(byte_size=8, signed=True), nullable=False),
        Field("name", StringType()),
        Field("amount", FloatingPointType(byte_size=8)),
        Field("active", BooleanType()),
    )),
)
F_LIST_OF_STRUCT = Field(
    "rows",
    ArrayType.from_item(Field(
        "item",
        StructType(fields=(
            Field("k", StringType()),
            Field("v", IntegerType(byte_size=8, signed=True)),
        )),
    )),
)
F_MAP_STR_LIST = Field(
    "buckets",
    MapType.from_key_value(
        key_field=Field("k", StringType(), nullable=False),
        value_field=Field(
            "v",
            ArrayType.from_item(Field("item", IntegerType(byte_size=8, signed=True))),
        ),
    ),
)
F_DEEP_STRUCT = Field(
    "row",
    StructType(fields=(
        Field("id", IntegerType(byte_size=8, signed=True), nullable=False),
        Field("name", StringType()),
        Field("amount", DecimalType(precision=18, scale=2)),
        Field("address", StructType(fields=(
            Field("street", StringType()),
            Field("city", StringType()),
            Field("zip", StringType()),
        ))),
        Field("tags", ArrayType.from_item(Field("item", StringType()))),
        Field("attrs", MapType.from_key_value(
            key_field=Field("k", StringType(), nullable=False),
            value_field=Field("v", StringType()),
        )),
    )),
)


# ---------------------------------------------------------------------------
# Source arrays — MATCH (same shape as target) + CAST (one inner type widened).
# ---------------------------------------------------------------------------


def _build_sources() -> dict[str, pa.Array]:
    list_match = pa.array([[i] for i in range(ROWS)], type=pa.list_(pa.int64()))
    list_widen = pa.array([[i] for i in range(ROWS)], type=pa.list_(pa.int32()))

    map_match = pa.array([[("k", i)] for i in range(ROWS)],
                         type=pa.map_(pa.string(), pa.int64()))
    map_widen = pa.array([[("k", i)] for i in range(ROWS)],
                         type=pa.map_(pa.string(), pa.int32()))

    struct_match_type = pa.struct([
        ("id", pa.int64()), ("name", pa.string()),
        ("amount", pa.float64()), ("active", pa.bool_()),
    ])
    struct_widen_type = pa.struct([
        ("id", pa.int32()), ("name", pa.string()),
        ("amount", pa.float32()), ("active", pa.bool_()),
    ])
    struct_payload = [
        {"id": i, "name": "x", "amount": 1.5, "active": True}
        for i in range(ROWS)
    ]
    struct_match = pa.array(struct_payload, type=struct_match_type)
    struct_widen = pa.array(struct_payload, type=struct_widen_type)

    list_of_struct_type = pa.list_(pa.struct([("k", pa.string()), ("v", pa.int64())]))
    list_of_struct_widen_type = pa.list_(pa.struct([("k", pa.string()), ("v", pa.int32())]))
    list_of_struct_payload = [[{"k": "x", "v": i}] for i in range(ROWS)]
    list_of_struct_match = pa.array(list_of_struct_payload, type=list_of_struct_type)
    list_of_struct_widen = pa.array(list_of_struct_payload, type=list_of_struct_widen_type)

    map_str_list_type = pa.map_(pa.string(), pa.list_(pa.int64()))
    map_str_list_widen_type = pa.map_(pa.string(), pa.list_(pa.int32()))
    map_str_list_payload = [[("k", [i, i + 1])] for i in range(ROWS)]
    map_str_list_match = pa.array(map_str_list_payload, type=map_str_list_type)
    map_str_list_widen = pa.array(map_str_list_payload, type=map_str_list_widen_type)

    deep_struct_type = pa.struct([
        ("id", pa.int64()), ("name", pa.string()),
        ("amount", pa.decimal128(18, 2)),
        ("address", pa.struct([
            ("street", pa.string()), ("city", pa.string()), ("zip", pa.string()),
        ])),
        ("tags", pa.list_(pa.string())),
        ("attrs", pa.map_(pa.string(), pa.string())),
    ])
    deep_struct_widen_type = pa.struct([
        ("id", pa.int32()), ("name", pa.string()),
        ("amount", pa.decimal128(18, 2)),
        ("address", pa.struct([
            ("street", pa.string()), ("city", pa.string()), ("zip", pa.string()),
        ])),
        ("tags", pa.list_(pa.string())),
        ("attrs", pa.map_(pa.string(), pa.string())),
    ])
    deep_struct_payload = [
        {"id": i, "name": "x", "amount": Decimal("1.50"),
         "address": {"street": "1", "city": "x", "zip": "00"},
         "tags": ["a", "b"], "attrs": [("k", "v")]}
        for i in range(ROWS)
    ]
    deep_struct_match = pa.array(deep_struct_payload, type=deep_struct_type)
    deep_struct_widen = pa.array(deep_struct_payload, type=deep_struct_widen_type)

    return {
        "list_match": list_match, "list_widen": list_widen,
        "map_match": map_match, "map_widen": map_widen,
        "struct_match": struct_match, "struct_widen": struct_widen,
        "list_of_struct_match": list_of_struct_match,
        "list_of_struct_widen": list_of_struct_widen,
        "map_str_list_match": map_str_list_match,
        "map_str_list_widen": map_str_list_widen,
        "deep_struct_match": deep_struct_match,
        "deep_struct_widen": deep_struct_widen,
    }


# ---------------------------------------------------------------------------
# Timing helpers.
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 100)):
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
    elif r["best"] >= 1e-3:
        scale, unit = 1e3, "ms"
    return (
        f"{r['label']:<70s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Engine adapters.
# ---------------------------------------------------------------------------


def _to_polars_series(arr: pa.Array):
    try:
        import polars as pl
    except ImportError:
        return None
    try:
        return pl.from_arrow(arr)
    except Exception:
        return None


def _to_pandas_series(arr: pa.Array):
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return None
    try:
        return arr.to_pandas()
    except Exception:
        return None


def _block(label_prefix: str, target: Field, arr_match: pa.Array, arr_cast: pa.Array,
           *, repeat: int, inner: int) -> list[dict]:
    opts = CastOptions(target=target)
    dtype = target.dtype
    out: list[dict] = []

    out.append(_time_one(
        f"arrow: {label_prefix} MATCH rows={ROWS}",
        lambda: dtype.cast_arrow_array(arr_match, opts),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"arrow: {label_prefix} CAST rows={ROWS}",
        lambda: dtype.cast_arrow_array(arr_cast, opts),
        repeat=repeat, inner=inner,
    ))

    pl_match = _to_polars_series(arr_match)
    pl_cast = _to_polars_series(arr_cast)
    if pl_match is not None and pl_cast is not None:
        try:
            out.append(_time_one(
                f"polars: {label_prefix} MATCH rows={ROWS}",
                lambda: dtype.cast_polars_series(pl_match, opts),
                repeat=repeat, inner=inner,
            ))
            out.append(_time_one(
                f"polars: {label_prefix} CAST rows={ROWS}",
                lambda: dtype.cast_polars_series(pl_cast, opts),
                repeat=repeat, inner=inner,
            ))
        except Exception as e:
            out.append({"label": f"polars: {label_prefix} SKIPPED ({type(e).__name__})",
                        "best": 0.0, "median": 0.0, "mean": 0.0})

    pd_match = _to_pandas_series(arr_match)
    pd_cast = _to_pandas_series(arr_cast)
    if pd_match is not None and pd_cast is not None:
        try:
            out.append(_time_one(
                f"pandas: {label_prefix} MATCH rows={ROWS}",
                lambda: dtype.cast_pandas_series(pd_match, opts),
                repeat=repeat, inner=inner,
            ))
            out.append(_time_one(
                f"pandas: {label_prefix} CAST rows={ROWS}",
                lambda: dtype.cast_pandas_series(pd_cast, opts),
                repeat=repeat, inner=inner,
            ))
        except Exception as e:
            out.append({"label": f"pandas: {label_prefix} SKIPPED ({type(e).__name__})",
                        "best": 0.0, "median": 0.0, "mean": 0.0})

    return out


def scenarios(repeat: int) -> list[dict]:
    s = _build_sources()
    out: list[dict] = []
    out.extend(_block("list<int64> (match / cast list<int32>→)",
                      F_LIST_INT, s["list_match"], s["list_widen"],
                      repeat=repeat, inner=200))
    out.extend(_block("map<str,int64> (match / cast map<str,int32>→)",
                      F_MAP_STR_INT, s["map_match"], s["map_widen"],
                      repeat=repeat, inner=200))
    out.extend(_block("struct (match / cast int32+f32→)",
                      F_STRUCT, s["struct_match"], s["struct_widen"],
                      repeat=repeat, inner=200))
    out.extend(_block("list<struct> (match / cast inner int32→)",
                      F_LIST_OF_STRUCT,
                      s["list_of_struct_match"], s["list_of_struct_widen"],
                      repeat=repeat, inner=100))
    out.extend(_block("map<str,list<int>> (match / cast inner int32→)",
                      F_MAP_STR_LIST,
                      s["map_str_list_match"], s["map_str_list_widen"],
                      repeat=repeat, inner=100))
    out.extend(_block("deep struct{addr,tags,attrs,decimal} (match / cast id int32→)",
                      F_DEEP_STRUCT,
                      s["deep_struct_match"], s["deep_struct_widen"],
                      repeat=repeat, inner=100))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}  rows={ROWS}")
    print(f"# {'label':<70s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
