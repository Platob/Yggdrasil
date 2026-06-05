"""Benchmark per-column + tabular cast paths across Arrow / Polars / pandas.

Why this exists
---------------

Covers both shapes the cast pipeline hits in production:

* Per-column kernels (``DataType.cast_arrow_array`` for
  :class:`pa.Array` / :class:`pa.ChunkedArray`,
  ``DataType.cast_polars_series`` for :class:`pl.Series`,
  ``DataType.cast_pandas_series`` for :class:`pd.Series`).
* Whole-table (``Schema.cast_arrow_tabular`` /
  ``Schema.cast_polars_tabular`` / ``Schema.cast_pandas_tabular``) with
  representative mixes of convenient types (int / float / string /
  timestamp) and nested types (struct / list / map).

For each engine + column type we measure three shapes:

* **MATCH**  — source dtype identical to target; the engine-type
  bypass should fire and the call returns the input unchanged.
* **WIDEN** — source narrower than target (e.g. int32 → int64);
  the kernel actually casts.
* **NARROW** — source wider than target (e.g. float64 → float32);
  exercises the lossy-cast / safe= path.

Per-row throughput across these three shapes is the right signal for
"is the bypass kicking in" and "did the cast kernel regress".

Usage::

    PYTHONPATH=src python benchmarks/bench_cast_data.py
    PYTHONPATH=src python benchmarks/bench_cast_data.py --rows 100000 --repeat 5
    PYTHONPATH=src python benchmarks/bench_cast_data.py --engines arrow,polars
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
from yggdrasil.data.types.nested import ArrayType, MapType, StructType
from yggdrasil.data.types.primitive import (
    BooleanType,
    DateType,
    DecimalType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
)


# ---------------------------------------------------------------------------
# Timing helpers
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
    scale = 1e6
    unit = "us"
    if r["best"] < 1e-6:
        scale = 1e9
        unit = "ns"
    return (
        f"{r['label']:<62s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Arrow scenarios
# ---------------------------------------------------------------------------


def _arrow_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []

    int64_target = Field("id", IntegerType(byte_size=8, signed=True), nullable=False)
    int32_target = Field("id", IntegerType(byte_size=4, signed=True), nullable=False)
    float32_target = Field("v", FloatingPointType(byte_size=4))
    string_target = Field("s", StringType())
    ts_target = Field(
        "ts",
        TimestampType(unit="us", tz="UTC"),
    )

    int_target = int64_target.dtype
    int_match = pa.array(range(rows), type=pa.int64())
    int_widen = pa.array(range(rows), type=pa.int32())
    int_narrow = pa.array(range(rows), type=pa.int64())  # narrow target = int32
    chunked = pa.chunked_array(
        [pa.array(range(rows // 2), type=pa.int64()),
         pa.array(range(rows // 2, rows), type=pa.int64())]
    )

    float_target = float32_target.dtype
    float_match = pa.array([1.5] * rows, type=pa.float32())
    float_widen = pa.array([1.5] * rows, type=pa.float32())  # widen target = float64
    float_narrow = pa.array([1.5] * rows, type=pa.float64())

    str_match = pa.array(["x"] * rows, type=pa.string())
    int_to_str = pa.array(range(rows), type=pa.int32())

    ts_match = pa.array([0] * rows, type=pa.timestamp("us", tz="UTC"))
    ts_naive = pa.array([0] * rows, type=pa.timestamp("us"))

    opts_int64 = CastOptions(target=int64_target)
    opts_int32 = CastOptions(target=int32_target)
    opts_float32 = CastOptions(target=float32_target)
    opts_float64 = CastOptions(target=Field("v", FloatingPointType(byte_size=8)))
    opts_string = CastOptions(target=string_target)
    opts_ts = CastOptions(target=ts_target)

    out.append(_time_one(
        f"arrow: cast_arrow_array int64 MATCH rows={rows}",
        lambda: int_target.cast_arrow_array(int_match, opts_int64),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"arrow: cast_arrow_array int32->int64 WIDEN rows={rows}",
        lambda: int_target.cast_arrow_array(int_widen, opts_int64),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"arrow: cast_arrow_array int64->int32 NARROW rows={rows}",
        lambda: int32_target.dtype.cast_arrow_array(int_narrow, opts_int32),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"arrow: cast_arrow_array chunked int64 MATCH rows={rows}",
        lambda: int_target.cast_arrow_array(chunked, opts_int64),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"arrow: cast_arrow_array float32 MATCH rows={rows}",
        lambda: float_target.cast_arrow_array(float_match, opts_float32),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"arrow: cast_arrow_array float32->float64 WIDEN rows={rows}",
        lambda: opts_float64.target.dtype.cast_arrow_array(float_widen, opts_float64),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"arrow: cast_arrow_array float64->float32 NARROW rows={rows}",
        lambda: float_target.cast_arrow_array(float_narrow, opts_float32),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"arrow: cast_arrow_array string MATCH rows={rows}",
        lambda: string_target.dtype.cast_arrow_array(str_match, opts_string),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"arrow: cast_arrow_array int->string CAST rows={rows}",
        lambda: string_target.dtype.cast_arrow_array(int_to_str, opts_string),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"arrow: cast_arrow_array ts MATCH rows={rows}",
        lambda: ts_target.dtype.cast_arrow_array(ts_match, opts_ts),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"arrow: cast_arrow_array ts naive->UTC CAST rows={rows}",
        lambda: ts_target.dtype.cast_arrow_array(ts_naive, opts_ts),
        repeat=repeat, inner=500,
    ))

    return out


# ---------------------------------------------------------------------------
# Polars scenarios
# ---------------------------------------------------------------------------


def _polars_scenarios(rows: int, repeat: int) -> list[dict]:
    try:
        import polars as pl
    except ImportError:
        return [{"label": "polars: not installed — skipped",
                 "best": 0.0, "median": 0.0, "mean": 0.0}]

    out: list[dict] = []

    int64_target = Field("id", IntegerType(byte_size=8, signed=True), nullable=False)
    int32_target = Field("id", IntegerType(byte_size=4, signed=True), nullable=False)
    float32_target = Field("v", FloatingPointType(byte_size=4))
    string_target = Field("s", StringType())

    s_int_match = pl.Series("id", list(range(rows)), dtype=pl.Int64)
    s_int_widen = pl.Series("id", list(range(rows)), dtype=pl.Int32)
    s_int_narrow = pl.Series("id", list(range(rows)), dtype=pl.Int64)
    s_float_match = pl.Series("v", [1.5] * rows, dtype=pl.Float32)
    s_float_narrow = pl.Series("v", [1.5] * rows, dtype=pl.Float64)
    s_str_match = pl.Series("s", ["x"] * rows, dtype=pl.String)
    s_int_to_str = pl.Series("s", list(range(rows)), dtype=pl.Int32)

    opts_int64 = CastOptions(target=int64_target)
    opts_int32 = CastOptions(target=int32_target)
    opts_float32 = CastOptions(target=float32_target)
    opts_string = CastOptions(target=string_target)

    out.append(_time_one(
        f"polars: cast_polars_series int64 MATCH rows={rows}",
        lambda: int64_target.dtype.cast_polars_series(s_int_match, opts_int64),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"polars: cast_polars_series int32->int64 WIDEN rows={rows}",
        lambda: int64_target.dtype.cast_polars_series(s_int_widen, opts_int64),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"polars: cast_polars_series int64->int32 NARROW rows={rows}",
        lambda: int32_target.dtype.cast_polars_series(s_int_narrow, opts_int32),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"polars: cast_polars_series float32 MATCH rows={rows}",
        lambda: float32_target.dtype.cast_polars_series(s_float_match, opts_float32),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"polars: cast_polars_series float64->float32 NARROW rows={rows}",
        lambda: float32_target.dtype.cast_polars_series(s_float_narrow, opts_float32),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"polars: cast_polars_series string MATCH rows={rows}",
        lambda: string_target.dtype.cast_polars_series(s_str_match, opts_string),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"polars: cast_polars_series int->string CAST rows={rows}",
        lambda: string_target.dtype.cast_polars_series(s_int_to_str, opts_string),
        repeat=repeat, inner=500,
    ))

    return out


# ---------------------------------------------------------------------------
# pandas scenarios
# ---------------------------------------------------------------------------


def _pandas_scenarios(rows: int, repeat: int) -> list[dict]:
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        return [{"label": "pandas: not installed — skipped",
                 "best": 0.0, "median": 0.0, "mean": 0.0}]

    out: list[dict] = []

    int64_target = Field("id", IntegerType(byte_size=8, signed=True), nullable=False)
    int32_target = Field("id", IntegerType(byte_size=4, signed=True), nullable=False)
    float32_target = Field("v", FloatingPointType(byte_size=4))
    string_target = Field("s", StringType())

    s_int_match = pd.Series(range(rows), dtype="int64", name="id")
    s_int_widen = pd.Series(range(rows), dtype="int32", name="id")
    s_int_narrow = pd.Series(range(rows), dtype="int64", name="id")
    s_float_match = pd.Series([1.5] * rows, dtype="float32", name="v")
    s_float_narrow = pd.Series([1.5] * rows, dtype="float64", name="v")
    s_str_match = pd.Series(["x"] * rows, name="s", dtype="object")
    s_int_to_str = pd.Series(range(rows), dtype="int32", name="s")

    opts_int64 = CastOptions(target=int64_target)
    opts_int32 = CastOptions(target=int32_target)
    opts_float32 = CastOptions(target=float32_target)
    opts_string = CastOptions(target=string_target)

    out.append(_time_one(
        f"pandas: cast_pandas_series int64 MATCH rows={rows}",
        lambda: int64_target.dtype.cast_pandas_series(s_int_match, opts_int64),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"pandas: cast_pandas_series int32->int64 WIDEN rows={rows}",
        lambda: int64_target.dtype.cast_pandas_series(s_int_widen, opts_int64),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"pandas: cast_pandas_series int64->int32 NARROW rows={rows}",
        lambda: int32_target.dtype.cast_pandas_series(s_int_narrow, opts_int32),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"pandas: cast_pandas_series float32 MATCH rows={rows}",
        lambda: float32_target.dtype.cast_pandas_series(s_float_match, opts_float32),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"pandas: cast_pandas_series float64->float32 NARROW rows={rows}",
        lambda: float32_target.dtype.cast_pandas_series(s_float_narrow, opts_float32),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"pandas: cast_pandas_series string MATCH rows={rows}",
        lambda: string_target.dtype.cast_pandas_series(s_str_match, opts_string),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"pandas: cast_pandas_series int->string CAST rows={rows}",
        lambda: string_target.dtype.cast_pandas_series(s_int_to_str, opts_string),
        repeat=repeat, inner=500,
    ))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tabular scenarios — whole-table casts with convenient + nested types.
# ---------------------------------------------------------------------------


def _convenient_schema() -> Schema:
    """Common analytics shape: id / amount / qty / name / ts / paid / placed_on."""
    return Schema.from_fields([
        Field("id", IntegerType(byte_size=8, signed=True), nullable=False),
        Field("amount", DecimalType(precision=18, scale=2)),
        Field("qty", IntegerType(byte_size=4, signed=True)),
        Field("name", StringType()),
        Field("ts", TimestampType(unit="us", tz="UTC")),
        Field("paid", BooleanType()),
        Field("placed_on", DateType()),
    ])


def _nested_schema() -> Schema:
    """Nested shape: tags array, attributes map, address struct."""
    return Schema.from_fields([
        Field("id", IntegerType(byte_size=8, signed=True), nullable=False),
        Field("tags", ArrayType.from_item(Field("item", StringType()))),
        Field("attributes", MapType.from_key_value(
            key_field=Field("k", StringType(), nullable=False),
            value_field=Field("v", StringType()),
        )),
        Field("address", StructType(fields=(
            Field("street", StringType()),
            Field("city", StringType()),
            Field("zip", StringType()),
        ))),
    ])


def _arrow_convenient_table(rows: int) -> pa.Table:
    return pa.table({
        "id": pa.array(range(rows), type=pa.int64()),
        "amount": pa.array(["1.50"] * rows, type=pa.string()),
        "qty": pa.array([1] * rows, type=pa.int32()),
        "name": pa.array(["x"] * rows, type=pa.string()),
        "ts": pa.array([0] * rows, type=pa.timestamp("us", tz="UTC")),
        "paid": pa.array([True] * rows, type=pa.bool_()),
        "placed_on": pa.array([0] * rows, type=pa.date32()),
    })


def _arrow_nested_table(rows: int) -> pa.Table:
    return pa.table({
        "id": pa.array(range(rows), type=pa.int64()),
        "tags": pa.array([["a", "b"]] * rows, type=pa.list_(pa.string())),
        "attributes": pa.array(
            [[("k", "v")]] * rows,
            type=pa.map_(pa.string(), pa.string()),
        ),
        "address": pa.array(
            [{"street": "1", "city": "x", "zip": "00"}] * rows,
            type=pa.struct([
                ("street", pa.string()),
                ("city", pa.string()),
                ("zip", pa.string()),
            ]),
        ),
    })


def _tabular_scenarios(rows: int, repeat: int, engines: list[str]) -> list[dict]:
    out: list[dict] = []

    convenient = _convenient_schema()
    nested = _nested_schema()

    arrow_conv = _arrow_convenient_table(rows)
    arrow_nested = _arrow_nested_table(rows)

    if "arrow" in engines:
        opts_conv = CastOptions(target=convenient)
        opts_nested = CastOptions(target=nested)
        out.append(_time_one(
            f"arrow: cast_arrow_tabular convenient rows={rows}",
            lambda: convenient.dtype.cast_arrow_tabular(arrow_conv, opts_conv),
            repeat=repeat, inner=100,
        ))
        out.append(_time_one(
            f"arrow: cast_arrow_tabular nested rows={rows}",
            lambda: nested.dtype.cast_arrow_tabular(arrow_nested, opts_nested),
            repeat=repeat, inner=100,
        ))

    if "polars" in engines:
        try:
            import polars as pl
            pl_conv = pl.from_arrow(arrow_conv)
            pl_nested = pl.from_arrow(arrow_nested)
            opts_conv = CastOptions(target=convenient)
            opts_nested = CastOptions(target=nested)
            out.append(_time_one(
                f"polars: cast_polars_tabular convenient rows={rows}",
                lambda: convenient.dtype.cast_polars_tabular(pl_conv, opts_conv),
                repeat=repeat, inner=100,
            ))
            out.append(_time_one(
                f"polars: cast_polars_tabular nested rows={rows}",
                lambda: nested.dtype.cast_polars_tabular(pl_nested, opts_nested),
                repeat=repeat, inner=100,
            ))
        except ImportError:
            pass

    if "pandas" in engines:
        try:
            import pandas as pd  # noqa: F401
            # Pandas convenient: convert via arrow → pandas to keep schema clean.
            pd_conv = arrow_conv.to_pandas(types_mapper=None)
            opts_conv = CastOptions(target=convenient)
            out.append(_time_one(
                f"pandas: cast_pandas_tabular convenient rows={rows}",
                lambda: convenient.dtype.cast_pandas_tabular(pd_conv, opts_conv),
                repeat=repeat, inner=100,
            ))
            # Pandas nested round-trip is brittle (Map / Struct support varies
            # by pandas version); omit nested for pandas tabular to keep the
            # bench focused on shapes that round-trip cleanly across versions.
        except ImportError:
            pass

    return out


_RUNNERS = {
    "arrow": _arrow_scenarios,
    "polars": _polars_scenarios,
    "pandas": _pandas_scenarios,
}


def scenarios(rows: int, repeat: int, engines: list[str]) -> list[dict]:
    results: list[dict] = []
    for engine in engines:
        runner = _RUNNERS.get(engine)
        if runner is None:
            results.append({
                "label": f"{engine}: unknown engine — skipped",
                "best": 0.0, "median": 0.0, "mean": 0.0,
            })
            continue
        results.extend(runner(rows, repeat))
    results.extend(_tabular_scenarios(rows, repeat, engines))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000,
                    help="Per-call row count (default 10_000).")
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    ap.add_argument(
        "--engines",
        default="arrow,polars,pandas",
        help="Comma-separated subset of engines to bench.",
    )
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    print(f"# rows={args.rows}  repeat={args.repeat}  engines={','.join(engines)}")
    print(f"# {'label':<62s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    for row in scenarios(args.rows, args.repeat, engines):
        print(_fmt(row))


if __name__ == "__main__":
    main()
