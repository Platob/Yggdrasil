"""Benchmark primitive :class:`DataType` cast kernels — arrow / polars / pandas.

Mirrors the source tree: lives next to
``yggdrasil/data/types/primitive``. Covers ``IntegerType``,
``FloatingPointType``, ``StringType``, ``TimestampType`` casts in
three shapes per type:

* **MATCH** — source matches target; engine-type bypass should fire.
* **WIDEN** — source narrower than target (e.g. int32 → int64).
* **NARROW** — source wider than target (e.g. float64 → float32).

The companion ``bench_nested_cast.py`` (under
``data/types/nested/``) covers list / map / struct casts.

Usage::

    PYTHONPATH=src python benchmarks/data/types/primitive/bench_primitive_cast.py
    PYTHONPATH=src python benchmarks/data/types/primitive/bench_primitive_cast.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import pyarrow as pa

from datetime import date, time as dtime, timedelta
from decimal import Decimal

from yggdrasil.data import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.primitive import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DurationType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
    TimeType,
)


ROWS = 10_000


# ---------------------------------------------------------------------------
# Field targets.
# ---------------------------------------------------------------------------


F_INT64 = Field("v", IntegerType(byte_size=8, signed=True), nullable=False)
F_INT32 = Field("v", IntegerType(byte_size=4, signed=True), nullable=False)
F_UINT16 = Field("v", IntegerType(byte_size=2, signed=False))
F_F32 = Field("v", FloatingPointType(byte_size=4))
F_F64 = Field("v", FloatingPointType(byte_size=8))
F_STR = Field("v", StringType())
F_BOOL = Field("v", BooleanType())
F_BIN = Field("v", BinaryType())
F_TS = Field("v", TimestampType(unit="us", tz="UTC"))
F_TS_NAIVE = Field("v", TimestampType(unit="us"))
F_TS_MS = Field("v", TimestampType(unit="ms", tz="UTC"))
F_DATE = Field("v", DateType())
F_TIME = Field("v", TimeType(unit="us"))
F_DURATION = Field("v", DurationType(unit="us"))
F_DEC_18_2 = Field("v", DecimalType(precision=18, scale=2))
F_DEC_38_6 = Field("v", DecimalType(precision=38, scale=6))


# ---------------------------------------------------------------------------
# Source arrays.
# ---------------------------------------------------------------------------


def _build_sources() -> dict[str, pa.Array]:
    return {
        "int64": pa.array(range(ROWS), type=pa.int64()),
        "int32": pa.array(range(ROWS), type=pa.int32()),
        "int16": pa.array([i % 256 for i in range(ROWS)], type=pa.int16()),
        "uint16": pa.array([i % 256 for i in range(ROWS)], type=pa.uint16()),
        "f64": pa.array([1.5] * ROWS, type=pa.float64()),
        "f32": pa.array([1.5] * ROWS, type=pa.float32()),
        "str": pa.array(["x"] * ROWS, type=pa.string()),
        "str_int": pa.array([str(i) for i in range(ROWS)], type=pa.string()),
        "str_float": pa.array([f"{i}.5" for i in range(ROWS)], type=pa.string()),
        "str_ts": pa.array(["2024-05-12T10:00:00Z"] * ROWS, type=pa.string()),
        "str_date": pa.array(["2024-05-12"] * ROWS, type=pa.string()),
        "str_bool": pa.array(["true"] * ROWS, type=pa.string()),
        "str_decimal": pa.array(["1.50"] * ROWS, type=pa.string()),
        "bool": pa.array([i % 2 == 0 for i in range(ROWS)], type=pa.bool_()),
        "bin": pa.array([b"abc"] * ROWS, type=pa.binary()),
        "large_str": pa.array(["x"] * ROWS, type=pa.large_string()),
        "ts_utc": pa.array([0] * ROWS, type=pa.timestamp("us", tz="UTC")),
        "ts_naive": pa.array([0] * ROWS, type=pa.timestamp("us")),
        "ts_ms_utc": pa.array([0] * ROWS, type=pa.timestamp("ms", tz="UTC")),
        "ts_ns_naive": pa.array([0] * ROWS, type=pa.timestamp("ns")),
        "date": pa.array([date(2024, 5, 12)] * ROWS, type=pa.date32()),
        "time_us": pa.array([dtime(10, 30, 0)] * ROWS, type=pa.time64("us")),
        "duration_us": pa.array(
            [timedelta(seconds=1, microseconds=500)] * ROWS,
            type=pa.duration("us"),
        ),
        "dec_18_2": pa.array(
            [Decimal("1.50")] * ROWS, type=pa.decimal128(18, 2)
        ),
        "dec_10_2": pa.array(
            [Decimal("1.50")] * ROWS, type=pa.decimal128(10, 2)
        ),
        "int_for_str": pa.array(range(ROWS), type=pa.int32()),
        "f64_for_dec": pa.array([1.5] * ROWS, type=pa.float64()),
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
        f"{r['label']:<64s}  "
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
    """Run MATCH and CAST across arrow / polars / pandas for one Field target."""
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
    # --- numeric -----------------------------------------------------
    out.extend(_block("int64 (match int64 / cast int32→)", F_INT64, s["int64"], s["int32"],
                      repeat=repeat, inner=500))
    out.extend(_block("int32 (match int32 / cast int64→)", F_INT32, s["int32"], s["int64"],
                      repeat=repeat, inner=500))
    out.extend(_block("uint16 (match uint16 / narrow int32→)", F_UINT16, s["uint16"], s["int32"],
                      repeat=repeat, inner=500))
    out.extend(_block("float64 (match f64 / cast f32→)", F_F64, s["f64"], s["f32"],
                      repeat=repeat, inner=500))
    out.extend(_block("float32 (match f32 / cast f64→)", F_F32, s["f32"], s["f64"],
                      repeat=repeat, inner=500))
    out.extend(_block("int→float64 (match f64 / widen int64→)", F_F64, s["f64"], s["int64"],
                      repeat=repeat, inner=500))
    # --- decimal -----------------------------------------------------
    out.extend(_block("decimal(18,2) (match dec / rescale dec(10,2)→)",
                      F_DEC_18_2, s["dec_18_2"], s["dec_10_2"],
                      repeat=repeat, inner=300))
    out.extend(_block("decimal(38,6) (match dec / cast f64→)",
                      F_DEC_38_6, s["dec_18_2"], s["f64_for_dec"],
                      repeat=repeat, inner=200))
    # --- boolean / binary --------------------------------------------
    out.extend(_block("boolean (match bool / cast int32→)", F_BOOL, s["bool"], s["int32"],
                      repeat=repeat, inner=500))
    out.extend(_block("binary (match bin / cast str→)", F_BIN, s["bin"], s["str"],
                      repeat=repeat, inner=300))
    # --- string ------------------------------------------------------
    out.extend(_block("string (match str / cast int→)", F_STR, s["str"], s["int_for_str"],
                      repeat=repeat, inner=200))
    out.extend(_block("string (match str / cast large_string→)",
                      F_STR, s["str"], s["large_str"],
                      repeat=repeat, inner=300))
    out.extend(_block("int32→string (match int_for_str→str / cast f64→str)",
                      F_STR, s["str_int"], s["f64_for_dec"],
                      repeat=repeat, inner=200))
    # --- string → typed (parse paths) --------------------------------
    out.extend(_block("string→int32 (match int / parse str)",
                      F_INT32, s["int32"], s["str_int"],
                      repeat=repeat, inner=200))
    out.extend(_block("string→ts UTC (match ts / parse str)",
                      F_TS, s["ts_utc"], s["str_ts"],
                      repeat=repeat, inner=100))
    out.extend(_block("string→date (match date / parse str)",
                      F_DATE, s["date"], s["str_date"],
                      repeat=repeat, inner=200))
    out.extend(_block("string→decimal (match dec / parse str)",
                      F_DEC_18_2, s["dec_18_2"], s["str_decimal"],
                      repeat=repeat, inner=200))
    # --- temporal ----------------------------------------------------
    out.extend(_block("ts UTC (match ts_utc / cast naive→)", F_TS, s["ts_utc"], s["ts_naive"],
                      repeat=repeat, inner=500))
    out.extend(_block("ts naive (match naive / cast ms_utc→)",
                      F_TS_NAIVE, s["ts_naive"], s["ts_ms_utc"],
                      repeat=repeat, inner=500))
    out.extend(_block("ts ms UTC (match / unit-cast us→ms)",
                      F_TS_MS, s["ts_ms_utc"], s["ts_utc"],
                      repeat=repeat, inner=500))
    out.extend(_block("ts UTC (match / ns→us widen)",
                      F_TS, s["ts_utc"], s["ts_ns_naive"],
                      repeat=repeat, inner=500))
    out.extend(_block("date (match date / cast ts→)",
                      F_DATE, s["date"], s["ts_utc"],
                      repeat=repeat, inner=300))
    out.extend(_block("time us (match / cast ts→)",
                      F_TIME, s["time_us"], s["ts_utc"],
                      repeat=repeat, inner=300))
    out.extend(_block("duration us (match / cast int64→)",
                      F_DURATION, s["duration_us"], s["int64"],
                      repeat=repeat, inner=300))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}  rows={ROWS}")
    print(f"# {'label':<64s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
