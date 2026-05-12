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

from yggdrasil.data import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.primitive import (
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
)


ROWS = 10_000


# ---------------------------------------------------------------------------
# Field targets.
# ---------------------------------------------------------------------------


F_INT64 = Field("v", IntegerType(byte_size=8, signed=True), nullable=False)
F_INT32 = Field("v", IntegerType(byte_size=4, signed=True), nullable=False)
F_F32 = Field("v", FloatingPointType(byte_size=4))
F_F64 = Field("v", FloatingPointType(byte_size=8))
F_STR = Field("v", StringType())
F_TS = Field("v", TimestampType(unit="us", tz="UTC"))


# ---------------------------------------------------------------------------
# Source arrays.
# ---------------------------------------------------------------------------


def _build_sources() -> dict[str, pa.Array]:
    return {
        "int64": pa.array(range(ROWS), type=pa.int64()),
        "int32": pa.array(range(ROWS), type=pa.int32()),
        "f64": pa.array([1.5] * ROWS, type=pa.float64()),
        "f32": pa.array([1.5] * ROWS, type=pa.float32()),
        "str": pa.array(["x"] * ROWS, type=pa.string()),
        "ts_utc": pa.array([0] * ROWS, type=pa.timestamp("us", tz="UTC")),
        "ts_naive": pa.array([0] * ROWS, type=pa.timestamp("us")),
        "int_for_str": pa.array(range(ROWS), type=pa.int32()),
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
    out.extend(_block("int64 (match int64 / cast int32→)", F_INT64, s["int64"], s["int32"],
                      repeat=repeat, inner=500))
    out.extend(_block("int32 (match int32 / cast int64→)", F_INT32, s["int32"], s["int64"],
                      repeat=repeat, inner=500))
    out.extend(_block("float64 (match f64 / cast f32→)", F_F64, s["f64"], s["f32"],
                      repeat=repeat, inner=500))
    out.extend(_block("float32 (match f32 / cast f64→)", F_F32, s["f32"], s["f64"],
                      repeat=repeat, inner=500))
    out.extend(_block("string (match str / cast int→)", F_STR, s["str"], s["int_for_str"],
                      repeat=repeat, inner=200))
    out.extend(_block("ts UTC (match ts_utc / cast naive→)", F_TS, s["ts_utc"], s["ts_naive"],
                      repeat=repeat, inner=500))
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
