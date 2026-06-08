"""Edge-case cast benchmarks across Arrow / Polars / pandas / Spark.

Why this exists
---------------

The existing :mod:`bench_cast_data` covers the canonical
MATCH / WIDEN / NARROW per-column shapes. Production folder-of-folders
reads also hit a long tail of *edge* shapes that the steady-state
benches don't measure but that show up clearly when a regression
lands:

* **empty inputs** — zero-row arrays where the per-call overhead
  dominates the engine kernel cost.
* **all-null** — every cell ``None``; the cast still has to allocate
  + tag the null buffer but the kernel cost should collapse.
* **sparse nulls** — half the cells null; the kernel walks all rows
  but each row check pays the validity-bitmap branch.
* **dictionary / categorical** — Arrow ``DictionaryArray`` and
  Polars ``Categorical`` sources that often live behind
  Parquet-read frames and round-trip into flat dtypes on cast.
* **chunked / many small chunks** — ``pa.ChunkedArray`` with many
  small chunks (Parquet row-group reads).
* **timestamp tz convert** — ``UTC`` -> ``America/New_York`` cast
  + ``naive -> UTC`` localize, which goes through ``pc.cast`` for
  the tz convert and the polars/pandas equivalents.
* **decimal precision / scale rescale** — ``decimal(10, 2)`` ->
  ``decimal(18, 6)``.
* **JSON / nested** — JSON parse / large_string -> string /
  list-of-struct round-trips.
* **cross-engine** — going Arrow -> Polars + Polars -> Arrow via
  the cast registry (the converter dispatch + zero-copy bridges).
* **engine-bypass** — same engine type, same target field; the
  ``need_cast`` fast path should fire and return the input.
* **with default fill** — non-nullable target + ``default``; the
  cast pipeline must replace nulls in the source.

Usage::

    PYTHONPATH=src python benchmarks/data/bench_cast_edges.py
    PYTHONPATH=src python benchmarks/data/bench_cast_edges.py --rows 50000 --repeat 5
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.data import Field
from yggdrasil.data.cast import convert
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.nested import ArrayType, MapType, StructType
from yggdrasil.data.types.primitive import (
    BooleanType,
    DecimalType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
)
from yggdrasil.data.types.primitive.json import SJsonType

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:  # pragma: no cover - bench-only optional path
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  # pragma: no cover - bench-only optional path
    HAS_PANDAS = False


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
    scale = 1e6
    unit = "us"
    if r["best"] < 1e-6:
        scale, unit = 1e9, "ns"
    return (
        f"{r['label']:<70s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Arrow edge cases.
# ---------------------------------------------------------------------------


def _arrow_edges(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    inner = max(20, 200_000 // max(rows, 1))

    int_target = Field("id", IntegerType(byte_size=8, signed=True))
    int32_target = Field("id", IntegerType(byte_size=4, signed=True))
    string_target = Field("s", StringType())
    ts_utc_target = Field("ts", TimestampType(unit="us", tz="UTC"))
    ts_ny_target = Field("ts", TimestampType(unit="us", tz="America/New_York"))
    dec_in = Field("d", DecimalType(precision=10, scale=2))
    dec_out = Field("d", DecimalType(precision=18, scale=6))

    opts_int = CastOptions(target=int_target)
    opts_int32 = CastOptions(target=int32_target)
    opts_string = CastOptions(target=string_target)
    opts_ts_utc = CastOptions(target=ts_utc_target)
    opts_ts_ny = CastOptions(target=ts_ny_target)
    opts_dec_out = CastOptions(target=dec_out)

    # ---- fixtures -----------------------------------------------------
    # Empty / single / large.
    empty_int = pa.array([], type=pa.int64())
    single_int = pa.array([7], type=pa.int64())

    # All-null / sparse-null versions of the same shape.
    all_null = pa.array([None] * rows, type=pa.int64())
    sparse_null = pa.array(
        [i if i % 2 == 0 else None for i in range(rows)],
        type=pa.int64(),
    )

    # Plain matched + cast inputs.
    int_match = pa.array(list(range(rows)), type=pa.int64())
    int_widen = pa.array(list(range(rows)), type=pa.int32())

    # Dictionary array — Parquet reads emit these for low-cardinality
    # string columns; cast registry should dictionary-decode to flat
    # string without materializing the whole dict-of-indices.
    dict_arr = pa.DictionaryArray.from_arrays(
        pa.array([i % 5 for i in range(rows)], type=pa.int32()),
        pa.array(["a", "b", "c", "d", "e"]),
    )
    string_match = pa.array([f"x{i}" for i in range(min(rows, 1000))] * (rows // 1000 + 1))[:rows]

    # Chunked array with many small chunks — the cast hot path's
    # per-chunk loop dominates here.
    chunk_sz = max(1, rows // 64)
    chunks = [
        pa.array(list(range(start, min(start + chunk_sz, rows))), type=pa.int64())
        for start in range(0, rows, chunk_sz)
    ]
    chunked_int = pa.chunked_array(chunks, type=pa.int64())

    # Timestamps: naive vs UTC, plus UTC -> NY tz convert.
    ts_utc = pa.array([0] * rows, type=pa.timestamp("us", tz="UTC"))
    ts_naive = pa.array([0] * rows, type=pa.timestamp("us"))

    # Decimals: rescale precision/scale.
    dec_in_arr = pa.array(
        [pa.scalar(i, type=pa.decimal128(10, 2)).as_py() for i in range(min(rows, 100))] * (rows // 100 + 1),
        type=pa.decimal128(10, 2),
    )[:rows]

    # JSON-shaped string source for json parse target.
    json_str = pa.array(['{"a": 1, "b": "x"}'] * rows, type=pa.string())

    # ---- empty / single -----------------------------------------------
    out.append(_time_one(
        f"arrow: cast int64 MATCH empty",
        lambda: int_target.dtype.cast_arrow_array(empty_int, opts_int),
        repeat=repeat, inner=inner * 20,
    ))
    out.append(_time_one(
        f"arrow: cast int64 MATCH single-row",
        lambda: int_target.dtype.cast_arrow_array(single_int, opts_int),
        repeat=repeat, inner=inner * 20,
    ))

    # ---- null shapes --------------------------------------------------
    out.append(_time_one(
        f"arrow: cast int64 MATCH all-null rows={rows}",
        lambda: int_target.dtype.cast_arrow_array(all_null, opts_int),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"arrow: cast int32->int64 WIDEN all-null rows={rows}",
        lambda: int_target.dtype.cast_arrow_array(
            pa.array([None] * rows, type=pa.int32()), opts_int
        ),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"arrow: cast int64 MATCH sparse-null rows={rows}",
        lambda: int_target.dtype.cast_arrow_array(sparse_null, opts_int),
        repeat=repeat, inner=inner,
    ))

    # ---- engine-bypass ------------------------------------------------
    # Same target Field, same source dtype: ``need_cast`` should return
    # the input array unchanged.
    out.append(_time_one(
        f"arrow: cast int64 MATCH bypass rows={rows}",
        lambda: int_target.dtype.cast_arrow_array(int_match, opts_int),
        repeat=repeat, inner=inner,
    ))

    # ---- dictionary source -------------------------------------------
    out.append(_time_one(
        f"arrow: cast dictionary<str>->string rows={rows}",
        lambda: string_target.dtype.cast_arrow_array(dict_arr, opts_string),
        repeat=repeat, inner=inner,
    ))

    # ---- chunked array -----------------------------------------------
    out.append(_time_one(
        f"arrow: cast chunked int64 MATCH rows={rows} chunks={len(chunks)}",
        lambda: int_target.dtype.cast_arrow_array(chunked_int, opts_int),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"arrow: cast chunked int64->int32 NARROW rows={rows}",
        lambda: int32_target.dtype.cast_arrow_array(chunked_int, opts_int32),
        repeat=repeat, inner=inner,
    ))

    # ---- temporal tz ------------------------------------------------
    out.append(_time_one(
        f"arrow: cast ts(UTC)->ts(NY) rows={rows}",
        lambda: ts_ny_target.dtype.cast_arrow_array(ts_utc, opts_ts_ny),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"arrow: cast ts(naive)->ts(UTC) rows={rows}",
        lambda: ts_utc_target.dtype.cast_arrow_array(ts_naive, opts_ts_utc),
        repeat=repeat, inner=inner,
    ))

    # ---- decimal rescale ---------------------------------------------
    out.append(_time_one(
        f"arrow: cast decimal(10,2)->decimal(18,6) rows={rows}",
        lambda: dec_out.dtype.cast_arrow_array(dec_in_arr, opts_dec_out),
        repeat=repeat, inner=inner,
    ))

    # ---- JSON parse path --------------------------------------------
    sjson_target = Field("j", SJsonType())
    opts_sjson = CastOptions(target=sjson_target)
    out.append(_time_one(
        f"arrow: cast string->sjson rows={rows}",
        lambda: sjson_target.dtype.cast_arrow_array(json_str, opts_sjson),
        repeat=repeat, inner=max(20, inner // 10),
    ))

    # ---- default-fill non-nullable target ---------------------------
    nn_target = Field(
        "id",
        IntegerType(byte_size=8, signed=True),
        nullable=False,
        default=0,
    )
    opts_nn = CastOptions(target=nn_target)
    out.append(_time_one(
        f"arrow: cast int64 sparse-null + fill-default rows={rows}",
        lambda: nn_target.dtype.cast_arrow_array(sparse_null, opts_nn),
        repeat=repeat, inner=inner,
    ))

    return out


# ---------------------------------------------------------------------------
# Polars edge cases.
# ---------------------------------------------------------------------------


def _polars_edges(rows: int, repeat: int) -> list[dict]:
    if not HAS_POLARS:
        return []
    out: list[dict] = []
    inner = max(20, 200_000 // max(rows, 1))

    int_target = Field("id", IntegerType(byte_size=8, signed=True))
    int32_target = Field("id", IntegerType(byte_size=4, signed=True))
    string_target = Field("s", StringType())
    ts_ny_target = Field("ts", TimestampType(unit="us", tz="America/New_York"))

    opts_int = CastOptions(target=int_target)
    opts_int32 = CastOptions(target=int32_target)
    opts_string = CastOptions(target=string_target)
    opts_ts_ny = CastOptions(target=ts_ny_target)

    s_empty = pl.Series("id", [], dtype=pl.Int64)
    s_all_null = pl.Series("id", [None] * rows, dtype=pl.Int64)
    s_sparse_null = pl.Series(
        "id",
        [i if i % 2 == 0 else None for i in range(rows)],
        dtype=pl.Int64,
    )
    s_match = pl.Series("id", list(range(rows)), dtype=pl.Int64)
    s_widen = pl.Series("id", list(range(rows)), dtype=pl.Int32)

    # Categorical (Polars equivalent of Arrow dictionary).
    s_cat = pl.Series(
        "s",
        ["a", "b", "c", "d", "e"] * (rows // 5 + 1),
    )[:rows].cast(pl.Categorical)

    s_ts_utc = pl.Series(
        "ts",
        [0] * rows,
        dtype=pl.Datetime("us", "UTC"),
    )

    out.append(_time_one(
        f"polars: cast int64 MATCH empty",
        lambda: int_target.dtype.cast_polars_series(s_empty, opts_int),
        repeat=repeat, inner=inner * 20,
    ))
    out.append(_time_one(
        f"polars: cast int64 MATCH all-null rows={rows}",
        lambda: int_target.dtype.cast_polars_series(s_all_null, opts_int),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"polars: cast int64 MATCH sparse-null rows={rows}",
        lambda: int_target.dtype.cast_polars_series(s_sparse_null, opts_int),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"polars: cast int64 MATCH bypass rows={rows}",
        lambda: int_target.dtype.cast_polars_series(s_match, opts_int),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"polars: cast int32->int64 WIDEN rows={rows}",
        lambda: int_target.dtype.cast_polars_series(s_widen, opts_int),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"polars: cast int64->int32 NARROW rows={rows}",
        lambda: int32_target.dtype.cast_polars_series(s_match, opts_int32),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"polars: cast categorical->string rows={rows}",
        lambda: string_target.dtype.cast_polars_series(s_cat, opts_string),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"polars: cast ts(UTC)->ts(NY) rows={rows}",
        lambda: ts_ny_target.dtype.cast_polars_series(s_ts_utc, opts_ts_ny),
        repeat=repeat, inner=inner,
    ))

    return out


# ---------------------------------------------------------------------------
# Pandas edge cases.
# ---------------------------------------------------------------------------


def _pandas_edges(rows: int, repeat: int) -> list[dict]:
    if not HAS_PANDAS:
        return []
    out: list[dict] = []
    inner = max(20, 200_000 // max(rows, 1))

    int_target = Field("id", IntegerType(byte_size=8, signed=True))
    string_target = Field("s", StringType())

    opts_int = CastOptions(target=int_target)
    opts_string = CastOptions(target=string_target)

    s_empty = pd.Series([], dtype="int64", name="id")
    s_all_null = pd.Series([None] * rows, dtype="Int64", name="id")
    s_match = pd.Series(list(range(rows)), dtype="int64", name="id")
    # Pandas category dtype — same logical shape as Arrow dictionary.
    s_cat = pd.Series(["a", "b", "c", "d", "e"] * (rows // 5 + 1), name="s")[:rows].astype("category")

    out.append(_time_one(
        f"pandas: cast int64 MATCH empty",
        lambda: int_target.dtype.cast_pandas_series(s_empty, opts_int),
        repeat=repeat, inner=inner * 10,
    ))
    out.append(_time_one(
        f"pandas: cast int64 MATCH all-null rows={rows}",
        lambda: int_target.dtype.cast_pandas_series(s_all_null, opts_int),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"pandas: cast int64 MATCH bypass rows={rows}",
        lambda: int_target.dtype.cast_pandas_series(s_match, opts_int),
        repeat=repeat, inner=inner,
    ))
    out.append(_time_one(
        f"pandas: cast category->string rows={rows}",
        lambda: string_target.dtype.cast_pandas_series(s_cat, opts_string),
        repeat=repeat, inner=inner,
    ))

    return out


# ---------------------------------------------------------------------------
# Cross-engine registry dispatch.
# ---------------------------------------------------------------------------


def _cross_engine_edges(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    inner = max(20, 100_000 // max(rows, 1))

    pa_table = pa.table({
        "id": pa.array(list(range(rows)), type=pa.int64()),
        "name": pa.array(["x"] * rows, type=pa.string()),
        "amount": pa.array([1.5] * rows, type=pa.float64()),
    })
    int_arr = pa_table["id"].combine_chunks()

    if HAS_POLARS:
        pl_df = pl.DataFrame({
            "id": list(range(rows)),
            "name": ["x"] * rows,
            "amount": [1.5] * rows,
        })
        out.append(_time_one(
            f"registry: pa.Table -> pl.DataFrame rows={rows}",
            lambda: convert(pa_table, pl.DataFrame),
            repeat=repeat, inner=max(10, inner // 2),
        ))
        out.append(_time_one(
            f"registry: pl.DataFrame -> pa.Table rows={rows}",
            lambda: convert(pl_df, pa.Table),
            repeat=repeat, inner=max(10, inner // 2),
        ))

    if HAS_PANDAS:
        pd_series = pd.Series(list(range(rows)), dtype="int64", name="id")
        pd_df = pd.DataFrame({
            "id": list(range(rows)),
            "name": ["x"] * rows,
            "amount": [1.5] * rows,
        })
        out.append(_time_one(
            f"registry: pa.Array -> pd.Series rows={rows}",
            lambda: convert(int_arr, pd.Series),
            repeat=repeat, inner=max(10, inner // 2),
        ))
        out.append(_time_one(
            f"registry: pd.Series -> pa.Array rows={rows}",
            lambda: convert(pd_series, pa.Array),
            repeat=repeat, inner=max(10, inner // 2),
        ))
        out.append(_time_one(
            f"registry: pa.Table -> pd.DataFrame rows={rows}",
            lambda: convert(pa_table, pd.DataFrame),
            repeat=repeat, inner=max(10, inner // 2),
        ))
        out.append(_time_one(
            f"registry: pd.DataFrame -> pa.Table rows={rows}",
            lambda: convert(pd_df, pa.Table),
            repeat=repeat, inner=max(10, inner // 2),
        ))

    return out


# ---------------------------------------------------------------------------
# Nested struct edge cases — mixed match/widen children + null structs.
# ---------------------------------------------------------------------------


def _nested_edges(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    inner = max(20, 100_000 // max(rows, 1))

    # struct[id:int32, name:string] -> struct[id:int64, name:string]
    # — one child widens, the other matches. Tests that the per-child
    # bypass fires on the matching column.
    src_struct = pa.StructArray.from_arrays(
        [
            pa.array(list(range(rows)), type=pa.int32()),
            pa.array(["x"] * rows, type=pa.string()),
        ],
        names=["id", "name"],
    )
    target_struct = Field(
        "row",
        StructType(fields=(
            Field("id", IntegerType(byte_size=8, signed=True)),
            Field("name", StringType()),
        )),
    )
    opts_struct = CastOptions(target=target_struct)

    out.append(_time_one(
        f"arrow: cast struct[int32+str]->struct[int64+str] rows={rows}",
        lambda: target_struct.dtype.cast_arrow_array(src_struct, opts_struct),
        repeat=repeat, inner=inner,
    ))

    # list<int32> -> list<int64> — one child widens.
    src_list = pa.array(
        [list(range(5)) for _ in range(rows)],
        type=pa.list_(pa.int32()),
    )
    target_list = Field(
        "tags",
        ArrayType(item_field=Field("item", IntegerType(byte_size=8, signed=True))),
    )
    opts_list = CastOptions(target=target_list)
    out.append(_time_one(
        f"arrow: cast list<int32>->list<int64> rows={rows}",
        lambda: target_list.dtype.cast_arrow_array(src_list, opts_list),
        repeat=repeat, inner=max(10, inner // 4),
    ))

    # map<str, int32> -> map<str, int64>
    src_map = pa.array(
        [[("a", 1), ("b", 2)] for _ in range(rows)],
        type=pa.map_(pa.string(), pa.int32()),
    )
    target_map = Field(
        "lookup",
        MapType(item_field=Field("entry", StructType(fields=[
            Field("key", StringType(), nullable=False),
            Field("value", IntegerType(byte_size=8, signed=True)),
        ]))),
    )
    opts_map = CastOptions(target=target_map)
    out.append(_time_one(
        f"arrow: cast map<str,int32>->map<str,int64> rows={rows}",
        lambda: target_map.dtype.cast_arrow_array(src_map, opts_map),
        repeat=repeat, inner=max(10, inner // 4),
    ))

    return out


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def scenarios(rows: int, repeat: int, engines: list[str]) -> list[dict]:
    out: list[dict] = []
    if "arrow" in engines:
        out.extend(_arrow_edges(rows, repeat))
    if "polars" in engines:
        out.extend(_polars_edges(rows, repeat))
    if "pandas" in engines:
        out.extend(_pandas_edges(rows, repeat))
    if "registry" in engines:
        out.extend(_cross_engine_edges(rows, repeat))
    if "nested" in engines:
        out.extend(_nested_edges(rows, repeat))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--engines",
        type=str,
        default="arrow,polars,pandas,registry,nested",
        help="Comma-separated subset of arrow,polars,pandas,registry,nested",
    )
    args = parser.parse_args()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    results = scenarios(args.rows, args.repeat, engines)
    print(f"# rows={args.rows} repeat={args.repeat}")
    print(f"# {'label':<70s}  {'best':>15}  {'median':>15}  {'mean':>15}")
    for r in results:
        print(_fmt(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
