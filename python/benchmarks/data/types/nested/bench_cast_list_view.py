"""Benchmark ``ArrayType`` casts on ``list_view`` / ``large_list_view``
sources across pyarrow / polars / pandas.

Why this exists
---------------

``list_view`` and ``large_list_view`` carry per-row ``(offset, size)``
pairs — rows can point anywhere into the shared values buffer in any
order, and ranges may overlap. ``pyarrow.compute.cast(list_view → list)``
silently drops rows whose offsets don't pack into monotone List
offsets; we normalise on the way in via
:func:`yggdrasil.data.types.nested.array.cast_arrow_list_array` so
every row survives, including out-of-order and overlapping layouts.

ListView / LargeListView *targets* aren't supported (Parquet has no
view encoding and pyarrow's view-side casts are inconsistent across
builds) — the cast helper raises in that direction. This bench only
measures source-side normalisation.

Workloads measured here:

* ``list_view<int64>`` → ``list<int32>`` — minimal item, the cheap
  baseline.
* ``list_view<struct{...wide...}>`` → ``list<struct{narrowed}>`` — the
  realistic event-array shape with many struct items per row. Two
  axes: items-per-row and out-of-order layouts (``ordered`` /
  ``out_of_order``).
* ``large_list_view`` source — int64 offsets path.
* End-to-end Parquet round-trip — Parquet has no native list_view
  encoding, so the full ingest flow is "cast list_view source → list
  → Parquet write → Parquet read". Measures the whole chain.
* Cross-engine bridges:

  - **polars**: polars' Rust core rejects ``list_view`` directly
    (``DataType "+vl" is still not supported``). The realistic flow
    is ``cast_arrow_list_array(list_view → list)`` then
    ``polars.from_arrow``. Bench measures both legs.
  - **pandas**: pandas ingests list_view via ``Array.to_pandas`` (one
    object Series per row). Bench measures the full
    ``cast_arrow → to_pandas`` path on the same wide ``list_view<struct>``.

Usage::

    PYTHONPATH=src python benchmarks/data/types/nested/bench_cast_list_view.py
    PYTHONPATH=src python benchmarks/data/types/nested/bench_cast_list_view.py --rows 50000
    PYTHONPATH=src python benchmarks/data/types/nested/bench_cast_list_view.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.nested import ArrayType, StructType
from yggdrasil.data.types.nested.array import cast_arrow_list_array
from yggdrasil.data.types.primitive import (
    FloatingPointType,
    IntegerType,
    StringType,
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
    scale, unit = 1e6, "us"
    if r["best"] < 1e-6:
        scale, unit = 1e9, "ns"
    elif r["best"] >= 1e-3:
        scale, unit = 1e3, "ms"
    return (
        f"{r['label']:<70s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Source builders
# ---------------------------------------------------------------------------


def _wide_struct_arrow_type(int_byte_size: int = 8) -> pa.StructType:
    int_t = pa.int64() if int_byte_size == 8 else pa.int32()
    return pa.struct(
        [(f"i{k:02d}", int_t) for k in range(8)]
        + [(f"s{k:02d}", pa.string()) for k in range(8)]
    )


def _wide_struct_target_field(int_byte_size: int = 4) -> Field:
    int_t = IntegerType(byte_size=int_byte_size, signed=True)
    return Field(
        "item",
        StructType(fields=tuple(
            [Field(f"i{k:02d}", int_t) for k in range(8)]
            + [Field(f"s{k:02d}", StringType()) for k in range(8)]
        )),
    )


def _wide_payload(items_per_row: int, rows: int) -> list:
    return [
        None if (r % 11 == 0) else [
            {
                **{f"i{k:02d}": r * items_per_row + k for k in range(8)},
                **{f"s{k:02d}": f"r{r}-k{k}" for k in range(8)},
            }
            for _ in range(items_per_row)
        ]
        for r in range(rows)
    ]


def _build_int_list_view(rows: int) -> pa.ListViewArray:
    return pa.array(
        [None if (r % 7 == 0) else list(range(r % 5 + 1))
         for r in range(rows)],
        type=pa.list_view(pa.int64()),
    )


def _build_out_of_order_list_view(rows: int, items_per_row: int) -> pa.ListViewArray:
    """Build a list_view whose offsets walk backwards row-by-row.

    Total entries = rows * items_per_row. Row r reads from
    ``(rows-1-r) * items_per_row``. Forces the cast path through the
    flatten() gather: pyarrow's pc.cast would corrupt this layout.
    """
    total = rows * items_per_row
    flat_values = pa.array(
        [{
            **{f"i{k:02d}": i + k for k in range(8)},
            **{f"s{k:02d}": f"v{i}-{k}" for k in range(8)},
         } for i in range(total)],
        type=_wide_struct_arrow_type(int_byte_size=8),
    )
    offsets = pa.array(
        [(rows - 1 - r) * items_per_row for r in range(rows)],
        type=pa.int32(),
    )
    sizes = pa.array([items_per_row] * rows, type=pa.int32())
    return pa.ListViewArray.from_arrays(
        offsets=offsets, sizes=sizes, values=flat_values,
    )


def _build_wide_list_view(items_per_row: int, rows: int) -> pa.ListViewArray:
    return pa.array(
        _wide_payload(items_per_row, rows),
        type=pa.list_view(_wide_struct_arrow_type()),
    )


def _build_wide_large_list_view(items_per_row: int, rows: int) -> pa.LargeListViewArray:
    return pa.array(
        _wide_payload(items_per_row, rows),
        type=pa.large_list_view(_wide_struct_arrow_type()),
    )


# ---------------------------------------------------------------------------
# Scenario blocks
# ---------------------------------------------------------------------------


def _arrow_scenarios(rows: int, items_per_row: int, repeat: int) -> list[dict]:
    out: list[dict] = []

    int_lv = _build_int_list_view(rows)
    int_target = Field(
        "vals",
        ArrayType.from_item(Field("item", IntegerType(byte_size=4, signed=True))),
    )

    wide_lv = _build_wide_list_view(items_per_row, rows)
    wide_lv_oo = _build_out_of_order_list_view(rows, items_per_row)
    wide_target = Field(
        "rows", ArrayType.from_item(_wide_struct_target_field(int_byte_size=4)),
    )

    wide_large_lv = _build_wide_large_list_view(items_per_row, rows)
    wide_large_target = Field(
        "rows",
        ArrayType.from_item(_wide_struct_target_field(int_byte_size=4), large=True),
    )

    out.append(_time_one(
        f"arrow: list_view<int64> -> list<int32> rows={rows}",
        lambda: cast_arrow_list_array(
            int_lv, CastOptions(target=int_target)),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"arrow: list_view<struct{{16}}>x{items_per_row} -> list<struct> rows={rows}",
        lambda: cast_arrow_list_array(
            wide_lv, CastOptions(target=wide_target)),
        repeat=repeat, inner=20,
    ))
    out.append(_time_one(
        f"arrow: list_view<struct{{16}}>x{items_per_row} OUT-OF-ORDER -> list rows={rows}",
        lambda: cast_arrow_list_array(
            wide_lv_oo, CastOptions(target=wide_target)),
        repeat=repeat, inner=20,
    ))
    out.append(_time_one(
        f"arrow: large_list_view<struct{{16}}>x{items_per_row} -> large_list rows={rows}",
        lambda: cast_arrow_list_array(
            wide_large_lv, CastOptions(target=wide_large_target)),
        repeat=repeat, inner=20,
    ))
    return out


def _parquet_scenarios(rows: int, items_per_row: int, repeat: int, tmp_dir: str) -> list[dict]:
    """End-to-end ingest: list_view source -> list cast -> Parquet write
    -> Parquet read. The shape that matters for batch warehouses."""
    import os

    out: list[dict] = []
    wide_lv = _build_wide_list_view(items_per_row, rows)
    wide_target = Field(
        "rows", ArrayType.from_item(_wide_struct_target_field(int_byte_size=4)),
    )

    # Pre-cast once — bench measures the write-then-read leg, not the
    # cast (covered above).
    casted = cast_arrow_list_array(wide_lv, CastOptions(target=wide_target))
    table = pa.table({"rows": casted})
    path = os.path.join(tmp_dir, f"list_view_struct_{rows}_{items_per_row}.parquet")

    def _write_then_read():
        pq.write_table(table, path)
        return pq.read_table(path)

    def _full_pipeline():
        c = cast_arrow_list_array(wide_lv, CastOptions(target=wide_target))
        pq.write_table(pa.table({"rows": c}), path)
        return pq.read_table(path)

    out.append(_time_one(
        f"parquet: write+read pre-casted rows={rows} items/row={items_per_row}",
        _write_then_read,
        repeat=repeat, inner=3,
    ))
    out.append(_time_one(
        f"parquet: full cast+write+read rows={rows} items/row={items_per_row}",
        _full_pipeline,
        repeat=repeat, inner=3,
    ))
    return out


def _polars_scenarios(rows: int, items_per_row: int, repeat: int) -> list[dict]:
    try:
        import polars as pl
    except ImportError:
        return [{"label": "polars: not installed — skipped",
                 "best": 0.0, "median": 0.0, "mean": 0.0}]

    out: list[dict] = []
    wide_lv = _build_wide_list_view(items_per_row, rows)
    wide_target = Field(
        "rows", ArrayType.from_item(_wide_struct_target_field(int_byte_size=4)),
    )

    # Polars' Rust core rejects ``pl.from_arrow(list_view)``. Realistic
    # flow: cast list_view -> list via our Arrow path, then bridge to
    # polars. We bench both legs separately to make the bridge cost
    # visible.
    casted = cast_arrow_list_array(wide_lv, CastOptions(target=wide_target))

    out.append(_time_one(
        f"polars: from_arrow(list<struct>) post-cast rows={rows} items/row={items_per_row}",
        lambda: pl.from_arrow(casted),
        repeat=repeat, inner=10,
    ))
    out.append(_time_one(
        f"polars: cast(list_view) + from_arrow rows={rows} items/row={items_per_row}",
        lambda: pl.from_arrow(
            cast_arrow_list_array(wide_lv, CastOptions(target=wide_target))),
        repeat=repeat, inner=10,
    ))
    return out


def _pandas_scenarios(rows: int, items_per_row: int, repeat: int) -> list[dict]:
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return [{"label": "pandas: not installed — skipped",
                 "best": 0.0, "median": 0.0, "mean": 0.0}]

    out: list[dict] = []
    wide_lv = _build_wide_list_view(items_per_row, rows)
    wide_target = Field(
        "rows", ArrayType.from_item(_wide_struct_target_field(int_byte_size=4)),
    )
    casted = cast_arrow_list_array(wide_lv, CastOptions(target=wide_target))

    out.append(_time_one(
        f"pandas: to_pandas(list<struct>) post-cast rows={rows} items/row={items_per_row}",
        lambda: casted.to_pandas(),
        repeat=repeat, inner=5,
    ))
    out.append(_time_one(
        f"pandas: cast(list_view) + to_pandas rows={rows} items/row={items_per_row}",
        lambda: cast_arrow_list_array(
            wide_lv, CastOptions(target=wide_target)).to_pandas(),
        repeat=repeat, inner=5,
    ))
    return out


def scenarios(rows: int, items_per_row: int, repeat: int, tmp_dir: str) -> list[dict]:
    return [
        *_arrow_scenarios(rows, items_per_row, repeat),
        *_parquet_scenarios(rows, items_per_row, repeat, tmp_dir),
        *_polars_scenarios(rows, items_per_row, repeat),
        *_pandas_scenarios(rows, items_per_row, repeat),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import tempfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000,
                    help="Outer row count.")
    ap.add_argument("--items-per-row", type=int, default=8,
                    help="Inner items per list_view row (struct count).")
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# rows={args.rows}  items/row={args.items_per_row}  repeat={args.repeat}")
    print(f"# {'label':<70s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    with tempfile.TemporaryDirectory() as tmp:
        for row in scenarios(args.rows, args.items_per_row, args.repeat, tmp):
            print(_fmt(row))


if __name__ == "__main__":
    main()
