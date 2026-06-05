"""Benchmark :mod:`yggdrasil.polars.cast` — the Polars-side conversion surface.

Mirrors the structure of ``benchmarks/arrow/bench_arrow_cast.py`` against
the polars converters registered in ``yggdrasil.polars.cast``:

* :func:`cast_polars_array` — ``pl.Series`` / ``pl.Expr`` → narrow cast.
* :func:`cast_polars_dataframe` / :func:`cast_polars_lazyframe` —
  full-frame cast on eager and lazy frames.
* :func:`any_to_polars_dataframe` — the cross-engine entry point.
  Exercises every namespace branch (pa.Table, pa.RecordBatch, pa.
  RecordBatchReader, pandas, list[dict], dict-of-cols, ``None``).
* :func:`polars_dataframe_to_arrow_table` — the polars→arrow exit.

What we measure for each shape:

* **MATCH** path — source schema already lines up with the target, so
  the engine-level bypass should short-circuit (see
  ``BaseDataType._cast_polars_series`` and
  ``cast_polars_tabular``'s ``source_pl_schema == target_pl_schema``
  bypass).  This number is the steady-state cost of "no real cast,
  please" and dominates real pipelines.
* **CAST** path — source widens (``id`` int32 → int64) so the cast
  kernel actually fires per column.  This number is the cost of the
  expression-tree rebuild + the polars `.select` over the cast plan.

Coverage that is **not** here (covered elsewhere):

* Arrow kernels (``benchmarks/arrow/bench_arrow_cast.py``).
* Registry dispatch only (``benchmarks/data/bench_cast.py``).

Usage::

    PYTHONPATH=src python benchmarks/polars/bench_polars_cast.py
    PYTHONPATH=src python benchmarks/polars/bench_polars_cast.py --rows 50000 --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.data import Field
from yggdrasil.data.cast.registry import convert
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema

# Polars is the subject under test — fail loudly if it isn't installed so
# the bench can't silently no-op like the legacy "polars: not installed"
# string in ``benchmarks/data/bench_cast.py``.
import polars as pl  # noqa: E402

from yggdrasil.polars.cast import (  # noqa: E402
    any_to_polars_dataframe,
    cast_polars_array,
    cast_polars_dataframe,
    cast_polars_lazyframe,
    polars_dataframe_to_arrow_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


TARGET_SCHEMA = Schema.from_fields(
    [
        Field("id", "int64", nullable=False),
        Field("amount", "float64"),
        Field("qty", "int32"),
        Field("name", "string"),
        Field("ts", "timestamp(us, UTC)"),
        Field("active", "bool"),
    ]
)


def _arrow_table(rows: int, *, mismatch: bool = False) -> pa.Table:
    """6-column analytics shape; ``mismatch`` widens ``id`` from int32→int64.

    MATCH builds against the target's pyarrow schema so the engine-level
    ``source_pl_schema == target_pl_schema`` bypass actually fires —
    otherwise the timestamp default (UTC) silently widens the MATCH
    case into the CAST path and we measure the rebuild on both numbers.
    """
    id_type = pa.int32() if mismatch else pa.int64()
    ts = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    return pa.table(
        {
            "id": pa.array(range(rows), type=id_type),
            "amount": pa.array([1.5] * rows, type=pa.float64()),
            "qty": pa.array([2] * rows, type=pa.int32()),
            "name": pa.array(
                ["row-" + str(i % 100) for i in range(rows)], type=pa.string()
            ),
            "ts": pa.array([ts] * rows, type=pa.timestamp("us", tz="UTC")),
            "active": pa.array(
                [(i % 2 == 0) for i in range(rows)], type=pa.bool_()
            ),
        }
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
        scale, unit = 1e9, "ns"
    elif r["best"] >= 1e-3:
        scale, unit = 1e3, "ms"
    return (
        f"{r['label']:<72s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _series_scenarios(rows: int, repeat: int) -> list[dict]:
    """Per-column casts — the inner-loop cost when callers cast one
    Series at a time (cross-engine UDF wrappers, ``select(...).to_series()``).
    """
    out: list[dict] = []
    id_field = TARGET_SCHEMA.field_by("id")
    opts = CastOptions(target=id_field)

    arrow_match = pa.array(range(rows), type=pa.int64())
    arrow_cast = pa.array(range(rows), type=pa.int32())
    series_match = pl.Series("id", arrow_match)
    series_cast = pl.Series("id", arrow_cast)

    out.append(_time_one(
        f"series: cast_polars_array(Series) MATCH rows={rows}",
        lambda: cast_polars_array(series_match, opts),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        f"series: cast_polars_array(Series) CAST rows={rows}",
        lambda: cast_polars_array(series_cast, opts),
        repeat=repeat, inner=2_000,
    ))

    return out


def _dataframe_scenarios(rows: int, repeat: int) -> list[dict]:
    """Eager DataFrame casts — the dominant pipeline shape."""
    out: list[dict] = []
    opts = CastOptions(target=TARGET_SCHEMA)

    df_match = pl.from_arrow(_arrow_table(rows, mismatch=False))
    df_cast = pl.from_arrow(_arrow_table(rows, mismatch=True))

    out.append(_time_one(
        f"frame: cast_polars_dataframe MATCH rows={rows}",
        lambda: cast_polars_dataframe(df_match, opts),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        f"frame: cast_polars_dataframe CAST rows={rows}",
        lambda: cast_polars_dataframe(df_cast, opts),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"frame: cast_polars_dataframe no-target rows={rows}",
        lambda: cast_polars_dataframe(df_match, None),
        repeat=repeat, inner=200_000,
    ))
    # Registry entry — third-party callers hit this path.
    out.append(_time_one(
        f"frame: convert(df, pl.DataFrame) MATCH rows={rows}",
        lambda: convert(df_match, pl.DataFrame, options=opts),
        repeat=repeat, inner=2_000,
    ))
    return out


def _lazyframe_scenarios(rows: int, repeat: int) -> list[dict]:
    """LazyFrame casts — the cast wraps the plan, no compute fires."""
    out: list[dict] = []
    opts = CastOptions(target=TARGET_SCHEMA)

    lf_match = pl.from_arrow(_arrow_table(rows, mismatch=False)).lazy()
    lf_cast = pl.from_arrow(_arrow_table(rows, mismatch=True)).lazy()

    out.append(_time_one(
        f"lazy: cast_polars_lazyframe MATCH rows={rows}",
        lambda: cast_polars_lazyframe(lf_match, opts),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"lazy: cast_polars_lazyframe CAST rows={rows}",
        lambda: cast_polars_lazyframe(lf_cast, opts),
        repeat=repeat, inner=200,
    ))
    return out


def _any_to_polars_scenarios(rows: int, repeat: int) -> list[dict]:
    """``any_to_polars_dataframe`` — every input namespace branch.

    Most pipelines hit this through ``convert(obj, pl.DataFrame)`` so
    the per-call namespace dispatch cost matters even when the body
    work is cheap.
    """
    out: list[dict] = []
    opts = CastOptions(target=TARGET_SCHEMA)

    pa_table = _arrow_table(rows, mismatch=False)
    pa_table_cast = _arrow_table(rows, mismatch=True)
    pa_batch = pa_table.combine_chunks().to_batches()[0]
    df_match = pl.from_arrow(pa_table)
    lf_match = df_match.lazy()
    reader_schema = pa_table.schema

    # pl.DataFrame passthrough (cheapest branch).
    out.append(_time_one(
        f"any2pl: any_to_polars_dataframe(DataFrame) MATCH rows={rows}",
        lambda: any_to_polars_dataframe(df_match, opts),
        repeat=repeat, inner=2_000,
    ))
    # pl.LazyFrame → .collect() + cast.
    out.append(_time_one(
        f"any2pl: any_to_polars_dataframe(LazyFrame) MATCH rows={rows}",
        lambda: any_to_polars_dataframe(lf_match, opts),
        repeat=repeat, inner=500,
    ))
    # pa.Table — zero-copy bridge.
    out.append(_time_one(
        f"any2pl: any_to_polars_dataframe(pa.Table) MATCH rows={rows}",
        lambda: any_to_polars_dataframe(pa_table, opts),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"any2pl: any_to_polars_dataframe(pa.Table) CAST rows={rows}",
        lambda: any_to_polars_dataframe(pa_table_cast, opts),
        repeat=repeat, inner=200,
    ))
    # pa.RecordBatch — wrap in Table then bridge.
    out.append(_time_one(
        f"any2pl: any_to_polars_dataframe(pa.RecordBatch) MATCH rows={rows}",
        lambda: any_to_polars_dataframe(pa_batch, opts),
        repeat=repeat, inner=500,
    ))
    # pa.RecordBatchReader — has to be rebuilt each iteration (one-shot reads).
    out.append(_time_one(
        f"any2pl: any_to_polars_dataframe(Reader) rows={rows}",
        lambda: any_to_polars_dataframe(
            pa.RecordBatchReader.from_batches(reader_schema, [pa_batch]),
            opts,
        ),
        repeat=repeat, inner=500,
    ))
    # ``None`` → empty frame with target schema.
    out.append(_time_one(
        "any2pl: any_to_polars_dataframe(None) -> empty",
        lambda: any_to_polars_dataframe(None, opts),
        repeat=repeat, inner=2_000,
    ))
    # dict-of-columns — common Python-side handoff.
    dict_payload = {
        "id": list(range(rows)),
        "amount": [1.5] * rows,
        "qty": [2] * rows,
        "name": ["row-" + str(i % 100) for i in range(rows)],
        "ts": [dt.datetime(2024, 1, 1)] * rows,
        "active": [(i % 2 == 0) for i in range(rows)],
    }
    out.append(_time_one(
        f"any2pl: any_to_polars_dataframe(dict) rows={rows}",
        lambda: any_to_polars_dataframe(dict_payload, opts),
        repeat=repeat, inner=100,
    ))
    return out


def _polars_to_arrow_scenarios(rows: int, repeat: int) -> list[dict]:
    """``polars_dataframe_to_arrow_table`` — the polars → arrow exit."""
    out: list[dict] = []
    opts = CastOptions(target=TARGET_SCHEMA)

    df_match = pl.from_arrow(_arrow_table(rows, mismatch=False))
    df_cast = pl.from_arrow(_arrow_table(rows, mismatch=True))
    lf_match = df_match.lazy()

    out.append(_time_one(
        f"exit: polars_dataframe_to_arrow_table MATCH rows={rows}",
        lambda: polars_dataframe_to_arrow_table(df_match, opts),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"exit: polars_dataframe_to_arrow_table CAST rows={rows}",
        lambda: polars_dataframe_to_arrow_table(df_cast, opts),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"exit: polars_dataframe_to_arrow_table(Lazy) MATCH rows={rows}",
        lambda: polars_dataframe_to_arrow_table(lf_match, opts),
        repeat=repeat, inner=200,
    ))
    return out


def scenarios(rows: int, repeat: int) -> list[dict]:
    return [
        *_series_scenarios(rows, repeat),
        *_dataframe_scenarios(rows, repeat),
        *_lazyframe_scenarios(rows, repeat),
        *_any_to_polars_scenarios(rows, repeat),
        *_polars_to_arrow_scenarios(rows, repeat),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000,
                    help="Row count for the in-memory fixture table.")
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# rows={args.rows}  repeat={args.repeat}")
    print(f"# {'label':<72s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    for row in scenarios(args.rows, args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
