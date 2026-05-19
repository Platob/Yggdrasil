"""Benchmark :mod:`yggdrasil.dataclasses.safe_function`.

Why this exists
---------------

:func:`check_function_args` and :func:`build_row_invoker` run on the
hot edge of every ``Dataset.apply`` / ``Dataset.map`` call: an
N-row transform pays the per-call inspection cost N times if it
goes through ``check_function_args`` for each row, but only once
when the caller pre-builds an invoker. This bench quantifies the
gap so regressions land in a visible number rather than a slow
``apply``.

The four scenarios mirror the four invoker shapes
:func:`build_row_invoker` recognises:

* single-arg unannotated — the cheapest path (no coercion).
* single-arg annotated — one ``convert(value, ann)`` per row.
* multi-arg dict-spread — pre-built coercer + ``**row``.
* ``**kwargs`` catch-all — full dict spread, no per-row inspect.

Each scenario also runs the "naive" baseline that calls
``check_function_args(func, args, kwargs)`` per row — that's
roughly what a hand-rolled ``apply`` does without the
``build_row_invoker`` cache. The delta is the win.

Usage::

    PYTHONPATH=src python benchmarks/dataclasses/bench_safe_function.py
    PYTHONPATH=src python benchmarks/dataclasses/bench_safe_function.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.dataclasses.safe_function import (
    build_batch_invoker,
    build_row_invoker,
    check_function_args,
)


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
        f"{r['label']:<60s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Fixture callables — module-scope so :func:`inspect.signature` resolves them
# the same way under all repeats.
# ---------------------------------------------------------------------------


def _single_unannotated(x):
    return x


def _single_annotated(x: int) -> int:
    return x


def _multi_arg(id: int, name: str) -> tuple:
    return (id, name)


def _var_kw(**row) -> dict:
    return row


def _var_pos(*xs) -> tuple:
    return xs


# ---------------------------------------------------------------------------
# Scenarios.
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    int_row = 42
    str_row = "42"
    dict_row = {"id": 1, "name": "x"}
    dict_row_str = {"id": "1", "name": "x"}
    tuple_row = (1, 2, 3)

    # ---- build_row_invoker: one-shot construction cost ---------------------
    out.append(_time_one(
        "build_row_invoker: _single_unannotated",
        lambda: build_row_invoker(_single_unannotated),
        repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "build_row_invoker: _single_annotated",
        lambda: build_row_invoker(_single_annotated),
        repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "build_row_invoker: _multi_arg",
        lambda: build_row_invoker(_multi_arg),
        repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        "build_row_invoker: _var_kw",
        lambda: build_row_invoker(_var_kw),
        repeat=repeat, inner=10_000,
    ))

    # ---- per-row dispatch (pre-built invoker) -----------------------------
    invoke_unann = build_row_invoker(_single_unannotated)
    invoke_ann = build_row_invoker(_single_annotated)
    invoke_multi = build_row_invoker(_multi_arg)
    invoke_var_kw = build_row_invoker(_var_kw)
    invoke_var_pos = build_row_invoker(_var_pos)

    out.append(_time_one(
        "invoker(single, unannotated, int row)",
        lambda: invoke_unann(int_row),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "invoker(single, annotated int, int row)",
        lambda: invoke_ann(int_row),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "invoker(single, annotated int, str row -> coerced)",
        lambda: invoke_ann(str_row),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "invoker(multi-arg, dict row spread as kwargs)",
        lambda: invoke_multi(dict_row),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "invoker(multi-arg, dict row with str id -> coerced)",
        lambda: invoke_multi(dict_row_str),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "invoker(var_kw, dict row spread)",
        lambda: invoke_var_kw(dict_row),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "invoker(var_pos, tuple row spread)",
        lambda: invoke_var_pos(tuple_row),
        repeat=repeat, inner=200_000,
    ))

    # ---- per-call check_function_args (no cache, the slower baseline) -----
    out.append(_time_one(
        "check_function_args(_multi_arg, dict_row) -- per call",
        lambda: check_function_args(_multi_arg, (), dict_row),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "check_function_args(_single_annotated, (str_row,)) -- per call",
        lambda: check_function_args(_single_annotated, (str_row,), None),
        repeat=repeat, inner=20_000,
    ))

    # ---- build_batch_invoker — vectorised vs per-row cost ----------------
    # Single positional + annotated + arg name matches a column → the
    # batch invoker should cast the column in one shot and skip the
    # dict materialisation. Per-row fallback (multi-arg / no name match)
    # quantifies the cost the vectorised path saves.
    def _by_name(id: int) -> int:
        return id + 1

    int_batch_2k = pa.RecordBatch.from_pydict({
        "id": pa.array(list(range(2_000)), type=pa.int64()),
        "name": pa.array([f"r{i}" for i in range(2_000)], type=pa.string()),
    })
    str_batch_2k = pa.RecordBatch.from_pydict({
        "id": pa.array([str(i) for i in range(2_000)], type=pa.string()),
        "name": pa.array([f"r{i}" for i in range(2_000)], type=pa.string()),
    })

    invoke_batch_typed = build_batch_invoker(_by_name)
    invoke_batch_multi = build_batch_invoker(_multi_arg)
    row_invoker_by_name = build_row_invoker(_by_name)

    out.append(_time_one(
        "build_batch_invoker: _by_name",
        lambda: build_batch_invoker(_by_name),
        repeat=repeat, inner=5_000,
    ))
    out.append(_time_one(
        "batch_invoker: single arg, int col, 2k rows (vectorised, no cast needed)",
        lambda: invoke_batch_typed(int_batch_2k),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        "batch_invoker: single arg, str col -> int, 2k rows (pa.compute.cast)",
        lambda: invoke_batch_typed(str_batch_2k),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        "batch_invoker: multi-arg (fallback to_pylist + per-row), 2k rows",
        lambda: invoke_batch_multi(int_batch_2k),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        "row_invoker baseline: 2k rows through to_pylist + per-row",
        lambda: [row_invoker_by_name(r) for r in int_batch_2k.to_pylist()],
        repeat=repeat, inner=500,
    ))

    # ---- whole-batch tabular path — one call per batch -------------------
    # ``def f(batch: pa.RecordBatch)`` is the cheapest possible apply
    # shape: the user function sees the whole batch, the invoker doesn't
    # walk row dicts. ``def f(df: pl.DataFrame)`` adds a single
    # Arrow→Polars zero-copy conversion before the call.
    def _whole_arrow_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
        return batch

    invoke_arrow = build_batch_invoker(_whole_arrow_batch)
    out.append(_time_one(
        "batch_invoker: whole pa.RecordBatch (identity), 2k rows",
        lambda: invoke_arrow(int_batch_2k),
        repeat=repeat, inner=20_000,
    ))

    def _whole_arrow_table(table: pa.Table) -> pa.Table:
        return table

    invoke_table = build_batch_invoker(_whole_arrow_table)
    out.append(_time_one(
        "batch_invoker: whole pa.Table (identity), 2k rows",
        lambda: invoke_table(int_batch_2k),
        repeat=repeat, inner=20_000,
    ))

    try:
        import polars as pl
        def _whole_polars(df: pl.DataFrame) -> pl.DataFrame:
            return df

        invoke_polars = build_batch_invoker(_whole_polars)
        out.append(_time_one(
            "batch_invoker: whole pl.DataFrame (zero-copy from Arrow), 2k rows",
            lambda: invoke_polars(int_batch_2k),
            repeat=repeat, inner=10_000,
        ))
    except ImportError:
        pass

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5,
                        help="Number of timing repeats (default: 5)")
    args = parser.parse_args()

    results = scenarios(args.repeat)
    print(f"# repeat={args.repeat}")
    print(f"# {'label':<60s}  {'best':>15}  {'median':>15}  {'mean':>15}")
    for r in results:
        print(_fmt(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
