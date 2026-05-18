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

from yggdrasil.dataclasses.safe_function import (
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
