"""Benchmark the :mod:`yggdrasil.data.cast.registry` hot paths.

Why this exists
---------------

Every ``convert(value, target)`` call funnels through
:func:`find_converter` first to pick a converter, then through the
chosen converter to produce the result. Both halves run on the hot
path of every cast — per-row coercion in
:func:`yggdrasil.dataclasses.safe_function.check_function_args`, per-cell
arrow → python conversion, per-frame cast through ``CastOptions``,
and per-batch tabular dispatch in
:func:`yggdrasil.dataclasses.build_batch_invoker`. A 10% regression
in either half is a 10% regression in every apply pipeline downstream.

This bench covers the seven dispatch steps from :func:`find_converter`
(``registry.py`` lines 198-205) plus the most common :func:`convert`
outcomes — identity, registered primitive, ``Optional`` unwrap,
container generic, enum, dataclass, cross-engine tabular — so a
regression in any one of them lands in a visible number rather than
a slow ``apply``.

Cache hit vs cold are tracked separately because the steady-state
of any long-running pipeline is cache-hit: the cold numbers tell
you the worst case (first call after process start), the warm
numbers tell you what production sees.

Usage::

    PYTHONPATH=src python benchmarks/data/bench_registry.py
    PYTHONPATH=src python benchmarks/data/bench_registry.py --repeat 7
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import enum
import statistics
import time
from typing import Any, Callable, Optional

import pyarrow as pa

from yggdrasil.data.cast import convert
from yggdrasil.data.cast.registry import _find_cache, find_converter


# ---------------------------------------------------------------------------
# Timing helpers — same shape as the other ``bench_*`` files so output is
# scannable in a side-by-side comparison.
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
        scale, unit = 1e9, "ns"
    return (
        f"{r['label']:<62s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


class _Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclasses.dataclass
class _Row:
    id: int
    name: str
    score: float = 0.0


class _Animal:
    pass


class _Dog(_Animal):
    pass


# ---------------------------------------------------------------------------
# find_converter — the seven dispatch steps from registry.py.
# Each step is exercised cold (cache invalidated) and warm (cache hit) so
# the steady-state vs first-call cost both have visible numbers.
# ---------------------------------------------------------------------------


def _find_converter_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # Pre-warm everything we'll probe so the "warm" runs hit the cache.
    pre_warm_pairs = [
        (str, int), (int, str), (str, float), (str, bool),
        (str, dt.date), (str, dt.datetime),
        (str, str), (int, int), (Any, int), (int, Any),
        (_Dog, _Animal),
        (pa.RecordBatch, pa.Table),
        (pa.RecordBatch, pa.RecordBatch),
        (int, _Color),
    ]
    for f, t in pre_warm_pairs:
        find_converter(f, t)

    # Step 1: exact registry hit — the steady-state hot path. Pure dict
    # lookup after the first call (cache hit).
    out.append(_time_one(
        "find_converter: exact str->int (cache hit)",
        lambda: find_converter(str, int),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "find_converter: exact int->str (cache hit)",
        lambda: find_converter(int, str),
        repeat=repeat, inner=500_000,
    ))

    # Step 2: identity (from == to or to is Any/object).
    out.append(_time_one(
        "find_converter: identity str->str (cache hit)",
        lambda: find_converter(str, str),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "find_converter: identity int->int (cache hit)",
        lambda: find_converter(int, int),
        repeat=repeat, inner=500_000,
    ))

    # Step 3: Any -> T wildcard.
    out.append(_time_one(
        "find_converter: Any->int (wildcard)",
        lambda: find_converter(Any, int),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "find_converter: int->Any (identity-ish)",
        lambda: find_converter(int, Any),
        repeat=repeat, inner=500_000,
    ))

    # Step 5: MRO walk — derived class to its base.
    out.append(_time_one(
        "find_converter: MRO walk _Dog->_Animal (cache hit)",
        lambda: find_converter(_Dog, _Animal),
        repeat=repeat, inner=500_000,
    ))

    # Cold path — invalidate the cache between calls so each iteration
    # walks the full dispatch chain. Quantifies the first-call cost users
    # see right after process start.
    def _cold_str_int():
        _find_cache.pop((str, int), None)
        return find_converter(str, int)

    out.append(_time_one(
        "find_converter: exact str->int (cache cold)",
        _cold_str_int,
        repeat=repeat, inner=100_000,
    ))

    def _cold_identity():
        _find_cache.pop((str, str), None)
        return find_converter(str, str)

    out.append(_time_one(
        "find_converter: identity str->str (cache cold)",
        _cold_identity,
        repeat=repeat, inner=100_000,
    ))

    def _cold_mro():
        _find_cache.pop((_Dog, _Animal), None)
        return find_converter(_Dog, _Animal)

    out.append(_time_one(
        "find_converter: MRO walk _Dog->_Animal (cache cold)",
        _cold_mro,
        repeat=repeat, inner=50_000,
    ))

    # Step 4: namespace-triggered late imports (polars / pandas / pyspark
    # / pyarrow). Warm only — the imports are one-shot side-effects so
    # the "cold" measurement is dominated by import time, not the lookup.
    out.append(_time_one(
        "find_converter: pa.RecordBatch->pa.Table (cache hit)",
        lambda: find_converter(pa.RecordBatch, pa.Table),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "find_converter: pa.RecordBatch->pa.RecordBatch (identity)",
        lambda: find_converter(pa.RecordBatch, pa.RecordBatch),
        repeat=repeat, inner=500_000,
    ))

    try:
        import polars as pl
        find_converter(pa.RecordBatch, pl.DataFrame)  # pre-warm
        out.append(_time_one(
            "find_converter: pa.RecordBatch->pl.DataFrame (cache hit)",
            lambda: find_converter(pa.RecordBatch, pl.DataFrame),
            repeat=repeat, inner=500_000,
        ))
    except ImportError:
        pass

    try:
        import pandas as pd
        find_converter(pa.RecordBatch, pd.DataFrame)  # pre-warm
        out.append(_time_one(
            "find_converter: pa.RecordBatch->pd.DataFrame (cache hit)",
            lambda: find_converter(pa.RecordBatch, pd.DataFrame),
            repeat=repeat, inner=500_000,
        ))
    except ImportError:
        pass

    # Negative path — "no converter" answer also caches and should stay
    # cheap on repeats. ``pa.RecordBatch -> int`` is the canonical miss
    # the batch invoker uses to decide between whole-batch vs per-row.
    find_converter(pa.RecordBatch, int)  # pre-warm
    out.append(_time_one(
        "find_converter: pa.RecordBatch->int  (miss, cache hit)",
        lambda: find_converter(pa.RecordBatch, int),
        repeat=repeat, inner=500_000,
    ))

    return out


# ---------------------------------------------------------------------------
# convert — the major dispatch outcomes covered by registry.convert.
# ---------------------------------------------------------------------------


def _convert_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # Identity / passthrough — value already matches the target.
    out.append(_time_one(
        "convert: identity 'x'->str",
        lambda: convert("x", str),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "convert: identity 42->int",
        lambda: convert(42, int),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "convert: identity 1.5->float",
        lambda: convert(1.5, float),
        repeat=repeat, inner=200_000,
    ))

    # ``Any`` / ``object`` target — no conversion attempted.
    out.append(_time_one(
        "convert: value->Any (passthrough)",
        lambda: convert("hello", Any),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "convert: value->object (passthrough)",
        lambda: convert(42, object),
        repeat=repeat, inner=500_000,
    ))

    # ``None`` handling: Optional unwrap vs default_scalar.
    out.append(_time_one(
        "convert: None->Optional[int]  (returns None)",
        lambda: convert(None, Optional[int]),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "convert: None->int  (returns default_scalar)",
        lambda: convert(None, int),
        repeat=repeat, inner=100_000,
    ))

    # Registered primitive conversions — the meat of every coercion call.
    out.append(_time_one(
        "convert: '42'->int",
        lambda: convert("42", int),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "convert: '1.5'->float",
        lambda: convert("1.5", float),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "convert: 'true'->bool",
        lambda: convert("true", bool),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "convert: 42->str",
        lambda: convert(42, str),
        repeat=repeat, inner=200_000,
    ))

    # Temporal — string ISO → date / datetime / time.
    out.append(_time_one(
        "convert: '2024-06-01'->date",
        lambda: convert("2024-06-01", dt.date),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "convert: '2024-06-01T12:30:45'->datetime",
        lambda: convert("2024-06-01T12:30:45", dt.datetime),
        repeat=repeat, inner=50_000,
    ))

    # Enum resolution — string member name / value.
    out.append(_time_one(
        "convert: 'red'->_Color (enum)",
        lambda: convert("red", _Color),
        repeat=repeat, inner=100_000,
    ))

    # Dataclass from mapping.
    row_dict = {"id": 5, "name": "x", "score": 1.5}
    out.append(_time_one(
        "convert: dict->_Row (dataclass)",
        lambda: convert(row_dict, _Row),
        repeat=repeat, inner=50_000,
    ))

    # Container generics — list / dict / set.
    src_list = ["1", "2", "3", "4", "5"]
    out.append(_time_one(
        "convert: list[str]->list[int]  (5 elems)",
        lambda: convert(src_list, list[int]),
        repeat=repeat, inner=50_000,
    ))
    src_dict = {"a": "1", "b": "2", "c": "3"}
    out.append(_time_one(
        "convert: dict[str,str]->dict[str,int]  (3 pairs)",
        lambda: convert(src_dict, dict[str, int]),
        repeat=repeat, inner=20_000,
    ))

    # Tabular: cross-engine via the registry.
    rb = pa.RecordBatch.from_pydict({
        "id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
        "name": pa.array(["a", "b", "c", "d", "e"], type=pa.string()),
    })
    out.append(_time_one(
        "convert: pa.RecordBatch->pa.RecordBatch  (identity)",
        lambda: convert(rb, pa.RecordBatch),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "convert: pa.RecordBatch->pa.Table",
        lambda: convert(rb, pa.Table),
        repeat=repeat, inner=50_000,
    ))

    try:
        import polars as pl
        out.append(_time_one(
            "convert: pa.RecordBatch->pl.DataFrame  (zero-copy)",
            lambda: convert(rb, pl.DataFrame),
            repeat=repeat, inner=20_000,
        ))
    except ImportError:
        pass

    try:
        import pandas as pd
        out.append(_time_one(
            "convert: pa.RecordBatch->pd.DataFrame",
            lambda: convert(rb, pd.DataFrame),
            repeat=repeat, inner=5_000,
        ))
    except ImportError:
        pass

    return out


def scenarios(repeat: int) -> list[dict]:
    return [
        *_find_converter_scenarios(repeat),
        *_convert_scenarios(repeat),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5,
                        help="Number of timing repeats (default: 5)")
    args = parser.parse_args()

    results = scenarios(args.repeat)
    print(f"# repeat={args.repeat}")
    print(f"# {'label':<62s}  {'best':>15}  {'median':>15}  {'mean':>15}")
    for r in results:
        print(_fmt(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
