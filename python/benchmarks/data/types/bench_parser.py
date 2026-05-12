"""Benchmark :mod:`yggdrasil.data.types.parser` — lex + parse hot path.

Why this exists
---------------

``parse_data_type`` is the front door for every ``DataType.from_str``
call site and the JSON branch of ``DataType.from_dict``. Schema
ingestion (Spark / Databricks ``SHOW CREATE TABLE`` output, Arrow
schemas, Polars / pandas frame round-trips, FastAPI request payloads)
hits it once per field and the same handful of spellings — ``int``,
``bigint``, ``string``, ``timestamp_ntz``, ``array<struct<...>>`` —
repeat across thousands of columns.

The parser already memoizes successful parses through
:func:`_parse_cached`, so two costs matter:

1. **Cache hit** — a frozen string keyed into an ``lru_cache``. This
   is the steady-state cost on a warm pipeline.
2. **Cache miss** — full lex + recursive-descent parse. This fires on
   every novel spelling and on every cold start of a worker process.

Both shapes show up in the scenario list below. Cache-miss timings
clear the LRU cache before each call so we measure the parser, not
the dict lookup.

Usage::

    PYTHONPATH=src python benchmarks/data/types/bench_parser.py
    PYTHONPATH=src python benchmarks/data/types/bench_parser.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

from yggdrasil.data.types.parser import (
    _parse_cached,
    parse_data_type,
)


# ---------------------------------------------------------------------------
# Inputs — grouped by complexity so the report reads top-to-bottom from
# the cheapest single-token spellings to the deepest nested DDL.
# ---------------------------------------------------------------------------


SIMPLE_TOKENS = [
    "int", "bigint", "smallint", "tinyint",
    "int32", "int64", "uint32", "uint64",
    "float", "double", "float32", "float64",
    "bool", "boolean",
    "string", "varchar", "text",
    "binary", "bytes",
    "date", "time", "timestamp",
    "json", "jsonb",
    "null",
]

FAST_PATH_TOKENS = [
    "timestamp_ns", "timestamp_us", "timestamp_ms", "timestamp_s",
    "timestamp_ntz", "timestamp_ltz",
    "datetime_us", "duration_ns", "interval_year_month",
    "date32", "date64", "time32_ms", "time64_ns",
]

BRACKET_METADATA = [
    "decimal(10, 2)",
    "decimal(38, 18)",
    "varchar(255)",
    "timestamp[unit=us, tz=UTC]",
    "int32[nullable=false]",
    "string[encoding=utf8, nullable=true]",
]

NESTED = [
    "array<int>",
    "array<string>",
    "array<array<int>>",
    "map<string, int>",
    "map<string, struct<a: int, b: string>>",
    "struct<a: int, b: string, c: double>",
    "struct<id: bigint, name: string, meta: map<string, string>>",
    "array<struct<a: int, b: string>>",
]

PYTHON_TYPING = [
    "list[int]",
    "dict[str, int]",
    "Optional[int]",
    "Union[int, str, None]",
    "int | str | None",
    "tuple[int, str, float]",
]

POSTFIX_ARRAY = [
    "int[]",
    "text[][]",
    "struct<a:int>[]",
]

DEEPLY_NESTED = (
    "array<struct<"
    "id: bigint, "
    "name: string, "
    "tags: array<string>, "
    "meta: map<string, struct<k: string, v: int, t: timestamp_us>>, "
    "scores: array<array<double>>"
    ">>"
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


def _time_cold(
    label: str,
    fn: Callable[[], None],
    *,
    repeat: int,
    inner: int,
) -> dict:
    """Time ``fn`` with the parse LRU cache cleared before each call.

    Measures the lex + parse cost in isolation. We pay one dict
    ``cache_clear`` per inner iteration; that overhead is constant
    across scenarios and small relative to a real parse, so it shows
    up as a uniform floor rather than skewing the comparison.
    """
    for _ in range(min(inner, 100)):
        _parse_cached.cache_clear()
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            _parse_cached.cache_clear()
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
        f"{r['label']:<58s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios.
# ---------------------------------------------------------------------------


def _sweep(inputs: list[str]) -> Callable[[], None]:
    """Build a closure that parses every string in ``inputs`` once."""
    def run() -> None:
        for s in inputs:
            parse_data_type(s)
    return run


def _sweep_cold(inputs: list[str]) -> Callable[[], None]:
    """Same as :func:`_sweep` but for use inside :func:`_time_cold`."""
    return _sweep(inputs)


def scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []

    # ------------------------------------------------------------------
    # Cache-hit path — the steady state.
    # ------------------------------------------------------------------

    # Warm the cache for every fixture before measuring hit-path scenarios.
    for batch in (
        SIMPLE_TOKENS, FAST_PATH_TOKENS, BRACKET_METADATA,
        NESTED, PYTHON_TYPING, POSTFIX_ARRAY, [DEEPLY_NESTED],
    ):
        for s in batch:
            parse_data_type(s)

    results.append(_time_one(
        "cached: parse_data_type('int')",
        lambda: parse_data_type("int"),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "cached: parse_data_type('bigint')",
        lambda: parse_data_type("bigint"),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "cached: parse_data_type('timestamp_ntz')",
        lambda: parse_data_type("timestamp_ntz"),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "cached: parse_data_type('array<struct<...>>')",
        lambda: parse_data_type(DEEPLY_NESTED),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "cached: sweep over 25 simple tokens",
        _sweep(SIMPLE_TOKENS),
        repeat=repeat, inner=10_000,
    ))

    # ------------------------------------------------------------------
    # Cache-miss path — the cold-start / novel-spelling cost.
    # ------------------------------------------------------------------

    results.append(_time_cold(
        "cold:   parse_data_type('int')",
        lambda: parse_data_type("int"),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_cold(
        "cold:   parse_data_type('bigint')",
        lambda: parse_data_type("bigint"),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_cold(
        "cold:   parse_data_type('timestamp_ntz')",
        lambda: parse_data_type("timestamp_ntz"),
        repeat=repeat, inner=20_000,
    ))
    results.append(_time_cold(
        "cold:   parse_data_type('decimal(10, 2)')",
        lambda: parse_data_type("decimal(10, 2)"),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_cold(
        "cold:   parse_data_type('varchar(255)')",
        lambda: parse_data_type("varchar(255)"),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_cold(
        "cold:   parse_data_type('array<int>')",
        lambda: parse_data_type("array<int>"),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_cold(
        "cold:   parse_data_type('map<string, int>')",
        lambda: parse_data_type("map<string, int>"),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_cold(
        "cold:   parse_data_type('struct<a:int,b:string,c:double>')",
        lambda: parse_data_type("struct<a: int, b: string, c: double>"),
        repeat=repeat, inner=5_000,
    ))
    results.append(_time_cold(
        "cold:   parse_data_type('Union[int,str,None]')",
        lambda: parse_data_type("Union[int, str, None]"),
        repeat=repeat, inner=5_000,
    ))
    results.append(_time_cold(
        "cold:   parse_data_type('int | str | None')",
        lambda: parse_data_type("int | str | None"),
        repeat=repeat, inner=5_000,
    ))
    results.append(_time_cold(
        "cold:   parse_data_type('int[]')",
        lambda: parse_data_type("int[]"),
        repeat=repeat, inner=10_000,
    ))
    results.append(_time_cold(
        "cold:   parse_data_type(deeply_nested)",
        lambda: parse_data_type(DEEPLY_NESTED),
        repeat=repeat, inner=2_000,
    ))

    # ------------------------------------------------------------------
    # Realistic batch — 25 simple + 13 fast-path + 6 bracket + 8 nested
    # + 6 python-typing + 3 postfix-array, all cold. Mirrors a schema
    # parse on a worker that hasn't seen these types yet.
    # ------------------------------------------------------------------

    all_inputs = (
        SIMPLE_TOKENS + FAST_PATH_TOKENS + BRACKET_METADATA
        + NESTED + PYTHON_TYPING + POSTFIX_ARRAY
    )
    results.append(_time_cold(
        f"cold:   sweep over {len(all_inputs)} mixed inputs",
        _sweep_cold(all_inputs),
        repeat=repeat, inner=500,
    ))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<58s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
