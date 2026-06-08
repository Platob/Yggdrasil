"""Benchmark :class:`NDJSONFile` — newline-delimited JSON.

NDJson is per-row encode-cost dominated: serializing 50k rows is
expensive, but reading via :mod:`pyarrow.json` is fast (vectorised).
Production shapes covered: flat analytics + nested (NDJson is the
common ingest format for JSON-shaped event streams).

Usage::

    PYTHONPATH=src python benchmarks/io/bench_ndjson.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from yggdrasil.io.ndjson_file import NDJSONFile

from _common import (  # type: ignore[import-not-found]
    bench_roundtrip,
    flat_table,
    make_cli,
    nested_table,
)


def scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    out.extend(bench_roundtrip("ndjson flat 1k", NDJSONFile, flat_table(1_000),
                               repeat=repeat, inner=100))
    out.extend(bench_roundtrip("ndjson flat 50k", NDJSONFile, flat_table(50_000),
                               repeat=repeat, inner=10))
    out.extend(bench_roundtrip("ndjson nested 10k", NDJSONFile, nested_table(10_000),
                               repeat=repeat, inner=20))
    return out


main = make_cli(scenarios)


if __name__ == "__main__":
    main()
