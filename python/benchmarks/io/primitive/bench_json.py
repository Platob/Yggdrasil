"""Benchmark :class:`JSONFile` — single top-level JSON array.

JSON is the slowest write path of the primitive set (one large
top-level array serialized in one shot). Production shapes covered:
flat analytics + nested (matches the API-response ingest shape).

Usage::

    PYTHONPATH=src python benchmarks/io/primitive/bench_json.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from yggdrasil.io.primitive.json_file import JSONFile

from _common import (  # type: ignore[import-not-found]
    bench_roundtrip,
    flat_table,
    make_cli,
    nested_table,
)


def scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    out.extend(bench_roundtrip("json flat 1k", JSONFile, flat_table(1_000),
                               repeat=repeat, inner=100))
    out.extend(bench_roundtrip("json flat 50k", JSONFile, flat_table(50_000),
                               repeat=repeat, inner=5))
    out.extend(bench_roundtrip("json nested 10k", JSONFile, nested_table(10_000),
                               repeat=repeat, inner=10))
    return out


main = make_cli(scenarios)


if __name__ == "__main__":
    main()
