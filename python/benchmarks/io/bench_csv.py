"""Benchmark :class:`CSVFile` — string-encoded format.

CSV is the codec-bound case: read + write is dominated by the
string encode/decode pass. Production shapes include flat
analytics + the wide schema where a CSV header walk drives most
of the parse cost.

Usage::

    PYTHONPATH=src python benchmarks/io/bench_csv.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from yggdrasil.io.csv_file import CSVFile

from _common import (  # type: ignore[import-not-found]
    bench_roundtrip,
    flat_table,
    make_cli,
    time_one,
    wide_table,
)


def scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    # CSV doesn't support nested types — skip the nested fixture so
    # this bench stays clean.
    out.extend(bench_roundtrip("csv flat 1k", CSVFile, flat_table(1_000),
                               repeat=repeat, inner=100))
    out.extend(bench_roundtrip("csv flat 50k", CSVFile, flat_table(50_000),
                               repeat=repeat, inner=20))
    out.extend(bench_roundtrip("csv wide 32x10k", CSVFile, wide_table(10_000),
                               repeat=repeat, inner=20))

    sink = CSVFile(b"")
    sink.write_arrow_table(flat_table(50_000))
    sink.seek(0)
    payload = sink.read()
    out.append(time_one(
        "csv: collect_schema flat 50k",
        lambda: CSVFile(payload).collect_schema(),
        repeat=repeat, inner=500,
    ))
    return out


main = make_cli(scenarios)


if __name__ == "__main__":
    main()
