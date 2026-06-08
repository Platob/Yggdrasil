"""Benchmark the :mod:`yggdrasil.io.curation` hot paths.

Why this exists
---------------

The curation pipeline runs at integration boundaries (CSV reads, HTTP
JSON bodies, Power Query payloads), so anything that costs more than
"one pyarrow.compute pass over the column" is paying for itself in
real ingest latency. Three load-bearing dimensions to measure:

* **Per-column rule order on ``StringCurator``** — every column pays
  the cost of the trials that fired before the winning one. Integer
  columns hit ``_try_bool`` first (cheap), then ``_try_int`` (the
  match). Date / time / timestamp inputs walk further. We bench each
  shape so a regression in the early trials shows up against the
  baseline.
* **Tabular dispatch** — ``Curator.curate_arrow_tabular`` runs
  ``Curator.pick`` per column. That walk visits every subclass in
  the tree; a few extra subclasses or extra ``handles`` work per
  column adds up quickly on wide tables.
* **Numeric shrinkers** — ``IntegerCurator`` / ``FloatCurator`` run
  ``pc.min_max`` (int) or a round-trip equality check (float). For a
  table with N numeric columns the cost dominates if the kernel is
  invoked redundantly.

Usage::

    PYTHONPATH=src python benchmarks/io/curation/bench_curation.py
    PYTHONPATH=src python benchmarks/io/curation/bench_curation.py --rows 100000 --repeat 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.io.curation import (
    Curator,
    FloatCurator,
    IntegerCurator,
    NestedCurator,
    StringCurator,
)


# ============================================================ fixtures


def _string_int_column(rows: int) -> pa.Array:
    """Pure integer strings — most common CSV ingest shape."""
    return pa.array([str(i) for i in range(rows)], type=pa.string())


def _string_float_column(rows: int) -> pa.Array:
    return pa.array([f"{i}.5" for i in range(rows)], type=pa.string())


def _string_bool_column(rows: int) -> pa.Array:
    return pa.array(
        ["true" if i % 2 == 0 else "false" for i in range(rows)],
        type=pa.string(),
    )


def _string_iso_date_column(rows: int) -> pa.Array:
    return pa.array(
        [f"2024-01-{(i % 28) + 1:02d}" for i in range(rows)],
        type=pa.string(),
    )


def _string_iso_timestamp_column(rows: int) -> pa.Array:
    return pa.array(
        [
            f"2024-01-{(i % 28) + 1:02d}T10:30:00+02:00"
            for i in range(rows)
        ],
        type=pa.string(),
    )


def _string_label_column(rows: int) -> pa.Array:
    return pa.array(
        [f"label-{i}" for i in range(rows)], type=pa.string()
    )


def _string_table(rows: int) -> pa.Table:
    return pa.table(
        {
            "id": _string_int_column(rows),
            "amount": _string_float_column(rows),
            "flag": _string_bool_column(rows),
            "when": _string_iso_timestamp_column(rows),
            "label": _string_label_column(rows),
        }
    )


def _int64_column(rows: int) -> pa.Array:
    return pa.array(range(rows), type=pa.int64())


def _float64_column(rows: int) -> pa.Array:
    return pa.array([i + 0.5 for i in range(rows)], type=pa.float64())


def _struct_of_strings(rows: int) -> pa.Array:
    return pa.StructArray.from_arrays(
        [_string_int_column(rows), _string_float_column(rows)],
        names=["id", "amount"],
    )


def _list_of_strings(rows: int) -> pa.Array:
    values = _string_int_column(rows * 4)
    offsets = pa.array(
        [i * 4 for i in range(rows + 1)], type=pa.int32()
    )
    return pa.ListArray.from_arrays(offsets, values)


# ====================================================== timing helpers


def _time_one(
    label: str, fn: Callable[[], None], *, repeat: int, inner: int
) -> dict:
    # Warm-up — JIT-like effects in pyarrow.compute kernels, dispatch
    # caches in the cast registry, frozen-dataclass first-touch.
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
        scale = 1e9
        unit = "ns"
    return (
        f"{r['label']:<70s}  "
        f"best={r['best'] * scale:9.2f} {unit}  "
        f"median={r['median'] * scale:9.2f} {unit}  "
        f"mean={r['mean'] * scale:9.2f} {unit}"
    )


# ============================================================ scenarios


def _string_curator_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    curator = StringCurator()
    inner = 50 if rows >= 100_000 else 200

    int_arr = _string_int_column(rows)
    out.append(
        _time_one(
            f"StringCurator: int strings rows={rows}",
            lambda: curator.curate(int_arr),
            repeat=repeat,
            inner=inner,
        )
    )

    float_arr = _string_float_column(rows)
    out.append(
        _time_one(
            f"StringCurator: float strings rows={rows}",
            lambda: curator.curate(float_arr),
            repeat=repeat,
            inner=inner,
        )
    )

    bool_arr = _string_bool_column(rows)
    out.append(
        _time_one(
            f"StringCurator: bool strings rows={rows}",
            lambda: curator.curate(bool_arr),
            repeat=repeat,
            inner=inner,
        )
    )

    date_arr = _string_iso_date_column(rows)
    out.append(
        _time_one(
            f"StringCurator: ISO date strings rows={rows}",
            lambda: curator.curate(date_arr),
            repeat=repeat,
            inner=inner,
        )
    )

    ts_arr = _string_iso_timestamp_column(rows)
    out.append(
        _time_one(
            f"StringCurator: ISO timestamp strings rows={rows}",
            lambda: curator.curate(ts_arr),
            repeat=repeat,
            inner=inner,
        )
    )

    label_arr = _string_label_column(rows)
    out.append(
        _time_one(
            f"StringCurator: free strings (string fallback) rows={rows}",
            lambda: curator.curate(label_arr),
            repeat=repeat,
            inner=inner,
        )
    )

    return out


def _tabular_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    inner = 20 if rows >= 100_000 else 100

    table = _string_table(rows)
    out.append(
        _time_one(
            f"curate_arrow_tabular: 5-col string table rows={rows}",
            lambda: Curator.curate_arrow_tabular(table),
            repeat=repeat,
            inner=inner,
        )
    )

    mixed = pa.table(
        {
            "id": _int64_column(rows),
            "score": _float64_column(rows),
            "label": _string_label_column(rows),
        }
    )
    out.append(
        _time_one(
            f"curate_arrow_tabular: pretyped int+float+string rows={rows}",
            lambda: Curator.curate_arrow_tabular(mixed),
            repeat=repeat,
            inner=inner,
        )
    )

    return out


def _shrinker_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    inner = 200 if rows >= 100_000 else 1000

    int_arr = _int64_column(rows)
    int_curator = IntegerCurator()
    out.append(
        _time_one(
            f"IntegerCurator: int64 rows={rows}",
            lambda: int_curator.curate(int_arr),
            repeat=repeat,
            inner=inner,
        )
    )

    float_arr = _float64_column(rows)
    float_curator = FloatCurator()
    out.append(
        _time_one(
            f"FloatCurator: float64 rows={rows}",
            lambda: float_curator.curate(float_arr),
            repeat=repeat,
            inner=inner,
        )
    )

    return out


def _nested_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    inner = 50 if rows >= 10_000 else 200

    struct = _struct_of_strings(rows)
    out.append(
        _time_one(
            f"NestedCurator: struct<str, str> rows={rows}",
            lambda: NestedCurator().curate(struct),
            repeat=repeat,
            inner=inner,
        )
    )

    lst = _list_of_strings(rows)
    out.append(
        _time_one(
            f"NestedCurator: list<str> rows={rows} (x4 values)",
            lambda: NestedCurator().curate(lst),
            repeat=repeat,
            inner=inner,
        )
    )

    return out


def _pick_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    # Per-column dispatch cost on tabular. Cheap by itself, dominant
    # on wide tables.
    str_arr = pa.array(["x"])
    int_arr = pa.array([1], type=pa.int64())
    list_arr = pa.array([["a"]])

    out.append(
        _time_one(
            "Curator.pick(string array)",
            lambda: Curator.pick(str_arr),
            repeat=repeat,
            inner=10_000,
        )
    )
    out.append(
        _time_one(
            "Curator.pick(int64 array)",
            lambda: Curator.pick(int_arr),
            repeat=repeat,
            inner=10_000,
        )
    )
    out.append(
        _time_one(
            "Curator.pick(list array)",
            lambda: Curator.pick(list_arr),
            repeat=repeat,
            inner=10_000,
        )
    )
    return out


# =========================================================== entrypoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    print(f"=== curation bench  rows={args.rows} repeat={args.repeat} ===")
    print()
    scenarios: list[dict] = []
    scenarios.extend(_string_curator_scenarios(args.rows, args.repeat))
    scenarios.extend(_tabular_scenarios(args.rows, args.repeat))
    scenarios.extend(_shrinker_scenarios(args.rows, args.repeat))
    scenarios.extend(_nested_scenarios(args.rows, args.repeat))
    scenarios.extend(_pick_scenarios(args.repeat))
    for row in scenarios:
        print(_fmt(row))


if __name__ == "__main__":
    main()
