"""Benchmark the cast registry + engine cast hot paths.

Why this exists
---------------

Every dataframe handoff between engines goes through one of two
places:

* :func:`yggdrasil.data.cast.registry.convert` — the registry that
  dispatches by ``(from_hint, to_hint)``, walking exact → identity →
  ``Any`` wildcard → MRO fallback → one-hop composition.
* :meth:`CastOptions.cast_arrow_tabular` / ``cast_polars_tabular`` /
  ``cast_pandas_tabular`` — the engine entry points that hand frames
  to the per-field type kernels.

A folder-of-folders read with N batches × M columns pays both costs
on every batch handoff: the registry dispatch decides whether any
real cast is needed, the engine path runs the actual kernel (or the
``need_cast`` short-circuit when source / target already line up).
This benchmark measures both so the registry-side wins don't hide
behind the much larger per-batch engine cost — and vice versa.

Usage::

    PYTHONPATH=src python benchmarks/bench_cast.py
    PYTHONPATH=src python benchmarks/bench_cast.py --rows 100000 --repeat 5
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.data import Field
from yggdrasil.data.cast.registry import (
    convert,
    find_converter,
    register_converter,
)
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _arrow_table(rows: int, *, mismatch: bool = False) -> pa.Table:
    """Build a representative wire table.

    ``mismatch=False`` keeps every column type identical to the target —
    every per-field cast short-circuits via ``need_cast`` → engine-type
    bypass. ``mismatch=True`` widens ``id`` from int32 → int64 so the
    cast kernel actually fires; useful for measuring the engine-side
    cost the bypass saves.
    """
    id_type = pa.int32() if mismatch else pa.int64()
    return pa.table(
        {
            "id": pa.array(range(rows), type=id_type),
            "amount": pa.array([1.5] * rows, type=pa.float64()),
            "qty": pa.array([2] * rows, type=pa.int32()),
            "name": pa.array(["x"] * rows, type=pa.string()),
            "ts": pa.array([0] * rows, type=pa.timestamp("us")),
            "active": pa.array([True] * rows, type=pa.bool_()),
        }
    )


TARGET_SCHEMA = Schema.from_fields(
    [
        Field("id", "int64", nullable=False),
        Field("amount", "float64"),
        Field("qty", "int32"),
        Field("name", "string"),
        Field("ts", "timestamp(us)"),
        Field("active", "bool"),
    ]
)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 200)):
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
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _arrow_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    opts = CastOptions(target=TARGET_SCHEMA)

    match_table = _arrow_table(rows, mismatch=False)
    mismatch_table = _arrow_table(rows, mismatch=True)

    # Match path — every column already matches; the cast site
    # short-circuits via the engine-type bypass.
    out.append(_time_one(
        f"arrow: cast_arrow_tabular MATCH rows={rows}",
        lambda: opts.cast_arrow_tabular(match_table),
        repeat=repeat, inner=200,
    ))
    # Mismatch path — int32 → int64 cast kernel fires on the id col.
    out.append(_time_one(
        f"arrow: cast_arrow_tabular CAST rows={rows}",
        lambda: opts.cast_arrow_tabular(mismatch_table),
        repeat=repeat, inner=200,
    ))
    # opts.cast(table) — the public DataIO entry point. Hits the
    # registry dispatch before the engine entry.
    out.append(_time_one(
        f"arrow: opts.cast(table) MATCH rows={rows}",
        lambda: opts.cast(match_table),
        repeat=repeat, inner=200,
    ))
    return out


def _polars_scenarios(rows: int, repeat: int) -> list[dict]:
    try:
        import polars as pl
    except ImportError:
        return [{"label": "polars: not installed — skipped",
                 "best": 0.0, "median": 0.0, "mean": 0.0}]

    out: list[dict] = []
    opts = CastOptions(target=TARGET_SCHEMA)

    # Build a polars DataFrame whose schema already matches the target.
    # ``ts`` keeps datetime64[us] to stay on the engine-type bypass path.
    arrow_match = _arrow_table(rows, mismatch=False)
    arrow_cast = _arrow_table(rows, mismatch=True)
    df_match = pl.from_arrow(arrow_match)
    df_cast = pl.from_arrow(arrow_cast)

    out.append(_time_one(
        f"polars: cast_polars_tabular MATCH rows={rows}",
        lambda: opts.cast_polars_tabular(df_match),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"polars: cast_polars_tabular CAST rows={rows}",
        lambda: opts.cast_polars_tabular(df_cast),
        repeat=repeat, inner=200,
    ))
    # Registry entry — ``convert(df, pl.DataFrame, options=opts)`` is
    # the path third-party callers hit.
    out.append(_time_one(
        f"polars: convert(df, pl.DataFrame) MATCH rows={rows}",
        lambda: convert(df_match, pl.DataFrame, options=opts),
        repeat=repeat, inner=200,
    ))
    # Series cast — common per-column cast in pipeline code. Scoped
    # to a single Field target so the cast site lands on the column
    # itself, not the multi-column Schema.
    series_match = df_match["id"]
    series_cast = df_cast["id"]
    id_field = TARGET_SCHEMA.field_by("id")
    opts_id = CastOptions(target=id_field)
    out.append(_time_one(
        f"polars: cast_polars_series MATCH rows={rows}",
        lambda: opts_id.cast_polars_series(series_match),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"polars: cast_polars_series CAST rows={rows}",
        lambda: opts_id.cast_polars_series(series_cast),
        repeat=repeat, inner=200,
    ))
    return out


def _registry_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # Identity dispatch — same type both sides.
    out.append(_time_one(
        "registry: convert('x', str)  identity",
        lambda: convert("x", str),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "registry: convert(1, int)  identity",
        lambda: convert(1, int),
        repeat=repeat, inner=200_000,
    ))

    # Primitive registered conversions.
    out.append(_time_one(
        "registry: convert('42', int)",
        lambda: convert("42", int),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "registry: convert(42, str)",
        lambda: convert(42, str),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "registry: convert('1.5', float)",
        lambda: convert("1.5", float),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        "registry: convert('true', bool)",
        lambda: convert("true", bool),
        repeat=repeat, inner=50_000,
    ))

    # Dispatch resolution — pure lookup cost, no kernel.
    out.append(_time_one(
        "registry: find_converter(str, int)",
        lambda: find_converter(str, int),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "registry: find_converter(int, str)",
        lambda: find_converter(int, str),
        repeat=repeat, inner=200_000,
    ))

    # Cache-preservation cost — measure how many warm hits survive a new
    # registration (the targeted invalidation only drops None entries and the
    # exact registered key, not the whole cache).  We register a private
    # sentinel pair, then time a fully-warmed convert() call.  If the whole
    # cache were cleared each time, the subsequent convert() would re-walk
    # MRO/namespace for every warm pair; with targeted clearing the existing
    # hits stay intact and the cost stays at a single dict lookup.
    class _BenchSrc:
        pass

    class _BenchDst:
        pass

    @register_converter(_BenchSrc, _BenchDst)
    def _bench_noop(v: _BenchSrc, opts: None) -> _BenchDst:
        return _BenchDst()

    # Warm the cache for the pairs we care about.
    find_converter(str, int)
    find_converter(int, str)
    find_converter(str, float)

    def _hit_after_registration():
        @register_converter(_BenchSrc, _BenchDst)
        def _inner(v: _BenchSrc, opts: None) -> _BenchDst:
            return _BenchDst()
        # Immediately re-warm and look up a cached pair.
        return find_converter(str, int)

    out.append(_time_one(
        "registry: register + warm find_converter (targeted inval.)",
        _hit_after_registration,
        repeat=repeat, inner=5_000,
    ))

    return out


def scenarios(rows: int, repeat: int) -> list[dict]:
    return [
        *_registry_scenarios(repeat),
        *_arrow_scenarios(rows, repeat),
        *_polars_scenarios(rows, repeat),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000,
                    help="Per-batch row count for engine scenarios.")
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# rows={args.rows}  repeat={args.repeat}")
    print(f"# {'label':<60s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    for row in scenarios(args.rows, args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
