"""Benchmark the expression / predicate AST end-to-end.

The AST sits between every caller that builds a ``WHERE`` /
filter clause (Tabular row filters, cache-config key building,
SQL emitters, pyarrow / polars / spark / pandas pushdowns) and the
engine that runs it. Four cost surfaces matter in real workloads:

1. **Construction** — fluent factory + operator overloads. Each node
   is a frozen dataclass; ``InList.__post_init__`` dedupes
   ``values`` and ``Logical.__post_init__`` flattens same-op
   nesting, so the tree the caller hands downstream is already
   canonical. The benchmarks below quantify the cost of those
   construction-time rewrites against the do-nothing baseline
   (a raw ``InList`` with unique values, a single-level
   ``Logical``).
2. **Emit** — ``to_python`` / ``to_arrow`` / ``to_polars`` /
   ``to_sql`` / ``to_pyspark`` walk the tree once and produce
   the engine-side form. Hot for cache-key building (the SQL
   emitter is the string the cache hashes on) and for pyarrow /
   polars pushdowns that hand the predicate to the engine per batch.
3. **Engine filter** — the new ``filter_<engine>`` helpers on
   :class:`Predicate` apply the predicate to the engine's native
   representation. We benchmark the same predicate against an
   Arrow Table, a Polars DataFrame, a Pandas DataFrame, a pylist
   of dicts, a pydict of columns, and the generic ``filter``
   dispatcher so the numbers reveal the per-engine cost profile.
4. **Cast inside the predicate** — ``Cast(col, dtype)`` leaves run
   the cast kernel before the comparison. Arrow / polars / spark
   fuse the cast into the predicate so the filter result keeps the
   original (un-cast) column values; the benchmark contrasts a
   no-cast predicate vs the equivalent casted predicate so the
   fused-cast cost is visible per engine.

Optional engines (polars / pandas / pyspark) are skipped when
unavailable. ``--engines`` lets the caller scope a run to a subset
(e.g. ``--engines arrow,polars``).

Usage::

    PYTHONPATH=src python benchmarks/io/tabular/bench_predicate.py
    PYTHONPATH=src python benchmarks/io/tabular/bench_predicate.py --rows 50000 --values 32 --repeat 5
    PYTHONPATH=src python benchmarks/io/tabular/bench_predicate.py --engines arrow,polars
"""
from __future__ import annotations

import argparse
import functools
import operator
import statistics
import time
from typing import Any, Callable, Iterable

import pyarrow as pa
import pyarrow.dataset as pds

from yggdrasil.saga.expr import (
    Expression,
    Predicate,
    col,
)
from yggdrasil.saga.expr.backends.python import (
    filter_rows,
    to_python,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_or_chain(column: str, values: list) -> Expression:
    """``c == v[0] | c == v[1] | ... | c == v[-1]`` — left-leaning OR.

    ``Logical.__post_init__`` flattens the chain into one operand
    tuple regardless of associativity, so the cost shown is one
    flatten pass per build.
    """
    parts = [col(column) == v for v in values]
    return functools.reduce(operator.or_, parts)


def _build_inlist(column: str, values: list) -> Expression:
    return col(column).is_in(values)


def _build_arrow_table(rows: int) -> pa.Table:
    return pa.table(
        {
            "id": list(range(rows)),
            "side": ["buy" if i % 2 == 0 else "sell" for i in range(rows)],
            "price": [100.0 + (i % 50) for i in range(rows)],
            "as_string": [str(i % 100) for i in range(rows)],
        }
    )


def _build_python_rows(rows: int) -> list[dict]:
    return [
        {
            "id": i % 100,
            "side": "buy" if i % 2 == 0 else "sell",
            "price": 100.0 + (i % 50),
            "as_string": str(i % 100),
        }
        for i in range(rows)
    ]


def _build_pydict(rows: int) -> dict:
    return {
        "id": [i % 100 for i in range(rows)],
        "side": ["buy" if i % 2 == 0 else "sell" for i in range(rows)],
        "price": [100.0 + (i % 50) for i in range(rows)],
        "as_string": [str(i % 100) for i in range(rows)],
    }


# ---------------------------------------------------------------------------
# Timing — same shape as the rest of benchmarks/.
# ---------------------------------------------------------------------------


def _time_one(
    label: str,
    fn: Callable[[], None],
    *,
    repeat: int,
    inner: int,
) -> dict:
    # Warm-up — match the rest of the suite's "small warmup, then
    # repeat timed loops, take median across" pattern.
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
        f"{r['label']:<75s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _construction_scenarios(values_n: int, repeat: int) -> list[dict]:
    """Cost of allocating the AST nodes themselves."""
    values = list(range(values_n))
    out: list[dict] = []
    out.append(_time_one(
        f"build: OR-chain c == v0 | ... | v{values_n - 1}",
        lambda: _build_or_chain("id", values),
        repeat=repeat, inner=1_000,
    ))
    out.append(_time_one(
        f"build: InList c.is_in([0..{values_n - 1}])",
        lambda: _build_inlist("id", values),
        repeat=repeat, inner=10_000,
    ))
    # __post_init__ dedupes the duplicate-laden value list the caller
    # built by concatenating per-batch keys. The number here is the
    # construction + dedup combined; compare against the no-dup
    # InList row above to read the dedup overhead.
    dup_values = values * 2
    out.append(_time_one(
        f"build: InList 2× dups ({values_n * 2} → {values_n})",
        lambda: _build_inlist("id", dup_values),
        repeat=repeat, inner=10_000,
    ))
    # AND-of-InList is the canonical shape for primary-key / cache-key
    # lookups; build cost should be near-zero compared to the OR-chain
    # since the flatten pass only sees two operands.
    out.append(_time_one(
        "build: AND(InList, InList)",
        lambda: _build_inlist("id", values) & _build_inlist("side", ["buy", "sell"]),
        repeat=repeat, inner=10_000,
    ))
    return out


def _emit_scenarios(values_n: int, repeat: int, engines: set[str]) -> list[dict]:
    """``to_python`` / ``to_arrow`` / ``to_polars`` / ``to_sql`` /
    ``to_pyspark`` compile cost."""
    values = list(range(values_n))
    or_chain = _build_or_chain("id", values)
    inlist = _build_inlist("id", values)
    and_inlist = inlist & _build_inlist("side", ["buy", "sell"])

    out: list[dict] = []
    out.append(_time_one(
        f"emit: to_python(OR-chain, {values_n} eqs)",
        lambda: to_python(or_chain),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"emit: to_python(InList, {values_n} vals)",
        lambda: to_python(inlist),
        repeat=repeat, inner=5_000,
    ))
    if "sql" in engines:
        out.append(_time_one(
            f"emit: to_sql(OR-chain, {values_n} eqs)",
            lambda: or_chain.to_sql(),
            repeat=repeat, inner=500,
        ))
        out.append(_time_one(
            f"emit: to_sql(InList, {values_n} vals)",
            lambda: inlist.to_sql(),
            repeat=repeat, inner=5_000,
        ))
        out.append(_time_one(
            "emit: to_sql(AND(InList, InList))",
            lambda: and_inlist.to_sql(),
            repeat=repeat, inner=5_000,
        ))
    if "arrow" in engines:
        out.append(_time_one(
            f"emit: to_arrow(OR-chain, {values_n} eqs)",
            lambda: or_chain.to_arrow(),
            repeat=repeat, inner=500,
        ))
        out.append(_time_one(
            f"emit: to_arrow(InList, {values_n} vals)",
            lambda: inlist.to_arrow(),
            repeat=repeat, inner=5_000,
        ))
    if "polars" in engines:
        try:
            import polars  # noqa: F401
        except ImportError:
            engines.discard("polars")
        else:
            out.append(_time_one(
                f"emit: to_polars(OR-chain, {values_n} eqs)",
                lambda: or_chain.to_polars(),
                repeat=repeat, inner=500,
            ))
            out.append(_time_one(
                f"emit: to_polars(InList, {values_n} vals)",
                lambda: inlist.to_polars(),
                repeat=repeat, inner=5_000,
            ))
    if "spark" in engines:
        try:
            import pyspark  # noqa: F401
        except ImportError:
            engines.discard("spark")
        else:
            out.append(_time_one(
                f"emit: to_pyspark(InList, {values_n} vals)",
                lambda: inlist.to_pyspark(),
                repeat=repeat, inner=500,
            ))
    return out


def _engine_filter_scenarios(
    rows_n: int,
    values_n: int,
    repeat: int,
    engines: set[str],
) -> list[dict]:
    """``Predicate.filter_<engine>`` throughput across engines.

    Same predicate (an ``AND(InList, InList)`` shape — the typical
    cache-lookup clause) is run against every engine's native
    representation so the numbers read as a single table.
    """
    values = list(range(values_n))
    pred: Predicate = (
        _build_inlist("id", values) & _build_inlist("side", ["buy", "sell"])
    )  # type: ignore[assignment]

    out: list[dict] = []
    if "arrow" in engines:
        table = _build_arrow_table(rows_n)
        batch = table.to_batches()[0]
        out.append(_time_one(
            f"filter: arrow_table rows={rows_n} vals={values_n}",
            lambda: pred.filter_arrow_table(table),
            repeat=repeat, inner=20,
        ))
        out.append(_time_one(
            f"filter: arrow_batch rows={rows_n} vals={values_n}",
            lambda: pred.filter_arrow_batch(batch),
            repeat=repeat, inner=20,
        ))
        # ``filter_arrow_batches`` adds streaming overhead — measure
        # at smaller batch sizes that real callers actually hand it.
        small_batches = [
            table.slice(i, 256).to_batches()[0]
            for i in range(0, min(rows_n, 1024), 256)
        ]
        out.append(_time_one(
            f"filter: arrow_batches batches={len(small_batches)} of 256 rows",
            lambda: list(pred.filter_arrow_batches(small_batches)),
            repeat=repeat, inner=200,
        ))
    if "polars" in engines:
        try:
            import polars as pl
        except ImportError:
            engines.discard("polars")
        else:
            pframe = pl.from_arrow(_build_arrow_table(rows_n))
            out.append(_time_one(
                f"filter: polars_frame rows={rows_n} vals={values_n}",
                lambda: pred.filter_polars_frame(pframe),
                repeat=repeat, inner=20,
            ))
            lazy = pframe.lazy()
            out.append(_time_one(
                f"filter: polars_frame (lazy) rows={rows_n} vals={values_n}",
                lambda: pred.filter_polars_frame(lazy).collect(),
                repeat=repeat, inner=10,
            ))
    if "pandas" in engines:
        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            engines.discard("pandas")
        else:
            pddf = _build_arrow_table(rows_n).to_pandas()
            out.append(_time_one(
                f"filter: pandas_frame rows={rows_n} vals={values_n}",
                lambda: pred.filter_pandas_frame(pddf),
                repeat=repeat, inner=5,
            ))
    rows = _build_python_rows(rows_n)
    pydict = _build_pydict(rows_n)
    out.append(_time_one(
        f"filter: pylist rows={rows_n} vals={values_n}",
        lambda: pred.filter_pylist(rows),
        repeat=repeat, inner=5,
    ))
    out.append(_time_one(
        f"filter: pydict rows={rows_n} vals={values_n}",
        lambda: pred.filter_pydict(pydict),
        repeat=repeat, inner=20,
    ))
    # Iterable path — same data as pylist, expressed as a generator,
    # to surface the per-call dispatch overhead.
    out.append(_time_one(
        f"filter: iterable rows={rows_n} vals={values_n}",
        lambda: list(pred.filter_iterable(rows)),
        repeat=repeat, inner=5,
    ))
    return out


def _python_eval_scenarios(rows_n: int, values_n: int, repeat: int) -> list[dict]:
    """Lower-level ``filter_rows`` throughput against a Python dict stream.

    ``filter_rows`` is the legacy entry point (now also reachable as
    ``predicate.filter_iterable(rows)``); benchmarking both surfaces
    against the same data lets a reviewer spot dispatch overhead in
    the wrapper.
    """
    values = list(range(values_n))
    rows = _build_python_rows(rows_n)
    or_chain = _build_or_chain("id", values)
    inlist = _build_inlist("id", values)

    def _drain(expr: Expression) -> None:
        for _ in filter_rows(expr, rows):
            pass

    out: list[dict] = []
    out.append(_time_one(
        f"eval: filter_rows OR-chain rows={rows_n} vals={values_n}",
        lambda: _drain(or_chain),
        repeat=repeat, inner=5,
    ))
    out.append(_time_one(
        f"eval: filter_rows InList rows={rows_n} vals={values_n}",
        lambda: _drain(inlist),
        repeat=repeat, inner=5,
    ))
    return out


def _arrow_eval_scenarios(rows_n: int, values_n: int, repeat: int) -> list[dict]:
    """``Dataset.to_table(filter=...)`` — runs the predicate in C++."""
    values = list(range(values_n))
    table = _build_arrow_table(rows_n)
    ds = pds.dataset(table)

    or_chain_expr = _build_or_chain("id", values).to_arrow()
    inlist_expr = _build_inlist("id", values).to_arrow()

    out: list[dict] = []
    out.append(_time_one(
        f"eval: arrow dataset OR-chain rows={rows_n} vals={values_n}",
        lambda: ds.to_table(filter=or_chain_expr),
        repeat=repeat, inner=20,
    ))
    out.append(_time_one(
        f"eval: arrow dataset InList rows={rows_n} vals={values_n}",
        lambda: ds.to_table(filter=inlist_expr),
        repeat=repeat, inner=20,
    ))
    return out


def _cast_scenarios(
    rows_n: int,
    values_n: int,
    repeat: int,
    engines: set[str],
) -> list[dict]:
    """Cast-inside-predicate cost: ``col(as_string).cast(int) IN (...)``.

    Arrow / polars fuse the cast into the predicate kernel and apply
    the resulting mask to the original (un-cast) column — the row
    that survives still carries the original string value. The
    numbers contrast the casted predicate against the equivalent
    no-cast predicate so the fused-cast cost is visible per engine.
    """
    from yggdrasil.data.data_field import Field
    from yggdrasil.data.types.primitive import Int64Type, StringType

    values = list(range(values_n))
    no_cast = _build_inlist("id", values)
    # Bind the string source explicitly — the smart ``Expression.cast``
    # only wraps in a real :class:`Cast` node when the source dtype
    # is known to differ from the target. A bare ``col("as_string")``
    # would have ``ObjectType`` and the factory would *replace* the
    # type instead of casting, defeating the benchmark.
    string_col = col(Field(name="as_string", dtype=StringType()))
    casted = string_col.cast(Int64Type()).is_in(values)

    out: list[dict] = []
    if "arrow" in engines:
        table = _build_arrow_table(rows_n)
        out.append(_time_one(
            f"cast: arrow no-cast rows={rows_n} vals={values_n}",
            lambda: no_cast.filter_arrow_table(table),
            repeat=repeat, inner=20,
        ))
        out.append(_time_one(
            f"cast: arrow with cast rows={rows_n} vals={values_n}",
            lambda: casted.filter_arrow_table(table),
            repeat=repeat, inner=20,
        ))
    if "polars" in engines:
        try:
            import polars as pl
        except ImportError:
            engines.discard("polars")
        else:
            pframe = pl.from_arrow(_build_arrow_table(rows_n))
            out.append(_time_one(
                f"cast: polars no-cast rows={rows_n} vals={values_n}",
                lambda: no_cast.filter_polars_frame(pframe),
                repeat=repeat, inner=20,
            ))
            out.append(_time_one(
                f"cast: polars with cast rows={rows_n} vals={values_n}",
                lambda: casted.filter_polars_frame(pframe),
                repeat=repeat, inner=20,
            ))
    return out


def _temporal_scenarios(
    rows_n: int,
    repeat: int,
    engines: set[str],
) -> list[dict]:
    """Tz-pushdown cost: filter UTC arrow column by Paris-tz literal.

    The optimisation: convert the literal to the column's native tz
    once at filter time and let the engine compare against the
    storage values directly, instead of casting every row through
    a tz kernel before the comparison.

    Three scenarios to read together:

    - ``baseline``: column carries no bound :class:`Field`; the
      smart ``cast`` factory swallows the cast and pyarrow's tz-
      aware compare runs natively (no rewrite, but no per-row
      cast either — pyarrow does the conversion in the kernel).
    - ``cast wrap``: column carries a UTC :class:`Field`, so
      ``cast(Paris)`` wraps in a real :class:`Cast` node. The
      construction-time pushdown unwraps it and converts the
      literal to UTC.
    - ``target rewrite``: column carries a Paris-claiming
      :class:`Field` (intentionally mismatched), but the target's
      Arrow schema reports UTC. The filter-time rewrite firing
      against the target's schema is what makes the filter pick
      up the right rows.
    """
    if "arrow" not in engines:
        return []
    import datetime as _dt
    import zoneinfo
    import pyarrow as pa

    from yggdrasil.data.data_field import Field
    from yggdrasil.data.types.primitive.temporal import TimestampType

    ts_arr = pa.array(
        [
            _dt.datetime(2026, 1, 1, h, 0, tzinfo=_dt.timezone.utc)
            for h in range(rows_n % 24 or 24)
        ]
        * max(1, rows_n // 24),
        type=pa.timestamp("us", tz="UTC"),
    )[:rows_n]
    table = pa.table({"ts": ts_arr})
    paris_dt = _dt.datetime(
        2026, 1, 1, 12, 0, tzinfo=zoneinfo.ZoneInfo("Europe/Paris"),
    )

    out: list[dict] = []

    # 1. No bound field — the smart cast swallows the cast.
    p_no_field = col("ts").cast(TimestampType(tz="Europe/Paris")) == paris_dt
    out.append(_time_one(
        f"tz: arrow filter (no-field cast) rows={rows_n}",
        lambda: p_no_field.filter_arrow_table(table),
        repeat=repeat, inner=20,
    ))

    # 2. UTC-bound field — construction-time pushdown fires.
    utc_field = Field(name="ts", dtype=TimestampType(tz="UTC"))
    p_utc_field = (
        col("ts", field=utc_field).cast(TimestampType(tz="Europe/Paris"))
        == paris_dt
    )
    out.append(_time_one(
        f"tz: arrow filter (UTC-field cast) rows={rows_n}",
        lambda: p_utc_field.filter_arrow_table(table),
        repeat=repeat, inner=20,
    ))

    # 3. Paris-claiming field on a UTC target — filter-time rewrite
    # is the only way to fix the mismatch.
    paris_field = Field(name="ts", dtype=TimestampType(tz="Europe/Paris"))
    p_paris_field = col("ts", field=paris_field) == paris_dt
    out.append(_time_one(
        f"tz: arrow filter (target schema rewrite) rows={rows_n}",
        lambda: p_paris_field.filter_arrow_table(table),
        repeat=repeat, inner=20,
    ))

    # 4. Polars: predicate with Paris-claim Field against UTC frame.
    #    Polars refuses to compare two different tz-aware columns
    #    without the rewrite — this benchmark would error out
    #    without the bare-tz pushdown.
    if "polars" in engines:
        try:
            import polars as pl
        except ImportError:
            engines.discard("polars")
        else:
            pframe = pl.from_arrow(table)
            out.append(_time_one(
                f"tz: polars filter (target schema rewrite) rows={rows_n}",
                lambda: p_paris_field.filter_polars_frame(pframe),
                repeat=repeat, inner=20,
            ))

    return out

    out: list[dict] = []
    if "arrow" in engines:
        table = _build_arrow_table(rows_n)
        out.append(_time_one(
            f"cast: arrow no-cast rows={rows_n} vals={values_n}",
            lambda: no_cast.filter_arrow_table(table),
            repeat=repeat, inner=20,
        ))
        out.append(_time_one(
            f"cast: arrow with cast rows={rows_n} vals={values_n}",
            lambda: casted.filter_arrow_table(table),
            repeat=repeat, inner=20,
        ))
    if "polars" in engines:
        try:
            import polars as pl
        except ImportError:
            engines.discard("polars")
        else:
            pframe = pl.from_arrow(_build_arrow_table(rows_n))
            out.append(_time_one(
                f"cast: polars no-cast rows={rows_n} vals={values_n}",
                lambda: no_cast.filter_polars_frame(pframe),
                repeat=repeat, inner=20,
            ))
            out.append(_time_one(
                f"cast: polars with cast rows={rows_n} vals={values_n}",
                lambda: casted.filter_polars_frame(pframe),
                repeat=repeat, inner=20,
            ))
    return out


def _dispatch_scenarios(
    rows_n: int,
    values_n: int,
    repeat: int,
    engines: set[str],
) -> list[dict]:
    """``Predicate.filter`` dispatcher overhead against each engine.

    Measures the wrapper cost of going through the generic dispatcher
    vs calling the specific ``filter_<engine>`` directly. Should be
    flat (one ``isinstance`` chain per call).
    """
    values = list(range(values_n))
    pred = _build_inlist("id", values) & _build_inlist("side", ["buy", "sell"])

    out: list[dict] = []
    table = _build_arrow_table(rows_n)
    out.append(_time_one(
        f"dispatch: filter(arrow_table) rows={rows_n}",
        lambda: pred.filter(table),
        repeat=repeat, inner=20,
    ))
    pydict = _build_pydict(rows_n)
    out.append(_time_one(
        f"dispatch: filter(pydict) rows={rows_n}",
        lambda: pred.filter(pydict),
        repeat=repeat, inner=20,
    ))
    rows = _build_python_rows(rows_n)
    out.append(_time_one(
        f"dispatch: filter(pylist) rows={rows_n}",
        lambda: pred.filter(rows),
        repeat=repeat, inner=5,
    ))
    if "polars" in engines:
        try:
            import polars as pl
        except ImportError:
            engines.discard("polars")
        else:
            pframe = pl.from_arrow(table)
            out.append(_time_one(
                f"dispatch: filter(polars_frame) rows={rows_n}",
                lambda: pred.filter(pframe),
                repeat=repeat, inner=20,
            ))
    return out


def scenarios(
    rows_n: int,
    values_n: int,
    repeat: int,
    engines: set[str],
) -> list[dict]:
    return [
        *_construction_scenarios(values_n, repeat),
        *_emit_scenarios(values_n, repeat, engines),
        *_engine_filter_scenarios(rows_n, values_n, repeat, engines),
        *_python_eval_scenarios(rows_n, values_n, repeat),
        *(_arrow_eval_scenarios(rows_n, values_n, repeat) if "arrow" in engines else []),
        *_cast_scenarios(rows_n, values_n, repeat, engines),
        *_temporal_scenarios(rows_n, repeat, engines),
        *_dispatch_scenarios(rows_n, values_n, repeat, engines),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_ALL_ENGINES = {"arrow", "sql", "polars", "pandas", "spark"}


def _parse_engines(arg: str) -> set[str]:
    if not arg:
        return set(_ALL_ENGINES)
    selected = {e.strip().lower() for e in arg.split(",") if e.strip()}
    bad = selected - _ALL_ENGINES
    if bad:
        raise SystemExit(
            f"Unknown engines: {sorted(bad)}. Valid: {sorted(_ALL_ENGINES)}"
        )
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000,
                    help="Row count for the evaluation fixtures.")
    ap.add_argument("--values", type=int, default=16,
                    help="Number of equality literals in the OR / InList chain.")
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    ap.add_argument("--engines", type=str, default="",
                    help=f"Comma-separated engine subset; one of {sorted(_ALL_ENGINES)}.")
    args = ap.parse_args()

    engines = _parse_engines(args.engines)

    print(
        f"# rows={args.rows}  values={args.values}  repeat={args.repeat}  "
        f"engines={sorted(engines)}"
    )
    print(f"# {'label':<75s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    for row in scenarios(args.rows, args.values, args.repeat, engines):
        print(_fmt(row))


if __name__ == "__main__":
    main()
