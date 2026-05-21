"""Benchmark the expression / predicate AST.

The AST sits between every caller that builds a ``WHERE`` /
filter clause (Tabular row filters, cache-config key building,
SQL emitters, pyarrow scanner pushdowns) and the engine that
runs it. Three cost surfaces show up in real workloads:

1. **Construction** — fluent factory + operator overloads. The
   builder allocates one frozen dataclass per node, so a deep
   tree (``c == a | c == b | c == c | c == d | ...``) is cheap
   per node but ``__post_init__`` work adds up at scale.
2. **Simplify** — :func:`yggdrasil.io.tabular.execution.expr.simplify`
   is the algebraic rewrite pass (InList dedup, OR-of-EQ collapse,
   Logical flatten, AND dedup). Pipelines that build predicates
   programmatically (chained ``|`` calls inside a loop) feed it
   long OR chains; the cost has to stay sub-µs per node or the
   "always simplify before emit" pattern stops paying off.
3. **Emit** — ``to_python`` / ``to_arrow`` / ``to_polars`` /
   ``to_sql`` walk the tree once and produce the engine-side
   form. Hot for cache-key building (the SQL emitter is the
   string the cache hashes on) and for pyarrow / polars pushdowns
   that hand the predicate to the engine per batch.
4. **Evaluate** — the Python-backend filter compiles the tree to
   a callable and runs it per row; the pyarrow backend runs the
   whole filter inside C++. Two very different cost profiles.

Scenarios target the load-bearing rewrite the OR-collapse is
designed to win: a long chain of ``c == X`` literals (the shape
pipelines actually build from a list of allowed values via
``functools.reduce(operator.or_, …)``) vs the explicit
``c.is_in([...])`` shape. Numbers should show:

- Construction cost grows linearly in chain length for both
  shapes; the InList form is roughly ``1/N`` per value (one
  dataclass + one tuple normalize) where the OR chain is
  ``N`` Comparisons + ``N-1`` Logicals.
- Simplify on the OR chain runs once, collapses to a single
  InList, and the downstream emit / evaluate cost drops
  accordingly.
- pyarrow filter throughput on the collapsed form is ``isin``
  (one kernel) vs ``equal | equal | ...`` (one kernel per
  literal then logical reduction).

Usage::

    PYTHONPATH=src python benchmarks/io/tabular/bench_predicate.py
    PYTHONPATH=src python benchmarks/io/tabular/bench_predicate.py --rows 50000 --values 32 --repeat 5
"""
from __future__ import annotations

import argparse
import functools
import operator
import statistics
import time
from typing import Callable

import pyarrow as pa
import pyarrow.dataset as pds

from yggdrasil.io.tabular.execution.expr import (
    Expression,
    col,
    simplify,
)
from yggdrasil.io.tabular.execution.expr.backends.python import (
    filter_rows,
    to_python,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_or_chain(column: str, values: list) -> Expression:
    """``c == v[0] | c == v[1] | ... | c == v[-1]`` — left-leaning OR."""
    parts = [col(column) == v for v in values]
    return functools.reduce(operator.or_, parts)


def _build_inlist(column: str, values: list) -> Expression:
    """``c IN (v0, v1, ...)``."""
    return col(column).is_in(values)


def _build_arrow_table(rows: int) -> pa.Table:
    """Sample shape: integer key, string side, float price.

    The ``id`` column cycles through 0..99 so a predicate that
    matches ``id IN (0..15)`` keeps ~16% of the rows — enough work
    that the per-batch filter cost dominates over the scanner setup.
    """
    return pa.table({
        "id": pa.array([i % 100 for i in range(rows)], type=pa.int64()),
        "side": pa.array(["buy" if i % 2 == 0 else "sell" for i in range(rows)]),
        "price": pa.array([100.0 + (i % 50) for i in range(rows)], type=pa.float64()),
    })


def _build_python_rows(rows: int) -> list[dict]:
    return [
        {"id": i % 100, "side": "buy" if i % 2 == 0 else "sell", "price": 100.0 + (i % 50)}
        for i in range(rows)
    ]


# ---------------------------------------------------------------------------
# Timing — same shape as the rest of benchmarks/.
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
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
        f"{r['label']:<70s}  "
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
        f"build: OR-chain c == v0|...|v{values_n - 1}",
        lambda: _build_or_chain("id", values),
        repeat=repeat, inner=1_000,
    ))
    out.append(_time_one(
        f"build: InList c.is_in([0..{values_n - 1}])",
        lambda: _build_inlist("id", values),
        repeat=repeat, inner=10_000,
    ))
    # Long chain with duplicates — simplify dedups it later. Cost
    # here is just the construction; the win lands in the simplify
    # row below.
    dup_values = values * 2
    out.append(_time_one(
        f"build: InList with 2× duplicates ({values_n * 2} entries)",
        lambda: _build_inlist("id", dup_values),
        repeat=repeat, inner=10_000,
    ))
    return out


def _simplify_scenarios(values_n: int, repeat: int) -> list[dict]:
    """Cost of the rewrite pass — both no-op and actual-collapse shapes."""
    values = list(range(values_n))
    or_chain = _build_or_chain("id", values)
    inlist = _build_inlist("id", values)
    duplicated = _build_inlist("id", values + values)
    or_with_isnull = or_chain | col("id").is_null()
    mixed = (
        _build_or_chain("id", values[: values_n // 2])
        | _build_or_chain("side", ["buy", "sell"])
        | col("price").is_null()
    )

    out: list[dict] = []
    out.append(_time_one(
        f"simplify: OR-chain ({values_n} eqs) → InList",
        lambda: simplify(or_chain),
        repeat=repeat, inner=1_000,
    ))
    out.append(_time_one(
        f"simplify: InList no-op ({values_n} unique values)",
        lambda: simplify(inlist),
        repeat=repeat, inner=10_000,
    ))
    out.append(_time_one(
        f"simplify: InList dedup ({values_n * 2} → {values_n})",
        lambda: simplify(duplicated),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        f"simplify: OR-chain + IsNull ({values_n} eqs)",
        lambda: simplify(or_with_isnull),
        repeat=repeat, inner=1_000,
    ))
    out.append(_time_one(
        f"simplify: mixed OR (id + side + IsNull, {values_n // 2 + 3} ops)",
        lambda: simplify(mixed),
        repeat=repeat, inner=1_000,
    ))
    return out


def _emit_scenarios(values_n: int, repeat: int) -> list[dict]:
    """``to_python`` / ``to_arrow`` / ``to_sql`` compile cost."""
    values = list(range(values_n))
    or_chain = _build_or_chain("id", values)
    inlist = _build_inlist("id", values)
    simplified = simplify(or_chain)  # should be an InList

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
    out.append(_time_one(
        f"emit: to_python(simplified OR-chain)",
        lambda: to_python(simplified),
        repeat=repeat, inner=5_000,
    ))

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
        f"emit: to_arrow(OR-chain, {values_n} eqs)",
        lambda: or_chain.to_arrow(),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"emit: to_arrow(InList, {values_n} vals)",
        lambda: inlist.to_arrow(),
        repeat=repeat, inner=5_000,
    ))

    try:
        import polars  # noqa: F401
    except ImportError:
        return out
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
    return out


def _python_eval_scenarios(rows_n: int, values_n: int, repeat: int) -> list[dict]:
    """``filter_rows`` throughput against a Python dict stream."""
    values = list(range(values_n))
    rows = _build_python_rows(rows_n)
    or_chain = _build_or_chain("id", values)
    inlist = _build_inlist("id", values)
    simplified = simplify(or_chain)

    def _drain(expr: Expression) -> None:
        for _ in filter_rows(expr, rows):
            pass

    out: list[dict] = []
    out.append(_time_one(
        f"eval: python filter_rows OR-chain rows={rows_n} vals={values_n}",
        lambda: _drain(or_chain),
        repeat=repeat, inner=5,
    ))
    out.append(_time_one(
        f"eval: python filter_rows InList rows={rows_n} vals={values_n}",
        lambda: _drain(inlist),
        repeat=repeat, inner=5,
    ))
    out.append(_time_one(
        f"eval: python filter_rows simplified-OR rows={rows_n} vals={values_n}",
        lambda: _drain(simplified),
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
    simplified_expr = simplify(_build_or_chain("id", values)).to_arrow()

    out: list[dict] = []
    out.append(_time_one(
        f"eval: arrow filter OR-chain rows={rows_n} vals={values_n}",
        lambda: ds.to_table(filter=or_chain_expr),
        repeat=repeat, inner=20,
    ))
    out.append(_time_one(
        f"eval: arrow filter InList rows={rows_n} vals={values_n}",
        lambda: ds.to_table(filter=inlist_expr),
        repeat=repeat, inner=20,
    ))
    out.append(_time_one(
        f"eval: arrow filter simplified-OR rows={rows_n} vals={values_n}",
        lambda: ds.to_table(filter=simplified_expr),
        repeat=repeat, inner=20,
    ))
    return out


def scenarios(rows_n: int, values_n: int, repeat: int) -> list[dict]:
    return [
        *_construction_scenarios(values_n, repeat),
        *_simplify_scenarios(values_n, repeat),
        *_emit_scenarios(values_n, repeat),
        *_python_eval_scenarios(rows_n, values_n, repeat),
        *_arrow_eval_scenarios(rows_n, values_n, repeat),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000,
                    help="Row count for the evaluation fixtures.")
    ap.add_argument("--values", type=int, default=16,
                    help="Number of equality literals in the OR / InList chain.")
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# rows={args.rows}  values={args.values}  repeat={args.repeat}")
    print(f"# {'label':<70s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    for row in scenarios(args.rows, args.values, args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
