"""Shared helpers for the per-format primitive-IO benches.

Every ``bench_<format>.py`` in this directory follows the same shape:

* representative analytics tables (flat + nested + wide) built once;
* MATCH / projection-friendly tables for the production scenarios;
* a ``_bench_leaf`` helper that drives write + read across arrow /
  polars / pandas for a given :class:`Tabular` leaf class.

Splitting the runner here keeps the per-format files lean and the
behavior consistent across formats.
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time
from typing import Callable

import pyarrow as pa


# ---------------------------------------------------------------------------
# Tables — kept module-level so build cost doesn't show up in scenarios.
# ---------------------------------------------------------------------------


def flat_table(rows: int) -> pa.Table:
    """6-column analytics shape — int / float / string / ts / bool."""
    return pa.table(
        {
            "id": pa.array(range(rows), type=pa.int64()),
            "amount": pa.array([1.5] * rows, type=pa.float64()),
            "qty": pa.array([2] * rows, type=pa.int32()),
            "name": pa.array(["row-" + str(i % 100) for i in range(rows)],
                             type=pa.string()),
            "ts": pa.array([dt.datetime(2024, 1, 1)] * rows,
                           type=pa.timestamp("us")),
            "active": pa.array([(i % 2 == 0) for i in range(rows)],
                               type=pa.bool_()),
        }
    )


def nested_table(rows: int) -> pa.Table:
    """Nested analytics shape — list / map / struct columns alongside scalars."""
    return pa.table(
        {
            "id": pa.array(range(rows), type=pa.int64()),
            "tags": pa.array([["alpha", "beta"]] * rows,
                             type=pa.list_(pa.string())),
            "attrs": pa.array([[("k", "v")]] * rows,
                              type=pa.map_(pa.string(), pa.string())),
            "address": pa.array(
                [{"street": "1", "city": "x", "zip": "00"}] * rows,
                type=pa.struct([
                    ("street", pa.string()),
                    ("city", pa.string()),
                    ("zip", pa.string()),
                ]),
            ),
        }
    )


def wide_table(rows: int, cols: int = 32) -> pa.Table:
    """Many-column shape — covers projection-pushdown scenarios."""
    return pa.table({
        f"c{i:02d}": pa.array([i] * rows, type=pa.int64())
        for i in range(cols)
    })


# ---------------------------------------------------------------------------
# Timing helpers.
# ---------------------------------------------------------------------------


def time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    """Warm one call, then ``inner`` calls per ``repeat`` outer iterations."""
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


def fmt(r: dict) -> str:
    scale = 1e6
    unit = "us"
    if r["best"] < 1e-6:
        scale, unit = 1e9, "ns"
    elif r["best"] >= 1e-3:
        scale, unit = 1e3, "ms"
    return (
        f"{r['label']:<66s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Per-leaf runner — shared across every primitive-format bench.
# ---------------------------------------------------------------------------


def bench_roundtrip(
    label: str,
    leaf_cls: type,
    table: pa.Table,
    *,
    repeat: int,
    inner: int,
    frameworks: tuple[str, ...] = ("arrow", "polars", "pandas"),
) -> list[dict]:
    """Write + read benchmark for a single leaf class, per framework."""
    sink = leaf_cls(b"")
    sink.write_arrow_table(table)
    sink.seek(0)
    payload = sink.read()
    rows = table.num_rows

    out: list[dict] = []

    if "arrow" in frameworks:
        def write_arrow():
            leaf_cls(b"").write_arrow_table(table)
        def read_arrow():
            leaf_cls(payload).read_arrow_table()
        out.append(time_one(
            f"{label}: arrow write_arrow_table rows={rows}",
            write_arrow, repeat=repeat, inner=inner,
        ))
        out.append(time_one(
            f"{label}: arrow read_arrow_table rows={rows}",
            read_arrow, repeat=repeat, inner=inner,
        ))

    if "polars" in frameworks:
        try:
            import polars  # noqa: F401
            pl_frame = leaf_cls(payload).read_polars_frame()

            def write_polars():
                leaf_cls(b"").write_polars_frame(pl_frame)
            def read_polars():
                leaf_cls(payload).read_polars_frame()

            out.append(time_one(
                f"{label}: polars write_polars_frame rows={rows}",
                write_polars, repeat=repeat, inner=inner,
            ))
            out.append(time_one(
                f"{label}: polars read_polars_frame rows={rows}",
                read_polars, repeat=repeat, inner=inner,
            ))
        except ImportError:
            pass
        except Exception as e:
            out.append({
                "label": f"{label}: polars SKIPPED ({type(e).__name__}: {e})",
                "best": 0.0, "median": 0.0, "mean": 0.0,
            })

    if "pandas" in frameworks:
        try:
            import pandas  # noqa: F401
            pd_frame = leaf_cls(payload).read_pandas_frame()

            def write_pandas():
                leaf_cls(b"").write_pandas_frame(pd_frame)
            def read_pandas():
                leaf_cls(payload).read_pandas_frame()

            out.append(time_one(
                f"{label}: pandas write_pandas_frame rows={rows}",
                write_pandas, repeat=repeat, inner=inner,
            ))
            out.append(time_one(
                f"{label}: pandas read_pandas_frame rows={rows}",
                read_pandas, repeat=repeat, inner=inner,
            ))
        except ImportError:
            pass
        except Exception as e:
            out.append({
                "label": f"{label}: pandas SKIPPED ({type(e).__name__}: {e})",
                "best": 0.0, "median": 0.0, "mean": 0.0,
            })

    return out


# ---------------------------------------------------------------------------
# CLI helper.
# ---------------------------------------------------------------------------


def make_cli(scenarios_fn: Callable[[int], list[dict]]) -> Callable[[], None]:
    """Build a ``main()`` that takes ``--repeat`` and prints scenarios."""
    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--repeat", type=int, default=3,
                        help="Outer repeat count per scenario (median across).")
        args = ap.parse_args()

        print(f"# repeat={args.repeat}")
        print(f"# {'label':<66s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
        for row in scenarios_fn(args.repeat):
            print(fmt(row))
    return main
