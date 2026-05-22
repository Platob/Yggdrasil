"""Benchmark :class:`ArrowTabular` reads / writes under the
already-usable :class:`CastOptions` knobs.

The Tabular surface is the canonical handoff between every read /
write pipeline in the codebase. Whenever a pipeline binds a
``target`` schema, a ``row_size`` / ``byte_size`` rechunk cap, or
both, the cost shows up here — once per batch, on top of whatever
the format reader / writer is already paying.

Scenarios target the three knobs callers already reach for:

* **target cast** (MATCH bypass vs real CAST) on ``read_arrow_table``
  / ``read_arrow_batches`` / ``write_arrow_table`` — the cast site
  short-circuits via the engine-type bypass when source / target
  already line up, and runs the per-field kernel otherwise.
* **row_size rechunk** — zero-copy slicing into fixed-row chunks,
  applied at the seam via :meth:`CastOptions.cast_arrow_batch_iterator`
  or the standalone :func:`rechunk_arrow_batches` helper.
* **byte_size rechunk** — concat-then-slice to a target byte budget;
  derives the row target from the per-segment bytes/row ratio.

Plus the ``ArrowTabular`` ingest paths (``pa.Table``, list of
``RecordBatch``, ``list[dict]``) and :meth:`collect_schema`, since
both feed every pipeline that walks an in-memory holder.

Usage::

    PYTHONPATH=src python benchmarks/io/tabular/bench_tabular.py
    PYTHONPATH=src python benchmarks/io/tabular/bench_tabular.py --rows 50000 --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time
from typing import Callable, Iterable

import pyarrow as pa

from yggdrasil.arrow.cast import rechunk_arrow_batches, rechunk_arrow_table
from yggdrasil.data import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.arrow.tabular import ArrowTabular


# ---------------------------------------------------------------------------
# Fixtures — built once at module scope so build cost stays out of timing.
# ---------------------------------------------------------------------------


def _arrow_table(rows: int, *, mismatch: bool = False) -> pa.Table:
    """6-column analytics shape; ``mismatch`` widens id from int32→int64."""
    id_type = pa.int32() if mismatch else pa.int64()
    return pa.table(
        {
            "id": pa.array(range(rows), type=id_type),
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


def _split_into_batches(table: pa.Table, batch_rows: int) -> list[pa.RecordBatch]:
    """Slice *table* into fixed-row record batches.

    Used to seed multi-batch holders — the rechunker only has work to
    do when the source isn't already shaped the way the caller asked
    for.
    """
    out: list[pa.RecordBatch] = []
    total = table.num_rows
    for offset in range(0, total, batch_rows):
        chunk = table.slice(offset, batch_rows)
        out.extend(chunk.to_batches())
    return out


# ---------------------------------------------------------------------------
# Timing helpers — same shape as the rest of benchmarks/.
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
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
        f"{r['label']:<66s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


def _drain(it: Iterable) -> None:
    for _ in it:
        pass


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _ingest_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    table = _arrow_table(rows)
    batches = _split_into_batches(table, batch_rows=max(1, rows // 8))
    rows_pylist = table.to_pylist()  # endpoint-only — diagnostic fixture
    pydict = {name: col.to_pylist() for name, col in zip(table.column_names, table.columns)}

    # Fresh RecordBatchReader per call — readers consume on iteration.
    def _make_reader() -> pa.RecordBatchReader:
        return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())

    # Sibling source so the multi-input shape has two distinct tables
    # to concat (matches the typical "drain N statement results into
    # one holder" pattern).
    sibling = _arrow_table(rows)
    seed_io = ArrowTabular(table)  # reused as a Tabular source

    out.append(_time_one(
        f"ingest: ArrowTabular(table) rows={rows}",
        lambda: ArrowTabular(table),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        f"ingest: ArrowTabular(table, table, table) rows={rows}",
        lambda: ArrowTabular(table, sibling, table),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        f"ingest: ArrowTabular(list[RecordBatch]) rows={rows}",
        lambda: ArrowTabular(batches),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        f"ingest: ArrowTabular(RecordBatchReader) rows={rows}",
        lambda: ArrowTabular(_make_reader()),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        f"ingest: ArrowTabular(ArrowTabular) rows={rows}",
        lambda: ArrowTabular(seed_io),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        f"ingest: ArrowTabular(list[dict]) rows={rows}",
        lambda: ArrowTabular(rows_pylist),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"ingest: ArrowTabular(dict[str, list]) rows={rows}",
        lambda: ArrowTabular(pydict),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        "ingest: ArrowTabular() empty",
        lambda: ArrowTabular(),
        repeat=repeat, inner=50_000,
    ))
    return out


def _read_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    match_table = _arrow_table(rows, mismatch=False)
    cast_table = _arrow_table(rows, mismatch=True)
    io_match = ArrowTabular(match_table)
    io_cast = ArrowTabular(cast_table)

    opts_none = CastOptions()
    opts_match = CastOptions(target=TARGET_SCHEMA)
    # Same target schema; combined with a mismatched source forces the
    # id column through the int32 → int64 cast kernel on every batch.
    opts_cast = CastOptions(target=TARGET_SCHEMA)

    out.append(_time_one(
        f"read: read_arrow_table no-target rows={rows}",
        lambda: io_match.read_arrow_table(opts_none),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"read: read_arrow_table MATCH rows={rows}",
        lambda: io_match.read_arrow_table(opts_match),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"read: read_arrow_table CAST rows={rows}",
        lambda: io_cast.read_arrow_table(opts_cast),
        repeat=repeat, inner=200,
    ))

    out.append(_time_one(
        f"read: read_arrow_batches no-target rows={rows}",
        lambda: _drain(io_match.read_arrow_batches(opts_none)),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"read: read_arrow_batches MATCH rows={rows}",
        lambda: _drain(io_match.read_arrow_batches(opts_match)),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"read: read_arrow_batches CAST rows={rows}",
        lambda: _drain(io_cast.read_arrow_batches(opts_cast)),
        repeat=repeat, inner=200,
    ))

    # collect_schema — every downstream caller that needs the holder's
    # shape (Tabular.lazy(), Tabular.write_table, schema diffing) hits it.
    out.append(_time_one(
        "read: collect_schema()",
        lambda: io_match.collect_schema(),
        repeat=repeat, inner=20_000,
    ))

    # Spilled-state read: holder backed by an mmap'd Arrow IPC file
    # (the path the ArrowIPCFile spill writer produces). Confirms the
    # zero-copy read-back path doesn't pay an extra concat / cast hop.
    io_spilled = ArrowTabular(match_table, spill_bytes=1)
    out.append(_time_one(
        f"read: read_arrow_table spilled no-target rows={rows}",
        lambda: io_spilled.read_arrow_table(opts_none),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"read: read_arrow_table spilled MATCH rows={rows}",
        lambda: io_spilled.read_arrow_table(opts_match),
        repeat=repeat, inner=500,
    ))
    return out


def _write_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    match_table = _arrow_table(rows, mismatch=False)
    cast_table = _arrow_table(rows, mismatch=True)

    opts_none = CastOptions()
    opts_match = CastOptions(target=TARGET_SCHEMA)
    opts_cast = CastOptions(target=TARGET_SCHEMA)

    def _write(table: pa.Table, opts: CastOptions) -> None:
        sink = ArrowTabular()
        sink.write_arrow_table(table, opts)

    out.append(_time_one(
        f"write: write_arrow_table no-target rows={rows}",
        lambda: _write(match_table, opts_none),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"write: write_arrow_table MATCH rows={rows}",
        lambda: _write(match_table, opts_match),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"write: write_arrow_table CAST rows={rows}",
        lambda: _write(cast_table, opts_cast),
        repeat=repeat, inner=200,
    ))

    # write_pandas_frame — exercises the arrow-oriented schema hint
    # path on :meth:`_write_pandas_frame`. With a bound target,
    # ``pa.Table.from_pandas`` consumes the target's arrow schema and
    # the downstream cast collapses to its bypass.
    try:
        import pandas as pd  # noqa: F401

        pandas_frame = match_table.to_pandas()

        def _write_pandas(frame, opts: CastOptions) -> None:
            sink = ArrowTabular()
            sink.write_pandas_frame(frame, opts)

        out.append(_time_one(
            f"write: write_pandas_frame no-target rows={rows}",
            lambda: _write_pandas(pandas_frame, opts_none),
            repeat=repeat, inner=200,
        ))
        out.append(_time_one(
            f"write: write_pandas_frame MATCH target rows={rows}",
            lambda: _write_pandas(pandas_frame, opts_match),
            repeat=repeat, inner=200,
        ))
    except ImportError:
        pass

    # write_polars_frame — confirms the polars cast → arrow → write
    # chain doesn't regress when the target adds an arrow-oriented
    # schema.
    try:
        import polars as pl  # noqa: F401

        polars_frame = pl.from_arrow(match_table)

        def _write_polars(frame, opts: CastOptions) -> None:
            sink = ArrowTabular()
            sink.write_polars_frame(frame, opts)

        out.append(_time_one(
            f"write: write_polars_frame no-target rows={rows}",
            lambda: _write_polars(polars_frame, opts_none),
            repeat=repeat, inner=200,
        ))
        out.append(_time_one(
            f"write: write_polars_frame MATCH target rows={rows}",
            lambda: _write_polars(polars_frame, opts_match),
            repeat=repeat, inner=200,
        ))
    except ImportError:
        pass
    return out


def _rechunk_scenarios(rows: int, repeat: int) -> list[dict]:
    """Direct rechunker hot paths.

    Two shapes: the standalone helpers
    (``rechunk_arrow_table`` / ``rechunk_arrow_batches``) and the
    options-driven seam ``CastOptions.cast_arrow_batch_iterator``,
    which downstream pipelines reach for when chaining a cast +
    rechunk over a streaming batch source.
    """
    out: list[dict] = []
    table = _arrow_table(rows)
    # Many small batches — the rechunker has work whenever the source
    # chunk shape doesn't match the requested cap.
    batches = _split_into_batches(table, batch_rows=max(1, rows // 32))
    sample_nbytes = sum(b.nbytes for b in batches)

    half_rows = max(1, rows // 2)
    quarter_rows = max(1, rows // 4)
    target_bytes = max(1, sample_nbytes // 4)

    # rechunk_arrow_table — table-shaped wrapper. Hot for callers that
    # land a Table and want a controlled chunk shape before handing it
    # off to a sink.
    out.append(_time_one(
        f"rechunk: rechunk_arrow_table row_size={half_rows} rows={rows}",
        lambda: rechunk_arrow_table(table, row_size=half_rows),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"rechunk: rechunk_arrow_table byte_size~{target_bytes} rows={rows}",
        lambda: rechunk_arrow_table(table, byte_size=target_bytes),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"rechunk: rechunk_arrow_table both caps rows={rows}",
        lambda: rechunk_arrow_table(
            table, row_size=quarter_rows, byte_size=target_bytes,
        ),
        repeat=repeat, inner=200,
    ))

    # rechunk_arrow_batches — streamed; ``batches`` is consumed per
    # call, so rebuild the iterator for each timing iteration.
    out.append(_time_one(
        f"rechunk: rechunk_arrow_batches passthrough rows={rows}",
        lambda: _drain(rechunk_arrow_batches(iter(batches))),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"rechunk: rechunk_arrow_batches row_size={half_rows} rows={rows}",
        lambda: _drain(rechunk_arrow_batches(iter(batches), row_size=half_rows)),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"rechunk: rechunk_arrow_batches byte_size~{target_bytes} rows={rows}",
        lambda: _drain(rechunk_arrow_batches(iter(batches), byte_size=target_bytes)),
        repeat=repeat, inner=200,
    ))

    # CastOptions.cast_arrow_batch_iterator — same rechunker, exposed
    # at the seam the Tabular surface routes downstream pipelines through.
    opts_row = CastOptions(row_size=half_rows)
    opts_bytes = CastOptions(byte_size=target_bytes)
    opts_target_row = CastOptions(target=TARGET_SCHEMA, row_size=half_rows)
    out.append(_time_one(
        f"rechunk: opts.cast_arrow_batch_iterator row_size rows={rows}",
        lambda: _drain(opts_row.cast_arrow_batch_iterator(iter(batches))),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"rechunk: opts.cast_arrow_batch_iterator byte_size rows={rows}",
        lambda: _drain(opts_bytes.cast_arrow_batch_iterator(iter(batches))),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"rechunk: opts.cast_arrow_batch_iterator target+row_size rows={rows}",
        lambda: _drain(opts_target_row.cast_arrow_batch_iterator(iter(batches))),
        repeat=repeat, inner=200,
    ))
    return out


def _spark_scenarios(rows: int, repeat: int) -> list[dict]:
    """:class:`CastOptions` against pyspark frames.

    Skipped when pyspark isn't installed or a local SparkSession
    fails to come up — the bench still reports a single skip row
    so the gap is visible in the output.
    """
    out: list[dict] = []
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return out

    try:
        from yggdrasil.environ import PyEnv
        spark = PyEnv.spark_session(create=True)
        if spark is None:
            raise RuntimeError("spark_session returned None")
    except Exception as exc:
        out.append({
            "label": f"spark: SKIPPED ({type(exc).__name__}: {exc})",
            "best": 0.0, "median": 0.0, "mean": 0.0,
        })
        return out

    # Build the two fixtures once: a "match" frame whose schema
    # already lines up with TARGET_SCHEMA, and a "mismatch" frame
    # whose ``id`` column needs the int32 → int64 cast kernel.
    match_table = _arrow_table(rows, mismatch=False)
    cast_table = _arrow_table(rows, mismatch=True)
    spark_match = spark.createDataFrame(match_table.to_pandas())
    spark_cast = spark.createDataFrame(cast_table.to_pandas())

    opts_none = CastOptions()
    opts_match = CastOptions(target=TARGET_SCHEMA)
    opts_cast = CastOptions(target=TARGET_SCHEMA)

    # ``cast_spark_tabular`` builds the engine-side plan; without
    # ``.collect()`` the cost is the plan rewrite, not the row
    # materialization. That's the right thing to measure here —
    # downstream callers chain more ops before collecting and the
    # cost amortizes — but it also explains why the timings are
    # microseconds rather than the seconds spark.createDataFrame /
    # collect take.
    out.append(_time_one(
        f"spark: opts.cast_spark_tabular no-target rows={rows}",
        lambda: opts_none.cast_spark_tabular(spark_match),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"spark: opts.cast_spark_tabular MATCH rows={rows}",
        lambda: opts_match.cast_spark_tabular(spark_match),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"spark: opts.cast_spark_tabular CAST rows={rows}",
        lambda: opts_cast.cast_spark_tabular(spark_cast),
        repeat=repeat, inner=200,
    ))

    # Full write round-trip through ArrowTabular: pulls the spark
    # frame back across the Arrow bridge (``toArrow`` when available,
    # else ``toPandas``) and feeds it through the arrow-cast path.
    # This is the cost the downstream :meth:`Tabular.write_spark_frame`
    # call sites actually pay.
    def _write_spark(frame, opts: CastOptions) -> None:
        sink = ArrowTabular()
        sink.write_spark_frame(frame, opts)

    out.append(_time_one(
        f"spark: write_spark_frame no-target rows={rows}",
        lambda: _write_spark(spark_match, opts_none),
        repeat=repeat, inner=20,
    ))
    out.append(_time_one(
        f"spark: write_spark_frame MATCH rows={rows}",
        lambda: _write_spark(spark_match, opts_match),
        repeat=repeat, inner=20,
    ))
    out.append(_time_one(
        f"spark: write_spark_frame CAST rows={rows}",
        lambda: _write_spark(spark_cast, opts_cast),
        repeat=repeat, inner=20,
    ))
    return out


def scenarios(rows: int, repeat: int) -> list[dict]:
    return [
        *_ingest_scenarios(rows, repeat),
        *_read_scenarios(rows, repeat),
        *_write_scenarios(rows, repeat),
        *_rechunk_scenarios(rows, repeat),
        *_spark_scenarios(rows, repeat),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000,
                    help="Row count for the in-memory fixture table.")
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# rows={args.rows}  repeat={args.repeat}")
    print(f"# {'label':<66s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    for row in scenarios(args.rows, args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
