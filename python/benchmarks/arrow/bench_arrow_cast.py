"""Benchmark :mod:`yggdrasil.arrow.cast` — the Arrow-side conversion surface.

Every pipeline that lands in pyarrow funnels through this module: the
``any_to_arrow_*`` converters, the in-place ``cast_arrow_*`` wrappers,
:func:`get_arrow_nbytes` (sized by every rechunk decision in the
codebase), and the streaming :func:`rechunk_arrow_batches` /
:func:`rechunk_arrow_table` helpers.

Coverage in this bench:

* **Type / schema entry points** — ``any_to_arrow_field`` /
  ``any_to_arrow_schema`` from raw Arrow + :class:`Schema` / :class:`Field`.
* **Scalar casting** — ``any_to_arrow_scalar`` / ``cast_arrow_scalar``
  (pa.Scalar → pa.Scalar, None → default, Python → typed).
* **Array casting** — ``cast_arrow_array`` on flat ``pa.Array`` and
  multi-chunk ``pa.ChunkedArray``, MATCH vs CAST.
* **Tabular casting** — ``cast_arrow_tabular`` on ``pa.Table`` /
  ``pa.RecordBatch``, MATCH vs CAST.
* **Bulk converters** — ``any_to_arrow_table`` from pa.Table,
  pa.RecordBatch, pa.RecordBatchReader, list[dict].
* **Streaming converters** — ``any_to_arrow_batch_iterator`` /
  ``any_to_arrow_record_batch_reader``.
* **Size estimator** — ``get_arrow_nbytes`` on Array, ChunkedArray,
  RecordBatch, Table, plus the view-type fast-path.
* **Rechunkers** — ``rechunk_arrow_batches`` /
  ``rechunk_arrow_table`` at row / byte / both.

What we do **not** measure here (covered elsewhere):

* Per-format primitive readers (CSV / Parquet / NDJSON / IPC) —
  ``benchmarks/io/``.
* Pandas / Polars / Spark cast kernels — ``benchmarks/data/bench_cast.py``.
* ``ArrowTabular`` read/write seam — ``benchmarks/io/tabular/bench_tabular.py``.

Usage::

    PYTHONPATH=src python benchmarks/arrow/bench_arrow_cast.py
    PYTHONPATH=src python benchmarks/arrow/bench_arrow_cast.py --rows 50000 --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time
from typing import Callable, Iterable

import pyarrow as pa

from yggdrasil.arrow.cast import (
    any_to_arrow_batch_iterator,
    any_to_arrow_field,
    any_to_arrow_record_batch,
    any_to_arrow_record_batch_reader,
    any_to_arrow_scalar,
    any_to_arrow_schema,
    any_to_arrow_table,
    cast_arrow_array,
    cast_arrow_scalar,
    cast_arrow_tabular,
    get_arrow_nbytes,
    rechunk_arrow_batches,
    rechunk_arrow_table,
)
from yggdrasil.data import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema


# ---------------------------------------------------------------------------
# Fixtures
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


def _chunked(table: pa.Table, chunks: int) -> pa.ChunkedArray:
    """Slice column 0 into a fixed number of chunks."""
    col = table.column(0)
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    rows = len(col)
    step = max(1, rows // chunks)
    pieces = [col.slice(i, step) for i in range(0, rows, step)]
    return pa.chunked_array(pieces)


def _build_string_view_array(rows: int) -> pa.Array:
    """Best-effort string_view array; falls back to plain string."""
    base = pa.array(["row-" + str(i % 100) for i in range(rows)],
                    type=pa.string())
    try:
        return base.cast(pa.string_view())  # type: ignore[attr-defined]
    except (AttributeError, pa.ArrowNotImplementedError, pa.ArrowInvalid):
        return base


# ---------------------------------------------------------------------------
# Timing helpers
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


def _nbytes_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    table = _arrow_table(rows)
    batch = table.to_batches()[0]
    col_int = table.column("id")
    col_str = table.column("name")
    chunked_int = _chunked(table, chunks=8)
    view_arr = _build_string_view_array(rows)

    out.append(_time_one(
        f"nbytes: get_arrow_nbytes(Table) rows={rows}",
        lambda: get_arrow_nbytes(table),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        f"nbytes: get_arrow_nbytes(RecordBatch) rows={rows}",
        lambda: get_arrow_nbytes(batch),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        f"nbytes: get_arrow_nbytes(ChunkedArray x8) rows={rows}",
        lambda: get_arrow_nbytes(chunked_int),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        f"nbytes: get_arrow_nbytes(int64 Array) rows={rows}",
        lambda: get_arrow_nbytes(col_int),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        f"nbytes: get_arrow_nbytes(string Array) rows={rows}",
        lambda: get_arrow_nbytes(col_str),
        repeat=repeat, inner=50_000,
    ))
    out.append(_time_one(
        f"nbytes: get_arrow_nbytes(string_view Array) rows={rows}",
        lambda: get_arrow_nbytes(view_arr),
        repeat=repeat, inner=50_000,
    ))
    return out


def _field_schema_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    pa_field = pa.field("id", pa.int64(), nullable=False)
    pa_schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("amount", pa.float64()),
        pa.field("name", pa.string()),
    ])
    ygg_field = Field("id", "int64", nullable=False)
    ygg_schema = TARGET_SCHEMA

    out.append(_time_one(
        "schema: any_to_arrow_field(pa.Field)",
        lambda: any_to_arrow_field(pa_field),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "schema: any_to_arrow_field(Field)",
        lambda: any_to_arrow_field(ygg_field),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "schema: any_to_arrow_schema(pa.Schema)",
        lambda: any_to_arrow_schema(pa_schema),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "schema: any_to_arrow_schema(Schema)",
        lambda: any_to_arrow_schema(ygg_schema),
        repeat=repeat, inner=10_000,
    ))
    return out


def _scalar_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []
    opts_int = CastOptions(target=Field("id", "int64"))
    opts_none = CastOptions()
    sc = pa.scalar(42, type=pa.int32())

    out.append(_time_one(
        "scalar: any_to_arrow_scalar(int → int64)",
        lambda: any_to_arrow_scalar(42, opts_int),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "scalar: any_to_arrow_scalar(None → default)",
        lambda: any_to_arrow_scalar(None, opts_int),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "scalar: any_to_arrow_scalar(int, no target)",
        lambda: any_to_arrow_scalar(42, opts_none),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "scalar: cast_arrow_scalar(pa.Scalar → int64)",
        lambda: cast_arrow_scalar(sc, opts_int),
        repeat=repeat, inner=10_000,
    ))
    return out


def _array_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    arr_match = pa.array(range(rows), type=pa.int64())
    arr_cast = pa.array(range(rows), type=pa.int32())
    chunked_match = pa.chunked_array(
        [arr_match.slice(i, max(1, rows // 4)) for i in range(0, rows, max(1, rows // 4))]
    )
    chunked_cast = pa.chunked_array(
        [arr_cast.slice(i, max(1, rows // 4)) for i in range(0, rows, max(1, rows // 4))]
    )
    id_field = TARGET_SCHEMA.field_by("id")
    opts = CastOptions(target=id_field)

    out.append(_time_one(
        f"array: cast_arrow_array MATCH rows={rows}",
        lambda: cast_arrow_array(arr_match, opts),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        f"array: cast_arrow_array CAST rows={rows}",
        lambda: cast_arrow_array(arr_cast, opts),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        f"array: cast_arrow_array MATCH Chunked x4 rows={rows}",
        lambda: cast_arrow_array(chunked_match, opts),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        f"array: cast_arrow_array CAST Chunked x4 rows={rows}",
        lambda: cast_arrow_array(chunked_cast, opts),
        repeat=repeat, inner=2_000,
    ))
    return out


def _tabular_scenarios(rows: int, repeat: int) -> list[dict]:
    """``cast_arrow_tabular`` — pa.Table + pa.RecordBatch, MATCH/CAST."""
    out: list[dict] = []
    match_table = _arrow_table(rows, mismatch=False)
    cast_table = _arrow_table(rows, mismatch=True)
    match_batch = match_table.combine_chunks().to_batches()[0]
    cast_batch = cast_table.combine_chunks().to_batches()[0]

    opts = CastOptions(target=TARGET_SCHEMA)
    opts_none = CastOptions()

    out.append(_time_one(
        f"tabular: cast_arrow_tabular(Table) MATCH rows={rows}",
        lambda: cast_arrow_tabular(match_table, opts),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"tabular: cast_arrow_tabular(Table) CAST rows={rows}",
        lambda: cast_arrow_tabular(cast_table, opts),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"tabular: cast_arrow_tabular(Batch) MATCH rows={rows}",
        lambda: cast_arrow_tabular(match_batch, opts),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"tabular: cast_arrow_tabular(Batch) CAST rows={rows}",
        lambda: cast_arrow_tabular(cast_batch, opts),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"tabular: cast_arrow_tabular(Table) no-target rows={rows}",
        lambda: cast_arrow_tabular(match_table, opts_none),
        repeat=repeat, inner=200_000,
    ))
    return out


def _any_to_table_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    match_table = _arrow_table(rows, mismatch=False)
    cast_table = _arrow_table(rows, mismatch=True)
    match_batch = match_table.combine_chunks().to_batches()[0]
    pylist = match_table.to_pylist()  # endpoint-only — diagnostic fixture
    reader_schema = match_table.schema

    opts = CastOptions(target=TARGET_SCHEMA)

    out.append(_time_one(
        f"any2table: any_to_arrow_table(Table) MATCH rows={rows}",
        lambda: any_to_arrow_table(match_table, opts),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"any2table: any_to_arrow_table(Table) CAST rows={rows}",
        lambda: any_to_arrow_table(cast_table, opts),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"any2table: any_to_arrow_table(Batch) MATCH rows={rows}",
        lambda: any_to_arrow_table(match_batch, opts),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"any2table: any_to_arrow_record_batch(Batch) rows={rows}",
        lambda: any_to_arrow_record_batch(match_batch, opts),
        repeat=repeat, inner=500,
    ))
    # RecordBatchReader — has to be rebuilt each iteration since reads
    # are one-shot. Construction cost shows up but is small relative to
    # the read.
    out.append(_time_one(
        f"any2table: any_to_arrow_table(Reader) rows={rows}",
        lambda: any_to_arrow_table(
            pa.RecordBatchReader.from_batches(reader_schema, [match_batch]),
            opts,
        ),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"any2table: any_to_arrow_table(list[dict]) rows={rows}",
        lambda: any_to_arrow_table(pylist, opts),
        repeat=repeat, inner=100,
    ))
    return out


def _streaming_scenarios(rows: int, repeat: int) -> list[dict]:
    out: list[dict] = []
    match_table = _arrow_table(rows, mismatch=False)
    cast_table = _arrow_table(rows, mismatch=True)

    opts = CastOptions(target=TARGET_SCHEMA)
    opts_rechunk = CastOptions(target=TARGET_SCHEMA, row_size=max(1, rows // 4))

    out.append(_time_one(
        f"stream: any_to_arrow_batch_iterator(Table) MATCH rows={rows}",
        lambda: _drain(any_to_arrow_batch_iterator(match_table, opts)),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"stream: any_to_arrow_batch_iterator(Table) CAST rows={rows}",
        lambda: _drain(any_to_arrow_batch_iterator(cast_table, opts)),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        f"stream: any_to_arrow_batch_iterator(Table)+rechunk rows={rows}",
        lambda: _drain(any_to_arrow_batch_iterator(match_table, opts_rechunk)),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"stream: any_to_arrow_record_batch_reader(Table) MATCH rows={rows}",
        lambda: _drain(any_to_arrow_record_batch_reader(match_table, opts)),
        repeat=repeat, inner=500,
    ))
    return out


def _rechunk_scenarios(rows: int, repeat: int) -> list[dict]:
    """Direct rechunkers — covers the chunker hot path independently of
    the cast pipeline. Same shape as the Tabular bench but driven from
    the arrow.cast public API so a regression here is caught even when
    the Tabular layer changes shape.
    """
    out: list[dict] = []
    table = _arrow_table(rows)
    small_batch_rows = max(1, rows // 16)
    batches = []
    for offset in range(0, rows, small_batch_rows):
        batches.extend(table.slice(offset, small_batch_rows).to_batches())
    sample_nbytes = sum(b.nbytes for b in batches)
    target_bytes = max(1, sample_nbytes // 4)
    half_rows = max(1, rows // 2)

    out.append(_time_one(
        f"rechunk: rechunk_arrow_table passthrough rows={rows}",
        lambda: rechunk_arrow_table(table),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        f"rechunk: rechunk_arrow_table row_size={half_rows} rows={rows}",
        lambda: rechunk_arrow_table(table, row_size=half_rows),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"rechunk: rechunk_arrow_table byte_size~{target_bytes} rows={rows}",
        lambda: rechunk_arrow_table(table, byte_size=target_bytes),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"rechunk: rechunk_arrow_batches row_size={half_rows} rows={rows}",
        lambda: _drain(rechunk_arrow_batches(iter(batches), row_size=half_rows)),
        repeat=repeat, inner=500,
    ))
    out.append(_time_one(
        f"rechunk: rechunk_arrow_batches byte_size~{target_bytes} rows={rows}",
        lambda: _drain(rechunk_arrow_batches(iter(batches), byte_size=target_bytes)),
        repeat=repeat, inner=200,
    ))
    return out


def scenarios(rows: int, repeat: int) -> list[dict]:
    return [
        *_field_schema_scenarios(repeat),
        *_scalar_scenarios(repeat),
        *_nbytes_scenarios(rows, repeat),
        *_array_scenarios(rows, repeat),
        *_tabular_scenarios(rows, repeat),
        *_any_to_table_scenarios(rows, repeat),
        *_streaming_scenarios(rows, repeat),
        *_rechunk_scenarios(rows, repeat),
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
