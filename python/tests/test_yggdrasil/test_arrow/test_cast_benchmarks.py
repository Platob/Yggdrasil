"""Arrow cast benchmarks — surface where the per-batch cast cost lands.

Same shape as ``test_parquet_benchmarks.py``: regression-mode runs by
default with tiny inputs so the suite stays fast; ``YGG_BENCHMARK=1``
flips on real numbers.

Targets exercised here are the hot spots the wider data plane actually
hits — write-side per-batch cast on a Parquet write, struct-of-struct
rebuild on a column-renamed warehouse read, JSON-string decode on a
batch sourced from a CSV/HTTP payload, the Databricks
"concat-then-cast" flush shape.

Numbers from a typical run (250 k rows, single machine) call out:

* JSON-string decode of a list<struct> column dominates per-batch cast
  time by 2-3 orders of magnitude vs primitive casts. That's
  unavoidable — JSON parsing is the cost.
* Chunked array casts pay an extra ``pa.chunked_array(chunks, ...)``
  rebuild even when every chunk identity-returned. The patch on
  ``DataType._cast_chunked_array`` short-circuits that.
* Multi-chunk ChunkedArray inputs to the JSON decoder pay an extra
  ``combine_chunks()`` even on single-chunk inputs.  The patch on
  ``_cast_json._arrow_to_utf8`` skips the call when ``len(chunks) ==
  1``.
* The Databricks-style flush shape (concat fetched IPC batches →
  one tabular cast → yield) is already efficient — no per-row
  Python crossings, one cast pass per ``byte_size`` flush. Useful to
  guard against regressions.
"""
from __future__ import annotations

import os
import random
import string
import time
import unittest
from typing import Any

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions


def _benchmark_mode() -> bool:
    return os.environ.get("YGG_BENCHMARK", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _bench_rows() -> int:
    raw = os.environ.get("YGG_BENCHMARK_ROWS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 250_000


def _regression_rows() -> int:
    return 4_000


def _time_cast(fn) -> tuple[float, Any]:
    """Wall-clock the cast call once; return ``(elapsed_ms, result)``."""
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000.0, out


class _BenchReport:
    def __init__(self, name: str) -> None:
        self.name = name
        self.rows: list[tuple[str, float]] = []

    def add(self, label: str, elapsed_ms: float) -> None:
        self.rows.append((label, elapsed_ms))

    def flush(self) -> None:
        if not self.rows:
            return
        print(f"\n=== {self.name} ===")
        label_w = max(len(r[0]) for r in self.rows)
        for label, ms in self.rows:
            print(f"{label.ljust(label_w)}  ms={ms:8.2f}")


# ---------------------------------------------------------------------------
# Data factories
# ---------------------------------------------------------------------------


def _numeric_batch(n: int) -> pa.RecordBatch:
    """Three-column record batch shared by every benchmark.

    Columns are named ``i`` (int32), ``v`` (float64), ``t`` (string).
    The casts each test wants to exercise re-declare ``source_field`` /
    ``target_field`` against these names.
    """
    rnd = random.Random(0)
    cols = {
        "i": pa.array([rnd.randint(-1_000_000, 1_000_000) for _ in range(n)], type=pa.int32()),
        "v": pa.array([rnd.uniform(-1, 1) for _ in range(n)], type=pa.float64()),
        "t": pa.array(["".join(rnd.choices(string.ascii_lowercase, k=6)) for _ in range(n)]),
    }
    return pa.record_batch(cols)


def _json_string_array(n: int, *, bad_rate: float = 0.0) -> pa.Array:
    """String column whose rows are JSON-encoded list<struct>."""
    rnd = random.Random(1)
    rows: list[str] = []
    for _ in range(n):
        if rnd.random() < bad_rate:
            rows.append("not-json")
            continue
        m = rnd.randint(1, 3)
        rows.append(
            "["
            + ",".join(
                f'{{"a":{rnd.randint(0,100)},"b":"{rnd.choice("xyzqr")}"}}'
                for _ in range(m)
            )
            + "]"
        )
    return pa.array(rows, type=pa.string())


def _list_of_struct_target() -> pa.DataType:
    return pa.list_(
        pa.field("item", pa.struct([
            pa.field("a", pa.int64()),
            pa.field("b", pa.string()),
        ]))
    )


# ---------------------------------------------------------------------------
# Regression mode — small inputs, exercise the code path
# ---------------------------------------------------------------------------


class TestCastBenchmarkRegression(ArrowTestCase):
    """Run the same code paths the benchmarks hit, just on tiny inputs.

    Asserts only that the cast still produces a non-zero result — the
    benchmark mode below carries the actual perf numbers. This guards
    against the benchmarks silently rotting between releases.
    """

    def test_primitive_cast_path(self) -> None:
        n = _regression_rows()
        rb = _numeric_batch(n)
        src = Field.from_arrow(pa.schema([
            pa.field("i", pa.int32()),
            pa.field("v", pa.float64()),
            pa.field("t", pa.string()),
        ]))
        tgt = Field.from_arrow(pa.schema([
            pa.field("i", pa.int64()),
            pa.field("v", pa.float32()),
            pa.field("t", pa.string()),
        ]))
        out = CastOptions(source_field=src, target_field=tgt).cast_arrow_tabular(rb)
        self.assertEqual(out.num_rows, n)

    def test_chunked_identity_cast_path(self) -> None:
        # A chunked array whose source already matches target — exercises
        # the identity-chunk path that the optimization short-circuits.
        n = _regression_rows()
        ca = pa.chunked_array(
            [pa.array([i for i in range(n // 4)], type=pa.int64()) for _ in range(4)]
        )
        tgt = Field.from_arrow(pa.field("x", pa.int64()))
        src = Field.from_arrow(pa.field("x", pa.int64()))
        out = CastOptions(source_field=src, target_field=tgt).cast_arrow_array(ca)
        self.assertEqual(len(out), n - (n % 4))

    def test_json_string_decode_path(self) -> None:
        n = _regression_rows()
        arr = _json_string_array(n)
        src = Field.from_arrow(pa.field("payload", pa.string()))
        tgt = Field.from_arrow(pa.field("payload", _list_of_struct_target()))
        out = tgt.cast_arrow_array(arr, source_field=src)
        self.assertEqual(len(out), n)


# ---------------------------------------------------------------------------
# Benchmark mode — opt-in
# ---------------------------------------------------------------------------


@unittest.skipUnless(_benchmark_mode(), "set YGG_BENCHMARK=1 to run benchmarks")
class TestCastBenchmarks(ArrowTestCase):
    """Cast-path sweeps. Output is a printed report; tests pass when the
    cast produces a non-zero result."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.n_rows = _bench_rows()

    # -- primitive cast on a record batch ---------------------------------

    def test_primitive_tabular_cast(self) -> None:
        report = _BenchReport(
            f"primitive tabular cast (n={self.n_rows}) — "
            f"options.py:cast_arrow_tabular"
        )
        rb = _numeric_batch(self.n_rows)
        src = Field.from_arrow(pa.schema([
            pa.field("i", pa.int32()),
            pa.field("v", pa.float64()),
            pa.field("t", pa.string()),
        ]))
        widen = Field.from_arrow(pa.schema([
            pa.field("i", pa.int64()),
            pa.field("v", pa.float32()),
            pa.field("t", pa.string()),
        ]))
        # Identity: source = target. Hits the tabular bypass.
        ms, _ = _time_cast(
            lambda: CastOptions(source_field=src, target_field=src).cast_arrow_tabular(rb),
        )
        report.add("identity (source == target)", ms)
        # Widen i32→i64 + f64→f32.
        ms, _ = _time_cast(
            lambda: CastOptions(source_field=src, target_field=widen).cast_arrow_tabular(rb),
        )
        report.add("widen i32->i64, narrow f64->f32", ms)
        report.flush()

    # -- chunked vs flat -------------------------------------------------

    def test_chunked_array_cast(self) -> None:
        report = _BenchReport(
            f"chunked array cast (n={self.n_rows}) — base.py:_cast_chunked_array"
        )
        src = Field.from_arrow(pa.field("x", pa.int64()))
        tgt = Field.from_arrow(pa.field("x", pa.int64()))
        opts = CastOptions(source_field=src, target_field=tgt)

        flat = pa.array(range(self.n_rows), type=pa.int64())
        ms, _ = _time_cast(lambda: opts.cast_arrow_array(flat))
        report.add("flat Array, identity cast        ", ms)

        for k in (1, 4, 16, 64):
            chunk_size = self.n_rows // k
            ca = pa.chunked_array(
                [
                    pa.array(range(chunk_size), type=pa.int64())
                    for _ in range(k)
                ]
            )
            ms, _ = _time_cast(lambda c=ca: opts.cast_arrow_array(c))
            report.add(f"ChunkedArray k={k:3d}, identity cast", ms)
        report.flush()

    # -- JSON string decode --------------------------------------------

    def test_json_string_decode(self) -> None:
        report = _BenchReport(
            f"JSON-string -> list<struct> decode (n={self.n_rows}) — "
            f"_cast_json.py:cast_arrow_json_string_array"
        )
        src = Field.from_arrow(pa.field("payload", pa.string()))
        tgt = Field.from_arrow(pa.field("payload", _list_of_struct_target()))

        # Clean input, single chunk.
        arr = _json_string_array(self.n_rows)
        ms, _ = _time_cast(lambda: tgt.cast_arrow_array(arr, source_field=src))
        report.add("flat Array, all valid             ", ms)

        # Single-chunk ChunkedArray: combine_chunks() should be skipped
        # by the unwrap fast path.
        ca1 = pa.chunked_array([arr], type=pa.string())
        ms, _ = _time_cast(lambda c=ca1: tgt.cast_arrow_array(c, source_field=src))
        report.add("ChunkedArray k=1, unwrap fast path", ms)

        # Multi-chunk: combine_chunks() unavoidable.
        chunk_size = self.n_rows // 4
        chunks = [arr.slice(k * chunk_size, chunk_size) for k in range(4)]
        ca = pa.chunked_array(chunks, type=pa.string())
        ms, _ = _time_cast(lambda c=ca: tgt.cast_arrow_array(c, source_field=src))
        report.add("ChunkedArray k=4, all valid       ", ms)

        # Permissive on 5% bad rows — falls back to per-row Python.
        bad_arr = _json_string_array(self.n_rows, bad_rate=0.05)
        ms, _ = _time_cast(
            lambda: tgt.cast_arrow_array(bad_arr, source_field=src, safe=False),
        )
        report.add("flat Array, 5% bad, safe=False  ", ms)
        report.flush()

    # -- nested struct rebuild -----------------------------------------

    def test_nested_struct_rebuild(self) -> None:
        report = _BenchReport(
            f"struct-of-struct child-renaming rebuild (n={self.n_rows}) — "
            f"struct_arrow.py:cast_arrow_struct_array"
        )
        n = self.n_rows
        inner = pa.StructArray.from_arrays(
            [pa.array(range(n), type=pa.int32()), pa.array(["x"] * n)],
            names=["a", "b"],
        )
        outer = pa.StructArray.from_arrays([inner], names=["inner"])
        src = Field.from_arrow(pa.field("row", outer.type))
        # Widen inner.a from int32 → int64.
        tgt_type = pa.struct([
            pa.field("inner", pa.struct([
                pa.field("a", pa.int64()),
                pa.field("b", pa.string()),
            ])),
        ])
        tgt = Field.from_arrow(pa.field("row", tgt_type))
        ms, _ = _time_cast(
            lambda: tgt.cast_arrow_array(outer, source_field=src),
        )
        report.add("widen inner.a int32->int64", ms)
        report.flush()

    # -- Databricks-style concat + cast --------------------------------

    def test_databricks_concat_then_cast_pattern(self) -> None:
        """Mirror the Databricks external-link flush shape: many small
        fetched batches buffered to a byte threshold, then concat'd and
        cast in one pass. The cast is the expensive part — concat is
        a memcpy.

        The benchmark surfaces both costs so we can verify a regression
        in either one. The numbers here run against synthetic data,
        not the wire — network is the real cost in production.
        """
        report = _BenchReport(
            f"databricks-style concat+cast (n={self.n_rows}) — "
            f"warehouse/statement.py:_read_arrow_batches"
        )
        src = Field.from_arrow(pa.schema([
            pa.field("i", pa.int32()),
            pa.field("v", pa.float64()),
            pa.field("t", pa.string()),
        ]))
        tgt = Field.from_arrow(pa.schema([
            pa.field("i", pa.int64()),
            pa.field("v", pa.float32()),
            pa.field("t", pa.string()),
        ]))

        for k in (4, 16, 64):
            chunk_size = self.n_rows // k
            batches = [_numeric_batch(chunk_size) for _ in range(k)]
            ms, _ = _time_cast(
                lambda b=batches: CastOptions(
                    source_field=src, target_field=tgt,
                ).cast_arrow_tabular(
                    pa.Table.from_batches(b).combine_chunks().to_batches()[0]
                ),
            )
            report.add(f"k={k:3d} fetched batches, concat+cast", ms)
        report.flush()
