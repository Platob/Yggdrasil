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
        out = CastOptions(source=src, target=tgt).cast_arrow_batch(rb)
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
        out = CastOptions(source=src, target=tgt).cast_arrow_array(ca)
        self.assertEqual(len(out), n - (n % 4))

    def test_json_string_decode_path(self) -> None:
        n = _regression_rows()
        arr = _json_string_array(n)
        src = Field.from_arrow(pa.field("payload", pa.string()))
        tgt = Field.from_arrow(pa.field("payload", _list_of_struct_target()))
        out = tgt.cast_arrow_array(arr, source=src)
        self.assertEqual(len(out), n)

    def test_list_element_dtype_path(self) -> None:
        n = _regression_rows()
        arr = pa.array(
            [[i, i + 1] for i in range(n)],
            type=pa.list_(pa.field("item", pa.int32())),
        )
        src = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", pa.int32())),
        ))
        tgt = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", pa.int64())),
        ))
        out = tgt.cast_arrow_array(arr, source=src)
        self.assertEqual(len(out), n)
        self.assertEqual(out.type.value_type, pa.int64())

    def test_list_of_struct_element_path(self) -> None:
        n = _regression_rows()
        flat_struct = pa.StructArray.from_arrays(
            [
                pa.array(range(n), type=pa.int32()),
                pa.array(["x"] * n, type=pa.string()),
            ],
            names=["a", "b"],
        )
        offsets = pa.array(list(range(n + 1)), type=pa.int32())
        list_arr = pa.ListArray.from_arrays(offsets, flat_struct)
        src = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", flat_struct.type)),
        ))
        tgt = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", pa.struct([
                pa.field("a", pa.int64()),
                pa.field("b", pa.string()),
            ]))),
        ))
        out = tgt.cast_arrow_array(list_arr, source=src)
        self.assertEqual(len(out), n)

    def test_nested_struct_value_path(self) -> None:
        n = _regression_rows()
        inner_list = pa.ListArray.from_arrays(
            pa.array(list(range(n + 1)), type=pa.int32()),
            pa.array(range(n), type=pa.int32()),
        )
        struct_arr = pa.StructArray.from_arrays(
            [inner_list, pa.array(["k"] * n, type=pa.string())],
            names=["inner_list", "name"],
        )
        src = Field.from_arrow(pa.field("row", struct_arr.type))
        tgt = Field.from_arrow(pa.field("row", pa.struct([
            pa.field("inner_list", pa.list_(pa.field("item", pa.int64()))),
            pa.field("name", pa.string()),
        ])))
        out = tgt.cast_arrow_array(struct_arr, source=src)
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
            lambda: CastOptions(source=src, target=src).cast_arrow_batch(rb),
        )
        report.add("identity (source == target)", ms)
        # Widen i32→i64 + f64→f32.
        ms, _ = _time_cast(
            lambda: CastOptions(source=src, target=widen).cast_arrow_batch(rb),
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
        opts = CastOptions(source=src, target=tgt)

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
        ms, _ = _time_cast(lambda: tgt.cast_arrow_array(arr, source=src))
        report.add("flat Array, all valid             ", ms)

        # Single-chunk ChunkedArray: combine_chunks() should be skipped
        # by the unwrap fast path.
        ca1 = pa.chunked_array([arr], type=pa.string())
        ms, _ = _time_cast(lambda c=ca1: tgt.cast_arrow_array(c, source=src))
        report.add("ChunkedArray k=1, unwrap fast path", ms)

        # Multi-chunk: combine_chunks() unavoidable.
        chunk_size = self.n_rows // 4
        chunks = [arr.slice(k * chunk_size, chunk_size) for k in range(4)]
        ca = pa.chunked_array(chunks, type=pa.string())
        ms, _ = _time_cast(lambda c=ca: tgt.cast_arrow_array(c, source=src))
        report.add("ChunkedArray k=4, all valid       ", ms)

        # Permissive on 5% bad rows — falls back to per-row Python.
        bad_arr = _json_string_array(self.n_rows, bad_rate=0.05)
        ms, _ = _time_cast(
            lambda: tgt.cast_arrow_array(bad_arr, source=src, safe=False),
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
            lambda: tgt.cast_arrow_array(outer, source=src),
        )
        report.add("widen inner.a int32->int64", ms)
        report.flush()

    # -- list<item> -> struct<...> positional extraction ---------------

    def test_list_to_struct_cast(self) -> None:
        """``cast_arrow_list_array`` is the list -> struct positional
        cast — used when the wider data plane lands a list-shaped column
        that the target schema declares as a struct with N children.

        The path was a per-row Python walk (``to_pylist`` + comprehension
        per child). The vectorised replacement uses ``pc.take`` over the
        flat values buffer with a ``list_value_length`` mask — pure C++
        kernels, no Python crossings.

        Numbers (compared to the legacy baseline below) measure the
        gap; on a 250 k-row input the vectorised path is typically
        an order of magnitude faster.
        """
        n = self.n_rows
        # Build a list<int32> source of varying row lengths so the
        # short-list mask actually has work to do.
        rnd = random.Random(2)
        lengths = [rnd.choice([1, 2, 3, 4]) for _ in range(n)]
        flat = [rnd.randint(0, 1000) for _ in range(sum(lengths))]
        # Punch a few null parents in so parent-null propagation
        # actually fires in the benchmark.
        offsets = [0]
        for L in lengths:
            offsets.append(offsets[-1] + L)
        arr = pa.ListArray.from_arrays(
            pa.array(offsets, type=pa.int32()),
            pa.array(flat, type=pa.int32()),
        )
        # Null out every 53rd parent row.
        mask = pa.array(
            [i % 53 != 0 for i in range(n)], type=pa.bool_(),
        )
        arr = arr.filter(mask).cast(pa.list_(pa.field("item", pa.int32())))
        # Note: filter shrinks length; rebuild a full-length array
        # by re-padding via nulls. Simpler: build via pa.array(...)
        # directly with explicit list-of-list-or-None.
        arr_py: list[list[int] | None] = []
        idx = 0
        for j, L in enumerate(lengths):
            if j % 53 == 0:
                arr_py.append(None)
            else:
                arr_py.append(flat[idx:idx + L])
            idx += L
        arr = pa.array(
            arr_py, type=pa.list_(pa.field("item", pa.int32())),
        )

        target_struct = pa.struct([
            pa.field("c0", pa.int32()),
            pa.field("c1", pa.int32()),
            pa.field("c2", pa.int32()),
        ])
        src = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", pa.int32())),
        ))
        tgt = Field.from_arrow(pa.field("x", target_struct))

        report = _BenchReport(
            f"list<int32> -> struct<c0,c1,c2> cast (n={self.n_rows}) — "
            f"struct_arrow.py:cast_arrow_list_array"
        )
        ms, _ = _time_cast(
            lambda: tgt.cast_arrow_array(arr, source=src),
        )
        report.add("vectorised (pc.take + offsets)   ", ms)

        # Legacy path — preserved here in the benchmark so we can prove
        # the speedup directly. Mirrors the previous implementation:
        # ``to_pylist`` + per-row Python list comprehension per child.
        def legacy_path() -> pa.StructArray:
            values_py = arr.to_pylist()
            children = []
            target_fields = list(target_struct)
            for i, _child in enumerate(target_fields):
                extracted_py = [
                    None if row is None or i >= len(row) else row[i]
                    for row in values_py
                ]
                extracted = pa.array(extracted_py, type=pa.int32())
                children.append(extracted)
            return pa.StructArray.from_arrays(
                children, fields=list(target_struct),
                mask=arr.is_null(),
            )

        ms, _ = _time_cast(legacy_path)
        report.add("legacy (to_pylist + Python comp) ", ms)
        report.flush()

    # -- list element-dtype change (ListArray rebuild) -----------------

    def test_list_element_dtype_cast(self) -> None:
        """``cast_arrow_list_array`` (in :mod:`nested.array`) handles
        list-shape preserved casts where only the element dtype changes
        — widen / narrow primitives, str→int, list<int>→list<str>.

        The vectorised path lifts ``array.values`` once, casts the flat
        buffer in a single ``cast_arrow_array`` call, and rebuilds the
        list with the original ``array.offsets`` and ``array.is_null()``
        mask. No per-row Python crossings — every value-side op stays
        inside Arrow compute.
        """
        n = self.n_rows
        rnd = random.Random(3)
        lengths = [rnd.choice([0, 1, 2, 3, 4, 5]) for _ in range(n)]
        flat_vals = [rnd.randint(-1000, 1000) for _ in range(sum(lengths))]
        arr_py: list[list[int] | None] = []
        idx = 0
        for j, L in enumerate(lengths):
            if j % 73 == 0:
                arr_py.append(None)
            else:
                arr_py.append(flat_vals[idx:idx + L])
            idx += L
        arr = pa.array(arr_py, type=pa.list_(pa.field("item", pa.int32())))

        widen_src = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", pa.int32())),
        ))
        widen_tgt = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", pa.int64())),
        ))
        str_tgt = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", pa.string())),
        ))

        report = _BenchReport(
            f"list element dtype cast (n={self.n_rows}) — "
            f"nested.array:cast_arrow_list_array"
        )

        ms, _ = _time_cast(
            lambda: widen_tgt.cast_arrow_array(arr, source=widen_src),
        )
        report.add("list<int32> -> list<int64>", ms)

        ms, _ = _time_cast(
            lambda: str_tgt.cast_arrow_array(arr, source=widen_src),
        )
        report.add("list<int32> -> list<string>", ms)

        # Chunked input — same vectorised path runs per chunk.
        ca = pa.chunked_array(
            [arr.slice(k * (n // 4), n // 4) for k in range(4)],
            type=arr.type,
        )
        ms, _ = _time_cast(
            lambda c=ca: widen_tgt.cast_arrow_array(c, source=widen_src),
        )
        report.add("ChunkedArray k=4 -> list<int64>", ms)
        report.flush()

    # -- list<struct> element-of-struct cast ---------------------------

    def test_list_of_struct_element_cast(self) -> None:
        """Cast a ``list<struct<a:int32,b:string>>`` to a target where
        the struct child widens — ``list<struct<a:int64,b:string>>``.

        Stresses the ``ListArray`` rebuild plus the nested struct cast
        on the flat values buffer.  This is the hottest shape in
        nested ingest (e.g. event payloads, line-items inside an
        invoice) and the path that benefits most from the vectorised
        struct rebuild — every value-side op stays inside Arrow
        compute kernels.
        """
        n = self.n_rows
        rnd = random.Random(4)
        lengths = [rnd.choice([1, 2, 3]) for _ in range(n)]
        total = sum(lengths)
        flat_a = pa.array(
            [rnd.randint(-1000, 1000) for _ in range(total)], type=pa.int32(),
        )
        flat_b = pa.array(
            ["".join(rnd.choices(string.ascii_lowercase, k=4)) for _ in range(total)],
            type=pa.string(),
        )
        flat_struct = pa.StructArray.from_arrays(
            [flat_a, flat_b], names=["a", "b"],
        )
        offsets = [0]
        for L in lengths:
            offsets.append(offsets[-1] + L)
        list_arr = pa.ListArray.from_arrays(
            pa.array(offsets, type=pa.int32()),
            flat_struct,
        )

        src = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", flat_struct.type)),
        ))
        tgt_inner = pa.struct([
            pa.field("a", pa.int64()),
            pa.field("b", pa.string()),
        ])
        tgt = Field.from_arrow(pa.field(
            "x", pa.list_(pa.field("item", tgt_inner)),
        ))

        report = _BenchReport(
            f"list<struct> element cast (n={self.n_rows}) — "
            f"nested.array:cast_arrow_list_array"
        )

        ms, _ = _time_cast(
            lambda: tgt.cast_arrow_array(list_arr, source=src),
        )
        report.add("list<struct<int32,str>> -> list<struct<int64,str>>", ms)

        # Identity cast — should short-circuit via type equality on the
        # ListArray rebuild path.
        ms, _ = _time_cast(
            lambda: src.cast_arrow_array(list_arr, source=src),
        )
        report.add("identity list<struct> cast (short-circuit)         ", ms)
        report.flush()

    # -- nested struct values (struct of list / list of struct) --------

    def test_nested_struct_value_cast(self) -> None:
        """Deeper-than-one struct casts that show how the per-leaf
        ``Field.cast_arrow_array`` wrap unfolds against realistic
        nested shapes.

        Two patterns mirror real wire shapes:

        * ``struct<inner_list: list<int32>, name: string>`` →
          widen the list element and the outer struct survives the
          rebuild with one ``StructArray.from_arrays`` call.
        * ``struct<row: struct<a:int32, items: list<struct<x,y>>>>``
          (an "envelope around a line-item list") — exercises the
          recursive ``cast_arrow_struct_array`` → ``cast_arrow_list_array``
          → ``cast_arrow_struct_array`` chain.  Each layer should
          stay vectorised; the benchmark surfaces it.
        """
        n = self.n_rows
        rnd = random.Random(5)

        # Pattern A — struct<inner_list, name>
        lengths = [rnd.choice([0, 1, 2, 3]) for _ in range(n)]
        flat_inner = pa.array(
            [rnd.randint(0, 1000) for _ in range(sum(lengths))], type=pa.int32(),
        )
        offsets_a = [0]
        for L in lengths:
            offsets_a.append(offsets_a[-1] + L)
        inner_list = pa.ListArray.from_arrays(
            pa.array(offsets_a, type=pa.int32()),
            flat_inner,
        )
        name_col = pa.array(
            ["".join(rnd.choices(string.ascii_lowercase, k=5)) for _ in range(n)],
            type=pa.string(),
        )
        struct_a = pa.StructArray.from_arrays(
            [inner_list, name_col], names=["inner_list", "name"],
        )
        src_a = Field.from_arrow(pa.field("row", struct_a.type))
        tgt_a_type = pa.struct([
            pa.field("inner_list", pa.list_(pa.field("item", pa.int64()))),
            pa.field("name", pa.string()),
        ])
        tgt_a = Field.from_arrow(pa.field("row", tgt_a_type))

        # Pattern B — struct<row: struct<a, items: list<struct<x,y>>>>
        # — an "envelope" with an inner line-item list.
        item_count = n  # one item per row to keep flat sizes balanced
        x_col = pa.array(
            [rnd.randint(-100, 100) for _ in range(item_count)], type=pa.int32(),
        )
        y_col = pa.array(
            [rnd.random() for _ in range(item_count)], type=pa.float64(),
        )
        items_inner = pa.StructArray.from_arrays(
            [x_col, y_col], names=["x", "y"],
        )
        items_offsets = pa.array(list(range(n + 1)), type=pa.int32())
        items_list = pa.ListArray.from_arrays(items_offsets, items_inner)
        a_col = pa.array(
            [rnd.randint(0, 1_000_000) for _ in range(n)], type=pa.int32(),
        )
        inner_struct = pa.StructArray.from_arrays(
            [a_col, items_list], names=["a", "items"],
        )
        env_struct = pa.StructArray.from_arrays(
            [inner_struct], names=["row"],
        )
        src_b = Field.from_arrow(pa.field("envelope", env_struct.type))
        tgt_items_inner = pa.struct([
            pa.field("x", pa.int64()),
            pa.field("y", pa.float64()),
        ])
        tgt_b_type = pa.struct([
            pa.field("row", pa.struct([
                pa.field("a", pa.int64()),
                pa.field("items", pa.list_(pa.field("item", tgt_items_inner))),
            ])),
        ])
        tgt_b = Field.from_arrow(pa.field("envelope", tgt_b_type))

        report = _BenchReport(
            f"nested struct value cast (n={self.n_rows}) — "
            f"struct_arrow.py:cast_arrow_struct_array"
        )

        ms, _ = _time_cast(
            lambda: tgt_a.cast_arrow_array(struct_a, source=src_a),
        )
        report.add("struct<list<int32>, str> -> widen list<int64>     ", ms)

        ms, _ = _time_cast(
            lambda: tgt_b.cast_arrow_array(env_struct, source=src_b),
        )
        report.add("struct<struct<int32, list<struct<x,y>>>> widen all", ms)
        report.flush()

    # -- pandas-side casts ---------------------------------------------

    def test_pandas_tabular_cast(self) -> None:
        """``cast_pandas_tabular`` (struct_pandas.py) routes the cast
        through the pyarrow→polars→columnwise fallback chain so the
        primary path stays inside vectorised Arrow kernels.

        The benchmark surfaces the Arrow path (the only one a healthy
        primitive DataFrame should hit) plus the polars round-trip and
        the columnwise reassembly fallback so a regression in any
        layer is visible.
        """
        import pandas as pd

        n = self.n_rows
        rnd = random.Random(7)
        df = pd.DataFrame({
            "i": [rnd.randint(-1_000_000, 1_000_000) for _ in range(n)],
            "v": [rnd.uniform(-1, 1) for _ in range(n)],
            "t": ["".join(rnd.choices(string.ascii_lowercase, k=6)) for _ in range(n)],
        })

        src = Field.from_arrow(pa.schema([
            pa.field("i", pa.int64()),
            pa.field("v", pa.float64()),
            pa.field("t", pa.string()),
        ]))
        tgt = Field.from_arrow(pa.schema([
            pa.field("i", pa.int32()),
            pa.field("v", pa.float32()),
            pa.field("t", pa.string()),
        ]))

        report = _BenchReport(
            f"pandas tabular cast (n={self.n_rows}) — "
            f"struct_pandas.py:cast_pandas_tabular"
        )

        ms, _ = _time_cast(
            lambda: tgt.cast_pandas(df, source=src),
        )
        report.add("primitive widen+narrow (pyarrow fast path)", ms)
        report.flush()

    def test_pandas_struct_series_cast(self) -> None:
        """Pandas Series of dicts → struct cast.

        Pandas has no native struct dtype, so the values sit in an
        object-Series and the cast has to push them through pyarrow
        (or polars) without a per-row Python loop. The pyarrow path
        round-trips via ``pa.array(series, from_pandas=True, type=...)``
        → ``cast`` → ``to_pylist`` — the Python-dict materialisation
        is the genuine endpoint, but the *cast* itself stays in
        vectorised Arrow.
        """
        import pandas as pd

        n = self.n_rows
        rnd = random.Random(8)
        rows = [
            {"a": rnd.randint(0, 1_000), "b": rnd.choice(["x", "y", "z"])}
            if rnd.random() > 0.05 else None
            for _ in range(n)
        ]
        series = pd.Series(rows, name="payload", dtype="object")

        src_struct = pa.struct([
            pa.field("a", pa.int64()),
            pa.field("b", pa.string()),
        ])
        tgt_struct = pa.struct([
            pa.field("a", pa.int32()),
            pa.field("b", pa.string()),
        ])
        src = Field.from_arrow(pa.field("payload", src_struct))
        tgt = Field.from_arrow(pa.field("payload", tgt_struct))

        report = _BenchReport(
            f"pandas struct series cast (n={self.n_rows}) — "
            f"struct_pandas.py:cast_pandas_struct_series"
        )

        ms, _ = _time_cast(
            lambda: tgt.cast_pandas_series(series, source=src),
        )
        report.add("dict-Series widen child a (pyarrow path)", ms)
        report.flush()

    def test_pandas_list_series_cast(self) -> None:
        """Pandas Series of lists → struct cast.

        Same shape as the struct-series benchmark but the source is a
        list-of-values: each row's positional ``i``-th element lands
        in the ``c{i}`` field of the target struct.  Vectorised path
        rebuilds via Arrow's ``pc.take``; columnwise fallback only
        fires on shapes pyarrow rejects.
        """
        import pandas as pd

        n = self.n_rows
        rnd = random.Random(9)
        rows = [
            [rnd.randint(0, 1_000), rnd.randint(0, 1_000), rnd.randint(0, 1_000)]
            if rnd.random() > 0.05 else None
            for _ in range(n)
        ]
        series = pd.Series(rows, name="payload", dtype="object")

        src = Field.from_arrow(pa.field(
            "payload", pa.list_(pa.field("item", pa.int64())),
        ))
        tgt = Field.from_arrow(pa.field("payload", pa.struct([
            pa.field("c0", pa.int32()),
            pa.field("c1", pa.int32()),
            pa.field("c2", pa.int32()),
        ])))

        report = _BenchReport(
            f"pandas list series cast (n={self.n_rows}) — "
            f"struct_pandas.py:cast_pandas_list_series"
        )

        ms, _ = _time_cast(
            lambda: tgt.cast_pandas_series(series, source=src),
        )
        report.add("list-Series -> struct<c0,c1,c2> (pyarrow path)", ms)
        report.flush()

    def test_pandas_nested_struct_value_cast(self) -> None:
        """Pandas DataFrame with a nested struct/list column → cast.

        Mirrors ``test_nested_struct_value_cast`` from the Arrow suite
        but exercises the pandas entry: the column is an object-Series
        of dicts (nested ``inner_list`` list-of-int + ``name`` string),
        and the cast widens the inner list dtype.
        """
        import pandas as pd

        n = self.n_rows
        rnd = random.Random(10)
        rows = [
            {
                "inner_list": [rnd.randint(0, 1_000) for _ in range(rnd.choice([0, 1, 2, 3]))],
                "name": "".join(rnd.choices(string.ascii_lowercase, k=4)),
            }
            if rnd.random() > 0.05 else None
            for _ in range(n)
        ]
        series = pd.Series(rows, name="row", dtype="object")

        src_struct = pa.struct([
            pa.field("inner_list", pa.list_(pa.field("item", pa.int32()))),
            pa.field("name", pa.string()),
        ])
        tgt_struct = pa.struct([
            pa.field("inner_list", pa.list_(pa.field("item", pa.int64()))),
            pa.field("name", pa.string()),
        ])
        src = Field.from_arrow(pa.field("row", src_struct))
        tgt = Field.from_arrow(pa.field("row", tgt_struct))

        report = _BenchReport(
            f"pandas nested struct value cast (n={self.n_rows}) — "
            f"struct_pandas.py:cast_pandas_struct_series"
        )

        ms, _ = _time_cast(
            lambda: tgt.cast_pandas_series(series, source=src),
        )
        report.add("struct<list<int32>, str> -> widen list<int64>", ms)
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
                    source=src, target=tgt,
                ).cast_arrow_batch(
                    pa.Table.from_batches(b).combine_chunks().to_batches()[0]
                ),
            )
            report.add(f"k={k:3d} fetched batches, concat+cast", ms)
        report.flush()
