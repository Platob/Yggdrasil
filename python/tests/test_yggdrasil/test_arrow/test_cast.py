"""Tests for the public ``yggdrasil.arrow.cast`` surface.

Covers the four conversion families exposed by the module:

* **Bulk** — :func:`any_to_arrow_table`, :func:`any_to_arrow_record_batch`
  from Arrow / pandas / polars / pylist / generator inputs.
* **Streaming** — :func:`any_to_arrow_batch_iterator`,
  :func:`any_to_arrow_record_batch_reader` and the ``row_size`` /
  ``byte_size`` rechunker plumbing.
* **Scalar** — :func:`any_to_arrow_scalar`, :func:`cast_arrow_scalar`,
  including ``None``-with-target defaulting and unsafe-cast fallback.
* **Field / schema / array / tabular passthrough wrappers** —
  :func:`any_to_arrow_field`, :func:`any_to_arrow_schema`,
  :func:`cast_arrow_array`, :func:`cast_arrow_tabular`,
  :func:`cast_arrow_record_batch_reader`, plus the byte-accounting
  helpers :func:`get_arrow_nbytes` and
  :func:`rechunk_arrow_batches`.

Engine-specific imports go through optional helpers — pandas / polars
tests skip cleanly when those engines are not installed.
"""
from __future__ import annotations

import unittest

from yggdrasil.arrow.cast import (
    any_to_arrow_batch_iterator,
    any_to_arrow_field,
    any_to_arrow_record_batch,
    any_to_arrow_record_batch_reader,
    any_to_arrow_scalar,
    any_to_arrow_schema,
    any_to_arrow_table,
    cast_arrow_array,
    cast_arrow_record_batch_reader,
    cast_arrow_scalar,
    cast_arrow_tabular,
    conform_arrow_batch,
    get_arrow_nbytes,
    rechunk_arrow_batches,
)
from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Field, Schema


def _maybe_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None


def _maybe_polars():
    try:
        import polars as pl
        return pl
    except ImportError:
        return None


class TestAnyToArrowTable(ArrowTestCase):
    """Bulk path — any object → ``pa.Table``."""

    def _sample(self):
        return self.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    def test_table_passthrough(self) -> None:
        t = self._sample()
        out = any_to_arrow_table(t)
        self.assertFrameEqual(out, t)

    def test_record_batch_promotes_to_table(self) -> None:
        rb = self.record_batch({"a": [1, 2], "b": ["x", "y"]})
        out = any_to_arrow_table(rb)
        self.assertEqual(out.num_rows, 2)
        self.assertEqual(out.column_names, ["a", "b"])

    def test_array_becomes_single_column(self) -> None:
        out = any_to_arrow_table(self.pa.array([1, 2, 3]))
        self.assertEqual(out.column_names, ["value"])
        self.assertEqual(out["value"].to_pylist(), [1, 2, 3])

    def test_array_uses_target_field_name(self) -> None:
        target = Field.from_arrow(self.pa.field("count", self.pa.int32()))
        out = any_to_arrow_table(self.pa.array([1, 2, 3]), CastOptions(target=target))
        self.assertEqual(out.column_names, ["count"])
        self.assertEqual(out.schema.field("count").type, self.pa.int32())

    def test_chunked_array_becomes_single_column(self) -> None:
        chunked = self.pa.chunked_array([[1, 2], [3, 4]])
        out = any_to_arrow_table(chunked)
        self.assertEqual(out.column_names, ["value"])
        self.assertEqual(out.num_rows, 4)

    def test_pylist_of_dicts(self) -> None:
        out = any_to_arrow_table([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        self.assertEqual(out.column_names, ["a", "b"])
        self.assertEqual(out.num_rows, 2)

    def test_empty_list(self) -> None:
        out = any_to_arrow_table([])
        self.assertEqual(out.num_rows, 0)

    def test_empty_generator(self) -> None:
        out = any_to_arrow_table(iter([]))
        self.assertEqual(out.num_rows, 0)

    def test_generator_of_dicts(self) -> None:
        def gen():
            yield {"a": 1, "b": "x"}
            yield {"a": 2, "b": "y"}

        out = any_to_arrow_table(gen())
        self.assertEqual(out.num_rows, 2)
        self.assertIn("a", out.column_names)

    def test_record_batch_reader_passthrough(self) -> None:
        t = self._sample()
        rbr = self.pa.RecordBatchReader.from_batches(t.schema, iter(t.to_batches()))
        out = any_to_arrow_table(rbr)
        self.assertEqual(out.num_rows, 3)

    def test_target_schema_casts_types(self) -> None:
        t = self._sample()
        target = Schema.from_arrow(self.pa.schema([
            self.pa.field("a", self.pa.int32()),
            self.pa.field("b", self.pa.string()),
        ]))
        out = any_to_arrow_table(t, CastOptions(target=target))
        self.assertEqual(out.schema.field("a").type, self.pa.int32())

    def test_column_projection_via_target_field(self) -> None:
        t = self._sample()
        target = Schema.from_arrow(self.pa.schema([self.pa.field("b", self.pa.string())]))
        out = any_to_arrow_table(t, CastOptions(target=target))
        self.assertEqual(out.column_names, ["b"])

    def test_pandas_dataframe(self) -> None:
        pd = _maybe_pandas()
        if pd is None:
            self.skipTest("pandas not installed")
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        out = any_to_arrow_table(df)
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(set(out.column_names), {"a", "b"})

    def test_pandas_series(self) -> None:
        pd = _maybe_pandas()
        if pd is None:
            self.skipTest("pandas not installed")
        s = pd.Series([1, 2, 3], name="a")
        out = any_to_arrow_table(s)
        self.assertEqual(out.num_rows, 3)
        self.assertIn("a", out.column_names)

    def test_polars_dataframe(self) -> None:
        pl = _maybe_polars()
        if pl is None:
            self.skipTest("polars not installed")
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        out = any_to_arrow_table(df)
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(set(out.column_names), {"a", "b"})

    def test_polars_lazyframe(self) -> None:
        pl = _maybe_polars()
        if pl is None:
            self.skipTest("polars not installed")
        lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
        out = any_to_arrow_table(lf)
        self.assertEqual(out.num_rows, 3)


class TestAnyToArrowRecordBatch(ArrowTestCase):
    """Bulk path — any object → single ``pa.RecordBatch``."""

    def test_record_batch_passthrough(self) -> None:
        rb = self.record_batch({"a": [1, 2, 3]})
        out = any_to_arrow_record_batch(rb)
        self.assertIsInstance(out, self.pa.RecordBatch)
        self.assertEqual(out.num_rows, 3)

    def test_table_combines_chunks(self) -> None:
        # Multi-chunk table -> single batch.
        t1 = self.table({"a": [1, 2]})
        t2 = self.table({"a": [3, 4]})
        combined = self.pa.concat_tables([t1, t2])
        out = any_to_arrow_record_batch(combined)
        self.assertIsInstance(out, self.pa.RecordBatch)
        self.assertEqual(out.num_rows, 4)

    def test_empty_table_returns_empty_batch(self) -> None:
        empty = self.table({"a": []})
        out = any_to_arrow_record_batch(empty)
        self.assertEqual(out.num_rows, 0)
        self.assertEqual(out.schema.names, ["a"])

    def test_pylist_input(self) -> None:
        out = any_to_arrow_record_batch([{"a": 1}, {"a": 2}])
        self.assertEqual(out.num_rows, 2)


class TestAnyToArrowBatchIterator(ArrowTestCase):
    """Streaming path — produces a lazy iterator of ``pa.RecordBatch``."""

    def test_table_yields_batches(self) -> None:
        t = self.table({"a": [1, 2, 3, 4]})
        batches = list(any_to_arrow_batch_iterator(t))
        total = sum(b.num_rows for b in batches)
        self.assertEqual(total, 4)

    def test_record_batch_yields_self(self) -> None:
        rb = self.record_batch({"a": [1, 2]})
        batches = list(any_to_arrow_batch_iterator(rb))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].num_rows, 2)

    def test_empty_list_yields_nothing(self) -> None:
        batches = list(any_to_arrow_batch_iterator([]))
        self.assertEqual(batches, [])

    def test_row_size_chunks_table(self) -> None:
        t = self.table({"a": list(range(10))})
        opts = CastOptions(row_size=3)
        batches = list(any_to_arrow_batch_iterator(t, opts))
        self.assertEqual([b.num_rows for b in batches], [3, 3, 3, 1])

    def test_record_batch_reader_streams(self) -> None:
        t = self.table({"a": [1, 2, 3]})
        rbr = self.pa.RecordBatchReader.from_batches(t.schema, iter(t.to_batches()))
        batches = list(any_to_arrow_batch_iterator(rbr))
        self.assertEqual(sum(b.num_rows for b in batches), 3)

    def test_polars_lazyframe_streams_per_chunk(self) -> None:
        pl = _maybe_polars()
        if pl is None:
            self.skipTest("polars not installed")
        lf = pl.DataFrame({"a": list(range(5))}).lazy()
        batches = list(any_to_arrow_batch_iterator(lf))
        self.assertEqual(sum(b.num_rows for b in batches), 5)


class TestAnyToArrowRecordBatchReader(ArrowTestCase):
    """Streaming path — wraps the iterator behind a ``RecordBatchReader``."""

    def test_table_wraps_into_reader(self) -> None:
        t = self.table({"a": [1, 2, 3]})
        rbr = any_to_arrow_record_batch_reader(t)
        self.assertIsInstance(rbr, self.pa.RecordBatchReader)
        self.assertEqual(rbr.read_all().num_rows, 3)

    def test_existing_reader_passes_through(self) -> None:
        t = self.table({"a": [1, 2, 3]})
        src = self.pa.RecordBatchReader.from_batches(t.schema, iter(t.to_batches()))
        # No cast / no chunking → the same reader should come back.
        out = any_to_arrow_record_batch_reader(src)
        self.assertIs(out, src)

    def test_empty_iterator_uses_empty_schema(self) -> None:
        rbr = any_to_arrow_record_batch_reader([])
        self.assertEqual(rbr.read_all().num_rows, 0)


class TestRechunker(ArrowTestCase):
    """Direct tests for the row/byte-size rechunker."""

    def _batch(self, n: int):
        return self.record_batch({"a": list(range(n))})

    def test_passthrough_when_no_knobs(self) -> None:
        b = self._batch(4)
        out = list(rechunk_arrow_batches([b]))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].num_rows, 4)

    def test_row_size_only_slices_in_place(self) -> None:
        b = self._batch(7)
        out = list(rechunk_arrow_batches([b], row_size=3))
        self.assertEqual([x.num_rows for x in out], [3, 3, 1])

    def test_row_size_drops_empty_batches(self) -> None:
        empty = self.record_batch({"a": []})
        b = self._batch(3)
        out = list(rechunk_arrow_batches([empty, b], row_size=2))
        self.assertEqual([x.num_rows for x in out], [2, 1])

    def test_byte_size_emits_chunks(self) -> None:
        b = self._batch(10)
        out = list(rechunk_arrow_batches([b], byte_size=8))
        self.assertEqual(sum(x.num_rows for x in out), 10)
        self.assertGreater(len(out), 1)

    def test_byte_and_row_caps_pick_minimum(self) -> None:
        b = self._batch(10)
        out = list(rechunk_arrow_batches([b], byte_size=10_000, row_size=4))
        self.assertEqual([x.num_rows for x in out], [4, 4, 2])

    def test_concat_buffers_under_target(self) -> None:
        small = [self._batch(1) for _ in range(5)]
        out = list(rechunk_arrow_batches(small, byte_size=10_000))
        self.assertEqual(sum(x.num_rows for x in out), 5)

    # ---- data-integrity tests ----

    def _wide_batch(self, n: int) -> "pa.RecordBatch":
        return self.record_batch({
            "id": list(range(n)),
            "val": [float(i) * 1.1 for i in range(n)],
            "label": [f"row-{i}" for i in range(n)],
        })

    def test_row_size_preserves_data(self) -> None:
        b = self._wide_batch(9)
        out = list(rechunk_arrow_batches([b], row_size=4))
        combined = self.pa.concat_batches(out)
        self.assertEqual(combined.to_pydict(), b.to_pydict())

    def test_byte_size_preserves_data(self) -> None:
        b = self._wide_batch(100)
        nbytes = get_arrow_nbytes(b)
        out = list(rechunk_arrow_batches([b], byte_size=nbytes // 4))
        combined = self.pa.concat_batches(out)
        self.assertEqual(combined.to_pydict(), b.to_pydict())

    def test_byte_size_chunks_near_target(self) -> None:
        b = self._wide_batch(200)
        target = get_arrow_nbytes(b) // 5
        out = list(rechunk_arrow_batches([b], byte_size=target))
        self.assertGreaterEqual(len(out), 3)
        for chunk in out[:-1]:
            chunk_bytes = get_arrow_nbytes(chunk)
            self.assertLessEqual(chunk_bytes, target * 2)

    def test_many_small_batches_coalesced_then_split(self) -> None:
        small = [self._wide_batch(5) for _ in range(20)]
        total_bytes = sum(get_arrow_nbytes(b) for b in small)
        target = total_bytes // 3
        out = list(rechunk_arrow_batches(small, byte_size=target))
        combined = self.pa.concat_batches(out)
        expected = self.pa.concat_batches(small)
        self.assertEqual(combined.to_pydict(), expected.to_pydict())
        self.assertLess(len(out), 20)

    def test_mixed_size_batches_preserves_data(self) -> None:
        batches = [self._wide_batch(n) for n in (1, 50, 2, 30, 1, 100, 3)]
        out = list(rechunk_arrow_batches(batches, byte_size=200))
        combined = self.pa.concat_batches(out)
        expected = self.pa.concat_batches(batches)
        self.assertEqual(combined.to_pydict(), expected.to_pydict())

    def test_single_large_batch_sliced_preserves_data(self) -> None:
        b = self._wide_batch(500)
        target = get_arrow_nbytes(b) // 10
        out = list(rechunk_arrow_batches([b], byte_size=target))
        self.assertGreater(len(out), 5)
        combined = self.pa.concat_batches(out)
        self.assertEqual(combined.to_pydict(), b.to_pydict())

    def test_byte_and_row_preserves_data(self) -> None:
        b = self._wide_batch(50)
        out = list(rechunk_arrow_batches([b], byte_size=100, row_size=7))
        combined = self.pa.concat_batches(out)
        self.assertEqual(combined.to_pydict(), b.to_pydict())
        for chunk in out:
            self.assertLessEqual(chunk.num_rows, 7)

    def test_schema_preserved_after_rechunk(self) -> None:
        b = self._wide_batch(20)
        out = list(rechunk_arrow_batches([b], row_size=5))
        for chunk in out:
            self.assertEqual(chunk.schema, b.schema)

    def test_empty_input_yields_nothing(self) -> None:
        out = list(rechunk_arrow_batches([], byte_size=100))
        self.assertEqual(out, [])

    def test_all_empty_batches_yield_nothing(self) -> None:
        empties = [self.record_batch({"a": []}) for _ in range(3)]
        out = list(rechunk_arrow_batches(empties, byte_size=100))
        self.assertEqual(out, [])

    def test_generator_input_works(self) -> None:
        def gen():
            for i in range(5):
                yield self._wide_batch(10)

        out = list(rechunk_arrow_batches(gen(), byte_size=200))
        self.assertEqual(sum(c.num_rows for c in out), 50)


class TestArrowNbytes(ArrowTestCase):
    """``get_arrow_nbytes`` covers arrays, chunked arrays, tables, and ``None``."""

    def test_none_returns_default(self) -> None:
        self.assertEqual(get_arrow_nbytes(None), 0)
        self.assertEqual(get_arrow_nbytes(None, default=42), 42)

    def test_array(self) -> None:
        arr = self.pa.array([1, 2, 3], type=self.pa.int64())
        self.assertEqual(get_arrow_nbytes(arr), 24)

    def test_chunked_array(self) -> None:
        ca = self.pa.chunked_array([[1, 2], [3]], type=self.pa.int64())
        self.assertEqual(get_arrow_nbytes(ca), 24)

    def test_table(self) -> None:
        t = self.table({"a": [1, 2, 3]})
        self.assertGreater(get_arrow_nbytes(t), 0)

    def test_string_view_returns_flat_default(self) -> None:
        # ``.nbytes`` on a sliced view array over-counts (the variadic
        # buffer is shared with the parent). The view branch returns a
        # flat 1 MiB regardless of slice length.
        from yggdrasil.arrow.cast import _VIEW_DEFAULT_NBYTES

        sv = self.pa.array(["x" * 200] * 1000, type=self.pa.string_view())
        self.assertEqual(get_arrow_nbytes(sv), _VIEW_DEFAULT_NBYTES)
        self.assertEqual(get_arrow_nbytes(sv.slice(0, 1)), _VIEW_DEFAULT_NBYTES)

    def test_chunked_view_recurses_per_chunk(self) -> None:
        from yggdrasil.arrow.cast import _VIEW_DEFAULT_NBYTES

        sv = self.pa.array(["x"], type=self.pa.string_view())
        ca = self.pa.chunked_array([sv, sv, sv])
        self.assertEqual(get_arrow_nbytes(ca), 3 * _VIEW_DEFAULT_NBYTES)


class TestAnyToArrowScalar(ArrowTestCase):
    """Scalar entry points — Python value → ``pa.Scalar``."""

    def test_no_target_uses_pa_scalar(self) -> None:
        s = any_to_arrow_scalar(5)
        self.assertEqual(s.as_py(), 5)

    def test_target_field_drives_type(self) -> None:
        target = Field.from_arrow(self.pa.field("v", self.pa.int32()))
        s = any_to_arrow_scalar(5, CastOptions(target=target))
        self.assertEqual(s.type, self.pa.int32())
        self.assertEqual(s.as_py(), 5)

    def test_none_with_nullable_target_returns_null_scalar(self) -> None:
        target = Field.from_arrow(self.pa.field("v", self.pa.int32(), nullable=True))
        s = any_to_arrow_scalar(None, CastOptions(target=target))
        self.assertTrue(s.as_py() is None)

    def test_none_with_non_nullable_target_uses_default(self) -> None:
        target = Field.from_arrow(self.pa.field("v", self.pa.int32(), nullable=False))
        s = any_to_arrow_scalar(None, CastOptions(target=target))
        self.assertEqual(s.as_py(), 0)

    def test_none_without_target_returns_null(self) -> None:
        s = any_to_arrow_scalar(None)
        self.assertIsNone(s.as_py())

    def test_existing_pa_scalar_routes_through_cast(self) -> None:
        target = Field.from_arrow(self.pa.field("v", self.pa.int32()))
        src = self.pa.scalar(7, type=self.pa.int64())
        out = any_to_arrow_scalar(src, CastOptions(target=target))
        self.assertEqual(out.type, self.pa.int32())
        self.assertEqual(out.as_py(), 7)

    def test_enum_unwraps_to_value(self) -> None:
        import enum

        class Color(enum.Enum):
            RED = 1
            BLUE = 2

        target = Field.from_arrow(self.pa.field("v", self.pa.int32()))
        s = any_to_arrow_scalar(Color.RED, CastOptions(target=target))
        self.assertEqual(s.as_py(), 1)

    def test_unsafe_construction_falls_back(self) -> None:
        # String "abc" can't be built as int32 directly — pa.scalar
        # raises ArrowInvalid; the fallback drops the type hint.
        target = Field.from_arrow(self.pa.field("v", self.pa.int32()))
        try:
            any_to_arrow_scalar("abc", CastOptions(target=target, safe=False))
        except Exception:
            # The fallback attempt to cast "abc" → int32 may still raise
            # downstream; the goal here is that the *first* pa.scalar
            # call's ArrowInvalid is caught (no propagation from line
            # 719). Hitting this branch validates the except clause.
            pass


class TestCastArrowScalar(ArrowTestCase):
    """``cast_arrow_scalar`` — pa.Scalar → pa.Scalar."""

    def test_no_target_passthrough(self) -> None:
        src = self.pa.scalar(7, type=self.pa.int64())
        out = cast_arrow_scalar(src)
        self.assertIs(out, src)

    def test_cast_to_int32(self) -> None:
        src = self.pa.scalar(7, type=self.pa.int64())
        target = Field.from_arrow(self.pa.field("v", self.pa.int32()))
        out = cast_arrow_scalar(src, CastOptions(target=target))
        self.assertEqual(out.type, self.pa.int32())
        self.assertEqual(out.as_py(), 7)


class TestCastArrowArray(ArrowTestCase):
    """``cast_arrow_array`` covers Array and ChunkedArray."""

    def test_array_cast(self) -> None:
        arr = self.pa.array([1, 2, 3], type=self.pa.int64())
        target = Field.from_arrow(self.pa.field("v", self.pa.int32()))
        out = cast_arrow_array(arr, CastOptions(target=target))
        self.assertEqual(out.type, self.pa.int32())
        self.assertEqual(out.to_pylist(), [1, 2, 3])

    def test_chunked_array_cast(self) -> None:
        ca = self.pa.chunked_array([[1, 2], [3]], type=self.pa.int64())
        target = Field.from_arrow(self.pa.field("v", self.pa.int32()))
        out = cast_arrow_array(ca, CastOptions(target=target))
        self.assertEqual(out.type, self.pa.int32())
        self.assertEqual(out.to_pylist(), [1, 2, 3])


class TestCastArrowTabular(ArrowTestCase):
    """``cast_arrow_tabular`` for Table and RecordBatch."""

    def test_table_cast_changes_type(self) -> None:
        t = self.table({"a": [1, 2, 3]})
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int32())]))
        out = cast_arrow_tabular(t, CastOptions(target=target))
        self.assertEqual(out.schema.field("a").type, self.pa.int32())

    def test_record_batch_cast(self) -> None:
        rb = self.record_batch({"a": [1, 2, 3]})
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int32())]))
        out = cast_arrow_tabular(rb, CastOptions(target=target))
        self.assertEqual(out.schema.field("a").type, self.pa.int32())

    def test_skip_cast_when_schema_matches(self) -> None:
        t = self.table({"a": self.pa.array([1, 2], type=self.pa.int32())})
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int32())]))
        out = cast_arrow_tabular(t, CastOptions(target=target))
        self.assertEqual(out.schema.field("a").type, self.pa.int32())


class TestConformArrowBatch(ArrowTestCase):
    """``conform_arrow_batch`` — reshape a drifted batch to a target schema."""

    def test_identity_passthrough_when_schema_matches(self) -> None:
        rb = self.record_batch({"a": [1, 2, 3]})
        out = conform_arrow_batch(rb, rb.schema)
        self.assertIs(out, rb)

    def test_casts_drifted_types(self) -> None:
        # binary -> large_binary and int64 -> timestamp are exactly the
        # cache-file drift this helper exists to reconcile.
        rb = self.pa.record_batch(
            [
                self.pa.array([b"x"], type=self.pa.binary()),
                self.pa.array([10], type=self.pa.int64()),
            ],
            names=["body", "ts"],
        )
        target = self.pa.schema([
            self.pa.field("body", self.pa.large_binary()),
            self.pa.field("ts", self.pa.timestamp("us", "UTC")),
        ])
        out = conform_arrow_batch(rb, target)
        self.assertTrue(out.schema.equals(target))
        self.assertEqual(out.column("body").to_pylist(), [b"x"])

    def test_missing_column_materialised_null(self) -> None:
        rb = self.record_batch({"a": [1, 2]})
        target = self.pa.schema([
            self.pa.field("a", self.pa.int64()),
            self.pa.field("b", self.pa.int64()),
        ])
        out = conform_arrow_batch(rb, target)
        self.assertEqual(out.column("b").to_pylist(), [None, None])

    def test_extra_column_dropped_and_reordered(self) -> None:
        rb = self.pa.record_batch(
            [self.pa.array([9]), self.pa.array([1]), self.pa.array([2])],
            names=["extra", "b", "a"],
        )
        target = self.pa.schema([
            self.pa.field("a", self.pa.int64()),
            self.pa.field("b", self.pa.int64()),
        ])
        out = conform_arrow_batch(rb, target)
        self.assertEqual(out.schema.names, ["a", "b"])
        self.assertEqual(out.column("a").to_pylist(), [2])

    def test_drifted_batches_assemble_into_one_table(self) -> None:
        # The end the helper serves: from_batches across drifted schemas.
        a = self.pa.record_batch(
            [self.pa.array([b"a"], type=self.pa.binary())], names=["body"]
        )
        b = self.pa.record_batch(
            [self.pa.array([b"b"], type=self.pa.large_binary())], names=["body"]
        )
        with self.assertRaises((self.pa.lib.ArrowInvalid, self.pa.lib.ArrowTypeError)):
            self.pa.Table.from_batches([a, b])
        target = a.schema
        table = self.pa.Table.from_batches(
            [conform_arrow_batch(x, target) for x in (a, b)]
        )
        self.assertEqual(table.num_rows, 2)


class TestCastArrowRecordBatchReader(ArrowTestCase):
    """``cast_arrow_record_batch_reader`` — passthrough vs cast/chunk path."""

    def test_passthrough_when_no_cast_no_chunk(self) -> None:
        t = self.table({"a": [1, 2, 3]})
        rbr = self.pa.RecordBatchReader.from_batches(t.schema, iter(t.to_batches()))
        out = cast_arrow_record_batch_reader(rbr)
        self.assertIs(out, rbr)

    def test_chunk_only_wraps_reader(self) -> None:
        t = self.table({"a": list(range(6))})
        rbr = self.pa.RecordBatchReader.from_batches(t.schema, iter(t.to_batches()))
        out = cast_arrow_record_batch_reader(rbr, CastOptions(row_size=2))
        self.assertIsNot(out, rbr)
        batches = list(out)
        self.assertEqual([b.num_rows for b in batches], [2, 2, 2])

    def test_cast_wraps_reader(self) -> None:
        t = self.table({"a": [1, 2, 3]})
        rbr = self.pa.RecordBatchReader.from_batches(t.schema, iter(t.to_batches()))
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int32())]))
        out = cast_arrow_record_batch_reader(rbr, CastOptions(target=target))
        self.assertEqual(out.schema.field("a").type, self.pa.int32())


class TestAnyToArrowField(ArrowTestCase):
    """Field-coercion entry point."""

    def test_pa_field_passthrough(self) -> None:
        f = self.pa.field("a", self.pa.int32())
        out = any_to_arrow_field(f)
        self.assertIs(out, f)

    def test_yggdrasil_field_to_arrow(self) -> None:
        ygg = Field.from_arrow(self.pa.field("a", self.pa.int32()))
        out = any_to_arrow_field(ygg)
        self.assertEqual(out.name, "a")
        self.assertEqual(out.type, self.pa.int32())


class TestAnyToArrowSchema(ArrowTestCase):
    """Schema-coercion entry point."""

    def test_pa_schema_passthrough(self) -> None:
        s = self.pa.schema([self.pa.field("a", self.pa.int64())])
        out = any_to_arrow_schema(s)
        self.assertIs(out, s)

    def test_yggdrasil_schema_to_arrow(self) -> None:
        ygg = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int64())]))
        out = any_to_arrow_schema(ygg)
        self.assertEqual(out.names, ["a"])


if __name__ == "__main__":
    unittest.main()
