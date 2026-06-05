"""Polars → Arrow conversion tests.

Exercises the polars branch of :func:`yggdrasil.arrow.cast.any_to_arrow_table`
(which reaches :func:`_polars_to_arrow`/:func:`_polars_eager_to_arrow`)
and :func:`any_to_arrow_batch_iterator` (which reaches
:func:`_polars_lazy_to_batch_iterator`). Goal is to confirm:

* DataFrame, Series, and LazyFrame inputs all flow through.
* In-engine cast (``CastOptions.cast_polars``) fuses into the plan
  before serialization — so the output schema matches the target
  without an Arrow-side rebuild.
* Lazy streaming via ``collect_batches(lazy=True)`` yields multiple
  batches whose row-counts sum to the source size, and respects
  ``row_size``.
* Categorical-friendly types (``Utf8`` / ``string_view``) survive the
  Arrow round trip.

Multi-inherits :class:`PolarsTestCase` and :class:`ArrowTestCase` so a
missing optional dependency on either side skips the class cleanly.
"""
from __future__ import annotations

import unittest

from yggdrasil.arrow.cast import (
    any_to_arrow_batch_iterator,
    any_to_arrow_record_batch,
    any_to_arrow_record_batch_reader,
    any_to_arrow_table,
)
from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.polars.tests import PolarsTestCase


class TestPolarsDataFrameToArrow(PolarsTestCase, ArrowTestCase):
    def test_basic_dataframe(self) -> None:
        df = self.df({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        out = any_to_arrow_table(df)
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(set(out.column_names), {"a", "b"})
        self.assertEqual(out["a"].to_pylist(), [1, 2, 3])

    def test_empty_dataframe(self) -> None:
        df = self.df({"a": [], "b": []}, schema={"a": self.pl.Int64, "b": self.pl.Utf8})
        out = any_to_arrow_table(df)
        self.assertEqual(out.num_rows, 0)
        self.assertEqual(set(out.column_names), {"a", "b"})

    def test_target_field_casts_int64_to_int32(self) -> None:
        df = self.df({"a": [1, 2, 3]})
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int32())]))
        out = any_to_arrow_table(df, CastOptions(target=target))
        self.assertEqual(out.schema.field("a").type, self.pa.int32())
        self.assertEqual(out["a"].to_pylist(), [1, 2, 3])

    def test_target_field_projects_subset(self) -> None:
        df = self.df({"a": [1, 2], "b": ["x", "y"], "c": [True, False]})
        target = Schema.from_arrow(self.pa.schema([
            self.pa.field("c", self.pa.bool_()),
            self.pa.field("a", self.pa.int64()),
        ]))
        out = any_to_arrow_table(df, CastOptions(target=target))
        self.assertEqual(out.column_names, ["c", "a"])

    def test_string_column(self) -> None:
        df = self.df({"a": ["x", None, "z"]})
        out = any_to_arrow_table(df)
        self.assertEqual(out["a"].to_pylist(), ["x", None, "z"])

    def test_target_cast_float_to_int(self) -> None:
        # Float→int crosses DataTypeId boundaries so need_cast() fires
        # and the in-engine polars cast actually runs (string vs
        # large_string vs string_view all share DataTypeId.STRING and
        # would short-circuit).
        df = self.df({"a": [1.5, 2.5, 3.5]})
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int32())]))
        out = any_to_arrow_table(df, CastOptions(target=target))
        self.assertEqual(out.schema.field("a").type, self.pa.int32())
        self.assertEqual(out["a"].to_pylist(), [1, 2, 3])

    def test_record_batch_path(self) -> None:
        df = self.df({"a": [1, 2, 3]})
        rb = any_to_arrow_record_batch(df)
        self.assertEqual(rb.num_rows, 3)

    def test_record_batch_reader(self) -> None:
        df = self.df({"a": [1, 2, 3]})
        rbr = any_to_arrow_record_batch_reader(df)
        self.assertEqual(rbr.read_all().num_rows, 3)


class TestPolarsSeriesToArrow(PolarsTestCase, ArrowTestCase):
    def test_named_series(self) -> None:
        s = self.series("vals", [1, 2, 3])
        out = any_to_arrow_table(s)
        self.assertEqual(out.num_rows, 3)
        self.assertIn("vals", out.column_names)

    def test_series_target_cast(self) -> None:
        s = self.series("vals", [1, 2, 3])
        target = Schema.from_arrow(self.pa.schema([self.pa.field("vals", self.pa.int16())]))
        out = any_to_arrow_table(s, CastOptions(target=target))
        self.assertEqual(out.schema.field("vals").type, self.pa.int16())

    def test_series_with_nulls(self) -> None:
        s = self.series("vals", [1.0, None, 3.0])
        out = any_to_arrow_table(s)
        self.assertEqual(out["vals"].to_pylist(), [1.0, None, 3.0])


class TestPolarsLazyFrameToArrow(PolarsTestCase, ArrowTestCase):
    def test_basic_lazyframe(self) -> None:
        lf = self.lazy({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        out = any_to_arrow_table(lf)
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(set(out.column_names), {"a", "b"})

    def test_lazyframe_target_cast(self) -> None:
        lf = self.lazy({"a": [1, 2, 3]})
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int32())]))
        out = any_to_arrow_table(lf, CastOptions(target=target))
        self.assertEqual(out.schema.field("a").type, self.pa.int32())

    def test_lazyframe_projection(self) -> None:
        lf = self.lazy({"a": [1, 2], "b": ["x", "y"], "c": [True, False]})
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int64())]))
        out = any_to_arrow_table(lf, CastOptions(target=target))
        self.assertEqual(out.column_names, ["a"])
        self.assertEqual(out["a"].to_pylist(), [1, 2])

    def test_lazyframe_streams_per_chunk(self) -> None:
        lf = self.lazy({"a": list(range(8))})
        batches = list(any_to_arrow_batch_iterator(lf))
        # collect_batches(lazy=True) yields one or more batches.
        self.assertGreaterEqual(len(batches), 1)
        self.assertEqual(sum(b.num_rows for b in batches), 8)

    def test_lazyframe_streams_with_row_size(self) -> None:
        lf = self.lazy({"a": list(range(7))})
        batches = list(any_to_arrow_batch_iterator(lf, CastOptions(row_size=3)))
        self.assertEqual(sum(b.num_rows for b in batches), 7)
        for b in batches:
            self.assertLessEqual(b.num_rows, 3)

    def test_lazyframe_streams_with_target_cast(self) -> None:
        lf = self.lazy({"a": list(range(5))})
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int32())]))
        batches = list(any_to_arrow_batch_iterator(lf, CastOptions(target=target)))
        for b in batches:
            self.assertEqual(b.schema.field("a").type, self.pa.int32())


if __name__ == "__main__":
    unittest.main()
