"""Pandas → Arrow conversion tests.

Exercises the pandas branch of :func:`yggdrasil.arrow.cast.any_to_arrow_table`
(which reaches :func:`_pandas_to_arrow`) plus the streaming wrapper. Goal
is to confirm:

* DataFrame and Series both flow through.
* In-engine cast (``CastOptions.cast_pandas``) actually fires before
  serialization — pandas extension dtypes survive instead of going
  through a lossy numpy detour.
* Target-field projection / type cast / nullable-fill behave the way
  the cast pipeline promises.
* Streaming entry points emit batches that round-trip back to the
  same data.

Multi-inherits :class:`PandasTestCase` and :class:`ArrowTestCase` so a
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
from yggdrasil.pandas.tests import PandasTestCase


class TestPandasDataFrameToArrow(PandasTestCase, ArrowTestCase):
    def test_basic_dataframe(self) -> None:
        df = self.df({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        out = any_to_arrow_table(df)
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(set(out.column_names), {"a", "b"})
        self.assertEqual(out["a"].to_pylist(), [1, 2, 3])
        self.assertEqual(out["b"].to_pylist(), ["x", "y", "z"])

    def test_empty_dataframe_preserves_columns(self) -> None:
        df = self.df({"a": [], "b": []})
        out = any_to_arrow_table(df)
        self.assertEqual(out.num_rows, 0)
        self.assertEqual(set(out.column_names), {"a", "b"})

    def test_target_field_casts_int64_to_int32(self) -> None:
        df = self.df({"a": [1, 2, 3]})
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int32())]))
        out = any_to_arrow_table(df, CastOptions(target_field=target))
        self.assertEqual(out.schema.field("a").type, self.pa.int32())
        self.assertEqual(out["a"].to_pylist(), [1, 2, 3])

    def test_target_field_projects_subset(self) -> None:
        df = self.df({"a": [1, 2], "b": ["x", "y"], "c": [True, False]})
        target = Schema.from_arrow(self.pa.schema([
            self.pa.field("c", self.pa.bool_()),
            self.pa.field("a", self.pa.int64()),
        ]))
        out = any_to_arrow_table(df, CastOptions(target_field=target))
        self.assertEqual(out.column_names, ["c", "a"])

    def test_nullable_int_extension_dtype(self) -> None:
        # pandas Int64 is the nullable extension dtype — without the
        # in-engine cast path, this trips the "via numpy → lose NaN"
        # bug.  With cast_pandas, the NA survives as a real null.
        df = self.df({"a": self.pd.array([1, None, 3], dtype="Int64")})
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.int64())]))
        out = any_to_arrow_table(df, CastOptions(target_field=target))
        self.assertEqual(out["a"].to_pylist(), [1, None, 3])

    def test_string_dtype(self) -> None:
        df = self.df({"a": self.pd.array(["x", None, "z"], dtype="string")})
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.string())]))
        out = any_to_arrow_table(df, CastOptions(target_field=target))
        self.assertEqual(out["a"].to_pylist(), ["x", None, "z"])

    def test_bool_column(self) -> None:
        df = self.df({"a": [True, False, True]})
        out = any_to_arrow_table(df)
        self.assertEqual(out.schema.field("a").type, self.pa.bool_())
        self.assertEqual(out["a"].to_pylist(), [True, False, True])

    def test_float_column(self) -> None:
        df = self.df({"a": [1.5, 2.5, 3.5]})
        target = Schema.from_arrow(self.pa.schema([self.pa.field("a", self.pa.float32())]))
        out = any_to_arrow_table(df, CastOptions(target_field=target))
        self.assertEqual(out.schema.field("a").type, self.pa.float32())

    def test_named_index_is_preserved(self) -> None:
        # ``preserve_index`` is wired to ``bool(df.index.name)`` in
        # ``_pandas_to_arrow`` — a named index should round-trip as a
        # column.
        df = self.df({"a": [1, 2, 3]})
        df.index = self.pd.Index([10, 20, 30], name="idx")
        out = any_to_arrow_table(df)
        self.assertIn("idx", out.column_names)

    def test_unnamed_index_is_dropped(self) -> None:
        df = self.df({"a": [1, 2, 3]})
        out = any_to_arrow_table(df)
        self.assertNotIn("__index_level_0__", out.column_names)
        self.assertEqual(out.column_names, ["a"])

    def test_record_batch_path(self) -> None:
        df = self.df({"a": [1, 2, 3]})
        rb = any_to_arrow_record_batch(df)
        self.assertEqual(rb.num_rows, 3)
        self.assertIn("a", rb.schema.names)

    def test_batch_iterator_round_trips(self) -> None:
        df = self.df({"a": list(range(10))})
        batches = list(any_to_arrow_batch_iterator(df))
        total = sum(b.num_rows for b in batches)
        self.assertEqual(total, 10)

    def test_batch_iterator_with_row_size(self) -> None:
        df = self.df({"a": list(range(7))})
        batches = list(any_to_arrow_batch_iterator(df, CastOptions(row_size=3)))
        self.assertEqual([b.num_rows for b in batches], [3, 3, 1])

    def test_record_batch_reader(self) -> None:
        df = self.df({"a": [1, 2, 3]})
        rbr = any_to_arrow_record_batch_reader(df)
        self.assertEqual(rbr.read_all().num_rows, 3)


class TestPandasSeriesToArrow(PandasTestCase, ArrowTestCase):
    def test_named_series(self) -> None:
        s = self.series([1, 2, 3], name="vals")
        out = any_to_arrow_table(s)
        self.assertEqual(out.num_rows, 3)
        self.assertIn("vals", out.column_names)

    def test_unnamed_series(self) -> None:
        # ``Series.to_frame()`` defaults to column name 0 when unnamed
        # — pyarrow will accept it but coerces to string in the schema.
        s = self.series([1, 2, 3])
        out = any_to_arrow_table(s)
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(len(out.column_names), 1)

    def test_series_target_cast(self) -> None:
        s = self.series([1, 2, 3], name="vals")
        target = Schema.from_arrow(self.pa.schema([self.pa.field("vals", self.pa.int16())]))
        out = any_to_arrow_table(s, CastOptions(target_field=target))
        self.assertEqual(out.schema.field("vals").type, self.pa.int16())

    def test_series_with_nulls(self) -> None:
        s = self.pd.Series([1.0, None, 3.0], name="vals")
        out = any_to_arrow_table(s)
        self.assertEqual(out["vals"].to_pylist(), [1.0, None, 3.0])


if __name__ == "__main__":
    unittest.main()
