"""Tests for the frame-shaped lift on :class:`Tabular`.

``Tabular(...)`` and ``Tabular.from_(...)`` accept the shapes a
real caller has on hand — pyarrow Table / RecordBatch, polars
DataFrame / LazyFrame, pandas DataFrame, ``list[dict]`` rows,
``dict[str, list]`` columns, and any iterable of those — and lift
each one to a :class:`MemoryArrowIO`. Skips the polars / pandas /
spark-specific sub-cases when the optional dep is missing.
"""

from __future__ import annotations

import unittest

import pyarrow as pa

from yggdrasil.io.tabular import Tabular
from yggdrasil.io.tabular import MemoryArrowIO


class TestConstructorLift(unittest.TestCase):
    def test_pyarrow_table(self) -> None:
        io = Tabular(pa.table({"a": [1, 2, 3]}))
        self.assertIsInstance(io, MemoryArrowIO)
        self.assertEqual(io.read_arrow_table().num_rows, 3)

    def test_pyarrow_record_batch(self) -> None:
        batch = pa.record_batch([pa.array([1, 2])], names=["a"])
        io = Tabular(batch)
        self.assertIsInstance(io, MemoryArrowIO)
        self.assertEqual(io.read_arrow_table().num_rows, 2)

    def test_list_of_dicts(self) -> None:
        io = Tabular([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        self.assertIsInstance(io, MemoryArrowIO)
        out = io.read_arrow_table()
        self.assertEqual(out.column_names, ["a", "b"])
        self.assertEqual(out.num_rows, 2)

    def test_dict_of_columns(self) -> None:
        io = Tabular({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        self.assertIsInstance(io, MemoryArrowIO)
        self.assertEqual(io.read_arrow_table().num_rows, 3)

    def test_polars_dataframe(self) -> None:
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        io = Tabular(pl.DataFrame({"a": [1, 2, 3]}))
        self.assertIsInstance(io, MemoryArrowIO)
        self.assertEqual(io.read_arrow_table().num_rows, 3)

    def test_polars_lazyframe_collects(self) -> None:
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        io = Tabular(pl.LazyFrame({"a": [1, 2, 3, 4]}))
        self.assertIsInstance(io, MemoryArrowIO)
        self.assertEqual(io.read_arrow_table().num_rows, 4)

    def test_pandas_dataframe(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        io = Tabular(pd.DataFrame({"a": [1, 2]}))
        self.assertIsInstance(io, MemoryArrowIO)
        self.assertEqual(io.read_arrow_table().num_rows, 2)


class TestFromLift(unittest.TestCase):
    def test_from_pyarrow_table(self) -> None:
        io = Tabular.from_(pa.table({"a": [1, 2]}))
        self.assertIsInstance(io, MemoryArrowIO)
        self.assertEqual(io.read_arrow_table().num_rows, 2)

    def test_from_already_tabular_io_passes_through(self) -> None:
        original = MemoryArrowIO(pa.table({"a": [1]}))
        same = Tabular.from_(original)
        self.assertIs(same, original)

    def test_from_unsupported_raises_with_helpful_message(self) -> None:
        class Mystery:
            pass

        with self.assertRaises(RuntimeError) as cx:
            Tabular.from_(Mystery())
        msg = str(cx.exception)
        self.assertIn("pyarrow", msg.lower())
        self.assertIn("polars", msg.lower())

    def test_from_polars(self) -> None:
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        io = Tabular.from_(pl.DataFrame({"a": [1, 2, 3]}))
        self.assertIsInstance(io, MemoryArrowIO)
        self.assertEqual(io.read_arrow_table().num_rows, 3)


class TestIngestExtras(unittest.TestCase):
    """Direct :meth:`MemoryArrowIO._ingest` coverage."""

    def test_ingest_iterable_of_tables(self) -> None:
        tables = [pa.table({"a": [1]}), pa.table({"a": [2]})]
        io = MemoryArrowIO(tables)
        self.assertEqual(io.read_arrow_table().num_rows, 2)

    def test_ingest_unknown_type_raises(self) -> None:
        with self.assertRaises(TypeError) as cx:
            MemoryArrowIO(42)  # int is not iterable
        self.assertIn("MemoryArrowIO can't ingest", str(cx.exception))


if __name__ == "__main__":
    unittest.main()
