"""Tests for the in-memory tabular dispatch on :meth:`IO.__new__`.

Calling :class:`yggdrasil.io.IO` directly with a pure-data tabular
shape (a :class:`pa.Table`, a Spark / polars / pandas frame, a
``list[dict]`` / ``dict[str, list]``, or an existing
:class:`Tabular`) returns the right in-memory holder instead of
forcing the input through the byte-backed scheme / format
registries. These cover the dispatch surface — the regression
paths (path-shaped strings, bytes, file-like) stay unchanged.
"""

from __future__ import annotations

import unittest
from unittest import mock

import pyarrow as pa

from yggdrasil.io.holder import IO, _resolve_in_memory_tabular
from yggdrasil.io.tabular import SparkTabular
from yggdrasil.arrow.tabular import ArrowTabular


class TestIOInMemoryDispatch(unittest.TestCase):
    """Pure in-memory shapes route to the right :class:`Tabular`."""

    def _table(self) -> pa.Table:
        return pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    def test_pa_table_routes_to_arrow_tabular(self) -> None:
        io = IO(self._table())
        self.assertIsInstance(io, ArrowTabular)
        self.assertEqual(io.num_rows, 3)

    def test_pa_record_batch_routes_to_arrow_tabular(self) -> None:
        batch = self._table().to_batches()[0]
        io = IO(batch)
        self.assertIsInstance(io, ArrowTabular)
        self.assertEqual(io.num_rows, 3)

    def test_pa_record_batch_reader_routes_to_arrow_tabular(self) -> None:
        t = self._table()
        reader = pa.RecordBatchReader.from_batches(t.schema, t.to_batches())
        io = IO(reader)
        self.assertIsInstance(io, ArrowTabular)
        self.assertEqual(io.num_rows, 3)

    def test_list_of_dicts_routes_to_arrow_tabular(self) -> None:
        io = IO([{"x": 1}, {"x": 2}])
        self.assertIsInstance(io, ArrowTabular)
        self.assertEqual(io.num_rows, 2)

    def test_dict_of_lists_routes_to_arrow_tabular(self) -> None:
        io = IO({"x": [1, 2, 3]})
        self.assertIsInstance(io, ArrowTabular)
        self.assertEqual(io.num_rows, 3)

    def test_existing_tabular_returned_as_is(self) -> None:
        seed = ArrowTabular(self._table())
        io = IO(seed)
        # Same instance — no re-wrap.
        self.assertIs(io, seed)

    def test_empty_list_is_not_in_memory_dispatch(self) -> None:
        # Empty list isn't a row-list — falls through to existing
        # dispatch (which lands on Memory). The point: the in-memory
        # branch shouldn't claim shapes it can't actually wrap.
        self.assertIsNone(_resolve_in_memory_tabular([]))

    def test_empty_dict_is_not_in_memory_dispatch(self) -> None:
        self.assertIsNone(_resolve_in_memory_tabular({}))

    def test_mixed_list_is_not_in_memory_dispatch(self) -> None:
        # ``[1, 2, 3]`` isn't a row-list (entries aren't dicts).
        self.assertIsNone(_resolve_in_memory_tabular([1, 2, 3]))


class TestIOPolarsPandasDispatch(unittest.TestCase):
    """Polars / pandas frames route to :class:`ArrowTabular`."""

    def test_polars_dataframe(self) -> None:
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        frame = pl.DataFrame({"x": [1, 2, 3]})
        io = IO(frame)
        self.assertIsInstance(io, ArrowTabular)
        self.assertEqual(io.num_rows, 3)

    def test_pandas_dataframe(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        frame = pd.DataFrame({"x": [1, 2, 3]})
        io = IO(frame)
        self.assertIsInstance(io, ArrowTabular)
        self.assertEqual(io.num_rows, 3)


class TestIOSparkDispatch(unittest.TestCase):
    """Spark DataFrame routes to :class:`SparkTabular` (kept lazy).

    Uses a mock to fake a pyspark module on ``sys.modules`` so the
    dispatch logic is exercised without requiring the optional dep.
    The real Spark integration path is covered under SparkTestCase.
    """

    def test_spark_dataframe_routes_to_spark_tabular(self) -> None:
        # Build a fake instance whose ``type(...).__module__`` starts
        # with ``pyspark`` and whose class name contains ``DataFrame``.
        fake_df_cls = type(
            "DataFrame",
            (object,),
            {"__module__": "pyspark.sql.dataframe"},
        )
        fake_df = fake_df_cls()
        # ``SparkTabular`` reads ``getattr(frame, "sparkSession", None)``
        # in __init__; setting it to ``None`` is fine for the routing
        # check.
        fake_df.sparkSession = None

        target = _resolve_in_memory_tabular(fake_df)
        self.assertIs(target, SparkTabular)

        # And the full ``IO(...)`` round trip lands a SparkTabular.
        io = IO(fake_df)
        self.assertIsInstance(io, SparkTabular)
        self.assertIs(io.frame, fake_df)

    def test_other_pyspark_objects_not_routed(self) -> None:
        # ``pyspark`` module but not a DataFrame → no in-memory route.
        fake_col_cls = type(
            "Column",
            (object,),
            {"__module__": "pyspark.sql.column"},
        )
        fake_col = fake_col_cls()
        self.assertIsNone(_resolve_in_memory_tabular(fake_col))


class TestIODispatchPassesKwargs(unittest.TestCase):
    """Holder kwargs flow through the dispatch into the concrete holder."""

    def test_arrow_tabular_kwargs_pass_through(self) -> None:
        # spill_bytes is an ArrowTabular kwarg; passing it through
        # IO(...) should land on the holder.
        t = pa.table({"x": list(range(100))})
        io = IO(t, spill_bytes=1)
        self.assertIsInstance(io, ArrowTabular)
        self.assertEqual(io.spill_bytes, 1)
        self.assertTrue(io.spilled)


class TestIORegressionPathDispatch(unittest.TestCase):
    """The pre-existing scheme / format dispatch still works."""

    def test_path_string_still_resolves_format_leaf(self) -> None:
        # Path-shaped string → format leaf (ParquetFile in this case).
        io = IO("scratch.parquet")
        # Not an ArrowTabular — went through the format registry.
        self.assertNotIsInstance(io, ArrowTabular)
        # Still satisfies the Tabular contract.
        from yggdrasil.io.tabular import Tabular
        self.assertIsInstance(io, Tabular)

    def test_bytes_still_resolves_to_memory(self) -> None:
        io = IO(b"\x00\x01\x02")
        from yggdrasil.io.memory import Memory
        self.assertIsInstance(io, Memory)

    def test_in_memory_dispatch_only_on_io_class(self) -> None:
        # When called on a concrete subclass (e.g. Memory), the
        # in-memory branch shouldn't fire — that subclass's __new__
        # owns the dispatch decision.
        from yggdrasil.io.memory import Memory

        # ``Memory`` doesn't claim to handle pa.Table; the existing
        # Memory dispatch decides. We just verify the IO-level branch
        # is gated on ``cls is IO``.
        with mock.patch(
            "yggdrasil.io.holder._resolve_in_memory_tabular",
        ) as resolver:
            try:
                Memory(b"hi")
            except Exception:
                pass
            # The resolver isn't consulted for Memory(...) — the
            # in-memory branch is ``cls is IO`` only.
            resolver.assert_not_called()


class TestResolveInMemoryTabular(unittest.TestCase):
    """Unit-level coverage of the resolver helper itself."""

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_resolve_in_memory_tabular(None))

    def test_string_returns_none(self) -> None:
        # Strings are path-shaped; the byte-backed dispatch handles them.
        self.assertIsNone(_resolve_in_memory_tabular("x.parquet"))

    def test_bytes_returns_none(self) -> None:
        self.assertIsNone(_resolve_in_memory_tabular(b"xyz"))

    def test_pa_table_returns_arrow_tabular(self) -> None:
        t = pa.table({"x": [1]})
        self.assertIs(_resolve_in_memory_tabular(t), ArrowTabular)

    def test_pa_record_batch_returns_arrow_tabular(self) -> None:
        batch = pa.table({"x": [1]}).to_batches()[0]
        self.assertIs(_resolve_in_memory_tabular(batch), ArrowTabular)

    def test_arbitrary_object_returns_none(self) -> None:
        self.assertIsNone(_resolve_in_memory_tabular(object()))
