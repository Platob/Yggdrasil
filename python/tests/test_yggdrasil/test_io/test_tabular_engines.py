"""Tests for the in-memory engine-frame Tabular holders.

Covers :class:`yggdrasil.io.tabular.polars.PolarsTabular` and
:class:`yggdrasil.io.tabular.pandas.PandasTabular` — the contract
mirrors :class:`yggdrasil.io.tabular.spark.SparkTabular`: the held
frame is the holder's only state, reads return it as-is, writes
mutate it in place.
"""

from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.io.tabular import PandasTabular, PolarsTabular
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase


class TestPolarsTabular(PolarsTestCase, ArrowTestCase):
    """In-memory Polars holder — engine-native + Arrow round-trip."""

    def test_constructs_from_polars_frame(self) -> None:
        df = self.df({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        pt = PolarsTabular(df)
        self.assertEqual(pt.num_rows, 3)
        self.assertFalse(pt.is_empty())
        self.assertFrameEqual(pt.read_polars_frame(), df)

    def test_constructs_empty(self) -> None:
        pt = PolarsTabular()
        self.assertTrue(pt.is_empty())
        self.assertEqual(pt.num_rows, 0)
        self.assertFalse(bool(pt))
        # Empty read returns an empty frame, not an error.
        self.assertEqual(pt.read_polars_frame().height, 0)

    def test_ingests_lazyframe(self) -> None:
        lf = self.lazy({"a": [1, 2]})
        pt = PolarsTabular(lf)
        self.assertEqual(pt.frame["a"].to_list(), [1, 2])

    def test_ingests_arrow_table(self) -> None:
        table = self.table({"k": [1, 2, 3]})
        pt = PolarsTabular(table)
        self.assertEqual(pt.frame["k"].to_list(), [1, 2, 3])

    def test_read_arrow_round_trip(self) -> None:
        df = self.df({"x": [1, 2, 3]})
        pt = PolarsTabular(df)
        out = pt.read_arrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(out.column_names, ["x"])

    def test_overwrite_replaces_frame(self) -> None:
        pt = PolarsTabular(self.df({"x": [1, 2]}))
        pt.write_polars_frame(self.df({"x": [9]}))
        self.assertEqual(pt.frame["x"].to_list(), [9])

    def test_append_concatenates(self) -> None:
        pt = PolarsTabular(self.df({"x": [1, 2]}))
        pt.write_polars_frame(self.df({"x": [3, 4]}), mode="APPEND")
        self.assertEqual(pt.frame["x"].to_list(), [1, 2, 3, 4])

    def test_append_diagonal_relaxed_fills_missing(self) -> None:
        pt = PolarsTabular(self.df({"a": [1], "b": [10]}))
        pt.write_polars_frame(self.df({"a": [2], "c": [20]}), mode="APPEND")
        # Both sides survive, missing columns fill with null.
        self.assertEqual(set(pt.frame.columns), {"a", "b", "c"})
        self.assertEqual(pt.frame.height, 2)

    def test_ignore_on_non_empty_keeps_existing(self) -> None:
        pt = PolarsTabular(self.df({"x": [1]}))
        pt.write_polars_frame(self.df({"x": [99]}), mode="IGNORE")
        self.assertEqual(pt.frame["x"].to_list(), [1])

    def test_ignore_on_empty_writes(self) -> None:
        pt = PolarsTabular()
        pt.write_polars_frame(self.df({"x": [1]}), mode="IGNORE")
        self.assertEqual(pt.frame["x"].to_list(), [1])

    def test_error_if_exists_raises_when_non_empty(self) -> None:
        pt = PolarsTabular(self.df({"x": [1]}))
        with self.assertRaises(FileExistsError):
            pt.write_polars_frame(self.df({"x": [2]}), mode="ERROR_IF_EXISTS")

    def test_write_arrow_table_routes_through_polars(self) -> None:
        pt = PolarsTabular()
        pt.write_arrow_table(self.table({"k": [1, 2, 3]}))
        self.assertEqual(pt.frame["k"].to_list(), [1, 2, 3])

    def test_persist_swaps_data(self) -> None:
        pt = PolarsTabular(self.df({"x": [1]}))
        pt.persist(data=self.df({"x": [2, 3]}))
        self.assertEqual(pt.frame["x"].to_list(), [2, 3])

    def test_unpersist_clears(self) -> None:
        pt = PolarsTabular(self.df({"x": [1]}))
        pt.unpersist()
        self.assertTrue(pt.is_empty())


class TestPandasTabular(PandasTestCase, ArrowTestCase):
    """In-memory pandas holder — engine-native + Arrow round-trip."""

    def test_constructs_from_pandas_frame(self) -> None:
        df = self.df({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        pt = PandasTabular(df)
        self.assertEqual(pt.num_rows, 3)
        self.assertFalse(pt.is_empty())
        self.assertFrameEqual(pt.read_pandas_frame(), df)

    def test_constructs_empty(self) -> None:
        pt = PandasTabular()
        self.assertTrue(pt.is_empty())
        self.assertEqual(pt.num_rows, 0)
        self.assertFalse(bool(pt))
        self.assertEqual(len(pt.read_pandas_frame()), 0)

    def test_ingests_arrow_table(self) -> None:
        table = self.table({"k": [1, 2, 3]})
        pt = PandasTabular(table)
        self.assertEqual(pt.frame["k"].tolist(), [1, 2, 3])

    def test_ingests_dict(self) -> None:
        pt = PandasTabular({"k": [1, 2]})
        self.assertEqual(pt.frame["k"].tolist(), [1, 2])

    def test_read_arrow_round_trip(self) -> None:
        df = self.df({"x": [1, 2, 3]})
        pt = PandasTabular(df)
        out = pt.read_arrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(out.column_names, ["x"])

    def test_overwrite_replaces_frame(self) -> None:
        pt = PandasTabular(self.df({"x": [1, 2]}))
        pt.write_pandas_frame(self.df({"x": [9]}))
        self.assertEqual(pt.frame["x"].tolist(), [9])

    def test_append_concatenates(self) -> None:
        pt = PandasTabular(self.df({"x": [1, 2]}))
        pt.write_pandas_frame(self.df({"x": [3, 4]}), mode="APPEND")
        self.assertEqual(pt.frame["x"].tolist(), [1, 2, 3, 4])

    def test_append_resets_index(self) -> None:
        # ``ignore_index=True`` keeps the held frame on a contiguous
        # default range index — important so subsequent
        # ``pa.Table.from_pandas`` calls don't leak an index column.
        pt = PandasTabular(self.df({"x": [1, 2]}))
        pt.write_pandas_frame(self.df({"x": [3, 4]}), mode="APPEND")
        self.assertEqual(list(pt.frame.index), [0, 1, 2, 3])

    def test_ignore_on_non_empty_keeps_existing(self) -> None:
        pt = PandasTabular(self.df({"x": [1]}))
        pt.write_pandas_frame(self.df({"x": [99]}), mode="IGNORE")
        self.assertEqual(pt.frame["x"].tolist(), [1])

    def test_error_if_exists_raises_when_non_empty(self) -> None:
        pt = PandasTabular(self.df({"x": [1]}))
        with self.assertRaises(FileExistsError):
            pt.write_pandas_frame(self.df({"x": [2]}), mode="ERROR_IF_EXISTS")

    def test_write_arrow_table_routes_through_pandas(self) -> None:
        pt = PandasTabular()
        pt.write_arrow_table(self.table({"k": [1, 2, 3]}))
        self.assertEqual(pt.frame["k"].tolist(), [1, 2, 3])

    def test_persist_swaps_data(self) -> None:
        pt = PandasTabular(self.df({"x": [1]}))
        pt.persist(data=self.df({"x": [2, 3]}))
        self.assertEqual(pt.frame["x"].tolist(), [2, 3])

    def test_unpersist_clears(self) -> None:
        pt = PandasTabular(self.df({"x": [1]}))
        pt.unpersist()
        self.assertTrue(pt.is_empty())
