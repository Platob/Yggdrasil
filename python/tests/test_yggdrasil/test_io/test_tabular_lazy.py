"""Tests for :class:`yggdrasil.io.tabular.lazy.LazyTabular`."""

from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.io.tabular import ArrowTabular, LazyTabular


class TestLazyTabular(PolarsTestCase, ArrowTestCase):
    """Verify lazy-op composition + pushdown on top of an ArrowTabular."""

    def _source(self) -> ArrowTabular:
        return ArrowTabular(
            self.table(
                {
                    "x": [1, 2, 3, 4, 5],
                    "y": [10, 20, 30, 40, 50],
                    "g": ["a", "b", "a", "b", "a"],
                }
            )
        )

    def test_passthrough_no_ops(self) -> None:
        src = self._source()
        lazy = LazyTabular(src)
        self.assertEqual(lazy.read_arrow_table().num_rows, 5)
        self.assertEqual(lazy.collect_schema().names, ["x", "y", "g"])

    def test_select_projection(self) -> None:
        lazy = LazyTabular(self._source()).select("x", "g")
        table = lazy.read_arrow_table()
        self.assertEqual(table.column_names, ["x", "g"])
        self.assertEqual(table.num_rows, 5)

    def test_filter_chain_conjoins(self) -> None:
        pl = self.pl
        lazy = (
            LazyTabular(self._source())
            .where(pl.col("x") > 1)
            .where(pl.col("x") < 5)
        )
        # Two filter ops are recorded but the planner conjoins them
        # into a single filter node before the scan.
        kinds = [op[0] for op in lazy.ops]
        self.assertEqual(kinds, ["filter", "filter"])

        df = lazy.read_polars_frame()
        self.assertEqual(df["x"].to_list(), [2, 3, 4])

    def test_chained_select_after_filter(self) -> None:
        pl = self.pl
        lazy = (
            LazyTabular(self._source())
            .where(pl.col("g") == "a")
            .select("x", "y")
        )
        df = lazy.read_polars_frame()
        self.assertEqual(df.columns, ["x", "y"])
        self.assertEqual(df["x"].to_list(), [1, 3, 5])

    def test_group_by_agg(self) -> None:
        pl = self.pl
        lazy = (
            LazyTabular(self._source())
            .group_by("g")
            .agg(pl.col("x").sum().alias("x_sum"))
        )
        df = lazy.read_polars_frame().sort("g")
        self.assertEqual(df.columns, ["g", "x_sum"])
        self.assertEqual(
            dict(zip(df["g"].to_list(), df["x_sum"].to_list())),
            {"a": 9, "b": 6},
        )

    def test_immutability_of_chain(self) -> None:
        pl = self.pl
        base = LazyTabular(self._source())
        a = base.where(pl.col("x") > 2)
        b = base.where(pl.col("x") < 3)
        # Each branch sees only its own filter; ``base`` is unchanged.
        self.assertEqual(base.ops, ())
        self.assertEqual(a.read_polars_frame()["x"].to_list(), [3, 4, 5])
        self.assertEqual(b.read_polars_frame()["x"].to_list(), [1, 2])

    def test_collect_schema_lazy(self) -> None:
        pl = self.pl
        lazy = LazyTabular(self._source()).select(
            pl.col("x").alias("x2"),
            pl.col("y"),
        )
        names = lazy.collect_schema().names
        self.assertEqual(names, ["x2", "y"])

    def test_scan_polars_frame_returns_lazyframe(self) -> None:
        pl = self.pl
        lazy = LazyTabular(self._source()).where(pl.col("x") > 2)
        lf = lazy.scan_polars_frame()
        self.assertIsInstance(lf, pl.LazyFrame)
        self.assertEqual(lf.collect()["x"].to_list(), [3, 4, 5])

    def test_writes_forward_to_inner(self) -> None:
        pl = self.pl
        src = ArrowTabular()  # empty sink
        lazy = LazyTabular(src).where(pl.col("x") > 0)
        # Writing through the lazy wrapper should populate the inner
        # untouched (lazy ops only describe the read-side view).
        lazy.write_arrow_table(self.table({"x": [10, 11], "y": [1, 2]}))
        self.assertEqual(src.num_rows, 2)
        self.assertEqual(src.read_arrow_table().column_names, ["x", "y"])

    def test_select_no_args_raises(self) -> None:
        with self.assertRaises(ValueError):
            LazyTabular(self._source()).select()

    def test_filter_no_args_raises(self) -> None:
        with self.assertRaises(ValueError):
            LazyTabular(self._source()).filter()

    def test_apply_escape_hatch(self) -> None:
        pl = self.pl
        lazy = LazyTabular(self._source()).apply(
            lambda lf: lf.sort("x", descending=True)
        )
        df = lazy.read_polars_frame()
        self.assertEqual(df["x"].to_list(), [5, 4, 3, 2, 1])

    def test_apply_non_callable_raises(self) -> None:
        with self.assertRaises(TypeError):
            LazyTabular(self._source()).apply("not callable")

    def test_stacked_lazytabular_composes(self) -> None:
        pl = self.pl
        first = LazyTabular(self._source()).where(pl.col("x") > 1)
        second = LazyTabular(first).select("x", "g")
        df = second.read_polars_frame()
        self.assertEqual(df.columns, ["x", "g"])
        self.assertEqual(df["x"].to_list(), [2, 3, 4, 5])
