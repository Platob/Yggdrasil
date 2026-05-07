"""Tests for :class:`yggdrasil.io.tabular.union.UnionTabular`."""

from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.data.expr import col
from yggdrasil.io.tabular import ArrowTabular, UnionTabular


class TestUnionTabular(PolarsTestCase, ArrowTestCase):

    def _t1(self) -> ArrowTabular:
        return ArrowTabular(self.table({"x": [1, 2], "y": [10, 20]}))

    def _t2(self) -> ArrowTabular:
        return ArrowTabular(self.table({"x": [3, 4], "y": [30, 40]}))

    def test_basic_union_preserves_order(self) -> None:
        u = UnionTabular([self._t1(), self._t2()])
        df = u.read_polars_frame()
        self.assertEqual(df["x"].to_list(), [1, 2, 3, 4])
        self.assertEqual(df["y"].to_list(), [10, 20, 30, 40])

    def test_empty_union_reads_zero_rows(self) -> None:
        u = UnionTabular([])
        self.assertEqual(u.read_arrow_table().num_rows, 0)

    def test_single_child_union_passthrough(self) -> None:
        u = UnionTabular([self._t1()])
        df = u.read_polars_frame()
        self.assertEqual(df["x"].to_list(), [1, 2])

    def test_schema_sync_missing_columns(self) -> None:
        # Child A has columns x, y; child B has x, z. Union should
        # surface all three with nulls where each child lacks one.
        a = ArrowTabular(self.table({"x": [1], "y": [10]}))
        b = ArrowTabular(self.table({"x": [2], "z": [200]}))
        u = UnionTabular([a, b])
        schema = u.collect_schema()
        self.assertEqual(sorted(schema.names), ["x", "y", "z"])

        df = u.read_polars_frame().sort("x")
        # The row from A has null z; the row from B has null y.
        self.assertEqual(df["x"].to_list(), [1, 2])
        a_row = df.filter(self.pl.col("x") == 1).to_dicts()[0]
        b_row = df.filter(self.pl.col("x") == 2).to_dicts()[0]
        self.assertIsNone(a_row["z"])
        self.assertIsNone(b_row["y"])

    def test_merged_schema_helper(self) -> None:
        a = ArrowTabular(self.table({"x": [1], "y": [10]}))
        b = ArrowTabular(self.table({"x": [2], "z": [200]}))
        u = UnionTabular([a, b])
        merged = u.merged_schema()
        self.assertEqual(sorted(merged.names), ["x", "y", "z"])

    def test_filter_pushdown_broadcasts_per_child(self) -> None:
        u = UnionTabular([self._t1(), self._t2()]).where(col("x") > 2)
        # Op is recorded; broadcast happens at execution time (visible
        # via the planner). Behaviorally: no row with x<=2 escapes.
        df = u.read_polars_frame()
        self.assertEqual(df["x"].to_list(), [3, 4])

    def test_select_pushdown(self) -> None:
        u = UnionTabular([self._t1(), self._t2()]).select("y")
        df = u.read_polars_frame()
        self.assertEqual(df.columns, ["y"])
        self.assertEqual(df["y"].to_list(), [10, 20, 30, 40])

    def test_group_by_after_union(self) -> None:
        pl = self.pl
        a = ArrowTabular(self.table({"g": ["a", "b"], "x": [1, 2]}))
        b = ArrowTabular(self.table({"g": ["a", "b"], "x": [10, 20]}))
        u = (
            UnionTabular([a, b])
            .group_by("g")
            .agg(pl.col("x").sum().alias("x_sum"))
        )
        out = u.read_polars_frame().sort("g")
        self.assertEqual(
            dict(zip(out["g"].to_list(), out["x_sum"].to_list())),
            {"a": 11, "b": 22},
        )

    def test_filter_then_groupby_splits_correctly(self) -> None:
        # Filter is in the pushdown prefix, group_by in the post-union
        # tail. Result should match a non-pushdown reference.
        pl = self.pl
        a = ArrowTabular(self.table({"g": ["a", "b", "a"], "x": [1, 2, 3]}))
        b = ArrowTabular(self.table({"g": ["a", "b", "b"], "x": [10, 20, 30]}))
        u = (
            UnionTabular([a, b])
            .where(col("x") > 1)
            .group_by("g")
            .agg(pl.col("x").sum().alias("x_sum"))
        )
        # Op kinds in expected order — filter pushed, group_by tail.
        self.assertEqual(
            [op[0] for op in u.ops],
            ["filter", "group_by"],
        )
        out = u.read_polars_frame().sort("g")
        self.assertEqual(
            dict(zip(out["g"].to_list(), out["x_sum"].to_list())),
            {"a": 13, "b": 52},
        )

    def test_chaining_returns_uniontabular(self) -> None:
        u = UnionTabular([self._t1(), self._t2()]).where(col("x") > 0)
        # ``where`` should preserve the subclass — not silently downcast
        # to a plain LazyTabular and lose the children.
        self.assertIsInstance(u, UnionTabular)
        self.assertEqual(len(u.children), 2)

    def test_writes_raise(self) -> None:
        u = UnionTabular([self._t1(), self._t2()])
        with self.assertRaises(TypeError):
            u.write_arrow_table(self.table({"x": [99], "y": [99]}))

    def test_split_ops_pushdown_prefix(self) -> None:
        u = (
            UnionTabular([self._t1(), self._t2()])
            .where(col("x") > 0)
            .select("x", "y")
        )
        prefix, tail = UnionTabular._split_ops(u.ops)
        self.assertEqual([op[0] for op in prefix], ["filter", "select"])
        self.assertEqual(tail, ())

    def test_split_ops_post_union_tail(self) -> None:
        pl = self.pl
        u = (
            UnionTabular([self._t1(), self._t2()])
            .where(col("x") > 0)
            .group_by("y")
            .agg(pl.col("x").sum())
            .where(col("x") > 0)
        )
        prefix, tail = UnionTabular._split_ops(u.ops)
        # group_by is the boundary; everything after stays in the tail
        # even though the second filter would otherwise be pushdownable.
        self.assertEqual([op[0] for op in prefix], ["filter"])
        self.assertEqual([op[0] for op in tail], ["group_by", "filter"])
