"""Tests for :class:`yggdrasil.io.tabular.lazy.LazyTabular`."""

from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.execution.expr import Logical, Predicate, col
from yggdrasil.execution.plan import (
    Apply,
    ExecutionPlan,
    Filter,
    GroupByAgg,
    Select,
)
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

    def test_passthrough_no_plan(self) -> None:
        src = self._source()
        lazy = LazyTabular(src)
        self.assertTrue(lazy.plan.is_empty())
        self.assertEqual(lazy.read_arrow_table().num_rows, 5)
        self.assertEqual(lazy.collect_schema().names, ["x", "y", "g"])

    def test_select_projection(self) -> None:
        lazy = LazyTabular(self._source()).select("x", "g")
        self.assertEqual(len(lazy.plan), 1)
        self.assertIsInstance(lazy.plan.ops[0], Select)
        table = lazy.read_arrow_table()
        self.assertEqual(table.column_names, ["x", "g"])
        self.assertEqual(table.num_rows, 5)

    def test_filter_chain_fuses_into_one_op(self) -> None:
        pl = self.pl
        lazy = (
            LazyTabular(self._source())
            .where(pl.col("x") > 1)
            .where(pl.col("x") < 5)
        )
        # Two adjacent filter calls fuse into one Filter op via
        # ExecutionPlan.append → Filter.extend.
        self.assertEqual(len(lazy.plan), 1)
        op = lazy.plan.ops[0]
        self.assertIsInstance(op, Filter)
        self.assertEqual(len(op.predicates), 2)

        df = lazy.read_polars_frame()
        self.assertEqual(df["x"].to_list(), [2, 3, 4])

    def test_filter_accepts_yggdrasil_predicate(self) -> None:
        lazy = LazyTabular(self._source()).where(col("x") > 2)
        df = lazy.read_polars_frame()
        self.assertEqual(df["x"].to_list(), [3, 4, 5])

    def test_filter_yggdrasil_predicates_merge_into_logical(self) -> None:
        lazy = (
            LazyTabular(self._source())
            .where(col("x") > 1)
            .where(col("x") < 5)
        )
        op = lazy.plan.ops[0]
        self.assertIsInstance(op, Filter)
        self.assertEqual(len(op.predicates), 1)
        self.assertIsInstance(op.predicates[0], Logical)
        self.assertIsInstance(op.predicates[0], Predicate)

        df = lazy.read_polars_frame()
        self.assertEqual(df["x"].to_list(), [2, 3, 4])

    def test_filter_accepts_sql_string(self) -> None:
        lazy = LazyTabular(self._source()).where("x >= 3 AND g = 'a'")
        df = lazy.read_polars_frame()
        self.assertEqual(df["x"].to_list(), [3, 5])

    def test_filter_rejects_non_predicate_yggdrasil(self) -> None:
        with self.assertRaises(TypeError):
            LazyTabular(self._source()).where(col("x") + 1)

    def test_chained_select_after_filter(self) -> None:
        pl = self.pl
        lazy = (
            LazyTabular(self._source())
            .where(pl.col("g") == "a")
            .select("x", "y")
        )
        self.assertEqual(
            [type(op).__name__ for op in lazy.plan.ops],
            ["Filter", "Select"],
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
        self.assertIsInstance(lazy.plan.ops[-1], GroupByAgg)
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
        self.assertTrue(base.plan.is_empty())
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

    def test_writes_forward_to_source(self) -> None:
        pl = self.pl
        src = ArrowTabular()  # empty sink
        lazy = LazyTabular(src).where(pl.col("x") > 0)
        # Writing through the lazy wrapper should populate the source
        # untouched (lazy plan only describes the read-side view).
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
        self.assertIsInstance(lazy.plan.ops[-1], Apply)
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

    def test_constructed_with_explicit_plan(self) -> None:
        plan = ExecutionPlan().append(Filter((col("x") > 2,)))
        lazy = LazyTabular(self._source(), plan=plan)
        self.assertIs(lazy.plan, plan)
        self.assertEqual(
            lazy.read_polars_frame()["x"].to_list(), [3, 4, 5],
        )

    def test_execute_plan_on_plain_tabular(self) -> None:
        # Tabular.execute_plan on a non-lazy source wraps in LazyTabular.
        plan = ExecutionPlan().append(Filter((col("x") > 2,)))
        result = self._source().execute_plan(plan)
        self.assertIsInstance(result, LazyTabular)
        self.assertEqual(
            result.read_polars_frame()["x"].to_list(), [3, 4, 5],
        )

    def test_execute_plan_empty_returns_self(self) -> None:
        src = self._source()
        self.assertIs(src.execute_plan(ExecutionPlan.empty()), src)
        self.assertIs(src.execute_plan(None), src)

    def test_execute_plan_on_lazy_tabular_composes(self) -> None:
        # Stacking via execute_plan should fold ops into the existing
        # plan rather than wrapping in a second LazyTabular.
        first = LazyTabular(self._source()).where(col("x") > 1)
        plan = ExecutionPlan().append(Select(("x",)))
        second = first.execute_plan(plan)
        self.assertIsInstance(second, LazyTabular)
        self.assertIs(second.source, first.source)
        # Plans fuse: filter + select on the same plan, no nesting.
        self.assertEqual(
            [type(o).__name__ for o in second.plan.ops],
            ["Filter", "Select"],
        )
        self.assertEqual(
            second.read_polars_frame()["x"].to_list(), [2, 3, 4, 5],
        )

    def test_plan_split_pushdownable(self) -> None:
        pl = self.pl
        lazy = (
            LazyTabular(self._source())
            .where(col("x") > 0)
            .select("x", "g")
            .group_by("g")
            .agg(pl.col("x").sum())
        )
        prefix, tail = lazy.plan.split_pushdownable()
        self.assertEqual(
            [type(o).__name__ for o in prefix.ops],
            ["Filter", "Select"],
        )
        self.assertEqual(
            [type(o).__name__ for o in tail.ops],
            ["GroupByAgg"],
        )
