"""Tests for :mod:`yggdrasil.plan`."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.execution.expr import col
from yggdrasil.plan import ExecutionPlan, LazyTabular, SelectPlan


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def users_table():
    return ArrowTabular(pa.table({
        "id": [1, 2, 3, 4, 5],
        "name": ["alice", "bob", "carol", "dave", "eve"],
        "region": ["US", "EU", "US", "EU", "US"],
    }))


@pytest.fixture
def scores_table():
    return ArrowTabular(pa.table({
        "id": [1, 2, 3, 6],
        "score": [90, 80, 95, 50],
    }))


@pytest.fixture
def dupes_table():
    return ArrowTabular(pa.table({
        "id": [1, 1, 2, 2, 3],
        "val": [10, 20, 30, 40, 50],
    }))


# ---------------------------------------------------------------------------
# ExecutionPlan — identity / introspection
# ---------------------------------------------------------------------------

class TestExecutionPlanIdentity:
    def test_empty_plan_is_identity(self):
        plan = SelectPlan()
        assert plan.is_identity

    def test_non_empty_plan_is_not_identity(self):
        plan = SelectPlan()
        plan.select("a")
        assert not plan.is_identity

    def test_clear_resets_to_identity(self):
        plan = SelectPlan()
        plan.select("a").filter("x > 1").limit(10)
        plan.clear()
        assert plan.is_identity

    def test_identity_plan_returns_source(self, users_table):
        plan = SelectPlan()
        result = plan.execute(users_table)
        assert result is users_table


# ---------------------------------------------------------------------------
# ExecutionPlan — select / drop
# ---------------------------------------------------------------------------

class TestExecutionPlanProjection:
    def test_select(self, users_table):
        plan = SelectPlan()
        plan.select("id", "name")
        result = plan.execute(users_table).read_arrow_table()
        assert result.column_names == ["id", "name"]
        assert result.num_rows == 5

    def test_select_replaces(self):
        plan = SelectPlan()
        plan.select("a", "b")
        assert plan.columns == ["a", "b"]
        plan.select("x")
        assert plan.columns == ["x"]

    def test_select_clears_drop(self):
        plan = SelectPlan()
        plan.drop("a")
        plan.select("b", "c")
        assert plan._drop is None

    def test_drop(self, users_table):
        plan = SelectPlan()
        plan.drop("region")
        result = plan.execute(users_table).read_arrow_table()
        assert result.column_names == ["id", "name"]

    def test_select_empty_raises(self):
        plan = SelectPlan()
        with pytest.raises(ValueError):
            plan.select()


# ---------------------------------------------------------------------------
# ExecutionPlan — filter
# ---------------------------------------------------------------------------

class TestExecutionPlanFilter:
    def test_filter_sql_string(self, users_table):
        plan = SelectPlan()
        plan.filter("id > 3")
        result = plan.execute(users_table).read_arrow_table()
        assert result.num_rows == 2
        assert result.column("id").to_pylist() == [4, 5]

    def test_filter_expression(self, users_table):
        plan = SelectPlan()
        plan.filter(col("id") > 3)
        result = plan.execute(users_table).read_arrow_table()
        assert result.num_rows == 2

    def test_filter_and_accumulates(self, dupes_table):
        plan = SelectPlan()
        plan.filter(col("id") > 1).filter(col("val") < 50)
        result = plan.execute(dupes_table).read_arrow_table()
        assert all(r["id"] > 1 and r["val"] < 50 for r in result.to_pylist())

    def test_clear_filter(self, users_table):
        plan = SelectPlan()
        plan.filter("id > 100")
        plan.clear_filter()
        assert plan.predicate is None
        result = plan.execute(users_table).read_arrow_table()
        assert result.num_rows == 5


# ---------------------------------------------------------------------------
# ExecutionPlan — join
# ---------------------------------------------------------------------------

class TestExecutionPlanJoin:
    def test_inner_join(self, users_table, scores_table):
        plan = SelectPlan()
        plan.join(scores_table, on="id", how="inner")
        result = plan.execute(users_table).read_arrow_table()
        assert "score" in result.column_names
        assert result.num_rows == 3

    def test_left_join(self, users_table, scores_table):
        plan = SelectPlan()
        plan.join(scores_table, on="id", how="left")
        result = plan.execute(users_table).read_arrow_table()
        assert result.num_rows == 5

    def test_join_with_lazy(self, users_table, scores_table):
        lazy_scores = scores_table.lazy().filter(col("score") > 70)
        plan = SelectPlan()
        plan.join(lazy_scores, on="id", how="inner")
        result = plan.execute(users_table).read_arrow_table()
        assert all(r["score"] > 70 for r in result.to_pylist())

    def test_join_then_filter(self, users_table, scores_table):
        plan = SelectPlan()
        plan.join(scores_table, on="id", how="inner")
        plan.filter(col("score") > 85)
        result = plan.execute(users_table).read_arrow_table()
        assert all(r["score"] > 85 for r in result.to_pylist())

    def test_join_then_select(self, users_table, scores_table):
        plan = SelectPlan()
        plan.join(scores_table, on="id", how="inner")
        plan.select("name", "score")
        result = plan.execute(users_table).read_arrow_table()
        assert result.column_names == ["name", "score"]


# ---------------------------------------------------------------------------
# ExecutionPlan — union
# ---------------------------------------------------------------------------

class TestExecutionPlanUnion:
    def test_union(self, users_table):
        other = ArrowTabular(pa.table({
            "id": [10, 11],
            "name": ["frank", "grace"],
            "region": ["EU", "US"],
        }))
        plan = SelectPlan()
        plan.union(other)
        result = plan.execute(users_table).read_arrow_table()
        assert result.num_rows == 7


# ---------------------------------------------------------------------------
# ExecutionPlan — unique
# ---------------------------------------------------------------------------

class TestExecutionPlanUnique:
    def test_unique(self, dupes_table):
        plan = SelectPlan()
        plan.unique("id")
        result = plan.execute(dupes_table).read_arrow_table()
        ids = result.column("id").to_pylist()
        assert sorted(ids) == [1, 2, 3]


# ---------------------------------------------------------------------------
# ExecutionPlan — limit
# ---------------------------------------------------------------------------

class TestExecutionPlanLimit:
    def test_limit(self, users_table):
        plan = SelectPlan()
        plan.limit(2)
        result = plan.execute(users_table).read_arrow_table()
        assert result.num_rows == 2

    def test_limit_none_removes(self, users_table):
        plan = SelectPlan()
        plan.limit(2)
        plan.limit(None)
        result = plan.execute(users_table).read_arrow_table()
        assert result.num_rows == 5


# ---------------------------------------------------------------------------
# ExecutionPlan — copy
# ---------------------------------------------------------------------------

class TestExecutionPlanCopy:
    def test_copy_is_independent(self):
        plan = SelectPlan()
        plan.select("a", "b").filter("x > 1")
        clone = plan.copy()
        clone.select("c")
        assert plan.columns == ["a", "b"]
        assert clone.columns == ["c"]


# ---------------------------------------------------------------------------
# ExecutionPlan — repr
# ---------------------------------------------------------------------------

class TestExecutionPlanRepr:
    def test_identity_repr(self):
        assert "identity" in repr(SelectPlan())

    def test_non_empty_repr(self):
        plan = SelectPlan()
        plan.select("a").filter("x > 1").limit(10)
        r = repr(plan)
        assert "select=" in r
        assert "filter=" in r
        assert "limit=" in r


# ---------------------------------------------------------------------------
# LazyTabular
# ---------------------------------------------------------------------------

class TestLazyTabular:
    def test_lazy_returns_lazy_tabular(self, users_table):
        lazy = users_table.lazy()
        assert isinstance(lazy, LazyTabular)

    def test_lazy_on_lazy_returns_self(self, users_table):
        lazy = users_table.lazy()
        assert lazy.lazy() is lazy

    def test_source_accessor(self, users_table):
        lazy = users_table.lazy()
        assert lazy.source is users_table

    def test_plan_accessor(self, users_table):
        lazy = users_table.lazy()
        assert isinstance(lazy.plan, ExecutionPlan)
        assert lazy.plan.is_identity

    def test_select_returns_self(self, users_table):
        lazy = users_table.lazy()
        result = lazy.select("id")
        assert result is lazy

    def test_filter_returns_self(self, users_table):
        lazy = users_table.lazy()
        result = lazy.filter("id > 1")
        assert result is lazy

    def test_read_arrow_table(self, users_table):
        result = users_table.lazy().select("id", "name").read_arrow_table()
        assert result.column_names == ["id", "name"]

    def test_read_arrow_batches(self, users_table):
        batches = list(users_table.lazy().select("id").read_arrow_batches())
        assert len(batches) >= 1
        assert batches[0].schema.names == ["id"]

    def test_read_pylist(self, users_table):
        rows = users_table.lazy().filter("id <= 2").read_pylist()
        assert len(rows) == 2

    def test_read_pydict(self, users_table):
        d = users_table.lazy().select("name").read_pydict()
        assert "name" in d
        assert len(d["name"]) == 5

    def test_count(self, users_table):
        assert users_table.lazy().filter("id > 3").count() == 2

    def test_collect_schema(self, users_table):
        schema = users_table.lazy().select("id", "name").collect_schema()
        assert "id" in schema.names

    def test_collect(self, users_table):
        result = users_table.lazy().filter("id > 3").collect()
        assert result.read_arrow_table().num_rows == 2

    def test_chain_select_filter_limit(self, users_table):
        result = (users_table.lazy()
                  .filter(col("id") > 1)
                  .select("id", "name")
                  .limit(2)
                  .read_arrow_table())
        assert result.column_names == ["id", "name"]
        assert result.num_rows == 2

    def test_join(self, users_table, scores_table):
        result = (users_table.lazy()
                  .join(scores_table, on="id", how="inner")
                  .read_arrow_table())
        assert "score" in result.column_names
        assert result.num_rows == 3

    def test_join_two_lazys(self, users_table, scores_table):
        lazy_scores = scores_table.lazy().filter(col("score") > 80)
        result = (users_table.lazy()
                  .join(lazy_scores, on="id", how="inner")
                  .read_arrow_table())
        assert all(r["score"] > 80 for r in result.to_pylist())

    def test_union(self, users_table):
        other = ArrowTabular(pa.table({
            "id": [10], "name": ["x"], "region": ["US"],
        }))
        result = users_table.lazy().union(other).read_arrow_table()
        assert result.num_rows == 6

    def test_unique(self, dupes_table):
        result = dupes_table.lazy().unique("id").read_arrow_table()
        assert len(set(result.column("id").to_pylist())) == 3

    def test_drop(self, users_table):
        result = users_table.lazy().drop("region").read_arrow_table()
        assert "region" not in result.column_names

    def test_write_raises(self, users_table):
        lazy = users_table.lazy()
        with pytest.raises(TypeError, match="read-only"):
            lazy.write_arrow_table(pa.table({"x": [1]}))

    def test_copy_branches(self, users_table):
        a = users_table.lazy().filter("id > 1")
        b = a.copy()
        b.select("name")
        assert a.plan.columns is None
        assert b.plan.columns == ["name"]

    def test_executes_on_every_read(self, users_table):
        lazy = users_table.lazy().select("id")
        r1 = lazy.read_arrow_table()
        r2 = lazy.read_arrow_table()
        assert r1.equals(r2)
        assert r1 is not r2
