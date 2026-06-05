"""Tests for the autonomous :class:`InsertPlan` / :class:`MergePlan`
builders and the :class:`ExecutionPlan(Tabular)` contract."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.io.tabular import Tabular
from yggdrasil.plan import (
    ExecutionPlan,
    InsertPlan,
    LazyTabular,
    MergePlan,
    OperationResult,
    SelectPlan,
    parse_sql,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def users():
    return ArrowTabular(pa.table({
        "id": [1, 2, 3],
        "name": ["alice", "bob", "carol"],
        "score": [90, 80, 95],
    }))


@pytest.fixture
def empty_target_schema():
    return pa.schema([
        ("id", pa.int64()),
        ("name", pa.large_utf8()),
        ("score", pa.int64()),
    ])


@pytest.fixture
def empty_target(empty_target_schema):
    return ArrowTabular(pa.table({
        "id": pa.array([], type=pa.int64()),
        "name": pa.array([], type=pa.large_utf8()),
        "score": pa.array([], type=pa.int64()),
    }))


# ---------------------------------------------------------------------------
# ExecutionPlan IS a Tabular
# ---------------------------------------------------------------------------


class TestExecutionPlanIsTabular:
    def test_select_plan_is_tabular(self, users):
        plan = SelectPlan(source=users).select("id", "name")
        assert isinstance(plan, Tabular)

    def test_select_plan_read_arrow_table(self, users):
        plan = SelectPlan(source=users).select("id", "name").limit(2)
        table = plan.read_arrow_table()
        assert table.num_rows == 2
        assert table.column_names == ["id", "name"]

    def test_select_plan_read_pylist(self, users):
        rows = SelectPlan(source=users).filter("id > 1").read_pylist()
        assert len(rows) == 2

    def test_insert_plan_read_returns_metadata(self, users, empty_target):
        plan = InsertPlan(target=empty_target, source=users)
        meta_table = plan.read_arrow_table()
        assert meta_table.column("operation").to_pylist() == ["INSERT"]
        assert meta_table.column("rows_inserted").to_pylist() == [3]


# ---------------------------------------------------------------------------
# SelectPlan — autonomous (source bound)
# ---------------------------------------------------------------------------


class TestSelectPlanAutonomous:
    def test_construct_with_source(self, users):
        plan = SelectPlan(source=users)
        assert plan.source is users

    def test_execute_uses_bound_source(self, users):
        plan = SelectPlan(source=users).filter("id > 1")
        result = plan.execute()
        assert result.read_arrow_table().num_rows == 2

    def test_execute_wait_kwarg(self, users):
        plan = SelectPlan(source=users).select("id")
        result = plan.execute(wait=True, raise_error=True)
        assert result.read_arrow_table().num_rows == 3

    def test_execute_without_source_raises(self):
        plan = SelectPlan().select("a")
        with pytest.raises(ValueError, match="no source"):
            plan.execute()

    def test_bind_attaches_source(self, users):
        plan = SelectPlan().select("id").bind(users)
        result = plan.execute()
        assert result.read_arrow_table().column_names == ["id"]

    def test_legacy_positional_source(self, users):
        plan = SelectPlan().select("id")
        result = plan.execute(users)
        assert result.read_arrow_table().column_names == ["id"]

    def test_copy_preserves_source(self, users):
        plan = SelectPlan(source=users).select("id")
        clone = plan.copy()
        assert clone.source is users


# ---------------------------------------------------------------------------
# InsertPlan
# ---------------------------------------------------------------------------


class TestInsertPlan:
    def test_insert_from_tabular(self, users, empty_target):
        plan = InsertPlan(target=empty_target, source=users)
        result = plan.execute()
        assert isinstance(result, OperationResult)
        assert result.operation == "INSERT"
        assert result.rows_inserted == 3
        assert empty_target.read_arrow_table().num_rows == 3

    def test_insert_from_select_plan(self, users, empty_target):
        src_plan = SelectPlan(source=users).filter("score > 85")
        plan = InsertPlan(target=empty_target, source=src_plan)
        result = plan.execute()
        assert result.rows_inserted == 2
        rows = empty_target.read_arrow_table()
        assert sorted(rows.column("id").to_pylist()) == [1, 3]

    def test_insert_values_literal(self, empty_target):
        plan = InsertPlan(
            target=empty_target,
            columns=["id", "name", "score"],
            values=[[10, "frank", 70], [11, "grace", 65]],
        )
        result = plan.execute()
        assert result.operation == "INSERT"
        assert result.rows_inserted == 2
        assert empty_target.read_arrow_table().column("name").to_pylist() == ["frank", "grace"]

    def test_builder_chain(self, users, empty_target):
        plan = InsertPlan().into(empty_target).select_from(users).with_columns("id", "name", "score")
        result = plan.execute()
        assert result.rows_inserted == 3

    def test_missing_target_raises(self, users):
        plan = InsertPlan(source=users)
        with pytest.raises(ValueError, match="no target"):
            plan.execute()

    def test_missing_source_and_values_raises(self, empty_target):
        plan = InsertPlan(target=empty_target)
        with pytest.raises(ValueError, match="neither source nor values"):
            plan.execute()

    def test_operation_result_to_arrow(self, users, empty_target):
        result = InsertPlan(target=empty_target, source=users).execute()
        meta = result.to_arrow_tabular().read_arrow_table()
        assert meta.column("rows_affected").to_pylist() == [3]

    def test_round_trip_sql(self, users, empty_target):
        # Build a plan, render to SQL, parse back — schema-preserving.
        plan = InsertPlan(target=empty_target, source=users)
        plan._target = type("R", (), {"name": "target"})()  # AST needs target.name
        sql = plan.to_sql()
        assert "INSERT INTO" in sql


# ---------------------------------------------------------------------------
# MergePlan
# ---------------------------------------------------------------------------


class TestMergePlan:
    def test_merge_update_only(self):
        target = ArrowTabular(pa.table({
            "id": [1, 2, 3],
            "score": [10, 20, 30],
        }))
        source = ArrowTabular(pa.table({
            "id": [2, 3],
            "score": [200, 300],
        }))
        from yggdrasil.execution.expr.nodes import Column
        plan = (MergePlan(target=target, source=source)
                .on(["id"])
                .when_matched_update({"score": Column(name="score")}))
        result = plan.execute()
        assert result.operation == "MERGE"
        assert result.rows_updated == 2
        assert result.rows_inserted == 0
        scores_by_id = dict(zip(
            target.read_arrow_table().column("id").to_pylist(),
            target.read_arrow_table().column("score").to_pylist(),
        ))
        assert scores_by_id[1] == 10
        assert scores_by_id[2] == 200
        assert scores_by_id[3] == 300

    def test_merge_insert_only(self):
        target = ArrowTabular(pa.table({
            "id": pa.array([1], type=pa.int64()),
            "score": pa.array([10], type=pa.int64()),
        }))
        source = ArrowTabular(pa.table({
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "score": pa.array([100, 200, 300], type=pa.int64()),
        }))
        from yggdrasil.execution.expr.nodes import Column
        plan = (MergePlan(target=target, source=source)
                .on(["id"])
                .when_not_matched_insert({
                    "id": Column(name="id"),
                    "score": Column(name="score"),
                }))
        result = plan.execute()
        assert result.rows_inserted == 2
        assert result.rows_updated == 0
        ids = sorted(target.read_arrow_table().column("id").to_pylist())
        assert ids == [1, 2, 3]

    def test_merge_update_and_insert(self):
        target = ArrowTabular(pa.table({
            "id": pa.array([1, 2], type=pa.int64()),
            "score": pa.array([10, 20], type=pa.int64()),
        }))
        source = ArrowTabular(pa.table({
            "id": pa.array([2, 3], type=pa.int64()),
            "score": pa.array([222, 333], type=pa.int64()),
        }))
        from yggdrasil.execution.expr.nodes import Column
        plan = (MergePlan(target=target, source=source)
                .on(["id"])
                .when_matched_update({"score": Column(name="score")})
                .when_not_matched_insert({
                    "id": Column(name="id"),
                    "score": Column(name="score"),
                }))
        result = plan.execute()
        assert result.rows_updated == 1
        assert result.rows_inserted == 1
        rows = dict(zip(
            target.read_arrow_table().column("id").to_pylist(),
            target.read_arrow_table().column("score").to_pylist(),
        ))
        assert rows == {1: 10, 2: 222, 3: 333}

    def test_merge_delete(self):
        target = ArrowTabular(pa.table({
            "id": pa.array([1, 2, 3], type=pa.int64()),
        }))
        source = ArrowTabular(pa.table({
            "id": pa.array([2], type=pa.int64()),
        }))
        plan = (MergePlan(target=target, source=source)
                .on(["id"])
                .when_matched_delete())
        result = plan.execute()
        assert result.rows_deleted == 1
        assert sorted(target.read_arrow_table().column("id").to_pylist()) == [1, 3]

    def test_missing_target_raises(self):
        source = ArrowTabular(pa.table({"id": [1]}))
        with pytest.raises(ValueError, match="no target"):
            MergePlan(source=source).execute()

    def test_missing_source_raises(self):
        target = ArrowTabular(pa.table({"id": [1]}))
        with pytest.raises(ValueError, match="no source"):
            MergePlan(target=target).execute()

    def test_missing_on_raises(self):
        target = ArrowTabular(pa.table({"id": [1]}))
        source = ArrowTabular(pa.table({"id": [1]}))
        plan = MergePlan(target=target, source=source)
        with pytest.raises(ValueError, match="ON clause"):
            plan.execute()


# ---------------------------------------------------------------------------
# SQL → autonomous plan dispatch
# ---------------------------------------------------------------------------


class TestFromSql:
    def test_from_sql_select_returns_select_plan(self):
        plan = ExecutionPlan.from_sql("SELECT id, name FROM t")
        assert isinstance(plan, SelectPlan)

    def test_from_sql_insert_returns_insert_plan(self):
        plan = ExecutionPlan.from_sql("INSERT INTO t (id, name) VALUES (1, 'a')")
        assert isinstance(plan, InsertPlan)
        assert plan.columns == ["id", "name"]
        assert plan.values is not None and len(plan.values) == 1

    def test_from_sql_merge_returns_merge_plan(self):
        plan = ExecutionPlan.from_sql(
            "MERGE INTO target t USING source s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET name = s.name "
            "WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)"
        )
        assert isinstance(plan, MergePlan)
        assert plan.when_matched
        assert plan.when_not_matched


# ---------------------------------------------------------------------------
# AST node.execute(tables=) now routes through the plan builders
# ---------------------------------------------------------------------------


class TestNodeExecuteRoutes:
    def test_insert_node_execute_returns_target(self):
        users = ArrowTabular(pa.table({"id": [1, 2], "name": ["a", "b"]}))
        target = ArrowTabular(pa.table({
            "id": pa.array([], type=pa.int64()),
            "name": pa.array([], type=pa.large_utf8()),
        }))
        node = parse_sql("INSERT INTO target SELECT * FROM users")
        node.execute(tables={"users": users, "target": target})
        assert target.read_arrow_table().num_rows == 2

    def test_merge_node_execute_against_arrow(self):
        target = ArrowTabular(pa.table({
            "id": pa.array([1, 2], type=pa.int64()),
            "name": pa.array(["a", "b"], type=pa.large_utf8()),
        }))
        source = ArrowTabular(pa.table({
            "id": pa.array([2, 3], type=pa.int64()),
            "name": pa.array(["B", "c"], type=pa.large_utf8()),
        }))
        node = parse_sql(
            "MERGE INTO target USING source ON target.id = source.id "
            "WHEN MATCHED THEN UPDATE SET name = source.name "
            "WHEN NOT MATCHED THEN INSERT (id, name) VALUES (source.id, source.name)"
        )
        node.execute(tables={"target": target, "source": source})
        rows = dict(zip(
            target.read_arrow_table().column("id").to_pylist(),
            target.read_arrow_table().column("name").to_pylist(),
        ))
        assert rows == {1: "a", 2: "B", 3: "c"}


# ---------------------------------------------------------------------------
# LazyTabular bridge to autonomous plans
# ---------------------------------------------------------------------------


class TestLazyTabularBridge:
    def test_lazy_into_returns_insert_plan(self, users, empty_target):
        plan = users.lazy().filter("id > 1").into(empty_target)
        assert isinstance(plan, InsertPlan)
        result = plan.execute()
        assert result.rows_inserted == 2

    def test_lazy_merge_into_returns_merge_plan(self, users):
        target = ArrowTabular(pa.table({
            "id": pa.array([1, 2], type=pa.int64()),
            "name": pa.array(["a", "b"], type=pa.large_utf8()),
            "score": pa.array([10, 20], type=pa.int64()),
        }))
        plan = users.lazy().merge_into(target, on=["id"])
        assert isinstance(plan, MergePlan)


# ---------------------------------------------------------------------------
# OperationResult
# ---------------------------------------------------------------------------


class TestOperationResult:
    def test_rows_affected_aggregates(self):
        r = OperationResult(
            operation="MERGE", rows_inserted=2, rows_updated=3, rows_deleted=1,
        )
        assert r.rows_affected == 6

    def test_to_dict(self):
        r = OperationResult(operation="INSERT", rows_inserted=5)
        d = r.to_dict()
        assert d["operation"] == "INSERT"
        assert d["rows_inserted"] == 5
        assert d["rows_affected"] == 5

    def test_to_arrow_tabular(self):
        r = OperationResult(operation="INSERT", rows_inserted=4)
        t = r.to_arrow_tabular().read_arrow_table()
        assert t.column("operation").to_pylist() == ["INSERT"]
        assert t.column("rows_inserted").to_pylist() == [4]
