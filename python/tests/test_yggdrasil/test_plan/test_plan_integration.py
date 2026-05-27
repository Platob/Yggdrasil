"""Integration tests — execute parsed SQL against real Tabular data."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.plan import (
    ExecutionPlan,
    LazyTabular,
    PlanNode,
    SelectNode,
    SelectPlan,
    parse_sql,
)
from yggdrasil.plan.nodes import ScanNode
from yggdrasil.plan.ops import TableRef


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def users():
    return ArrowTabular(pa.table({
        "id": [1, 2, 3, 4, 5],
        "name": ["alice", "bob", "carol", "dave", "eve"],
        "region": ["US", "EU", "US", "EU", "US"],
        "score": [90, 80, 95, 70, 85],
    }))


@pytest.fixture
def orders():
    return ArrowTabular(pa.table({
        "id": [1, 2, 1, 3],
        "order_id": [101, 102, 103, 104],
        "amount": [10.0, 20.0, 15.0, 30.0],
    }))


# ---------------------------------------------------------------------------
# Protocol tests — abstract ExecutionPlan contract
# ---------------------------------------------------------------------------

class TestExecutionPlanProtocol:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            ExecutionPlan()

    def test_select_plan_is_execution_plan(self):
        assert issubclass(SelectPlan, ExecutionPlan)

    def test_from_sql_returns_select_plan(self):
        plan = ExecutionPlan.from_sql("SELECT a FROM t WHERE a > 1")
        assert isinstance(plan, SelectPlan)

    def test_to_sql_on_select_plan(self):
        plan = SelectPlan()
        plan.select("a", "b").filter("id > 1").limit(10)
        sql = plan.to_sql(dialect="databricks")
        assert "SELECT" in sql


# ---------------------------------------------------------------------------
# PlanNode.execute integration
# ---------------------------------------------------------------------------

class TestPlanNodeExecution:
    def test_scan_node(self, users):
        node = ScanNode(name="users", tabular=users)
        result = node.execute()
        assert result.read_arrow_table().num_rows == 5

    def test_select_node_filter(self, users):
        node = parse_sql("SELECT * FROM users WHERE score > 80")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert all(s > 80 for s in table.column("score").to_pylist())

    def test_select_node_projection(self, users):
        node = parse_sql("SELECT id, name FROM users")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert table.column_names == ["id", "name"]

    def test_select_node_limit(self, users):
        node = parse_sql("SELECT * FROM users LIMIT 3")
        result = node.execute(tables={"users": users})
        assert result.read_arrow_table().num_rows == 3

    def test_select_node_order_by(self, users):
        node = parse_sql("SELECT * FROM users ORDER BY score DESC")
        result = node.execute(tables={"users": users})
        scores = result.read_arrow_table().column("score").to_pylist()
        assert scores == sorted(scores, reverse=True)

    def test_select_node_distinct(self, users):
        node = parse_sql("SELECT DISTINCT region FROM users")
        result = node.execute(tables={"users": users})
        regions = result.read_arrow_table().column("region").to_pylist()
        assert len(regions) == len(set(regions))

    def test_select_node_group_by(self, users):
        node = parse_sql(
            "SELECT region, COUNT(*) AS cnt FROM users GROUP BY region"
        )
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert "region" in table.column_names
        assert table.num_rows == 2

    def test_select_node_join(self, users, orders):
        node = parse_sql(
            "SELECT * FROM users INNER JOIN orders ON users.id = orders.id"
        )
        result = node.execute(tables={"users": users, "orders": orders})
        table = result.read_arrow_table()
        assert table.num_rows > 0
        assert "order_id" in table.column_names

    def test_cte_execution(self, users):
        node = parse_sql(
            "WITH top_users AS (SELECT * FROM users WHERE score > 80) "
            "SELECT * FROM top_users"
        )
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert all(s > 80 for s in table.column("score").to_pylist())


# ---------------------------------------------------------------------------
# LazyTabular with SQL round-trip
# ---------------------------------------------------------------------------

class TestLazyTabularSQL:
    def test_lazy_select_filter(self, users):
        lazy = users.lazy().select("id", "name").filter("score > 80")
        result = lazy.read_arrow_table()
        assert result.column_names == ["id", "name"]
        assert result.num_rows > 0

    def test_lazy_to_sql(self, users):
        lazy = users.lazy()
        lazy.select("id", "name").filter("score > 80").limit(2)
        sql = lazy.plan.to_sql(dialect="databricks")
        assert "SELECT" in sql

    def test_lazy_join_with_tabular(self, users, orders):
        result = (users.lazy()
                  .join(orders, on="id", how="inner")
                  .read_arrow_table())
        assert result.num_rows > 0

    def test_lazy_plan_is_select_plan(self, users):
        lazy = users.lazy()
        assert isinstance(lazy.plan, SelectPlan)


# ---------------------------------------------------------------------------
# SelectPlan.from_sql
# ---------------------------------------------------------------------------

class TestSelectPlanFromSQL:
    def test_basic(self):
        plan = ExecutionPlan.from_sql("SELECT a, b FROM t WHERE a > 1 LIMIT 10")
        assert isinstance(plan, SelectPlan)
        assert plan.columns is not None
        assert plan.limit_rows == 10
        assert plan.predicate is not None

    def test_execute_from_sql(self, users):
        plan = ExecutionPlan.from_sql("SELECT id, name FROM users WHERE score > 80 LIMIT 2")
        # Execute by applying the parsed filters programmatically
        result = plan.execute(users)
        table = result.read_arrow_table()
        assert table.num_rows <= 2
