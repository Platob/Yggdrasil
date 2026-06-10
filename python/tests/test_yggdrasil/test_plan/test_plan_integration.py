"""Integration tests — execute parsed SQL against real Tabular data."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.plan import (
    ExecutionPlan,
    SelectNode,
    SelectPlan,
    parse_sql,
)
from yggdrasil.plan.nodes import ScanNode
from yggdrasil.plan.ops import JoinClause, TableRef


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


# ---------------------------------------------------------------------------
# TestGroupByExecution — verify aggregation results
# ---------------------------------------------------------------------------

class TestGroupByExecution:
    def test_count_star(self, users):
        node = parse_sql("SELECT region, COUNT(*) AS cnt FROM users GROUP BY region")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert table.num_rows == 2
        # Verify counts add up
        cnts = table.column("cnt").to_pylist()
        assert sum(cnts) == 5

    def test_sum(self, users):
        node = parse_sql("SELECT region, SUM(score) AS total FROM users GROUP BY region")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert table.num_rows == 2
        # US: 90 + 95 + 85 = 270, EU: 80 + 70 = 150
        for row in table.to_pylist():
            if row["region"] == "US":
                assert row["total"] == 270
            else:
                assert row["total"] == 150

    def test_avg(self, users):
        node = parse_sql("SELECT region, AVG(score) AS avg_score FROM users GROUP BY region")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert table.num_rows == 2

    def test_min_max(self, users):
        node = parse_sql("SELECT region, MIN(score) AS lo, MAX(score) AS hi FROM users GROUP BY region")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        for row in table.to_pylist():
            if row["region"] == "US":
                assert row["lo"] == 85
                assert row["hi"] == 95
            else:
                assert row["lo"] == 70
                assert row["hi"] == 80


# ---------------------------------------------------------------------------
# TestHavingExecution
# ---------------------------------------------------------------------------

class TestHavingExecution:
    def test_having_filters_groups(self, users):
        # Use the alias column name in HAVING so the arrow filter can resolve it;
        # the engine evaluates HAVING as a post-aggregation filter on column names.
        node = parse_sql(
            "SELECT region, COUNT(*) AS cnt FROM users "
            "GROUP BY region HAVING cnt > 2"
        )
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        # US has 3 users (passes), EU has 2 (fails)
        assert table.num_rows == 1
        assert table.column("region").to_pylist() == ["US"]


# ---------------------------------------------------------------------------
# TestUnionExecution
# ---------------------------------------------------------------------------

class TestUnionExecution:
    def test_union_all(self, users):
        extra = ArrowTabular(pa.table({
            "id": [10, 11],
            "name": ["frank", "grace"],
            "region": ["EU", "US"],
            "score": [88, 92],
        }))
        node = parse_sql("SELECT * FROM users UNION ALL SELECT * FROM extra")
        result = node.execute(tables={"users": users, "extra": extra})
        table = result.read_arrow_table()
        assert table.num_rows == 7  # 5 + 2


# ---------------------------------------------------------------------------
# TestOrderByExecution
# ---------------------------------------------------------------------------

class TestOrderByExecution:
    def test_order_by_multiple_columns(self, users):
        node = parse_sql("SELECT * FROM users ORDER BY region ASC, score DESC")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        rows = table.to_pylist()
        # First EU rows (sorted by score DESC), then US rows (sorted by score DESC)
        eu_rows = [r for r in rows if r["region"] == "EU"]
        us_rows = [r for r in rows if r["region"] == "US"]
        assert eu_rows[0]["score"] >= eu_rows[-1]["score"]
        assert us_rows[0]["score"] >= us_rows[-1]["score"]
        # EU comes before US (alphabetical ASC)
        eu_idx = [i for i, r in enumerate(rows) if r["region"] == "EU"]
        us_idx = [i for i, r in enumerate(rows) if r["region"] == "US"]
        assert max(eu_idx) < min(us_idx)


# ---------------------------------------------------------------------------
# TestOffsetExecution
# ---------------------------------------------------------------------------

class TestOffsetExecution:
    def test_limit_offset(self, users):
        node = parse_sql("SELECT * FROM users ORDER BY id LIMIT 2 OFFSET 2")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert table.num_rows == 2
        ids = table.column("id").to_pylist()
        assert ids == [3, 4]  # skipped 1,2 and took 2


# ---------------------------------------------------------------------------
# TestDistinctExecution
# ---------------------------------------------------------------------------

class TestDistinctExecution:
    def test_distinct_values(self, users):
        node = parse_sql("SELECT DISTINCT region FROM users")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        regions = table.column("region").to_pylist()
        assert len(regions) == 2
        assert set(regions) == {"US", "EU"}


# ---------------------------------------------------------------------------
# TestCTEExecution
# ---------------------------------------------------------------------------

class TestCTEExecution:
    def test_cte_with_filter(self, users):
        node = parse_sql(
            "WITH high_scorers AS ("
            "  SELECT * FROM users WHERE score >= 85"
            ") "
            "SELECT name, score FROM high_scorers"
        )
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert all(s >= 85 for s in table.column("score").to_pylist())

    def test_multiple_ctes(self, users, orders):
        node = parse_sql(
            "WITH us_users AS ("
            "  SELECT * FROM users WHERE region = 'US'"
            "), us_orders AS ("
            "  SELECT * FROM orders WHERE id IN (1, 3, 5)"
            ") "
            "SELECT * FROM us_users"
        )
        result = node.execute(tables={"users": users, "orders": orders})
        table = result.read_arrow_table()
        assert all(r == "US" for r in table.column("region").to_pylist())


# ---------------------------------------------------------------------------
# TestSQLEmitterRoundTrips — parse -> emit -> parse -> verify structure matches
# ---------------------------------------------------------------------------

class TestSQLEmitterRoundTrips:
    def _roundtrip(self, sql: str, dialect: str = "databricks"):
        """Parse, emit, re-parse, verify key structures survive."""
        node1 = parse_sql(sql, dialect=dialect)
        emitted = node1.to_sql(dialect=dialect)
        node2 = parse_sql(emitted, dialect=dialect)
        return node1, node2, emitted

    def test_roundtrip_select_where(self):
        _, node2, sql = self._roundtrip("SELECT a, b FROM t WHERE id > 10")
        assert len(node2.projections) == 2
        assert node2.where is not None
        assert isinstance(node2.from_clause, TableRef)

    def test_roundtrip_group_by(self):
        _, node2, _ = self._roundtrip(
            "SELECT region, COUNT(*) FROM t GROUP BY region"
        )
        assert node2.group_by is not None

    def test_roundtrip_order_by_desc(self):
        _, node2, sql = self._roundtrip("SELECT * FROM t ORDER BY score DESC")
        assert "DESC" in sql
        assert node2.order_by is not None
        assert node2.order_by[0].ascending is False

    def test_roundtrip_distinct(self):
        _, node2, sql = self._roundtrip("SELECT DISTINCT region FROM t")
        assert "DISTINCT" in sql
        assert node2.distinct is True

    def test_roundtrip_join(self):
        _, node2, sql = self._roundtrip(
            "SELECT * FROM a INNER JOIN b ON a.id = b.id"
        )
        assert "JOIN" in sql
        assert isinstance(node2.from_clause, JoinClause)

    def test_roundtrip_having(self):
        _, node2, sql = self._roundtrip(
            "SELECT region, COUNT(*) FROM t GROUP BY region HAVING COUNT(*) > 5"
        )
        assert "HAVING" in sql
        assert node2.having is not None

    def test_roundtrip_cte(self):
        _, node2, sql = self._roundtrip(
            "WITH cte AS (SELECT a FROM t) SELECT * FROM cte"
        )
        assert "WITH" in sql
        assert node2.ctes is not None

    def test_roundtrip_union(self):
        _, node2, sql = self._roundtrip(
            "SELECT a FROM t1 UNION ALL SELECT a FROM t2"
        )
        assert "UNION ALL" in sql
        assert node2.set_ops is not None

    def test_roundtrip_lateral_view(self):
        _, node2, sql = self._roundtrip(
            "SELECT id, val FROM t LATERAL VIEW EXPLODE(arr) vals AS val",
            dialect="databricks",
        )
        assert "LATERAL VIEW" in sql
        assert node2.lateral_views is not None

    def test_roundtrip_case_when(self):
        _, node2, sql = self._roundtrip(
            "SELECT CASE WHEN a > 0 THEN 1 ELSE 0 END FROM t"
        )
        assert "CASE" in sql
        assert "WHEN" in sql

    def test_roundtrip_window_function(self):
        _, node2, sql = self._roundtrip(
            "SELECT ROW_NUMBER() OVER (PARTITION BY region ORDER BY id) FROM t",
            dialect="databricks",
        )
        assert "OVER" in sql
        assert "PARTITION BY" in sql

    def test_roundtrip_function_alias(self):
        _, node2, sql = self._roundtrip(
            "SELECT COUNT(*) AS total FROM t"
        )
        assert "AS" in sql

    def test_roundtrip_insert(self):
        _, _, sql = self._roundtrip(
            "INSERT INTO target SELECT * FROM source"
        )
        assert "INSERT INTO" in sql

    def test_roundtrip_limit_offset(self):
        _, node2, sql = self._roundtrip("SELECT * FROM t LIMIT 10 OFFSET 5")
        assert "LIMIT 10" in sql
        assert "OFFSET 5" in sql
        assert node2.limit == 10
        assert node2.offset == 5


# ---------------------------------------------------------------------------
# TestSelectPlanNodeConversion
# ---------------------------------------------------------------------------

class TestSelectPlanNodeConversion:
    def test_to_plan_node_select(self):
        plan = SelectPlan()
        plan.select("a", "b")
        node = plan.to_plan_node()
        assert isinstance(node, SelectNode)
        # Should have Column projections
        from yggdrasil.execution.expr.nodes import Column
        assert any(isinstance(p, Column) and p.name == "a" for p in node.projections)

    def test_to_plan_node_with_filter(self):
        plan = SelectPlan()
        plan.filter("x > 1")
        node = plan.to_plan_node()
        assert node.where is not None

    def test_to_plan_node_with_limit(self):
        plan = SelectPlan()
        plan.limit(10)
        node = plan.to_plan_node()
        assert node.limit == 10

    def test_to_sql_basic(self):
        plan = SelectPlan()
        plan.select("a", "b").filter("id > 5").limit(10)
        sql = plan.to_sql(dialect="databricks")
        assert "SELECT" in sql
        assert "LIMIT 10" in sql


# ---------------------------------------------------------------------------
# TestComplexQueryExecution — end-to-end complex queries
# ---------------------------------------------------------------------------

class TestComplexQueryExecution:
    def test_filter_and_order_and_limit(self, users):
        node = parse_sql(
            "SELECT id, name, score FROM users WHERE score >= 80 ORDER BY score DESC LIMIT 3"
        )
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert table.num_rows == 3
        scores = table.column("score").to_pylist()
        assert scores == sorted(scores, reverse=True)
        assert all(s >= 80 for s in scores)

    def test_join_and_filter(self, users, orders):
        node = parse_sql(
            "SELECT users.name, orders.amount "
            "FROM users INNER JOIN orders ON users.id = orders.id "
            "WHERE orders.amount > 15"
        )
        result = node.execute(tables={"users": users, "orders": orders})
        table = result.read_arrow_table()
        assert all(a > 15 for a in table.column("amount").to_pylist())

    def test_group_by_and_order(self, users):
        node = parse_sql(
            "SELECT region, COUNT(*) AS cnt FROM users "
            "GROUP BY region ORDER BY cnt DESC"
        )
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        cnts = table.column("cnt").to_pylist()
        assert cnts == sorted(cnts, reverse=True)


# ---------------------------------------------------------------------------
# HIGH priority gap: INSERT execution
# ---------------------------------------------------------------------------

class TestInsertExecution:
    def test_insert_select(self, users):
        target = ArrowTabular(pa.table({
            "id": pa.array([], type=pa.int64()),
            "name": pa.array([], type=pa.utf8()),
            "region": pa.array([], type=pa.utf8()),
            "score": pa.array([], type=pa.int64()),
        }))
        node = parse_sql("INSERT INTO target SELECT * FROM users")
        node.execute(tables={"users": users, "target": target})
        result = target.read_arrow_table()
        assert result.num_rows == 5

    def test_insert_values(self, users):
        target = ArrowTabular(pa.table({
            "id": pa.array([], type=pa.int64()),
            "name": pa.array([], type=pa.utf8()),
        }))
        node = parse_sql("INSERT INTO target (id, name) VALUES (1, 'alice'), (2, 'bob')")
        node.execute(tables={"target": target})
        result = target.read_arrow_table()
        assert result.num_rows == 2
        assert result.column("id").to_pylist() == [1, 2]


# ---------------------------------------------------------------------------
# HIGH priority gap: aggregate without GROUP BY
# ---------------------------------------------------------------------------

class TestAggregateWithoutGroupBy:
    def test_count_star_no_group_by(self, users):
        node = parse_sql("SELECT COUNT(*) AS cnt FROM users")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert table.num_rows == 1
        assert table.column("cnt").to_pylist() == [5]

    def test_sum_no_group_by(self, users):
        node = parse_sql("SELECT SUM(score) AS total FROM users")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert table.column("total").to_pylist() == [420]

    def test_avg_no_group_by(self, users):
        node = parse_sql("SELECT AVG(score) AS avg_score FROM users")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert table.num_rows == 1
        avg = table.column("avg_score").to_pylist()[0]
        assert abs(avg - 84.0) < 0.01

    def test_min_max_no_group_by(self, users):
        node = parse_sql("SELECT MIN(score) AS lo, MAX(score) AS hi FROM users")
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        assert table.column("lo").to_pylist() == [70]
        assert table.column("hi").to_pylist() == [95]


# ---------------------------------------------------------------------------
# MEDIUM priority: HAVING with COUNT(*)
# ---------------------------------------------------------------------------

class TestHavingCountStar:
    def test_having_count_star(self, users):
        node = parse_sql(
            "SELECT region, COUNT(*) AS cnt FROM users "
            "GROUP BY region HAVING COUNT(*) > 2"
        )
        result = node.execute(tables={"users": users})
        table = result.read_arrow_table()
        # US has 3 users (>2), EU has 2 (not >2)
        assert table.num_rows == 1
        assert table.column("region").to_pylist() == ["US"]


# ---------------------------------------------------------------------------
# MEDIUM priority: Subquery IN
# ---------------------------------------------------------------------------

class TestSubqueryIn:
    def test_in_subquery_parses(self):
        node = parse_sql(
            "SELECT * FROM t WHERE id IN (SELECT id FROM users)"
        )
        assert isinstance(node, SelectNode)
        assert node.where is not None


# ---------------------------------------------------------------------------
# MEDIUM priority: Expression node equality/hashing
# ---------------------------------------------------------------------------

class TestExpressionNodeEquality:
    def test_function_call_equality(self):
        from yggdrasil.execution.expr.nodes import FunctionCall, Column
        fc1 = FunctionCall("COUNT", (Column(name="id"),), distinct=False)
        fc2 = FunctionCall("COUNT", (Column(name="id"),), distinct=False)
        assert fc1.equals(fc2)
        assert hash(fc1) == hash(fc2)

    def test_function_call_distinct_differs(self):
        from yggdrasil.execution.expr.nodes import FunctionCall, Column
        fc1 = FunctionCall("COUNT", (Column(name="id"),), distinct=True)
        fc2 = FunctionCall("COUNT", (Column(name="id"),), distinct=False)
        assert not fc1.equals(fc2)

    def test_star_equality(self):
        from yggdrasil.execution.expr.nodes import Star
        assert Star().equals(Star())
        assert Star(qualifier="t").equals(Star(qualifier="t"))
        assert not Star().equals(Star(qualifier="t"))

    def test_alias_equality(self):
        from yggdrasil.execution.expr.nodes import Alias, Column
        a1 = Alias(Column(name="x"), "y")
        a2 = Alias(Column(name="x"), "y")
        assert a1.equals(a2)
        assert hash(a1) == hash(a2)

    def test_subscript_equality(self):
        from yggdrasil.execution.expr.nodes import Subscript, Column, Literal
        s1 = Subscript(Column(name="arr"), Literal(0))
        s2 = Subscript(Column(name="arr"), Literal(0))
        assert s1.equals(s2)
        assert hash(s1) == hash(s2)

    def test_case_when_equality(self):
        from yggdrasil.execution.expr.nodes import CaseWhen, Comparison, Column, Literal
        from yggdrasil.execution.expr.operators import CompareOp
        branch = (Comparison(Column(name="a"), CompareOp.GT, Literal(0)), Literal(1))
        cw1 = CaseWhen(branches=(branch,), else_expr=Literal(0))
        cw2 = CaseWhen(branches=(branch,), else_expr=Literal(0))
        assert cw1.equals(cw2)

    def test_sort_order_equality(self):
        from yggdrasil.execution.expr.nodes import SortOrder, Column
        so1 = SortOrder(Column(name="a"), ascending=False, nulls_first=True)
        so2 = SortOrder(Column(name="a"), ascending=False, nulls_first=True)
        assert so1.equals(so2)


# ---------------------------------------------------------------------------
# MEDIUM priority: expression-level SQL round-trip for new nodes
# ---------------------------------------------------------------------------

class TestExpressionSQLRoundTrip:
    def test_function_call_render(self):
        from yggdrasil.execution.expr.backends.sql import to_sql
        from yggdrasil.execution.expr.nodes import FunctionCall, Column
        fc = FunctionCall("UPPER", (Column(name="name"),))
        sql = to_sql(fc, dialect="databricks")
        assert sql == "UPPER(`name`)"

    def test_subscript_render(self):
        from yggdrasil.execution.expr.backends.sql import to_sql
        from yggdrasil.execution.expr.nodes import Subscript, Column, Literal
        s = Subscript(Column(name="arr"), Literal(0))
        sql = to_sql(s, dialect="databricks")
        assert sql == "`arr`[0]"

    def test_case_when_render(self):
        from yggdrasil.execution.expr.backends.sql import to_sql, from_sql
        expr = from_sql("CASE WHEN x > 0 THEN 1 ELSE 0 END")
        sql = to_sql(expr, dialect="databricks")
        assert "CASE" in sql
        assert "WHEN" in sql
        assert "ELSE" in sql
        assert "END" in sql

    def test_window_function_render(self):
        from yggdrasil.execution.expr.backends.sql import to_sql, from_sql
        expr = from_sql("ROW_NUMBER() OVER (PARTITION BY region ORDER BY id)")
        sql = to_sql(expr, dialect="databricks")
        assert "OVER" in sql
        assert "PARTITION BY" in sql

    def test_star_render(self):
        from yggdrasil.execution.expr.backends.sql import to_sql
        from yggdrasil.execution.expr.nodes import Star
        assert to_sql(Star(), dialect="databricks") == "*"
        assert ".*" in to_sql(Star(qualifier="t"), dialect="databricks")

    def test_alias_render(self):
        from yggdrasil.execution.expr.backends.sql import to_sql
        from yggdrasil.execution.expr.nodes import Alias, Column
        a = Alias(Column(name="x"), "y")
        sql = to_sql(a, dialect="databricks")
        assert "AS" in sql

    def test_sort_order_render(self):
        from yggdrasil.execution.expr.backends.sql import to_sql
        from yggdrasil.execution.expr.nodes import SortOrder, Column
        so = SortOrder(Column(name="a"), ascending=False, nulls_first=True)
        sql = to_sql(so, dialect="databricks")
        assert "DESC" in sql
        assert "NULLS FIRST" in sql


# ---------------------------------------------------------------------------
# MEDIUM priority: walk() traversal for new nodes
# ---------------------------------------------------------------------------

class TestWalkNewNodes:
    def test_walk_function_call(self):
        from yggdrasil.execution.expr import free_columns
        from yggdrasil.execution.expr.nodes import FunctionCall, Column
        fc = FunctionCall("UPPER", (Column(name="name"),))
        cols = free_columns(fc)
        assert "name" in cols

    def test_walk_subscript(self):
        from yggdrasil.execution.expr import free_columns
        from yggdrasil.execution.expr.nodes import Subscript, Column
        s = Subscript(Column(name="arr"), Column(name="idx"))
        cols = free_columns(s)
        assert "arr" in cols
        assert "idx" in cols

    def test_walk_case_when(self):
        from yggdrasil.execution.expr import free_columns
        from yggdrasil.execution.expr.nodes import CaseWhen, Comparison, Column, Literal
        from yggdrasil.execution.expr.operators import CompareOp
        cw = CaseWhen(
            branches=((Comparison(Column(name="a"), CompareOp.GT, Literal(0)), Column(name="b")),),
            else_expr=Column(name="c"),
        )
        cols = free_columns(cw)
        assert set(cols) == {"a", "b", "c"}

    def test_walk_window_function(self):
        from yggdrasil.execution.expr import free_columns
        from yggdrasil.execution.expr.nodes import (
            FunctionCall, WindowFunction, WindowSpec, SortOrder, Column,
        )
        wf = WindowFunction(
            function=FunctionCall("ROW_NUMBER", ()),
            window=WindowSpec(
                partition_by=(Column(name="region"),),
                order_by=(SortOrder(Column(name="id"), ascending=True),),
            ),
        )
        cols = free_columns(wf)
        assert "region" in cols
        assert "id" in cols


# ---------------------------------------------------------------------------
# MEDIUM priority: ExecutionPlan.from_sql for non-SELECT
# ---------------------------------------------------------------------------

class TestExecutionPlanFromSQLEdgeCases:
    def test_from_sql_with_group_by(self):
        plan = ExecutionPlan.from_sql(
            "SELECT region, COUNT(*) FROM t GROUP BY region"
        )
        assert isinstance(plan, SelectPlan)

    def test_from_sql_complex_expression_projections(self):
        plan = ExecutionPlan.from_sql("SELECT a + b FROM t")
        # Complex expressions don't map to _select columns
        assert plan.columns is None
