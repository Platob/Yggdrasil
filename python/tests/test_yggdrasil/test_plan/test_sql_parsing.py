"""SQL parsing tests — covers Databricks SQL queries.

Tests: basic SELECT, FROM, WHERE, GROUP BY, JOIN, UNION,
WITH (CTE), ORDER BY, LIMIT, LATERAL VIEW, function calls,
CASE WHEN, window functions, and Databricks-specific features.
"""

from __future__ import annotations

import pytest

from yggdrasil.plan import SelectNode, parse_sql, ScanNode, InsertNode, MergeNode
from yggdrasil.plan.nodes import PlanNode
from yggdrasil.plan.ops import (
    CTE, JoinClause, LateralViewItem, SetOp, SubqueryRef, TableRef,
)
from yggdrasil.execution.expr.nodes import (
    Alias,
    Arithmetic,
    Between,
    CaseWhen,
    Cast,
    Column,
    Comparison,
    Expression,
    FunctionCall,
    InList,
    IsNull,
    Like,
    Literal,
    Logical,
    Not,
    SortOrder,
    Star,
    Subscript,
    WindowFunction,
    WindowSpec,
)
from yggdrasil.execution.expr.operators import CompareOp, LogicalOp


# ---------------------------------------------------------------------------
# Basic SELECT
# ---------------------------------------------------------------------------

class TestSelectBasic:
    def test_select_star(self):
        node = parse_sql("SELECT * FROM t")
        assert isinstance(node, SelectNode)
        assert len(node.projections) == 1
        assert isinstance(node.projections[0], Star)

    def test_select_columns(self):
        node = parse_sql("SELECT a, b, c FROM t")
        assert len(node.projections) == 3
        assert all(isinstance(p, Column) for p in node.projections)
        assert [p.name for p in node.projections] == ["a", "b", "c"]

    def test_select_alias(self):
        node = parse_sql("SELECT a AS x, b y FROM t")
        assert len(node.projections) == 2
        p0 = node.projections[0]
        assert isinstance(p0, Alias)
        assert p0.name == "x"
        p1 = node.projections[1]
        assert isinstance(p1, Alias)
        assert p1.name == "y"

    def test_select_distinct(self):
        node = parse_sql("SELECT DISTINCT a FROM t")
        assert node.distinct is True

    def test_select_expression(self):
        node = parse_sql("SELECT a + b FROM t")
        assert len(node.projections) == 1
        assert isinstance(node.projections[0], Arithmetic)

    def test_select_literal(self):
        node = parse_sql("SELECT 1, 'hello', NULL")
        assert len(node.projections) == 3
        assert isinstance(node.projections[0], Literal)
        assert node.projections[0].value == 1
        assert isinstance(node.projections[1], Literal)
        assert node.projections[1].value == "hello"
        assert isinstance(node.projections[2], Literal)
        assert node.projections[2].value is None

    def test_select_qualified_star(self):
        node = parse_sql("SELECT t.* FROM t")
        assert len(node.projections) == 1
        assert isinstance(node.projections[0], Star)
        assert node.projections[0].qualifier == "t"


# ---------------------------------------------------------------------------
# FROM clause
# ---------------------------------------------------------------------------

class TestFromClause:
    def test_from_table(self):
        node = parse_sql("SELECT * FROM users")
        assert isinstance(node.from_clause, TableRef)
        assert node.from_clause.name == "users"

    def test_from_table_alias(self):
        node = parse_sql("SELECT * FROM users u")
        assert isinstance(node.from_clause, TableRef)
        assert node.from_clause.name == "users"
        assert node.from_clause.alias == "u"

    def test_from_schema_table(self):
        node = parse_sql("SELECT * FROM myschema.users")
        assert isinstance(node.from_clause, TableRef)
        assert node.from_clause.schema == "myschema"
        assert node.from_clause.name == "users"

    def test_from_catalog_schema_table(self):
        node = parse_sql("SELECT * FROM catalog.myschema.users")
        assert isinstance(node.from_clause, TableRef)
        assert node.from_clause.catalog == "catalog"
        assert node.from_clause.schema == "myschema"
        assert node.from_clause.name == "users"

    def test_from_subquery(self):
        node = parse_sql("SELECT * FROM (SELECT a FROM t) sub")
        assert isinstance(node.from_clause, SubqueryRef)
        assert node.from_clause.alias == "sub"
        inner = node.from_clause.plan
        assert isinstance(inner, SelectNode)


# ---------------------------------------------------------------------------
# WHERE clause
# ---------------------------------------------------------------------------

class TestWhereClause:
    def test_simple_comparison(self):
        node = parse_sql("SELECT * FROM t WHERE id > 10")
        assert isinstance(node.where, Comparison)
        assert node.where.op == CompareOp.GT

    def test_and_or(self):
        node = parse_sql("SELECT * FROM t WHERE a > 1 AND b < 2 OR c = 3")
        assert isinstance(node.where, Logical)

    def test_in_list(self):
        node = parse_sql("SELECT * FROM t WHERE id IN (1, 2, 3)")
        assert isinstance(node.where, InList)
        assert node.where.values == (1, 2, 3)

    def test_between(self):
        node = parse_sql("SELECT * FROM t WHERE id BETWEEN 1 AND 10")
        assert isinstance(node.where, Between)

    def test_like(self):
        node = parse_sql("SELECT * FROM t WHERE name LIKE '%test%'")
        assert isinstance(node.where, Like)
        assert node.where.pattern == "%test%"

    def test_is_null(self):
        node = parse_sql("SELECT * FROM t WHERE x IS NULL")
        assert isinstance(node.where, IsNull)
        assert node.where.negated is False

    def test_is_not_null(self):
        node = parse_sql("SELECT * FROM t WHERE x IS NOT NULL")
        assert isinstance(node.where, IsNull)
        assert node.where.negated is True

    def test_not_in(self):
        node = parse_sql("SELECT * FROM t WHERE id NOT IN (1, 2)")
        assert isinstance(node.where, InList)
        assert node.where.negated is True

    def test_complex_predicate(self):
        node = parse_sql(
            "SELECT * FROM t WHERE (a > 1 AND b < 2) OR c IN (1, 2, 3)"
        )
        assert isinstance(node.where, Logical)
        assert node.where.op == LogicalOp.OR


# ---------------------------------------------------------------------------
# GROUP BY / HAVING
# ---------------------------------------------------------------------------

class TestGroupBy:
    def test_group_by(self):
        node = parse_sql("SELECT region, COUNT(*) FROM t GROUP BY region")
        assert node.group_by is not None
        assert len(node.group_by) == 1
        assert isinstance(node.group_by[0], Column)
        assert node.group_by[0].name == "region"

    def test_group_by_multiple(self):
        node = parse_sql("SELECT a, b, SUM(c) FROM t GROUP BY a, b")
        assert len(node.group_by) == 2

    def test_having(self):
        node = parse_sql(
            "SELECT region, COUNT(*) cnt FROM t "
            "GROUP BY region HAVING COUNT(*) > 10"
        )
        assert node.having is not None
        assert isinstance(node.having, Comparison)


# ---------------------------------------------------------------------------
# JOIN
# ---------------------------------------------------------------------------

class TestJoin:
    def test_inner_join(self):
        node = parse_sql(
            "SELECT * FROM a INNER JOIN b ON a.id = b.id"
        )
        assert isinstance(node.from_clause, JoinClause)
        assert node.from_clause.join_type.is_inner

    def test_left_join(self):
        node = parse_sql(
            "SELECT * FROM a LEFT JOIN b ON a.id = b.id"
        )
        assert isinstance(node.from_clause, JoinClause)
        assert node.from_clause.join_type.is_outer

    def test_right_join(self):
        node = parse_sql(
            "SELECT * FROM a RIGHT OUTER JOIN b ON a.id = b.id"
        )
        assert node.from_clause.join_type.name == "RIGHT_OUTER"

    def test_full_join(self):
        node = parse_sql(
            "SELECT * FROM a FULL OUTER JOIN b ON a.id = b.id"
        )
        assert node.from_clause.join_type.name == "FULL_OUTER"

    def test_cross_join(self):
        node = parse_sql("SELECT * FROM a CROSS JOIN b")
        assert node.from_clause.join_type.is_cross

    def test_multi_join(self):
        node = parse_sql(
            "SELECT * FROM a JOIN b ON a.id = b.id "
            "LEFT JOIN c ON b.id = c.id"
        )
        # Outer join wraps the inner join
        assert isinstance(node.from_clause, JoinClause)
        assert isinstance(node.from_clause.left, JoinClause)

    def test_join_subquery(self):
        node = parse_sql(
            "SELECT * FROM a JOIN (SELECT * FROM b) sub ON a.id = sub.id"
        )
        jc = node.from_clause
        assert isinstance(jc, JoinClause)
        assert isinstance(jc.right, SubqueryRef)


# ---------------------------------------------------------------------------
# ORDER BY / LIMIT / OFFSET
# ---------------------------------------------------------------------------

class TestOrderByLimit:
    def test_order_by(self):
        node = parse_sql("SELECT * FROM t ORDER BY a ASC, b DESC")
        assert node.order_by is not None
        assert len(node.order_by) == 2
        assert isinstance(node.order_by[0], SortOrder)
        assert node.order_by[0].ascending is True
        assert node.order_by[1].ascending is False

    def test_order_by_nulls(self):
        node = parse_sql("SELECT * FROM t ORDER BY a ASC NULLS FIRST")
        assert node.order_by[0].nulls_first is True

    def test_limit(self):
        node = parse_sql("SELECT * FROM t LIMIT 10")
        assert node.limit == 10

    def test_limit_offset(self):
        node = parse_sql("SELECT * FROM t LIMIT 10 OFFSET 20")
        assert node.limit == 10
        assert node.offset == 20


# ---------------------------------------------------------------------------
# UNION / INTERSECT / EXCEPT
# ---------------------------------------------------------------------------

class TestSetOps:
    def test_union_all(self):
        node = parse_sql("SELECT a FROM t1 UNION ALL SELECT a FROM t2")
        assert node.set_ops is not None
        assert len(node.set_ops) == 1
        assert node.set_ops[0].kind == "UNION ALL"

    def test_union(self):
        node = parse_sql("SELECT a FROM t1 UNION SELECT a FROM t2")
        assert node.set_ops[0].kind == "UNION"

    def test_intersect(self):
        node = parse_sql("SELECT a FROM t1 INTERSECT SELECT a FROM t2")
        assert node.set_ops[0].kind == "INTERSECT"

    def test_except(self):
        node = parse_sql("SELECT a FROM t1 EXCEPT SELECT a FROM t2")
        assert node.set_ops[0].kind == "EXCEPT"


# ---------------------------------------------------------------------------
# WITH (CTE)
# ---------------------------------------------------------------------------

class TestCTE:
    def test_simple_cte(self):
        node = parse_sql(
            "WITH cte AS (SELECT a FROM t) SELECT * FROM cte"
        )
        assert node.ctes is not None
        assert len(node.ctes) == 1
        assert node.ctes[0].name == "cte"

    def test_multiple_ctes(self):
        node = parse_sql(
            "WITH a AS (SELECT 1 x), b AS (SELECT 2 y) "
            "SELECT * FROM a JOIN b ON a.x = b.y"
        )
        assert len(node.ctes) == 2
        assert node.ctes[0].name == "a"
        assert node.ctes[1].name == "b"


# ---------------------------------------------------------------------------
# Function calls
# ---------------------------------------------------------------------------

class TestFunctionCalls:
    def test_count_star(self):
        node = parse_sql("SELECT COUNT(*) FROM t")
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)
        assert fc.name == "COUNT"
        assert isinstance(fc.args[0], Star)

    def test_count_distinct(self):
        node = parse_sql("SELECT COUNT(DISTINCT id) FROM t")
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)
        assert fc.distinct is True

    def test_sum_avg_min_max(self):
        for func in ("SUM", "AVG", "MIN", "MAX"):
            node = parse_sql(f"SELECT {func}(val) FROM t")
            fc = node.projections[0]
            assert isinstance(fc, FunctionCall)
            assert fc.name == func

    def test_nested_function(self):
        node = parse_sql("SELECT COALESCE(a, b, 0) FROM t")
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)
        assert fc.name == "COALESCE"
        assert len(fc.args) == 3

    def test_cast(self):
        node = parse_sql("SELECT CAST(id AS BIGINT) FROM t")
        assert isinstance(node.projections[0], Cast)


# ---------------------------------------------------------------------------
# Databricks date/time functions
# ---------------------------------------------------------------------------

class TestDatabricksDateFunctions:
    def test_date_trunc(self):
        node = parse_sql("SELECT DATE_TRUNC('month', ts) FROM t",
                          dialect="databricks")
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)
        assert fc.name == "DATE_TRUNC"
        assert len(fc.args) == 2

    def test_date_add(self):
        node = parse_sql("SELECT DATE_ADD(dt, 7) FROM t", dialect="databricks")
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)
        assert fc.name == "DATE_ADD"

    def test_datediff(self):
        node = parse_sql("SELECT DATEDIFF(a, b) FROM t", dialect="databricks")
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)
        assert fc.name == "DATEDIFF"

    def test_date_format(self):
        node = parse_sql(
            "SELECT DATE_FORMAT(ts, 'yyyy-MM-dd') FROM t",
            dialect="databricks",
        )
        fc = node.projections[0]
        assert fc.name == "DATE_FORMAT"

    def test_to_date(self):
        node = parse_sql("SELECT TO_DATE(s, 'yyyy-MM-dd') FROM t",
                          dialect="databricks")
        assert isinstance(node.projections[0], FunctionCall)

    def test_to_timestamp(self):
        node = parse_sql("SELECT TO_TIMESTAMP(s) FROM t", dialect="databricks")
        assert isinstance(node.projections[0], FunctionCall)

    def test_current_date(self):
        node = parse_sql("SELECT CURRENT_DATE() FROM t")
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)
        assert fc.name == "CURRENT_DATE"

    def test_current_timestamp(self):
        node = parse_sql("SELECT CURRENT_TIMESTAMP() FROM t")
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)


# ---------------------------------------------------------------------------
# Databricks string / null / math functions
# ---------------------------------------------------------------------------

class TestDatabricksStringFunctions:
    def test_concat(self):
        node = parse_sql("SELECT CONCAT(a, b) FROM t", dialect="databricks")
        assert isinstance(node.projections[0], FunctionCall)

    def test_substring(self):
        node = parse_sql("SELECT SUBSTRING(s, 1, 3) FROM t",
                          dialect="databricks")
        fc = node.projections[0]
        assert fc.name == "SUBSTRING"
        assert len(fc.args) == 3

    def test_upper_lower(self):
        for f in ("UPPER", "LOWER"):
            node = parse_sql(f"SELECT {f}(name) FROM t", dialect="databricks")
            assert node.projections[0].name == f

    def test_coalesce(self):
        node = parse_sql("SELECT COALESCE(a, b, 0) FROM t",
                          dialect="databricks")
        fc = node.projections[0]
        assert fc.name == "COALESCE"

    def test_nvl(self):
        node = parse_sql("SELECT NVL(a, 0) FROM t", dialect="databricks")
        assert node.projections[0].name == "NVL"

    def test_abs_round(self):
        for f in ("ABS", "ROUND", "CEIL", "FLOOR"):
            node = parse_sql(f"SELECT {f}(val) FROM t", dialect="databricks")
            assert node.projections[0].name == f

    def test_regexp_replace(self):
        node = parse_sql(
            "SELECT REGEXP_REPLACE(s, '[0-9]+', 'X') FROM t",
            dialect="databricks",
        )
        assert node.projections[0].name == "REGEXP_REPLACE"

    def test_split(self):
        node = parse_sql("SELECT SPLIT(s, ',') FROM t", dialect="databricks")
        assert node.projections[0].name == "SPLIT"


# ---------------------------------------------------------------------------
# LATERAL VIEW / EXPLODE
# ---------------------------------------------------------------------------

class TestLateralView:
    def test_lateral_view_explode(self):
        node = parse_sql(
            "SELECT id, val FROM t "
            "LATERAL VIEW EXPLODE(arr) vals AS val",
            dialect="databricks",
        )
        assert node.lateral_views is not None
        assert len(node.lateral_views) == 1
        lv = node.lateral_views[0]
        assert lv.table_alias == "vals"
        assert lv.column_aliases == ["val"]
        assert isinstance(lv.function, FunctionCall)
        assert lv.function.name == "EXPLODE"

    def test_lateral_view_outer(self):
        node = parse_sql(
            "SELECT * FROM t LATERAL VIEW OUTER EXPLODE(arr) vals AS val",
            dialect="databricks",
        )
        assert len(node.lateral_views) == 1


# ---------------------------------------------------------------------------
# CASE WHEN
# ---------------------------------------------------------------------------

class TestCaseWhen:
    def test_searched_case(self):
        node = parse_sql(
            "SELECT CASE WHEN a > 0 THEN 'pos' WHEN a < 0 THEN 'neg' "
            "ELSE 'zero' END FROM t"
        )
        cw = node.projections[0]
        assert isinstance(cw, CaseWhen)
        assert cw.operand is None
        assert len(cw.branches) == 2
        assert isinstance(cw.else_expr, Literal)

    def test_simple_case(self):
        node = parse_sql(
            "SELECT CASE status WHEN 1 THEN 'a' WHEN 2 THEN 'b' END FROM t"
        )
        cw = node.projections[0]
        assert isinstance(cw, CaseWhen)
        assert cw.operand is not None


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

class TestWindowFunctions:
    def test_row_number(self):
        node = parse_sql(
            "SELECT ROW_NUMBER() OVER (PARTITION BY region ORDER BY id) FROM t",
            dialect="databricks",
        )
        wf = node.projections[0]
        assert isinstance(wf, WindowFunction)
        assert isinstance(wf.function, FunctionCall)
        assert wf.function.name == "ROW_NUMBER"
        win = wf.window
        assert isinstance(win, WindowSpec)
        assert len(win.partition_by) == 1
        assert len(win.order_by) == 1

    def test_rank(self):
        node = parse_sql(
            "SELECT RANK() OVER (ORDER BY score DESC) FROM t",
            dialect="databricks",
        )
        wf = node.projections[0]
        assert isinstance(wf, WindowFunction)
        assert wf.function.name == "RANK"
        assert wf.window.order_by[0].ascending is False

    def test_lag_lead(self):
        for fn in ("LAG", "LEAD"):
            node = parse_sql(
                f"SELECT {fn}(val, 1) OVER (ORDER BY ts) FROM t",
                dialect="databricks",
            )
            wf = node.projections[0]
            assert isinstance(wf, WindowFunction)
            assert wf.function.name == fn

    def test_window_frame(self):
        node = parse_sql(
            "SELECT SUM(val) OVER ("
            "PARTITION BY id ORDER BY ts "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            ") FROM t",
            dialect="databricks",
        )
        wf = node.projections[0]
        assert isinstance(wf, WindowFunction)
        win = wf.window
        assert win.frame_start == "UNBOUNDED PRECEDING"
        assert win.frame_end == "CURRENT ROW"


# ---------------------------------------------------------------------------
# TRY_CAST (Databricks)
# ---------------------------------------------------------------------------

class TestTryCast:
    def test_try_cast(self):
        node = parse_sql("SELECT TRY_CAST(x AS INT) FROM t",
                          dialect="databricks")
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)
        assert fc.name == "TRY_CAST"


# ---------------------------------------------------------------------------
# INSERT
# ---------------------------------------------------------------------------

class TestInsert:
    def test_insert_select(self):
        node = parse_sql("INSERT INTO target SELECT * FROM source")
        assert isinstance(node, InsertNode)
        assert node.target.name == "target"
        assert isinstance(node.source, SelectNode)

    def test_insert_values(self):
        node = parse_sql("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
        assert isinstance(node, InsertNode)
        assert len(node.values) == 2

    def test_insert_columns(self):
        node = parse_sql("INSERT INTO t (id, name) SELECT id, name FROM s")
        assert node.columns == ["id", "name"]


# ---------------------------------------------------------------------------
# MERGE
# ---------------------------------------------------------------------------

class TestMerge:
    def test_basic_merge(self):
        node = parse_sql(
            "MERGE INTO target t USING source s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET t.name = s.name "
            "WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)"
        )
        assert isinstance(node, MergeNode)
        assert node.target.name == "target"
        assert isinstance(node.on, Comparison)
        assert node.when_matched is not None
        assert node.when_not_matched is not None


# ---------------------------------------------------------------------------
# SQL round-trip (parse → emit)
# ---------------------------------------------------------------------------

class TestSQLRoundTrip:
    def _rt(self, sql: str, dialect: str = "databricks") -> str:
        node = parse_sql(sql, dialect=dialect)
        return node.to_sql(dialect=dialect)

    def test_simple_select(self):
        result = self._rt("SELECT a, b FROM t WHERE id > 10")
        assert "SELECT" in result
        assert "FROM" in result
        assert "WHERE" in result

    def test_select_limit(self):
        result = self._rt("SELECT * FROM t LIMIT 10")
        assert "LIMIT 10" in result

    def test_join(self):
        result = self._rt("SELECT * FROM a INNER JOIN b ON a.id = b.id")
        assert "JOIN" in result

    def test_group_by(self):
        result = self._rt("SELECT region, COUNT(*) FROM t GROUP BY region")
        assert "GROUP BY" in result

    def test_order_by(self):
        result = self._rt("SELECT * FROM t ORDER BY a DESC")
        assert "ORDER BY" in result
        assert "DESC" in result

    def test_union(self):
        result = self._rt("SELECT a FROM t1 UNION ALL SELECT a FROM t2")
        assert "UNION ALL" in result

    def test_cte(self):
        result = self._rt(
            "WITH cte AS (SELECT a FROM t) SELECT * FROM cte"
        )
        assert "WITH" in result


# ---------------------------------------------------------------------------
# PlanNode.from_sql convenience
# ---------------------------------------------------------------------------

class TestPlanNodeFromSQL:
    def test_from_sql(self):
        node = PlanNode.from_sql("SELECT 1")
        assert isinstance(node, SelectNode)

    def test_from_sql_databricks(self):
        node = PlanNode.from_sql(
            "SELECT DATE_TRUNC('month', ts) FROM t",
            dialect="databricks",
        )
        assert isinstance(node, SelectNode)
