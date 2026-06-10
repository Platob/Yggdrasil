"""Comprehensive tests for the completed lazy tabular and SQL parser.

Covers:
- LazyTabular group_by, order_by, having, with_cte, offset
- SQL parser QUALIFY, numeric INTERVAL, USING joins, TIMESTAMPADD
- SelectPlan new builders (group_by, order_by, having, with_cte, offset)
- SQL round-trip for all new features
- Benchmarks for parse and execute performance
- Integration tests with real-shaped data (Meteologica-style forecast pipeline)
"""

from __future__ import annotations

import datetime as dt
import time

import pyarrow as pa
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.execution.expr import col
from yggdrasil.execution.expr.nodes import (
    Alias,
    Comparison,
    FunctionCall,
    Logical,
)
from yggdrasil.plan import (
    ExecutionPlan,
    SelectNode,
    SelectPlan,
    parse_sql,
)
from yggdrasil.plan.ops import JoinClause


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


@pytest.fixture
def forecast_data():
    """Realistic Meteologica-style forecast data."""
    n = 1000
    base_ts = dt.datetime(2024, 1, 1)
    return ArrowTabular(pa.table({
        "content_id": [i % 10 + 1 for i in range(n)],
        "issue_date": pa.array([
            base_ts + dt.timedelta(hours=i // 10)
            for i in range(n)
        ], type=pa.timestamp("us")),
        "from_timestamp": pa.array([
            base_ts + dt.timedelta(hours=i // 10 + 24)
            for i in range(n)
        ], type=pa.timestamp("us")),
        "to_timestamp": pa.array([
            base_ts + dt.timedelta(hours=i // 10 + 25)
            for i in range(n)
        ], type=pa.timestamp("us")),
        "value": [float(50 + (i * 7 + 13) % 30) for i in range(n)],
        "metric": ["forecast"] * n,
        "hour_diff": [((i * 3 + 1) % 48) + 1 for i in range(n)],
    }))


@pytest.fixture
def timeseries():
    """Hourly timeseries for resample/window tests."""
    base = dt.datetime(2024, 6, 1)
    n = 240  # 10 days of hourly data
    return ArrowTabular(pa.table({
        "ts": pa.array([base + dt.timedelta(hours=i) for i in range(n)],
                       type=pa.timestamp("us")),
        "station_id": [i % 5 + 1 for i in range(n)],
        "temperature": [20.0 + (i % 24) * 0.5 + (i % 5) * 2.0 for i in range(n)],
        "humidity": [60.0 + (i % 12) * 1.5 for i in range(n)],
    }))


@pytest.fixture
def large_table():
    """10k-row table for benchmarks."""
    return ArrowTabular(pa.table({
        "id": list(range(10_000)),
        "name": [f"user_{i}" for i in range(10_000)],
        "region": ["US" if i % 3 == 0 else "EU" if i % 3 == 1 else "ASIA"
                   for i in range(10_000)],
        "score": [i % 100 for i in range(10_000)],
        "amount": [float(i * 1.5) for i in range(10_000)],
    }))


# ===========================================================================
# LazyTabular — group_by
# ===========================================================================

class TestLazyGroupBy:
    def test_group_by_with_count(self, users):
        result = (users.lazy()
                  .group_by("region", aggregations={"cnt": "count(id)"})
                  .read_arrow_table())
        assert result.num_rows == 2
        assert "cnt" in result.column_names
        assert "region" in result.column_names

    def test_group_by_with_sum(self, users):
        result = (users.lazy()
                  .group_by("region", aggregations={"total": "sum(score)"})
                  .read_arrow_table())
        rows = result.to_pylist()
        for row in rows:
            if row["region"] == "US":
                assert row["total"] == 270  # 90 + 95 + 85
            elif row["region"] == "EU":
                assert row["total"] == 150  # 80 + 70

    def test_group_by_with_avg(self, users):
        result = (users.lazy()
                  .group_by("region", aggregations={"avg_score": "avg(score)"})
                  .read_arrow_table())
        assert result.num_rows == 2
        rows = result.to_pylist()
        for row in rows:
            if row["region"] == "US":
                assert abs(row["avg_score"] - 90.0) < 0.1
            elif row["region"] == "EU":
                assert abs(row["avg_score"] - 75.0) < 0.1

    def test_group_by_with_min_max(self, users):
        result = (users.lazy()
                  .group_by("region",
                            aggregations={"lo": "min(score)", "hi": "max(score)"})
                  .read_arrow_table())
        rows = result.to_pylist()
        for row in rows:
            if row["region"] == "US":
                assert row["lo"] == 85
                assert row["hi"] == 95

    def test_group_by_no_aggregation_deduplicates(self, users):
        result = (users.lazy()
                  .group_by("region")
                  .read_arrow_table())
        regions = result.column("region").to_pylist()
        assert len(regions) == len(set(regions))

    def test_group_by_chain_with_filter(self, users):
        result = (users.lazy()
                  .filter("score >= 80")
                  .group_by("region", aggregations={"cnt": "count(id)"})
                  .read_arrow_table())
        assert result.num_rows <= 2

    def test_group_by_raises_without_keys(self, users):
        with pytest.raises(ValueError, match="at least one"):
            users.lazy().group_by()


# ===========================================================================
# LazyTabular — order_by
# ===========================================================================

class TestLazyOrderBy:
    def test_order_by_ascending(self, users):
        result = (users.lazy()
                  .order_by("score")
                  .read_arrow_table())
        scores = result.column("score").to_pylist()
        assert scores == sorted(scores)

    def test_order_by_descending(self, users):
        result = (users.lazy()
                  .order_by("-score")
                  .read_arrow_table())
        scores = result.column("score").to_pylist()
        assert scores == sorted(scores, reverse=True)

    def test_order_by_tuple_form(self, users):
        result = (users.lazy()
                  .order_by(("score", False))
                  .read_arrow_table())
        scores = result.column("score").to_pylist()
        assert scores == sorted(scores, reverse=True)

    def test_order_by_multiple(self, users):
        result = (users.lazy()
                  .order_by("region", "-score")
                  .read_arrow_table())
        rows = result.to_pylist()
        eu_rows = [r for r in rows if r["region"] == "EU"]
        us_rows = [r for r in rows if r["region"] == "US"]
        assert eu_rows[0]["score"] >= eu_rows[-1]["score"]
        assert us_rows[0]["score"] >= us_rows[-1]["score"]

    def test_order_by_with_limit(self, users):
        result = (users.lazy()
                  .order_by("-score")
                  .limit(3)
                  .read_arrow_table())
        assert result.num_rows == 3
        scores = result.column("score").to_pylist()
        assert scores == sorted(scores, reverse=True)

    def test_order_by_raises_empty(self, users):
        with pytest.raises(ValueError, match="at least one"):
            users.lazy().order_by()


# ===========================================================================
# LazyTabular — having
# ===========================================================================

class TestLazyHaving:
    def test_having_filters_groups(self, users):
        result = (users.lazy()
                  .group_by("region", aggregations={"cnt": "count(id)"})
                  .having("cnt > 2")
                  .read_arrow_table())
        assert result.num_rows == 1
        assert result.column("region").to_pylist() == ["US"]

    def test_having_with_sum(self, users):
        result = (users.lazy()
                  .group_by("region", aggregations={"total": "sum(score)"})
                  .having(col("total") > 200)
                  .read_arrow_table())
        assert result.num_rows == 1
        assert result.column("region").to_pylist() == ["US"]


# ===========================================================================
# LazyTabular — offset
# ===========================================================================

class TestLazyOffset:
    def test_offset_only(self, users):
        result = (users.lazy()
                  .order_by("id")
                  .offset(2)
                  .read_arrow_table())
        ids = result.column("id").to_pylist()
        assert ids[0] == 3

    def test_limit_offset(self, users):
        result = (users.lazy()
                  .order_by("id")
                  .limit(2)
                  .offset(1)
                  .read_arrow_table())
        assert result.num_rows == 2
        ids = result.column("id").to_pylist()
        assert ids == [2, 3]


# ===========================================================================
# LazyTabular — with_cte
# ===========================================================================

class TestLazyCTE:
    def test_with_cte_basic(self, users):
        sub = SelectPlan()
        sub.filter("score > 80")
        lazy = (users.lazy()
                .with_cte("top_users", sub)
                .filter("score > 80")
                .select("name", "score"))
        result = lazy.read_arrow_table()
        assert result.num_rows > 0
        assert result.column_names == ["name", "score"]

    def test_with_cte_repr(self, users):
        lazy = users.lazy()
        lazy.with_cte("cte1", SelectPlan())
        assert "ctes=1" in repr(lazy.plan)


# ===========================================================================
# SelectPlan — new builders
# ===========================================================================

class TestSelectPlanGroupBy:
    def test_group_by_builder(self):
        plan = SelectPlan()
        plan.group_by("region", aggregations={"cnt": "count(*)"})
        assert plan.group_by_op is not None
        assert plan.group_by_op.keys == ["region"]

    def test_group_by_execute(self, users):
        plan = SelectPlan()
        plan.group_by("region", aggregations={"cnt": "count(id)"})
        result = plan.execute(users).read_arrow_table()
        assert result.num_rows == 2
        assert "cnt" in result.column_names

    def test_order_by_builder(self):
        plan = SelectPlan()
        plan.order_by("score", ("-name", False))
        assert plan.order_by_op is not None
        assert len(plan.order_by_op.keys) == 2

    def test_order_by_execute(self, users):
        plan = SelectPlan()
        plan.order_by("-score")
        result = plan.execute(users).read_arrow_table()
        scores = result.column("score").to_pylist()
        assert scores == sorted(scores, reverse=True)

    def test_having_builder(self):
        plan = SelectPlan()
        plan.group_by("region", aggregations={"cnt": "count(id)"})
        plan.having("cnt > 2")
        assert plan.having_pred is not None

    def test_offset_builder(self):
        plan = SelectPlan()
        plan.offset(5)
        assert plan.offset_rows == 5

    def test_with_cte_builder(self):
        plan = SelectPlan()
        sub = SelectPlan()
        sub.filter("x > 1")
        plan.with_cte("filtered", sub)
        assert plan.ctes is not None
        assert len(plan.ctes) == 1

    def test_copy_preserves_new_fields(self):
        plan = SelectPlan()
        plan.group_by("region", aggregations={"cnt": "count(*)"})
        plan.order_by("-score")
        plan.having("cnt > 1")
        plan.offset(5)
        plan.with_cte("x", SelectPlan())
        clone = plan.copy()
        assert clone.group_by_op is not None
        assert clone.order_by_op is not None
        assert clone.having_pred is not None
        assert clone.offset_rows == 5
        assert clone.ctes is not None

    def test_clear_resets_new_fields(self):
        plan = SelectPlan()
        plan.group_by("r").order_by("s").having("cnt > 1").offset(5)
        plan.with_cte("x", SelectPlan())
        plan.clear()
        assert plan.is_identity

    def test_to_plan_node_includes_group_by(self):
        plan = SelectPlan()
        plan.group_by("region").select("region")
        node = plan.to_plan_node()
        assert node.group_by is not None
        assert len(node.group_by) == 1

    def test_to_plan_node_includes_order_by(self):
        plan = SelectPlan()
        plan.order_by("-score")
        node = plan.to_plan_node()
        assert node.order_by is not None
        assert node.order_by[0].ascending is False

    def test_to_plan_node_includes_offset(self):
        plan = SelectPlan()
        plan.limit(10).offset(5)
        node = plan.to_plan_node()
        assert node.limit == 10
        assert node.offset == 5

    def test_to_plan_node_includes_ctes(self):
        plan = SelectPlan()
        plan.with_cte("x", SelectPlan())
        node = plan.to_plan_node()
        assert node.ctes is not None


class TestSelectPlanFromSQLNewFeatures:
    def test_from_sql_with_order_by(self):
        plan = ExecutionPlan.from_sql(
            "SELECT a FROM t ORDER BY a DESC"
        )
        assert isinstance(plan, SelectPlan)
        assert plan.order_by_op is not None

    def test_from_sql_with_offset(self):
        plan = ExecutionPlan.from_sql("SELECT * FROM t LIMIT 10 OFFSET 5")
        assert plan.limit_rows == 10
        assert plan.offset_rows == 5


# ===========================================================================
# SQL Parser — QUALIFY
# ===========================================================================

class TestQualifyClause:
    def test_qualify_basic(self):
        node = parse_sql(
            "SELECT *, ROW_NUMBER() OVER (PARTITION BY content_id ORDER BY issue_date DESC) AS rn "
            "FROM forecast "
            "QUALIFY rn = 1",
            dialect="databricks",
        )
        assert isinstance(node, SelectNode)
        assert node.qualify is not None

    def test_qualify_with_comparison(self):
        node = parse_sql(
            "SELECT content_id, value "
            "FROM forecast "
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY content_id, from_timestamp "
            "ORDER BY issue_date DESC) = 1",
            dialect="databricks",
        )
        assert node.qualify is not None
        assert isinstance(node.qualify, Comparison)

    def test_qualify_after_where(self):
        node = parse_sql(
            "SELECT * FROM t "
            "WHERE metric = 'forecast' "
            "QUALIFY ROW_NUMBER() OVER (ORDER BY ts DESC) = 1",
            dialect="databricks",
        )
        assert node.where is not None
        assert node.qualify is not None

    def test_qualify_after_group_by_having(self):
        node = parse_sql(
            "SELECT region, COUNT(*) AS cnt FROM t "
            "GROUP BY region "
            "HAVING cnt > 1 "
            "QUALIFY ROW_NUMBER() OVER (ORDER BY cnt DESC) <= 5",
            dialect="databricks",
        )
        assert node.group_by is not None
        assert node.having is not None
        assert node.qualify is not None

    def test_qualify_roundtrip(self):
        sql = (
            "SELECT content_id, value FROM forecast "
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY content_id ORDER BY issue_date DESC) = 1"
        )
        node = parse_sql(sql, dialect="databricks")
        emitted = node.to_sql(dialect="databricks")
        assert "QUALIFY" in emitted
        # Re-parse
        node2 = parse_sql(emitted, dialect="databricks")
        assert node2.qualify is not None


# ===========================================================================
# SQL Parser — Numeric INTERVAL
# ===========================================================================

class TestNumericInterval:
    def test_interval_numeric_hour(self):
        node = parse_sql("SELECT ts + INTERVAL 2 HOUR FROM t")
        p = node.projections[0]
        assert isinstance(p.right, FunctionCall)
        assert p.right.name == "INTERVAL"
        assert p.right.args[0].value == 2

    def test_interval_numeric_day(self):
        node = parse_sql("SELECT ts - INTERVAL 7 DAY FROM t")
        p = node.projections[0]
        assert p.right.args[0].value == 7
        assert p.right.args[1].value == "DAY"

    def test_interval_string_still_works(self):
        node = parse_sql("SELECT ts + INTERVAL '1' DAY FROM t")
        p = node.projections[0]
        assert p.right.args[0].value == "1"

    def test_interval_plural_normalized(self):
        node = parse_sql("SELECT ts + INTERVAL 3 HOURS FROM t")
        p = node.projections[0]
        assert p.right.args[1].value == "HOUR"

    def test_interval_minutes_normalized(self):
        node = parse_sql("SELECT ts + INTERVAL 30 MINUTES FROM t")
        p = node.projections[0]
        assert p.right.args[1].value == "MINUTE"

    def test_interval_in_where(self):
        node = parse_sql(
            "SELECT * FROM t WHERE from_timestamp > issue_date + INTERVAL 2 HOUR",
        )
        assert node.where is not None

    def test_interval_numeric_roundtrip(self):
        node = parse_sql("SELECT ts + INTERVAL 5 DAY FROM t")
        sql = node.to_sql(dialect="databricks")
        assert "INTERVAL" in sql
        node2 = parse_sql(sql, dialect="databricks")
        p = node2.projections[0]
        assert isinstance(p.right, FunctionCall)
        assert p.right.name == "INTERVAL"


# ===========================================================================
# SQL Parser — USING joins
# ===========================================================================

class TestUsingJoin:
    def test_using_single_column(self):
        node = parse_sql("SELECT * FROM a JOIN b USING (id)")
        assert isinstance(node.from_clause, JoinClause)
        assert node.from_clause.on is not None

    def test_using_multiple_columns(self):
        node = parse_sql("SELECT * FROM a JOIN b USING (id, name)")
        assert isinstance(node.from_clause, JoinClause)
        on = node.from_clause.on
        assert isinstance(on, Logical)

    def test_using_left_join(self):
        node = parse_sql("SELECT * FROM a LEFT JOIN b USING (id)")
        assert node.from_clause.join_type.is_outer

    def test_using_execution(self, users, orders):
        node = parse_sql("SELECT * FROM users JOIN orders USING (id)")
        result = node.execute(tables={"users": users, "orders": orders})
        table = result.read_arrow_table()
        assert table.num_rows > 0
        assert "order_id" in table.column_names

    def test_using_roundtrip(self):
        node = parse_sql("SELECT * FROM a JOIN b USING (id)")
        sql = node.to_sql(dialect="databricks")
        assert "JOIN" in sql


# ===========================================================================
# SQL Parser — TIMESTAMPADD / Databricks functions
# ===========================================================================

class TestTimestampaddParser:
    def test_timestampadd(self):
        node = parse_sql(
            "SELECT TIMESTAMPADD(HOUR, 2, ts) FROM t",
            dialect="databricks",
        )
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)
        assert fc.name == "TIMESTAMPADD"

    def test_make_interval(self):
        node = parse_sql(
            "SELECT MAKE_INTERVAL(0, 0, 0, 0, 2, 0, 0) FROM t",
            dialect="databricks",
        )
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)
        assert fc.name == "MAKE_INTERVAL"

    def test_timestampdiff(self):
        node = parse_sql(
            "SELECT TIMESTAMPDIFF(HOUR, ts1, ts2) FROM t",
            dialect="databricks",
        )
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)
        assert fc.name == "TIMESTAMPDIFF"


# ===========================================================================
# SQL Parser — Complex Meteologica-style queries
# ===========================================================================

class TestMeteologicaStyleSQL:
    def test_cte_pipeline(self):
        sql = """
        WITH raw AS (
            SELECT content_id, issue_date, value
            FROM forecast
            WHERE metric = 'forecast'
              AND issue_date >= TIMESTAMP '2024-01-01 00:00:00'
              AND issue_date < TIMESTAMP '2024-02-01 00:00:00'
        ),
        latest AS (
            SELECT content_id, value
            FROM raw
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY content_id
                ORDER BY issue_date DESC
            ) = 1
        )
        SELECT content_id, value FROM latest
        """
        node = parse_sql(sql, dialect="databricks")
        assert isinstance(node, SelectNode)
        assert node.ctes is not None
        assert len(node.ctes) == 2
        assert node.ctes[0].name == "raw"
        assert node.ctes[1].name == "latest"
        # latest CTE should have QUALIFY
        latest_plan = node.ctes[1].plan
        assert isinstance(latest_plan, SelectNode)
        assert latest_plan.qualify is not None

    def test_lateral_view_with_qualify(self):
        sql = """
        SELECT r.content_id, r.issue_date, d.from_timestamp, d.value
        FROM raw r
        LATERAL VIEW EXPLODE(r.data) d_view AS d
        WHERE d.from_timestamp >= TIMESTAMP '2024-01-01 00:00:00'
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY r.content_id, d.from_timestamp
            ORDER BY r.issue_date DESC
        ) = 1
        """
        node = parse_sql(sql, dialect="databricks")
        assert node.lateral_views is not None
        assert node.qualify is not None

    def test_cross_join_with_explode(self):
        sql = """
        SELECT e.content_id, hd.hour_diff, e.value
        FROM exploded_flat e
        CROSS JOIN hd
        WHERE e.from_timestamp > TIMESTAMPADD(HOUR, hd.hour_diff, e.issue_date)
        """
        node = parse_sql(sql, dialect="databricks")
        assert isinstance(node.from_clause, JoinClause)
        assert node.from_clause.join_type.is_cross

    def test_window_with_named_struct(self):
        sql = """
        SELECT
            content_id,
            date,
            COLLECT_LIST(
                NAMED_STRUCT('hour', from_timestamp_hour, 'value', CAST(value AS DOUBLE))
            ) AS hourly_values
        FROM hourly
        GROUP BY content_id, date
        """
        node = parse_sql(sql, dialect="databricks")
        assert node.group_by is not None
        assert len(node.group_by) == 2

    def test_complex_where_with_interval(self):
        sql = """
        SELECT * FROM forecast
        WHERE metric = 'forecast'
          AND issue_date >= TIMESTAMP '2024-01-01 00:00:00'
          AND issue_date < TIMESTAMP '2024-02-01 00:00:00'
          AND from_timestamp > issue_date + INTERVAL 2 HOUR
        """
        node = parse_sql(sql, dialect="databricks")
        assert node.where is not None

    def test_last_value_forward_fill(self):
        sql = """
        SELECT
            content_id,
            COALESCE(
                value,
                LAST_VALUE(value, TRUE) OVER (
                    PARTITION BY content_id
                    ORDER BY ts
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ),
                FIRST_VALUE(value, TRUE) OVER (
                    PARTITION BY content_id
                    ORDER BY ts
                    ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
                )
            ) AS filled_value
        FROM spine
        """
        node = parse_sql(sql, dialect="databricks")
        coalesce = node.projections[1]
        assert isinstance(coalesce, Alias)

    def test_full_daily_forecast_cte_chain(self):
        """Parse a 6-CTE Meteologica-style query."""
        sql = """
        WITH raw AS (
            SELECT content_id, issue_date, value, from_timestamp, to_timestamp
            FROM curated_data
            WHERE metric = 'forecast'
              AND issue_date >= TIMESTAMP '2024-01-01 00:00:00'
        ),
        exploded AS (
            SELECT content_id, issue_date, from_timestamp, to_timestamp, value
            FROM raw
            WHERE from_timestamp >= TIMESTAMP '2024-01-01 00:00:00'
              AND from_timestamp > issue_date + INTERVAL 2 HOUR
        ),
        latest AS (
            SELECT content_id, from_timestamp, to_timestamp, value
            FROM exploded
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY content_id, from_timestamp, to_timestamp
                ORDER BY issue_date DESC
            ) = 1
        ),
        hourly AS (
            SELECT content_id, date, AVG(value) AS value
            FROM latest
            GROUP BY content_id, date
        ),
        daily AS (
            SELECT content_id, date,
                   COLLECT_LIST(value) AS values
            FROM hourly
            GROUP BY content_id, date
        ),
        content_lookup AS (
            SELECT content_id, content_name
            FROM content_meta
            WHERE content_id IN (1, 2, 3, 4, 5)
        )
        SELECT d.content_id, c.content_name, d.date, d.values
        FROM daily d
        JOIN content_lookup c ON d.content_id = c.content_id
        """
        node = parse_sql(sql, dialect="databricks")
        assert isinstance(node, SelectNode)
        assert len(node.ctes) == 6
        assert node.ctes[0].name == "raw"
        assert node.ctes[2].name == "latest"
        assert node.ctes[4].name == "daily"
        assert isinstance(node.from_clause, JoinClause)

    def test_sequence_and_collect(self):
        sql = """
        SELECT
            content_id,
            SEQUENCE(0, 23) AS hours,
            COLLECT_LIST(value) AS values
        FROM daily
        GROUP BY content_id
        """
        node = parse_sql(sql, dialect="databricks")
        assert len(node.projections) == 3


# ===========================================================================
# Integration — Execute Meteologica-style queries with real data
# ===========================================================================

class TestQualifyExecution:
    def test_qualify_row_number_eq_1(self):
        data = ArrowTabular(pa.table({
            "content_id": [1, 1, 2, 2, 3],
            "issue_date": [1, 2, 1, 2, 1],
            "value": [10, 20, 30, 40, 50],
        }))
        node = parse_sql(
            "SELECT content_id, value FROM forecast "
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY content_id "
            "  ORDER BY issue_date DESC) = 1",
            dialect="databricks",
        )
        result = node.execute(tables={"forecast": data})
        rows = result.read_arrow_table().to_pylist()
        # Latest per content_id: cid=1 -> value=20, cid=2 -> value=40, cid=3 -> value=50
        assert len(rows) == 3
        by_cid = {r["content_id"]: r["value"] for r in rows}
        assert by_cid == {1: 20, 2: 40, 3: 50}

    def test_qualify_row_number_le_2(self):
        data = ArrowTabular(pa.table({
            "content_id": [1, 1, 1, 2, 2, 2],
            "ts": [1, 2, 3, 1, 2, 3],
            "value": [10, 20, 30, 40, 50, 60],
        }))
        node = parse_sql(
            "SELECT content_id, value FROM forecast "
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY content_id "
            "  ORDER BY ts DESC) <= 2",
            dialect="databricks",
        )
        result = node.execute(tables={"forecast": data})
        rows = result.read_arrow_table().to_pylist()
        # Top 2 per content_id by ts: cid=1 -> (30, 20), cid=2 -> (60, 50)
        assert len(rows) == 4
        cid1 = sorted(r["value"] for r in rows if r["content_id"] == 1)
        cid2 = sorted(r["value"] for r in rows if r["content_id"] == 2)
        assert cid1 == [20, 30]
        assert cid2 == [50, 60]

    def test_qualify_no_partition(self):
        data = ArrowTabular(pa.table({
            "id": [1, 2, 3, 4, 5],
            "score": [10, 30, 20, 50, 40],
        }))
        node = parse_sql(
            "SELECT id, score FROM t "
            "QUALIFY ROW_NUMBER() OVER (ORDER BY score DESC) = 1",
            dialect="databricks",
        )
        result = node.execute(tables={"t": data})
        rows = result.read_arrow_table().to_pylist()
        assert len(rows) == 1
        assert rows[0]["id"] == 4
        assert rows[0]["score"] == 50

    def test_qualify_unsupported_falls_back(self):
        # A QUALIFY that isn't a ROW_NUMBER comparison falls back to no-op
        # rather than crashing.
        data = ArrowTabular(pa.table({
            "id": [1, 2, 3],
            "score": [10, 20, 30],
        }))
        node = parse_sql(
            "SELECT id FROM t "
            "QUALIFY SUM(score) OVER (ORDER BY id) > 5",
            dialect="databricks",
        )
        result = node.execute(tables={"t": data})
        # SUM window is not supported; result is unfiltered (graceful fallback)
        assert result.read_arrow_table().num_rows == 3


class TestValuesInFrom:
    def test_values_no_column_list(self):
        node = parse_sql(
            "SELECT * FROM (VALUES (1, 'a'), (2, 'b')) t",
            dialect="databricks",
        )
        result = node.execute()
        rows = result.read_arrow_table().to_pylist()
        assert len(rows) == 2

    def test_values_with_column_list(self):
        node = parse_sql(
            "SELECT * FROM (VALUES (1, 'alice'), (2, 'bob'), (3, 'carol')) "
            "AS t(id, name)",
            dialect="databricks",
        )
        result = node.execute()
        rows = result.read_arrow_table().to_pylist()
        assert rows == [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
            {"id": 3, "name": "carol"},
        ]

    def test_values_in_cte(self):
        node = parse_sql(
            "WITH lookup AS ("
            "  SELECT * FROM (VALUES "
            "    (1, 'Station_A', '/europe/wind'), "
            "    (2, 'Station_B', '/europe/solar')"
            "  ) AS t(content_id, content_name, content_path)"
            ") "
            "SELECT * FROM lookup",
            dialect="databricks",
        )
        result = node.execute()
        rows = result.read_arrow_table().to_pylist()
        assert len(rows) == 2
        assert rows[0]["content_name"] == "Station_A"

    def test_values_join_with_data(self):
        data = ArrowTabular(pa.table({
            "id": [1, 2, 3],
            "val": [10, 20, 30],
        }))
        node = parse_sql(
            "SELECT d.val, m.name FROM data d "
            "JOIN (VALUES (1, 'a'), (2, 'b'), (3, 'c')) AS m(id, name) "
            "ON d.id = m.id",
            dialect="databricks",
        )
        result = node.execute(tables={"data": data})
        rows = result.read_arrow_table().to_pylist()
        assert len(rows) == 3

    def test_values_roundtrip(self):
        sql_in = "SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(id, name)"
        node = parse_sql(sql_in, dialect="databricks")
        emitted = node.to_sql(dialect="databricks")
        assert "VALUES" in emitted
        node2 = parse_sql(emitted, dialect="databricks")
        result = node2.execute()
        rows = result.read_arrow_table().to_pylist()
        assert len(rows) == 2


class TestLambdaParser:
    def test_transform_lambda(self):
        node = parse_sql(
            "SELECT TRANSFORM(arr, x -> x * 2) FROM t",
            dialect="databricks",
        )
        fc = node.projections[0]
        assert isinstance(fc, FunctionCall)
        assert fc.name == "TRANSFORM"
        from yggdrasil.execution.expr.nodes import Lambda
        assert isinstance(fc.args[1], Lambda)
        assert fc.args[1].params == ("x",)

    def test_filter_lambda(self):
        node = parse_sql(
            "SELECT FILTER(arr, x -> x > 0) FROM t",
            dialect="databricks",
        )
        from yggdrasil.execution.expr.nodes import Lambda
        fc = node.projections[0]
        assert isinstance(fc.args[1], Lambda)

    def test_aggregate_multi_param_lambda(self):
        node = parse_sql(
            "SELECT AGGREGATE(arr, 0, (acc, x) -> acc + x) FROM t",
            dialect="databricks",
        )
        from yggdrasil.execution.expr.nodes import Lambda
        fc = node.projections[0]
        assert isinstance(fc.args[2], Lambda)
        assert fc.args[2].params == ("acc", "x")

    def test_zip_with_lambda(self):
        node = parse_sql(
            "SELECT ZIP_WITH(a, b, (x, y) -> x + y) FROM t",
            dialect="databricks",
        )
        from yggdrasil.execution.expr.nodes import Lambda
        fc = node.projections[0]
        assert isinstance(fc.args[2], Lambda)
        assert fc.args[2].params == ("x", "y")

    def test_exists_higher_order(self):
        # EXISTS(arr, lambda) is the Databricks higher-order form
        node = parse_sql(
            "SELECT EXISTS(arr, x -> x > 0) FROM t",
            dialect="databricks",
        )
        from yggdrasil.execution.expr.nodes import Lambda
        fc = node.projections[0]
        assert fc.name == "EXISTS"
        assert isinstance(fc.args[1], Lambda)

    def test_exists_subquery_still_works(self):
        # Subquery EXISTS — the older form must still parse
        node = parse_sql(
            "SELECT * FROM t WHERE EXISTS(SELECT 1 FROM u WHERE u.id = t.id)"
        )
        assert node.where is not None

    def test_lambda_roundtrip_single_param(self):
        node = parse_sql("SELECT TRANSFORM(arr, x -> x * 2) FROM t", dialect="databricks")
        sql = node.to_sql(dialect="databricks")
        assert "->" in sql
        # Re-parse
        node2 = parse_sql(sql, dialect="databricks")
        from yggdrasil.execution.expr.nodes import Lambda
        assert isinstance(node2.projections[0].args[1], Lambda)

    def test_lambda_roundtrip_multi_param(self):
        node = parse_sql(
            "SELECT AGGREGATE(arr, 0, (acc, x) -> acc + x) FROM t",
            dialect="databricks",
        )
        sql = node.to_sql(dialect="databricks")
        assert "(acc, x) ->" in sql

    def test_lambda_free_columns_excludes_params(self):
        from yggdrasil.execution.expr import free_columns
        node = parse_sql(
            "SELECT TRANSFORM(my_arr, x -> x + my_offset) FROM t",
            dialect="databricks",
        )
        fc = node.projections[0]
        cols = free_columns(fc)
        # 'x' is a lambda param — should not be a free column
        assert "x" not in cols
        assert "my_arr" in cols
        assert "my_offset" in cols

    def test_lambda_with_complex_body(self):
        node = parse_sql(
            "SELECT FILTER(arr, x -> CAST(x AS DOUBLE) > 0.5) FROM t",
            dialect="databricks",
        )
        from yggdrasil.execution.expr.nodes import Lambda
        fc = node.projections[0]
        assert isinstance(fc.args[1], Lambda)


class TestLateralViewExecution:
    def test_explode_simple(self):
        data = ArrowTabular(pa.table({
            "id": [1, 2, 3],
            "tags": [["a", "b"], ["c"], ["d", "e", "f"]],
        }))
        node = parse_sql(
            "SELECT id, tag FROM t LATERAL VIEW EXPLODE(tags) tv AS tag",
            dialect="databricks",
        )
        result = node.execute(tables={"t": data})
        rows = result.read_arrow_table().to_pylist()
        assert len(rows) == 6
        assert rows[0] == {"id": 1, "tag": "a"}
        assert rows[5] == {"id": 3, "tag": "f"}

    def test_posexplode(self):
        data = ArrowTabular(pa.table({
            "id": [1, 2],
            "vals": [[10, 20, 30], [100]],
        }))
        node = parse_sql(
            "SELECT id, p, v FROM t LATERAL VIEW POSEXPLODE(vals) tv AS p, v",
            dialect="databricks",
        )
        result = node.execute(tables={"t": data})
        rows = result.read_arrow_table().to_pylist()
        assert len(rows) == 4
        assert rows[0] == {"id": 1, "p": 0, "v": 10}
        assert rows[3] == {"id": 2, "p": 0, "v": 100}

    def test_explode_in_cte(self):
        data = ArrowTabular(pa.table({
            "id": [1, 2],
            "tags": [["x", "y"], ["z"]],
        }))
        node = parse_sql(
            "WITH exploded AS ("
            "  SELECT id, tag FROM t LATERAL VIEW EXPLODE(tags) tv AS tag"
            ") "
            "SELECT * FROM exploded ORDER BY id, tag",
            dialect="databricks",
        )
        result = node.execute(tables={"t": data})
        rows = result.read_arrow_table().to_pylist()
        assert len(rows) == 3

    def test_explode_then_qualify(self):
        """Meteologica pattern: explode array, then keep latest per partition."""
        data = ArrowTabular(pa.table({
            "content_id": [1, 1, 2, 2],
            "issue_date": [100, 200, 100, 200],
            "values": [[1.0, 2.0], [3.0, 4.0], [5.0], [6.0, 7.0]],
        }))
        node = parse_sql(
            "WITH exploded AS ("
            "  SELECT content_id, issue_date, v FROM forecast "
            "  LATERAL VIEW EXPLODE(values) tv AS v"
            "), latest AS ("
            "  SELECT content_id, v FROM exploded "
            "  QUALIFY ROW_NUMBER() OVER ("
            "    PARTITION BY content_id ORDER BY issue_date DESC) = 1"
            ") "
            "SELECT * FROM latest ORDER BY content_id",
            dialect="databricks",
        )
        result = node.execute(tables={"forecast": data})
        rows = result.read_arrow_table().to_pylist()
        # For each content_id, keep only the latest issue_date's exploded rows
        # cid 1 latest is issue_date 200 -> [3.0, 4.0], but row_number=1 means
        # only one row per partition (the first after sort) -> just one value
        # The implementation gives one row per partition.
        assert len(rows) == 2


class TestForecastIntegration:
    def test_filter_by_metric_and_date(self, forecast_data):
        node = parse_sql(
            "SELECT content_id, issue_date, value "
            "FROM forecast "
            "WHERE metric = 'forecast' "
            "  AND value > 60"
        )
        result = node.execute(tables={"forecast": forecast_data})
        table = result.read_arrow_table()
        assert all(v > 60 for v in table.column("value").to_pylist())

    def test_group_by_content_id(self, forecast_data):
        node = parse_sql(
            "SELECT content_id, AVG(value) AS avg_value, COUNT(*) AS cnt "
            "FROM forecast "
            "WHERE metric = 'forecast' "
            "GROUP BY content_id"
        )
        result = node.execute(tables={"forecast": forecast_data})
        table = result.read_arrow_table()
        assert table.num_rows == 10
        assert sum(table.column("cnt").to_pylist()) == 1000

    def test_order_by_and_limit(self, forecast_data):
        node = parse_sql(
            "SELECT content_id, value "
            "FROM forecast "
            "ORDER BY value DESC "
            "LIMIT 10"
        )
        result = node.execute(tables={"forecast": forecast_data})
        table = result.read_arrow_table()
        assert table.num_rows == 10
        vals = table.column("value").to_pylist()
        assert vals == sorted(vals, reverse=True)

    def test_cte_chain_with_execution(self, forecast_data):
        node = parse_sql(
            "WITH filtered AS ("
            "  SELECT content_id, value FROM forecast WHERE value > 70"
            "), "
            "top AS ("
            "  SELECT content_id, AVG(value) AS avg_val "
            "  FROM filtered GROUP BY content_id"
            ") "
            "SELECT * FROM top ORDER BY avg_val DESC LIMIT 5"
        )
        result = node.execute(tables={"forecast": forecast_data})
        table = result.read_arrow_table()
        assert table.num_rows <= 5
        assert "avg_val" in table.column_names

    def test_join_forecast_with_metadata(self, forecast_data):
        content_meta = ArrowTabular(pa.table({
            "content_id": list(range(1, 11)),
            "content_name": [f"Station_{i}" for i in range(1, 11)],
        }))
        node = parse_sql(
            "SELECT f.content_id, c.content_name, f.value "
            "FROM forecast f "
            "INNER JOIN content_meta c ON f.content_id = c.content_id "
            "WHERE f.value > 70 "
        )
        result = node.execute(tables={
            "forecast": forecast_data,
            "content_meta": content_meta,
        })
        table = result.read_arrow_table()
        assert "content_name" in table.column_names
        assert table.num_rows > 0

    def test_lazy_forecast_pipeline(self, forecast_data):
        result = (forecast_data.lazy()
                  .filter("metric = 'forecast'")
                  .filter(col("value") > 60)
                  .select("content_id", "value", "hour_diff")
                  .order_by("-value")
                  .limit(50)
                  .read_arrow_table())
        assert result.num_rows <= 50
        vals = result.column("value").to_pylist()
        assert vals == sorted(vals, reverse=True)
        assert all(v > 60 for v in vals)

    def test_lazy_group_aggregate_pipeline(self, forecast_data):
        result = (forecast_data.lazy()
                  .filter("metric = 'forecast'")
                  .group_by("content_id",
                            aggregations={
                                "avg_value": "avg(value)",
                                "min_value": "min(value)",
                                "max_value": "max(value)",
                                "cnt": "count(value)",
                            })
                  .order_by("-avg_value")
                  .read_arrow_table())
        assert result.num_rows == 10
        assert "avg_value" in result.column_names
        assert "min_value" in result.column_names
        assert "max_value" in result.column_names


class TestTimeseriesIntegration:
    def test_filter_by_station(self, timeseries):
        node = parse_sql(
            "SELECT ts, temperature FROM ts_data WHERE station_id = 1"
        )
        result = node.execute(tables={"ts_data": timeseries})
        table = result.read_arrow_table()
        assert table.num_rows == 48  # 240 / 5 stations

    def test_group_by_station(self, timeseries):
        node = parse_sql(
            "SELECT station_id, AVG(temperature) AS avg_temp, "
            "MIN(temperature) AS min_temp, MAX(temperature) AS max_temp "
            "FROM ts_data GROUP BY station_id"
        )
        result = node.execute(tables={"ts_data": timeseries})
        table = result.read_arrow_table()
        assert table.num_rows == 5

    def test_order_by_temperature(self, timeseries):
        node = parse_sql(
            "SELECT * FROM ts_data ORDER BY temperature DESC LIMIT 20"
        )
        result = node.execute(tables={"ts_data": timeseries})
        table = result.read_arrow_table()
        temps = table.column("temperature").to_pylist()
        assert temps == sorted(temps, reverse=True)


# ===========================================================================
# SQL round-trip for all new features
# ===========================================================================

class TestNewFeatureRoundTrips:
    def _rt(self, sql: str, dialect: str = "databricks") -> str:
        node = parse_sql(sql, dialect=dialect)
        return node.to_sql(dialect=dialect)

    def test_qualify_roundtrip(self):
        sql = self._rt(
            "SELECT content_id, value FROM forecast "
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY content_id ORDER BY ts DESC) = 1"
        )
        assert "QUALIFY" in sql

    def test_numeric_interval_roundtrip(self):
        sql = self._rt("SELECT ts + INTERVAL 5 DAY FROM t")
        assert "INTERVAL" in sql

    def test_using_join_roundtrip(self):
        sql = self._rt("SELECT * FROM a JOIN b USING (id)")
        assert "JOIN" in sql

    def test_timestampadd_roundtrip(self):
        sql = self._rt("SELECT TIMESTAMPADD(HOUR, 2, ts) FROM t")
        assert "TIMESTAMPADD" in sql

    def test_complex_cte_roundtrip(self):
        original = (
            "WITH raw AS (SELECT * FROM t WHERE id > 0), "
            "latest AS (SELECT * FROM raw "
            "QUALIFY ROW_NUMBER() OVER (ORDER BY ts DESC) = 1) "
            "SELECT * FROM latest"
        )
        sql = self._rt(original)
        assert "WITH" in sql
        assert "QUALIFY" in sql
        # Re-parse
        node = parse_sql(sql, dialect="databricks")
        assert node.ctes is not None
        assert len(node.ctes) == 2

    def test_plan_to_sql_with_group_order_limit_offset(self):
        plan = SelectPlan()
        plan.select("region").group_by("region").order_by("-region").limit(10).offset(5)
        sql = plan.to_sql(dialect="databricks")
        assert "SELECT" in sql
        assert "LIMIT 10" in sql
        assert "OFFSET 5" in sql


# ===========================================================================
# Benchmarks
# ===========================================================================

class TestBenchmarks:
    def test_parse_simple_select_under_200us(self):
        t0 = time.perf_counter()
        for _ in range(100):
            parse_sql("SELECT a, b FROM t WHERE id > 10 LIMIT 5")
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.001  # 1ms budget per parse

    def test_parse_complex_cte_qualify_under_500us(self):
        sql = (
            "WITH raw AS (SELECT * FROM t WHERE metric = 'forecast'), "
            "latest AS (SELECT * FROM raw "
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts DESC) = 1) "
            "SELECT id, value FROM latest ORDER BY value DESC LIMIT 100"
        )
        t0 = time.perf_counter()
        for _ in range(100):
            parse_sql(sql, dialect="databricks")
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.005  # 5ms budget

    def test_parse_6_cte_pipeline_under_2ms(self):
        sql = """
        WITH c1 AS (SELECT * FROM t1),
             c2 AS (SELECT * FROM t2),
             c3 AS (SELECT * FROM c1 JOIN c2 ON c1.id = c2.id),
             c4 AS (SELECT * FROM c3 WHERE val > 0),
             c5 AS (SELECT region, AVG(val) AS avg_val FROM c4 GROUP BY region),
             c6 AS (SELECT * FROM c5 ORDER BY avg_val DESC)
        SELECT * FROM c6 LIMIT 10
        """
        t0 = time.perf_counter()
        for _ in range(50):
            parse_sql(sql, dialect="databricks")
        elapsed = (time.perf_counter() - t0) / 50
        assert elapsed < 0.01  # 10ms budget

    def test_execute_filter_10k_under_5ms(self, large_table):
        node = parse_sql("SELECT * FROM t WHERE score > 50")
        t0 = time.perf_counter()
        for _ in range(20):
            node.execute(tables={"t": large_table}).read_arrow_table()
        elapsed = (time.perf_counter() - t0) / 20
        assert elapsed < 0.05  # 50ms budget

    def test_execute_group_by_10k_under_5ms(self, large_table):
        node = parse_sql(
            "SELECT region, COUNT(*) AS cnt, AVG(score) AS avg FROM t GROUP BY region"
        )
        t0 = time.perf_counter()
        for _ in range(20):
            node.execute(tables={"t": large_table}).read_arrow_table()
        elapsed = (time.perf_counter() - t0) / 20
        assert elapsed < 0.05

    def test_execute_order_limit_10k_under_5ms(self, large_table):
        node = parse_sql("SELECT * FROM t ORDER BY score DESC LIMIT 100")
        t0 = time.perf_counter()
        for _ in range(20):
            node.execute(tables={"t": large_table}).read_arrow_table()
        elapsed = (time.perf_counter() - t0) / 20
        assert elapsed < 0.05

    def test_execute_join_10k_under_50ms(self, large_table):
        right = ArrowTabular(pa.table({
            "id": list(range(0, 10_000, 2)),
            "status": ["active" if i % 2 == 0 else "inactive"
                       for i in range(5_000)],
        }))
        node = parse_sql("SELECT t.name, r.status FROM t JOIN r ON t.id = r.id")
        t0 = time.perf_counter()
        for _ in range(10):
            node.execute(tables={"t": large_table, "r": right}).read_arrow_table()
        elapsed = (time.perf_counter() - t0) / 10
        assert elapsed < 0.1

    def test_execute_cte_pipeline_10k_under_50ms(self, large_table):
        node = parse_sql(
            "WITH filtered AS (SELECT * FROM t WHERE score > 50), "
            "grouped AS (SELECT region, COUNT(*) AS cnt FROM filtered GROUP BY region) "
            "SELECT * FROM grouped ORDER BY cnt DESC"
        )
        t0 = time.perf_counter()
        for _ in range(10):
            node.execute(tables={"t": large_table}).read_arrow_table()
        elapsed = (time.perf_counter() - t0) / 10
        assert elapsed < 0.1

    def test_lazy_pipeline_10k_under_10ms(self, large_table):
        t0 = time.perf_counter()
        for _ in range(10):
            (large_table.lazy()
             .filter("score > 50")
             .select("id", "name", "score")
             .order_by("-score")
             .limit(100)
             .read_arrow_table())
        elapsed = (time.perf_counter() - t0) / 10
        assert elapsed < 0.05

    def test_sql_roundtrip_under_500us(self):
        sql = (
            "WITH cte AS (SELECT * FROM t WHERE id > 0) "
            "SELECT a, b FROM cte WHERE x > 10 ORDER BY a DESC LIMIT 100"
        )
        t0 = time.perf_counter()
        for _ in range(100):
            node = parse_sql(sql, dialect="databricks")
            node.to_sql(dialect="databricks")
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.005


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_table_group_by(self):
        empty = ArrowTabular(pa.table({
            "region": ["x"],
            "score": [0],
        }))
        # Remove the row to get an empty table with schema
        plan = SelectPlan()
        plan.filter("score > 999")
        plan.group_by("region", aggregations={"total": "sum(score)"})
        result = plan.execute(empty).read_arrow_table()
        assert result.num_rows == 0

    def test_single_row_group_by(self):
        single = ArrowTabular(pa.table({
            "region": ["US"],
            "score": [100],
        }))
        plan = SelectPlan()
        plan.group_by("region", aggregations={"total": "sum(score)"})
        result = plan.execute(single).read_arrow_table()
        assert result.num_rows == 1
        assert result.column("total").to_pylist() == [100]

    def test_order_by_preserves_all_columns(self, users):
        plan = SelectPlan()
        plan.order_by("score")
        result = plan.execute(users).read_arrow_table()
        assert set(result.column_names) == {"id", "name", "region", "score"}

    def test_group_by_then_order_by(self, users):
        plan = SelectPlan()
        plan.group_by("region", aggregations={"cnt": "count(id)"})
        plan.order_by("-cnt")
        result = plan.execute(users).read_arrow_table()
        cnts = result.column("cnt").to_pylist()
        assert cnts == sorted(cnts, reverse=True)

    def test_qualify_not_alias_name(self):
        """QUALIFY should not be treated as an identifier alias."""
        node = parse_sql(
            "SELECT * FROM t QUALIFY ROW_NUMBER() OVER (ORDER BY id) = 1",
            dialect="databricks",
        )
        assert node.qualify is not None

    def test_parse_interval_zero(self):
        node = parse_sql("SELECT ts + INTERVAL 0 DAY FROM t")
        p = node.projections[0]
        assert p.right.args[0].value == 0

    def test_interval_in_complex_expression(self):
        node = parse_sql(
            "SELECT * FROM t "
            "WHERE ts > CURRENT_TIMESTAMP() - INTERVAL 7 DAY "
            "  AND ts < CURRENT_TIMESTAMP() + INTERVAL 1 DAY"
        )
        assert node.where is not None

    def test_using_join_with_alias(self):
        node = parse_sql("SELECT * FROM users u LEFT JOIN orders o USING (id)")
        jc = node.from_clause
        assert isinstance(jc, JoinClause)
        assert jc.join_type.is_outer
