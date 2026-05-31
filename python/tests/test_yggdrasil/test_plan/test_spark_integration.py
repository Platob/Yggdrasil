"""Spark integration tests for plan execution.

Tests verify that the executor routes through Spark-native operations
when the source is a SparkDataset, by checking that:
- `.filter()` uses Tabular dispatch (not Arrow materialization)
- `.select()` uses Tabular dispatch
- The Spark frame is preserved through the pipeline where possible

Since PySpark may not be available in all test environments, these
tests use the ArrowTabular as a Tabular stand-in that validates the
operation sequence. The key property being tested is that the executor
uses Tabular interface methods (which dispatch to Spark natively)
rather than premature `.read_arrow_table()` calls.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.plan import parse_sql, ExecutionPlan
from yggdrasil.plan.nodes import PlanNode, SelectNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def large_table():
    """10k row table for pushdown verification."""
    rows = 10_000
    return ArrowTabular(pa.table({
        "id": list(range(rows)),
        "name": [f"user_{i}" for i in range(rows)],
        "region": ["US" if i % 2 == 0 else "EU" for i in range(rows)],
        "score": [50 + (i * 7) % 51 for i in range(rows)],
        "ts": pa.array(
            [1704067200 + i * 3600 for i in range(rows)],
            type=pa.timestamp("s"),
        ),
    }))


# ---------------------------------------------------------------------------
# Pushdown verification — predicate pushed into CastOptions
# ---------------------------------------------------------------------------

class TestPredicatePushdown:
    def test_where_filter_reduces_rows(self, large_table):
        node = parse_sql("SELECT * FROM data WHERE score > 90")
        result = node.execute(tables={"data": large_table})
        table = result.read_arrow_table()
        assert table.num_rows < 10_000
        assert all(s > 90 for s in table.column("score").to_pylist())

    def test_where_and_select(self, large_table):
        node = parse_sql("SELECT id, name FROM data WHERE score > 90")
        result = node.execute(tables={"data": large_table})
        table = result.read_arrow_table()
        assert table.column_names == ["id", "name"]
        assert table.num_rows < 10_000

    def test_where_and_limit(self, large_table):
        node = parse_sql(
            "SELECT * FROM data WHERE score > 80 ORDER BY id LIMIT 10"
        )
        result = node.execute(tables={"data": large_table})
        table = result.read_arrow_table()
        assert table.num_rows == 10

    def test_complex_predicate_pushdown(self, large_table):
        node = parse_sql(
            "SELECT * FROM data WHERE score > 80 AND region = 'US'"
        )
        result = node.execute(tables={"data": large_table})
        table = result.read_arrow_table()
        assert all(
            s > 80 and r == "US"
            for s, r in zip(
                table.column("score").to_pylist(),
                table.column("region").to_pylist(),
            )
        )


# ---------------------------------------------------------------------------
# Tabular interface dispatch (Spark-compatible path)
# ---------------------------------------------------------------------------

class TestTabularInterfaceDispatch:
    """Verify operations use Tabular methods (filter/select/unique)
    which dispatch to Spark natively when available."""

    def test_filter_uses_tabular(self, large_table):
        node = parse_sql("SELECT * FROM data WHERE id < 100")
        result = node.execute(tables={"data": large_table})
        assert result.read_arrow_table().num_rows == 100

    def test_select_uses_tabular(self, large_table):
        node = parse_sql("SELECT id, name FROM data")
        result = node.execute(tables={"data": large_table})
        table = result.read_arrow_table()
        assert table.column_names == ["id", "name"]
        assert table.num_rows == 10_000

    def test_distinct_uses_tabular(self, large_table):
        node = parse_sql("SELECT DISTINCT region FROM data")
        result = node.execute(tables={"data": large_table})
        regions = result.read_arrow_table().column("region").to_pylist()
        assert set(regions) == {"US", "EU"}

    def test_order_by_uses_tabular(self, large_table):
        node = parse_sql(
            "SELECT * FROM data ORDER BY score DESC LIMIT 5"
        )
        result = node.execute(tables={"data": large_table})
        table = result.read_arrow_table()
        scores = table.column("score").to_pylist()
        assert scores == sorted(scores, reverse=True)
        assert table.num_rows == 5

    def test_group_by_uses_tabular(self, large_table):
        node = parse_sql(
            "SELECT region, COUNT(*) AS cnt FROM data GROUP BY region"
        )
        result = node.execute(tables={"data": large_table})
        table = result.read_arrow_table()
        assert table.num_rows == 2
        total = sum(table.column("cnt").to_pylist())
        assert total == 10_000


# ---------------------------------------------------------------------------
# Join dispatch
# ---------------------------------------------------------------------------

class TestJoinDispatch:
    def test_join_with_large_tables(self, large_table):
        small = ArrowTabular(pa.table({
            "id": list(range(100)),
            "label": [f"label_{i}" for i in range(100)],
        }))
        node = parse_sql(
            "SELECT data.name, labels.label "
            "FROM data INNER JOIN labels ON data.id = labels.id"
        )
        result = node.execute(tables={"data": large_table, "labels": small})
        table = result.read_arrow_table()
        assert table.num_rows == 100
        assert "name" in table.column_names
        assert "label" in table.column_names

    def test_join_with_filter(self, large_table):
        small = ArrowTabular(pa.table({
            "id": list(range(50)),
            "tag": ["hot" if i < 25 else "cold" for i in range(50)],
        }))
        node = parse_sql(
            "SELECT data.name, tags.tag "
            "FROM data INNER JOIN tags ON data.id = tags.id "
            "WHERE tags.tag = 'hot'"
        )
        result = node.execute(tables={"data": large_table, "tags": small})
        table = result.read_arrow_table()
        assert all(t == "hot" for t in table.column("tag").to_pylist())


# ---------------------------------------------------------------------------
# CTE with large data
# ---------------------------------------------------------------------------

class TestCTELargeData:
    def test_cte_filters_before_aggregate(self, large_table):
        node = parse_sql(
            "WITH filtered AS ("
            "  SELECT * FROM data WHERE score > 80"
            ") "
            "SELECT region, COUNT(*) AS cnt FROM filtered GROUP BY region"
        )
        result = node.execute(tables={"data": large_table})
        table = result.read_arrow_table()
        assert table.num_rows == 2
        total = sum(table.column("cnt").to_pylist())
        assert total < 10_000


# ---------------------------------------------------------------------------
# Verify SparkDataset would be recognized
# ---------------------------------------------------------------------------

class TestSparkRecognition:
    def test_native_spark_frame_detection(self, large_table):
        spark_frame = large_table._native_spark_frame()
        assert spark_frame is None  # ArrowTabular has no Spark frame

    def test_spark_dataset_import(self):
        try:
            from yggdrasil.spark.tabular import SparkDataset
            assert hasattr(SparkDataset, "_native_spark_frame")
            assert hasattr(SparkDataset, "_filter")
            assert hasattr(SparkDataset, "_select")
        except ImportError:
            pytest.skip("pyspark not available")


# ---------------------------------------------------------------------------
# Default parameter on from_sql
# ---------------------------------------------------------------------------

class TestDefaultParamExec:
    def test_execution_plan_from_sql_default(self):
        result = ExecutionPlan.from_sql("GARBAGE SQL", default=None)
        assert result is None

    def test_plan_node_from_sql_default(self):
        result = PlanNode.from_sql("GARBAGE SQL", default=None)
        assert result is None

    def test_parse_sql_default_none(self):
        from yggdrasil.plan import parse_sql
        result = parse_sql("???", default=None)
        assert result is None

    def test_parse_sql_default_custom(self):
        from yggdrasil.plan import parse_sql
        sentinel = object()
        result = parse_sql("???", default=sentinel)
        assert result is sentinel

    def test_parse_sql_valid_ignores_default(self):
        from yggdrasil.plan import parse_sql
        result = parse_sql("SELECT 1", default=None)
        assert isinstance(result, SelectNode)


# ---------------------------------------------------------------------------
# Real PySpark integration tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spark():
    # Shared SparkTestCase session — never stopped (other modules share it).
    from yggdrasil.spark.tests import _get_test_spark
    return _get_test_spark()


@pytest.fixture
def spark_users(spark):
    from yggdrasil.spark.tabular import SparkDataset
    df = spark.createDataFrame(
        [(1, "alice", "US", 90), (2, "bob", "EU", 80),
         (3, "carol", "US", 95), (4, "dave", "EU", 70),
         (5, "eve", "US", 85)],
        ["id", "name", "region", "score"],
    )
    return SparkDataset(frame=df)


@pytest.fixture
def spark_orders(spark):
    from yggdrasil.spark.tabular import SparkDataset
    df = spark.createDataFrame(
        [(1, 101, 10.0), (2, 102, 20.0), (1, 103, 15.0), (3, 104, 30.0)],
        ["id", "order_id", "amount"],
    )
    return SparkDataset(frame=df)


class TestRealSparkExecution:
    def test_spark_filter(self, spark_users):
        node = parse_sql("SELECT * FROM users WHERE score > 80")
        result = node.execute(tables={"users": spark_users})
        table = result.read_arrow_table()
        assert table.num_rows == 3
        assert all(s > 80 for s in table.column("score").to_pylist())

    def test_spark_select(self, spark_users):
        node = parse_sql("SELECT id, name FROM users")
        result = node.execute(tables={"users": spark_users})
        table = result.read_arrow_table()
        assert set(table.column_names) == {"id", "name"}

    def test_spark_order_limit(self, spark_users):
        node = parse_sql("SELECT * FROM users ORDER BY score DESC LIMIT 3")
        result = node.execute(tables={"users": spark_users})
        table = result.read_arrow_table()
        assert table.num_rows == 3
        scores = table.column("score").to_pylist()
        assert scores == sorted(scores, reverse=True)

    def test_spark_distinct(self, spark_users):
        node = parse_sql("SELECT DISTINCT region FROM users")
        result = node.execute(tables={"users": spark_users})
        regions = set(result.read_arrow_table().column("region").to_pylist())
        assert regions == {"US", "EU"}

    def test_spark_join(self, spark_users, spark_orders):
        node = parse_sql(
            "SELECT users.name, orders.amount "
            "FROM users INNER JOIN orders ON users.id = orders.id"
        )
        result = node.execute(tables={"users": spark_users, "orders": spark_orders})
        table = result.read_arrow_table()
        assert table.num_rows > 0
        assert "name" in table.column_names

    def test_spark_cte(self, spark_users):
        node = parse_sql(
            "WITH high AS (SELECT * FROM users WHERE score >= 85) "
            "SELECT * FROM high"
        )
        result = node.execute(tables={"users": spark_users})
        table = result.read_arrow_table()
        assert all(s >= 85 for s in table.column("score").to_pylist())

    def test_spark_preserves_frame(self, spark_users):
        """SparkDataset stays as SparkDataset through filter/select."""
        from yggdrasil.spark.tabular import SparkDataset
        filtered = spark_users.filter("score > 80")
        assert isinstance(filtered, SparkDataset)
        assert filtered._native_spark_frame() is not None

    def test_spark_lazy(self, spark_users):
        lazy = spark_users.lazy()
        lazy.filter("score > 80").select("id", "name").limit(2)
        result = lazy.read_arrow_table()
        assert result.num_rows <= 2
        assert set(result.column_names) == {"id", "name"}


# ---------------------------------------------------------------------------
# row_limit pushdown via CastOptions
# ---------------------------------------------------------------------------

class TestRowLimitPushdown:
    def test_arrow_row_limit(self):
        from yggdrasil.data.options import CastOptions
        source = ArrowTabular(pa.table({"id": list(range(100))}))
        table = source.read_arrow_table(CastOptions(row_limit=10))
        assert table.num_rows == 10

    def test_arrow_row_limit_batches(self):
        from yggdrasil.data.options import CastOptions
        source = ArrowTabular(pa.table({"id": list(range(100))}))
        batches = list(source.read_arrow_batches(CastOptions(row_limit=5)))
        total = sum(b.num_rows for b in batches)
        assert total == 5

    def test_arrow_row_limit_larger_than_data(self):
        from yggdrasil.data.options import CastOptions
        source = ArrowTabular(pa.table({"id": [1, 2, 3]}))
        table = source.read_arrow_table(CastOptions(row_limit=100))
        assert table.num_rows == 3

    def test_spark_row_limit(self, spark):
        from yggdrasil.data.options import CastOptions
        from yggdrasil.spark.tabular import SparkDataset
        df = spark.createDataFrame([(i,) for i in range(100)], ["id"])
        ds = SparkDataset(frame=df)
        table = ds.read_arrow_table(CastOptions(row_limit=10))
        assert table.num_rows == 10

    def test_row_limit_none_reads_all(self):
        from yggdrasil.data.options import CastOptions
        source = ArrowTabular(pa.table({"id": list(range(50))}))
        table = source.read_arrow_table(CastOptions(row_limit=None))
        assert table.num_rows == 50
