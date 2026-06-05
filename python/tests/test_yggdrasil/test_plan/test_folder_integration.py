"""Integration tests — execute plans against Folder-backed Tabulars.

Tests plan execution with disk-backed data (Folder + Parquet) to
verify the full pipeline: SQL parsing → plan node → execution →
Arrow materialization from disk.
"""

from __future__ import annotations

import shutil
import tempfile

import pyarrow as pa
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.path.folder import Folder
from yggdrasil.path.local_path import LocalPath
from yggdrasil.plan import (
    LazyTabular,
    SelectPlan,
    parse_sql,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_folder(path: str, table: pa.Table) -> Folder:
    lp = LocalPath(path)
    folder = Folder(path=lp)
    folder.write_table(table)
    return folder


@pytest.fixture
def users_folder(tmpdir):
    return _make_folder(f"{tmpdir}/users", pa.table({
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "name": ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi"],
        "region": ["US", "EU", "US", "EU", "US", "EU", "US", "EU"],
        "score": [90, 80, 95, 70, 85, 60, 75, 55],
    }))


@pytest.fixture
def orders_folder(tmpdir):
    return _make_folder(f"{tmpdir}/orders", pa.table({
        "id": [1, 2, 1, 3, 5, 2, 4, 6],
        "order_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "amount": [10.0, 20.0, 15.0, 30.0, 25.0, 12.0, 18.0, 22.0],
    }))


# ---------------------------------------------------------------------------
# Basic Folder reads via plan
# ---------------------------------------------------------------------------

class TestFolderBasicExecution:
    def test_select_star(self, users_folder):
        node = parse_sql("SELECT * FROM users")
        result = node.execute(tables={"users": users_folder})
        assert result.read_arrow_table().num_rows == 8

    def test_select_columns(self, users_folder):
        node = parse_sql("SELECT id, name FROM users")
        result = node.execute(tables={"users": users_folder})
        table = result.read_arrow_table()
        assert table.column_names == ["id", "name"]
        assert table.num_rows == 8

    def test_filter(self, users_folder):
        node = parse_sql("SELECT * FROM users WHERE score > 80")
        result = node.execute(tables={"users": users_folder})
        table = result.read_arrow_table()
        assert all(s > 80 for s in table.column("score").to_pylist())

    def test_order_by_and_limit(self, users_folder):
        node = parse_sql(
            "SELECT * FROM users ORDER BY score DESC LIMIT 3"
        )
        result = node.execute(tables={"users": users_folder})
        table = result.read_arrow_table()
        assert table.num_rows == 3
        scores = table.column("score").to_pylist()
        assert scores == [95, 90, 85]

    def test_distinct(self, users_folder):
        node = parse_sql("SELECT DISTINCT region FROM users")
        result = node.execute(tables={"users": users_folder})
        regions = result.read_arrow_table().column("region").to_pylist()
        assert set(regions) == {"US", "EU"}


# ---------------------------------------------------------------------------
# Aggregation on Folder
# ---------------------------------------------------------------------------

class TestFolderAggregation:
    def test_count_by_region(self, users_folder):
        node = parse_sql(
            "SELECT region, COUNT(*) AS cnt FROM users GROUP BY region"
        )
        result = node.execute(tables={"users": users_folder})
        table = result.read_arrow_table()
        assert table.num_rows == 2
        total = sum(table.column("cnt").to_pylist())
        assert total == 8

    def test_sum_by_region(self, users_folder):
        node = parse_sql(
            "SELECT region, SUM(score) AS total FROM users GROUP BY region"
        )
        result = node.execute(tables={"users": users_folder})
        table = result.read_arrow_table()
        for row in table.to_pylist():
            if row["region"] == "US":
                assert row["total"] == 90 + 95 + 85 + 75
            else:
                assert row["total"] == 80 + 70 + 60 + 55

    def test_avg_score(self, users_folder):
        node = parse_sql("SELECT AVG(score) AS avg_score FROM users")
        result = node.execute(tables={"users": users_folder})
        table = result.read_arrow_table()
        avg = table.column("avg_score").to_pylist()[0]
        expected = (90 + 80 + 95 + 70 + 85 + 60 + 75 + 55) / 8
        assert abs(avg - expected) < 0.01


# ---------------------------------------------------------------------------
# Join Folder with Folder
# ---------------------------------------------------------------------------

class TestFolderJoin:
    def test_inner_join(self, users_folder, orders_folder):
        node = parse_sql(
            "SELECT users.name, orders.amount "
            "FROM users INNER JOIN orders ON users.id = orders.id"
        )
        result = node.execute(tables={
            "users": users_folder,
            "orders": orders_folder,
        })
        table = result.read_arrow_table()
        assert table.num_rows > 0
        assert "name" in table.column_names
        assert "amount" in table.column_names

    def test_join_with_filter(self, users_folder, orders_folder):
        node = parse_sql(
            "SELECT users.name, orders.amount "
            "FROM users INNER JOIN orders ON users.id = orders.id "
            "WHERE orders.amount > 20"
        )
        result = node.execute(tables={
            "users": users_folder,
            "orders": orders_folder,
        })
        table = result.read_arrow_table()
        assert all(a > 20 for a in table.column("amount").to_pylist())


# ---------------------------------------------------------------------------
# CTE with Folder
# ---------------------------------------------------------------------------

class TestFolderCTE:
    def test_cte_with_folder(self, users_folder):
        node = parse_sql(
            "WITH top_scorers AS ("
            "  SELECT * FROM users WHERE score >= 80"
            ") "
            "SELECT name, score FROM top_scorers ORDER BY score DESC"
        )
        result = node.execute(tables={"users": users_folder})
        table = result.read_arrow_table()
        scores = table.column("score").to_pylist()
        assert all(s >= 80 for s in scores)
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# LazyTabular on Folder
# ---------------------------------------------------------------------------

class TestFolderLazy:
    def test_lazy_select_filter(self, users_folder):
        lazy = users_folder.lazy()
        result = lazy.select("id", "name").filter("score > 80").read_arrow_table()
        assert result.column_names == ["id", "name"]
        assert result.num_rows > 0

    def test_lazy_chain(self, users_folder):
        result = (users_folder.lazy()
                  .filter("region = 'US'")
                  .select("name", "score")
                  .limit(2)
                  .read_arrow_table())
        assert result.num_rows <= 2
        assert result.column_names == ["name", "score"]


# ---------------------------------------------------------------------------
# Multiple writes and reads
# ---------------------------------------------------------------------------

class TestFolderMultiWrite:
    def test_append_and_read(self, tmpdir):
        folder = _make_folder(f"{tmpdir}/append_test", pa.table({
            "id": [1, 2, 3],
            "val": ["a", "b", "c"],
        }))
        folder.write_table(
            pa.table({"id": [4, 5], "val": ["d", "e"]}),
            options={"mode": "append"},
        )
        node = parse_sql("SELECT * FROM data ORDER BY id")
        result = node.execute(tables={"data": folder})
        table = result.read_arrow_table()
        assert table.num_rows == 5
        assert table.column("id").to_pylist() == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestFunctionRegistry:
    def test_builtin_count(self):
        from yggdrasil.plan import BUILTIN_REGISTRY
        assert len(BUILTIN_REGISTRY) > 250

    def test_known_functions(self):
        from yggdrasil.plan import BUILTIN_REGISTRY
        for name in ["COUNT", "SUM", "AVG", "MIN", "MAX",
                      "DATE_TRUNC", "UPPER", "LOWER", "COALESCE",
                      "ROW_NUMBER", "LAG", "LEAD", "EXPLODE",
                      "ABS", "ROUND", "HASH", "UUID",
                      "CURRENT_DATE", "CURRENT_TIMESTAMP",
                      "EXTRACT", "INTERVAL"]:
            assert BUILTIN_REGISTRY.is_known(name), f"{name} not in registry"

    def test_register_udf(self):
        from yggdrasil.plan import BUILTIN_REGISTRY
        reg = BUILTIN_REGISTRY.copy()
        meta = reg.register("MY_UDF", category="udf", min_args=1, max_args=3)
        assert reg.is_known("MY_UDF")
        assert meta.category == "udf"
        assert meta.min_args == 1
        assert meta.max_args == 3

    def test_categories(self):
        from yggdrasil.plan import BUILTIN_REGISTRY
        categories = {m.category for m in BUILTIN_REGISTRY._functions.values()}
        assert "aggregate" in categories
        assert "datetime" in categories
        assert "string" in categories
        assert "window" in categories
        assert "math" in categories
        assert "collection" in categories

    def test_function_metadata(self):
        from yggdrasil.plan import BUILTIN_REGISTRY
        meta = BUILTIN_REGISTRY.get("DATE_TRUNC")
        assert meta is not None
        assert meta.category == "datetime"
        assert meta.min_args == 2
        assert meta.max_args == 2
        assert meta.deterministic is True
