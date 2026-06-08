"""Tests for :class:`yggdrasil.saga.Saga` — the unified data engine."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.saga import ExecutionPlan, LazyTabular, Saga, SelectPlan
from yggdrasil.saga.plan.nodes import SelectNode


@pytest.fixture
def users() -> ArrowTabular:
    return ArrowTabular(pa.table({
        "id": [1, 2, 3, 4, 5],
        "name": ["alice", "bob", "carol", "dave", "eve"],
        "region": ["US", "EU", "US", "EU", "US"],
        "score": [90, 80, 95, 70, 85],
    }))


@pytest.fixture
def saga(users: ArrowTabular) -> Saga:
    return Saga(dialect="databricks").register("users", users)


def test_register_is_upsert_and_chainable(users: ArrowTabular) -> None:
    s = Saga()
    assert s.register("t", users) is s          # chainable
    assert "t" in s
    other = ArrowTabular(pa.table({"x": [1]}))
    s.register("t", other)                       # upsert by name
    assert s["t"] is other


def test_sql_end_to_end(saga: Saga) -> None:
    res = saga.sql("SELECT name, score FROM users WHERE score > 80 ORDER BY score DESC")
    assert res.read_arrow_table().to_pylist() == [
        {"name": "carol", "score": 95},
        {"name": "alice", "score": 90},
        {"name": "eve", "score": 85},
    ]


def test_sql_extra_tables_dont_mutate_catalog(saga: Saga) -> None:
    extra = ArrowTabular(pa.table({"id": [1, 2], "tag": ["a", "b"]}))
    res = saga.sql("SELECT id, tag FROM tags ORDER BY id", tables={"tags": extra})
    assert res.read_arrow_table().to_pylist() == [{"id": 1, "tag": "a"}, {"id": 2, "tag": "b"}]
    assert "tags" not in saga          # one-off table did not leak into the catalog


def test_parse_returns_node_and_uses_default_dialect(saga: Saga) -> None:
    node = saga.parse("SELECT * FROM users WHERE region = 'US'")
    assert isinstance(node, SelectNode)
    assert "users" in saga.to_sql(node)


def test_parse_default_on_failure(saga: Saga) -> None:
    assert saga.parse("NOT VALID SQL", default=None) is None


def test_plan_returns_execution_plan(saga: Saga) -> None:
    plan = saga.plan("SELECT name FROM users LIMIT 2")
    assert isinstance(plan, (ExecutionPlan, SelectPlan))


def test_scan_builds_lazy_pipeline(saga: Saga) -> None:
    out = (saga.scan("users")
               .filter("region = 'US'")
               .select("name", "score")
               .limit(2)
               .read_arrow_table())
    rows = out.to_pylist()
    assert isinstance(saga.scan("users"), LazyTabular)
    assert len(rows) == 2
    assert {r["name"] for r in rows} <= {"alice", "carol", "eve"}


def test_scan_accepts_tabular_directly(users: ArrowTabular) -> None:
    lazy = Saga().scan(users)
    assert isinstance(lazy, LazyTabular)
    assert lazy.select("name").read_arrow_table().column_names == ["name"]


def test_execute_node_and_plan(saga: Saga) -> None:
    node = saga.parse("SELECT id FROM users WHERE id = 3")
    assert saga.execute(node).read_arrow_table().to_pylist() == [{"id": 3}]


def test_table_miss_raises(saga: Saga) -> None:
    with pytest.raises(KeyError):
        saga.table("missing")


def test_with_dialect_shares_catalog(saga: Saga) -> None:
    pg = saga.with_dialect("postgres")
    assert pg.dialect == "postgres"
    assert "users" in pg                 # catalog shared
    assert pg.tables is saga.tables


def test_unregister(saga: Saga) -> None:
    saga.unregister("users")
    assert "users" not in saga
    saga.unregister("nope")              # missing_ok by default
    with pytest.raises(KeyError):
        saga.unregister("nope", missing_ok=False)
