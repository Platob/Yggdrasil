"""Tests for :class:`yggdrasil.saga.Saga` — the unified data engine.

Saga holds no catalog: it parses the ``FROM`` sources and live-builds them
(path/URL via IO, in-memory frames via :meth:`Tabular.new`), and owns a
local-disk staging session for spilling results.
"""

from __future__ import annotations

import pathlib

import pyarrow as pa
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.saga import ExecutionPlan, LazyTabular, Saga, SagaSession, SelectPlan
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
def users_parquet(users: ArrowTabular, tmp_path: pathlib.Path) -> str:
    import pyarrow.parquet as pq

    path = tmp_path / "users.parquet"
    pq.write_table(users.read_arrow_table(), str(path))
    return str(path)


# -- Parsing -------------------------------------------------------------

def test_parse_returns_node_and_uses_default_dialect() -> None:
    saga = Saga(dialect="databricks")
    node = saga.parse("SELECT * FROM t WHERE region = 'US'")
    assert isinstance(node, SelectNode)
    assert "t" in saga.to_sql(node)


def test_parse_default_on_failure() -> None:
    assert Saga().parse("NOT VALID SQL", default=None) is None


def test_plan_returns_execution_plan() -> None:
    plan = Saga().plan("SELECT name FROM t LIMIT 2")
    assert isinstance(plan, (ExecutionPlan, SelectPlan))


# -- Live FROM resolution + execution ------------------------------------

def test_sql_live_resolves_path_source(users_parquet: str) -> None:
    saga = Saga()
    res = saga.sql(f"SELECT name, score FROM '{users_parquet}' WHERE score > 80 ORDER BY score DESC")
    assert res.read_arrow_table().to_pylist() == [
        {"name": "carol", "score": 95},
        {"name": "alice", "score": 90},
        {"name": "eve", "score": 85},
    ]


def test_sql_ad_hoc_tables_not_stored(users: ArrowTabular) -> None:
    saga = Saga()
    res = saga.sql("SELECT id, name FROM users WHERE id = 3", tables={"users": users})
    assert res.read_arrow_table().to_pylist() == [{"id": 3, "name": "carol"}]
    # Nothing is stored on the engine — re-running without the table fails.
    with pytest.raises(Exception):
        saga.sql("SELECT * FROM users").read_arrow_table()


def test_execute_coerces_raw_frames_via_new() -> None:
    pl = pytest.importorskip("polars")
    saga = Saga()
    node = saga.parse("SELECT a FROM t WHERE a > 1")
    res = saga.execute(node, tables={"t": pl.DataFrame({"a": [1, 2, 3]})})
    assert res.read_arrow_table().to_pylist() == [{"a": 2}, {"a": 3}]


# -- Lazy scanning -------------------------------------------------------

def test_scan_accepts_tabular(users: ArrowTabular) -> None:
    out = (Saga().scan(users)
                 .filter("region = 'US'")
                 .select("name", "score")
                 .limit(2)
                 .read_arrow_table())
    assert isinstance(Saga().scan(users), LazyTabular)
    assert len(out.to_pylist()) == 2


def test_scan_accepts_raw_frame() -> None:
    pl = pytest.importorskip("polars")
    lazy = Saga().scan(pl.DataFrame({"name": ["a"], "score": [1]}))
    assert isinstance(lazy, LazyTabular)
    assert lazy.select("name").read_arrow_table().column_names == ["name"]


def test_scan_accepts_path(users_parquet: str) -> None:
    out = Saga().scan(users_parquet).select("name").read_arrow_table()
    assert out.column_names == ["name"]
    assert out.num_rows == 5


# -- Dialect -------------------------------------------------------------

def test_with_dialect_shares_session() -> None:
    saga = Saga(dialect="databricks")
    sess = saga.session
    pg = saga.with_dialect("postgres")
    assert pg.dialect == "postgres"
    assert pg.session is sess  # session shared, not duplicated


# -- Disk-spill staging session -----------------------------------------

def test_session_staging_path_and_autoclean(tmp_path: pathlib.Path) -> None:
    sess = SagaSession(root=tmp_path)
    staging = sess.stage_dir()
    assert staging.exists()
    assert staging == tmp_path / sess.session_id / "staging"
    sess.close()
    assert not (tmp_path / sess.session_id).exists()  # tree removed
    sess.close()  # idempotent


def test_collect_spills_result_to_session(users: ArrowTabular, tmp_path: pathlib.Path) -> None:
    saga = Saga(session=SagaSession(root=tmp_path))
    spilled = saga.collect("SELECT * FROM t", tables={"t": users}, spill=True)
    # Same data, now backed by a staging-aware holder.
    assert spilled.read_arrow_table().num_rows == 5
    saga.close()
    assert not (tmp_path / saga.session.session_id).exists()


def test_saga_context_manager_cleans_up(tmp_path: pathlib.Path) -> None:
    with Saga(session=SagaSession(root=tmp_path)) as saga:
        sid = saga.session.session_id
        saga.session.stage_dir()
        assert (tmp_path / sid).exists()
    assert not (tmp_path / sid).exists()
