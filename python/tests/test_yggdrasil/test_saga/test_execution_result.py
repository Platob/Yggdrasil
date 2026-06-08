"""Tests for :class:`yggdrasil.saga.ExecutionResult` — lazy awaitable plan run."""

from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.dataclasses.awaitable import Awaitable
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.saga import ExecutionResult, Saga


@pytest.fixture
def users() -> ArrowTabular:
    return ArrowTabular(pa.table({
        "id": [1, 2, 3, 4, 5],
        "name": ["a", "b", "c", "d", "e"],
        "score": [90, 80, 95, 70, 85],
    }))


@pytest.fixture
def saga() -> Saga:
    return Saga()


def test_is_tabular_and_awaitable(saga: Saga, users: ArrowTabular) -> None:
    r = saga.submit("SELECT * FROM t", tables={"t": users})
    assert isinstance(r, Tabular)
    assert isinstance(r, Awaitable)


def test_lazy_until_read(saga: Saga, users: ArrowTabular) -> None:
    r = saga.submit("SELECT name FROM t", tables={"t": users})
    assert not r.started
    assert r.state.is_idle
    assert r.result is None
    # reading triggers execution
    out = r.read_arrow_table()
    assert out.column_names == ["name"]
    assert r.done and r.state.is_succeeded


def test_read_returns_plan_result(saga: Saga, users: ArrowTabular) -> None:
    r = saga.submit("SELECT name, score FROM t WHERE score > 80 ORDER BY score DESC",
                    tables={"t": users})
    assert r.read_arrow_table().to_pylist() == [
        {"name": "c", "score": 95},
        {"name": "a", "score": 90},
        {"name": "e", "score": 85},
    ]


def test_start_wait_lifecycle(saga: Saga, users: ArrowTabular) -> None:
    r = saga.submit("SELECT id FROM t WHERE id = 3", tables={"t": users})
    r.start(wait=False)
    assert r.started
    r.wait()
    assert r.done
    assert r.collect().read_arrow_table().to_pylist() == [{"id": 3}]


def test_collect_caches_result(saga: Saga, users: ArrowTabular) -> None:
    r = saga.submit("SELECT id FROM t", tables={"t": users})
    first = r.collect()
    assert r.result is first  # raw plan result cached
    assert r.collect() is first  # no re-execution


def test_await_runs_and_resolves(saga: Saga, users: ArrowTabular) -> None:
    async def go() -> None:
        r = saga.submit("SELECT COUNT(*) AS n FROM t", tables={"t": users})
        awaited = await r
        assert awaited is r
        assert r.state.is_succeeded
        assert r.collect().read_arrow_table().to_pylist() == [{"n": 5}]

    asyncio.run(go())


def test_error_propagation(saga: Saga) -> None:
    r = saga.submit("SELECT * FROM does_not_exist")
    with pytest.raises(Exception):
        r.read_arrow_table()
    assert r.failed
    assert r.error is not None
    with pytest.raises(Exception):
        r.raise_for_status()


def test_select_has_no_operation_result(saga: Saga, users: ArrowTabular) -> None:
    r = saga.submit("SELECT * FROM t", tables={"t": users})
    r.collect()
    assert r.operation_result is None


def test_plan_submit(saga: Saga, users: ArrowTabular) -> None:
    plan = saga.plan("SELECT score FROM t")
    plan.bind(users)
    r = plan.submit()
    assert isinstance(r, ExecutionResult)
    assert r.read_arrow_table().num_rows == 5


def test_node_submit(saga: Saga, users: ArrowTabular) -> None:
    node = saga.parse("SELECT name FROM t")
    r = node.submit(tables={"t": users})
    assert isinstance(r, ExecutionResult)
    assert r.read_arrow_table().column_names == ["name"]


def test_repr(saga: Saga, users: ArrowTabular) -> None:
    r = saga.submit("SELECT * FROM t", tables={"t": users})
    assert "ExecutionResult" in repr(r)
    assert "idle" in repr(r).lower()
