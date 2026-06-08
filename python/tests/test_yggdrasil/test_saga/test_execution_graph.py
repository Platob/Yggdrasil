"""Graph execution + display for :class:`ExecutionResult` / :class:`LazyTabular`."""

from __future__ import annotations

import time

import pyarrow as pa
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.saga import ExecutionResult
from yggdrasil.saga.plan import LazyTabular, SelectPlan


def test_lazytabular_is_executionresult() -> None:
    assert LazyTabular is ExecutionResult


def test_lazy_is_unstarted_plan() -> None:
    t = ArrowTabular(pa.table({"id": [1, 2, 3]}))
    lz = t.lazy()
    assert isinstance(lz, ExecutionResult)
    assert lz.state.is_idle and not lz.started
    assert lz.source is t


def test_builder_chain_then_read() -> None:
    t = ArrowTabular(pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"], "v": [9, 1, 5]}))
    out = (t.lazy().filter("v > 2").select("id", "name").order_by("-id").read_arrow_table())
    assert out.to_pylist() == [{"id": 3, "name": "c"}, {"id": 1, "name": "a"}]


def test_cannot_transform_after_start() -> None:
    t = ArrowTabular(pa.table({"id": [1, 2]}))
    lz = t.lazy().filter("id > 0")
    lz.read_arrow_table()
    with pytest.raises(RuntimeError):
        lz.filter("id > 1")


def test_join_builds_child_graph() -> None:
    users = ArrowTabular(pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]}))
    orders = ArrowTabular(pa.table({"id": [1, 2, 3], "amt": [10, 20, 30]}))
    joined = users.lazy().filter("id > 1").join(orders.lazy().filter("amt >= 20"), on="id")
    # the lazy right side is a scheduled child
    assert len(joined.children) == 1
    assert joined.read_arrow_table().to_pylist() == [
        {"id": 2, "name": "b", "amt": 20},
        {"id": 3, "name": "c", "amt": 30},
    ]


def _parent_of_three(slow: bool = False):
    cls = _SlowTabular if slow else ArrowTabular
    a = cls(pa.table({"id": [1], "v": [10]})).lazy().filter("v >= 0")
    b = cls(pa.table({"id": [2], "v": [20]})).lazy().filter("v >= 0")
    c = cls(pa.table({"id": [3], "v": [30]})).lazy().filter("v >= 0")
    return ExecutionResult(SelectPlan(source=a)).union(b).union(c)


class _SlowTabular(ArrowTabular):
    def _read_arrow_batches(self, options):  # type: ignore[override]
        time.sleep(0.2)
        yield from super()._read_arrow_batches(options)


def test_parallel_children_and_unique_ids() -> None:
    parent = _parent_of_three()
    kids = parent.children
    assert len(kids) == 3
    assert parent.child_mode == "parallel"
    # ids are distinct across the graph
    ids = {parent.short_id, *(c.short_id for c in kids)}
    assert len(ids) == 4
    assert sorted(r["id"] for r in parent.read_arrow_table().to_pylist()) == [1, 2, 3]


def test_parallel_actually_overlaps() -> None:
    parent = _parent_of_three(slow=True)
    t0 = time.perf_counter()
    parent.read_arrow_table()
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5  # 3×0.2s overlap, not 0.6s sequential


def test_sequential_mode_via_max_concurrency() -> None:
    a = ArrowTabular(pa.table({"id": [1], "v": [1]})).lazy().filter("v >= 0")
    b = ArrowTabular(pa.table({"id": [2], "v": [2]})).lazy().filter("v >= 0")
    parent = ExecutionResult(SelectPlan(source=a), max_concurrency=1).union(b)
    assert parent.child_mode == "sequential"
    assert sorted(r["id"] for r in parent.read_arrow_table().to_pylist()) == [1, 2]


def test_tree_and_graph_display() -> None:
    parent = _parent_of_three()
    tree = parent.tree()
    assert "∥ parallel" in tree
    assert f"#{parent.short_id}" in tree
    for c in parent.children:
        assert f"#{c.short_id}" in tree
    g = parent.graph()
    assert g["mode"] == "parallel"
    assert len(g["children"]) == 3
    # glyph reflects state after a run
    parent.read_arrow_table()
    assert "●" in parent.tree()


def test_display_prints(capsys: pytest.CaptureFixture) -> None:
    t = ArrowTabular(pa.table({"id": [1, 2]}))
    t.lazy().filter("id > 0").display()
    assert "SelectPlan" in capsys.readouterr().out
