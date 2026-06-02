"""Unit tests for the job DAG view and the awaitable JobTask (no live cluster).

:class:`JobDag` turns SDK task lists (job settings ``Task`` or run ``RunTask``)
into a printable, topologically-ordered graph; :class:`JobTask` is the awaitable
per-task handle that polls its parent run until the task is terminal.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from databricks.sdk.service.jobs import (
    RunLifeCycleState,
    RunResultState,
    RunState,
)

from yggdrasil.databricks.job.dag import JobDag, JobDagNode
from yggdrasil.databricks.job.run import JobTask
from yggdrasil.enums.state import State


def _dep(key: str) -> SimpleNamespace:
    return SimpleNamespace(task_key=key)


def _task(key: str, deps: list[str] | None = None, state=None) -> SimpleNamespace:
    return SimpleNamespace(
        task_key=key,
        depends_on=[_dep(d) for d in (deps or [])],
        state=state,
    )


def _run_state(lc: RunLifeCycleState, result=None) -> RunState:
    return RunState(life_cycle_state=lc, result_state=result)


# --------------------------------------------------------------------------- #
# JobDag
# --------------------------------------------------------------------------- #
class TestJobDag:
    def test_from_tasks_builds_nodes_and_edges(self):
        dag = JobDag.from_tasks([
            _task("a"),
            _task("b", ["a"]),
            _task("c", ["a"]),
            _task("d", ["b", "c"]),
        ])
        assert len(dag) == 4
        assert dag.keys == ["a", "b", "c", "d"]
        assert dag.roots() == ["a"]
        assert sorted(dag.leaves()) == ["d"]
        assert ("a", "b") in dag.edges()
        assert ("b", "d") in dag.edges() and ("c", "d") in dag.edges()
        assert dag.node("d").depends_on == ("b", "c")

    def test_topological_order_respects_dependencies(self):
        # input order is intentionally reversed vs. dependency order
        dag = JobDag.from_tasks([
            _task("d", ["b", "c"]),
            _task("c", ["a"]),
            _task("b", ["a"]),
            _task("a"),
        ])
        order = [n.key for n in dag.topological_order()]
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_does_not_drop_tasks(self):
        # a → b → a is a cycle; render/order must still surface both nodes.
        dag = JobDag.from_tasks([_task("a", ["b"]), _task("b", ["a"])])
        order = [n.key for n in dag.topological_order()]
        assert sorted(order) == ["a", "b"]

    def test_render_includes_state_and_edges(self):
        dag = JobDag.from_tasks(
            [
                _task("a", state=_run_state(RunLifeCycleState.TERMINATED, RunResultState.SUCCESS)),
                _task("b", ["a"], state=_run_state(RunLifeCycleState.RUNNING)),
            ],
            state_of=lambda t: _resolve(t.state),
        )
        text = dag.render()
        assert "JobDag (2 tasks)" in text
        assert "b  ← a" in text
        assert "[SUCCEEDED]" in text and "[RUNNING]" in text
        assert str(dag) == text

    def test_empty_dag(self):
        dag = JobDag.from_tasks([])
        assert not dag and len(dag) == 0
        assert dag.render() == "JobDag (0 tasks)"


# --------------------------------------------------------------------------- #
# JobTask (awaitable)
# --------------------------------------------------------------------------- #
class TestJobTask:
    def test_snapshot_state_without_parent(self):
        raw = _task("ingest", state=_run_state(RunLifeCycleState.RUNNING))
        t = JobTask(raw)
        assert t.task_key == "ingest"
        assert t.state is State.RUNNING and not t.is_done

    def test_wait_polls_parent_run_until_terminal(self):
        # The parent run flips the task RUNNING → SUCCESS across two refreshes.
        states = [
            _run_state(RunLifeCycleState.RUNNING),
            _run_state(RunLifeCycleState.TERMINATED, RunResultState.SUCCESS),
        ]
        run = MagicMock()

        def _refresh():
            run.details.tasks = [_task("ingest", state=states[min(_refresh.n, 1)])]
            _refresh.n += 1
        _refresh.n = 0
        run.refresh.side_effect = _refresh

        t = JobTask(_task("ingest", state=states[0]), run=run)
        t.wait(wait=5)
        assert t.is_done and t.is_succeeded
        assert run.refresh.call_count >= 1

    def test_wait_raises_on_failed_task(self):
        run = MagicMock()
        run.details.tasks = [
            _task("ingest", state=_run_state(RunLifeCycleState.INTERNAL_ERROR, RunResultState.FAILED))
        ]
        t = JobTask(_task("ingest", state=_run_state(RunLifeCycleState.RUNNING)), run=run)
        try:
            t.wait(wait=5)
        except RuntimeError as e:
            assert "ingest" in str(e)
        else:  # pragma: no cover
            raise AssertionError("expected RuntimeError on failed task")
        assert t.is_failed


def _resolve(state):
    from yggdrasil.databricks.job.run import _resolve_state

    return _resolve_state(state)
