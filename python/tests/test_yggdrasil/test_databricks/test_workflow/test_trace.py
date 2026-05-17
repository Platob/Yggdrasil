"""Tests for the trace-on-deploy DAG capture mechanism.

Covers the contract that ``@task`` calls inside a ``@flow`` body, when
the trace context is active, produce :class:`TaskNode` futures with
the right upstream edges and key suffixing on collision.
"""
from __future__ import annotations

import unittest

from yggdrasil.databricks.workflow import (
    FlowParam,
    TaskNode,
    TraceContext,
    current_trace,
    flow,
    task,
)


class TestTraceCaptureOutsideFlow(unittest.TestCase):
    """``@task`` calls outside a trace run as plain Python."""

    def test_call_returns_real_value_when_no_trace(self) -> None:
        @task
        def step(x: int) -> int:
            return x * 2

        self.assertEqual(step(3), 6)

    def test_current_trace_is_none_by_default(self) -> None:
        self.assertIsNone(current_trace())


class TestTraceCaptureInsideFlow(unittest.TestCase):
    """Inside a :class:`TraceContext` every ``@task`` call becomes a node."""

    def test_simple_dag_records_upstream_dep(self) -> None:
        @task
        def a(x: int) -> str:
            return f"a-{x}"

        @task
        def b(s: str) -> str:
            return f"b-{s}"

        with TraceContext() as ctx:
            r = a(7)
            self.assertIsInstance(r, TaskNode)
            self.assertEqual(r.task_key, "a")
            r2 = b(r)
            self.assertIsInstance(r2, TaskNode)
            self.assertEqual(r2.task_key, "b")
            self.assertEqual(r2.depends_on, [r])

        self.assertEqual([n.task_key for n in ctx.nodes], ["a", "b"])

    def test_duplicate_task_keys_are_suffixed(self) -> None:
        @task
        def step(x: int) -> int:
            return x + 1

        with TraceContext() as ctx:
            step(1)
            step(2)
            step(3)

        self.assertEqual(
            [n.task_key for n in ctx.nodes],
            ["step", "step_2", "step_3"],
        )

    def test_flow_param_kept_as_node_arg(self) -> None:
        @task
        def step(value: str) -> str:
            return value.upper()

        @flow(name="param-flow")
        def my_flow(value: str = "default"):
            step(value)

        nodes = my_flow.trace()
        self.assertEqual(len(nodes), 1)
        self.assertIsInstance(nodes[0].args[0], FlowParam)
        self.assertEqual(nodes[0].args[0].name, "value")
        self.assertEqual(nodes[0].args[0].default, "default")

    def test_trace_with_override_pins_value(self) -> None:
        """Overrides bypass FlowParam — the literal value lands on the node."""

        @task
        def step(value: str) -> str:
            return value

        @flow
        def my_flow(value: str = "default"):
            step(value)

        nodes = my_flow.trace(value="pinned")
        self.assertEqual(nodes[0].args, ("pinned",))

    def test_unknown_override_raises(self) -> None:
        @task
        def step():
            return None

        @flow
        def my_flow():
            step()

        with self.assertRaisesRegex(TypeError, "unknown parameter"):
            my_flow.trace(missing="boom")

    def test_explicit_after_attaches_extra_dep(self) -> None:
        @task
        def upstream():
            return None

        # ``@other.after(upstream)`` adds a hard edge that wouldn't be
        # inferred from arguments alone — side-effect-only ordering.
        @task
        def downstream():
            return None

        # Imitate the decorator pattern by hand to avoid relying on
        # decorator order in the test source: get a wrapped callable
        # back from ``after`` and call it within the trace.
        with TraceContext() as ctx:
            up = upstream()
            after_wrapped = downstream.after(up)(downstream)
            down = after_wrapped()
            self.assertIn(up, down.depends_on)

        self.assertEqual(len(ctx.nodes), 2)


class TestEmptyFlowDeployErrors(unittest.TestCase):
    def test_deploy_with_no_tasks_raises(self) -> None:
        @flow
        def empty_flow():
            pass

        # Trace produces zero nodes — deploy should fail loudly.
        from unittest.mock import MagicMock

        jobs = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "produced no tasks"):
            empty_flow.deploy(service=jobs)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
