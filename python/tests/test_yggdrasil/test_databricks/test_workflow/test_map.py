"""Tests for :meth:`WorkflowTask.map` — Prefect-style parallel fan-out.

Covers the two modes ``.map`` collapses into:

* **Local mode** (no active trace) — the task body runs across a real
  :class:`concurrent.futures.ThreadPoolExecutor` /
  :class:`~concurrent.futures.ProcessPoolExecutor`, preserving input
  order, with ``SecretRef`` defaults resolving the same way
  :meth:`WorkflowTask.__call__` resolves them.

* **Trace mode** (inside :class:`TraceContext`) — one
  :class:`TaskNode` per iterable element, with the
  collision-suffixing logic giving every fan-out task a unique key.
  Downstream tasks that take the resulting list as a single argument
  pick up the inferred ``depends_on`` edges via the container-walking
  :meth:`TaskNode._resolve_deps`.
"""
from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import threading
import time
import unittest

from yggdrasil.databricks.workflow import (
    TaskNode,
    TraceContext,
    flow,
    task,
)


_HAS_DILL = importlib.util.find_spec("dill") is not None


# ---------------------------------------------------------------- #
# Module-level fixtures so ProcessPoolExecutor can pickle them.
# ---------------------------------------------------------------- #


@task
def _double(x: int) -> int:
    return x * 2


@task
def _square_with_offset(x: int, offset: int = 0) -> int:
    return x * x + offset


@task(pool="thread", max_workers=4)
def _thread_pooled(x: int) -> int:
    return x + 1


@task
def _cube(x: int) -> int:
    return x ** 3


class TestMapLocal(unittest.TestCase):
    """``.map`` outside a trace runs the body in a local pool."""

    def test_returns_results_in_input_order(self) -> None:
        results = _double.map([1, 2, 3, 4, 5])

        self.assertEqual(results, [2, 4, 6, 8, 10])

    def test_results_are_returned_as_list(self) -> None:
        # ``.map`` materialises into a list so the trace-mode return
        # shape (list of TaskNodes) matches local mode (list of values).
        results = _double.map([7, 8])

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    def test_constants_pass_through_unchanged(self) -> None:
        results = _square_with_offset.map([1, 2, 3], offset=10)

        self.assertEqual(results, [11, 14, 19])

    def test_positional_constants_pass_through(self) -> None:
        results = _square_with_offset.map([1, 2, 3], 100)

        self.assertEqual(results, [101, 104, 109])

    def test_empty_iterable_returns_empty_list(self) -> None:
        results = _double.map([])

        self.assertEqual(results, [])

    def test_generator_iterable_is_materialised(self) -> None:
        # A generator must work the same as a list — ``.map`` materialises
        # internally so the pool sees a known-length sequence.
        results = _double.map(i for i in range(4))

        self.assertEqual(results, [0, 2, 4, 6])

    def test_uses_a_thread_pool_for_concurrency(self) -> None:
        # I/O-bound shape: a barrier that completes only when ``n``
        # workers reach it simultaneously. Serial execution would hang;
        # parallel execution releases together.
        barrier = threading.Barrier(parties=3, timeout=2.0)

        @task(pool="thread", max_workers=3)
        def wait_then_double(x: int) -> int:
            barrier.wait()
            return x * 2

        start = time.monotonic()
        results = wait_then_double.map([1, 2, 3])
        elapsed = time.monotonic() - start

        self.assertEqual(results, [2, 4, 6])
        # Three threads tripped the same barrier — they ran concurrently.
        # Allow generous headroom for slow CI.
        self.assertLess(elapsed, 1.5)

    def test_max_workers_caps_concurrency(self) -> None:
        # Tighten the barrier to ``max_workers=2``; if the executor
        # actually respects the cap, three items processed against a
        # 2-party barrier never all run at once and the barrier waits
        # for at least one rotation.
        observed_concurrency = []
        lock = threading.Lock()
        active = [0]

        @task(pool="thread", max_workers=2)
        def step(x: int) -> int:
            with lock:
                active[0] += 1
                observed_concurrency.append(active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return x

        step.map([1, 2, 3, 4, 5, 6])

        self.assertLessEqual(max(observed_concurrency), 2)

    def test_external_executor_is_not_shut_down(self) -> None:
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            results = _double.map([1, 2, 3], executor=ex)
            self.assertEqual(results, [2, 4, 6])

            # The decorator should not close the externally-owned executor.
            fut = ex.submit(lambda: 42)
            self.assertEqual(fut.result(timeout=1), 42)

    @unittest.skipUnless(
        _HAS_DILL,
        "Process-pool pickling of an @task-decorated callable needs dill "
        "(install via 'ygg[pickle]'); the @task decorator shadows the "
        "function's module-level name, breaking stock pickle.",
    )
    def test_pool_override_at_call_site(self) -> None:
        # The default pool is "thread"; ``pool="process"`` should drive
        # the run through a ProcessPoolExecutor. With dill installed
        # (and the ``yggdrasil.pyutils.parallel`` import-time patch
        # active), the decorated callable round-trips across the fork
        # boundary.
        # Trigger the dill side-effect patch on ``ForkingPickler``.
        import yggdrasil.pyutils.parallel  # noqa: F401

        results = _cube.map([0, 1, 2, 3], pool="process")

        self.assertEqual(results, [0, 1, 8, 27])

    def test_invalid_pool_at_call_site_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 'thread' or 'process'"):
            _double.map([1, 2, 3], pool="weird")

    def test_invalid_pool_at_decoration_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 'thread' or 'process'"):
            @task(pool="weird")
            def bad(x: int) -> int:
                return x


class TestMapInsideTrace(unittest.TestCase):
    """``.map`` inside a :class:`TraceContext` registers one node per item."""

    def test_records_one_node_per_element(self) -> None:
        with TraceContext() as ctx:
            nodes = _double.map([10, 20, 30])

        self.assertEqual(len(nodes), 3)
        for node in nodes:
            self.assertIsInstance(node, TaskNode)
        self.assertEqual(len(ctx.nodes), 3)

    def test_task_keys_are_collision_suffixed(self) -> None:
        with TraceContext() as ctx:
            _double.map([1, 2, 3, 4])

        self.assertEqual(
            [n.task_key for n in ctx.nodes],
            ["_double", "_double_2", "_double_3", "_double_4"],
        )

    def test_each_node_carries_its_item_as_first_arg(self) -> None:
        with TraceContext() as ctx:
            _double.map([7, 8, 9])

        self.assertEqual([n.args for n in ctx.nodes], [(7,), (8,), (9,)])

    def test_constants_propagate_to_every_node(self) -> None:
        with TraceContext() as ctx:
            _square_with_offset.map([1, 2, 3], offset=99)

        for node, expected_item in zip(ctx.nodes, [1, 2, 3]):
            self.assertEqual(node.args, (expected_item,))
            self.assertEqual(node.kwargs, {"offset": 99})

    def test_upstream_taskdep_propagates_to_every_mapped_node(self) -> None:
        @task
        def root() -> list[int]:
            return [1, 2, 3]

        with TraceContext() as ctx:
            up = root()
            # Pass ``up`` as a constant — every mapped node should
            # depend on it.
            mapped = _square_with_offset.map([10, 20, 30], offset=0)
            # …and now manually thread ``up`` through to assert deps
            # don't leak when the upstream isn't actually referenced.
            for node in mapped:
                self.assertEqual(node.depends_on, [])

            # When the upstream IS referenced, it lands on every node.
            mapped2 = _square_with_offset.map([1, 2], offset=up)
            for node in mapped2:
                self.assertEqual(node.depends_on, [up])

        # Sanity: root + first map (3) + second map (2) = 6 nodes.
        self.assertEqual(len(ctx.nodes), 6)

    def test_reducer_picks_up_all_mapped_nodes_via_container(self) -> None:
        @task
        def reducer(items: list) -> int:
            return sum(items)

        @flow
        def my_flow():
            mapped = _double.map([1, 2, 3])
            reducer(mapped)

        nodes = my_flow.trace()

        # 3 fan-out + 1 reducer = 4 nodes.
        self.assertEqual(len(nodes), 4)
        reducer_node = nodes[-1]
        self.assertEqual(reducer_node.task_key, "reducer")
        # The reducer's depends_on must include every mapped node.
        self.assertEqual(len(reducer_node.depends_on), 3)
        self.assertEqual(
            [d.task_key for d in reducer_node.depends_on],
            ["_double", "_double_2", "_double_3"],
        )

    def test_pool_kwargs_are_ignored_in_trace_mode(self) -> None:
        # In trace mode Databricks' scheduler owns parallelism — local
        # ``pool`` / ``max_workers`` overrides are no-ops; we just
        # record nodes regardless.
        with TraceContext() as ctx:
            nodes = _double.map(
                [1, 2, 3],
                pool="process",
                max_workers=99,
            )

        self.assertEqual(len(nodes), 3)
        self.assertEqual(len(ctx.nodes), 3)


class TestMapResolveDepsContainerWalk(unittest.TestCase):
    """``_resolve_deps`` recurses into list/tuple/dict containers."""

    def test_list_arg_carrying_task_nodes_creates_edges(self) -> None:
        @task
        def step() -> int:
            return 1

        @task
        def collect(items: list) -> int:
            return sum(items)

        with TraceContext() as ctx:
            a = step()
            b = step()
            c = step()
            r = collect([a, b, c])

        self.assertEqual([d.task_key for d in r.depends_on], ["step", "step_2", "step_3"])
        self.assertEqual(len(ctx.nodes), 4)

    def test_dict_arg_carrying_task_nodes_creates_edges(self) -> None:
        @task
        def step() -> int:
            return 1

        @task
        def consume(mapping: dict) -> int:
            return sum(mapping.values())

        with TraceContext():
            a = step()
            b = step()
            r = consume({"a": a, "b": b})

        self.assertEqual(set(d.task_key for d in r.depends_on), {"step", "step_2"})

    def test_string_args_are_not_walked(self) -> None:
        # Strings are iterable but never carry TaskNodes — walking
        # them would yield no nodes anyway, but the recursion guard
        # protects against pathological repr cost on long strings.
        @task
        def step() -> str:
            return "hello"

        @task
        def consume(s: str) -> int:
            return len(s)

        with TraceContext():
            r = consume("a long string that should not be walked")

        self.assertEqual(r.depends_on, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
