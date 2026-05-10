"""Unit tests for :class:`Session`'s parallel cache-insert helpers.

Covers:

- ``Session._run_concurrently`` runs tasks in parallel, propagates the
  first exception, and short-circuits the empty / single-task cases.
- ``Session._enable_fair_spark_scheduler`` calls ``conf.set`` with
  ``spark.scheduler.mode=FAIR`` and is forgiving on Spark builds that
  reject the runtime change.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from yggdrasil.io.session import Session


# ---------------------------------------------------------------------------
# _run_concurrently
# ---------------------------------------------------------------------------


class TestRunConcurrently:

    def test_empty_tasks_is_noop(self) -> None:
        Session._run_concurrently([])

    def test_single_task_runs_inline(self) -> None:
        # Single task must execute on the calling thread — no pool
        # spawn — so the saved thread name matches.
        seen: "list[str]" = []
        caller = threading.current_thread().name

        def task() -> None:
            seen.append(threading.current_thread().name)

        Session._run_concurrently([task])
        assert seen == [caller]

    def test_runs_in_parallel(self) -> None:
        # Each task blocks on a barrier so the wall clock proves the
        # tasks ran simultaneously rather than head-to-tail.
        n = 4
        barrier = threading.Barrier(n)
        durations: "list[float]" = []

        def task() -> None:
            t0 = time.monotonic()
            barrier.wait(timeout=2.0)
            durations.append(time.monotonic() - t0)

        start = time.monotonic()
        Session._run_concurrently([task] * n)
        elapsed = time.monotonic() - start

        # If they ran serially, total time would be roughly the sum
        # of the per-task waits; running concurrently caps it at the
        # single-task wait + scheduling overhead. Generous tolerance
        # for a slow CI box.
        assert elapsed < 1.0
        assert len(durations) == n

    def test_propagates_first_exception(self) -> None:
        completed: "list[int]" = []

        def good(idx: int) -> Any:
            time.sleep(0.05)
            completed.append(idx)
            return idx

        def bad() -> Any:
            raise RuntimeError("planned")

        with pytest.raises(RuntimeError, match="planned"):
            Session._run_concurrently([
                lambda: good(0),
                bad,
                lambda: good(1),
            ])

        # All other tasks were awaited before the exception surfaces
        # — no leaked workers, no half-finished state.
        assert sorted(completed) == [0, 1]

    def test_caps_workers_to_task_count(self) -> None:
        # Two tasks but max_workers=8 → pool is sized to 2 (no idle
        # threads spun up). Verified indirectly: distinct thread
        # names per task.
        names: "set[str]" = set()
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def task() -> None:
            barrier.wait(timeout=2.0)
            with lock:
                names.add(threading.current_thread().name)

        Session._run_concurrently(
            [task, task],
            max_workers=8,
            thread_name_prefix="ygg-test",
        )
        assert len(names) == 2
        assert all(n.startswith("ygg-test") for n in names)


# ---------------------------------------------------------------------------
# _enable_fair_spark_scheduler
# ---------------------------------------------------------------------------


class _FakeSparkConf:

    def __init__(self) -> None:
        self.set_calls: "list[tuple[str, str]]" = []

    def set(self, key: str, value: str) -> None:
        self.set_calls.append((key, value))


class _FakeSparkSession:

    def __init__(self) -> None:
        self.conf = _FakeSparkConf()


class _FakeSparkSessionRejecting:

    def __init__(self) -> None:
        class _Rejector:
            def set(self, key: str, value: str) -> None:
                raise RuntimeError(f"runtime change refused for {key}")
        self.conf = _Rejector()


class TestEnableFairSparkScheduler:

    def test_sets_fair_mode(self) -> None:
        spark = _FakeSparkSession()
        Session._enable_fair_spark_scheduler(spark)
        assert spark.conf.set_calls == [("spark.scheduler.mode", "FAIR")]

    def test_swallows_rejection(self) -> None:
        spark = _FakeSparkSessionRejecting()
        # Managed clusters that reject runtime scheduler-mode changes
        # must not break the request flow.
        Session._enable_fair_spark_scheduler(spark)

    def test_none_session_is_noop(self) -> None:
        Session._enable_fair_spark_scheduler(None)
