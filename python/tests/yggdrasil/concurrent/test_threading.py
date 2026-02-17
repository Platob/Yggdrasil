# test_threading.py
#
# Pytest unit tests for yggdrasil.concurrent.threading (Job + JobThreadPoolExecutor)
#
# Design goals:
# - No flaky assumptions about OS thread scheduling.
# - Avoid long sleeps; use Events to control when work completes.
# - IMPORTANT: as_completed() is a generator; it does not submit/prime until iterated.
# - Prefer max_in_flight (new API). We still validate max_buffer back-compat.
#
# Run:
#   pytest -q

from __future__ import annotations

import threading
import time
from typing import Callable, List, Set

import pytest

from yggdrasil.concurrent.threading import Job, JobThreadPoolExecutor


# ----------------- helpers -----------------

def _spin_until(pred: Callable[[], bool], timeout: float = 1.0, interval: float = 0.005) -> bool:
    t0 = time.time()
    while time.time() - t0 <= timeout:
        if pred():
            return True
        time.sleep(interval)
    return False


def _drain_futures(gen, out_results: List[int | str], *, stop_after: int | None = None) -> None:
    """
    Drain a futures generator into out_results. Optionally stop after N results.
    NOTE: Caller controls generator lifetime; breaking early may leave in-flight tasks running.
    """
    for fut in gen:
        out_results.append(fut.result(timeout=1.0))
        if stop_after is not None and len(out_results) >= stop_after:
            break


# ----------------- tests -----------------

def test_job_make_and_run_basic():
    def add(a, b, *, c=0):
        return a + b + c

    job = Job.make(add, 1, 2, c=3)
    assert job.run() == 6
    assert job.args == (1, 2)
    assert job.kwargs == {"c": 3}


def test_as_completed_rejects_non_positive_limits():
    with JobThreadPoolExecutor(max_workers=2) as ex:
        with pytest.raises(ValueError):
            list(ex.as_completed([], max_in_flight=0))
        with pytest.raises(ValueError):
            list(ex.as_completed([], max_in_flight=-5))

        # Back-compat path: max_buffer should also reject
        with pytest.raises(ValueError):
            list(ex.as_completed([], max_buffer=0))
        with pytest.raises(ValueError):
            list(ex.as_completed([], max_buffer=-1))


def test_as_completed_empty_iterable_yields_nothing_ordered_and_unordered():
    with JobThreadPoolExecutor(max_workers=2) as ex:
        assert list(ex.as_completed([], ordered=False, max_in_flight=10)) == []
    with JobThreadPoolExecutor(max_workers=2) as ex:
        assert list(ex.as_completed([], ordered=True, max_in_flight=10)) == []


def test_unordered_completion_allows_overtake():
    """
    ordered=False yields completion order (fast can overtake slow head).
    We create: slow, fast1, fast2. With max_in_flight=3 they all submit.
    First two yields must be fast ones (any order), last is slow after release.
    """
    gate = threading.Event()

    def slow():
        gate.wait(timeout=2.0)
        return "slow"

    def fast(tag: str):
        return tag

    jobs = [Job.make(slow), Job.make(fast, "fast1"), Job.make(fast, "fast2")]

    with JobThreadPoolExecutor(max_workers=3) as ex:
        gen = ex.as_completed(jobs, ordered=False, max_in_flight=3)

        f1 = next(gen)
        f2 = next(gen)
        r1 = f1.result(timeout=1.0)
        r2 = f2.result(timeout=1.0)
        assert {r1, r2} == {"fast1", "fast2"}

        gate.set()
        f3 = next(gen)
        assert f3.result(timeout=1.0) == "slow"

        with pytest.raises(StopIteration):
            next(gen)


def test_ordered_blocks_behind_head_and_preserves_order():
    """
    ordered=True yields submission order and blocks behind slow head.
    Validate blocking by calling next(gen) in another thread and ensuring it doesn't return
    until we release the head.
    """
    gate = threading.Event()

    def slow():
        gate.wait(timeout=2.0)
        return "slow"

    def fast():
        return "fast"

    jobs = [Job.make(slow), Job.make(fast)]

    with JobThreadPoolExecutor(max_workers=2) as ex:
        gen = ex.as_completed(jobs, ordered=True, max_in_flight=2)

        box = {}

        def pull_first():
            box["f0"] = next(gen)

        t = threading.Thread(target=pull_first, daemon=True)
        t.start()

        # Should still be blocked shortly after start (head not released).
        t.join(timeout=0.05)
        assert t.is_alive()

        gate.set()
        t.join(timeout=1.0)
        assert not t.is_alive()

        f0 = box["f0"]
        assert f0.result(timeout=1.0) == "slow"

        f1 = next(gen)
        assert f1.result(timeout=1.0) == "fast"

        with pytest.raises(StopIteration):
            next(gen)


def test_unordered_respects_max_in_flight_peak_concurrency():
    """
    Ensure unordered mode never runs > max_in_flight tasks concurrently.
    Key detail: generator submits only when iterated, so we start draining in a thread.
    """
    lock = threading.Lock()
    gate = threading.Event()
    started_any = threading.Event()
    current = 0
    peak = 0

    def tracked(i: int) -> int:
        nonlocal current, peak
        with lock:
            current += 1
            peak = max(peak, current)
            started_any.set()
        gate.wait(timeout=2.0)
        with lock:
            current -= 1
        return i

    max_in_flight = 2
    jobs = [Job.make(tracked, i) for i in range(20)]
    results: List[int] = []

    with JobThreadPoolExecutor(max_workers=20) as ex:
        gen = ex.as_completed(jobs, ordered=False, max_in_flight=max_in_flight)

        t = threading.Thread(target=_drain_futures, args=(gen, results), daemon=True)
        t.start()

        assert started_any.wait(timeout=1.0)
        assert _spin_until(lambda: peak >= 1, timeout=1.0)
        assert peak <= max_in_flight

        gate.set()
        t.join(timeout=2.0)
        assert not t.is_alive()

    assert sorted(results) == list(range(20))


def test_ordered_respects_max_in_flight_peak_concurrency_and_output_order():
    """
    Ensure ordered mode never runs > max_in_flight tasks concurrently AND yields exact order.
    """
    lock = threading.Lock()
    gate = threading.Event()
    started_any = threading.Event()
    current = 0
    peak = 0

    def tracked(i: int) -> int:
        nonlocal current, peak
        with lock:
            current += 1
            peak = max(peak, current)
            started_any.set()
        gate.wait(timeout=2.0)
        with lock:
            current -= 1
        return i

    max_in_flight = 3
    jobs = [Job.make(tracked, i) for i in range(12)]
    results: List[int] = []

    with JobThreadPoolExecutor(max_workers=12) as ex:
        gen = ex.as_completed(jobs, ordered=True, max_in_flight=max_in_flight)

        t = threading.Thread(target=_drain_futures, args=(gen, results), daemon=True)
        t.start()

        assert started_any.wait(timeout=1.0)
        assert _spin_until(lambda: peak >= 1, timeout=1.0)
        assert peak <= max_in_flight

        gate.set()
        t.join(timeout=2.0)
        assert not t.is_alive()

    assert results == list(range(12))


def test_unordered_yields_all_results_exactly_once():
    """
    Unordered mode: no dups, no drops.
    """
    def work(i: int) -> int:
        return i

    n = 200
    jobs = [Job.make(work, i) for i in range(n)]

    with JobThreadPoolExecutor(max_workers=8) as ex:
        results = [f.result(timeout=1.0) for f in ex.as_completed(jobs, ordered=False, max_in_flight=25)]

    assert len(results) == n
    assert set(results) == set(range(n))


def test_ordered_yields_all_results_exactly_once_in_order():
    """
    Ordered mode: must be exactly submission order.
    """
    def work(i: int) -> int:
        return i

    n = 200
    jobs = [Job.make(work, i) for i in range(n)]

    with JobThreadPoolExecutor(max_workers=8) as ex:
        results = [f.result(timeout=1.0) for f in ex.as_completed(jobs, ordered=True, max_in_flight=25)]

    assert results == list(range(n))


def test_breaking_early_from_huge_stream_is_safe_and_plausible_subset():
    """
    Break early from a huge stream without hanging.
    ordered=False => no ordering guarantee.

    We assert:
      - we got k results
      - all are from the first max_in_flight submissions (only those can be in-flight initially)
    """
    def work(i: int) -> int:
        # tiny jitter to prevent accidental consistent ordering on some machines
        if i % 7 == 0:
            time.sleep(0.002)
        return i * 2

    def job_stream():
        for i in range(10_000):
            yield Job.make(work, i)

    k = 5
    max_in_flight = 10

    with JobThreadPoolExecutor(max_workers=4) as ex:
        gen = ex.as_completed(job_stream(), ordered=False, max_in_flight=max_in_flight)
        out: List[int] = []
        _drain_futures(gen, out, stop_after=k)

    assert len(out) == k
    assert all(x % 2 == 0 for x in out)
    allowed: Set[int] = {2 * i for i in range(max_in_flight)}
    assert set(out).issubset(allowed)


def test_max_buffer_alias_overrides_max_in_flight():
    """
    Back-compat / precedence behavior: if max_buffer is explicitly passed, it should take precedence.
    We validate by setting max_in_flight high and max_buffer low, then asserting peak <= low.
    """
    lock = threading.Lock()
    gate = threading.Event()
    started_any = threading.Event()
    current = 0
    peak = 0

    def tracked(i: int) -> int:
        nonlocal current, peak
        with lock:
            current += 1
            peak = max(peak, current)
            started_any.set()
        gate.wait(timeout=2.0)
        with lock:
            current -= 1
        return i

    max_in_flight = 10
    max_buffer = 2
    jobs = [Job.make(tracked, i) for i in range(20)]
    results: List[int] = []

    with JobThreadPoolExecutor(max_workers=20) as ex:
        gen = ex.as_completed(
            jobs,
            ordered=False,
            max_in_flight=max_in_flight,
            max_buffer=max_buffer,  # should win
        )

        t = threading.Thread(target=_drain_futures, args=(gen, results), daemon=True)
        t.start()

        assert started_any.wait(timeout=1.0)
        assert _spin_until(lambda: peak >= 1, timeout=1.0)
        assert peak <= max_buffer

        gate.set()
        t.join(timeout=2.0)
        assert not t.is_alive()

    assert sorted(results) == list(range(20))
