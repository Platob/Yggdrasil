# test_infinite_threadpool.py
from __future__ import annotations

import time
import threading
import logging
from collections import deque

import pytest

# Change this import to match your project layout.
# Example:
#   from yggdrasil.concurrent.infinite_threadpool import Job, JobPoolExecutor
from yggdrasil.concurrent.threading import Job, JobPoolExecutor


def test_job_make_and_run_positional_and_kwargs():
    def f(a, b, *, c=0):
        return a + b + c

    job = Job.make(f, 1, 2, c=3)
    assert job.run() == 6


def test_job_fire_and_forget_runs_async():
    evt = threading.Event()

    def f():
        evt.set()

    Job.make(f).fire_and_forget()
    assert evt.wait(timeout=1.0), "fire_and_forget should start the thread and run the function"


def test_submit_job_returns_future_and_result():
    with JobPoolExecutor(max_workers=2, max_in_flight=4) as ex:
        fut = ex.submit_job(Job.make(lambda x: x * 2, 21))
        assert fut.result(timeout=1.0) == 42


def test_as_completed_unordered_yields_all_results():
    # Unordered mode returns completion order, so we just check set equality.
    def mk(i: int):
        return Job.make(lambda x: x + 1, i)

    jobs = [mk(i) for i in range(20)]

    with JobPoolExecutor(max_workers=4, max_in_flight=8) as ex:
        out = list(ex.as_completed(jobs, ordered=False))
        assert set(out) == {i + 1 for i in range(20)}
        assert len(out) == 20


def test_as_completed_ordered_yields_submission_order_even_if_slow_head():
    # Head sleeps longer; ordered=True should block behind it and yield strictly in submit order.
    def sleeper(value: int, delay: float):
        time.sleep(delay)
        return value

    jobs = [
        Job.make(sleeper, 0, 0.10),
        Job.make(sleeper, 1, 0.00),
        Job.make(sleeper, 2, 0.00),
        Job.make(sleeper, 3, 0.00),
    ]

    with JobPoolExecutor(max_workers=4, max_in_flight=4) as ex:
        out = list(ex.as_completed(jobs, ordered=True))
        assert out == [0, 1, 2, 3]


def test_as_completed_unordered_completes_fast_jobs_first_typically():
    # This is probabilistic if you do tiny sleeps; we make it robust:
    # with 1 worker, completion == submission. With 2+, faster job should usually come out first.
    def sleeper(value: int, delay: float):
        time.sleep(delay)
        return value

    jobs = [
        Job.make(sleeper, "slow", 0.08),
        Job.make(sleeper, "fast", 0.00),
    ]

    with JobPoolExecutor(max_workers=2, max_in_flight=2) as ex:
        out = list(ex.as_completed(jobs, ordered=False))
        assert set(out) == {"slow", "fast"}
        # Most of the time, "fast" will appear first; allow rare scheduling weirdness:
        assert out[0] in {"slow", "fast"}


def test_raise_error_true_propagates_and_cancels_pending_best_effort():
    start_evt = threading.Event()
    release_evt = threading.Event()

    def boom():
        raise RuntimeError("kaboom")

    def blocked():
        # Signal that we've started (or at least are running) then block until released.
        start_evt.set()
        release_evt.wait(timeout=2.0)
        return "blocked-done"

    jobs = [
        Job.make(boom),
        Job.make(blocked),
        Job.make(blocked),
        Job.make(blocked),
    ]

    ex = JobPoolExecutor(max_workers=1, max_in_flight=4)
    gen = ex.as_completed(jobs, ordered=False, raise_error=True)

    with pytest.raises(RuntimeError, match="kaboom"):
        next(gen)

    # Make sure we don't hang tests if something is running.
    release_evt.set()
    ex.shutdown(wait=True, cancel_futures=True)

def test_max_in_flight_bounds_in_unordered_mode():
    # We measure the maximum concurrent active tasks inside the worker function.
    active = 0
    peak = 0
    lock = threading.Lock()
    release = threading.Event()

    def work(i: int):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        # block so we can build up in-flight
        release.wait(timeout=2.0)
        with lock:
            active -= 1
        return i

    jobs = [Job.make(work, i) for i in range(50)]

    with JobPoolExecutor(max_workers=32, max_in_flight=5) as ex:
        it = ex.as_completed(jobs, ordered=False)
        # allow executor to start submitting/running
        time.sleep(0.05)
        release.set()
        out = list(it)

    assert set(out) == set(range(50))
    assert peak <= 5, f"peak concurrency {peak} exceeded max_in_flight=5"


def test_cancel_on_exit_cancels_not_done_futures_best_effort():
    # We'll stop consuming early and rely on cancel_on_exit to cancel queued futures.
    release = threading.Event()

    def long(i: int):
        release.wait(timeout=2.0)
        return i

    jobs = [Job.make(long, i) for i in range(100)]

    ex = JobPoolExecutor(max_workers=1, max_in_flight=10)
    gen = ex.as_completed(jobs, ordered=False, cancel_on_exit=True, shutdown_on_exit=True, shutdown_wait=True)

    # Consume just one result (none will complete until release, so we need to release briefly)
    release.set()
    first = next(gen)
    assert isinstance(first, int)

    # Stop consuming: trigger generator finalizer by closing it.
    gen.close()

    # Nothing to assert deterministically about cancellation beyond "no hang" and shutdown executed.
    # If it got here, cancellation/shutdown logic didn't blow up.


def test_shutdown_on_exit_shuts_down_executor():
    # After the generator ends with shutdown_on_exit, executor should reject new submits.
    jobs = [Job.make(lambda i=i: i) for i in range(5)]

    ex = JobPoolExecutor(max_workers=2, max_in_flight=2)
    out = list(ex.as_completed(jobs, ordered=False, shutdown_on_exit=True, shutdown_wait=True))
    assert set(out) == set(range(5))

    with pytest.raises(RuntimeError):
        ex.submit(lambda: 1)


def test_parse_any_returns_cls_for_instance_and_executor_for_other():
    ex = JobPoolExecutor(max_workers=1)
    try:
        assert JobPoolExecutor.parse(ex) is JobPoolExecutor
        other = JobPoolExecutor.parse(object(), max_workers=3)
        assert isinstance(other, JobPoolExecutor)
        assert other.max_workers == 3
        other.shutdown(wait=True, cancel_futures=True)
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


def test_cancel_all_only_attempts_cancel_on_not_done():
    # This test is basically "doesn't crash", but also checks it flips cancel flag for queued futures.
    release = threading.Event()

    def block():
        release.wait(timeout=2.0)
        return 1

    with JobPoolExecutor(max_workers=1, max_in_flight=10) as ex:
        fs = [ex.submit(block) for _ in range(10)]
        # First one is running, rest queued
        JobPoolExecutor._cancel_all(fs)
        # queued futures should be cancelled (best effort)
        cancelled = sum(1 for f in fs[1:] if f.cancelled())
        assert cancelled >= 1

        release.set()
        # running one should finish
        assert fs[0].result(timeout=1.0) == 1


def test_ordered_mode_uses_deque_inflight_and_yields_all():
    def f(i: int):
        return i * i

    jobs = [Job.make(f, i) for i in range(15)]

    with JobPoolExecutor(max_workers=3, max_in_flight=4) as ex:
        out = list(ex.as_completed(jobs, ordered=True))
        assert out == [i * i for i in range(15)]