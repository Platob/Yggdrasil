"""Safety + semantics for :class:`JobPoolExecutor.as_completed`.

``JobPoolExecutor`` is the concurrency engine behind ``HTTPSession.send_many``
(and every other fan-out in the codebase): it keeps at most *max_in_flight*
jobs submitted at once so an unbounded job stream never floods the pool, and
streams :class:`JobResult` objects in completion order (``ordered=False``) or
strict submission order (``ordered=True``).

These tests pin the properties that make the HTTP fan-out safe:

* the in-flight window is genuinely bounded — peak concurrency never exceeds
  ``min(window, max_workers)``, even for a stream far larger than the window;
* every submitted job yields exactly one result; ordered mode preserves
  submission order, unordered mode returns the full set;
* ``raise_error=True`` surfaces the first failure (and stops feeding the
  stream); ``raise_error=False`` captures failures as :class:`JobResult`;
* a lazily-generated / effectively-infinite source is consumed incrementally,
  not eagerly drained;
* ``cancel_on_exit`` / ``shutdown_on_exit`` clean up when the consumer walks
  away early.
"""
from __future__ import annotations

import threading
import time
from itertools import count

import pytest

from yggdrasil.concurrent.job import Job
from yggdrasil.concurrent.job_result import JobResult
from yggdrasil.concurrent.pool import JobPoolExecutor


# ---------------------------------------------------------------------------
# A concurrency probe: records the peak number of jobs running simultaneously.
# ---------------------------------------------------------------------------


class _Probe:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.ran: list[int] = []

    def task(self, i: int, hold: float = 0.02):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(hold)
            with self.lock:
                self.ran.append(i)
            return i
        finally:
            with self.lock:
                self.active -= 1


def _jobs(probe: _Probe, n: int, hold: float = 0.02):
    return (Job.make(probe.task, i, hold) for i in range(n))


# ---------------------------------------------------------------------------
# In-flight window is bounded
# ---------------------------------------------------------------------------


def test_window_caps_peak_concurrency():
    probe = _Probe()
    with JobPoolExecutor(max_workers=8) as pool:
        results = list(pool.as_completed(_jobs(probe, 40), max_in_flight=4))
    assert len(results) == 40
    # Never more than the window ran at once, and it actually reached it.
    assert probe.peak <= 4
    assert probe.peak >= 3


def test_window_capped_by_max_workers():
    probe = _Probe()
    # Window wider than the pool — workers become the real ceiling.
    with JobPoolExecutor(max_workers=3) as pool:
        list(pool.as_completed(_jobs(probe, 30), max_in_flight=20))
    assert probe.peak <= 3


def test_window_bounds_submission_of_infinite_stream():
    """An effectively-infinite generator must be consumed incrementally.

    If ``as_completed`` eagerly drained the source it would spin forever here;
    instead it pulls only ``window`` jobs ahead, so taking the first few
    results returns promptly and the generator is never fully realised.
    """
    submitted = count()
    produced: list[int] = []

    def gen():
        for i in submitted:
            produced.append(i)
            yield Job.make(lambda x=i: x)

    with JobPoolExecutor(max_workers=4) as pool:
        stream = pool.as_completed(gen(), max_in_flight=4)
        first_five = [next(stream).result for _ in range(5)]
        stream.close()

    assert len(first_five) == 5
    # Only a bounded look-ahead was produced — not millions of jobs.
    assert len(produced) <= 4 + 5 + 2


# ---------------------------------------------------------------------------
# Completeness + ordering
# ---------------------------------------------------------------------------


def test_unordered_returns_all_results():
    probe = _Probe()
    with JobPoolExecutor(max_workers=8) as pool:
        out = [r.result for r in pool.as_completed(_jobs(probe, 25), ordered=False)]
    assert sorted(out) == list(range(25))


def test_ordered_preserves_submission_order():
    # Stagger completion: later indices finish sooner. Ordered mode must still
    # yield 0,1,2,... in submission order regardless of completion timing.
    def hold_for(i: int):
        time.sleep((20 - i) * 0.002)
        return i

    jobs = (Job.make(hold_for, i) for i in range(20))
    with JobPoolExecutor(max_workers=8) as pool:
        out = [r.result for r in pool.as_completed(jobs, ordered=True)]
    assert out == list(range(20))


def test_empty_stream_yields_nothing():
    with JobPoolExecutor(max_workers=4) as pool:
        assert list(pool.as_completed(iter([]))) == []
        assert list(pool.as_completed(iter([]), ordered=True)) == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ordered", [False, True])
def test_raise_error_true_propagates(ordered):
    def boom(i: int):
        if i == 5:
            raise ValueError(f"boom-{i}")
        return i

    jobs = (Job.make(boom, i) for i in range(20))
    with JobPoolExecutor(max_workers=4) as pool:
        with pytest.raises(ValueError, match="boom-5"):
            list(pool.as_completed(jobs, ordered=ordered, raise_error=True))


@pytest.mark.parametrize("ordered", [False, True])
def test_raise_error_false_drops_failures(ordered):
    # ``raise_error=False`` swallows failing jobs (logged, not yielded) instead
    # of surfacing them — the stream returns only the successful results and
    # never raises. Pin this so the suppression can't silently change shape.
    def boom(i: int):
        if i % 4 == 0:
            raise RuntimeError(f"fail-{i}")
        return i

    jobs = (Job.make(boom, i) for i in range(16))
    with JobPoolExecutor(max_workers=4) as pool:
        results = list(pool.as_completed(jobs, ordered=ordered, raise_error=False))

    assert all(isinstance(r, JobResult) and r.ok for r in results)
    # i in {0,4,8,12} raise and are dropped; the other 12 come back intact.
    assert sorted(r.result for r in results) == [
        i for i in range(16) if i % 4 != 0
    ]


# ---------------------------------------------------------------------------
# Early-exit cleanup
# ---------------------------------------------------------------------------


def test_cancel_on_exit_cancels_pending_on_early_close():
    release = threading.Event()
    ran: list[int] = []
    ran_lock = threading.Lock()

    def task(i: int):
        with ran_lock:
            ran.append(i)
        if i == 0:
            return i  # first job returns immediately so next() can yield
        release.wait(2.0)  # the rest block, occupying / waiting on the pool
        return i

    # ``as_completed`` is a lazy generator — nothing is submitted until the
    # first ``next``. Pull exactly one result (job 0), then walk away: the
    # finally clause must cancel the still-pending window and shut the pool
    # down so the 200-job stream never gets fully dispatched.
    pool = JobPoolExecutor(max_workers=2)
    stream = pool.as_completed(
        (Job.make(task, i) for i in range(200)),
        max_in_flight=3,
        cancel_on_exit=True,
        shutdown_on_exit=True,
        shutdown_wait=False,
    )
    first = next(stream)
    assert first.result == 0
    stream.close()  # GeneratorExit → finally: cancel pending + shutdown
    release.set()

    # Only the initial window's worth of jobs was ever dispatched — nowhere
    # near the 200 queued.
    with ran_lock:
        assert len(ran) <= 8


def test_shutdown_on_exit_closes_pool():
    pool = JobPoolExecutor(max_workers=4)
    list(pool.as_completed(
        (Job.make(lambda x=i: x) for i in range(5)),
        shutdown_on_exit=True,
        shutdown_wait=True,
    ))
    # The pool was shut down inside as_completed — further submits must reject.
    with pytest.raises(RuntimeError):
        pool.submit(lambda: 1)
