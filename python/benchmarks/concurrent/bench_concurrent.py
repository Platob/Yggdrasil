"""Benchmark the :mod:`yggdrasil.concurrent` primitives.

Why this exists
---------------

Three primitives in this module land on every fan-out / retry / async
path in the codebase:

* :class:`Job` — immutable callable bundle built by every call site
  that wants "run this later" or "run this on a thread". Built per
  retry, per remote call, per ``fire_and_forget``.
* :class:`JobResult` — wrapper returned by ``JobPoolExecutor`` and
  ``ThreadJob`` — one allocation per completed unit of work.
* :class:`JobPoolExecutor` — bounded ``ThreadPoolExecutor`` driving
  every Databricks / external-link batch read, every per-statement
  retry, every HTTP send-many fan-out.

For thousand-job streams (the Databricks SDK or
``external_links`` path), the **per-job overhead in
``JobPoolExecutor.as_completed``** is the part that adds up; for
single fire-and-forget calls, **``Job.make`` + thread start** is the
floor.

Usage::

    PYTHONPATH=src python benchmarks/bench_concurrent.py
    PYTHONPATH=src python benchmarks/bench_concurrent.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

from yggdrasil.concurrent import (
    Job,
    JobPoolExecutor,
    JobResult,
    ThreadJob,
)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _identity(x: int) -> int:
    return x


def _noop() -> None:
    return None


def _add(a: int, b: int) -> int:
    return a + b


JOB_NOARG = Job.make(_noop)
JOB_ID = Job.make(_identity, 42)
JOB_KW = Job.make(_add, 1, b=2)
RESULT_OK = JobResult(result=42, exception=None)
RESULT_ERR = JobResult(result=None, exception=RuntimeError("boom"))


# ---------------------------------------------------------------------------
# Timing helpers.
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 1000)):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
    }


def _fmt(r: dict) -> str:
    scale = 1e6
    unit = "us"
    if r["best"] < 1e-6:
        scale = 1e9
        unit = "ns"
    return (
        f"{r['label']:<58s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios — Job / JobResult lightweight primitives.
# ---------------------------------------------------------------------------


def _job_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    out.append(_time_one(
        "Job.make(func) no-arg",
        lambda: Job.make(_noop),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "Job.make(func, 42)",
        lambda: Job.make(_identity, 42),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "Job.make(func, 1, b=2)",
        lambda: Job.make(_add, 1, b=2),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "Job.run() no-arg",
        lambda: JOB_NOARG.run(),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "Job.run() one positional",
        lambda: JOB_ID.run(),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "Job.run() mixed args / kwargs",
        lambda: JOB_KW.run(),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "Job.__call__() (pure call-site)",
        lambda: JOB_ID(),
        repeat=repeat, inner=500_000,
    ))

    # JobResult — built and inspected once per completion.
    out.append(_time_one(
        "JobResult(value, None) construct",
        lambda: JobResult(result=42, exception=None),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "JobResult(None, exc) construct",
        lambda: JobResult(result=None, exception=RuntimeError("x")),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "JobResult.ok (success)",
        lambda: RESULT_OK.ok,
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "JobResult.get() (success)",
        lambda: RESULT_OK.get(),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "bool(JobResult)",
        lambda: bool(RESULT_OK),
        repeat=repeat, inner=500_000,
    ))

    return out


# ---------------------------------------------------------------------------
# Scenarios — ThreadJob (real thread start + wait).
# ---------------------------------------------------------------------------


def _thread_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # Fire + wait — captures the per-job thread-start + join cost.
    def fire_and_wait():
        return Job.make(_noop).fire_and_forget().wait()

    out.append(_time_one(
        "Job.thread().wait() round-trip noop",
        fire_and_wait,
        repeat=repeat, inner=200,
    ))

    return out


# ---------------------------------------------------------------------------
# Scenarios — JobPoolExecutor (small + large streams, ordered + unordered).
# ---------------------------------------------------------------------------


def _make_pool() -> JobPoolExecutor:
    return JobPoolExecutor(max_workers=4, max_in_flight=8)


def _drain(it):
    for _ in it:
        pass


def _pool_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # 64-job stream of trivial functions — measures dispatch + queueing
    # cost rather than the job body. ``ordered=False`` is the default
    # consumer pattern; ``ordered=True`` is what
    # ``WarehouseStatementResult._read_arrow_batches`` and other
    # ordered-streaming callers hit.
    def run_unordered(n: int) -> None:
        with _make_pool() as ex:
            _drain(ex.as_completed(
                (Job.make(_identity, i) for i in range(n)),
                ordered=False,
            ))

    def run_ordered(n: int) -> None:
        with _make_pool() as ex:
            _drain(ex.as_completed(
                (Job.make(_identity, i) for i in range(n)),
                ordered=True,
            ))

    out.append(_time_one(
        "JobPoolExecutor: 64 jobs, unordered",
        lambda: run_unordered(64),
        repeat=repeat, inner=20,
    ))
    out.append(_time_one(
        "JobPoolExecutor: 64 jobs, ordered",
        lambda: run_ordered(64),
        repeat=repeat, inner=20,
    ))
    out.append(_time_one(
        "JobPoolExecutor: 256 jobs, unordered",
        lambda: run_unordered(256),
        repeat=repeat, inner=10,
    ))
    out.append(_time_one(
        "JobPoolExecutor: 256 jobs, ordered",
        lambda: run_ordered(256),
        repeat=repeat, inner=10,
    ))

    # Reused pool — what happens when the same executor drains
    # back-to-back streams (e.g. one statement result reading 4
    # external-link windows). Submission + completion only; no
    # pool teardown per call.
    pool = _make_pool()

    def reused(n: int) -> None:
        _drain(pool.as_completed(
            (Job.make(_identity, i) for i in range(n)),
            ordered=False,
        ))

    out.append(_time_one(
        "JobPoolExecutor: 64 jobs (reused pool)",
        lambda: reused(64),
        repeat=repeat, inner=20,
    ))
    pool.shutdown(wait=True)

    return out


# ---------------------------------------------------------------------------
# Aggregator + CLI.
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    return [
        *_job_scenarios(repeat),
        *_thread_scenarios(repeat),
        *_pool_scenarios(repeat),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<58s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
