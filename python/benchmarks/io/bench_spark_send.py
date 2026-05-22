"""Benchmarks for :meth:`Session.send_many` in tabular Spark mode.

What this covers
----------------

``Session.send_many(..., as_tabular=True, spark_session=spark)`` is the
lazy :class:`Tabular`-of-Response counterpart to the iterator-returning
``send_many``: cache lookups stay on the driver, the network fetch
fans out via ``mapInArrow``, and per-chunk
:class:`yggdrasil.io.response_batch.ResponseBatch` frames are
``unionByName``-stitched into one DataFrame wrapped as a
:class:`Dataset`. The interesting numbers split between two phases:

* **Plan build** — how much work the driver does before any Spark
  action fires. Drives the ``_send_many_batches`` chunking loop,
  scatters the requests through ``spark.createDataFrame`` +
  ``repartition``, and unions the per-chunk frames. No executor work
  yet — this is the cost callers pay even if they bail out before
  collecting; small N here is what makes the tabular Spark mode
  composable as a planner step.
* **Plan + collect** — the same plan, plus the executor round-trip:
  one ``mapInArrow`` job per chunk that runs the in-process
  :class:`StubSession`'s ``_local_send`` (so no real network, but the
  Arrow encode + Spark IPC is real). Measures the per-request lifted
  cost end-to-end.

The reference points alongside:

* **Python ``send_many``** — the driver-side iterator path that the
  tabular mode replaces when callers want frame composition instead
  of one ``Response`` at a time. Helps see when the Spark mode is
  paying its own scheduler tax vs. when it's a clear win.
* **``Session._responses_to_spark(N)``** — the underlying per-chunk
  lift; the bulk of the Spark mode's plan-time cost reduces to one
  call of this per chunk. Already covered in
  ``bench_session_cache_spark.py``; we re-run a small variant here so
  the cross-bench comparison reads in one file.

Skipped cleanly when pyspark isn't installed or a local SparkSession
fails to come up.

Usage::

    PYTHONPATH=src python benchmarks/io/bench_spark_send.py
    PYTHONPATH=src python benchmarks/io/bench_spark_send.py --repeat 5 --n 64
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, List

# Spark workers unpickle the broadcast session by importing its
# ``__module__``. Defining the stub inline in this script puts it under
# ``__main__``, which workers can't resolve; instead we keep the class
# in a sibling file (``_bench_stub_session.py``) and expose ``benchmarks``
# as a namespace package by adding ``python/`` to ``sys.path`` and to
# ``PYTHONPATH`` so the worker subprocesses inherit it.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.environ["PYTHONPATH"] = (
    str(_PROJECT_ROOT)
    + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
)

from yggdrasil.io.request import PreparedRequest  # noqa: E402
from yggdrasil.io.send_config import SendConfig  # noqa: E402
from yggdrasil.io.session import Session  # noqa: E402

from benchmarks.io._bench_stub_session import _StubBenchSession  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_HOST = "https://api.example.com"


def _build_requests(count: int) -> List[PreparedRequest]:
    """Build *count* distinct :class:`PreparedRequest` objects.

    Each request goes through the standard ``prepare`` path and gets
    its lazy identity hashes warmed so the bench's timing window
    doesn't pay for the first-call hash walk.
    """
    out: List[PreparedRequest] = []
    for i in range(count):
        req = PreparedRequest.prepare(
            "GET",
            f"{_HOST}/v1/accounts/{i:05d}/transactions?page={i % 7}",
            headers={"Content-Type": "application/json"},
        )
        _ = req.public_hash, req.public_url_hash, req.partition_key
        out.append(req)
    return out


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    # Warm up: run a handful before timing to amortise JIT / Spark
    # planner cache misses. Capped at 5 because the Spark-bound scenarios
    # are slow enough that more warm-up dominates the wall time.
    for _ in range(min(inner, 5)):
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
        scale, unit = 1e9, "ns"
    elif r["best"] >= 1e-3:
        scale, unit = 1e3, "ms"
    return (
        f"{r['label']:<70s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenarios(repeat: int, n: int) -> list[dict]:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return [{
            "label": "send_many(as_tabular=True): SKIPPED (pyspark not installed)",
            "best": 0.0, "median": 0.0, "mean": 0.0,
        }]

    try:
        from yggdrasil.environ import PyEnv
        spark = PyEnv.spark_session(create=True)
        if spark is None:
            raise RuntimeError("spark_session returned None")
    except Exception as exc:
        return [{
            "label": f"send_many(as_tabular=True): SKIPPED ({type(exc).__name__}: {exc})",
            "best": 0.0, "median": 0.0, "mean": 0.0,
        }]

    out: list[dict] = []
    session = _StubBenchSession()
    one_req = _build_requests(1)[0]
    requests = _build_requests(n)

    # ---- send_many(as_tabular=True, spark_session=spark): single request ----
    out.append(_time_one(
        "send_many([req], as_tabular=True) plan build (no .count())",
        lambda: session.send_many(
            iter([one_req]), as_tabular=True, spark_session=spark,
        ),
        repeat=repeat, inner=5,
    ))
    out.append(_time_one(
        "send_many([req], as_tabular=True).frame.count() (plan + 1-row collect)",
        lambda: session.send_many(
            iter([one_req]), as_tabular=True, spark_session=spark,
        ).frame.count(),
        repeat=repeat, inner=5,
    ))

    # ---- send_many(as_tabular=True): plan build only (no Spark action) ----
    out.append(_time_one(
        f"send_many({n}, as_tabular=True) plan build (no .count())",
        lambda: session.send_many(
            iter(requests), as_tabular=True, spark_session=spark,
        ),
        repeat=repeat, inner=5,
    ))

    # ---- send_many(as_tabular=True): plan + collect, single chunk ----
    out.append(_time_one(
        f"send_many({n}, as_tabular=True).frame.count() — one chunk",
        lambda: session.send_many(
            iter(requests),
            as_tabular=True,
            batch_size=n * 2,  # ensures a single chunk
            spark_session=spark,
        ).frame.count(),
        repeat=repeat, inner=3,
    ))

    # ---- send_many(as_tabular=True): plan + collect, multi-chunk union ----
    if n >= 4:
        chunk = max(1, n // 4)
        out.append(_time_one(
            f"send_many({n}, as_tabular=True).frame.count() — {n // chunk} chunks of {chunk}",
            lambda: session.send_many(
                iter(requests),
                as_tabular=True,
                batch_size=chunk,
                spark_session=spark,
            ).frame.count(),
            repeat=repeat, inner=3,
        ))

    # ---- Python send_many reference (no Spark) ----
    out.append(_time_one(
        f"send_many({n}) — Python path, materialised to list",
        lambda: list(session.send_many(iter(requests))),
        repeat=repeat, inner=20,
    ))

    # ---- Underlying lift (one chunk's worth of responses → SparkDataFrame) ----
    canned_responses = [
        session._local_send(req, SendConfig.check_arg(None))
        for req in requests
    ]
    out.append(_time_one(
        f"Session._responses_to_spark({n}) — per-chunk lift",
        lambda: Session._responses_to_spark(canned_responses, spark),
        repeat=repeat, inner=10,
    ))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    ap.add_argument("--n", type=int, default=32,
                    help="Number of requests for the multi-request scenarios.")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}  n={args.n}")
    print(f"# {'label':<70s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    for row in scenarios(args.repeat, args.n):
        print(_fmt(row))


if __name__ == "__main__":
    main()
