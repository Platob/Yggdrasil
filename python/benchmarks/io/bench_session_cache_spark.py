"""Spark-flavoured benchmarks for :class:`yggdrasil.io.session.Session` cache.

What this covers
----------------

The session's cache pipeline has two flavours: the driver-side
Python path (responses round-trip through arrow batches and end up as
:class:`yggdrasil.io.response.Response` objects) and the Spark path
(responses live as :class:`pyspark.sql.DataFrame` rows so the rest of
the bucket can stay frame-resident). This bench targets the second:

* :meth:`Session._responses_to_spark` — the lift from a ``list[Response]``
  to a :class:`SparkDataFrame`. Empty / 1 / 16 / 128 responses cover
  the per-call overhead, the small-batch case, and the large-batch
  case where ``pa.Table.from_batches`` plus ``createDataFrame`` start
  to dominate.
* :meth:`Session._cached_empty_spark_frame` — fast-path on the
  zero-response branch; reuses one cached empty :class:`SparkDataFrame`
  per ``SparkSession`` instead of rebuilding the schema each call.

Real remote-cache scenarios
(``_split_remote_cache_spark`` / ``_lookup_remote_table_spark``) need
a live Databricks SQL endpoint and aren't covered here — they're the
``integration`` marker's territory. The driver-side lift, by
contrast, runs against a local SparkSession in seconds.

Skipped cleanly when pyspark isn't installed or a local SparkSession
fails to come up — the bench prints a single skip row in that case.

Usage::

    PYTHONPATH=src python benchmarks/io/bench_session_cache_spark.py
    PYTHONPATH=src python benchmarks/io/bench_session_cache_spark.py --repeat 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time
from typing import Callable, List

from yggdrasil.io.memory import Memory
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response
from yggdrasil.io.session import Session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_HOST = "https://api.example.com"


def _build_responses(count: int) -> List[Response]:
    """Build *count* distinct :class:`Response` objects keyed on URL.

    Each response carries a small JSON-looking payload so the arrow
    batch encoding has real bytes to walk; the receive timestamp is
    fixed so the rows hash deterministically.
    """
    received = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    out: List[Response] = []
    for i in range(count):
        req = PreparedRequest.prepare(
            "GET",
            f"{_HOST}/v1/accounts/{i:05d}/transactions?page={i % 7}",
            headers={"Content-Type": "application/json"},
        )
        # Warm the lazy identity hashes — without this the first
        # ``to_arrow_batch`` call would pay the hash walk inside the
        # bench's timing window.
        _ = req.public_hash, req.partition_key
        resp = Response(
            request=req,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=Memory(binary=b'{"ok":true,"rows":[1,2,3]}'),
            received_at=received,
        )
        _ = resp.hash, resp.public_hash, resp.partition_key
        out.append(resp)
    return out


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 50)):
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
        f"{r['label']:<62s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return [{
            "label": "session-cache spark: SKIPPED (pyspark not installed)",
            "best": 0.0, "median": 0.0, "mean": 0.0,
        }]

    try:
        from yggdrasil.environ import PyEnv
        spark = PyEnv.spark_session(create=True)
        if spark is None:
            raise RuntimeError("spark_session returned None")
    except Exception as exc:
        return [{
            "label": f"session-cache spark: SKIPPED ({type(exc).__name__}: {exc})",
            "best": 0.0, "median": 0.0, "mean": 0.0,
        }]

    out: list[dict] = []

    one_resp = _build_responses(1)
    sixteen_resp = _build_responses(16)
    big_resp = _build_responses(128)

    # ``_responses_to_spark`` is a private classmethod on Session.
    # Reach for it via the public ``Session`` symbol — the test suite
    # already does this, and the private call signature is stable.
    lift = Session._responses_to_spark

    out.append(_time_one(
        "Session._responses_to_spark([]) empty (cached empty)",
        lambda: lift([], spark),
        repeat=repeat, inner=2_000,
    ))
    out.append(_time_one(
        "Session._responses_to_spark(1)",
        lambda: lift(one_resp, spark),
        repeat=repeat, inner=200,
    ))
    out.append(_time_one(
        "Session._responses_to_spark(16)",
        lambda: lift(sixteen_resp, spark),
        repeat=repeat, inner=100,
    ))
    out.append(_time_one(
        "Session._responses_to_spark(128)",
        lambda: lift(big_resp, spark),
        repeat=repeat, inner=20,
    ))

    # ``to_arrow_batch`` is the per-response cost ``_responses_to_spark``
    # pays before ``Table.from_batches`` + ``createDataFrame`` even
    # start. Capture it standalone so the lift cost reads as
    # "per-row encode + N-arrow-table glue + spark createDataFrame."
    out.append(_time_one(
        "Response.to_arrow_batch(parse=False)",
        lambda: one_resp[0].to_arrow_batch(parse=False),
        repeat=repeat, inner=2_000,
    ))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<62s}  {'best':>15s}  {'median':>18s}  {'mean':>15s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
