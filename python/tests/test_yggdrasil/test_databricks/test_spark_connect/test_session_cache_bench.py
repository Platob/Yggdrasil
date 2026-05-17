"""Live benchmarks for :meth:`DatabricksClient.spark` session reuse.

Each test re-enters one of the caching layers that sits between
``client.spark()`` and the Spark Connect gRPC channel — the
``getActiveSession()`` short-circuit, the
:class:`DatabricksSparkStatementExecutor` singleton, the
``_bind_spark_session`` rebind path — and quotes ``best`` / ``median``
wall-clock timings so a regression in any of those layers shows up as
"warm call now costs N ms" instead of being lost in the noise of a full
gRPC round trip.

Skipped unless ``DATABRICKS_HOST`` (and the matching credentials) are
exported and ``databricks-connect`` is importable — see
:class:`DatabricksIntegrationCase`. The base class already calls
``client.spark()`` once in ``setUpClass``, so every measurement here
runs against a warm in-process session: the numbers track *cache reuse*
cost, not first-handshake cost.
"""

from __future__ import annotations

import statistics
import time
import unittest
from typing import Callable

import pytest

from yggdrasil.databricks.sql.spark_executor import (
    DatabricksSparkStatementExecutor,
)

from .. import DatabricksIntegrationCase

__all__ = ["TestSparkSessionCacheBench"]


def _measure(fn: Callable[[], object], repeats: int) -> tuple[float, float]:
    """Run *fn* *repeats* times and return ``(best, median)`` seconds."""
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return min(samples), statistics.median(samples)


class TestSparkSessionCacheBench(DatabricksIntegrationCase):

    REPEATS = 20

    @classmethod
    def setUpClass(cls) -> None:
        pytest.importorskip("databricks.connect")
        super().setUpClass()
        if cls.spark is None:
            raise unittest.SkipTest(
                "client.spark() returned no session — skipping reuse benchmarks."
            )

    # ------------------------------------------------------------------
    # client.spark() — warm reuse via getActiveSession()
    # ------------------------------------------------------------------

    def test_client_spark_reuse_is_cheap(self) -> None:
        """A second ``client.spark()`` call must short-circuit on the
        in-process active session — no ``DatabricksSession.builder``
        handshake, no dependency classification, no wheel republish."""
        warm = self.client.spark()
        assert warm is self.spark, (
            "client.spark() must return the same active session on reuse"
        )

        best, median = _measure(self.client.spark, self.REPEATS)
        print(
            f"\n[bench] client.spark() reuse  best={best * 1e3:.2f}ms  "
            f"median={median * 1e3:.2f}ms  (n={self.REPEATS})"
        )
        # Pure ``getActiveSession()`` + ``_bind_spark_session`` is a
        # handful of attribute lookups; anything beyond ~50ms means a
        # round trip leaked back into the warm path.
        assert median < 0.05, (
            f"client.spark() reuse median={median * 1e3:.2f}ms exceeds 50ms "
            f"budget — the getActiveSession() short-circuit likely regressed"
        )

    # ------------------------------------------------------------------
    # SparkExecutor singleton — same (cls, client) → same executor
    # ------------------------------------------------------------------

    def test_spark_executor_singleton_caches_session(self) -> None:
        """``DatabricksSparkStatementExecutor`` is singleton-cached per
        ``(cls, client)`` and pins the resolved session on the instance;
        ``resolve_session`` must hand back the cached slot without a
        second ``getActiveSession()`` probe."""
        DatabricksSparkStatementExecutor._INSTANCES.clear()

        first = DatabricksSparkStatementExecutor(client=self.client)
        second = DatabricksSparkStatementExecutor(client=self.client)
        assert first is second, "executor must be singleton-cached per client"

        # Prime the cached session slot once.
        first.resolve_session(create=True)

        best, median = _measure(
            lambda: first.resolve_session(create=True), self.REPEATS,
        )
        print(
            f"\n[bench] executor.resolve_session() warm  best={best * 1e6:.1f}us  "
            f"median={median * 1e6:.1f}us  (n={self.REPEATS})"
        )
        # Warm slot is a single attribute read — sub-millisecond easily.
        assert median < 0.001, (
            f"resolve_session warm median={median * 1e6:.1f}us exceeds 1ms — "
            f"the executor cache likely regressed"
        )

    # ------------------------------------------------------------------
    # End-to-end — repeated SELECT 1 over the active session
    # ------------------------------------------------------------------

    def test_select_one_roundtrip_is_stable(self) -> None:
        """A trivial ``SELECT 1`` over the warm session is the cheapest
        real round trip available; quoting its timing pins the floor a
        regression has to climb above to be visible."""
        spark = self.spark

        def _roundtrip() -> None:
            rows = spark.sql("SELECT 1 AS one").collect()
            assert rows and rows[0]["one"] == 1

        # Discard the first call — JIT / first-statement gRPC channel
        # warm-up isn't representative of steady-state cost.
        _roundtrip()

        best, median = _measure(_roundtrip, self.REPEATS)
        print(
            f"\n[bench] spark.sql('SELECT 1').collect()  best={best * 1e3:.2f}ms  "
            f"median={median * 1e3:.2f}ms  (n={self.REPEATS})"
        )
