"""Sustained sequential flood of the Databricks SQL Statement Execution API.

Drives many back-to-back ``SELECT <i>`` statements through
:class:`SQLEngine.execute` against a live workspace, then asserts that
every single call returned its exact expected payload. The point is
twofold:

1. **Observe behavior under sustained load.** A real workload eventually
   bumps into per-warehouse quotas, statement-execution backpressure,
   and the SDK's transparent 429 / retry-after handling. Capturing a
   per-call latency timeline lets us see whether those policies absorb
   the burst (slower tail, no exceptions) or surface to the caller
   (failed call, full assertion diagnostic with iteration index).

2. **Pin the correctness contract under that load.** Each statement
   carries a unique value so the assertion isn't accidentally
   satisfied by a cached or duplicated response — under any latency
   regime, the warehouse must return *this* call's bound literals.

Sequential (no client-side concurrency) on purpose: this isolates the
warehouse-side rate path from any local thread-pool effects. For
concurrent-write stress, see :mod:`test_engine_integration.TestSQLConcurrentWrites`.

Knobs (env vars, all optional):

- ``DATABRICKS_FLOOD_ITERATIONS`` — how many statements to send.
  Default 200. Increase locally to push harder; CI keeps the default
  so the test stays bounded.
- ``DATABRICKS_FLOOD_MAX_DURATION`` — wall-clock cap in seconds.
  Default 120s. The loop stops early when exceeded (with a logged
  summary) so a slow warehouse doesn't hang the suite for hours.
"""
from __future__ import annotations

import logging
import os
import statistics
import time
from typing import ClassVar

from yggdrasil.databricks.sql.engine import SQLEngine

from .. import DatabricksIntegrationCase


__all__ = ["TestSQLWarehouseFlood"]


logger = logging.getLogger("yggdrasil.tests.warehouse_flood")


def _env_positive_int(name: str, default: int) -> int:
    """Read a positive-int env var with a typed error on garbage input."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a positive integer; got {raw!r}. "
            f"Unset it to use the default ({default})."
        ) from exc
    if value <= 0:
        raise ValueError(
            f"{name} must be > 0; got {value}. "
            f"Unset it to use the default ({default})."
        )
    return value


def _percentile(samples: list[float], p: float) -> float:
    """Inclusive percentile on a copy-sorted sample list.

    Hand-rolled instead of importing ``numpy`` so the test stays inside
    the pyarrow-only base install.
    """
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    # Rank-1 inclusive: position = p * (n - 1).
    pos = p * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


class TestSQLWarehouseFlood(DatabricksIntegrationCase):
    """Sustained sequential burst against the Statement Execution API."""

    engine: ClassVar[SQLEngine]
    ITERATIONS: ClassVar[int]
    MAX_DURATION: ClassVar[float]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # No catalog/schema scope: the flood runs ``SELECT <literals>``
        # only, so the engine never needs to resolve unqualified
        # identifiers against a namespace.
        cls.engine = cls.client.sql
        cls.ITERATIONS = _env_positive_int("DATABRICKS_FLOOD_ITERATIONS", 200)
        cls.MAX_DURATION = float(
            _env_positive_int("DATABRICKS_FLOOD_MAX_DURATION", 120)
        )

    # ------------------------------------------------------------------
    # Core flood
    # ------------------------------------------------------------------

    def test_select_literal_flood_returns_expected_payload(self) -> None:
        """Fire N back-to-back ``SELECT <i> AS i, '<tag>' AS t`` statements.

        Hard contract: every single iteration's result must equal
        ``[{"i": i, "t": f"tag-{i}"}]`` — the literals are unique per
        call so the assertion can't be satisfied by a stale cached
        response.

        On the first failure the iteration index and the partial
        latency summary (so the reader can see *when* the warehouse
        started misbehaving) are surfaced in the assertion message.
        """
        latencies: list[float] = []
        wall_start = time.perf_counter()
        completed = 0
        stopped_early = False

        for i in range(self.ITERATIONS):
            wall_elapsed = time.perf_counter() - wall_start
            if wall_elapsed > self.MAX_DURATION:
                stopped_early = True
                logger.warning(
                    "Flood stopped early at iter=%d after %.1fs > "
                    "DATABRICKS_FLOOD_MAX_DURATION=%.1fs",
                    i, wall_elapsed, self.MAX_DURATION,
                )
                break

            call_start = time.perf_counter()
            try:
                result = self.engine.execute(
                    f"SELECT CAST({i} AS BIGINT) AS i, 'tag-{i}' AS t"
                )
                rows = result.to_arrow_table().to_pylist()
            except Exception as exc:
                # Anything reaching here means the SDK's transparent
                # retry loop didn't absorb the failure — surface the
                # exact iteration and the timeline so far, then re-raise
                # with the original traceback intact.
                self._log_summary(
                    latencies, completed, wall_start,
                    note=f"failed at iter={i} with {type(exc).__name__}",
                )
                raise AssertionError(
                    f"flood iter={i}/{self.ITERATIONS} raised "
                    f"{type(exc).__name__}: {exc}. "
                    f"completed={completed} before failure. "
                    f"see WARN log above for latency stats."
                ) from exc
            latencies.append(time.perf_counter() - call_start)

            # Hard correctness: the bound literals must come back exactly.
            # A failed assertion here gives the reader the iteration
            # index, the raw row, and (via the summary log emitted in
            # tearDown via this branch's escape) the latency stats.
            if rows != [{"i": i, "t": f"tag-{i}"}]:
                self._log_summary(
                    latencies, completed, wall_start,
                    note=f"payload mismatch at iter={i}",
                )
                self.fail(
                    f"flood iter={i}/{self.ITERATIONS} returned {rows!r}; "
                    f"expected [{{'i': {i}, 't': 'tag-{i}'}}]. "
                    f"completed={completed} before mismatch."
                )

            completed += 1

        self._log_summary(
            latencies, completed, wall_start,
            note="stopped early by MAX_DURATION" if stopped_early else "ok",
        )

        # At least one statement has to have run — otherwise the env
        # was misconfigured (zero iterations, or MAX_DURATION ≤ 0
        # somehow slipped past the env-var validator).
        self.assertGreater(
            completed, 0,
            "flood completed zero iterations — check "
            "DATABRICKS_FLOOD_ITERATIONS / DATABRICKS_FLOOD_MAX_DURATION",
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _log_summary(
        latencies: list[float],
        completed: int,
        wall_start: float,
        *,
        note: str,
    ) -> None:
        """Emit a one-line summary of the flood at the WARN level.

        WARN (not INFO) so the line shows up even when the test run is
        configured to filter INFO chatter — flood reports are exactly
        the kind of "I want to see this every run" observational
        output the suite is written for.
        """
        wall = time.perf_counter() - wall_start
        if latencies:
            best = min(latencies)
            median = statistics.median(latencies)
            p95 = _percentile(latencies, 0.95)
            p99 = _percentile(latencies, 0.99)
            worst = max(latencies)
        else:
            best = median = p95 = p99 = worst = float("nan")
        throughput = completed / wall if wall > 0 else float("nan")
        logger.warning(
            "warehouse-flood %s: completed=%d wall=%.2fs throughput=%.2f/s "
            "best=%.3fs median=%.3fs p95=%.3fs p99=%.3fs worst=%.3fs",
            note, completed, wall, throughput,
            best, median, p95, p99, worst,
        )
