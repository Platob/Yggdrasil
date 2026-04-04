"""
Integration tests for ExecutionContext auto-close / reaper behaviour.

Structure mirrors test_mongoengine_data.py:
  - ``OfflineReaperIntegrationTest``  — fully offline with real reaper thread,
    mocked Databricks SDK.  Always runs in CI.
  - ``DatabricksLiveIntegrationTest`` — needs real DATABRICKS_HOST +
    DATABRICKS_TOKEN.  Skipped automatically when credentials are absent.

Run all tests:
    pytest tests/test_yggdrasil/test_databricks/compute/test_execution_context_integration.py -v

Run only offline tests:
    pytest -k "Offline" ...

Run only live tests (requires credentials):
    DATABRICKS_TOKEN=<tok> pytest -k "Live" ...
"""

from __future__ import annotations

import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from databricks.sdk.service.compute import Language

import yggdrasil.databricks.compute.execution_context as _ec_mod
from yggdrasil.databricks.compute.execution_context import (
    ContextPoolKey,
    ExecutionContext,
    close_all_pooled_contexts,
    _ensure_reaper_running,
)

# ---------------------------------------------------------------------------
# Skip guard for live tests
# ---------------------------------------------------------------------------

_HAVE_DATABRICKS_CREDS = bool(
    os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN")
)
_SKIP_LIVE = unittest.skipUnless(
    _HAVE_DATABRICKS_CREDS,
    "Live Databricks tests require DATABRICKS_HOST + DATABRICKS_TOKEN env vars.",
)


# ---------------------------------------------------------------------------
# Shared mock helpers  (same style as test_execution_context.py)
# ---------------------------------------------------------------------------

def _make_cluster(cluster_id: str = "cls-integ") -> MagicMock:
    cluster = MagicMock(name="Cluster")
    cluster.cluster_id = cluster_id
    url_mock = MagicMock(name="URL")
    url_mock.with_query_items.return_value = url_mock
    url_mock.to_string.return_value = f"https://dbc.example.com/?cluster={cluster_id}"
    cluster.url.return_value = url_mock
    cluster.is_in_databricks_environment.return_value = False
    return cluster


def _ws(cluster: MagicMock):
    return cluster.client.workspace_client.return_value.command_execution


def _ctx_response(ctx_id: str) -> MagicMock:
    r = MagicMock()
    r.response.id = ctx_id
    return r


# ---------------------------------------------------------------------------
# Base class: saves/restores module-level pool + reaper state
# ---------------------------------------------------------------------------

class _ReaperBase(unittest.TestCase):
    """Save and restore global state so tests are fully isolated."""

    def setUp(self):
        self._orig_pool = dict(_ec_mod._CONTEXT_POOL)
        _ec_mod._CONTEXT_POOL.clear()

        _ec_mod._REAPER_STOP.set()
        if _ec_mod._REAPER_THREAD is not None and _ec_mod._REAPER_THREAD.is_alive():
            _ec_mod._REAPER_THREAD.join(timeout=3.0)
        _ec_mod._REAPER_THREAD = None
        _ec_mod._REAPER_STOP.clear()

        self._orig_interval = _ec_mod._REAPER_INTERVAL
        _ec_mod._REAPER_INTERVAL = 0.05      # 50 ms — fast for tests

    def tearDown(self):
        _ec_mod._REAPER_STOP.set()
        if _ec_mod._REAPER_THREAD is not None and _ec_mod._REAPER_THREAD.is_alive():
            _ec_mod._REAPER_THREAD.join(timeout=3.0)
        _ec_mod._REAPER_THREAD = None
        _ec_mod._REAPER_STOP.clear()

        _ec_mod._CONTEXT_POOL.clear()
        _ec_mod._CONTEXT_POOL.update(self._orig_pool)
        _ec_mod._REAPER_INTERVAL = self._orig_interval


# ===========================================================================
# Offline integration tests — real reaper thread, mocked Databricks SDK
# ===========================================================================

class OfflineReaperIntegrationTest(_ReaperBase):
    """
    End-to-end reaper tests that are always executable (no credentials needed).
    Uses mocked SDK calls but a *real* background thread so the full timing
    and concurrency path is exercised.

    Mirrors test_mongoengine_data.MongoCase in style — each test is a named
    scenario that is self-contained and repeatable.
    """

    # ------------------------------------------------------------------
    # Scenario 1 — basic create → idle → auto-close
    # ------------------------------------------------------------------

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_scenario_idle_context_is_auto_closed(self, mock_unreg, mock_reg):
        """
        A context that sits idle beyond its ``close_after`` window must be
        closed and removed from the pool by the background reaper.
        """
        cluster = _make_cluster("cls-s1")
        _ws(cluster).create.return_value = _ctx_response("ctx-s1")

        ctx = ExecutionContext.get_or_create(cluster=cluster, close_after=0.05)
        pool_key = ExecutionContext._pool_key(
            cluster_id="cls-s1", language=Language.PYTHON, context_key=ctx.context_key
        )

        # Simulate the context being unused for > close_after
        ctx._last_used_at = time.time() - 1.0

        _ensure_reaper_running()
        time.sleep(0.4)

        self.assertNotIn(pool_key, _ec_mod._CONTEXT_POOL, "Idle context should have been evicted")
        self.assertFalse(ctx.context_id, "context_id should be cleared after auto-close")

    # ------------------------------------------------------------------
    # Scenario 2 — active context survives
    # ------------------------------------------------------------------

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_scenario_active_context_survives_reaper(self, mock_unreg, mock_reg):
        """
        A context that is touched before the timeout window closes must NOT
        be evicted by the reaper.
        """
        cluster = _make_cluster("cls-s2")
        _ws(cluster).create.return_value = _ctx_response("ctx-s2")

        ctx = ExecutionContext.get_or_create(cluster=cluster, close_after=60.0)
        pool_key = ExecutionContext._pool_key(
            cluster_id="cls-s2", language=Language.PYTHON, context_key=ctx.context_key
        )

        ctx.touch()   # reset idle timer

        _ensure_reaper_running()
        time.sleep(0.3)

        self.assertIn(pool_key, _ec_mod._CONTEXT_POOL, "Active context must remain pooled")
        self.assertTrue(ctx.context_id, "context_id should still be set")

    # ------------------------------------------------------------------
    # Scenario 3 — close_after=None pins the context forever
    # ------------------------------------------------------------------

    def test_scenario_pinned_context_never_evicted(self):
        """
        ``close_after=None`` must prevent eviction regardless of idle duration.
        """
        cluster = _make_cluster("cls-s3")
        ctx = ExecutionContext(
            cluster=cluster, context_id="ctx-pinned",
            context_key="pinned", close_after=None,
        )
        ctx._last_used_at = time.time() - 999_999.0
        key = ContextPoolKey(cluster_id="cls-s3", language="PYTHON", context_key="pinned")
        _ec_mod._CONTEXT_POOL[key] = ctx

        _ensure_reaper_running()
        time.sleep(0.3)

        self.assertIn(key, _ec_mod._CONTEXT_POOL)
        self.assertEqual(ctx.context_id, "ctx-pinned")

    # ------------------------------------------------------------------
    # Scenario 4 — multiple contexts with different timeouts
    # ------------------------------------------------------------------

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_scenario_mixed_timeouts_partial_eviction(self, mock_unreg, mock_reg):
        """
        With three contexts — one expired, one fresh, one pinned —
        only the expired one is evicted.
        """
        cluster = _make_cluster("cls-s4")

        # Expired (10 ms timeout, 1 s idle)
        _ws(cluster).create.return_value = _ctx_response("ctx-expired")
        ctx_exp = ExecutionContext.get_or_create(
            cluster=cluster, close_after=0.01, language=Language.PYTHON, context_key="exp"
        )
        ctx_exp._last_used_at = time.time() - 1.0

        # Fresh (3600 s timeout)
        _ws(cluster).create.return_value = _ctx_response("ctx-fresh")
        ctx_fresh = ExecutionContext.get_or_create(
            cluster=cluster, close_after=3600.0, language=Language.PYTHON, context_key="fresh"
        )
        ctx_fresh.touch()

        # Pinned (no timeout)
        ctx_pinned = ExecutionContext(
            cluster=cluster, context_id="ctx-pinned",
            context_key="pinned", close_after=None,
        )
        ctx_pinned._last_used_at = time.time() - 999_999.0
        key_pinned = ContextPoolKey(cluster_id="cls-s4", language="PYTHON", context_key="pinned")
        _ec_mod._CONTEXT_POOL[key_pinned] = ctx_pinned

        key_exp = ExecutionContext._pool_key(
            cluster_id="cls-s4", language=Language.PYTHON, context_key="exp"
        )
        key_fresh = ExecutionContext._pool_key(
            cluster_id="cls-s4", language=Language.PYTHON, context_key="fresh"
        )

        _ensure_reaper_running()
        time.sleep(0.4)

        self.assertNotIn(key_exp, _ec_mod._CONTEXT_POOL, "Expired context must be evicted")
        self.assertIn(key_fresh, _ec_mod._CONTEXT_POOL, "Fresh context must survive")
        self.assertIn(key_pinned, _ec_mod._CONTEXT_POOL, "Pinned context must survive")

    # ------------------------------------------------------------------
    # Scenario 5 — pool empties when all contexts expire
    # ------------------------------------------------------------------

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_scenario_all_contexts_evicted_pool_becomes_empty(self, mock_unreg, mock_reg):
        """When every pooled context is idle, the pool must drain to zero."""
        cluster = _make_cluster("cls-s5")

        for i in range(4):
            _ws(cluster).create.return_value = _ctx_response(f"ctx-s5-{i}")
            ctx = ExecutionContext.get_or_create(
                cluster=cluster, close_after=0.01,
                language=Language.PYTHON, context_key=f"s5k{i}"
            )
            ctx._last_used_at = time.time() - 1.0

        _ensure_reaper_running()
        time.sleep(0.4)

        self.assertEqual(len(_ec_mod._CONTEXT_POOL), 0, "Pool should be completely empty")

    # ------------------------------------------------------------------
    # Scenario 6 — get_or_create re-opens after eviction
    # ------------------------------------------------------------------

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_scenario_get_or_create_reopens_evicted_context(self, mock_unreg, mock_reg):
        """
        After the reaper evicts a context, the next get_or_create call must
        create a brand-new execution context on the cluster.
        """
        cluster = _make_cluster("cls-s6")
        _ws(cluster).create.return_value = _ctx_response("ctx-orig")

        ctx_orig = ExecutionContext.get_or_create(cluster=cluster, close_after=0.05)
        ctx_orig._last_used_at = time.time() - 1.0

        _ensure_reaper_running()
        time.sleep(0.4)
        self.assertFalse(ctx_orig.context_id, "Original must have been evicted")

        # New request — must create a fresh context
        _ws(cluster).create.return_value = _ctx_response("ctx-new")
        ctx_new = ExecutionContext.get_or_create(cluster=cluster, close_after=1800.0)

        self.assertEqual(ctx_new.context_id, "ctx-new")

    # ------------------------------------------------------------------
    # Scenario 7 — close_all_pooled_contexts clears the pool
    # ------------------------------------------------------------------

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_scenario_close_all_pooled_drains_pool(self, mock_unreg, mock_reg):
        """``close_all_pooled_contexts()`` must close every context and clear pool."""
        cluster = _make_cluster("cls-s7")
        contexts = []
        for i in range(3):
            _ws(cluster).create.return_value = _ctx_response(f"ctx-s7-{i}")
            ctx = ExecutionContext.get_or_create(
                cluster=cluster, close_after=1800.0,
                language=Language.PYTHON, context_key=f"s7k{i}"
            )
            contexts.append(ctx)

        close_all_pooled_contexts()

        self.assertEqual(len(_ec_mod._CONTEXT_POOL), 0)
        for ctx in contexts:
            self.assertFalse(ctx.context_id, "Each context should be closed")

    # ------------------------------------------------------------------
    # Scenario 8 — concurrent touch() races reaper safely
    # ------------------------------------------------------------------

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_scenario_concurrent_touch_prevents_eviction(self, mock_unreg, mock_reg):
        """
        A thread that repeatedly touches the context must prevent it from
        being evicted even with a very short close_after window.
        """
        cluster = _make_cluster("cls-s8")
        _ws(cluster).create.return_value = _ctx_response("ctx-s8")

        ctx = ExecutionContext.get_or_create(cluster=cluster, close_after=0.1)
        pool_key = ExecutionContext._pool_key(
            cluster_id="cls-s8", language=Language.PYTHON, context_key=ctx.context_key
        )

        stop_flag = threading.Event()

        def keep_alive():
            while not stop_flag.is_set():
                ctx.touch()
                time.sleep(0.02)

        keeper = threading.Thread(target=keep_alive, daemon=True)
        keeper.start()

        _ensure_reaper_running()
        time.sleep(0.4)

        stop_flag.set()
        keeper.join(timeout=1.0)

        # Should still be in pool because keep_alive kept touching it
        self.assertIn(pool_key, _ec_mod._CONTEXT_POOL)

    # ------------------------------------------------------------------
    # Scenario 9 — reaper thread is a daemon (won't block interpreter exit)
    # ------------------------------------------------------------------

    def test_scenario_reaper_thread_is_daemon(self):
        _ensure_reaper_running()
        self.assertTrue(
            _ec_mod._REAPER_THREAD.daemon,
            "Reaper thread must be a daemon so it never blocks interpreter shutdown",
        )

    # ------------------------------------------------------------------
    # Scenario 10 — reaper recovers from close() errors
    # ------------------------------------------------------------------

    def test_scenario_reaper_survives_close_error(self):
        """
        If ctx.close() raises, the reaper must log and continue — it must not
        crash and must still process subsequent evictions.
        """
        cluster = _make_cluster("cls-s10")

        # Context that raises on close
        ctx_bad = ExecutionContext(
            cluster=cluster, context_id="ctx-bad",
            context_key="bad", close_after=0.01,
        )
        ctx_bad._last_used_at = time.time() - 1.0

        # Context that closes cleanly
        ctx_good = ExecutionContext(
            cluster=cluster, context_id="ctx-good",
            context_key="good", close_after=0.01,
        )
        ctx_good._last_used_at = time.time() - 1.0

        key_bad = ContextPoolKey(cluster_id="cls-s10", language="PYTHON", context_key="bad")
        key_good = ContextPoolKey(cluster_id="cls-s10", language="PYTHON", context_key="good")
        _ec_mod._CONTEXT_POOL[key_bad] = ctx_bad
        _ec_mod._CONTEXT_POOL[key_good] = ctx_good

        with patch.object(ctx_bad, "close", side_effect=RuntimeError("network gone")):
            _ensure_reaper_running()
            time.sleep(0.4)

        # Both should be evicted from the pool (pool.pop happens before close())
        self.assertNotIn(key_bad, _ec_mod._CONTEXT_POOL)
        self.assertNotIn(key_good, _ec_mod._CONTEXT_POOL)
        # The good one should have been closed
        self.assertFalse(ctx_good.context_id)


# ===========================================================================
# Live integration tests — require real Databricks credentials
# ===========================================================================

@_SKIP_LIVE
class DatabricksLiveIntegrationTest(unittest.TestCase):
    """
    Tests that actually connect to a Databricks workspace.

    Set environment variables before running:
        export DATABRICKS_HOST=https://<workspace>.azuredatabricks.net
        export DATABRICKS_TOKEN=<personal-access-token>

    A running all-purpose cluster must exist (the first available one is used).
    """

    @classmethod
    def setUpClass(cls):
        from yggdrasil.databricks import DatabricksClient
        cls.client = DatabricksClient().connect()
        cls.cluster = cls.client.compute.clusters.all_purpose_cluster()

    def setUp(self):
        # Save reaper state
        self._orig_pool = dict(_ec_mod._CONTEXT_POOL)
        _ec_mod._CONTEXT_POOL.clear()
        _ec_mod._REAPER_STOP.set()
        if _ec_mod._REAPER_THREAD is not None and _ec_mod._REAPER_THREAD.is_alive():
            _ec_mod._REAPER_THREAD.join(timeout=5.0)
        _ec_mod._REAPER_THREAD = None
        _ec_mod._REAPER_STOP.clear()
        self._orig_interval = _ec_mod._REAPER_INTERVAL

    def tearDown(self):
        # Close all live contexts and restore module state
        close_all_pooled_contexts()
        _ec_mod._REAPER_STOP.set()
        if _ec_mod._REAPER_THREAD is not None and _ec_mod._REAPER_THREAD.is_alive():
            _ec_mod._REAPER_THREAD.join(timeout=5.0)
        _ec_mod._REAPER_THREAD = None
        _ec_mod._REAPER_STOP.clear()
        _ec_mod._CONTEXT_POOL.clear()
        _ec_mod._CONTEXT_POOL.update(self._orig_pool)
        _ec_mod._REAPER_INTERVAL = self._orig_interval

    def test_live_create_and_close(self):
        """Open a real context, verify it has a non-empty context_id, then close it."""
        ctx = ExecutionContext.get_or_create(
            cluster=self.cluster, close_after=1800.0
        )
        self.assertTrue(ctx.context_id, "Real context_id must be non-empty")
        ctx.close(wait=True, raise_error=True)
        self.assertFalse(ctx.context_id)

    def test_live_pool_reuses_context(self):
        """Two get_or_create calls on the same cluster must return the same object."""
        ctx1 = ExecutionContext.get_or_create(cluster=self.cluster)
        ctx2 = ExecutionContext.get_or_create(cluster=self.cluster)
        self.assertIs(ctx1, ctx2)

    def test_live_reaper_auto_closes_idle_context(self):
        """
        With a tiny close_after window, the background reaper must auto-close
        a real context without manual intervention.
        """
        _ec_mod._REAPER_INTERVAL = 1.0       # 1-second scan for live test

        ctx = ExecutionContext.get_or_create(
            cluster=self.cluster, close_after=2.0    # 2-second idle window
        )
        pool_key = ExecutionContext._pool_key(
            cluster_id=self.cluster.cluster_id,
            language=Language.PYTHON,
            context_key=ctx.context_key,
        )

        # Do NOT touch the context — let it go idle
        time.sleep(5.0)   # wait for reaper to fire (at least twice)

        self.assertNotIn(pool_key, _ec_mod._CONTEXT_POOL)
        self.assertFalse(ctx.context_id, "Reaper should have closed the live context")

    def test_live_context_manager(self):
        """Context manager must open and close a real remote context."""
        with ExecutionContext(cluster=self.cluster, temporary=True) as ctx:
            self.assertTrue(ctx.context_id)

    def test_live_close_after_none_stays_open(self):
        """close_after=None → reaper must leave the context alone."""
        _ec_mod._REAPER_INTERVAL = 0.5

        ctx = ExecutionContext.get_or_create(
            cluster=self.cluster, close_after=None
        )
        pool_key = ExecutionContext._pool_key(
            cluster_id=self.cluster.cluster_id,
            language=Language.PYTHON,
            context_key=ctx.context_key,
        )
        ctx._last_used_at = time.time() - 9999.0  # simulate long idle

        _ensure_reaper_running()
        time.sleep(2.0)

        self.assertIn(pool_key, _ec_mod._CONTEXT_POOL)
        self.assertTrue(ctx.context_id)


if __name__ == "__main__":
    unittest.main()

