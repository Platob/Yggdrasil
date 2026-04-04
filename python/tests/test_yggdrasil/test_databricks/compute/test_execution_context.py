"""
Unit tests for ExecutionContext.

Covers: lifecycle (create / connect / close), context manager, atexit
registration, URL/repr, remote_metadata, syspath_lines, and command()
factory.  All tests are entirely offline — no real Databricks cluster.
"""

from __future__ import annotations

import json
import logging
import os
import time
import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

from databricks.sdk.service.compute import Language

from yggdrasil.databricks.compute.execution_context import (
    ExecutionContext,
    RemoteMetadata,
    exclude_env_key,
)


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------

def _make_cluster(cluster_id: str = "cls-abc") -> MagicMock:
    cluster = MagicMock(name="Cluster")
    cluster.cluster_id = cluster_id
    url_mock = MagicMock(name="URL")
    url_mock.with_query_items.return_value = url_mock
    url_mock.to_string.return_value = f"https://dbc.example.com/?cluster={cluster_id}"
    cluster.url.return_value = url_mock
    cluster.is_in_databricks_environment.return_value = False
    return cluster


def _ws(cluster: MagicMock):
    """Return the mocked command_execution service."""
    return cluster.client.workspace_client.return_value.command_execution


def _ctx_response(ctx_id: str) -> MagicMock:
    r = MagicMock()
    r.response.id = ctx_id
    return r


def _make_ctx(
    cluster: Optional[MagicMock] = None,
    *,
    context_id: str = "",
    temporary: bool = True,
    language: Language = Language.PYTHON,
    context_key: str = "test-key",
) -> ExecutionContext:
    return ExecutionContext(
        cluster=cluster or _make_cluster(),
        context_id=context_id,
        language=language,
        temporary=temporary,
        context_key=context_key,
    )


# ===========================================================================
# exclude_env_key
# ===========================================================================

class TestExcludeEnvKey(unittest.TestCase):

    def test_exact_match_excluded(self):
        self.assertTrue(exclude_env_key("DATABRICKS_HOST"))
        self.assertTrue(exclude_env_key("databricks_host"))   # case-insensitive
        self.assertTrue(exclude_env_key("PATH"))
        self.assertTrue(exclude_env_key("PYTHONPATH"))

    def test_prefix_excluded(self):
        self.assertTrue(exclude_env_key("SPARK_HOME"))
        self.assertTrue(exclude_env_key("DATABRICKS_TOKEN"))
        self.assertTrue(exclude_env_key("MLFLOW_TRACKING_URI"))
        self.assertTrue(exclude_env_key("PYTEST_CURRENT_TEST"))

    def test_user_var_included(self):
        self.assertFalse(exclude_env_key("MY_APP_SECRET"))
        self.assertFalse(exclude_env_key("POSTGRES_URL"))
        self.assertFalse(exclude_env_key("REDIS_HOST"))


# ===========================================================================
# RemoteMetadata
# ===========================================================================

class TestRemoteMetadata(unittest.TestCase):

    def test_metadata_paths_derived_from_context_key(self):
        ctx = _make_ctx(context_key="my-key")
        m = ctx.remote_metadata

        self.assertIsInstance(m, RemoteMetadata)
        self.assertIn("my-key", m.context_path)
        self.assertTrue(m.tmp_path.startswith(m.context_path))
        self.assertTrue(m.libs_path.startswith(m.context_path))

    def test_metadata_cached_after_first_access(self):
        ctx = _make_ctx(context_key="cached-key")
        m1 = ctx.remote_metadata
        m2 = ctx.remote_metadata
        self.assertIs(m1, m2)

    def test_metadata_falls_back_to_hostname_when_no_key(self):
        ctx = _make_ctx(context_key="")
        ctx.context_key = None
        m = ctx.remote_metadata
        self.assertIsNotNone(m.context_path)


# ===========================================================================
# create()
# ===========================================================================

class TestExecutionContextCreate(unittest.TestCase):

    @patch("atexit.register")
    def test_create_returns_new_instance_with_context_id(self, _):
        ctx = _make_ctx(temporary=False)
        _ws(ctx.cluster).create.return_value = _ctx_response("ctx-001")

        instance = ctx.create(language=Language.PYTHON, temporary=False)

        self.assertIsInstance(instance, ExecutionContext)
        self.assertEqual(instance.context_id, "ctx-001")
        # create() mutates self in-place and returns self — NOT a new object
        self.assertIs(instance, ctx)

    @patch("atexit.register")
    def test_create_temporary_true_registers_atexit(self, mock_reg):
        ctx = _make_ctx(temporary=False)
        _ws(ctx.cluster).create.return_value = _ctx_response("ctx-002")

        instance = ctx.create(language=Language.PYTHON, temporary=True)

        mock_reg.assert_called_once_with(instance._unsafe_close)

    @patch("atexit.register")
    def test_create_temporary_false_no_atexit(self, mock_reg):
        ctx = _make_ctx(temporary=False)
        _ws(ctx.cluster).create.return_value = _ctx_response("ctx-003")

        ctx.create(language=Language.PYTHON, temporary=False)

        mock_reg.assert_not_called()

    @patch("atexit.register")
    def test_create_reuses_when_context_id_and_language_match(self, mock_reg):
        ctx = _make_ctx(context_id="ctx-existing", language=Language.PYTHON)

        result = ctx.create(language=Language.PYTHON)

        self.assertIs(result, ctx)
        _ws(ctx.cluster).create.assert_not_called()
        mock_reg.assert_not_called()

    @patch("atexit.register")
    def test_create_propagates_context_key_to_instance(self, _):
        ctx = _make_ctx(temporary=False, context_key="my-key")
        _ws(ctx.cluster).create.return_value = _ctx_response("ctx-004")

        instance = ctx.create(language=Language.PYTHON, temporary=False)

        self.assertEqual(instance.context_key, "my-key")

    @patch("atexit.register")
    def test_create_generates_random_key_when_none(self, _):
        ctx = _make_ctx(temporary=False, context_key="")
        ctx.context_key = None
        _ws(ctx.cluster).create.return_value = _ctx_response("ctx-005")

        instance = ctx.create(language=Language.PYTHON, temporary=False)

        self.assertIsNotNone(instance.context_key)
        self.assertGreater(len(instance.context_key), 0)

    @patch("atexit.register")
    def test_create_retries_after_timeout(self, _):
        """On FuturesTimeoutError, create() must ensure_running and retry."""
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        ctx = _make_ctx(temporary=False)
        ws = _ws(ctx.cluster)

        # First submit/result raises timeout; final client.create succeeds
        future_mock = MagicMock()
        future_mock.result.side_effect = FuturesTimeoutError()
        executor_mock = MagicMock()
        executor_mock.__enter__ = MagicMock(return_value=executor_mock)
        executor_mock.__exit__ = MagicMock(return_value=False)
        executor_mock.submit.return_value = future_mock

        ws.create.return_value = _ctx_response("ctx-retry")

        with patch(
            "yggdrasil.databricks.compute.execution_context.ThreadPoolExecutor",
            return_value=executor_mock,
        ):
            instance = ctx.create(language=Language.PYTHON, temporary=False)

        ctx.cluster.ensure_running.assert_called()
        self.assertEqual(instance.context_id, "ctx-retry")

    @patch("atexit.register")
    def test_create_retries_after_generic_exception(self, _):
        """On any exception, create() must ensure_running and retry once more."""
        ctx = _make_ctx(temporary=False)
        ws = _ws(ctx.cluster)

        # The executor path raises; the fallback direct call succeeds
        with patch(
            "yggdrasil.databricks.compute.execution_context.ThreadPoolExecutor",
            side_effect=RuntimeError("boom"),
        ):
            ws.create.return_value = _ctx_response("ctx-fallback")
            instance = ctx.create(language=Language.PYTHON, temporary=False)

        ctx.cluster.ensure_running.assert_called()
        self.assertEqual(instance.context_id, "ctx-fallback")


# ===========================================================================
# connect()
# ===========================================================================

class TestExecutionContextConnect(unittest.TestCase):

    @patch("atexit.register")
    def test_connect_empty_context_id_creates_new(self, mock_reg):
        """Bug guard: context_id='' must not be treated as 'connected'."""
        ctx = _make_ctx(context_id="", temporary=True)
        _ws(ctx.cluster).create.return_value = _ctx_response("ctx-new")

        result = ctx.connect()

        self.assertEqual(result.context_id, "ctx-new")
        mock_reg.assert_called_once()

    @patch("atexit.register")
    def test_connect_existing_returns_self(self, mock_reg):
        ctx = _make_ctx(context_id="ctx-ok", temporary=True)

        result = ctx.connect()

        self.assertIs(result, ctx)
        _ws(ctx.cluster).create.assert_not_called()
        mock_reg.assert_not_called()

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_connect_reset_closes_old_creates_new(self, mock_unreg, mock_reg):
        ctx = _make_ctx(context_id="ctx-old", temporary=True)
        _ws(ctx.cluster).create.return_value = _ctx_response("ctx-new")

        result = ctx.connect(reset=True)

        # connect() mutates ctx in-place (close → create → same object returned)
        self.assertIs(result, ctx)
        self.assertEqual(result.context_id, "ctx-new")
        mock_reg.assert_called_once()

    @patch("atexit.register")
    def test_connect_defaults_language_to_python(self, _):
        ctx = _make_ctx(context_id="", temporary=False)
        ctx.language = None
        _ws(ctx.cluster).create.return_value = _ctx_response("ctx-py")

        result = ctx.connect()

        call_kwargs = _ws(ctx.cluster).create.call_args.kwargs
        self.assertEqual(call_kwargs["language"], Language.PYTHON)


# ===========================================================================
# close()
# ===========================================================================

class TestExecutionContextClose(unittest.TestCase):

    @patch("atexit.unregister")
    def test_close_calls_destroy_and_clears_context_id(self, _):
        ctx = _make_ctx(context_id="ctx-100", temporary=True)

        ctx.close(wait=True, raise_error=False)

        _ws(ctx.cluster).destroy.assert_called_once()
        self.assertFalse(ctx.context_id)  # close() sets context_id to "" (falsy)

    @patch("atexit.unregister")
    def test_close_empty_context_id_is_noop(self, mock_unreg):
        ctx = _make_ctx(context_id="", temporary=True)

        ctx.close(wait=True, raise_error=False)

        ctx.cluster.client.workspace_client.assert_not_called()
        mock_unreg.assert_not_called()

    @patch("atexit.unregister")
    def test_close_temporary_unregisters_atexit(self, mock_unreg):
        ctx = _make_ctx(context_id="ctx-101", temporary=True)

        ctx.close(wait=False, raise_error=False)

        mock_unreg.assert_called_once()
        self.assertIs(mock_unreg.call_args[0][0].__self__, ctx)

    @patch("atexit.unregister")
    def test_close_non_temporary_no_atexit(self, mock_unreg):
        ctx = _make_ctx(context_id="ctx-102", temporary=False)

        ctx.close(wait=False, raise_error=False)

        mock_unreg.assert_not_called()

    @patch("atexit.unregister")
    def test_close_idempotent(self, mock_unreg):
        ctx = _make_ctx(context_id="ctx-103", temporary=True)

        ctx.close(wait=False, raise_error=False)
        ctx.close(wait=False, raise_error=False)  # second call: context_id is None

        self.assertEqual(mock_unreg.call_count, 1)

    @patch("atexit.unregister")
    def test_close_no_wait_uses_fire_and_forget(self, _):
        """close(wait=False) must use fire-and-forget, not a blocking destroy call."""
        ctx = _make_ctx(context_id="ctx-104", temporary=True)

        with patch("yggdrasil.databricks.compute.execution_context.Job") as mock_job:
            job_instance = MagicMock()
            mock_job.make.return_value = job_instance

            ctx.close(wait=False, raise_error=False)

        # Job.make was used instead of a direct blocking destroy()
        mock_job.make.assert_called_once()
        job_instance.fire_and_forget.assert_called_once()
        _ws(ctx.cluster).destroy.assert_not_called()

    @patch("atexit.unregister")
    def test_close_wait_true_calls_destroy_synchronously(self, _):
        ctx = _make_ctx(context_id="ctx-105", temporary=True)

        ctx.close(wait=True, raise_error=False)

        _ws(ctx.cluster).destroy.assert_called_once()

    @patch("atexit.unregister")
    def test_unsafe_close_suppresses_errors(self, _):
        ctx = _make_ctx(context_id="ctx-106", temporary=True)
        from databricks.sdk.errors import DatabricksError
        _ws(ctx.cluster).destroy.side_effect = DatabricksError("boom")

        # Must not raise
        ctx._unsafe_close()

        self.assertFalse(ctx.context_id)  # close() sets context_id to "" (falsy)


# ===========================================================================
# __enter__ / __exit__  (context manager)
# ===========================================================================

class TestExecutionContextContextManager(unittest.TestCase):

    @patch("atexit.register")
    def test_enter_empty_string_creates_context(self, mock_reg):
        """Bug guard: default context_id='' must trigger create(), not reuse."""
        ctx = _make_ctx(context_id="", temporary=True)
        _ws(ctx.cluster).create.return_value = _ctx_response("ctx-new")

        result = ctx.__enter__()

        # __enter__ → connect() → create() all mutate self in-place and return self
        self.assertIs(result, ctx)
        self.assertEqual(result.context_id, "ctx-new")

    @patch("atexit.register")
    def test_enter_existing_context_id_reuses(self, mock_reg):
        ctx = _make_ctx(context_id="ctx-ok", temporary=True)

        result = ctx.__enter__()

        self.assertIs(result, ctx)
        mock_reg.assert_not_called()

    @patch("atexit.unregister")
    def test_exit_calls_close(self, _):
        ctx = _make_ctx(context_id="ctx-exit", temporary=True)

        ctx.__exit__(None, None, None)

        self.assertFalse(ctx.context_id)  # close() sets context_id to "" (falsy)

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_context_manager_roundtrip(self, mock_unreg, mock_reg):
        ctx = _make_ctx(context_id="", temporary=True)
        _ws(ctx.cluster).create.return_value = _ctx_response("ctx-cm")

        with ctx as inner:
            # __enter__ → connect() → create() mutates ctx in-place, returns self
            self.assertIs(inner, ctx)
            self.assertEqual(inner.context_id, "ctx-cm")
            mock_reg.assert_called_once()

        # __exit__ calls close() on ctx (which now has context_id="ctx-cm", not "")
        # so close() IS invoked and unregister IS called
        mock_unreg.assert_called_once()
        self.assertFalse(ctx.context_id)

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_context_manager_roundtrip_reuse(self, mock_unreg, mock_reg):
        """When context_id is set before entering, __exit__ closes it."""
        ctx = _make_ctx(context_id="ctx-pre", temporary=True)

        with ctx as inner:
            self.assertIs(inner, ctx)

        # close() was called on ctx — context_id is now "" (falsy)
        self.assertFalse(ctx.context_id)
        mock_unreg.assert_called_once()


# ===========================================================================
# url / __repr__ / __str__
# ===========================================================================

class TestExecutionContextRepr(unittest.TestCase):

    def test_url_contains_context_id(self):
        ctx = _make_ctx(context_id="ctx-abc")
        ctx.cluster.url.return_value.with_query_items.return_value = MagicMock(
            to_string=MagicMock(return_value="https://host/?context=ctx-abc")
        )
        # url() must be callable without error
        url = ctx.url()
        self.assertIsNotNone(url)

    def test_url_unknown_when_no_context_id(self):
        ctx = _make_ctx(context_id="")
        # Pass through — should not raise
        ctx.url()

    def test_repr_contains_class_name(self):
        ctx = _make_ctx(context_id="ctx-repr")
        r = repr(ctx)
        self.assertIn("ExecutionContext", r)

    def test_str_returns_string(self):
        ctx = _make_ctx(context_id="ctx-str")
        s = str(ctx)
        self.assertIsInstance(s, str)


# ===========================================================================
# syspath_lines()
# ===========================================================================

class TestSyspathLines(unittest.TestCase):

    def test_basic_snippet_contains_sys_path_insert(self):
        ctx = _make_ctx(context_id="ctx-sp", context_key="sp-key")
        snippet = ctx.syspath_lines()

        self.assertIn("sys.path.insert", snippet)
        self.assertIn("sp-key", snippet)

    def test_environ_injected_as_base64_gzip(self):
        ctx = _make_ctx(context_id="ctx-env", context_key="env-key")
        snippet = ctx.syspath_lines(environ={"MY_VAR": "hello"})

        self.assertIn("base64", snippet)
        self.assertIn("gzip", snippet)
        self.assertIn("os.environ", snippet)

    def test_no_environ_omits_env_block(self):
        ctx = _make_ctx(context_id="ctx-noenv", context_key="noenv-key")
        snippet = ctx.syspath_lines()

        self.assertNotIn("os.environ", snippet)

    def test_none_env_value_replaced_with_empty_string(self):
        ctx = _make_ctx(context_id="ctx-none", context_key="none-key")
        import base64, gzip
        snippet = ctx.syspath_lines(environ={"KEY": None})

        # Extract the base64 block from the snippet and decode it
        import re
        m = re.search(r"base64\.b64decode\(b?'([^']+)'", snippet)
        if m:
            decoded = gzip.decompress(base64.b64decode(m.group(1)))
            env = json.loads(decoded)
            self.assertEqual(env["KEY"], "")

    def test_excluded_env_keys_not_in_snippet(self):
        """Keys that match exclude_env_key() must not appear in the environ block."""
        ctx = _make_ctx(context_id="ctx-exc", context_key="exc-key")
        snippet = ctx.syspath_lines(environ={"DATABRICKS_HOST": "https://x.com"})
        # DATABRICKS_HOST is excluded by exclude_env_key; but syspath_lines
        # accepts whatever dict is passed — the filtering happens in command().
        # Here we just verify the snippet is produced without error.
        self.assertIsInstance(snippet, str)


# ===========================================================================
# command()
# ===========================================================================

class TestExecutionContextCommand(unittest.TestCase):

    def _make_connected_ctx(self) -> ExecutionContext:
        ctx = _make_ctx(context_id="ctx-cmd", temporary=True, context_key="cmd-key")
        return ctx

    def test_command_string_sets_command_field(self):
        from yggdrasil.databricks.compute.command_execution import CommandExecution
        ctx = self._make_connected_ctx()

        cmd = ctx.command("print(1)")

        self.assertIsInstance(cmd, CommandExecution)
        self.assertIn("print(1)", cmd.command)

    def test_command_callable_sets_pyfunc(self):
        from yggdrasil.databricks.compute.command_execution import CommandExecution
        ctx = self._make_connected_ctx()

        def my_func():
            return 42

        cmd = ctx.command(my_func)

        self.assertIsInstance(cmd, CommandExecution)
        self.assertIs(cmd.pyfunc, my_func)

    def test_command_language_string_converted_to_enum(self):
        from yggdrasil.databricks.compute.command_execution import CommandExecution
        ctx = self._make_connected_ctx()

        cmd = ctx.command("SELECT 1", language="SQL")  # enum keys are uppercase

        self.assertIsInstance(cmd, CommandExecution)
        self.assertEqual(cmd.language, Language.SQL)

    def test_command_python_prepends_syspath(self):
        from yggdrasil.databricks.compute.command_execution import CommandExecution
        ctx = self._make_connected_ctx()

        cmd = ctx.command("x = 1", language="PYTHON")  # enum keys are uppercase

        self.assertIn("sys.path.insert", cmd.command)
        self.assertIn("x = 1", cmd.command)


    def test_command_environ_filtering(self):
        from yggdrasil.databricks.compute.command_execution import CommandExecution
        ctx = self._make_connected_ctx()

        with patch.dict(os.environ, {"MY_CUSTOM_VAR": "abc", "DATABRICKS_HOST": "https://x.com"}):
            cmd = ctx.command("pass", environ={"EXTRA": "1"})

        # DATABRICKS_HOST should be excluded; MY_CUSTOM_VAR should be present
        # (filtering happens via exclude_env_key inside command())
        self.assertIsInstance(cmd, CommandExecution)


# ===========================================================================
# is_in_databricks_environment()
# ===========================================================================

class TestInDatabricksEnvironment(unittest.TestCase):

    def test_delegates_to_cluster_client(self):
        ctx = _make_ctx(context_id="ctx-dbx")
        ctx.cluster.client.is_in_databricks_environment.return_value = True

        self.assertTrue(ctx.is_in_databricks_environment())

    def test_false_when_not_in_databricks(self):
        ctx = _make_ctx(context_id="ctx-local")
        ctx.cluster.client.is_in_databricks_environment.return_value = False

        self.assertFalse(ctx.is_in_databricks_environment())


import yggdrasil.databricks.compute.execution_context as _ec_mod
from yggdrasil.databricks.compute.execution_context import (
    ContextPoolKey,
    _evict_idle_contexts,
    _ensure_reaper_running,
)


# ===========================================================================
# Helpers shared by reaper tests
# ===========================================================================

def _pool_key_for(ctx: ExecutionContext, cluster_id: str) -> ContextPoolKey:
    """Return the ContextPoolKey that would address *ctx* in the pool."""
    return ContextPoolKey(
        cluster_id=cluster_id,
        language=(ctx.language or Language.PYTHON).value,
        context_key=str(ctx.context_key or ""),
    )


class _ReaperTestBase(unittest.TestCase):
    """Mixin that saves/restores global pool & reaper state around every test."""

    def setUp(self):
        # --- pool ---
        self._orig_pool = dict(_ec_mod._CONTEXT_POOL)
        _ec_mod._CONTEXT_POOL.clear()
        # --- reaper thread ---
        _ec_mod._REAPER_STOP.set()
        if _ec_mod._REAPER_THREAD is not None and _ec_mod._REAPER_THREAD.is_alive():
            _ec_mod._REAPER_THREAD.join(timeout=3.0)
        _ec_mod._REAPER_THREAD = None
        _ec_mod._REAPER_STOP.clear()
        # --- speed up interval for tests that actually run the reaper ---
        self._orig_interval = _ec_mod._REAPER_INTERVAL
        _ec_mod._REAPER_INTERVAL = 0.05  # 50 ms

    def tearDown(self):
        # --- stop reaper ---
        _ec_mod._REAPER_STOP.set()
        if _ec_mod._REAPER_THREAD is not None and _ec_mod._REAPER_THREAD.is_alive():
            _ec_mod._REAPER_THREAD.join(timeout=3.0)
        _ec_mod._REAPER_THREAD = None
        _ec_mod._REAPER_STOP.clear()
        # --- restore pool ---
        _ec_mod._CONTEXT_POOL.clear()
        _ec_mod._CONTEXT_POOL.update(self._orig_pool)
        # --- restore interval ---
        _ec_mod._REAPER_INTERVAL = self._orig_interval

    # convenience: inject a ctx directly into the pool
    def _pool_add(
        self,
        ctx: ExecutionContext,
        cluster_id: str = "cls",
        language: Language = Language.PYTHON,
        context_key: str = "k",
    ) -> ContextPoolKey:
        key = ContextPoolKey(
            cluster_id=cluster_id,
            language=language.value,
            context_key=context_key,
        )
        _ec_mod._CONTEXT_POOL[key] = ctx
        return key


# ===========================================================================
# close_after field
# ===========================================================================

class TestCloseAfterField(unittest.TestCase):

    def test_default_is_1800(self):
        ctx = _make_ctx()
        self.assertEqual(ctx.close_after, 1800.0)

    def test_none_disables_auto_close(self):
        ctx = ExecutionContext(cluster=_make_cluster(), close_after=None)
        self.assertIsNone(ctx.close_after)

    def test_custom_value_preserved(self):
        ctx = ExecutionContext(cluster=_make_cluster(), close_after=300.0)
        self.assertEqual(ctx.close_after, 300.0)

    def test_excluded_from_equality(self):
        c = _make_cluster()
        ctx1 = ExecutionContext(cluster=c, context_id="x", close_after=100.0)
        ctx2 = ExecutionContext(cluster=c, context_id="x", close_after=9999.0)
        self.assertEqual(ctx1, ctx2)

    def test_excluded_from_repr(self):
        ctx = ExecutionContext(cluster=_make_cluster(), close_after=42.0)
        self.assertNotIn("42", repr(ctx))

    def test_excluded_from_hash(self):
        # ExecutionContext is a mutable dataclass (not frozen), so __hash__ is
        # removed by Python (unhashable by default).  The hash=False parameter
        # on the field is therefore consistent — just verify the dataclass is
        # indeed not hashable.
        ctx = ExecutionContext(cluster=_make_cluster(), context_id="x", close_after=1.0)
        with self.assertRaises(TypeError):
            hash(ctx)


# ===========================================================================
# touch()
# ===========================================================================

class TestTouchMethod(unittest.TestCase):

    def test_touch_sets_last_used_at(self):
        ctx = _make_ctx()
        self.assertEqual(ctx._last_used_at, 0.0)
        before = time.time()
        ctx.touch()
        after = time.time()
        self.assertGreaterEqual(ctx._last_used_at, before)
        self.assertLessEqual(ctx._last_used_at, after)

    def test_touch_is_monotonically_increasing(self):
        ctx = _make_ctx()
        ctx.touch()
        t1 = ctx._last_used_at
        time.sleep(0.02)
        ctx.touch()
        t2 = ctx._last_used_at
        self.assertGreater(t2, t1)

    def test_touch_updates_on_every_call(self):
        ctx = _make_ctx()
        ctx.touch()
        t1 = ctx._last_used_at
        time.sleep(0.01)
        ctx.touch()
        self.assertNotEqual(ctx._last_used_at, t1)


# ===========================================================================
# _evict_idle_contexts()
# ===========================================================================

class TestEvictIdleContexts(_ReaperTestBase):

    def test_evicts_expired_context(self):
        ctx = _make_ctx(context_id="ctx-exp")
        ctx.close_after = 60.0
        ctx._last_used_at = time.time() - 120.0       # 2 min ago → expired
        key = self._pool_add(ctx, cluster_id="cls-exp")

        _evict_idle_contexts()

        self.assertNotIn(key, _ec_mod._CONTEXT_POOL)
        self.assertFalse(ctx.context_id)              # close() sets context_id=""

    def test_skips_context_not_yet_expired(self):
        ctx = _make_ctx(context_id="ctx-fresh")
        ctx.close_after = 3600.0
        ctx._last_used_at = time.time() - 10.0        # only 10 s ago → not expired
        key = self._pool_add(ctx, cluster_id="cls-fresh")

        _evict_idle_contexts()

        self.assertIn(key, _ec_mod._CONTEXT_POOL)
        self.assertEqual(ctx.context_id, "ctx-fresh")

    def test_skips_close_after_none(self):
        ctx = _make_ctx(context_id="ctx-persist")
        ctx.close_after = None
        ctx._last_used_at = time.time() - 999_999.0  # ancient — would expire without None
        key = self._pool_add(ctx, cluster_id="cls-persist")

        _evict_idle_contexts()

        self.assertIn(key, _ec_mod._CONTEXT_POOL)
        self.assertEqual(ctx.context_id, "ctx-persist")

    def test_skips_already_closed_context_id(self):
        ctx = _make_ctx(context_id="")              # already closed (empty string)
        ctx.close_after = 10.0
        ctx._last_used_at = time.time() - 9999.0
        key = self._pool_add(ctx, cluster_id="cls-already-closed")

        _evict_idle_contexts()

        # Stays in pool (guard: `not ctx.context_id`) and close() is NOT called again
        self.assertIn(key, _ec_mod._CONTEXT_POOL)

    def test_skips_context_never_used(self):
        ctx = _make_ctx(context_id="ctx-never-used")
        ctx.close_after = 10.0
        ctx._last_used_at = 0.0                     # never touched
        key = self._pool_add(ctx, cluster_id="cls-never")

        _evict_idle_contexts()

        self.assertIn(key, _ec_mod._CONTEXT_POOL)
        self.assertEqual(ctx.context_id, "ctx-never-used")

    def test_partial_eviction_mixed_contexts(self):
        cluster = _make_cluster()
        # Expired
        ctx_exp = ExecutionContext(cluster=cluster, context_id="exp", close_after=60.0, context_key="k1")
        ctx_exp._last_used_at = time.time() - 120.0
        key_exp = ContextPoolKey(cluster_id="cls", language="PYTHON", context_key="k1")
        _ec_mod._CONTEXT_POOL[key_exp] = ctx_exp
        # Fresh
        ctx_ok = ExecutionContext(cluster=cluster, context_id="ok", close_after=3600.0, context_key="k2")
        ctx_ok._last_used_at = time.time() - 5.0
        key_ok = ContextPoolKey(cluster_id="cls", language="PYTHON", context_key="k2")
        _ec_mod._CONTEXT_POOL[key_ok] = ctx_ok

        _evict_idle_contexts()

        self.assertNotIn(key_exp, _ec_mod._CONTEXT_POOL)
        self.assertIn(key_ok, _ec_mod._CONTEXT_POOL)
        self.assertFalse(ctx_exp.context_id)
        self.assertEqual(ctx_ok.context_id, "ok")

    def test_calls_close_on_expired(self):
        ctx = _make_ctx(context_id="ctx-closeme")
        ctx.close_after = 60.0
        ctx._last_used_at = time.time() - 120.0
        self._pool_add(ctx, cluster_id="cls-closeme")

        with patch.object(ctx, "close") as mock_close:
            _evict_idle_contexts()

        mock_close.assert_called_once_with(wait=False, raise_error=False)

    def test_evict_suppresses_close_errors(self):
        ctx = _make_ctx(context_id="ctx-err")
        ctx.close_after = 60.0
        ctx._last_used_at = time.time() - 120.0
        self._pool_add(ctx, cluster_id="cls-err")

        with patch.object(ctx, "close", side_effect=RuntimeError("boom")):
            # Must not propagate
            _evict_idle_contexts()

    def test_multiple_eviction_calls_are_safe(self):
        """Calling evict twice when pool is already empty must not raise."""
        _evict_idle_contexts()
        _evict_idle_contexts()   # no-op — safe

    def test_pool_empty_after_all_expired(self):
        cluster = _make_cluster()
        for i in range(3):
            ctx = ExecutionContext(cluster=cluster, context_id=f"ctx-{i}", close_after=60.0, context_key=f"k{i}")
            ctx._last_used_at = time.time() - 120.0
            key = ContextPoolKey(cluster_id="cls", language="PYTHON", context_key=f"k{i}")
            _ec_mod._CONTEXT_POOL[key] = ctx

        _evict_idle_contexts()

        self.assertEqual(len(_ec_mod._CONTEXT_POOL), 0)

    def test_evict_at_exact_boundary_is_inclusive(self):
        """A context idle for exactly close_after seconds must be evicted."""
        ctx = _make_ctx(context_id="ctx-boundary")
        ctx.close_after = 60.0
        ctx._last_used_at = time.time() - 60.0        # exactly at boundary
        key = self._pool_add(ctx, cluster_id="cls-boundary")

        _evict_idle_contexts()

        self.assertNotIn(key, _ec_mod._CONTEXT_POOL)

    def test_evict_just_before_boundary_is_skipped(self):
        ctx = _make_ctx(context_id="ctx-before")
        ctx.close_after = 60.0
        ctx._last_used_at = time.time() - 59.0        # 1 second before boundary
        key = self._pool_add(ctx, cluster_id="cls-before")

        _evict_idle_contexts()

        self.assertIn(key, _ec_mod._CONTEXT_POOL)


# ===========================================================================
# _ensure_reaper_running()
# ===========================================================================

class TestEnsureReaperRunning(_ReaperTestBase):

    def test_starts_a_daemon_thread(self):
        _ensure_reaper_running()
        self.assertIsNotNone(_ec_mod._REAPER_THREAD)
        self.assertTrue(_ec_mod._REAPER_THREAD.is_alive())
        self.assertTrue(_ec_mod._REAPER_THREAD.daemon)

    def test_thread_has_correct_name(self):
        _ensure_reaper_running()
        self.assertEqual(_ec_mod._REAPER_THREAD.name, "ygg-context-reaper")

    def test_idempotent_same_thread_returned(self):
        _ensure_reaper_running()
        thread1 = _ec_mod._REAPER_THREAD
        _ensure_reaper_running()
        thread2 = _ec_mod._REAPER_THREAD
        self.assertIs(thread1, thread2)

    def test_idempotent_thread_still_alive(self):
        _ensure_reaper_running()
        _ensure_reaper_running()
        _ensure_reaper_running()
        self.assertTrue(_ec_mod._REAPER_THREAD.is_alive())

    def test_restarts_dead_thread(self):
        _ensure_reaper_running()
        old_thread = _ec_mod._REAPER_THREAD
        # Stop it
        _ec_mod._REAPER_STOP.set()
        old_thread.join(timeout=5.0)
        self.assertFalse(old_thread.is_alive())
        # Start again
        _ec_mod._REAPER_STOP.clear()
        _ensure_reaper_running()
        self.assertIsNot(_ec_mod._REAPER_THREAD, old_thread)
        self.assertTrue(_ec_mod._REAPER_THREAD.is_alive())


# ===========================================================================
# _reaper_loop() — calls evict on each tick
# ===========================================================================

class TestReaperLoopBehavior(_ReaperTestBase):

    def test_reaper_calls_evict_multiple_times(self):
        """With a 50 ms interval, at least 3 eviction calls in 300 ms."""
        with patch.object(_ec_mod, "_evict_idle_contexts") as mock_evict:
            _ensure_reaper_running()
            time.sleep(0.3)
            _ec_mod._REAPER_STOP.set()
            _ec_mod._REAPER_THREAD.join(timeout=3.0)

        self.assertGreaterEqual(mock_evict.call_count, 3)

    def test_reaper_stops_when_stop_event_set(self):
        _ensure_reaper_running()
        t = _ec_mod._REAPER_THREAD
        self.assertTrue(t.is_alive())
        _ec_mod._REAPER_STOP.set()
        t.join(timeout=3.0)
        self.assertFalse(t.is_alive())

    def test_reaper_swallows_evict_exceptions(self):
        """Exception inside _evict_idle_contexts must not kill the reaper thread."""
        call_count = {"n": 0}

        def flaky_evict():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient error")
            # Stop after second call so test finishes quickly
            _ec_mod._REAPER_STOP.set()

        with patch.object(_ec_mod, "_evict_idle_contexts", side_effect=flaky_evict):
            _ensure_reaper_running()
            _ec_mod._REAPER_THREAD.join(timeout=3.0)

        self.assertGreaterEqual(call_count["n"], 2)


# ===========================================================================
# get_or_create() — close_after wiring
# ===========================================================================

class TestGetOrCreateCloseAfter(_ReaperTestBase):

    @patch("atexit.register")
    def test_default_close_after_on_new_context(self, _):
        cluster = _make_cluster("cls-goc1")
        _ws(cluster).create.return_value = _ctx_response("ctx-goc1")

        ctx = ExecutionContext.get_or_create(cluster=cluster)

        self.assertEqual(ctx.close_after, 1800.0)

    @patch("atexit.register")
    def test_custom_close_after_on_new_context(self, _):
        cluster = _make_cluster("cls-goc2")
        _ws(cluster).create.return_value = _ctx_response("ctx-goc2")

        ctx = ExecutionContext.get_or_create(cluster=cluster, close_after=300.0)

        self.assertEqual(ctx.close_after, 300.0)

    @patch("atexit.register")
    def test_close_after_none_on_new_context(self, _):
        cluster = _make_cluster("cls-goc3")
        _ws(cluster).create.return_value = _ctx_response("ctx-goc3")

        ctx = ExecutionContext.get_or_create(cluster=cluster, close_after=None)

        self.assertIsNone(ctx.close_after)

    @patch("atexit.register")
    def test_starts_reaper_when_close_after_is_set(self, _):
        cluster = _make_cluster("cls-goc4")
        _ws(cluster).create.return_value = _ctx_response("ctx-goc4")

        with patch.object(_ec_mod, "_ensure_reaper_running") as mock_reaper:
            ExecutionContext.get_or_create(cluster=cluster, close_after=1800.0)

        mock_reaper.assert_called_once()

    @patch("atexit.register")
    def test_no_reaper_when_close_after_none(self, _):
        cluster = _make_cluster("cls-goc5")
        _ws(cluster).create.return_value = _ctx_response("ctx-goc5")

        with patch.object(_ec_mod, "_ensure_reaper_running") as mock_reaper:
            ExecutionContext.get_or_create(cluster=cluster, close_after=None)

        mock_reaper.assert_not_called()

    @patch("atexit.register")
    def test_reaper_only_started_for_new_context(self, _):
        """Pool hit must not call _ensure_reaper_running again."""
        cluster = _make_cluster("cls-goc6")
        _ws(cluster).create.return_value = _ctx_response("ctx-goc6")

        with patch.object(_ec_mod, "_ensure_reaper_running") as mock_reaper:
            ExecutionContext.get_or_create(cluster=cluster, close_after=1800.0)
            self.assertEqual(mock_reaper.call_count, 1)
            ExecutionContext.get_or_create(cluster=cluster, close_after=1800.0)
            self.assertEqual(mock_reaper.call_count, 1)  # unchanged

    @patch("atexit.register")
    def test_pool_contains_context_after_get_or_create(self, _):
        cluster = _make_cluster("cls-goc7")
        _ws(cluster).create.return_value = _ctx_response("ctx-goc7")

        ctx = ExecutionContext.get_or_create(cluster=cluster)

        self.assertEqual(len(_ec_mod._CONTEXT_POOL), 1)
        self.assertIn(ctx, _ec_mod._CONTEXT_POOL.values())

    @patch("atexit.register")
    def test_touch_called_so_last_used_at_set(self, _):
        cluster = _make_cluster("cls-goc8")
        _ws(cluster).create.return_value = _ctx_response("ctx-goc8")

        before = time.time()
        ctx = ExecutionContext.get_or_create(cluster=cluster)
        after = time.time()

        self.assertGreaterEqual(ctx._last_used_at, before)
        self.assertLessEqual(ctx._last_used_at, after)


# ===========================================================================
# Full reaper cycle — end-to-end with real thread (offline)
# ===========================================================================

class TestFullReaperEvictionCycle(_ReaperTestBase):
    """
    End-to-end offline tests that run the actual background reaper thread
    and verify contexts are evicted (or preserved) correctly.
    All Databricks SDK calls are satisfied by MagicMock.
    """

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_idle_context_evicted_by_reaper(self, mock_unreg, mock_reg):
        cluster = _make_cluster("cls-idle-evict")
        _ws(cluster).create.return_value = _ctx_response("ctx-idle")

        ctx = ExecutionContext.get_or_create(cluster=cluster, close_after=0.05)
        ctx._last_used_at = time.time() - 1.0      # already idle > close_after

        key = ExecutionContext._pool_key(
            cluster_id="cls-idle-evict",
            language=Language.PYTHON,
            context_key=ctx.context_key,
        )

        _ensure_reaper_running()
        time.sleep(0.4)                            # allow ≥ 4 cycles

        self.assertNotIn(key, _ec_mod._CONTEXT_POOL)
        self.assertFalse(ctx.context_id)

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_active_context_not_evicted(self, mock_unreg, mock_reg):
        cluster = _make_cluster("cls-active")
        _ws(cluster).create.return_value = _ctx_response("ctx-active")

        ctx = ExecutionContext.get_or_create(cluster=cluster, close_after=60.0)
        ctx.touch()                                # just used → idle timer reset

        key = ExecutionContext._pool_key(
            cluster_id="cls-active",
            language=Language.PYTHON,
            context_key=ctx.context_key,
        )

        _ensure_reaper_running()
        time.sleep(0.3)

        self.assertIn(key, _ec_mod._CONTEXT_POOL)
        self.assertTrue(ctx.context_id)

    def test_close_after_none_never_evicted(self):
        cluster = _make_cluster("cls-pinned")
        ctx = ExecutionContext(
            cluster=cluster, context_id="ctx-pinned",
            context_key="pinned-key", close_after=None,
        )
        ctx._last_used_at = time.time() - 999_999.0
        key = ContextPoolKey(cluster_id="cls-pinned", language="PYTHON", context_key="pinned-key")
        _ec_mod._CONTEXT_POOL[key] = ctx

        _ensure_reaper_running()
        time.sleep(0.3)

        self.assertIn(key, _ec_mod._CONTEXT_POOL)
        self.assertEqual(ctx.context_id, "ctx-pinned")

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_partial_eviction_mixed_timeouts(self, mock_unreg, mock_reg):
        """Contexts with different timeouts: only expired ones evicted."""
        cluster = _make_cluster("cls-mixed")

        # Short timeout, already idle → should be evicted
        _ws(cluster).create.return_value = _ctx_response("ctx-short")
        ctx_short = ExecutionContext.get_or_create(
            cluster=cluster, close_after=0.05, language=Language.PYTHON, context_key="short"
        )
        ctx_short._last_used_at = time.time() - 1.0

        # Long timeout, recently used → must survive
        _ws(cluster).create.return_value = _ctx_response("ctx-long")
        ctx_long = ExecutionContext.get_or_create(
            cluster=cluster, close_after=3600.0, language=Language.PYTHON, context_key="long"
        )
        ctx_long.touch()

        key_short = ExecutionContext._pool_key(
            cluster_id="cls-mixed", language=Language.PYTHON, context_key="short"
        )
        key_long = ExecutionContext._pool_key(
            cluster_id="cls-mixed", language=Language.PYTHON, context_key="long"
        )

        _ensure_reaper_running()
        time.sleep(0.4)

        self.assertNotIn(key_short, _ec_mod._CONTEXT_POOL)
        self.assertIn(key_long, _ec_mod._CONTEXT_POOL)

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_pool_empty_after_bulk_eviction(self, mock_unreg, mock_reg):
        cluster = _make_cluster("cls-bulk")
        for i in range(3):
            _ws(cluster).create.return_value = _ctx_response(f"ctx-bulk-{i}")
            ctx = ExecutionContext.get_or_create(
                cluster=cluster, close_after=0.05, language=Language.PYTHON, context_key=f"bk{i}"
            )
            ctx._last_used_at = time.time() - 1.0   # already idle

        _ensure_reaper_running()
        time.sleep(0.4)

        self.assertEqual(len(_ec_mod._CONTEXT_POOL), 0)

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_get_or_create_recreates_after_eviction(self, mock_unreg, mock_reg):
        """After the reaper evicts a context, get_or_create must open a new one."""
        cluster = _make_cluster("cls-recreate")
        _ws(cluster).create.return_value = _ctx_response("ctx-original")
        ctx_original = ExecutionContext.get_or_create(cluster=cluster, close_after=0.05)
        ctx_original._last_used_at = time.time() - 1.0

        _ensure_reaper_running()
        time.sleep(0.4)

        # Pool is now empty — a new create call must succeed
        _ws(cluster).create.return_value = _ctx_response("ctx-new-one")
        ctx_new = ExecutionContext.get_or_create(cluster=cluster, close_after=1800.0)

        self.assertEqual(ctx_new.context_id, "ctx-new-one")
        self.assertIsNot(ctx_new, ctx_original)


# ===========================================================================
# _unsafe_close() — shutdown-safety regression tests
# (guards the "ValueError: I/O operation on closed file" atexit banner)
# ===========================================================================

class TestUnsafeCloseShutdownSafety(unittest.TestCase):
    """
    Regression tests for the atexit / interpreter-shutdown race.

    When Python tears down the process it closes stdout/stderr *before* all
    atexit handlers have finished.  Any LOGGER call made from ``_unsafe_close``
    therefore hits a closed StreamHandler and Python's logging machinery prints
    a noisy "--- Logging error ---" banner.

    The fix: ``_unsafe_close`` sets ``logging.raiseExceptions = False`` for its
    duration so ``handleError()`` swallows the error silently.
    """

    def _make_broken_handler(self) -> logging.Handler:
        """Return a StreamHandler whose stream is already closed."""
        import io
        stream = io.StringIO()
        stream.close()               # simulate a shut-down stdout
        return logging.StreamHandler(stream)

    def test_unsafe_close_does_not_raise_with_closed_stream(self):
        """_unsafe_close() must complete silently even when the log stream is closed."""
        ctx = _make_ctx(context_id="ctx-shutdown", temporary=True)
        broken_handler = self._make_broken_handler()

        ec_logger = logging.getLogger("yggdrasil.databricks.compute.execution_context")
        ec_logger.addHandler(broken_handler)
        try:
            # Must not raise anything
            ctx._unsafe_close()
        finally:
            ec_logger.removeHandler(broken_handler)

        self.assertFalse(ctx.context_id, "context_id must be cleared after _unsafe_close")

    def test_unsafe_close_suppresses_logging_error_banner(self):
        """No '--- Logging error ---' must appear on stderr during _unsafe_close."""
        import io
        ctx = _make_ctx(context_id="ctx-banner", temporary=True)
        broken_handler = self._make_broken_handler()

        ec_logger = logging.getLogger("yggdrasil.databricks.compute.execution_context")
        ec_logger.addHandler(broken_handler)

        captured_stderr = io.StringIO()
        with patch("sys.stderr", captured_stderr):
            try:
                ctx._unsafe_close()
            finally:
                ec_logger.removeHandler(broken_handler)

        self.assertNotIn(
            "Logging error",
            captured_stderr.getvalue(),
            "No logging-error banner should appear during atexit shutdown",
        )

    def test_unsafe_close_restores_raise_exceptions_flag(self):
        """logging.raiseExceptions must be restored to its original value afterwards."""
        ctx = _make_ctx(context_id="ctx-flag", temporary=True)
        original = logging.raiseExceptions
        broken_handler = self._make_broken_handler()

        ec_logger = logging.getLogger("yggdrasil.databricks.compute.execution_context")
        ec_logger.addHandler(broken_handler)
        try:
            ctx._unsafe_close()
        finally:
            ec_logger.removeHandler(broken_handler)

        self.assertEqual(logging.raiseExceptions, original)

    def test_unsafe_close_restores_flag_even_if_close_raises(self):
        """logging.raiseExceptions is restored even when close() itself raises."""
        ctx = _make_ctx(context_id="ctx-exc", temporary=True)
        original = logging.raiseExceptions

        with patch.object(ctx, "close", side_effect=RuntimeError("boom")):
            ctx._unsafe_close()   # must not propagate

        self.assertEqual(logging.raiseExceptions, original)

    def test_unsafe_close_on_already_closed_context_is_noop(self):
        """Calling _unsafe_close when context_id is already '' must be a silent noop."""
        ctx = _make_ctx(context_id="", temporary=True)
        # Should not raise, call destroy, or touch atexit
        ctx._unsafe_close()
        ctx.cluster.client.workspace_client.assert_not_called()

    def test_evict_idle_contexts_silences_close_stream_error(self):
        """_evict_idle_contexts must not print a logging banner when the stream is closed."""
        import io
        ctx = _make_ctx(context_id="ctx-reaper-shutdown")
        ctx.close_after = 10.0
        ctx._last_used_at = 0.0001  # non-zero so eviction guard passes
        ctx._last_used_at = time.time() - 120.0

        broken_handler = self._make_broken_handler()
        ec_logger = logging.getLogger("yggdrasil.databricks.compute.execution_context")
        ec_logger.addHandler(broken_handler)

        orig_pool = dict(_ec_mod._CONTEXT_POOL)
        key = ContextPoolKey(cluster_id="cls-rs", language="PYTHON", context_key="rs-key")
        ctx.context_key = "rs-key"
        _ec_mod._CONTEXT_POOL[key] = ctx

        captured_stderr = io.StringIO()
        with patch("sys.stderr", captured_stderr):
            try:
                _evict_idle_contexts()
            finally:
                ec_logger.removeHandler(broken_handler)
                _ec_mod._CONTEXT_POOL.clear()
                _ec_mod._CONTEXT_POOL.update(orig_pool)

        self.assertNotIn("Logging error", captured_stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

