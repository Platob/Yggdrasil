"""
Unit tests for CommandExecution.

Covers: construction, start/cancel/wait/result lifecycle, decode_response,
_cleanup_remote_payload, state properties (done/running), details property,
shutdown hook registration, error handling (InternalError, PermissionDenied,
ClientTerminatedSession, ModuleNotFoundError), and _zip_local_module.
All tests are entirely offline — no real Databricks cluster.
"""

from __future__ import annotations

import io
import unittest
import zipfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

from databricks.sdk.errors import InternalError
from databricks.sdk.service.compute import (
    CommandStatus,
    CommandStatusResponse,
    Language,
    ResultType,
    Results,
)

from yggdrasil.databricks.compute.command_execution import (
    CommandExecution,
    _get_tree_mtime,
)
from yggdrasil.databricks.compute.exceptions import ClientTerminatedSession
from yggdrasil.databricks.compute.execution_context import ExecutionContext


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_cluster(cluster_id: str = "cls-test") -> MagicMock:
    cluster = MagicMock(name="Cluster")
    cluster.cluster_id = cluster_id
    url = MagicMock(name="URL")
    url.with_query_items.return_value = url
    url.query_dict = {}
    url.to_string.return_value = f"https://dbc.example.com/?cluster={cluster_id}"
    cluster.url.return_value = url
    cluster.is_in_databricks_environment.return_value = False
    return cluster


def _ws(cluster: MagicMock):
    return cluster.client.workspace_client.return_value.command_execution


def _make_ctx(
    cluster: Optional[MagicMock] = None,
    *,
    context_id: str = "ctx-test",
    temporary: bool = True,
    context_key: str = "ck-test",
) -> ExecutionContext:
    return ExecutionContext(
        cluster=cluster or _make_cluster(),
        context_id=context_id,
        language=Language.PYTHON,
        temporary=temporary,
        context_key=context_key,
    )


def _make_cmd(
    ctx: Optional[ExecutionContext] = None,
    *,
    command_id: str | None = None,
    command: str = "print(1)",
    language: Language = Language.PYTHON,
    pyfunc=None,
) -> CommandExecution:
    ctx = ctx or _make_ctx()
    return CommandExecution(
        context=ctx,
        command_id=command_id,
        language=language,
        command=command,
        pyfunc=pyfunc,
    )


def _exec_response(command_id: str) -> MagicMock:
    r = MagicMock()
    r.response.id = command_id
    return r


def _status(
    command_id: str,
    status: CommandStatus,
    result_type: ResultType = ResultType.TEXT,
    data: str = "",
) -> CommandStatusResponse:
    return CommandStatusResponse(
        id=command_id,
        status=status,
        results=Results(result_type=result_type, data=data),
    )


def _finished(command_id: str = "cmd-done", data: str = "") -> CommandStatusResponse:
    return _status(command_id, CommandStatus.FINISHED, data=data)


def _running(command_id: str = "cmd-run") -> CommandStatusResponse:
    return _status(command_id, CommandStatus.RUNNING)


def _error(command_id: str = "cmd-err", cause: str = "oops") -> CommandStatusResponse:
    return CommandStatusResponse(
        id=command_id,
        status=CommandStatus.ERROR,
        results=Results(result_type=ResultType.ERROR, cause=cause),
    )


# ===========================================================================
# Construction / __post_init__
# ===========================================================================

class TestCommandExecutionConstruction(unittest.TestCase):

    def test_default_language_inferred_from_context(self):
        ctx = _make_ctx()
        ctx.language = Language.SQL
        cmd = CommandExecution(context=ctx, command="SELECT 1")
        self.assertEqual(cmd.language, Language.SQL)

    def test_language_defaults_to_python_when_pyfunc_given(self):
        ctx = _make_ctx()
        cmd = CommandExecution(context=ctx, pyfunc=lambda: 1)
        self.assertEqual(cmd.language, Language.PYTHON)

    def test_environ_dict_conversion(self):
        ctx = _make_ctx()
        with self.assertRaises(ValueError):
            CommandExecution(context=ctx, command="pass", environ=object())

    def test_shutdown_hook_initially_none(self):
        ctx = _make_ctx()
        cmd = CommandExecution(context=ctx, command="pass")
        self.assertIsNone(cmd._shutdown_hook)


# ===========================================================================
# url / __repr__ / __str__
# ===========================================================================

class TestCommandExecutionRepr(unittest.TestCase):

    def test_repr_contains_class_name(self):
        cmd = _make_cmd(command_id="cmd-repr")
        self.assertIn("CommandExecution", repr(cmd))

    def test_str_is_string(self):
        cmd = _make_cmd()
        self.assertIsInstance(str(cmd), str)


# ===========================================================================
# State properties: done / running / state
# ===========================================================================

class TestCommandExecutionState(unittest.TestCase):

    def test_done_true_when_finished(self):
        cmd = _make_cmd(command_id="cmd-1")
        cmd._details = _finished("cmd-1")
        self.assertTrue(cmd.done)

    def test_done_false_when_running(self):
        cmd = _make_cmd(command_id="cmd-2")
        cmd._details = _running("cmd-2")
        self.assertFalse(cmd.done)

    def test_running_true_when_running(self):
        cmd = _make_cmd(command_id="cmd-3")
        cmd._details = _running("cmd-3")
        _ws(cmd.context.cluster).command_status.return_value = _running("cmd-3")
        self.assertTrue(cmd.running)

    def test_running_false_when_finished(self):
        cmd = _make_cmd(command_id="cmd-4")
        cmd._details = _finished("cmd-4")
        self.assertFalse(cmd.running)

    def test_state_none_when_no_command_id(self):
        cmd = _make_cmd(command_id=None)
        self.assertIsNone(cmd.state)

    def test_done_false_when_no_command_id(self):
        cmd = _make_cmd(command_id=None)
        self.assertFalse(cmd.done)

    def test_running_false_when_no_command_id(self):
        cmd = _make_cmd(command_id=None)
        self.assertFalse(cmd.running)


# ===========================================================================
# details property
# ===========================================================================

class TestCommandExecutionDetails(unittest.TestCase):

    def test_details_returns_cached_done_response(self):
        cmd = _make_cmd(command_id="cmd-d1")
        resp = _finished("cmd-d1")
        cmd._details = resp
        self.assertIs(cmd.details, resp)

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_details_done_unregisters_shutdown_hook(self, mock_unreg):
        cmd = _make_cmd(command_id="cmd-d2")
        cmd._shutdown_hook = cmd._unsafe_cancel
        cmd._details = _finished("cmd-d2")
        _ = cmd.details
        mock_unreg.assert_called_with(cmd._unsafe_cancel)

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_details_running_does_not_unregister_shutdown_hook(self, mock_unreg):
        cmd = _make_cmd(command_id="cmd-d3")
        cmd._shutdown_hook = cmd._unsafe_cancel
        cmd._details = _running("cmd-d3")
        _ws(cmd.context.cluster).command_status.return_value = _running("cmd-d3")
        _ = cmd.details
        mock_unreg.assert_not_called()

    def test_details_fetches_from_api_when_running(self):
        cmd = _make_cmd(command_id="cmd-d4")
        cmd._details = _running("cmd-d4")
        _ws(cmd.context.cluster).command_status.return_value = _finished("cmd-d4")
        d = cmd.details
        _ws(cmd.context.cluster).command_status.assert_called_once()
        self.assertEqual(d.status, CommandStatus.FINISHED)

    def test_details_returns_finished_when_no_command_id(self):
        cmd = _make_cmd(command_id=None)
        d = cmd.details
        self.assertEqual(d.status, CommandStatus.FINISHED)


# ===========================================================================
# start()
# ===========================================================================

class TestCommandExecutionStart(unittest.TestCase):

    @patch("yggdrasil.environ.shutdown.register")
    def test_start_sets_command_id(self, _):
        ctx = _make_ctx(context_id="ctx-s1")
        _ws(ctx.cluster).execute.return_value = _exec_response("cmd-s1")
        cmd = _make_cmd(ctx=ctx)
        cmd.start()
        self.assertEqual(cmd.command_id, "cmd-s1")

    @patch("yggdrasil.environ.shutdown.register")
    def test_start_registers_shutdown_hook(self, mock_reg):
        ctx = _make_ctx(context_id="ctx-s2")
        _ws(ctx.cluster).execute.return_value = _exec_response("cmd-s2")
        mock_reg.return_value = None
        cmd = _make_cmd(ctx=ctx)
        cmd.start()
        mock_reg.assert_called_once()
        self.assertIs(mock_reg.call_args[0][0].__self__, cmd)

    def test_start_no_reset_returns_self_when_running(self):
        ctx = _make_ctx(context_id="ctx-s4")
        cmd = _make_cmd(ctx=ctx, command_id="cmd-s4-existing")
        with patch("yggdrasil.environ.shutdown.register") as mock_reg:
            result = cmd.start()
        self.assertIs(result, cmd)
        mock_reg.assert_not_called()
        _ws(ctx.cluster).execute.assert_not_called()

    @patch("yggdrasil.environ.shutdown.register")
    def test_start_asserts_command_present(self, _):
        ctx = _make_ctx(context_id="ctx-s5")
        cmd = _make_cmd(ctx=ctx, command="")
        cmd.command = None
        with self.assertRaises(AssertionError):
            cmd.start()

    # ------------------------------------------------------------------
    # start(reset=True)
    # ------------------------------------------------------------------

    @patch("yggdrasil.environ.shutdown.unregister")
    @patch("yggdrasil.environ.shutdown.register")
    def test_start_reset_running_cancels_and_reregisters(self, mock_reg, mock_unreg):
        ctx = _make_ctx(context_id="ctx-sr1")
        ws = _ws(ctx.cluster)
        ws.execute.return_value = _exec_response("cmd-new-sr1")
        ws.cancel.return_value = MagicMock()
        cmd = _make_cmd(ctx=ctx, command_id="cmd-old-sr1")
        cmd._details = _running("cmd-old-sr1")
        cmd._shutdown_hook = cmd._unsafe_cancel
        mock_reg.return_value = cmd._unsafe_cancel
        cmd.start(reset=True)
        ws.cancel.assert_called_once()
        mock_unreg.assert_called()
        mock_reg.assert_called_once()
        self.assertEqual(cmd.command_id, "cmd-new-sr1")

    @patch("yggdrasil.environ.shutdown.unregister")
    @patch("yggdrasil.environ.shutdown.register")
    def test_start_reset_done_unregisters_before_reregistering(self, mock_reg, mock_unreg):
        """Bug guard: done-reset must not leave a duplicate shutdown entry."""
        ctx = _make_ctx(context_id="ctx-sr2")
        _ws(ctx.cluster).execute.return_value = _exec_response("cmd-new-sr2")
        cmd = _make_cmd(ctx=ctx, command_id="cmd-done-sr2")
        cmd._details = _finished("cmd-done-sr2")
        cmd._shutdown_hook = cmd._unsafe_cancel
        mock_reg.return_value = cmd._unsafe_cancel
        cmd.start(reset=True)
        mock_unreg.assert_called()
        mock_reg.assert_called_once()
        self.assertEqual(cmd.command_id, "cmd-new-sr2")

    def test_start_reset_done_registration_count_is_one(self):
        """Verify exactly one unregister + one re-register happens on done-reset."""
        ctx = _make_ctx(context_id="ctx-sr3")
        _ws(ctx.cluster).execute.return_value = _exec_response("cmd-new-sr3")
        cmd = _make_cmd(ctx=ctx, command_id="cmd-done-sr3")
        cmd._details = _finished("cmd-done-sr3")
        cmd._shutdown_hook = cmd._unsafe_cancel

        registered: list = []
        unregistered: list = []

        def fake_register(f, **kw):
            registered.append(f)
            return f

        def fake_unregister(f):
            unregistered.append(f)
            return True

        with patch("yggdrasil.environ.shutdown.register", side_effect=fake_register), \
             patch("yggdrasil.environ.shutdown.unregister", side_effect=fake_unregister):
            cmd.start(reset=True)

        self.assertEqual(len(unregistered), 1)
        self.assertEqual(len(registered), 1)
        self.assertIsNotNone(cmd._shutdown_hook)

    # ------------------------------------------------------------------
    # start() — error handling
    # ------------------------------------------------------------------

    @patch("yggdrasil.environ.shutdown.register")
    def test_start_retries_on_internal_error(self, _):
        ctx = _make_ctx(context_id="ctx-se1")
        ws = _ws(ctx.cluster)
        ws.execute.side_effect = [InternalError("boom"), _exec_response("cmd-se1-retry")]
        ctx.cluster.client.workspace_client.return_value.command_execution.create.return_value = (
            MagicMock(response=MagicMock(id="ctx-se1-new"))
        )
        cmd = _make_cmd(ctx=ctx)
        cmd.start()
        self.assertEqual(ws.execute.call_count, 2)

    @patch("yggdrasil.environ.shutdown.register")
    def test_start_retries_on_context_id_error(self, _):
        ctx = _make_ctx(context_id="ctx-se2")
        ws = _ws(ctx.cluster)
        ws.execute.side_effect = [RuntimeError("invalid context_id"), _exec_response("cmd-se2-retry")]
        ctx.cluster.client.workspace_client.return_value.command_execution.create.return_value = (
            MagicMock(response=MagicMock(id="ctx-se2-new"))
        )
        cmd = _make_cmd(ctx=ctx)
        cmd.start()
        self.assertEqual(ws.execute.call_count, 2)

    @patch("yggdrasil.environ.shutdown.register")
    def test_start_propagates_unrecognised_error(self, _):
        ctx = _make_ctx(context_id="ctx-se3")
        _ws(ctx.cluster).execute.side_effect = RuntimeError("totally unexpected")
        cmd = _make_cmd(ctx=ctx)
        with self.assertRaises(RuntimeError):
            cmd.start()


# ===========================================================================
# cancel()
# ===========================================================================

class TestCommandExecutionCancel(unittest.TestCase):

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_cancel_calls_api_and_clears_command_id(self, _):
        ctx = _make_ctx(context_id="ctx-c1")
        _ws(ctx.cluster).cancel.return_value = MagicMock()
        cmd = _make_cmd(ctx=ctx, command_id="cmd-c1")
        cmd.cancel()
        _ws(ctx.cluster).cancel.assert_called_once()
        self.assertIsNone(cmd.command_id)

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_cancel_unregisters_shutdown_hook(self, mock_unreg):
        ctx = _make_ctx(context_id="ctx-c2")
        _ws(ctx.cluster).cancel.return_value = MagicMock()
        cmd = _make_cmd(ctx=ctx, command_id="cmd-c2")
        cmd._shutdown_hook = cmd._unsafe_cancel
        cmd.cancel()
        mock_unreg.assert_called_once_with(cmd._unsafe_cancel)

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_cancel_no_command_id_is_noop(self, mock_unreg):
        ctx = _make_ctx(context_id="ctx-c4")
        cmd = _make_cmd(ctx=ctx, command_id=None)
        cmd.cancel()
        _ws(ctx.cluster).cancel.assert_not_called()
        mock_unreg.assert_not_called()

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_cancel_no_context_id_is_noop(self, mock_unreg):
        ctx = _make_ctx(context_id="")
        cmd = _make_cmd(ctx=ctx, command_id="cmd-c5")
        cmd.cancel()
        _ws(ctx.cluster).cancel.assert_not_called()

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_cancel_suppresses_errors_by_default(self, _):
        ctx = _make_ctx(context_id="ctx-c6")
        _ws(ctx.cluster).cancel.side_effect = RuntimeError("oops")
        cmd = _make_cmd(ctx=ctx, command_id="cmd-c6")
        cmd.cancel(raise_error=False)
        self.assertIsNone(cmd.command_id)

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_cancel_propagates_when_raise_error_true(self, _):
        ctx = _make_ctx(context_id="ctx-c7")
        _ws(ctx.cluster).cancel.side_effect = RuntimeError("oops")
        cmd = _make_cmd(ctx=ctx, command_id="cmd-c7")
        with self.assertRaises(RuntimeError):
            cmd.cancel(raise_error=True)

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_unsafe_cancel_idempotent(self, _):
        ctx = _make_ctx(context_id="ctx-c8")
        _ws(ctx.cluster).cancel.return_value = MagicMock()
        cmd = _make_cmd(ctx=ctx, command_id="cmd-c8")
        cmd._unsafe_cancel()
        cmd._unsafe_cancel()
        self.assertIsNone(cmd.command_id)
        self.assertEqual(_ws(ctx.cluster).cancel.call_count, 1)


# ===========================================================================
# decode_response()
# ===========================================================================

class TestDecodeResponse(unittest.TestCase):

    def _make_text_response(self, data: str) -> CommandStatusResponse:
        return CommandStatusResponse(
            id="cmd-dr",
            status=CommandStatus.FINISHED,
            results=Results(result_type=ResultType.TEXT, data=data),
        )

    def test_no_tag_returns_none_logs_and_raw(self):
        cmd = _make_cmd()
        resp = self._make_text_response("hello world")
        logs, raw = cmd.decode_response(resp, Language.PYTHON, raise_error=False)
        self.assertIsNone(logs)
        self.assertEqual(raw, "hello world")

    def test_tag_splits_correctly(self):
        cmd = _make_cmd()
        resp = self._make_text_response("some logs\n__CALL_RESULT__result_payload")
        logs, raw = cmd.decode_response(resp, Language.PYTHON, raise_error=False)
        self.assertEqual(logs, "some logs\n")
        self.assertEqual(raw, "result_payload")

    def test_custom_tag(self):
        cmd = _make_cmd()
        resp = self._make_text_response("preamble||VALUE")
        logs, raw = cmd.decode_response(resp, Language.PYTHON, raise_error=False, tag="||")
        self.assertEqual(logs, "preamble")
        self.assertEqual(raw, "VALUE")

    def test_error_result_raises_when_raise_error_true(self):
        from yggdrasil.databricks.compute.command_execution import raise_error_from_response
        resp = CommandStatusResponse(
            id="cmd-err",
            status=CommandStatus.ERROR,
            results=Results(result_type=ResultType.ERROR, cause="Something went wrong"),
        )
        with self.assertRaises(Exception):
            raise_error_from_response(resp, Language.PYTHON, raise_error=True)

    def test_client_terminated_session_raised_on_keyword(self):
        from yggdrasil.databricks.compute.command_execution import raise_error_from_response
        resp = CommandStatusResponse(
            id="cmd-cts",
            status=CommandStatus.ERROR,
            results=Results(result_type=ResultType.ERROR, cause="client terminated the session unexpectedly"),
        )
        with self.assertRaises(ClientTerminatedSession):
            raise_error_from_response(resp, Language.PYTHON, raise_error=True)


# ===========================================================================
# _cleanup_remote_payload()
# ===========================================================================

class TestCleanupRemotePayload(unittest.TestCase):

    def test_no_payload_path_is_noop(self):
        cmd = _make_cmd()
        cmd._remote_payload_path = None
        cmd._cleanup_remote_payload()
        cmd.context.cluster.client.dbfs_path.assert_not_called()

    def test_removes_payload_and_clears_path(self):
        cmd = _make_cmd()
        cmd._remote_payload_path = "/dbfs/tmp/payload.b64"
        cmd._cleanup_remote_payload()
        cmd.context.cluster.client.dbfs_path.assert_called_once_with("/dbfs/tmp/payload.b64")
        self.assertIsNone(cmd._remote_payload_path)

    def test_suppresses_removal_error(self):
        cmd = _make_cmd()
        cmd._remote_payload_path = "/dbfs/tmp/missing.b64"
        cmd.context.cluster.client.dbfs_path.return_value.remove.side_effect = Exception("gone")
        cmd._cleanup_remote_payload()
        self.assertIsNone(cmd._remote_payload_path)


# ===========================================================================
# _result_inner() — retry and success paths
# ===========================================================================

class TestResultInner(unittest.TestCase):

    def _cmd_with_finished_details(self, data: str = "") -> CommandExecution:
        cmd = _make_cmd(command_id="cmd-ri")
        cmd._details = _finished("cmd-ri", data=data)
        return cmd

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_result_inner_returns_raw_when_no_tag(self, _):
        cmd = self._cmd_with_finished_details("plain output")
        result = cmd._result_inner(wait=False, raise_error=False)
        self.assertEqual(result, "plain output")

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_result_inner_deserialises_tagged_payload(self, _):
        from yggdrasil.pickle.ser import dumps
        payload = dumps({"key": "value"}, b64=True)
        cmd = self._cmd_with_finished_details(f"logs\n__CALL_RESULT__{payload}")
        result = cmd._result_inner(wait=False, raise_error=True)
        self.assertEqual(result, {"key": "value"})

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_result_inner_json_list_parsed(self, _):
        cmd = self._cmd_with_finished_details('__CALL_RESULT__["a","b","c"]')
        result = cmd._result_inner(wait=False, raise_error=True)
        self.assertEqual(result, ["a", "b", "c"])

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_result_inner_json_dict_parsed(self, _):
        cmd = self._cmd_with_finished_details('__CALL_RESULT__{"x": 1}')
        result = cmd._result_inner(wait=False, raise_error=True)
        self.assertEqual(result, {"x": 1})

    @patch("yggdrasil.environ.shutdown.register")
    @patch("yggdrasil.environ.shutdown.unregister")
    def test_result_inner_retries_on_internal_error(self, mock_unreg, mock_reg):
        ctx = _make_ctx(context_id="ctx-ri2")
        ws = _ws(ctx.cluster)
        ws.execute.return_value = _exec_response("cmd-ri2-new")
        ws.create.return_value = MagicMock(response=MagicMock(id="ctx-ri2-new"))
        cmd = _make_cmd(ctx=ctx, command_id="cmd-ri2")
        call_count = {"n": 0}

        def details_side_effect(self_cmd):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise InternalError("cluster down")
            return _finished("cmd-ri2-new", data="ok")

        with patch.object(CommandExecution, "details", new_callable=lambda: property(details_side_effect)):
            result = cmd._result_inner(wait=False, raise_error=True)

        ws.execute.assert_called()
        self.assertEqual(result, "ok")

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_result_inner_raises_when_exhausted_and_raise_error_true(self, _):
        cmd = _make_cmd(command_id="cmd-ri3")
        cmd._details = _error("cmd-ri3", cause="something broke")
        with self.assertRaises(Exception):
            cmd._result_inner(wait=False, raise_error=True)

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_result_inner_no_raise_returns_raw_on_empty_output(self, _):
        cmd = _make_cmd(command_id="cmd-ri4")
        cmd._details = _finished("cmd-ri4", data="")
        result = cmd._result_inner(wait=False, raise_error=False)
        self.assertEqual(result, "")


# ===========================================================================
# wait()
# ===========================================================================

class TestCommandExecutionWait(unittest.TestCase):

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_wait_on_finished_command_returns_self(self, _):
        cmd = _make_cmd(command_id="cmd-w1")
        cmd._details = _finished("cmd-w1")
        result = cmd.wait(wait=False, raise_error=False)
        self.assertIs(result, cmd)

    @patch("yggdrasil.environ.shutdown.register")
    def test_wait_on_no_command_id_calls_start_first(self, _):
        ctx = _make_ctx(context_id="ctx-w2")
        _ws(ctx.cluster).execute.return_value = _exec_response("cmd-w2")
        cmd = _make_cmd(ctx=ctx, command_id=None)
        with patch.object(type(cmd), "details", new_callable=lambda: property(lambda self: _finished("cmd-w2"))):
            cmd.wait(wait=True, raise_error=False)
        self.assertEqual(cmd.command_id, "cmd-w2")

    @patch("yggdrasil.environ.shutdown.unregister")
    def test_wait_raises_for_error_state_when_raise_error_true(self, _):
        cmd = _make_cmd(command_id="cmd-w3")
        cmd._details = _error("cmd-w3", cause="crash")
        with self.assertRaises(Exception):
            cmd.wait(wait=False, raise_error=True)


# ===========================================================================
# _zip_local_module()
# ===========================================================================

class TestZipLocalModule(unittest.TestCase):

    def test_zips_single_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mymod.py"
            p.write_text("x = 1")
            name, data = CommandExecution._zip_local_module(p)
            self.assertEqual(name, "mymod.py")
            zf = zipfile.ZipFile(io.BytesIO(data))
            self.assertIn("mymod.py", zf.namelist())

    def test_zips_directory_skipping_pycache(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "mypkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "core.py").write_text("pass")
            cache = pkg / "__pycache__"
            cache.mkdir()
            (cache / "core.cpython-312.pyc").write_bytes(b"\x00")
            name, data = CommandExecution._zip_local_module(pkg)
            zf = zipfile.ZipFile(io.BytesIO(data))
            names = zf.namelist()
            self.assertTrue(any("mypkg" in n for n in names))
            self.assertFalse(any("__pycache__" in n for n in names))


# ===========================================================================
# _get_tree_mtime()
# ===========================================================================

class TestGetTreeMtime(unittest.TestCase):

    def test_single_file(self):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            p = Path(f.name)
        try:
            mtime = _get_tree_mtime(p)
            self.assertGreater(mtime, 0)
        finally:
            p.unlink(missing_ok=True)

    def test_directory_returns_max_mtime(self):
        import tempfile, time
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.py").write_text("a")
            time.sleep(0.01)
            f2 = root / "b.py"
            f2.write_text("b")
            mtime = _get_tree_mtime(root)
            self.assertGreaterEqual(mtime, f2.stat().st_mtime)


# ===========================================================================
# install_module — blacklist filtering via cluster.install_libraries
# ===========================================================================

class TestInstallModuleBlacklist(unittest.TestCase):
    """
    Verify that install_module no longer calls a private _remote_pip_install
    method, and that blacklisted packages are silently skipped by
    cluster.install_libraries.
    """

    def test_no_remote_pip_install_method(self):
        """_remote_pip_install must not exist on CommandExecution anymore."""
        self.assertFalse(hasattr(CommandExecution, "_remote_pip_install"))

    def test_install_libraries_called_directly_for_site_packages(self):
        """install_module must call cluster.install_libraries for site-packages specs."""
        ctx = _make_ctx(context_id="ctx-pip")
        cmd = _make_cmd(ctx=ctx)

        fake_path = MagicMock(spec=Path)
        fake_path.parts = ("usr", "local", "lib", "site-packages", "mypkg")
        fake_path.name = "mypkg"
        fake_path.is_dir.return_value = True

        with patch.object(CommandExecution, "_get_local_module_path", return_value=fake_path), \
             patch("yggdrasil.databricks.compute.command_execution._get_local_distribution_compatible_spec",
                   return_value="mypkg~=1.2") as mock_spec, \
             patch.object(ctx.cluster, "install_libraries") as mock_install:
            result = cmd.install_module("mypkg")

        mock_install.assert_called_once_with(
            libraries=["mypkg~=1.2"],
            raise_error=True,
            remove_failed=True,
        )
        self.assertEqual(result, "mypkg~=1.2")

    def test_install_libraries_skips_none_spec(self):
        """When spec resolves to None (excluded lib), install_libraries is not called."""
        ctx = _make_ctx(context_id="ctx-pip-none")
        cmd = _make_cmd(ctx=ctx)

        fake_path = MagicMock(spec=Path)
        fake_path.parts = ("usr", "local", "lib", "site-packages", "ygg")
        fake_path.name = "ygg"
        fake_path.is_dir.return_value = True

        with patch.object(CommandExecution, "_get_local_module_path", return_value=fake_path), \
             patch("yggdrasil.databricks.compute.command_execution._get_local_distribution_compatible_spec",
                   return_value=None), \
             patch.object(ctx.cluster, "install_libraries") as mock_install:
            result = cmd.install_module("ygg")

        mock_install.assert_not_called()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

