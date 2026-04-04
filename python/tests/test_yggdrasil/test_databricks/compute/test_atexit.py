"""
Unit tests for atexit lifecycle management in ExecutionContext and CommandExecution.

These tests are entirely offline — no real Databricks cluster is needed.
Every external call is intercepted with ``unittest.mock``.

Bugs fixed (and guarded by tests below)
----------------------------------------
1. ``ExecutionContext.__enter__``:  guard was ``is None``; default ``context_id``
   is ``""`` so a fresh instance never created a context.  Fixed to ``not self.context_id``.

2. ``ExecutionContext.connect()``:  guard was ``is not None``; same root cause —
   ``""`` passed the guard and returned ``self`` without connecting.
   Fixed to ``if self.context_id:``.

3. ``CommandExecution.start(reset=True)`` when the command is already *done*:
   ``cancel()`` was skipped (correct — nothing to cancel), but ``cancel()`` is
   the only place ``atexit.unregister`` was called.  The old registration was
   never removed, so the subsequent ``atexit.register`` duplicated the entry.
   Fixed by an explicit ``atexit.unregister`` in the done-reset branch.
"""

from __future__ import annotations

import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

from databricks.sdk.service.compute import (
    CommandStatus,
    CommandStatusResponse,
    Language,
    ResultType,
    Results,
)

from yggdrasil.databricks.compute.command_execution import CommandExecution
from yggdrasil.databricks.compute.execution_context import ExecutionContext


# ---------------------------------------------------------------------------
# Test-object factories
# ---------------------------------------------------------------------------

def _make_cluster(cluster_id: str = "test-cluster-abc") -> MagicMock:
    """Return a minimal Cluster mock that satisfies ExecutionContext's interface."""
    cluster = MagicMock(name="Cluster")
    cluster.cluster_id = cluster_id
    cluster.url.return_value = MagicMock(name="URL")
    cluster.url.return_value.with_query_items.return_value = MagicMock(name="URL.with_query")
    cluster.url.return_value.with_query_items.return_value.to_string.return_value = (
        f"https://example.databricks.com/?cluster={cluster_id}"
    )
    return cluster


def _ws_cmd_exec(cluster: MagicMock):
    """Shortcut to the mocked command_execution service on the workspace client."""
    return cluster.client.workspace_client.return_value.command_execution


def _make_ctx(
    cluster: Optional[MagicMock] = None,
    *,
    context_id: str = "",
    temporary: bool = True,
    language: Language = Language.PYTHON,
) -> ExecutionContext:
    """Return a minimal ExecutionContext backed by mocks."""
    cluster = cluster or _make_cluster()
    return ExecutionContext(
        cluster=cluster,
        context_id=context_id,
        language=language,
        temporary=temporary,
    )


def _make_cmd(
    ctx: Optional[ExecutionContext] = None,
    *,
    command_id: Optional[str] = None,
    temporary: bool = True,
    command: str = "print(1)",
) -> CommandExecution:
    """Return a minimal CommandExecution backed by mocks."""
    ctx = ctx or _make_ctx()
    return CommandExecution(
        context=ctx,
        command_id=command_id,
        language=Language.PYTHON,
        command=command,
        temporary=temporary,
    )


def _done_response(command_id: str = "cmd-done") -> CommandStatusResponse:
    return CommandStatusResponse(
        id=command_id,
        status=CommandStatus.FINISHED,
        results=Results(result_type=ResultType.TEXT, data=""),
    )


def _running_response(command_id: str = "cmd-run") -> CommandStatusResponse:
    return CommandStatusResponse(
        id=command_id,
        status=CommandStatus.RUNNING,
    )


def _execute_response(command_id: str) -> MagicMock:
    """Simulate client.execute(...).response with a specific command_id."""
    resp = MagicMock()
    resp.response.id = command_id
    return resp


def _create_response(context_id: str) -> MagicMock:
    """Simulate client.create(...).response with a specific context_id."""
    resp = MagicMock()
    resp.response.id = context_id
    return resp


# ===========================================================================
# ExecutionContext — atexit tests
# ===========================================================================

class TestExecutionContextAtexit(unittest.TestCase):

    # -----------------------------------------------------------------------
    # create()
    # -----------------------------------------------------------------------

    @patch("atexit.register")
    def test_create_temporary_registers_atexit(self, mock_register: MagicMock):
        """create(temporary=True) must register _unsafe_close with atexit."""
        ctx = _make_ctx(temporary=False)
        _ws_cmd_exec(ctx.cluster).create.return_value = _create_response("ctx-001")

        instance = ctx.create(language=Language.PYTHON, temporary=True)

        mock_register.assert_called_once_with(instance._unsafe_close)
        self.assertEqual(instance.context_id, "ctx-001")
        self.assertTrue(instance.temporary)

    @patch("atexit.register")
    def test_create_non_temporary_does_not_register_atexit(self, mock_register: MagicMock):
        """create(temporary=False) must never touch atexit."""
        ctx = _make_ctx(temporary=False)
        _ws_cmd_exec(ctx.cluster).create.return_value = _create_response("ctx-002")

        ctx.create(language=Language.PYTHON, temporary=False)

        mock_register.assert_not_called()

    @patch("atexit.register")
    def test_create_reuses_matching_context_without_network_call(self, mock_register: MagicMock):
        """create() with matching context_id and language must return self immediately."""
        ctx = _make_ctx(context_id="ctx-existing", temporary=True)

        result = ctx.create(language=Language.PYTHON)

        self.assertIs(result, ctx)
        _ws_cmd_exec(ctx.cluster).create.assert_not_called()
        mock_register.assert_not_called()

    # -----------------------------------------------------------------------
    # close()
    # -----------------------------------------------------------------------

    @patch("atexit.unregister")
    def test_close_temporary_unregisters_atexit(self, mock_unregister: MagicMock):
        """close() on a temporary context must unregister its _unsafe_close."""
        ctx = _make_ctx(context_id="ctx-100", temporary=True)

        ctx.close(wait=False, raise_error=False)

        mock_unregister.assert_called_once()
        fn = mock_unregister.call_args[0][0]
        self.assertIs(fn.__self__, ctx)

    @patch("atexit.unregister")
    def test_close_non_temporary_does_not_touch_atexit(self, mock_unregister: MagicMock):
        """close() on a non-temporary context must not touch atexit."""
        ctx = _make_ctx(context_id="ctx-101", temporary=False)

        ctx.close(wait=False, raise_error=False)

        mock_unregister.assert_not_called()

    @patch("atexit.unregister")
    def test_close_empty_context_id_is_noop(self, mock_unregister: MagicMock):
        """close() when context_id='' must return immediately without any network call."""
        ctx = _make_ctx(context_id="", temporary=True)

        ctx.close(wait=False, raise_error=False)

        ctx.cluster.client.workspace_client.assert_not_called()
        mock_unregister.assert_not_called()

    @patch("atexit.unregister")
    def test_close_clears_context_id(self, _):
        """close() must set context_id to None after destruction."""
        ctx = _make_ctx(context_id="ctx-102", temporary=True)

        ctx.close(wait=False, raise_error=False)

        self.assertEqual(ctx.context_id, "")

    @patch("atexit.unregister")
    def test_close_idempotent_unregisters_only_once(self, mock_unregister: MagicMock):
        """Calling close() twice must only unregister once (second call is a no-op)."""
        ctx = _make_ctx(context_id="ctx-103", temporary=True)

        ctx.close(wait=False, raise_error=False)   # real close
        ctx.close(wait=False, raise_error=False)   # no-op (context_id is now None)

        self.assertEqual(mock_unregister.call_count, 1)

    # -----------------------------------------------------------------------
    # __enter__ — bug guard: context_id="" must create a new context
    # -----------------------------------------------------------------------

    @patch("atexit.register")
    def test_enter_empty_string_context_id_creates_new_context(self, mock_register: MagicMock):
        """
        Bug guard: context_id defaults to "" (falsy), NOT None.
        __enter__ must treat "" as 'not connected' and call create().

        Previously the guard was ``if self.context_id is None`` which never
        fired for the default empty-string, so the context manager silently
        returned an unconnected self.

        Note: create() mutates self in-place and returns self — the returned
        object IS the same instance, now connected.
        """
        ctx = _make_ctx(context_id="", temporary=True)
        _ws_cmd_exec(ctx.cluster).create.return_value = _create_response("ctx-new")

        result = ctx.__enter__()

        # create() mutates ctx in-place and returns self
        self.assertIs(result, ctx)
        self.assertEqual(result.context_id, "ctx-new")

    @patch("atexit.register")
    def test_enter_non_empty_context_id_reuses_without_create(self, mock_register: MagicMock):
        """__enter__ with an existing context_id must return self without calling create()."""
        ctx = _make_ctx(context_id="ctx-existing", temporary=True)

        result = ctx.__enter__()

        self.assertIs(result, ctx)
        _ws_cmd_exec(ctx.cluster).create.assert_not_called()
        mock_register.assert_not_called()

    # -----------------------------------------------------------------------
    # __exit__
    # -----------------------------------------------------------------------

    @patch("atexit.unregister")
    def test_exit_calls_close(self, _):
        """__exit__ must close the context (context_id becomes falsy)."""
        ctx = _make_ctx(context_id="ctx-200", temporary=True)

        ctx.__exit__(None, None, None)

        # close() sets context_id to "" (empty string, falsy) — not None
        self.assertFalse(ctx.context_id)

    # -----------------------------------------------------------------------
    # connect() — bug guard: context_id="" must NOT be treated as "connected"
    # -----------------------------------------------------------------------

    @patch("atexit.register")
    def test_connect_empty_string_context_id_creates_new_context(self, mock_register: MagicMock):
        """
        Bug guard: connect() guard was ``is not None``.  Empty string passed
        the guard, causing connect() to return self (unconnected) instead of
        calling create().
        """
        ctx = _make_ctx(context_id="", temporary=True)
        _ws_cmd_exec(ctx.cluster).create.return_value = _create_response("ctx-300")

        result = ctx.connect()

        # A new connected instance must be returned
        self.assertNotEqual(result.context_id, "")
        self.assertEqual(result.context_id, "ctx-300")

    @patch("atexit.register")
    def test_connect_non_empty_context_id_reuses(self, _):
        """connect() with an existing context_id and no reset must return self."""
        ctx = _make_ctx(context_id="ctx-existing", temporary=True)

        result = ctx.connect()

        self.assertIs(result, ctx)
        _ws_cmd_exec(ctx.cluster).create.assert_not_called()

    @patch("atexit.register")
    @patch("atexit.unregister")
    def test_connect_reset_closes_old_and_creates_new(self, mock_unregister, mock_register):
        """connect(reset=True) must close the old context and create a new one."""
        ctx = _make_ctx(context_id="ctx-old", temporary=True)
        _ws_cmd_exec(ctx.cluster).create.return_value = _create_response("ctx-new")

        result = ctx.connect(reset=True)

        # connect() mutates ctx in-place: close → create → return self
        # After the call, ctx IS result and has the new context_id
        self.assertIs(result, ctx)
        self.assertEqual(result.context_id, "ctx-new")
        mock_register.assert_called_once()


# ===========================================================================
# CommandExecution — atexit tests
# ===========================================================================

class TestCommandExecutionAtexit(unittest.TestCase):

    # -----------------------------------------------------------------------
    # start()
    # -----------------------------------------------------------------------

    @patch("atexit.register")
    def test_start_temporary_registers_atexit(self, mock_register: MagicMock):
        """start() on a temporary CommandExecution must register _unsafe_cancel."""
        ctx = _make_ctx(context_id="ctx-400", temporary=True)
        _ws_cmd_exec(ctx.cluster).execute.return_value = _execute_response("cmd-400")

        cmd = _make_cmd(ctx=ctx, temporary=True)
        cmd.start()

        mock_register.assert_called_once()
        fn = mock_register.call_args[0][0]
        self.assertIs(fn.__self__, cmd)
        self.assertEqual(cmd.command_id, "cmd-400")

    @patch("atexit.register")
    def test_start_non_temporary_does_not_register_atexit(self, mock_register: MagicMock):
        """start() on a non-temporary CommandExecution must not touch atexit."""
        ctx = _make_ctx(context_id="ctx-401", temporary=False)
        _ws_cmd_exec(ctx.cluster).execute.return_value = _execute_response("cmd-401")

        cmd = _make_cmd(ctx=ctx, temporary=False)
        cmd.start()

        mock_register.assert_not_called()

    def test_start_no_reset_with_existing_command_id_returns_self(self):
        """start() without reset on a command that is already started must return self immediately."""
        ctx = _make_ctx(context_id="ctx-402", temporary=True)
        cmd = _make_cmd(ctx=ctx, command_id="cmd-already", temporary=True)

        with patch("atexit.register") as mock_reg:
            result = cmd.start()

        self.assertIs(result, cmd)
        mock_reg.assert_not_called()
        _ws_cmd_exec(ctx.cluster).execute.assert_not_called()

    # -----------------------------------------------------------------------
    # start(reset=True) — running command
    # -----------------------------------------------------------------------

    @patch("atexit.unregister")
    @patch("atexit.register")
    def test_start_reset_running_command_cancels_then_reregisters(
        self, mock_register: MagicMock, mock_unregister: MagicMock
    ):
        """
        reset on a RUNNING command must:
          1. cancel() the old command → unregister is called inside cancel()
          2. start the new command → register is called once
        """
        ctx = _make_ctx(context_id="ctx-500", temporary=True)
        ws = _ws_cmd_exec(ctx.cluster)
        ws.execute.return_value = _execute_response("cmd-new-500")
        ws.cancel.return_value = MagicMock()

        cmd = _make_cmd(ctx=ctx, command_id="cmd-old-500", temporary=True)
        cmd._details = _running_response("cmd-old-500")  # force running state

        cmd.start(reset=True)

        # cancel() was invoked (done=False path) → unregister happened inside it
        ws.cancel.assert_called_once()
        mock_unregister.assert_called()

        # new command registered
        mock_register.assert_called_once()
        self.assertEqual(cmd.command_id, "cmd-new-500")

    # -----------------------------------------------------------------------
    # start(reset=True) — done command  ← BUG GUARD
    # -----------------------------------------------------------------------

    @patch("atexit.unregister")
    @patch("atexit.register")
    def test_start_reset_done_command_unregisters_before_reregistering(
        self, mock_register: MagicMock, mock_unregister: MagicMock
    ):
        """
        Bug guard: reset on a DONE command must explicitly call atexit.unregister
        before registering the new command.

        Previously the done branch skipped cancel() entirely (correct), but also
        skipped atexit.unregister — so _unsafe_cancel was registered twice and
        would fire twice at process exit.
        """
        ctx = _make_ctx(context_id="ctx-600", temporary=True)
        _ws_cmd_exec(ctx.cluster).execute.return_value = _execute_response("cmd-new-600")

        cmd = _make_cmd(ctx=ctx, command_id="cmd-done-600", temporary=True)
        cmd._details = _done_response("cmd-done-600")   # force done state

        cmd.start(reset=True)

        # Old entry must be removed BEFORE the new one is added
        mock_unregister.assert_called()
        unregistered_fn = mock_unregister.call_args[0][0]
        self.assertIs(unregistered_fn.__self__, cmd)

        # Exactly ONE new registration
        mock_register.assert_called_once()
        self.assertEqual(cmd.command_id, "cmd-new-600")

    @patch("atexit.unregister")
    @patch("atexit.register")
    def test_start_reset_done_command_registration_count_is_one(
        self, mock_register: MagicMock, mock_unregister: MagicMock
    ):
        """
        After a done-reset the atexit entry count must be exactly 1,
        not 2 (the pre-fix behaviour).
        """
        # Track real atexit state with a counter
        registrations: list = []

        def fake_register(fn, *a, **kw):
            registrations.append(fn)

        def fake_unregister(fn):
            registrations[:] = [f for f in registrations if f != fn]

        ctx = _make_ctx(context_id="ctx-601", temporary=True)
        _ws_cmd_exec(ctx.cluster).execute.return_value = _execute_response("cmd-new-601")

        cmd = _make_cmd(ctx=ctx, command_id="cmd-done-601", temporary=True)
        cmd._details = _done_response("cmd-done-601")

        with patch("atexit.register", side_effect=fake_register), \
             patch("atexit.unregister", side_effect=fake_unregister):
            # Simulate the initial registration that happened when the command first started
            registrations.append(cmd._unsafe_cancel)
            cmd.start(reset=True)

        self.assertEqual(len(registrations), 1,
                         "Expected exactly 1 atexit entry after done-reset, "
                         f"got {len(registrations)}")

    # -----------------------------------------------------------------------
    # cancel()
    # -----------------------------------------------------------------------

    @patch("atexit.unregister")
    def test_cancel_temporary_unregisters_atexit(self, mock_unregister: MagicMock):
        """cancel() on a temporary command must unregister _unsafe_cancel."""
        ctx = _make_ctx(context_id="ctx-700", temporary=True)
        _ws_cmd_exec(ctx.cluster).cancel.return_value = MagicMock()

        cmd = _make_cmd(ctx=ctx, command_id="cmd-700", temporary=True)
        cmd.cancel()

        mock_unregister.assert_called_once()
        fn = mock_unregister.call_args[0][0]
        self.assertIs(fn.__self__, cmd)

    @patch("atexit.unregister")
    def test_cancel_non_temporary_does_not_touch_atexit(self, mock_unregister: MagicMock):
        """cancel() on a non-temporary command must not touch atexit."""
        ctx = _make_ctx(context_id="ctx-701", temporary=False)
        _ws_cmd_exec(ctx.cluster).cancel.return_value = MagicMock()

        cmd = _make_cmd(ctx=ctx, command_id="cmd-701", temporary=False)
        cmd.cancel()

        mock_unregister.assert_not_called()

    @patch("atexit.unregister")
    def test_cancel_clears_command_id(self, _):
        """cancel() must set command_id to None regardless of temporary flag."""
        ctx = _make_ctx(context_id="ctx-702", temporary=True)
        _ws_cmd_exec(ctx.cluster).cancel.return_value = MagicMock()

        cmd = _make_cmd(ctx=ctx, command_id="cmd-702", temporary=True)
        cmd.cancel()

        self.assertIsNone(cmd.command_id)

    @patch("atexit.unregister")
    def test_cancel_with_no_command_id_is_noop(self, mock_unregister: MagicMock):
        """cancel() when command_id is None/'' must be a complete no-op."""
        ctx = _make_ctx(context_id="ctx-703", temporary=True)

        cmd = _make_cmd(ctx=ctx, command_id=None, temporary=True)
        cmd.cancel()  # must not raise

        _ws_cmd_exec(ctx.cluster).cancel.assert_not_called()
        mock_unregister.assert_not_called()

    # -----------------------------------------------------------------------
    # _unsafe_cancel — idempotency
    # -----------------------------------------------------------------------

    @patch("atexit.unregister")
    def test_unsafe_cancel_idempotent(self, _):
        """_unsafe_cancel() called twice must not raise on the second call."""
        ctx = _make_ctx(context_id="ctx-800", temporary=True)
        _ws_cmd_exec(ctx.cluster).cancel.return_value = MagicMock()

        cmd = _make_cmd(ctx=ctx, command_id="cmd-800", temporary=True)
        cmd._unsafe_cancel()
        cmd._unsafe_cancel()   # second call: command_id is already None → no-op

        self.assertIsNone(cmd.command_id)

    # -----------------------------------------------------------------------
    # details property — auto-unregister when done
    # -----------------------------------------------------------------------

    @patch("atexit.unregister")
    def test_details_done_state_unregisters_atexit(self, mock_unregister: MagicMock):
        """
        Accessing .details when the command is FINISHED must call
        atexit.unregister so the handler is not invoked at process exit.
        """
        ctx = _make_ctx(context_id="ctx-900", temporary=True)
        cmd = _make_cmd(ctx=ctx, command_id="cmd-900", temporary=True)

        # Pre-seed _details as FINISHED so the property takes the done-branch
        cmd._details = _done_response("cmd-900")

        _ = cmd.details

        mock_unregister.assert_called()

    @patch("atexit.unregister")
    def test_details_running_state_does_not_unregister_atexit(self, mock_unregister: MagicMock):
        """
        Accessing .details when the command is still RUNNING must NOT
        call atexit.unregister.
        """
        ctx = _make_ctx(context_id="ctx-901", temporary=True)
        ws = _ws_cmd_exec(ctx.cluster)
        ws.command_status.return_value = _running_response("cmd-901")

        cmd = _make_cmd(ctx=ctx, command_id="cmd-901", temporary=True)
        cmd._details = _running_response("cmd-901")

        _ = cmd.details

        mock_unregister.assert_not_called()


if __name__ == "__main__":
    unittest.main()

