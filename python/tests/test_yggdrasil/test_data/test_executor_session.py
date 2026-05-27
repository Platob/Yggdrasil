"""Tests covering the :class:`StatementExecutor` ``Session``-shaped surface.

:class:`StatementExecutor` IS a :class:`yggdrasil.io.session.Session`
over a SQL transport: it inherits the singleton-by-config plumbing,
exposes the ``prepare`` / ``send`` vocabulary, and pins type ClassVars
(``_PREPARED_CLASS`` / ``_RESPONSE_CLASS`` / ``_BATCH_CLASS``) lifted
onto the :class:`Session` base. These tests pin the contract: the
class lineage carries Session, the prepare → send pipeline produces
typed results, and ``send(start=False)`` returns an idled handle
matching :meth:`Session.send`'s ``start=False`` shape.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pyarrow as pa

from yggdrasil.data.executor import StatementExecutor
from yggdrasil.data.statement import (
    PreparedStatement,
    StatementBatch,
    StatementResult,
)
from yggdrasil.io.session import Session


class _DummyResult(StatementResult):
    """No-op result — every abstract hook lands in a terminal state."""

    def __init__(self, statement, *, executor=None, **kwargs):
        super().__init__(statement, executor=executor, **kwargs)
        self._terminal = False

    def _compute_state(self):
        from yggdrasil.data.statement import State
        return State.SUCCEEDED if self._terminal else State.IDLE

    def refresh_status(self):
        pass

    def _start(self):
        self._terminal = True

    def _cancel(self):
        pass

    def _error_for_status(self):
        return None

    def _read_arrow_batches(self, options):
        return iter([pa.RecordBatch.from_pylist([])])

    def _write_arrow_batches(self, batches, options):
        for _ in batches:
            pass


class _DummyExecutor(StatementExecutor):
    """Smallest executor that implements the required hooks."""

    _PREPARED_CLASS = PreparedStatement
    _RESPONSE_CLASS = _DummyResult
    _BATCH_CLASS = StatementBatch

    def _submit_statement(self, statement, start=True):
        result = self._RESPONSE_CLASS(statement=statement, executor=self)
        if start:
            result.start()
        return result


class TestExecutorInheritsSession:
    def test_statement_executor_is_singleton_subclass(self):
        from yggdrasil.dataclasses.singleton import Singleton
        assert issubclass(StatementExecutor, Singleton)

    def test_session_classvars_declared_on_base(self):
        """Session declares the prepared / response / batch slots so each
        transport pins them once instead of restating them per backend.
        The abstract Session leaves them unset — every concrete subclass
        (HTTPSession, StatementExecutor) sets them in turn — so we check
        the annotation slot exists rather than a default value."""
        annotations = Session.__annotations__
        assert "_PREPARED_CLASS" in annotations
        assert "_RESPONSE_CLASS" in annotations
        assert "_BATCH_CLASS" in annotations


class TestPrepareSend:
    def test_prepare_coerces_string_to_prepared_class(self):
        exe = _DummyExecutor()
        prepared = exe.prepare("SELECT 1")
        assert isinstance(prepared, exe._PREPARED_CLASS)

    def test_send_returns_response_instance(self):
        exe = _DummyExecutor()
        result = exe.send("SELECT 1")
        assert isinstance(result, _DummyResult)
        assert result.executor is exe
        assert result.started

    def test_send_start_false_returns_idled_result(self):
        """``send(start=False)`` returns the response without firing the
        backend — same shape as :meth:`Session.send` with ``start=False``."""
        exe = _DummyExecutor()
        result = exe.send("SELECT 1", start=False)
        assert isinstance(result, _DummyResult)
        assert result.started is False

    def test_send_binds_executor_when_subclass_forgets(self):
        """Safety net: if ``_submit_statement`` forgets to thread
        ``executor=self`` through the constructor, :meth:`send` sets it
        from one place."""
        sentinel = MagicMock(spec=_DummyResult)
        sentinel.executor = None

        class _Forgetful(_DummyExecutor):
            def _submit_statement(self, statement, start=True):
                return sentinel

        exe = _Forgetful()
        result = exe.send("SELECT 1")
        assert result is sentinel
        assert sentinel.executor is exe
