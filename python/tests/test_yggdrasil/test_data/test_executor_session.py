"""Tests covering the :class:`StatementExecutor` ``Session``-shaped surface.

:class:`StatementExecutor` IS a :class:`yggdrasil.io.session.Session`
over a SQL transport: it inherits the singleton-by-config plumbing,
exposes the ``prepare`` / ``send`` vocabulary, and pins type ClassVars
(``_PREPARED_CLASS`` / ``_RESPONSE_CLASS`` / ``_BATCH_CLASS``) lifted
onto the :class:`Session` base. These tests pin the contract: the
class lineage carries Session, the prepare → send pipeline produces
typed results, ``send(lazy=True)`` returns an idled handle, and the
historical ``_STATEMENT_*`` ClassVar names still pin the right types
through ``__init_subclass__``.
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
        self.started = False
        self._terminal = False

    def _compute_state(self):
        from yggdrasil.data.statement import State
        return State.SUCCEEDED if self._terminal else State.IDLE

    def refresh_status(self):
        pass

    def start(self, *, reset=False, wait=None, raise_error=False, **kwargs):
        self.started = True
        self._terminal = True
        return self

    def cancel(self):
        pass

    def _raise_for_status(self):
        pass

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


class _LegacyPinnedExecutor(StatementExecutor):
    """Older subclass shape — pins the historical ``_STATEMENT_*`` names.

    ``__init_subclass__`` should mirror those onto the new short
    ``_PREPARED_CLASS`` / ``_RESPONSE_CLASS`` / ``_BATCH_CLASS`` slots so
    the prepare → send pipeline still produces the right types.
    """

    _PREPARED_STATEMENT_CLASS = PreparedStatement
    _STATEMENT_RESULT_CLASS = _DummyResult
    _STATEMENT_BATCH_CLASS = StatementBatch

    def _submit_statement(self, statement, start=True):
        result = self._RESPONSE_CLASS(statement=statement, executor=self)
        if start:
            result.start()
        return result


class TestExecutorInheritsSession:
    def test_statement_executor_is_session_subclass(self):
        assert issubclass(StatementExecutor, Session)

    def test_session_classvars_lifted_onto_base(self):
        """Session exposes the prepared / response / batch slots so
        subclasses pin once instead of restating them per backend."""
        assert hasattr(Session, "_PREPARED_CLASS")
        assert hasattr(Session, "_RESPONSE_CLASS")
        assert hasattr(Session, "_BATCH_CLASS")

    def test_legacy_long_names_mirror_onto_short_aliases(self):
        """A subclass pinning ``_STATEMENT_*`` propagates to ``_*_CLASS``."""
        assert _LegacyPinnedExecutor._PREPARED_CLASS is PreparedStatement
        assert _LegacyPinnedExecutor._RESPONSE_CLASS is _DummyResult
        assert _LegacyPinnedExecutor._BATCH_CLASS is StatementBatch


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

    def test_send_lazy_returns_idled_result(self):
        """``send(lazy=True)`` returns the response without firing the
        backend — same shape as :meth:`Session.send` with ``lazy=True``."""
        exe = _DummyExecutor()
        result = exe.send("SELECT 1", lazy=True)
        assert isinstance(result, _DummyResult)
        assert result.started is False

    def test_submit_statement_routes_through_send(self):
        """The legacy ``submit_statement`` alias keeps the same lifecycle
        wiring — calling it should produce the same result as ``send``."""
        exe = _DummyExecutor()
        result = exe.submit_statement(PreparedStatement(text="SELECT 1"))
        assert isinstance(result, _DummyResult)
        assert result.executor is exe
        assert result.started

    def test_submit_statement_start_false_matches_send_lazy(self):
        exe = _DummyExecutor()
        result = exe.submit_statement(
            PreparedStatement(text="SELECT 1"),
            start=False,
        )
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
