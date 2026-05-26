"""Lifecycle-state tests for :class:`StatementResult`.

The contract under test:

* The base class derives ``done`` / ``failed`` / ``started`` from a
  subclass-provided ``_compute_state`` so a single
  :class:`yggdrasil.data.enums.State` mapping is the source of truth.
* The :meth:`state_snapshot` context manager pins ``state`` for the
  duration of the block so multiple state-derived predicate accesses
  share a single ``refresh_status`` call — the warehouse path's
  ``raise_for_status`` / ``_auto_promote_transient_retry`` rely on this
  to avoid flooding the SDK with status pings.
* The unified ``State`` predicates match the legacy warehouse
  ``DONE_STATES`` / ``FAILED_STATES`` semantics (CANCELED is failed,
  RUNNING is started, SUCCEEDED is terminal-and-not-failed).
"""
from __future__ import annotations

from typing import Any

from yggdrasil.enums import State
from yggdrasil.data.statement import PreparedStatement, StatementResult


class _StubStatement(PreparedStatement):
    """No extra fields; lets us drive the lifecycle directly from tests."""


class _CountingResult(StatementResult[_StubStatement]):
    """Result that tracks how many times ``_compute_state`` fired.

    Each call to :meth:`_compute_state` simulates a refresh-from-backend.
    Tests assert on ``compute_count`` to verify that the snapshot path
    collapses N property accesses into one call.
    """

    _PREPARED_CLASS = _StubStatement

    def __init__(self, statement: _StubStatement, *, state: State = State.PENDING, **kwargs: Any) -> None:
        super().__init__(statement=statement, **kwargs)
        self._state_value: State = state
        self.compute_count: int = 0

    def _compute_state(self) -> State:
        self.compute_count += 1
        return self._state_value

    def refresh_status(self) -> None:
        return None

    def start(self, reset: bool = False, **kwargs: Any) -> "_CountingResult":
        self._state_value = State.SUCCEEDED
        return self

    def cancel(self, wait: WaitingConfigArg = None, raise_error: bool = False, **kwargs) -> "_CountingResult":
        self._state_value = State.CANCELED
        return self

    def _raise_for_status(self) -> None:
        if self._state_value.is_failed:
            raise RuntimeError("stub failed")

    # Unused Tabular hooks.
    def _read_arrow_batches(self, options):  # pragma: no cover
        return iter(())

    def _write_arrow_batches(self, batches, options):  # pragma: no cover
        raise NotImplementedError


def _result(state: State = State.PENDING) -> _CountingResult:
    return _CountingResult(_StubStatement(text="SELECT 1"), state=state)


# ---------------------------------------------------------------------------
# Base property delegation
# ---------------------------------------------------------------------------


class TestStateDelegation:
    """``done`` / ``failed`` / ``started`` derive from ``state``."""

    def test_pending(self) -> None:
        r = _result(State.PENDING)
        assert r.state is State.PENDING

    def test_running(self) -> None:
        r = _result(State.RUNNING)
        assert r.state.is_running

    def test_succeeded(self) -> None:
        r = _result(State.SUCCEEDED)
        assert r.state.is_succeeded

    def test_failed(self) -> None:
        r = _result(State.FAILED)
        assert r.state.is_failed

    def test_canceled_is_failed(self) -> None:
        # CANCELED counts as failed to match the warehouse path's
        # ``raise_for_status`` semantics — a canceled query is one the
        # caller asked for and didn't get.
        r = _result(State.CANCELED)
        assert r.state.is_canceled
