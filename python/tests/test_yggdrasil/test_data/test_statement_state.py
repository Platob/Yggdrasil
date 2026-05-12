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

from yggdrasil.data.enums import State
from yggdrasil.data.statement import PreparedStatement, StatementResult


class _StubStatement(PreparedStatement):
    """No extra fields; lets us drive the lifecycle directly from tests."""


class _CountingResult(StatementResult[_StubStatement]):
    """Result that tracks how many times ``_compute_state`` fired.

    Each call to :meth:`_compute_state` simulates a refresh-from-backend.
    Tests assert on ``compute_count`` to verify that the snapshot path
    collapses N property accesses into one call.
    """

    _PREPARED_STATEMENT_CLASS = _StubStatement

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

    def cancel(self) -> "_CountingResult":
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
        assert not r.started
        assert not r.done
        assert not r.failed

    def test_running(self) -> None:
        r = _result(State.RUNNING)
        assert r.started
        assert not r.done
        assert not r.failed

    def test_succeeded(self) -> None:
        r = _result(State.SUCCEEDED)
        assert r.started
        assert r.done
        assert not r.failed

    def test_failed(self) -> None:
        r = _result(State.FAILED)
        assert r.started
        assert r.done
        assert r.failed

    def test_canceled_is_failed(self) -> None:
        # CANCELED counts as failed to match the warehouse path's
        # ``raise_for_status`` semantics — a canceled query is one the
        # caller asked for and didn't get.
        r = _result(State.CANCELED)
        assert r.started
        assert r.done
        assert r.failed


# ---------------------------------------------------------------------------
# state_snapshot()
# ---------------------------------------------------------------------------


class TestStateSnapshot:
    """``state_snapshot()`` pins ``state`` so a block of predicate accesses
    only refreshes once."""

    def test_four_predicate_accesses_outside_snapshot_compute_four_times(self) -> None:
        # Sanity check: without the snapshot, each predicate triggers
        # its own ``_compute_state`` call.
        r = _result(State.RUNNING)
        _ = r.state
        _ = r.started
        _ = r.done
        _ = r.failed
        assert r.compute_count == 4

    def test_inside_snapshot_computes_once(self) -> None:
        r = _result(State.RUNNING)
        with r.state_snapshot():
            _ = r.state
            _ = r.started
            _ = r.done
            _ = r.failed
        assert r.compute_count == 1

    def test_snapshot_yields_the_pinned_state(self) -> None:
        r = _result(State.RUNNING)
        with r.state_snapshot() as snap:
            assert snap is State.RUNNING
            assert r.state is State.RUNNING

    def test_snapshot_clears_on_exit(self) -> None:
        r = _result(State.RUNNING)
        with r.state_snapshot():
            assert r._state_snapshot is State.RUNNING
        assert r._state_snapshot is None

    def test_snapshot_clears_on_exception(self) -> None:
        r = _result(State.RUNNING)
        try:
            with r.state_snapshot():
                raise ValueError("boom")
        except ValueError:
            pass
        assert r._state_snapshot is None

    def test_nested_snapshot_reuses_outer(self) -> None:
        # Inner snapshot must not refresh again — it reuses the outer pin.
        r = _result(State.RUNNING)
        with r.state_snapshot() as outer:
            with r.state_snapshot() as inner:
                assert outer is inner
                _ = r.failed
                _ = r.done
        assert r.compute_count == 1

    def test_snapshot_does_not_freeze_after_exit(self) -> None:
        # After leaving the snapshot, ``state`` reflects fresh values
        # again so callers polling outside a block still see updates.
        r = _result(State.PENDING)
        with r.state_snapshot():
            assert r.state is State.PENDING
        r._state_value = State.SUCCEEDED
        assert r.state is State.SUCCEEDED

    def test_raise_for_status_uses_snapshot(self) -> None:
        # ``raise_for_status`` is the canonical multi-access block —
        # it reads ``failed`` and then ``_auto_promote_transient_retry``
        # reads ``failed`` two more times. With the snapshot wrapping,
        # ``_compute_state`` should fire once.
        r = _result(State.FAILED)
        try:
            r.raise_for_status()
        except RuntimeError:
            pass
        assert r.compute_count == 1
