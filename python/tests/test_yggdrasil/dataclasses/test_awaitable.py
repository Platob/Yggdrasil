"""Tests for :class:`yggdrasil.dataclasses.awaitable.Awaitable`."""
from __future__ import annotations

import pytest

from yggdrasil.dataclasses.awaitable import Awaitable
from yggdrasil.dataclasses.waiting import WaitingConfig
from yggdrasil.enums.state import State


class _InstantTask(Awaitable):
    def _poll(self):
        self._state = State.SUCCEEDED

    def _start(self):
        self._state = State.RUNNING

    def raise_for_status(self):
        if self.is_failed:
            raise RuntimeError(f"task {self._state.name}")


class _SlowTask(Awaitable):
    def __init__(self, polls_until_done: int = 3):
        self._polls = 0
        self._target = polls_until_done

    def _poll(self):
        self._polls += 1
        if self._polls >= self._target:
            self._state = State.SUCCEEDED

    def _start(self):
        self._state = State.RUNNING
        self._polls = 0

    def raise_for_status(self):
        if self.is_failed:
            raise RuntimeError(f"task {self._state.name}")


class _FailingTask(Awaitable):
    def _poll(self):
        self._state = State.FAILED

    def _start(self):
        self._state = State.RUNNING

    def raise_for_status(self):
        if self.is_failed:
            raise RuntimeError(f"task {self._state.name}")


class _RetryableTask(Awaitable):
    def __init__(self, fail_count: int = 1):
        self._fail_count = fail_count
        self._polls = 0

    @property
    def retryable(self) -> bool:
        return self._attempts <= self._fail_count

    def _poll(self):
        self._polls += 1
        if self._polls >= 2:
            if self._attempts <= self._fail_count:
                self._state = State.FAILED
            else:
                self._state = State.SUCCEEDED

    def _start(self):
        self._polls = 0
        self._state = State.RUNNING

    def raise_for_status(self):
        if self.is_failed:
            raise RuntimeError(f"task {self._state.name}")


class _AlwaysRetryableTask(Awaitable):
    @property
    def retryable(self) -> bool:
        return True

    def _poll(self):
        self._state = State.FAILED

    def _start(self):
        self._state = State.RUNNING

    def raise_for_status(self):
        if self.is_failed:
            raise RuntimeError(f"task {self._state.name}")


class _CancellableTask(Awaitable):
    cancelled: bool = False

    def _poll(self):
        pass

    def _start(self):
        self._state = State.RUNNING

    def _cancel(self):
        self.cancelled = True
        self._state = State.CANCELED

    def raise_for_status(self):
        if self.is_failed:
            raise RuntimeError(f"task {self._state.name}")


class TestAbstract:

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Awaitable()

    def test_abstract_methods(self):
        assert "_poll" in Awaitable.__abstractmethods__
        assert "_start" in Awaitable.__abstractmethods__
        assert "raise_for_status" in Awaitable.__abstractmethods__


class TestInitialState:

    def test_default_state_is_idle(self):
        t = _InstantTask()
        assert t.state is State.IDLE

    def test_is_idle(self):
        assert _InstantTask().is_idle

    def test_not_started(self):
        assert not _InstantTask().started

    def test_not_done(self):
        assert not _InstantTask().is_done

    def test_not_active(self):
        assert not _InstantTask().is_active

    def test_zero_attempts(self):
        assert _InstantTask().attempts == 0


class TestStateSetter:

    def test_set_from_enum(self):
        t = _InstantTask()
        t.state = State.RUNNING
        assert t.is_running

    def test_set_from_string(self):
        t = _InstantTask()
        t.state = "succeeded"
        assert t.is_succeeded

    def test_set_from_int(self):
        t = _InstantTask()
        t.state = State.FAILED.value
        assert t.is_failed


class TestStateAccessors:

    def test_is_running(self):
        t = _InstantTask()
        t._state = State.RUNNING
        assert t.is_running
        assert not t.is_done

    def test_is_succeeded(self):
        t = _InstantTask()
        t._state = State.SUCCEEDED
        assert t.is_succeeded
        assert t.is_done
        assert not t.is_failed

    def test_is_failed(self):
        t = _InstantTask()
        t._state = State.FAILED
        assert t.is_failed
        assert t.is_done

    def test_is_canceled(self):
        t = _InstantTask()
        t._state = State.CANCELED
        assert t.is_canceled
        assert t.is_done
        assert t.is_failed

    def test_is_active(self):
        t = _InstantTask()
        t._state = State.RUNNING
        assert t.is_active

    def test_started(self):
        t = _InstantTask()
        t._state = State.RUNNING
        assert t.started
        t._state = State.SUCCEEDED
        assert t.started

    def test_not_started_when_pending(self):
        t = _InstantTask()
        t._state = State.PENDING
        assert not t.started


class TestRetryableDefault:

    def test_default_retryable_is_false(self):
        assert not _InstantTask().retryable

    def test_default_retryable_is_false_on_failure(self):
        t = _FailingTask()
        t._state = State.FAILED
        assert not t.retryable


class TestStart:

    def test_start_sets_pending_then_delegates(self):
        t = _InstantTask()
        t.start(wait=False)
        assert t.state is State.RUNNING

    def test_start_increments_attempts(self):
        t = _InstantTask()
        t.start(wait=False)
        assert t.attempts == 1

    def test_start_with_wait_blocks_until_done(self):
        t = _InstantTask()
        t.start(wait=True)
        assert t.is_succeeded

    def test_start_idempotent_without_reset(self):
        t = _InstantTask()
        t._state = State.RUNNING
        result = t.start(wait=False)
        assert result is t
        assert t.state is State.RUNNING

    def test_start_reset_restarts(self):
        t = _InstantTask()
        t._state = State.SUCCEEDED
        t.start(reset=True, wait=False)
        assert t.state is State.RUNNING

    def test_start_reset_cancels_active(self):
        t = _CancellableTask()
        t._state = State.RUNNING
        t.start(reset=True, wait=False)
        assert t.cancelled

    def test_start_reset_increments_attempts(self):
        t = _InstantTask()
        t.start(wait=False)
        t.start(reset=True, wait=False)
        assert t.attempts == 2

    def test_start_returns_self(self):
        t = _InstantTask()
        assert t.start(wait=False) is t


class TestWait:

    def test_wait_false_polls_once(self):
        t = _InstantTask()
        t._state = State.RUNNING
        t.wait(wait=False)
        assert t.is_succeeded

    def test_wait_true_blocks_until_done(self):
        t = _SlowTask(polls_until_done=2)
        t._state = State.RUNNING
        t.wait(wait=True)
        assert t.is_succeeded

    def test_wait_returns_self(self):
        t = _InstantTask()
        t._state = State.RUNNING
        assert t.wait(wait=False) is t

    def test_wait_already_done_is_instant(self):
        t = _InstantTask()
        t._state = State.SUCCEEDED
        t.wait()
        assert t.is_succeeded

    def test_wait_timeout_raises(self):
        t = _SlowTask(polls_until_done=999)
        t._state = State.RUNNING
        wc = WaitingConfig(timeout=0.01, interval=0.001, backoff=1.0, max_interval=0.01)
        with pytest.raises(TimeoutError):
            t.wait(wait=wc)

    def test_wait_timeout_no_raise(self):
        t = _SlowTask(polls_until_done=999)
        t._state = State.RUNNING
        wc = WaitingConfig(timeout=0.01, interval=0.001, backoff=1.0, max_interval=0.01)
        result = t.wait(wait=wc, raise_error=False)
        assert result is t
        assert not t.is_done

    def test_wait_raises_on_failure(self):
        t = _FailingTask()
        t._state = State.RUNNING
        with pytest.raises(RuntimeError):
            t.wait(wait=False)

    def test_wait_no_raise_on_failure(self):
        t = _FailingTask()
        t._state = State.RUNNING
        t.wait(wait=False, raise_error=False)
        assert t.is_failed

    def test_wait_accepts_waiting_config(self):
        t = _SlowTask(polls_until_done=2)
        t._state = State.RUNNING
        wc = WaitingConfig(timeout=60, interval=0.001, backoff=1.0, max_interval=1.0)
        t.wait(wait=wc)
        assert t.is_succeeded

    def test_wait_accepts_numeric_seconds(self):
        t = _InstantTask()
        t._state = State.RUNNING
        t.wait(wait=5.0)
        assert t.is_succeeded


class TestRetry:

    def test_retryable_task_retries_on_failure(self):
        t = _RetryableTask(fail_count=1)
        t.start(wait=False)
        wc = WaitingConfig(timeout=5, interval=0.001, backoff=1.0, max_interval=0.01, max_attempts=None)
        t.wait(wait=wc)
        assert t.is_succeeded
        assert t.attempts == 2

    def test_retryable_task_multiple_retries(self):
        t = _RetryableTask(fail_count=3)
        t.start(wait=False)
        wc = WaitingConfig(timeout=5, interval=0.001, backoff=1.0, max_interval=0.01, max_attempts=None)
        t.wait(wait=wc)
        assert t.is_succeeded
        assert t.attempts == 4

    def test_non_retryable_failure_raises_immediately(self):
        t = _FailingTask()
        t._state = State.RUNNING
        with pytest.raises(RuntimeError):
            t.wait(wait=False)

    def test_always_retryable_times_out(self):
        t = _AlwaysRetryableTask()
        t.start(wait=False)
        wc = WaitingConfig(timeout=0.05, interval=0.001, backoff=1.0, max_interval=0.01, max_attempts=None)
        with pytest.raises(TimeoutError):
            t.wait(wait=wc)
        assert t.attempts > 1

    def test_always_retryable_no_raise_returns(self):
        t = _AlwaysRetryableTask()
        t.start(wait=False)
        wc = WaitingConfig(timeout=0.05, interval=0.001, backoff=1.0, max_interval=0.01, max_attempts=None)
        result = t.wait(wait=wc, raise_error=False)
        assert result is t

    def test_max_attempts_caps_retries(self):
        t = _AlwaysRetryableTask()
        t.start(wait=False)
        wc = WaitingConfig(timeout=5, interval=0.001, backoff=1.0, max_interval=0.01, max_attempts=3)
        with pytest.raises(RuntimeError):
            t.wait(wait=wc)
        assert t.attempts == 3

    def test_max_attempts_none_unlimited(self):
        t = _RetryableTask(fail_count=2)
        t.start(wait=False)
        wc = WaitingConfig(timeout=5, interval=0.001, backoff=1.0, max_interval=0.01, max_attempts=None)
        t.wait(wait=wc)
        assert t.is_succeeded
        assert t.attempts == 3

    def test_max_attempts_default_is_four(self):
        wc = WaitingConfig.from_(True)
        assert wc.max_attempts == 4

    def test_retryable_resets_iteration_counter(self):
        t = _RetryableTask(fail_count=1)
        t.start(wait=False)
        wc = WaitingConfig(timeout=5, interval=0.001, backoff=1.0, max_interval=0.01, max_attempts=None)
        t.wait(wait=wc)
        assert t._polls >= 2


class TestRaiseForStatus:

    def test_raise_for_status_is_abstract(self):
        assert "raise_for_status" in Awaitable.__abstractmethods__

    def test_raises_on_failed(self):
        t = _InstantTask()
        t._state = State.FAILED
        with pytest.raises(RuntimeError, match="FAILED"):
            t.raise_for_status()

    def test_raises_on_canceled(self):
        t = _InstantTask()
        t._state = State.CANCELED
        with pytest.raises(RuntimeError, match="CANCELED"):
            t.raise_for_status()

    def test_no_raise_on_succeeded(self):
        t = _InstantTask()
        t._state = State.SUCCEEDED
        t.raise_for_status()

    def test_no_raise_on_running(self):
        t = _InstantTask()
        t._state = State.RUNNING
        t.raise_for_status()

    def test_private_raise_for_status_delegates(self):
        t = _InstantTask()
        t._state = State.FAILED
        with pytest.raises(RuntimeError):
            t._raise_for_status()

    def test_private_raise_for_status_noop_when_not_failed(self):
        t = _InstantTask()
        t._state = State.SUCCEEDED
        t._raise_for_status()


class TestCancel:

    def test_cancel_active_task(self):
        t = _CancellableTask()
        t._state = State.RUNNING
        t.cancel(wait=False)
        assert t.is_canceled
        assert t.cancelled

    def test_cancel_idle_is_noop(self):
        t = _CancellableTask()
        t.cancel(wait=False)
        assert t.is_idle
        assert not t.cancelled

    def test_cancel_done_is_noop(self):
        t = _CancellableTask()
        t._state = State.SUCCEEDED
        t.cancel(wait=False)
        assert t.is_succeeded
        assert not t.cancelled

    def test_cancel_returns_self(self):
        t = _CancellableTask()
        assert t.cancel(wait=False) is t

    def test_default_cancel_sets_canceled(self):
        t = _InstantTask()
        t._state = State.RUNNING
        t.cancel(wait=False)
        assert t.is_canceled


class TestRepr:

    def test_repr_includes_class_and_state(self):
        t = _InstantTask()
        assert "_InstantTask" in repr(t)
        assert "idle" in repr(t)
