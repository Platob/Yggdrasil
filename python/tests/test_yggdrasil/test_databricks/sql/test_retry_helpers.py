"""Unit tests for retry helpers in :mod:`yggdrasil.databricks.sql.table`.

These tests exercise :func:`_drain_batch_with_retry` and
:func:`_execute_with_merge_fallback` against fully mocked batches —
they don't reach Databricks.  The goal is to lock in the behaviour
that callers rely on:

- Successful batches are not retried.
- Transient failures (e.g. ``ConcurrentAppendException``) get
  auto-promoted via ``raise_for_status`` and retried.
- The DELETE+INSERT fallback in ``Mode.MERGE`` *also* gets the same
  retry treatment when the primary batch can't recover.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any
from unittest.mock import MagicMock

from yggdrasil.databricks.sql.table import (
    _drain_batch_with_retry,
    _execute_with_merge_fallback,
)


def _result(*, failed: bool, retryable: bool, transient: bool = False) -> MagicMock:
    """Build a mocked :class:`StatementResult` with the right surface area."""
    result = MagicMock()
    result.failed = failed
    result.retryable = retryable

    if failed and transient:
        # raise_for_status auto-promotes then raises — replicate that here:
        # flip ``retryable`` on first call and raise.
        def _raise() -> None:
            result.retryable = True
            raise RuntimeError("ConcurrentAppendException: simulated")

        result.raise_for_status.side_effect = _raise
    elif failed:
        result.raise_for_status.side_effect = RuntimeError("non-transient")
    else:
        result.raise_for_status.return_value = None

    return result


def _batch(results: dict[str, MagicMock]) -> MagicMock:
    batch = MagicMock()
    batch.results = OrderedDict(results)
    batch.failed = any(r.failed for r in results.values())

    def _retry(*, wait: Any, raise_error: bool, **_: Any) -> Any:
        # Simulate a successful retry: every transient-failed item recovers.
        for r in results.values():
            if r.failed and r.retryable:
                r.failed = False
        batch.failed = any(r.failed for r in results.values())
        return batch

    batch.retry.side_effect = _retry
    return batch


class TestDrainBatchWithRetry:
    def test_noop_on_successful_batch(self):
        batch = _batch({"a": _result(failed=False, retryable=False)})

        _drain_batch_with_retry(
            batch, wait=True, target_location="cat.sch.tbl", label="primary",
        )

        batch.retry.assert_not_called()

    def test_retries_when_transient_failure_auto_promotes(self):
        result = _result(failed=True, retryable=False, transient=True)
        batch = _batch({"a": result})

        _drain_batch_with_retry(
            batch, wait=True, target_location="cat.sch.tbl", label="primary",
        )

        # raise_for_status auto-promoted to retryable; batch.retry() ran.
        result.raise_for_status.assert_called_once()
        batch.retry.assert_called_once()
        assert batch.failed is False

    def test_skips_retry_when_failure_not_retryable(self):
        result = _result(failed=True, retryable=False, transient=False)
        batch = _batch({"a": result})

        _drain_batch_with_retry(
            batch, wait=True, target_location="cat.sch.tbl", label="primary",
        )

        # raise_for_status was tried (auto-promote attempt) but didn't flip
        # retryable, so we don't call batch.retry().
        result.raise_for_status.assert_called_once()
        batch.retry.assert_not_called()

    def test_swallows_batch_retry_exception(self):
        result = _result(failed=True, retryable=False, transient=True)
        batch = _batch({"a": result})
        batch.retry.side_effect = RuntimeError("retry pool blew up")

        # Must not raise — the surrounding fallback path needs to keep going.
        _drain_batch_with_retry(
            batch, wait=True, target_location="cat.sch.tbl", label="primary",
        )

        batch.retry.assert_called_once()


class TestExecuteWithMergeFallback:
    def _mock_engine(self, *batches: MagicMock) -> MagicMock:
        engine = MagicMock()
        engine.execute_many.side_effect = list(batches)
        return engine

    def test_returns_primary_when_it_succeeds(self):
        primary = _batch({"a": _result(failed=False, retryable=False)})
        engine = self._mock_engine(primary)

        out = _execute_with_merge_fallback(
            engine,
            primary=["MERGE ..."],
            fallback_factory=lambda: ["DELETE ...", "INSERT ..."],
            wait=True,
            raise_error=True,
            engine_name="api",
            target_location="cat.sch.tbl",
        )

        assert out is primary
        # Fallback never built or submitted.
        assert engine.execute_many.call_count == 1

    def test_runs_fallback_after_primary_exhausts_retry(self):
        # Primary keeps failing even after retry.
        primary_result = _result(failed=True, retryable=True)
        primary = _batch({"merge": primary_result})

        # Override the success-on-retry side effect: simulate retry that
        # doesn't recover.
        def _failing_retry(**_: Any) -> Any:
            return primary

        primary.retry.side_effect = _failing_retry

        # Fallback succeeds.
        fallback = _batch({"delete": _result(failed=False, retryable=False),
                           "insert": _result(failed=False, retryable=False)})

        engine = self._mock_engine(primary, fallback)

        out = _execute_with_merge_fallback(
            engine,
            primary=["MERGE ..."],
            fallback_factory=lambda: ["DELETE ...", "INSERT ..."],
            wait=True,
            raise_error=True,
            engine_name="api",
            target_location="cat.sch.tbl",
        )

        assert out is fallback
        assert engine.execute_many.call_count == 2

    def test_fallback_batch_is_also_retried(self):
        # Primary fails and stays failed (retry is a no-op on it).
        primary_result = _result(failed=True, retryable=True)
        primary = _batch({"merge": primary_result})
        primary.retry.side_effect = lambda **_: primary

        # Fallback transiently fails on first submit, recovers on retry.
        fb_result = _result(failed=True, retryable=False, transient=True)
        fallback = _batch({"insert": fb_result})

        engine = self._mock_engine(primary, fallback)

        _execute_with_merge_fallback(
            engine,
            primary=["MERGE ..."],
            fallback_factory=lambda: ["DELETE ...", "INSERT ..."],
            wait=True,
            raise_error=True,
            engine_name="api",
            target_location="cat.sch.tbl",
        )

        # Both batches walked through the retry helper.
        primary.retry.assert_called()
        fallback.retry.assert_called_once()
