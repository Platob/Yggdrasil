"""Auto-promote + batch-retry behavior for transient statement failures.

Covers the Delta concurrent-append (and friends) pathway exercised by
``test_concurrent_upserts_overlapping_keys_no_duplicates``: a writer's
MERGE / DELETE+INSERT loses a metadata race, the backend returns a
``DELTA_CONCURRENT_APPEND`` error, and the table layer is supposed to
re-run the statement transparently rather than bubble the error up.

The tests use a stub :class:`StatementResult` subclass that mimics the
warehouse contract — ``failed`` until ``start(reset=True)`` flips it —
so the retry machinery can be exercised without standing up a real
Databricks workspace.
"""

from __future__ import annotations

from typing import Any, Optional

from yggdrasil.data.enums import State
from yggdrasil.data.statement import (
    PreparedStatement,
    StatementBatch,
    StatementResult,
)
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg

# A WaitingConfig that completes synchronously — interval=0 short-circuits
# the sleep call.  Two retries gives us "first attempt fails, second
# attempt succeeds, third attempt would raise" coverage.
_FAST_RETRY = WaitingConfig(timeout=10.0, interval=0.0, backoff=1.0, retries=2)


class _StubStatement(PreparedStatement):
    """PreparedStatement with no extra fields; uses the base contract."""


class _StubResult(StatementResult[_StubStatement]):
    """Stub result that fails with a configurable message until told otherwise.

    ``fail_until`` controls the failing-attempt budget.  Each call to
    :meth:`start` (with ``reset=True`` from the retry loop) increments
    ``self.attempts``; while ``attempts <= fail_until`` the result stays
    failed.  Once the budget is exhausted, ``failed`` flips to ``False``
    and the next ``done`` reflects that.
    """

    _PREPARED_STATEMENT_CLASS = _StubStatement

    def __init__(
        self,
        statement: _StubStatement,
        *,
        failure_message: str,
        fail_until: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(statement=statement, **kwargs)
        self.failure_message = failure_message
        self.fail_until = fail_until
        self.attempts = 0
        self._started = False
        self._failed = False
        self._done = False

    # ----- lifecycle ----------------------------------------------------------

    def _compute_state(self) -> State:
        if not self._started:
            return State.PENDING
        if self._failed:
            return State.FAILED
        if self._done:
            return State.SUCCEEDED
        return State.RUNNING

    def refresh_status(self) -> None:
        return None

    def start(
        self,
        reset: bool = False,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        **kwargs: Any,
    ) -> "_StubResult":
        if self._started and not reset:
            if raise_error:
                self.raise_for_status()
            return self
        self.attempts += 1
        self._started = True
        self._done = True
        self._failed = self.attempts <= self.fail_until
        if raise_error:
            self.raise_for_status()
        return self

    def cancel(self) -> "_StubResult":
        return self

    def _raise_for_status(self) -> None:
        if self._failed:
            raise RuntimeError(self.failure_message)

    # ----- Tabular I/O hooks (unused — tests only drive lifecycle) -----------

    def _read_arrow_batches(self, options):  # pragma: no cover - unused
        return iter(())

    def _write_arrow_batches(self, batches, options):  # pragma: no cover - unused
        raise NotImplementedError


class _StubExecutor:
    """Minimal executor stand-in for :class:`StatementBatch`.

    The batch only calls ``executor.execute(stmt, wait=False,
    raise_error=False)`` from :meth:`_submit_one`; everything else
    operates on the result object the executor returns.  Tests
    pre-build the result and hand it back here so we can pin the
    failure scenario per-statement.
    """

    def __init__(self) -> None:
        self.results: dict[str, _StubResult] = {}

    def register(self, key: str, result: _StubResult) -> None:
        self.results[key] = result

    def execute(self, statement, *, wait=False, raise_error=False, **kwargs):
        result = self.results[statement.key]
        return result.start(reset=False, wait=wait, raise_error=raise_error)


def _make_result(
    *,
    failure_message: str,
    fail_until: int,
    retry: Optional[WaitingConfigArg] = _FAST_RETRY,
    key: Optional[str] = None,
) -> _StubResult:
    stmt = _StubStatement(text="MERGE INTO ...", key=key, retry=retry)
    return _StubResult(
        stmt,
        failure_message=failure_message,
        fail_until=fail_until,
    )


# ---------------------------------------------------------------------------
# StatementResult._auto_promote_transient_retry
# ---------------------------------------------------------------------------


class TestAutoPromote:

    def test_promotes_transient_failure_to_retryable(self) -> None:
        result = _make_result(
            failure_message=(
                "[DELTA_CONCURRENT_APPEND.ROW_LEVEL_CHANGES] Transaction conflict"
            ),
            fail_until=1,
            retry=None,  # caller didn't opt into retry
        )
        result.start(reset=False, wait=False, raise_error=False)

        assert result.failed
        assert result.statement.retry is None
        assert result.retryable is False

        promoted = result._auto_promote_transient_retry()

        assert promoted is True
        assert result.statement.retry is not None
        assert result.retryable is True

    def test_non_transient_failure_is_not_promoted(self) -> None:
        result = _make_result(
            failure_message="SCHEMA_MISMATCH: column count mismatch",
            fail_until=1,
            retry=None,
        )
        result.start(reset=False, wait=False, raise_error=False)

        assert result.failed
        assert result._auto_promote_transient_retry() is False
        assert result.statement.retry is None
        assert result.retryable is False

    def test_promote_is_sticky(self) -> None:
        result = _make_result(
            failure_message="DELTA_CONCURRENT_APPEND",
            fail_until=1,
            retry=None,
        )
        result.start(reset=False, wait=False, raise_error=False)

        assert result._auto_promote_transient_retry() is True
        first_cfg = result.statement.retry

        # A second call must not overwrite the existing config or
        # re-log the promotion — sticky semantics.
        assert result._auto_promote_transient_retry() is True
        assert result.statement.retry is first_cfg

    def test_succeeded_result_passes_through(self) -> None:
        result = _make_result(
            failure_message="DELTA_CONCURRENT_APPEND",
            fail_until=0,
            retry=None,
        )
        result.start(reset=False, wait=False, raise_error=False)

        assert result.failed is False
        assert result._auto_promote_transient_retry() is False


# ---------------------------------------------------------------------------
# StatementBatch.retry — auto-promote + retry loop
# ---------------------------------------------------------------------------


class TestBatchRetryAutoPromote:

    def _make_batch(self, *results: _StubResult) -> StatementBatch:
        executor = _StubExecutor()
        batch = StatementBatch(executor=executor)
        for r in results:
            executor.register(r.key, r)
            batch.results[r.key] = r
            r.start(reset=False, wait=False, raise_error=False)
        return batch

    def test_retry_picks_up_unpromoted_transient_failure(self) -> None:
        """A caller running execute_many(raise_error=False) never hits
        raise_for_status, so without batch-side promotion the failure
        would stay non-retryable.  After the fix, batch.retry()
        promotes-and-retries in one call."""
        result = _make_result(
            failure_message="DELTA_CONCURRENT_APPEND: writer race",
            fail_until=1,
            retry=None,
        )
        batch = self._make_batch(result)

        assert result.failed
        assert result.retryable is False  # not yet promoted

        batch.retry(wait=False, raise_error=True)

        assert result.failed is False
        assert result.attempts == 2  # initial + one retry
        assert result.statement.retry is not None  # sticky promotion

    def test_retry_exhausts_budget_and_raises(self) -> None:
        # fail_until exceeds total_try_count so retry can't succeed.  Pass
        # _FAST_RETRY explicitly so auto-promote keeps the caller's policy
        # (otherwise the default WaitingConfig kicks in with 8 retries
        # and the test would wait minutes).
        result = _make_result(
            failure_message="DELTA_CONCURRENT_APPEND",
            fail_until=20,
            retry=_FAST_RETRY,
        )
        batch = self._make_batch(result)

        try:
            batch.retry(wait=False, raise_error=True)
        except RuntimeError as exc:
            # Latest backend failure must surface directly (not wrapped
            # in a generic "Batch item ... failed" RuntimeError).
            assert "DELTA_CONCURRENT_APPEND" in str(exc)
            assert "Batch item" not in str(exc)
        else:
            raise AssertionError("retry() should have raised after exhausting budget")

        # Initial start (1) + retry loop (total_try_count) attempts.
        assert result.attempts == 1 + _FAST_RETRY.total_try_count

    def test_raise_for_status_surfaces_backend_error_directly(self) -> None:
        """``raise_for_status`` must propagate the typed backend error
        rather than wrapping it in ``RuntimeError("Batch item ...
        failed")`` — callers rely on the error message (and the typed
        exception class on real backends, e.g. ``SQLError``) for
        diagnostics."""
        result = _make_result(
            failure_message="DELTA_CONCURRENT_APPEND.ROW_LEVEL_CHANGES detail",
            fail_until=10,  # never succeeds
            retry=None,
        )
        batch = self._make_batch(result)

        try:
            batch.raise_for_status()
        except RuntimeError as exc:
            assert str(exc) == "DELTA_CONCURRENT_APPEND.ROW_LEVEL_CHANGES detail"
        else:
            raise AssertionError("raise_for_status() should have raised")

    def test_raise_for_status_surfaces_latest_failure(self) -> None:
        """When several items failed, the latest one's exception
        propagates; earlier ones are logged but not re-wrapped."""
        first = _make_result(
            failure_message="first failure",
            fail_until=10,
            retry=None,
            key="first",
        )
        last = _make_result(
            failure_message="latest failure (DELTA_CONCURRENT_APPEND)",
            fail_until=10,
            retry=None,
            key="last",
        )
        batch = self._make_batch(first, last)

        try:
            batch.raise_for_status()
        except RuntimeError as exc:
            assert "latest failure" in str(exc)
            assert "first failure" not in str(exc)
        else:
            raise AssertionError("raise_for_status() should have raised")

    def test_retry_skips_non_transient_failures(self) -> None:
        non_transient = _make_result(
            failure_message="SCHEMA_MISMATCH",
            fail_until=10,
            retry=None,
            key="non-transient",
        )
        transient = _make_result(
            failure_message="DELTA_CONCURRENT_APPEND",
            fail_until=1,
            retry=None,
            key="transient",
        )
        batch = self._make_batch(non_transient, transient)

        try:
            batch.retry(wait=False, raise_error=True)
        except Exception:
            pass  # expected — non-transient surfaces

        # Transient was retried once; non-transient was never retried.
        assert transient.attempts == 2
        assert non_transient.attempts == 1
        assert non_transient.retryable is False
