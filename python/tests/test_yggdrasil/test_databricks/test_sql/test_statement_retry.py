"""Tests for :class:`WarehouseStatementResult` retryable logic and
:class:`WarehouseStatementBatch` alias substitution.

Covers the paths that are most expensive when misconfigured: the
result-level retry predicate (prevents infinite retry storms) and the
batch coercer (prevents stale/missing alias substitution on re-submit).
"""
from __future__ import annotations

import copy
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from databricks.sdk.service.sql import (
    Disposition,
    ResultData,
    ResultManifest,
    ServiceError,
    StatementResponse,
    StatementState,
    StatementStatus,
)

from yggdrasil.databricks.sql import SQLEngine  # noqa: F401 — import-order fix
from yggdrasil.databricks.warehouse import SQLWarehouse, WarehousePreparedStatement
from yggdrasil.databricks.warehouse.statement import (
    WarehouseStatementBatch,
    WarehouseStatementResult,
    _RETRYABLE_ERROR_CODES,
    _RETRYABLE_ELAPSED_LIMIT,
    _RETRYABLE_ITERATION_LIMIT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _warehouse() -> SQLWarehouse:
    service = MagicMock(name="Warehouses")
    return SQLWarehouse(service=service, warehouse_id="wh-1", warehouse_name="wh")


def _result_in_state(
    state: StatementState,
    *,
    error_message: Optional[str] = None,
    iteration: int = 1,
    elapsed: Optional[float] = None,
) -> WarehouseStatementResult:
    status = StatementStatus(
        state=state,
        error=ServiceError(message=error_message) if error_message else None,
    )
    resp = StatementResponse(
        statement_id="stmt-1",
        status=status,
        manifest=ResultManifest(total_row_count=0),
        result=ResultData(external_links=[]),
    )
    stmt = WarehousePreparedStatement("SELECT 1", disposition=Disposition.EXTERNAL_LINKS)
    r = WarehouseStatementResult(
        executor=_warehouse(),
        statement=stmt,
        statement_id="stmt-1",
        _response=resp,
    )
    r.iteration = iteration
    if elapsed is not None:
        r.start_timestamp = r.start_timestamp or 0.0
        # Fake elapsed by patching elapsed_timestamp via direct attr.
        object.__setattr__(r, "start_timestamp", -elapsed)
    return r


# ---------------------------------------------------------------------------
# WarehouseStatementResult.retryable
# ---------------------------------------------------------------------------


class TestRetryable:
    """The ``retryable`` predicate guards the retry loop — it must be
    True for known-transient codes and False for everything else."""

    def test_false_when_succeeded(self) -> None:
        r = _result_in_state(StatementState.SUCCEEDED)
        assert r.retryable is False

    def test_false_when_no_error(self) -> None:
        # FAILED status without a matching retryable error code.
        r = _result_in_state(StatementState.FAILED, error_message="PERMISSION_DENIED")
        assert r.retryable is False

    def test_true_for_retryable_code(self) -> None:
        # Any code in ``_RETRYABLE_ERROR_CODES`` must yield ``retryable=True``.
        code = next(iter(_RETRYABLE_ERROR_CODES))
        r = _result_in_state(StatementState.FAILED, error_message=code)
        assert r.retryable is True

    def test_false_when_iteration_limit_exhausted(self) -> None:
        code = next(iter(_RETRYABLE_ERROR_CODES))
        r = _result_in_state(
            StatementState.FAILED,
            error_message=code,
            iteration=_RETRYABLE_ITERATION_LIMIT,
        )
        assert r.retryable is False

    def test_false_when_elapsed_limit_exceeded(self) -> None:
        code = next(iter(_RETRYABLE_ERROR_CODES))
        r = _result_in_state(StatementState.FAILED, error_message=code)
        # Fake elapsed by setting start_timestamp far in the past.
        import time
        r.start_timestamp = time.time() - (_RETRYABLE_ELAPSED_LIMIT + 1)
        assert r.retryable is False


# ---------------------------------------------------------------------------
# WarehouseStatementResult.cancel
# ---------------------------------------------------------------------------


class TestCancel:
    """``cancel`` must short-circuit on terminal state without hitting the API."""

    def test_noop_when_not_started(self) -> None:
        wh = _warehouse()
        stmt = WarehousePreparedStatement("SELECT 1")
        r = WarehouseStatementResult(executor=wh, statement=stmt)
        r.cancel()  # must not raise

    def test_noop_when_already_terminal(self) -> None:
        r = _result_in_state(StatementState.SUCCEEDED)
        # cancel() checks _response.status.state before hitting the API.
        r.cancel(wait=True)
        # No SDK call should have been made — the cached response is terminal.
        r.executor.client.workspace_client.return_value\
            .statement_execution.cancel_execution.assert_not_called()

    def test_fires_cancel_when_running(self) -> None:
        r = _result_in_state(StatementState.RUNNING)
        r.cancel(wait=False)
        r.executor.client.workspace_client.return_value\
            .statement_execution.cancel_execution.assert_called_once_with(
                statement_id="stmt-1",
            )


# ---------------------------------------------------------------------------
# WarehouseStatementBatch._coerce — alias substitution
# ---------------------------------------------------------------------------


class TestBatchCoerceAliasSubstitution:
    """The batch coercer must rewrite ``{alias}`` placeholders without
    mutating the caller's original statement, and must resolve aliases
    from all three sources in the documented precedence order."""

    def _batch(self) -> WarehouseStatementBatch:
        return WarehouseStatementBatch(executor=_warehouse())

    def test_no_aliases_passes_through_unchanged(self) -> None:
        batch = self._batch()
        stmt = WarehousePreparedStatement("SELECT 1")
        coerced = batch._coerce(stmt)
        assert coerced.text == "SELECT 1"

    def test_batch_wide_alias_substituted(self) -> None:
        from unittest.mock import MagicMock
        from yggdrasil.databricks.fs import VolumePath
        path = MagicMock(spec=VolumePath)
        path.full_path.return_value = "/Volumes/c/s/v/data.parquet"
        batch = WarehouseStatementBatch(
            executor=_warehouse(),
            external_paths={"tbl": path},
        )
        stmt = WarehousePreparedStatement(
            "SELECT * FROM {tbl}",
        )
        coerced = batch._coerce(stmt)
        assert "parquet.`/Volumes/c/s/v/data.parquet`" in coerced.text

    def test_s3_path_alias_substituted(self) -> None:
        # An S3Path external source is read in place just like a VolumePath.
        from unittest.mock import MagicMock
        from yggdrasil.aws.fs.path import S3Path
        path = MagicMock(spec=S3Path)
        path.full_path.return_value = "s3://bucket/data.parquet"
        batch = WarehouseStatementBatch(
            executor=_warehouse(),
            external_paths={"tbl": path},
        )
        stmt = WarehousePreparedStatement("SELECT * FROM {tbl}")
        coerced = batch._coerce(stmt)
        assert "parquet.`s3://bucket/data.parquet`" in coerced.text

    def test_check_external_data_passes_through_s3_path(self) -> None:
        # check_external_data accepts an S3Path verbatim — no staging.
        from unittest.mock import MagicMock
        from yggdrasil.aws.fs.path import S3Path
        path = MagicMock(spec=S3Path)
        out = WarehousePreparedStatement.check_external_data({"src": path})
        assert out == {"src": path}

    def test_per_statement_alias_overrides_batch_alias(self) -> None:
        from unittest.mock import MagicMock
        from yggdrasil.databricks.fs import VolumePath

        batch_path = MagicMock(spec=VolumePath)
        batch_path.full_path.return_value = "/Volumes/c/s/v/batch.parquet"

        stmt_path = MagicMock(spec=VolumePath)
        stmt_path.full_path.return_value = "/Volumes/c/s/v/stmt.parquet"

        batch = WarehouseStatementBatch(
            executor=_warehouse(),
            external_paths={"src": batch_path},
        )
        stmt = WarehousePreparedStatement(
            "SELECT * FROM {src}",
            external_volume_paths={"src": stmt_path},
        )
        coerced = batch._coerce(stmt)
        # Per-statement path wins over batch-wide path.
        assert "/Volumes/c/s/v/stmt.parquet" in coerced.text
        assert "/Volumes/c/s/v/batch.parquet" not in coerced.text

    def test_does_not_mutate_original_statement(self) -> None:
        from unittest.mock import MagicMock
        from yggdrasil.databricks.fs import VolumePath

        path = MagicMock(spec=VolumePath)
        path.full_path.return_value = "/Volumes/c/s/v/x.parquet"
        batch = WarehouseStatementBatch(
            executor=_warehouse(),
            external_paths={"t": path},
        )
        original_text = "SELECT * FROM {t}"
        stmt = WarehousePreparedStatement(original_text)
        batch._coerce(stmt)
        assert stmt.text == original_text

    def test_no_placeholder_match_returns_original_object(self) -> None:
        from unittest.mock import MagicMock
        from yggdrasil.databricks.fs import VolumePath

        path = MagicMock(spec=VolumePath)
        path.full_path.return_value = "/Volumes/c/s/v/x.parquet"
        batch = WarehouseStatementBatch(
            executor=_warehouse(),
            external_paths={"other_alias": path},
        )
        stmt = WarehousePreparedStatement("SELECT 1")
        coerced = batch._coerce(stmt)
        # No substitution → same object returned (no unnecessary copy).
        assert coerced is stmt
