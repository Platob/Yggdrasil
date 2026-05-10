"""Tests for :meth:`SQLWarehouse._submit_with_retry` — the
submission-level retry loop that wraps the SDK
``execute_statement`` call to absorb cold/busy-warehouse
``DeadlineExceeded`` (and ``InternalError``) responses.

The retry sequence under test is:

1. First failure  → back off per ``submit_wait`` and retry on the
   same warehouse.
2. Second failure → :meth:`_failover_to_sibling` for a serverless
   sibling, submit there.
3. Third failure  → re-raise as :class:`DeadlineExceeded` with the
   original cause chained on.

These tests stub the SDK, ``time.sleep``, and the sibling
failover so the loop runs deterministically without a live
workspace.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.errors import DeadlineExceeded, InternalError

# Importing the SQL package first sidesteps the circular import
# between ``warehouse`` and ``sql.engine`` (engine pulls SQLWarehouse
# from the warehouse package, warehouse.statement pulls SQLError
# from the sql package).
from yggdrasil.databricks.sql import SQLEngine  # noqa: F401  -- import-order fix
from yggdrasil.databricks.warehouse import SQLWarehouse, WarehousePreparedStatement
from yggdrasil.dataclasses.waiting import WaitingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _warehouse(
    *,
    warehouse_id: str = "wh-1",
    warehouse_name: str = "wh",
) -> SQLWarehouse:
    """Build an :class:`SQLWarehouse` with a mocked service so the
    constructor skips the live ``find_warehouse`` lookup.

    Both ``warehouse_id`` and ``warehouse_name`` are passed so neither
    of the resolution branches in :meth:`SQLWarehouse.__init__` fires.
    """
    service = MagicMock(name="Warehouses")
    return SQLWarehouse(
        service=service,
        warehouse_id=warehouse_id,
        warehouse_name=warehouse_name,
    )


# Tiny submit-wait policy: zero-ish back-off so the patched
# ``time.sleep`` can no-op without changing observable behaviour.
_FAST_WAIT = WaitingConfig(
    timeout=10.0,
    interval=0.001,
    backoff=1.0,
    max_interval=0.001,
)


def _submit_kwargs(wh: SQLWarehouse, sdk: MagicMock) -> dict:
    """Common kwargs for :meth:`SQLWarehouse._submit_with_retry`."""
    return dict(
        sdk_client=sdk,
        statement=WarehousePreparedStatement("SELECT 1"),
        target_wh_id=wh.warehouse_id,
        disposition=MagicMock(name="Disposition"),
        format_=MagicMock(name="Format"),
        submit_wait=_FAST_WAIT,
        deadline=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSubmitWithRetryDeadlineExceeded:
    """Submission-level retry on ``DeadlineExceeded`` / ``InternalError``."""

    def test_succeeds_on_first_attempt_no_retry(self) -> None:
        wh = _warehouse()
        sdk = MagicMock(name="StatementExecutionAPI")
        response = MagicMock(name="StatementResponse")
        sdk.execute_statement.return_value = response

        with patch(
            "yggdrasil.databricks.warehouse.warehouse.time.sleep",
        ) as sleep:
            out = wh._submit_with_retry(**_submit_kwargs(wh, sdk))

        assert out is response
        sdk.execute_statement.assert_called_once()
        sleep.assert_not_called()
        assert sdk.execute_statement.call_args.kwargs["warehouse_id"] == "wh-1"

    def test_retries_once_on_same_warehouse_after_deadline_exceeded(self) -> None:
        """First attempt hits ``DeadlineExceeded``; second attempt on
        the original warehouse succeeds — no failover needed."""
        wh = _warehouse()
        sdk = MagicMock(name="StatementExecutionAPI")
        response = MagicMock(name="StatementResponse")
        sdk.execute_statement.side_effect = [
            DeadlineExceeded("warehouse cold"),
            response,
        ]

        with patch.object(SQLWarehouse, "_failover_to_sibling") as failover, \
             patch("yggdrasil.databricks.warehouse.warehouse.time.sleep"):
            out = wh._submit_with_retry(**_submit_kwargs(wh, sdk))

        assert out is response
        assert sdk.execute_statement.call_count == 2
        # Both submissions land on the original warehouse.
        assert [
            c.kwargs["warehouse_id"]
            for c in sdk.execute_statement.call_args_list
        ] == ["wh-1", "wh-1"]
        failover.assert_not_called()

    def test_fails_over_to_sibling_after_second_deadline_exceeded(self) -> None:
        """Original retry also fails → loop calls
        :meth:`_failover_to_sibling` and submits against the sibling
        warehouse, which succeeds."""
        wh = _warehouse()
        sibling = MagicMock(name="SiblingWarehouse")
        sibling.warehouse_id = "wh-2"
        sdk = MagicMock(name="StatementExecutionAPI")
        response = MagicMock(name="StatementResponse")
        sdk.execute_statement.side_effect = [
            DeadlineExceeded("busy 1"),
            DeadlineExceeded("busy 2"),
            response,
        ]

        with patch.object(
            SQLWarehouse, "_failover_to_sibling", return_value=sibling,
        ) as failover, \
             patch("yggdrasil.databricks.warehouse.warehouse.time.sleep"):
            out = wh._submit_with_retry(**_submit_kwargs(wh, sdk))

        assert out is response
        assert sdk.execute_statement.call_count == 3
        warehouse_ids = [
            c.kwargs["warehouse_id"]
            for c in sdk.execute_statement.call_args_list
        ]
        assert warehouse_ids == ["wh-1", "wh-1", "wh-2"]
        # Sibling lookup is keyed on the *original* busy warehouse.
        failover.assert_called_once()
        assert failover.call_args.kwargs["busy_wh_id"] == "wh-1"

    def test_raises_after_three_deadline_exceeded_with_wrapped_message(self) -> None:
        """Three back-to-back failures exhaust both the same-warehouse
        retry and the sibling failover; the loop re-raises a fresh
        :class:`DeadlineExceeded` chaining the original cause."""
        wh = _warehouse()
        sibling = MagicMock(name="SiblingWarehouse")
        sibling.warehouse_id = "wh-2"
        sdk = MagicMock(name="StatementExecutionAPI")
        last_cause = DeadlineExceeded("sibling also busy")
        sdk.execute_statement.side_effect = [
            DeadlineExceeded("busy 1"),
            DeadlineExceeded("busy 2"),
            last_cause,
        ]

        with patch.object(
            SQLWarehouse, "_failover_to_sibling", return_value=sibling,
        ), patch("yggdrasil.databricks.warehouse.warehouse.time.sleep"):
            with pytest.raises(DeadlineExceeded) as exc_info:
                wh._submit_with_retry(**_submit_kwargs(wh, sdk))

        assert sdk.execute_statement.call_count == 3
        # Wrapped message names both warehouses for diagnostics, and
        # ``__cause__`` points at the final SDK exception.
        msg = str(exc_info.value)
        assert "wh-1" in msg
        assert "wh-2" in msg
        assert exc_info.value.__cause__ is last_cause

    def test_internal_error_treated_as_retriable_like_deadline_exceeded(self) -> None:
        """``InternalError`` shares the retry path with
        ``DeadlineExceeded`` — one back-off then a retry on the
        original warehouse."""
        wh = _warehouse()
        sdk = MagicMock(name="StatementExecutionAPI")
        response = MagicMock(name="StatementResponse")
        sdk.execute_statement.side_effect = [
            InternalError("transient 5xx"),
            response,
        ]

        with patch("yggdrasil.databricks.warehouse.warehouse.time.sleep"):
            out = wh._submit_with_retry(**_submit_kwargs(wh, sdk))

        assert out is response
        assert sdk.execute_statement.call_count == 2
