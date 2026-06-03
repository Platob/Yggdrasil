"""``SQLWarehouse.send`` logging — no backwards "Started" line on direct success.

A statement that succeeds within ``wait_timeout`` returns already terminal, so
``set_api_response`` logs "finished"; ``send`` must then *not* also log
"Started" (which would be a second, out-of-order line). A still-running
statement keeps the "Started" line.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from databricks.sdk.service.sql import (
    StatementResponse,
    StatementState,
    StatementStatus,
)

# Import the SQL package first to sidestep the warehouse<->sql.engine
# circular import (see test_warehouse_polars_lazy.py).
from yggdrasil.databricks.sql import SQLEngine  # noqa: F401  -- import-order fix
from yggdrasil.databricks.warehouse import SQLWarehouse, WarehousePreparedStatement


def _warehouse(state: StatementState) -> SQLWarehouse:
    service = MagicMock(name="Warehouses")
    # ``SQLWarehouse.client`` resolves to ``service.client``.
    service.client.workspace_client.return_value.statement_execution.execute_statement.return_value = (
        StatementResponse(statement_id="stmt-1", status=StatementStatus(state=state))
    )
    return SQLWarehouse(service=service, warehouse_id="wh-1", warehouse_name="wh")


def _started_logged(log: MagicMock) -> bool:
    return any(call.args and call.args[0] == "Started %r" for call in log.info.call_args_list)


class TestSendStartedLog:
    def test_no_started_log_when_already_done(self) -> None:
        wh = _warehouse(StatementState.SUCCEEDED)
        with patch("yggdrasil.databricks.warehouse.warehouse.LOGGER") as log:
            result = wh.send(WarehousePreparedStatement("SELECT 1"))
        assert result.done
        assert not _started_logged(log)  # bypassed — set_api_response logged "finished"

    def test_started_log_when_still_running(self) -> None:
        wh = _warehouse(StatementState.RUNNING)
        with patch("yggdrasil.databricks.warehouse.warehouse.LOGGER") as log:
            result = wh.send(WarehousePreparedStatement("SELECT 1"))
        assert not result.done
        assert _started_logged(log)
