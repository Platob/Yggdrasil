"""Unit tests for :mod:`yggdrasil.databricks.sql.statements`."""

from __future__ import annotations

import datetime as _dt
from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.sql import (
    StatementResponse,
    StatementState,
    StatementStatus,
)

from yggdrasil.databricks.client import DatabricksClient, DatabricksService
from yggdrasil.databricks.sql.statement import Statement
from yggdrasil.databricks.sql.statements import Statements


# ---------------------------------------------------------------------------
# Service shape
# ---------------------------------------------------------------------------


def test_statements_is_databricks_service():
    assert issubclass(Statements, DatabricksService)


def test_statements_service_name():
    assert Statements.service_name() == "statements"


def test_client_exposes_statements_property():
    client = DatabricksClient(host="https://example.cloud.databricks.com")
    assert isinstance(client.statements, Statements)
    # Cached: repeated access returns the same instance.
    assert client.statements is client.statements


# ---------------------------------------------------------------------------
# statement() factory
# ---------------------------------------------------------------------------


def test_statement_factory_builds_bound_statement():
    client = DatabricksClient(host="https://example.cloud.databricks.com")
    svc = client.statements

    stmt = svc.statement("SELECT :x", parameters={"x": 1})

    assert isinstance(stmt, Statement)
    assert stmt.service is svc
    assert stmt.text == "SELECT :x"
    assert stmt.parameters == {"x": 1}


def test_statement_factory_accepts_existing_statement():
    client = DatabricksClient(host="https://example.cloud.databricks.com")
    svc = client.statements
    original = Statement(text="SELECT 1")

    rebound = svc.statement(original)

    assert rebound.service is svc
    assert rebound.text == "SELECT 1"


# ---------------------------------------------------------------------------
# find_statement
# ---------------------------------------------------------------------------


def test_find_statement_returns_statement_bound_to_service():
    client = DatabricksClient(host="https://example.cloud.databricks.com")
    svc = client.statements

    ws = MagicMock()
    ws.statement_execution.get_statement.return_value = StatementResponse(
        statement_id="abc",
        status=StatementStatus(state=StatementState.SUCCEEDED),
    )
    with patch.object(type(client), "workspace_client", return_value=ws):
        stmt = svc.find_statement("abc")

    assert isinstance(stmt, Statement)
    assert stmt.service is svc
    assert stmt.statement_id == "abc"
    assert stmt._response.status.state == StatementState.SUCCEEDED


def test_find_statement_returns_none_when_missing_and_no_raise():
    client = DatabricksClient(host="https://example.cloud.databricks.com")
    svc = client.statements

    ws = MagicMock()
    ws.statement_execution.get_statement.side_effect = ResourceDoesNotExist("nope")
    with patch.object(type(client), "workspace_client", return_value=ws):
        stmt = svc.find_statement("missing", raise_error=False)

    assert stmt is None


def test_find_statement_raises_by_default():
    client = DatabricksClient(host="https://example.cloud.databricks.com")
    svc = client.statements

    ws = MagicMock()
    ws.statement_execution.get_statement.side_effect = ResourceDoesNotExist("nope")
    with patch.object(type(client), "workspace_client", return_value=ws):
        with pytest.raises(ResourceDoesNotExist):
            svc.find_statement("missing")


# ---------------------------------------------------------------------------
# list_statements query composition
# ---------------------------------------------------------------------------


def _captured_query(svc: Statements, rows=None, **list_kwargs):
    result = MagicMock()
    result.to_pylist.return_value = rows or []

    sql_proxy = MagicMock()
    sql_proxy.execute.return_value = result

    with patch.object(type(svc.client), "sql", new_callable=lambda: property(lambda self: sql_proxy)):
        # Drain the generator so the SQL call is made.
        list(svc.list_statements(**list_kwargs))

    args, kwargs = sql_proxy.execute.call_args
    return args[0]


def test_list_statements_builds_query_against_system_table():
    client = DatabricksClient(host="https://example.cloud.databricks.com")
    svc = client.statements

    query = _captured_query(svc, limit=10)

    assert "FROM system.query.history" in query
    assert "ORDER BY start_time DESC" in query
    assert "LIMIT 10" in query


def test_list_statements_adds_filters_from_arguments():
    client = DatabricksClient(host="https://example.cloud.databricks.com")
    svc = client.statements

    query = _captured_query(
        svc,
        warehouse_id="wh1",
        executed_by="alice@example.com",
        status=StatementState.FAILED,
        start_time_from=_dt.datetime(2026, 4, 1, 0, 0, 0),
        text_contains="orders",
        limit=None,
    )

    assert "warehouse_id = 'wh1'" in query
    assert "executed_by = 'alice@example.com'" in query
    assert "execution_status = 'FAILED'" in query
    assert "start_time >= TIMESTAMP" in query
    assert "lower(statement_text) like '%orders%'" in query
    # No LIMIT clause when limit=None.
    assert "LIMIT" not in query


def test_list_statements_yields_statement_resources():
    client = DatabricksClient(host="https://example.cloud.databricks.com")
    svc = client.statements

    rows = [
        {
            "statement_id": "s1",
            "statement_text": "SELECT 1",
            "warehouse_id": "wh1",
            "execution_status": "FINISHED",
        },
        {
            "statement_id": "s2",
            "statement_text": "SELECT 2",
            "warehouse_id": "wh1",
            "execution_status": "FINISHED",
        },
    ]

    result = MagicMock()
    result.to_pylist.return_value = rows
    sql_proxy = MagicMock()
    sql_proxy.execute.return_value = result

    with patch.object(type(client), "sql", new_callable=lambda: property(lambda self: sql_proxy)):
        statements = list(svc.list_statements())

    assert [s.statement_id for s in statements] == ["s1", "s2"]
    assert all(isinstance(s, Statement) for s in statements)
    assert all(s.service is svc for s in statements)
    assert statements[0]._history == rows[0]


def test_list_statements_inherits_service_defaults():
    client = DatabricksClient(host="https://example.cloud.databricks.com")
    svc = Statements(client=client, warehouse_id="wh-default")

    query = _captured_query(svc, limit=1)
    assert "warehouse_id = 'wh-default'" in query


# ---------------------------------------------------------------------------
# Dict-like access
# ---------------------------------------------------------------------------


def test_getitem_delegates_to_find_statement():
    client = DatabricksClient(host="https://example.cloud.databricks.com")
    svc = client.statements

    ws = MagicMock()
    ws.statement_execution.get_statement.return_value = StatementResponse(
        statement_id="abc",
        status=StatementStatus(state=StatementState.SUCCEEDED),
    )
    with patch.object(type(client), "workspace_client", return_value=ws):
        stmt = svc["abc"]

    assert stmt.statement_id == "abc"


def test_contains_uses_non_raising_find():
    client = DatabricksClient(host="https://example.cloud.databricks.com")
    svc = client.statements

    ws = MagicMock()
    ws.statement_execution.get_statement.side_effect = ResourceDoesNotExist("nope")
    with patch.object(type(client), "workspace_client", return_value=ws):
        assert ("missing" in svc) is False
