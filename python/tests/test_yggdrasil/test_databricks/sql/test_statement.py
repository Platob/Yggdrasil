"""Unit tests for :mod:`yggdrasil.databricks.sql.statement` and :class:`PreparedStatement`."""

from __future__ import annotations

import pytest

from yggdrasil.databricks.sql.statement import PreparedStatement, StatementResult


# ---------------------------------------------------------------------------
# PreparedStatement (config) — prepare / mutators / parameter list / looks_like_query
# ---------------------------------------------------------------------------


class TestStatementPrepare:
    def test_prepare_from_string_wraps_text(self):
        stmt = PreparedStatement.prepare("SELECT 1")
        assert isinstance(stmt, PreparedStatement)
        assert stmt.text == "SELECT 1"
        assert stmt.parameters == {}
        assert stmt.external_tables == {}

    def test_prepare_returns_existing_statement_untouched(self):
        original = PreparedStatement(text="SELECT 1", parameters={"x": 1})
        out = PreparedStatement.prepare(original)
        assert out is original

    def test_prepare_merges_parameters_on_existing_statement(self):
        original = PreparedStatement(text="SELECT 1", parameters={"x": 1})
        out = PreparedStatement.prepare(original, parameters={"y": 2})
        assert out is not original
        assert out.parameters == {"x": 1, "y": 2}
        assert out.text == "SELECT 1"

    def test_prepare_merges_parameters_overriding_existing(self):
        original = PreparedStatement(text="SELECT :x", parameters={"x": 1})
        out = PreparedStatement.prepare(original, parameters={"x": 99})
        assert out.parameters == {"x": 99}

    def test_prepare_merges_external_tables_on_existing_statement(self):
        original = PreparedStatement(text="SELECT * FROM {t}", external_tables={"t": "A"})
        out = PreparedStatement.prepare(original, external_tables={"u": "B"})
        assert out.external_tables == {"t": "A", "u": "B"}

    def test_prepare_builds_fresh_statement_from_string(self):
        out = PreparedStatement.prepare(
            "SELECT :x",
            parameters={"x": 1},
            external_tables={"t": "A"},
        )
        assert out.text == "SELECT :x"
        assert out.parameters == {"x": 1}
        assert out.external_tables == {"t": "A"}


class TestStatementMutators:
    def test_bind_returns_new_instance(self):
        base = PreparedStatement(text="SELECT :x")
        out = base.bind(x=1)
        assert out is not base
        assert base.parameters == {}
        assert out.parameters == {"x": 1}

    def test_bind_with_no_args_returns_same_instance(self):
        base = PreparedStatement(text="SELECT 1")
        assert base.bind() is base

    def test_bind_merges_over_existing(self):
        base = PreparedStatement(text="SELECT :x", parameters={"x": 1, "y": 2})
        out = base.bind(y=99, z=3)
        assert out.parameters == {"x": 1, "y": 99, "z": 3}

    def test_with_external_tables_returns_new_instance(self):
        base = PreparedStatement(text="SELECT * FROM {a}")
        out = base.with_external_tables(a="X")
        assert out is not base
        assert base.external_tables == {}
        assert out.external_tables == {"a": "X"}

    def test_with_external_tables_no_args_returns_same_instance(self):
        base = PreparedStatement(text="SELECT 1")
        assert base.with_external_tables() is base

    def test_clear_resets_all_fields(self):
        base = PreparedStatement(
            text="SELECT :x FROM {a}",
            parameters={"x": 1},
            external_tables={"a": "A"},
        )
        cleared = base.clear()
        assert cleared.text == ""
        assert cleared.parameters == {}
        assert cleared.external_tables == {}
        # Original unchanged
        assert base.text == "SELECT :x FROM {a}"
        assert base.parameters == {"x": 1}
        assert base.external_tables == {"a": "A"}

    def test_with_text_returns_new_instance(self):
        base = PreparedStatement(text="SELECT 1", parameters={"x": 1})
        out = base.with_text("SELECT 2")
        assert out is not base
        assert out.text == "SELECT 2"
        assert out.parameters == {"x": 1}

    def test_with_text_same_text_returns_self(self):
        base = PreparedStatement(text="SELECT 1")
        assert base.with_text("SELECT 1") is base


class TestStatementToParameterList:
    def test_empty_parameters_returns_none(self):
        assert PreparedStatement(text="SELECT 1").to_parameter_list() is None

    def test_parameter_list_renders_named_items(self):
        stmt = PreparedStatement(text="SELECT :x", parameters={"x": 42, "y": "hi"})
        items = stmt.to_parameter_list()
        assert items is not None
        assert [i.name for i in items] == ["x", "y"]
        assert [i.value for i in items] == ["42", "hi"]

    def test_parameter_list_preserves_none_values(self):
        stmt = PreparedStatement(text="SELECT :x", parameters={"x": None})
        items = stmt.to_parameter_list()
        assert items is not None
        assert items[0].value is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("SELECT 1", True),
        ("select * from t", True),
        ("  \n\t  SELECT 1", True),
        ("-- a comment\nSELECT 1", True),
        ("/* block */ WITH cte AS (SELECT 1) SELECT * FROM cte", True),
        ("VALUES (1, 2)", True),
        ("FROM t | SELECT *", True),
        ("TABLE my.tbl", True),
        ("my.table", False),
        ("hello world", False),
        ("INSERT INTO t VALUES (1)", False),
        ("UPDATE t SET x = 1", False),
        ("", False),
        (None, False),
        (123, False),
        ([1, 2, 3], False),
    ],
)
def test_looks_like_query(text, expected):
    assert PreparedStatement.looks_like_query(text) is expected


# ---------------------------------------------------------------------------
# StatementResult (Databricks handler)
# ---------------------------------------------------------------------------


class TestStatementResultPrepare:
    def test_prepare_from_string_wraps_in_result(self):
        result = StatementResult.prepare("SELECT 1")
        assert isinstance(result, StatementResult)
        assert result.statement.text == "SELECT 1"
        assert result.text == "SELECT 1"  # via property

    def test_prepare_from_config_wraps_in_result(self):
        cfg = PreparedStatement(text="SELECT :x", parameters={"x": 1})
        result = StatementResult.prepare(cfg)
        assert result.statement is cfg

    def test_prepare_reuses_existing_result(self):
        result = StatementResult(statement=PreparedStatement(text="SELECT 1"))
        out = StatementResult.prepare(result)
        assert out is result

    def test_prepare_merges_parameters_into_existing_result(self):
        result = StatementResult(statement=PreparedStatement(text="SELECT 1", parameters={"x": 1}))
        out = StatementResult.prepare(result, parameters={"y": 2})
        assert out is not result
        assert out.statement.parameters == {"x": 1, "y": 2}


class TestStatementResultConfigShortcuts:
    def test_text_property_delegates_to_statement(self):
        result = StatementResult(statement=PreparedStatement(text="SELECT 1"))
        assert result.text == "SELECT 1"

    def test_parameters_property_delegates(self):
        result = StatementResult(statement=PreparedStatement(text="X", parameters={"x": 1}))
        assert result.parameters == {"x": 1}

    def test_external_tables_property_delegates(self):
        result = StatementResult(statement=PreparedStatement(text="X", external_tables={"t": "A"}))
        assert result.external_tables == {"t": "A"}

    def test_with_text_returns_new_result(self):
        result = StatementResult(statement=PreparedStatement(text="SELECT 1", parameters={"x": 1}))
        out = result.with_text("SELECT 2")
        assert out is not result
        assert out.statement.text == "SELECT 2"
        assert out.statement.parameters == {"x": 1}

    def test_bind_returns_new_result(self):
        result = StatementResult(statement=PreparedStatement(text="SELECT :x"))
        out = result.bind(x=1)
        assert out is not result
        assert out.statement.parameters == {"x": 1}

    def test_with_external_tables_returns_new_result(self):
        result = StatementResult(statement=PreparedStatement(text="SELECT * FROM {a}"))
        out = result.with_external_tables(a="X")
        assert out is not result
        assert out.statement.external_tables == {"a": "X"}


def test_statement_result_has_default_service():
    from yggdrasil.databricks.sql.statements import Statements

    result = StatementResult(statement=PreparedStatement(text="SELECT 1"))
    assert isinstance(result.service, Statements)
    assert result.client is result.service.client


def test_started_false_without_statement_id():
    result = StatementResult(statement=PreparedStatement(text="SELECT 1"))
    assert result.started is False


def test_started_true_when_statement_id_set():
    result = StatementResult(statement=PreparedStatement(text="SELECT 1"))
    object.__setattr__(result, "statement_id", "abc123")
    assert result.started is True


def test_start_is_idempotent_when_started():
    result = StatementResult(statement=PreparedStatement(text="SELECT 1"))
    object.__setattr__(result, "statement_id", "abc123")
    # Should return self without submitting (warehouse arg is unused).
    assert result.start() is result


def test_cancel_noop_when_not_started():
    result = StatementResult(statement=PreparedStatement(text="SELECT 1"))
    # Not started -> no client calls, returns self.
    assert result.cancel() is result


def test_cancel_noop_for_spark_statements():
    result = StatementResult(statement=PreparedStatement(text="SELECT 1"))
    object.__setattr__(result, "statement_id", "SparkSQL")
    assert result.cancel() is result


def test_cancel_noop_when_already_done():
    from databricks.sdk.service.sql import (
        StatementResponse,
        StatementState,
        StatementStatus,
    )

    result = StatementResult(statement=PreparedStatement(text="SELECT 1"))
    object.__setattr__(result, "statement_id", "abc123")
    object.__setattr__(
        result,
        "_response",
        StatementResponse(
            statement_id="abc123",
            status=StatementStatus(state=StatementState.SUCCEEDED),
        ),
    )
    assert result.cancel() is result


def test_cancel_calls_sdk_when_running():
    from unittest.mock import MagicMock

    from databricks.sdk.service.sql import (
        StatementResponse,
        StatementState,
        StatementStatus,
    )

    result = StatementResult(statement=PreparedStatement(text="SELECT 1"))
    object.__setattr__(result, "statement_id", "abc123")
    object.__setattr__(
        result,
        "_response",
        StatementResponse(
            statement_id="abc123",
            status=StatementStatus(state=StatementState.RUNNING),
        ),
    )

    # Stub the workspace client to record the cancel + get_statement calls.
    ws = MagicMock()
    ws.statement_execution.get_statement.return_value = StatementResponse(
        statement_id="abc123",
        status=StatementStatus(state=StatementState.CANCELED),
    )

    client = MagicMock()
    client.workspace_client.return_value = ws

    service = MagicMock()
    service.client = client
    object.__setattr__(result, "service", service)

    out = result.cancel()

    assert out is result
    ws.statement_execution.cancel_execution.assert_called_once_with(statement_id="abc123")
    assert result._response.status.state == StatementState.CANCELED
