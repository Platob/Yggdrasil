"""Unit tests for :mod:`yggdrasil.databricks.sql.statement`."""

from __future__ import annotations

import pytest

from yggdrasil.databricks.sql.statement import Statement


# ---------------------------------------------------------------------------
# Construction / prepare
# ---------------------------------------------------------------------------


class TestStatementPrepare:
    def test_prepare_from_string_wraps_text(self):
        stmt = Statement.prepare("SELECT 1")
        assert isinstance(stmt, Statement)
        assert stmt.text == "SELECT 1"
        assert stmt.parameters == {}
        assert stmt.temporary_tables == {}

    def test_prepare_returns_existing_statement_untouched(self):
        original = Statement(text="SELECT 1", parameters={"x": 1})
        out = Statement.prepare(original)
        assert out is original

    def test_prepare_merges_parameters_on_existing_statement(self):
        original = Statement(text="SELECT 1", parameters={"x": 1})
        out = Statement.prepare(original, parameters={"y": 2})
        assert out is not original
        assert out.parameters == {"x": 1, "y": 2}
        assert out.text == "SELECT 1"

    def test_prepare_merges_parameters_overriding_existing(self):
        original = Statement(text="SELECT :x", parameters={"x": 1})
        out = Statement.prepare(original, parameters={"x": 99})
        assert out.parameters == {"x": 99}

    def test_prepare_merges_temporary_tables_on_existing_statement(self):
        original = Statement(text="SELECT * FROM {t}", temporary_tables={"t": "A"})
        out = Statement.prepare(original, temporary_tables={"u": "B"})
        assert out.temporary_tables == {"t": "A", "u": "B"}

    def test_prepare_builds_fresh_statement_from_string(self):
        out = Statement.prepare(
            "SELECT :x",
            parameters={"x": 1},
            temporary_tables={"t": "A"},
        )
        assert out.text == "SELECT :x"
        assert out.parameters == {"x": 1}
        assert out.temporary_tables == {"t": "A"}


# ---------------------------------------------------------------------------
# bind / with_temporary_tables / clear
# ---------------------------------------------------------------------------


class TestStatementMutators:
    def test_bind_returns_new_instance(self):
        base = Statement(text="SELECT :x")
        out = base.bind(x=1)
        assert out is not base
        assert base.parameters == {}
        assert out.parameters == {"x": 1}

    def test_bind_with_no_args_returns_same_instance(self):
        base = Statement(text="SELECT 1")
        assert base.bind() is base

    def test_bind_merges_over_existing(self):
        base = Statement(text="SELECT :x", parameters={"x": 1, "y": 2})
        out = base.bind(y=99, z=3)
        assert out.parameters == {"x": 1, "y": 99, "z": 3}

    def test_with_temporary_tables_returns_new_instance(self):
        base = Statement(text="SELECT * FROM {a}")
        out = base.with_temporary_tables(a="X")
        assert out is not base
        assert base.temporary_tables == {}
        assert out.temporary_tables == {"a": "X"}

    def test_with_temporary_tables_no_args_returns_same_instance(self):
        base = Statement(text="SELECT 1")
        assert base.with_temporary_tables() is base

    def test_clear_resets_all_fields(self):
        base = Statement(
            text="SELECT :x FROM {a}",
            parameters={"x": 1},
            temporary_tables={"a": "A"},
        )
        cleared = base.clear()
        assert cleared.text == ""
        assert cleared.parameters == {}
        assert cleared.temporary_tables == {}
        # Original unchanged
        assert base.text == "SELECT :x FROM {a}"
        assert base.parameters == {"x": 1}
        assert base.temporary_tables == {"a": "A"}


# ---------------------------------------------------------------------------
# to_parameter_list
# ---------------------------------------------------------------------------


class TestStatementToParameterList:
    def test_empty_parameters_returns_none(self):
        assert Statement(text="SELECT 1").to_parameter_list() is None

    def test_parameter_list_renders_named_items(self):
        stmt = Statement(text="SELECT :x", parameters={"x": 42, "y": "hi"})
        items = stmt.to_parameter_list()
        assert items is not None
        assert [i.name for i in items] == ["x", "y"]
        assert [i.value for i in items] == ["42", "hi"]

    def test_parameter_list_preserves_none_values(self):
        stmt = Statement(text="SELECT :x", parameters={"x": None})
        items = stmt.to_parameter_list()
        assert items is not None
        assert items[0].value is None


# ---------------------------------------------------------------------------
# looks_like_query
# ---------------------------------------------------------------------------


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
    assert Statement.looks_like_query(text) is expected


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_statement_is_frozen():
    stmt = Statement(text="SELECT 1")
    with pytest.raises(Exception):
        stmt.text = "SELECT 2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StatementResult.statement property
# ---------------------------------------------------------------------------


def test_statement_result_defaults_to_empty_statement():
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.sql.statement_result import StatementResult

    result = StatementResult(client=DatabricksClient.__new__(DatabricksClient))
    assert isinstance(result.statement, Statement)
    assert result.statement.text == ""


def test_statement_result_carries_provided_statement():
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.sql.statement_result import StatementResult

    stmt = Statement(text="SELECT 1", parameters={"x": 1})
    result = StatementResult(
        client=DatabricksClient.__new__(DatabricksClient),
        statement=stmt,
    )
    assert result.statement is stmt
