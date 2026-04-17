from __future__ import annotations

from yggdrasil.databricks.sql.sql_utils import (
    databricks_tag_literal,
    escape_sql_string,
    normalize_databricks_collation,
    quote_ident,
    quote_principal,
    quote_qualified_ident,
    sql_literal,
)


def test_escape_sql_string_doubles_quotes():
    assert escape_sql_string("O'Reilly") == "O''Reilly"


def test_quote_qualified_ident_quotes_each_segment():
    assert quote_qualified_ident("main.sales.orders") == "`main`.`sales`.`orders`"


def test_quote_principal_quotes_single_name():
    assert quote_principal("Data Scientists") == "`Data Scientists`"


def test_sql_literal_handles_scalars_and_sql_prefix():
    assert sql_literal("true") == "TRUE"
    assert sql_literal("12.5") == "12.5"
    assert sql_literal("sql:current_timestamp()") == "current_timestamp()"
    assert sql_literal("hello") == "'hello'"


def test_quote_ident_is_reexport_safe():
    assert quote_ident("table`name") == "`table``name`"


def test_normalize_databricks_collation_handles_builtin_families():
    assert normalize_databricks_collation("utf8_binary_rtrim") == "UTF8_BINARY_RTRIM"
    assert normalize_databricks_collation("unicode_ai_ci_rtrim") == "UNICODE_CI_AI_RTRIM"


def test_normalize_databricks_collation_handles_locale_forms():
    assert normalize_databricks_collation("de_CI_AI_rtrim") == "de_CI_AI_RTRIM"


def test_normalize_databricks_collation_rejects_unsupported_suffixes():
    try:
        normalize_databricks_collation("UTF8_BINARY_CI")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid UTF8_BINARY collation to raise")


def test_databricks_tag_literal_adds_default_collation():
    assert databricks_tag_literal("env") == "'env'"


def test_databricks_tag_literal_can_disable_collation():
    assert databricks_tag_literal("env", collation=None) == "'env'"
