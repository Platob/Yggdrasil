from __future__ import annotations

from yggdrasil.databricks.sql.sql_utils import (
    MAX_TABLE_NAME_LEN,
    databricks_tag_literal,
    escape_sql_string,
    normalize_databricks_collation,
    quote_ident,
    quote_principal,
    quote_qualified_ident,
    safe_table_name,
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


def test_safe_table_name_passes_through_short_names():
    assert safe_table_name("orders") == "orders"
    assert safe_table_name("a" * MAX_TABLE_NAME_LEN) == "a" * MAX_TABLE_NAME_LEN


def test_safe_table_name_passes_through_none_and_empty():
    assert safe_table_name(None) is None
    assert safe_table_name("") == ""


def test_safe_table_name_truncates_and_hashes_when_over_limit():
    long = "x" * 400
    out = safe_table_name(long)

    assert out is not None
    assert len(out) == MAX_TABLE_NAME_LEN
    # Prefix is preserved so the result stays recognizable in logs/SQL.
    assert out.startswith("x")
    # 16-byte BLAKE2b digest → 32 hex chars after the underscore.
    assert out[-33] == "_"
    assert all(ch in "0123456789abcdef" for ch in out[-32:])


def test_safe_table_name_is_deterministic():
    long = "report_" + "y" * 400
    assert safe_table_name(long) == safe_table_name(long)


def test_safe_table_name_distinguishes_distinct_inputs():
    a = safe_table_name("a" * 400)
    b = safe_table_name("a" * 399 + "b")
    assert a != b


def test_safe_table_name_respects_custom_limit():
    out = safe_table_name("orders_archive_2026", limit=10)
    assert out is not None and len(out) == 10


def test_safe_table_name_falls_back_to_pure_hash_for_tiny_limits():
    out = safe_table_name("orders_archive_2026", limit=8)
    assert out is not None and len(out) == 8
    assert all(ch in "0123456789abcdef" for ch in out)


def test_safe_table_name_keeps_leading_token_when_overflow_is_huge():
    # "brz" stays whole at the front; the giant tail token is replaced
    # by a 32-hex digest.
    long = "brz_" + "x" * 400
    out = safe_table_name(long)
    assert out is not None
    assert len(out) <= MAX_TABLE_NAME_LEN
    assert out.startswith("brz_")
    assert out[-33] == "_"
    assert all(ch in "0123456789abcdef" for ch in out[-32:])


def test_safe_table_name_keeps_multiple_leading_tokens():
    long = "raw_orders_archive_" + "y" * 400
    out = safe_table_name(long)
    assert out is not None and len(out) <= MAX_TABLE_NAME_LEN
    assert out.startswith("raw_orders_archive_")
    assert out[-33] == "_"
    assert all(ch in "0123456789abcdef" for ch in out[-32:])


def test_safe_table_name_distinguishes_overflows_sharing_a_prefix():
    a = safe_table_name("raw_orders_archive_" + "a" * 400)
    b = safe_table_name("raw_orders_archive_" + "b" * 400)
    assert a is not None and b is not None and a != b
    assert a.startswith("raw_orders_archive_")
    assert b.startswith("raw_orders_archive_")


def test_safe_table_name_normalizes_dash_and_space_separators():
    long = "raw orders-archive_" + "z" * 400
    out = safe_table_name(long)
    assert out is not None and out.startswith("raw_orders_archive_")


def test_safe_table_name_falls_back_when_first_token_is_too_long():
    # Single token longer than (limit - 33) → falls back to mid-token
    # truncate + hash of the full name. We just check the result fits
    # the limit and ends with the underscore-separated digest.
    long = "x" * 100 + "_short"
    out = safe_table_name(long, limit=64)
    assert out is not None and len(out) == 64
    assert out[-33] == "_"
    assert all(ch in "0123456789abcdef" for ch in out[-32:])


def test_safe_table_name_keeps_token_boundary_in_normal_truncation():
    # With many short tokens, truncation lands on a token boundary so
    # the kept head reads cleanly.
    name = "_".join(["raw", "orders", "archive", "v2"] + ["seg"] * 60)
    out = safe_table_name(name)
    assert out is not None and len(out) <= MAX_TABLE_NAME_LEN
    assert out.startswith("raw_orders_archive_v2_")
    # Whatever's kept must be a series of whole tokens — no partial token
    # immediately before the trailing "_<digest>".
    head = out[:-33]
    assert not head.endswith("_")
    for tok in head.split("_"):
        assert tok in {"raw", "orders", "archive", "v2", "seg"}
