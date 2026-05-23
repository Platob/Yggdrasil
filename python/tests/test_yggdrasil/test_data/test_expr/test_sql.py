"""Tests for the SQL backend.

Pinned outputs by dialect — quoting, literal escaping, NULL-aware
``IN`` expansion, ``LIKE`` / ``ILIKE`` keyword selection. The
:func:`from_sql` round-trip suite exercises the hand-rolled
tokenizer + recursive-descent parser, so it has no optional
dependency to guard.
"""

from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.execution.expr import col, lit
from yggdrasil.execution.expr.backends.sql import from_sql, to_sql


class TestQuotingByDialect:
    def test_databricks_uses_backticks(self):
        assert (col("price") >= 100).to_sql(dialect="databricks") == (
            "`price` >= 100"
        )

    def test_postgres_uses_double_quotes(self):
        assert (col("price") >= 100).to_sql(dialect="postgres") == (
            '"price" >= 100'
        )

    def test_alias_qualifies_column(self):
        assert (
            col("price", alias="t") >= 100
        ).to_sql(dialect="databricks") == "`t`.`price` >= 100"


class TestLiteralRendering:
    def test_string_escapes_single_quotes(self):
        assert (col("c") == "O'Hara").to_sql() == "`c` = 'O''Hara'"

    def test_datetime_uses_timestamp_keyword(self):
        sql = (col("t") == dt.datetime(2025, 1, 1)).to_sql()
        assert sql == "`t` = TIMESTAMP '2025-01-01 00:00:00.000000'"

    def test_date_uses_date_keyword(self):
        assert (col("d") == dt.date(2025, 1, 1)).to_sql() == (
            "`d` = DATE '2025-01-01'"
        )

    def test_bool_renders_as_true_false(self):
        assert (col("x") == True).to_sql() == "`x` = TRUE"  # noqa: E712
        assert (col("x") == False).to_sql() == "`x` = FALSE"  # noqa: E712

    def test_none_renders_as_null(self):
        # Note: ``col == None`` is *not* the SQL-aware way (it
        # always evaluates UNKNOWN); use ``is_null``. Still, the
        # literal renderer should produce ``NULL`` correctly.
        assert lit(None).to_sql() == "NULL"


class TestNullAwareIn:
    def test_explicit_null_expands_to_or_is_null(self):
        sql = col("c").is_in([1, 2, None]).to_sql()
        assert sql == "`c` IN (1, 2) OR `c` IS NULL"

    def test_not_in_with_null_uses_and_is_not_null(self):
        sql = col("c").not_in([1, None]).to_sql()
        assert sql == "`c` NOT IN (1) AND `c` IS NOT NULL"

    def test_empty_in_renders_as_false_predicate(self):
        # Defensive — most SQL dialects reject literal ``IN ()``;
        # rendering ``FALSE`` keeps the predicate well-formed.
        assert col("c").is_in([]).to_sql() == "FALSE"
        assert col("c").not_in([]).to_sql() == "TRUE"


class TestLikeIlike:
    def test_databricks_renders_ilike_natively(self):
        sql = col("s").like("foo%", case_insensitive=True).to_sql(
            dialect="databricks",
        )
        assert sql == "`s` ILIKE 'foo%'"

    def test_negated_like(self):
        assert col("s").not_like("foo%").to_sql() == "`s` NOT LIKE 'foo%'"


class TestComposition:
    def test_and_or_precedence_minimal_parens(self):
        sql = ((col("a") == 1) & (col("b") == 2) | (col("c") == 3)).to_sql()
        # AND binds tighter than OR — no parens needed around the
        # AND chain when it's an OR child.
        assert sql == "`a` = 1 AND `b` = 2 OR `c` = 3"

    def test_or_inside_and_gets_wrapped(self):
        sql = ((col("a") == 1) | (col("b") == 2)) & (col("c") == 3)
        assert sql.to_sql() == "(`a` = 1 OR `b` = 2) AND `c` = 3"

    def test_not_renders_keyword(self):
        sql = (~(col("a") == 1)).to_sql()
        assert sql == "NOT `a` = 1"


class TestUnknownDialectRaises:
    def test_passing_unknown_dialect_string_raises(self):
        with pytest.raises(ValueError, match="Unknown SQL dialect"):
            (col("x") == 1).to_sql(dialect="oracle")


class TestFromSqlRoundtrip:
    def test_simple_predicate_round_trips(self):
        original = (col("price") >= 100) & (col("side") == "buy")
        sql = original.to_sql()
        back = from_sql(sql)
        # The parser keeps operand order — emit and compare the
        # rendered SQL of the lifted tree, not the operand tuple,
        # so the test stays portable across construction-time rewrites.
        assert "`price` >= 100" in to_sql(back)
        assert "`side` = 'buy'" in to_sql(back)

    def test_in_lifts_to_inlist_node(self):
        sql = col("c").is_in([1, 2, 3]).to_sql()
        back = from_sql(sql)
        # Round-trip should preserve the IN value set.
        assert to_sql(back) == "`c` IN (1, 2, 3)"

    def test_between_round_trips(self):
        sql = col("d").between(1, 10).to_sql()
        back = from_sql(sql)
        assert to_sql(back) == "`d` BETWEEN 1 AND 10"

    def test_between_timestamp_call_lifts_typed_literals(self):
        # ``TIMESTAMP('...')`` parses as ``Cast(string-lit, TIMESTAMP)``
        # in sqlglot. The lifter folds the cast into a typed
        # :class:`Literal` carrying a Python ``datetime`` so the
        # rendered SQL regenerates the ``TIMESTAMP 'iso'`` form.
        back = from_sql(
            "issue_date BETWEEN TIMESTAMP('2024-01-01 00:00:00') "
            "AND TIMESTAMP('2024-02-01 00:00:00')"
        )
        assert to_sql(back) == (
            "`issue_date` BETWEEN "
            "TIMESTAMP '2024-01-01 00:00:00.000000' "
            "AND TIMESTAMP '2024-02-01 00:00:00.000000'"
        )
        fn = back.to_python()
        assert fn({"issue_date": dt.datetime(2024, 1, 15)}) is True
        assert fn({"issue_date": dt.datetime(2023, 12, 31)}) is False

    def test_between_date_call_lifts_typed_literals(self):
        back = from_sql(
            "d BETWEEN DATE('2024-01-01') AND DATE('2024-02-01')"
        )
        assert to_sql(back) == (
            "`d` BETWEEN DATE '2024-01-01' AND DATE '2024-02-01'"
        )
        fn = back.to_python()
        assert fn({"d": dt.date(2024, 1, 15)}) is True
        assert fn({"d": dt.date(2024, 2, 1)}) is True  # inclusive

    def test_compound_in_and_between_timestamp(self):
        back = from_sql(
            "content_id IN (1, 2) "
            "AND issue_date BETWEEN TIMESTAMP('2024-01-01 00:00:00') "
            "AND TIMESTAMP('2024-02-01 00:00:00')"
        )
        rendered = to_sql(back)
        assert "`content_id` IN (1, 2)" in rendered
        assert "`issue_date` BETWEEN TIMESTAMP '2024-01-01" in rendered

    def test_typed_date_literal_round_trips(self):
        # ``DATE '2024-01-01'`` and ``CAST('2024-01-01' AS DATE)``
        # both fold to the same typed Literal.
        back = from_sql("d >= DATE '2024-01-01'")
        assert to_sql(back) == "`d` >= DATE '2024-01-01'"

        back = from_sql("d >= CAST('2024-01-01' AS DATE)")
        assert to_sql(back) == "`d` >= DATE '2024-01-01'"

    def test_not_between_lifts_negated(self):
        back = from_sql("x NOT BETWEEN 1 AND 10")
        assert to_sql(back) == "`x` NOT BETWEEN 1 AND 10"

    def test_not_in_lifts_negated(self):
        back = from_sql("x NOT IN (1, 2, 3)")
        assert to_sql(back) == "`x` NOT IN (1, 2, 3)"

    def test_is_null_and_is_not_null(self):
        assert to_sql(from_sql("c IS NULL")) == "`c` IS NULL"
        assert to_sql(from_sql("c IS NOT NULL")) == "`c` IS NOT NULL"

    def test_quoted_identifier_round_trips(self):
        # Backticks (Databricks / MySQL) and double quotes (ANSI /
        # Postgres / SQLite) both round-trip as identifiers.
        assert to_sql(from_sql("`weird name` = 1")) == "`weird name` = 1"
        assert to_sql(
            from_sql('"col" = 1', dialect="postgres"),
            dialect="postgres",
        ) == '"col" = 1'

    def test_double_quote_is_string_on_databricks(self):
        # Databricks treats ``"…"`` as a string literal, matching
        # the source SQL the call form ``TIMESTAMP("…")`` emits.
        back = from_sql('s = "hello"')
        assert to_sql(back) == "`s` = 'hello'"

    def test_string_literal_doubled_quote_escape(self):
        back = from_sql("c = 'O''Hara'")
        assert to_sql(back) == "`c` = 'O''Hara'"

    def test_negative_numeric_literal_folds(self):
        back = from_sql("x = -5")
        assert to_sql(back) == "`x` = -5"

    def test_arithmetic_lifts_to_arithmetic_node(self):
        back = from_sql("a + 1 > b * 2")
        assert to_sql(back) == "`a` + 1 > `b` * 2"

    def test_line_and_block_comments_are_skipped(self):
        back = from_sql(
            "x = 1 -- trailing comment\n  AND /* inline */ y = 2"
        )
        assert to_sql(back) == "`x` = 1 AND `y` = 2"

    def test_compound_not_between_and_in_chained(self):
        back = from_sql(
            "content_id IN (1, 2) AND issue_date NOT BETWEEN "
            "DATE('2024-01-01') AND DATE('2024-02-01')"
        )
        rendered = to_sql(back)
        assert "`content_id` IN (1, 2)" in rendered
        assert "`issue_date` NOT BETWEEN DATE '2024-01-01'" in rendered

    def test_trailing_garbage_raises_with_position(self):
        with pytest.raises(ValueError, match="trailing token"):
            from_sql("x = 1 wibble")
