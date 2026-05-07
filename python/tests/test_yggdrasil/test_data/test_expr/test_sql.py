"""Tests for the SQL backend.

Pinned outputs by dialect — quoting, literal escaping, NULL-aware
``IN`` expansion, ``LIKE`` / ``ILIKE`` keyword selection.
:func:`from_sql` round-trips through ``sqlglot`` so the lifter
test is gated on the optional dependency.
"""

from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.io.tabular.execution.expr import col, lit
from yggdrasil.io.tabular.execution.expr.backends.sql import from_sql, to_sql


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


sqlglot = pytest.importorskip("sqlglot")


class TestFromSqlRoundtrip:
    def test_simple_predicate_round_trips(self):
        original = (col("price") >= 100) & (col("side") == "buy")
        sql = original.to_sql()
        back = from_sql(sql)
        # sqlglot reorders commutative AND/OR — emit and compare
        # the rendered SQL of the lifted tree, not the operand
        # tuple.
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
