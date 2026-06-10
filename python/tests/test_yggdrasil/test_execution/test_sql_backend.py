"""SQL backend tests — ``to_sql`` emission per dialect and the
``from_sql`` parser round-trip."""

from __future__ import annotations

import datetime as dt

from yggdrasil.execution.expr import Expression, col, lit
from yggdrasil.execution.expr.nodes import (
    Alias,
    CaseWhen,
    FunctionCall,
    SortOrder,
    Star,
    Subscript,
    WindowFunction,
    WindowSpec,
)


def test_default_dialect_backtick_quotes():
    sql = ((col("price") >= 100) & col("side").is_in(["buy", "sell"])).to_sql()
    assert sql == "`price` >= 100 AND `side` IN ('buy', 'sell')"


def test_postgres_dialect_double_quotes():
    sql = (col("price") >= 100).to_sql("postgres")
    assert sql == '"price" >= 100'


def test_dialect_keyword_alias():
    assert (col("x") == 1).to_sql(dialect="postgres") == '"x" = 1'


def test_string_literals_escape_quotes():
    assert (col("s") == "it's").to_sql() == "`s` = 'it''s'"


def test_temporal_literals_render_typed():
    assert (col("d") == dt.date(2026, 1, 2)).to_sql() == "`d` = DATE '2026-01-02'"
    ts = (col("t") == dt.datetime(2026, 1, 2, 3, 4, 5)).to_sql()
    assert ts.startswith("`t` = TIMESTAMP '2026-01-02")


def test_predicate_shapes_render():
    assert col("x").is_null().to_sql() == "`x` IS NULL"
    assert col("x").is_not_null().to_sql() == "`x` IS NOT NULL"
    assert col("x").not_in([1, 2]).to_sql() == "`x` NOT IN (1, 2)"
    assert col("x").between(1, 5).to_sql() == "`x` BETWEEN 1 AND 5"
    assert col("x").not_between(1, 5).to_sql() == "`x` NOT BETWEEN 1 AND 5"
    assert col("x").like("a%").to_sql() == "`x` LIKE 'a%'"
    assert (col("x") != 1).to_sql() == "`x` != 1"


def test_arithmetic_parenthesises_by_precedence():
    assert ((col("a") + 1) * 2 == 4).to_sql() == "(`a` + 1) * 2 = 4"


def test_qualifier_and_alias_render():
    assert (col("x", qualifier="t") == 1).to_sql() == "`t`.`x` = 1"
    aliased = Alias(col("x"), "y")
    assert "AS" in aliased.to_sql()


def test_function_call_and_star():
    count = FunctionCall("count", (Star(),))
    assert count.to_sql() == "COUNT(*)"
    distinct = FunctionCall("count", (col("x"),), distinct=True)
    assert distinct.to_sql() == "COUNT(DISTINCT `x`)"


def test_window_function_renders_over_clause():
    win = WindowFunction(
        FunctionCall("row_number"),
        WindowSpec(
            partition_by=(col("region"),),
            order_by=(SortOrder(col("ts"), ascending=False),),
        ),
    )
    sql = win.to_sql()
    assert "ROW_NUMBER() OVER (PARTITION BY `region` ORDER BY `ts` DESC" in sql


def test_case_when_renders():
    case = CaseWhen(
        branches=((col("a") == 1, lit("one")),),
        else_expr=lit("other"),
    )
    sql = case.to_sql()
    assert sql.startswith("CASE WHEN")
    assert sql.endswith("END")
    assert "ELSE 'other'" in sql


def test_subscript_renders():
    assert Subscript(col("tags"), lit(0)).to_sql() == "`tags`[0]"


def test_from_sql_round_trip_preserves_structure():
    original = (
        (col("price") >= 100)
        & col("side").is_in(["buy", "sell"])
        & col("note").is_null()
    )
    lifted = Expression.from_sql(original.to_sql())
    assert lifted.equals(original)
    assert lifted.to_sql() == original.to_sql()


def test_from_sql_parses_operator_precedence():
    parsed = Expression.from_sql("a = 1 OR b = 2 AND c = 3")
    # AND binds tighter than OR.
    assert parsed.to_sql() == "`a` = 1 OR `b` = 2 AND `c` = 3"
    fn = parsed.to_python()
    assert fn({"a": 0, "b": 2, "c": 3}) is True
    assert fn({"a": 0, "b": 2, "c": 0}) is False


def test_from_sql_via_generic_lifter():
    parsed = Expression.from_("x > 5")
    assert parsed.equals(col("x") > 5)
