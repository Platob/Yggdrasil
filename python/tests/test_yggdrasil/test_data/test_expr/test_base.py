"""Behavior of :mod:`yggdrasil.data.expr.base` — the SQL expression builder.

The expression builder is what callers reach for when they need to
emit dialect-correct SQL fragments (a Databricks ``WHERE`` clause, a
Postgres parameterized query) without hand-rolling string concat. This
file pins the user-visible contract:

* **Leaf operators** — `eq` / `ne` / `lt` / `le` / `gt` / `ge` / `like`
  / `ilike` / `between` / `is_null` / `is_not_null`.
* **Datetime literals** — typed `TIMESTAMP '...'` / `DATE '...'` /
  `TIME '...'` per dialect.
* **NULL-aware `IN` / `NOT IN`** — `None` in the list expands into
  `OR IS NULL` (or `AND IS NOT NULL` for `NOT IN`).
* **Composition + precedence** — chained `and_` / `or_` flatten,
  mixed precedence picks up parens, `not_().not_()` collapses.
* **Parameterized rendering** — qmark / numeric / named / pyformat,
  with `start=` offset.
* **Dialect flavors** — Databricks (default), Postgres, MySQL, T-SQL,
  SQLite, Spark.
* **Validation** — empty columns, scalar/null type guards.
* **`between()` collection form** — `between([5, 1, 3])` → min/max,
  with NULL handling and per-call type checks.
* **Array-like sniffing** — duck-typed `to_list()` / `tolist()` plus
  real Arrow / Polars / Pandas / Numpy inputs.
* **Compaction** — large contiguous IN lists collapse to BETWEEN
  ranges (or runs of BETWEENs).
* **Table alias** — `with_table_alias("t")` qualifies bare columns
  across the entire predicate tree.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal

import pytest

from yggdrasil.data.expr.base import NULL, Dialect, Expr, flavor_of


# ---------------------------------------------------------------------------
# Leaf operators (default Databricks flavor)
# ---------------------------------------------------------------------------


class TestLeafOperators:

    def test_eq_int(self) -> None:
        assert str(Expr("price").eq(75)) == "`price` = 75"

    def test_eq_string_doubles_quote(self) -> None:
        assert str(Expr("name").eq("O'Brien")) == "`name` = 'O''Brien'"

    def test_ne(self) -> None:
        assert (
            str(Expr("status").ne("cancelled")) == "`status` != 'cancelled'"
        )

    @pytest.mark.parametrize(
        "method,op",
        [("lt", "<"), ("le", "<="), ("gt", ">"), ("ge", ">=")],
    )
    def test_inequalities(self, method: str, op: str) -> None:
        assert str(getattr(Expr("q"), method)(10)) == f"`q` {op} 10"

    def test_like(self) -> None:
        assert str(Expr("n").like("Shell%")) == "`n` LIKE 'Shell%'"

    def test_ilike_native_databricks(self) -> None:
        assert str(Expr("n").ilike("%X%")) == "`n` ILIKE '%X%'"

    def test_between(self) -> None:
        assert str(Expr("p").between(70, 90)) == "`p` BETWEEN 70 AND 90"

    def test_not_between(self) -> None:
        assert (
            str(Expr("p").not_between(70, 90)) == "`p` NOT BETWEEN 70 AND 90"
        )

    def test_is_null_and_is_not_null(self) -> None:
        assert str(Expr("c").is_null()) == "`c` IS NULL"
        assert str(Expr("c").is_not_null()) == "`c` IS NOT NULL"

    def test_eq_none_routes_to_is_null(self) -> None:
        assert str(Expr("c").eq(None)) == "`c` IS NULL"
        assert str(Expr("c").eq(NULL)) == "`c` IS NULL"

    def test_ne_none_routes_to_is_not_null(self) -> None:
        assert str(Expr("c").ne(None)) == "`c` IS NOT NULL"

    def test_dotted_column(self) -> None:
        assert str(Expr("a.b.c").eq(1)) == "`a`.`b`.`c` = 1"

    def test_decimal_literal(self) -> None:
        assert str(Expr("p").eq(Decimal("1.5"))) == "`p` = 1.5"

    def test_bool_literals_use_keywords(self) -> None:
        assert str(Expr("a").eq(True)) == "`a` = TRUE"
        assert str(Expr("a").eq(False)) == "`a` = FALSE"

    def test_bytes_literal_hex(self) -> None:
        assert str(Expr("c").eq(b"\xde\xad")) == "`c` = X'dead'"

    def test_identifier_doubles_quote_char(self) -> None:
        assert str(Expr("we`ird").eq(1)) == "`we``ird` = 1"


# ---------------------------------------------------------------------------
# Datetime / date / time literals
# ---------------------------------------------------------------------------


class TestDatetimeLiterals:

    def test_databricks_timestamp(self) -> None:
        ts = datetime(2026, 4, 28, 14, 30, 5)
        assert (
            str(Expr("ts").eq(ts)) == "`ts` = TIMESTAMP '2026-04-28 14:30:05'"
        )

    def test_databricks_date(self) -> None:
        assert (
            str(Expr("d").eq(date(2026, 12, 1))) == "`d` = DATE '2026-12-01'"
        )

    def test_databricks_time(self) -> None:
        assert str(Expr("t").eq(time(14, 30))) == "`t` = TIME '14:30:00'"

    def test_postgres_timestamp(self) -> None:
        ts = datetime(2026, 1, 2, 3, 4, 5)
        assert (
            str(Expr("ts", flavor="postgres").eq(ts))
            == '"ts" = TIMESTAMP \'2026-01-02 03:04:05\''
        )

    def test_mysql_timestamp_untyped(self) -> None:
        ts = datetime(2026, 1, 2, 3, 4, 5)
        assert (
            str(Expr("ts", flavor="mysql").eq(ts))
            == "`ts` = '2026-01-02 03:04:05'"
        )

    def test_sqlite_date_untyped(self) -> None:
        assert (
            str(Expr("d", flavor="sqlite").eq(date(2026, 12, 1)))
            == '"d" = \'2026-12-01\''
        )

    def test_databricks_timestamps_in_in_list(self) -> None:
        # Each value gets the typed prefix individually.
        a = datetime(2026, 1, 1)
        b = datetime(2026, 2, 1)
        assert str(Expr("ts").in_([a, b])) == (
            "`ts` IN (TIMESTAMP '2026-01-01 00:00:00', TIMESTAMP '2026-02-01 00:00:00')"
        )

    def test_databricks_dates_in_between(self) -> None:
        assert str(Expr("d").between(date(2026, 1, 1), date(2026, 12, 31))) == (
            "`d` BETWEEN DATE '2026-01-01' AND DATE '2026-12-31'"
        )


# ---------------------------------------------------------------------------
# IN / NOT IN with NULL
# ---------------------------------------------------------------------------


class TestNullAwareIn:

    def test_in_no_null(self) -> None:
        assert str(Expr("c").in_([1, 2, 3])) == "`c` IN (1, 2, 3)"

    def test_in_with_python_none_expands_to_or(self) -> None:
        assert (
            str(Expr("c").in_([1, 2, None])) == "`c` IN (1, 2) OR `c` IS NULL"
        )

    def test_in_with_NULL_sentinel_expands(self) -> None:
        assert (
            str(Expr("c").in_([1, NULL, 2])) == "`c` IN (1, 2) OR `c` IS NULL"
        )

    def test_in_only_null_collapses_to_is_null(self) -> None:
        assert str(Expr("c").in_([None])) == "`c` IS NULL"

    def test_in_tuple_with_none(self) -> None:
        assert str(Expr("c").in_((1, None))) == "`c` IN (1) OR `c` IS NULL"

    def test_null_expansion_under_and_gets_parens(self) -> None:
        p = Expr("c").in_([1, None]).and_(Expr("d").eq(5))
        assert str(p) == "(`c` IN (1) OR `c` IS NULL) AND `d` = 5"

    def test_not_in_no_null(self) -> None:
        assert str(Expr("c").not_in([1, 2])) == "`c` NOT IN (1, 2)"

    def test_not_in_with_none_expands_to_and_is_not_null(self) -> None:
        # Without expansion, NOT IN with NULL silently filters every row.
        assert (
            str(Expr("c").not_in([1, None]))
            == "`c` NOT IN (1) AND `c` IS NOT NULL"
        )

    def test_not_in_only_null_collapses_to_is_not_null(self) -> None:
        assert str(Expr("c").not_in([None])) == "`c` IS NOT NULL"


class TestInParam:

    def test_no_null(self) -> None:
        sql, params = Expr("c").in_([1, 2, 3]).to_param()

        assert sql == "`c` IN (?, ?, ?)"
        assert params == [1, 2, 3]

    def test_null_expansion_skips_param_bind(self) -> None:
        sql, params = Expr("c").in_([1, 2, None]).to_param()

        assert sql == "`c` IN (?, ?) OR `c` IS NULL"
        assert params == [1, 2]

    def test_not_in_with_null_expansion(self) -> None:
        sql, params = Expr("c").not_in([1, None]).to_param()

        assert sql == "`c` NOT IN (?) AND `c` IS NOT NULL"
        assert params == [1]


# ---------------------------------------------------------------------------
# Composition + precedence
# ---------------------------------------------------------------------------


class TestComposition:

    def test_and(self) -> None:
        p = Expr("a").eq(1).and_(Expr("b").eq(2))
        assert str(p) == "`a` = 1 AND `b` = 2"

    def test_or(self) -> None:
        p = Expr("a").eq(1).or_(Expr("b").eq(2))
        assert str(p) == "`a` = 1 OR `b` = 2"

    def test_and_chain_flattens(self) -> None:
        p = Expr("a").eq(1).and_(Expr("b").eq(2)).and_(Expr("c").eq(3))
        assert str(p) == "`a` = 1 AND `b` = 2 AND `c` = 3"

    def test_or_chain_flattens(self) -> None:
        p = Expr("a").eq(1).or_(Expr("b").eq(2)).or_(Expr("c").eq(3))
        assert str(p) == "`a` = 1 OR `b` = 2 OR `c` = 3"

    def test_or_inside_and_gets_parens(self) -> None:
        p = Expr("a").eq(1).and_(Expr("b").eq(2).or_(Expr("c").eq(3)))
        assert str(p) == "`a` = 1 AND (`b` = 2 OR `c` = 3)"

    def test_or_left_of_and_gets_parens(self) -> None:
        p = Expr("a").eq(1).or_(Expr("b").eq(2)).and_(Expr("c").eq(3))
        assert str(p) == "(`a` = 1 OR `b` = 2) AND `c` = 3"

    def test_not_on_leaf(self) -> None:
        assert str(Expr("a").eq(1).not_()) == "NOT (`a` = 1)"

    def test_not_on_compound(self) -> None:
        p = Expr("a").eq(1).and_(Expr("b").eq(2)).not_()
        assert str(p) == "NOT (`a` = 1 AND `b` = 2)"

    def test_double_not_collapses(self) -> None:
        p = Expr("a").eq(1).not_().not_()
        assert str(p) == "`a` = 1"

    def test_op_overloads_match_and_or_not(self) -> None:
        assert str((Expr("a") >= 1) & (Expr("b") < 10)) == (
            "`a` >= 1 AND `b` < 10"
        )
        assert str((Expr("a") == 1) | (Expr("b") == 2)) == (
            "`a` = 1 OR `b` = 2"
        )
        assert str(~((Expr("a") == 1) & (Expr("b") == 2))) == (
            "NOT (`a` = 1 AND `b` = 2)"
        )


# ---------------------------------------------------------------------------
# Parameterized rendering
# ---------------------------------------------------------------------------


class TestParam:

    def test_qmark_default(self) -> None:
        sql, params = Expr("a").eq(1).to_param()

        assert sql == "`a` = ?"
        assert params == [1]

    def test_numeric_style(self) -> None:
        sql, params = (
            Expr("a").eq(1).and_(Expr("b").in_([2, 3])).to_param(style="numeric")
        )

        assert sql == "`a` = $1 AND `b` IN ($2, $3)"
        assert params == [1, 2, 3]

    def test_named_style(self) -> None:
        sql, params = (
            Expr("a").eq(1).and_(Expr("b").in_([2, 3])).to_param(style="named")
        )

        assert sql == "`a` = :p0 AND `b` IN (:p1, :p2)"
        assert params == {"p0": 1, "p1": 2, "p2": 3}

    def test_pyformat_style(self) -> None:
        sql, params = Expr("a").eq(1).to_param(style="pyformat")

        assert sql == "`a` = %(p0)s"
        assert params == {"p0": 1}

    def test_start_offset_numeric(self) -> None:
        sql, params = (
            Expr("a")
            .eq(1)
            .and_(Expr("b").in_([2, 3]))
            .to_param(style="numeric", start=5)
        )

        assert sql == "`a` = $6 AND `b` IN ($7, $8)"
        assert params == [1, 2, 3]

    def test_start_offset_named(self) -> None:
        sql, params = Expr("a").eq(1).to_param(style="named", start=10)

        assert sql == "`a` = :p10"
        assert params == {"p10": 1}

    def test_unary_no_params(self) -> None:
        sql, params = Expr("c").is_null().to_param()

        assert sql == "`c` IS NULL"
        assert params == []

    def test_not_renders_params_only_once(self) -> None:
        # Regression: prior version walked the NOT child twice and double-bound params.
        sql, params = Expr("a").eq(1).and_(Expr("b").eq(2)).not_().to_param()

        assert sql == "NOT (`a` = ? AND `b` = ?)"
        assert params == [1, 2]

    def test_unknown_style_raises(self) -> None:
        with pytest.raises(ValueError):
            Expr("a").eq(1).to_param(style="bogus")

    def test_datetime_passed_through_to_driver(self) -> None:
        ts = datetime(2026, 4, 28, 14, 30)
        sql, params = Expr("ts").eq(ts).to_param()

        assert sql == "`ts` = ?"
        assert params == [ts]


# ---------------------------------------------------------------------------
# Dialect flavors
# ---------------------------------------------------------------------------


class TestFlavors:

    def test_default_is_databricks(self) -> None:
        assert str(Expr("c").eq(1)) == "`c` = 1"

    def test_postgres_uses_double_quotes(self) -> None:
        assert str(Expr("c", flavor="postgres").eq(1)) == '"c" = 1'

    def test_mysql_uses_backticks(self) -> None:
        assert str(Expr("c", flavor="mysql").eq(1)) == "`c` = 1"

    def test_tsql_uses_brackets(self) -> None:
        assert str(Expr("c", flavor="tsql").eq(1)) == "[c] = 1"

    def test_tsql_renders_bool_as_int(self) -> None:
        assert str(Expr("a", flavor="tsql").eq(True)) == "[a] = 1"
        assert str(Expr("a", flavor="tsql").eq(False)) == "[a] = 0"

    def test_tsql_bytes_use_0x_prefix(self) -> None:
        assert (
            str(Expr("c", flavor="tsql").eq(b"\xde\xad")) == "[c] = 0xdead"
        )

    def test_mysql_ilike_falls_back_to_lower(self) -> None:
        assert (
            str(Expr("n", flavor="mysql").ilike("Foo%"))
            == "LOWER(`n`) LIKE LOWER('Foo%')"
        )

    def test_mysql_ilike_param_uses_lower(self) -> None:
        sql, params = Expr("n", flavor="mysql").ilike("Foo%").to_param()

        assert sql == "LOWER(`n`) LIKE LOWER(?)"
        assert params == ["Foo%"]

    def test_databricks_ilike_native_with_params(self) -> None:
        sql, params = Expr("n").ilike("Foo%").to_param()

        assert sql == "`n` ILIKE ?"
        assert params == ["Foo%"]

    def test_dialect_enum_accepted(self) -> None:
        assert str(Expr("c", flavor=Dialect.POSTGRES).eq(1)) == '"c" = 1'

    def test_flavor_object_accepted(self) -> None:
        f = flavor_of("postgres")
        assert str(Expr("c", flavor=f).eq(1)) == '"c" = 1'

    def test_unknown_dialect_raises(self) -> None:
        with pytest.raises(ValueError):
            Expr("c", flavor="oracle").eq(1)

    def test_with_flavor_swaps_rendering(self) -> None:
        p = Expr("price").ge(70).and_(Expr("ts").eq(datetime(2026, 1, 2, 3, 4, 5)))
        assert str(p) == "`price` >= 70 AND `ts` = TIMESTAMP '2026-01-02 03:04:05'"

        assert (
            str(p.with_flavor("postgres"))
            == "\"price\" >= 70 AND \"ts\" = TIMESTAMP '2026-01-02 03:04:05'"
        )

    def test_with_flavor_on_expr(self) -> None:
        e = Expr("price").with_flavor("postgres")

        assert isinstance(e, Expr)
        assert str(e.eq(70)) == '"price" = 70'

    def test_with_flavor_preserves_alias(self) -> None:
        e = Expr("price", alias="t").with_flavor("postgres")
        assert str(e.eq(70)) == '"t"."price" = 70'

    def test_flavor_propagates_through_composition(self) -> None:
        a = Expr("a", flavor="postgres").eq(1)
        b = Expr("b", flavor="postgres").eq(2)

        assert str(a.and_(b)) == '"a" = 1 AND "b" = 2'

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("MSSQL", Dialect.TSQL),
            ("SqlServer", Dialect.TSQL),
            ("ANSI", Dialect.STANDARD),
            ("sparksql", Dialect.SPARK),
        ],
    )
    def test_dialect_aliases(self, alias: str, expected: Dialect) -> None:
        assert Dialect.parse(alias) is expected


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:

    @pytest.mark.parametrize("name", ["", "   "])
    def test_empty_or_whitespace_column_raises(self, name: str) -> None:
        with pytest.raises(ValueError):
            Expr(name)

    def test_eq_non_scalar_raises(self) -> None:
        with pytest.raises(TypeError):
            Expr("c").eq([1, 2])

    def test_lt_null_raises(self) -> None:
        with pytest.raises(ValueError):
            Expr("c").lt(None)

    def test_in_empty_list_raises(self) -> None:
        with pytest.raises(ValueError):
            Expr("c").in_([])

    def test_in_none_arg_raises(self) -> None:
        with pytest.raises(ValueError):
            Expr("c").in_(None)

    def test_like_non_string_raises(self) -> None:
        with pytest.raises(TypeError):
            Expr("c").like(5)

    def test_and_with_non_predicate_raises(self) -> None:
        with pytest.raises(TypeError):
            Expr("a").eq(1).and_("not a predicate")  # type: ignore[arg-type]

    def test_or_with_non_predicate_raises(self) -> None:
        with pytest.raises(TypeError):
            Expr("a").eq(1).or_(42)  # type: ignore[arg-type]

    def test_dotted_empty_segment_raises(self) -> None:
        with pytest.raises(ValueError):
            Expr("a..b").eq(1)

    def test_between_with_null_bound_raises(self) -> None:
        with pytest.raises(ValueError):
            Expr("c").between(None, 5)


# ---------------------------------------------------------------------------
# between() / not_between() — collection form
# ---------------------------------------------------------------------------


class TestBetweenCollectionPair:
    """Pair form (the original API) — preserved for back-compat."""

    def test_between(self) -> None:
        assert str(Expr("p").between(70, 90)) == "`p` BETWEEN 70 AND 90"

    def test_not_between(self) -> None:
        assert (
            str(Expr("p").not_between(70, 90)) == "`p` NOT BETWEEN 70 AND 90"
        )

    @pytest.mark.parametrize(
        "args", [(None, 5), (5, None)]
    )
    def test_pair_form_rejects_null(self, args: tuple) -> None:
        with pytest.raises(ValueError):
            Expr("p").between(*args)


class TestBetweenCollectionMinMax:

    def test_list(self) -> None:
        assert str(Expr("p").between([5, 1, 3, 9, 4])) == "`p` BETWEEN 1 AND 9"

    def test_tuple(self) -> None:
        assert str(Expr("p").between((10, 2, 7))) == "`p` BETWEEN 2 AND 10"

    def test_set(self) -> None:
        assert str(Expr("p").between({3, 1, 5, 2})) == "`p` BETWEEN 1 AND 5"

    def test_generator(self) -> None:
        gen = (i for i in [4, 1, 7, 2])
        assert str(Expr("p").between(gen)) == "`p` BETWEEN 1 AND 7"

    def test_singleton_collection_uses_same_lo_hi(self) -> None:
        assert str(Expr("p").between([3])) == "`p` BETWEEN 3 AND 3"


class TestBetweenCollectionNullHandling:

    def test_list_with_none_expands_to_or_is_null(self) -> None:
        assert (
            str(Expr("p").between([5, None, 1, 3]))
            == "`p` BETWEEN 1 AND 5 OR `p` IS NULL"
        )

    def test_NULL_sentinel_treated_as_none(self) -> None:
        assert (
            str(Expr("p").between([5, NULL, 1, 3]))
            == "`p` BETWEEN 1 AND 5 OR `p` IS NULL"
        )

    def test_no_null_no_expansion(self) -> None:
        assert str(Expr("p").between([5, 1, 3])) == "`p` BETWEEN 1 AND 5"

    def test_only_nones_raises(self) -> None:
        with pytest.raises(ValueError):
            Expr("p").between([None, None, NULL])

    def test_empty_collection_raises(self) -> None:
        with pytest.raises(ValueError):
            Expr("p").between([])

    def test_singleton_with_nones_collapses(self) -> None:
        assert (
            str(Expr("p").between([None, 7, None]))
            == "`p` BETWEEN 7 AND 7 OR `p` IS NULL"
        )

    def test_expansion_under_and_gets_parens(self) -> None:
        p = Expr("p").between([5, None, 1]).and_(Expr("q").eq(0))
        assert str(p) == "(`p` BETWEEN 1 AND 5 OR `p` IS NULL) AND `q` = 0"


class TestBetweenCollectionNotBetween:

    def test_no_null(self) -> None:
        assert (
            str(Expr("p").not_between([5, 1, 3])) == "`p` NOT BETWEEN 1 AND 5"
        )

    def test_with_none_expands_to_and_is_not_null(self) -> None:
        assert (
            str(Expr("p").not_between([5, None, 1, 3]))
            == "`p` NOT BETWEEN 1 AND 5 AND `p` IS NOT NULL"
        )

    def test_only_nones_raises(self) -> None:
        with pytest.raises(ValueError):
            Expr("p").not_between([None, NULL])


class TestBetweenCollectionTypes:

    def test_floats(self) -> None:
        assert (
            str(Expr("p").between([1.5, 0.5, 2.5])) == "`p` BETWEEN 0.5 AND 2.5"
        )

    def test_strings_with_none(self) -> None:
        assert (
            str(Expr("name").between(["m", "a", "z", None]))
            == "`name` BETWEEN 'a' AND 'z' OR `name` IS NULL"
        )

    def test_dates_with_none(self) -> None:
        d1, d2, d3 = date(2026, 6, 1), date(2026, 1, 1), date(2026, 12, 31)

        assert (
            str(Expr("d").between([d1, d2, d3, None]))
            == "`d` BETWEEN DATE '2026-01-01' AND DATE '2026-12-31' OR `d` IS NULL"
        )

    def test_postgres_datetimes_with_none(self) -> None:
        a = datetime(2026, 4, 28, 14, 30)
        b = datetime(2026, 1, 2, 3, 4, 5)

        assert str(Expr("ts", flavor="postgres").between([a, None, b])) == (
            "\"ts\" BETWEEN TIMESTAMP '2026-01-02 03:04:05' "
            "AND TIMESTAMP '2026-04-28 14:30:00' "
            "OR \"ts\" IS NULL"
        )


class TestBetweenCollectionParam:

    def test_no_null(self) -> None:
        sql, params = Expr("p").between([5, 1, 3]).to_param()

        assert sql == "`p` BETWEEN ? AND ?"
        assert params == [1, 5]

    def test_with_null_doesnt_bind_extra_params(self) -> None:
        sql, params = Expr("p").between([5, None, 1, 3]).to_param()

        assert sql == "`p` BETWEEN ? AND ? OR `p` IS NULL"
        assert params == [1, 5]

    def test_numeric_with_null(self) -> None:
        sql, params = (
            Expr("p").between([5, None, 1, 3]).to_param(style="numeric")
        )

        assert sql == "`p` BETWEEN $1 AND $2 OR `p` IS NULL"
        assert params == [1, 5]

    def test_not_between_with_null(self) -> None:
        sql, params = Expr("p").not_between([5, None, 1]).to_param()

        assert sql == "`p` NOT BETWEEN ? AND ? AND `p` IS NOT NULL"
        assert params == [1, 5]


class TestBetweenCollectionErrors:

    def test_scalar_one_arg_raises(self) -> None:
        with pytest.raises(TypeError):
            Expr("p").between(5)

    def test_string_one_arg_raises_not_iterated(self) -> None:
        # Bare string is NOT iterated character-by-character (consistent with in_).
        with pytest.raises(TypeError):
            Expr("name").between("hello")

    def test_bytes_one_arg_raises(self) -> None:
        with pytest.raises(TypeError):
            Expr("c").between(b"abc")

    def test_mixed_types_raises(self) -> None:
        with pytest.raises(TypeError):
            Expr("p").between([1, "two", 3])

    def test_none_one_arg_raises(self) -> None:
        with pytest.raises(TypeError):
            Expr("p").between(None)


# ---------------------------------------------------------------------------
# Array-like sniffing — Arrow / polars / pandas / numpy / duck-typed
# ---------------------------------------------------------------------------


class _FakeToList:
    """Duck-typed Arrow/polars-style array exposing ``to_list()``."""

    def __init__(self, values) -> None:
        self._v = values

    def to_list(self):
        return list(self._v)


class _FakeTolist:
    """Duck-typed pandas/numpy-style array exposing ``tolist()`` (lowercase l)."""

    def __init__(self, values) -> None:
        self._v = values

    def tolist(self):
        return list(self._v)


class _FakeBoth:
    """Has both ``to_list()`` and ``tolist()``; ``to_list()`` should win."""

    def to_list(self):
        return [1, 2, 3]

    def tolist(self):
        # If sniffer falls through to this, [9, 9, 9] would leak through.
        return [9, 9, 9]


class _FakeBadToList:
    """``to_list()`` raises TypeError — sniffer must fall through to ``tolist()``."""

    def to_list(self, *args, **kwargs):
        raise TypeError("requires args")

    def tolist(self):
        return [10, 20]


class TestArrayLikeSniffingDuckTyped:

    def test_in_with_to_list(self) -> None:
        assert (
            str(Expr("c").in_(_FakeToList([1, 2, 3]))) == "`c` IN (1, 2, 3)"
        )

    def test_in_with_tolist(self) -> None:
        assert (
            str(Expr("c").in_(_FakeTolist([1, 2, 3]))) == "`c` IN (1, 2, 3)"
        )

    def test_in_with_to_list_containing_none(self) -> None:
        assert (
            str(Expr("c").in_(_FakeToList([1, None, 2])))
            == "`c` IN (1, 2) OR `c` IS NULL"
        )

    def test_to_list_preferred_over_tolist(self) -> None:
        assert str(Expr("c").in_(_FakeBoth())) == "`c` IN (1, 2, 3)"

    def test_to_list_typeerror_falls_through_to_tolist(self) -> None:
        assert str(Expr("c").in_(_FakeBadToList())) == "`c` IN (10, 20)"

    def test_not_in_with_to_list(self) -> None:
        assert (
            str(Expr("c").not_in(_FakeToList([1, None])))
            == "`c` NOT IN (1) AND `c` IS NOT NULL"
        )

    def test_between_with_to_list(self) -> None:
        assert (
            str(Expr("p").between(_FakeToList([5, 1, 3])))
            == "`p` BETWEEN 1 AND 5"
        )

    def test_between_with_to_list_and_none(self) -> None:
        assert (
            str(Expr("p").between(_FakeToList([5, None, 1, 3])))
            == "`p` BETWEEN 1 AND 5 OR `p` IS NULL"
        )

    def test_between_with_tolist(self) -> None:
        assert (
            str(Expr("p").between(_FakeTolist([10, 2, 7])))
            == "`p` BETWEEN 2 AND 10"
        )

    def test_not_between_with_tolist_and_none(self) -> None:
        assert (
            str(Expr("p").not_between(_FakeTolist([5, None, 1])))
            == "`p` NOT BETWEEN 1 AND 5 AND `p` IS NOT NULL"
        )

    def test_in_param_with_to_list(self) -> None:
        sql, params = Expr("c").in_(_FakeToList([1, None, 2])).to_param()

        assert sql == "`c` IN (?, ?) OR `c` IS NULL"
        assert params == [1, 2]


class TestArrayLikeNanHandling:

    def test_nan_in_collection_treated_as_null(self) -> None:
        assert (
            str(Expr("p").between([5.0, float("nan"), 1.0, 3.0]))
            == "`p` BETWEEN 1.0 AND 5.0 OR `p` IS NULL"
        )

    def test_nan_in_eq_routes_to_is_null(self) -> None:
        assert str(Expr("c").eq(float("nan"))) == "`c` IS NULL"

    def test_nan_in_lt_rejected(self) -> None:
        with pytest.raises(ValueError):
            Expr("c").lt(float("nan"))


class TestArrayLikeNativeContainers:

    def test_native_list_unchanged(self) -> None:
        assert str(Expr("c").in_([1, 2, 3])) == "`c` IN (1, 2, 3)"


class TestArrayLikeRealLibraries:

    def test_arrow_array(self) -> None:
        pa = pytest.importorskip("pyarrow")

        arr = pa.array([1, 2, None, 3])
        assert str(Expr("c").in_(arr)) == "`c` IN (1, 2, 3) OR `c` IS NULL"

    def test_arrow_chunked_array(self) -> None:
        pa = pytest.importorskip("pyarrow")

        arr = pa.chunked_array([[1, 2], [None, 3]])
        assert str(Expr("c").in_(arr)) == "`c` IN (1, 2, 3) OR `c` IS NULL"

    def test_arrow_between(self) -> None:
        pa = pytest.importorskip("pyarrow")

        arr = pa.array([5, None, 1, 3])
        assert (
            str(Expr("p").between(arr))
            == "`p` BETWEEN 1 AND 5 OR `p` IS NULL"
        )

    def test_polars_series(self) -> None:
        pl = pytest.importorskip("polars")

        assert (
            str(Expr("c").in_(pl.Series([1, 2, None, 3])))
            == "`c` IN (1, 2, 3) OR `c` IS NULL"
        )

    def test_polars_between(self) -> None:
        pl = pytest.importorskip("polars")

        assert (
            str(Expr("p").between(pl.Series([5, None, 1, 3])))
            == "`p` BETWEEN 1 AND 5 OR `p` IS NULL"
        )

    def test_pandas_series_upcasts_to_float(self) -> None:
        pd = pytest.importorskip("pandas")

        # pandas upcasts [1, 2, None, 3] to float64 since None forces a numeric
        # dtype that can hold NaN — surviving values come out as floats.
        s = pd.Series([1, 2, None, 3])
        assert (
            str(Expr("c").in_(s))
            == "`c` IN (1.0, 2.0, 3.0) OR `c` IS NULL"
        )

    def test_pandas_object_series_preserves_int(self) -> None:
        pd = pytest.importorskip("pandas")

        s = pd.Series([1, 2, None, 3], dtype=object)
        assert str(Expr("c").in_(s)) == "`c` IN (1, 2, 3) OR `c` IS NULL"

    def test_pandas_index(self) -> None:
        pd = pytest.importorskip("pandas")

        idx = pd.Index([1, 2, 3])
        assert str(Expr("c").in_(idx)) == "`c` IN (1, 2, 3)"

    def test_numpy_array(self) -> None:
        np = pytest.importorskip("numpy")

        assert str(Expr("c").in_(np.array([1, 2, 3]))) == "`c` IN (1, 2, 3)"

    def test_numpy_array_with_nan(self) -> None:
        np = pytest.importorskip("numpy")

        assert (
            str(Expr("c").in_(np.array([1.0, float("nan"), 3.0])))
            == "`c` IN (1.0, 3.0) OR `c` IS NULL"
        )


# ---------------------------------------------------------------------------
# Compaction — large IN lists collapse to BETWEEN runs
# ---------------------------------------------------------------------------


class TestCompactionContiguous:

    def test_one_run_collapses_to_single_between(self) -> None:
        p = Expr("id").in_(list(range(1, 5001)))
        assert str(p) == "`id` BETWEEN 1 AND 5000"

    def test_descending_input_still_compacts(self) -> None:
        p = Expr("id").in_(list(range(5000, 0, -1)))
        assert str(p) == "`id` BETWEEN 1 AND 5000"

    def test_duplicates_dropped_silently(self) -> None:
        vals = list(range(1, 2001)) + list(range(500, 600))
        p = Expr("id").in_(vals)

        assert str(p) == "`id` BETWEEN 1 AND 2000"


class TestCompactionMultipleRuns:

    def test_two_runs_plus_singletons(self) -> None:
        vals = list(range(1, 1500)) + list(range(2000, 2010)) + [3000, 5000]
        p = Expr("id").in_(vals)

        assert str(p) == (
            "`id` BETWEEN 1 AND 1499 "
            "OR `id` BETWEEN 2000 AND 2009 "
            "OR `id` = 3000 "
            "OR `id` = 5000"
        )

    def test_two_runs_param(self) -> None:
        vals = list(range(1, 1500)) + list(range(3000, 3010))
        sql, params = Expr("id").in_(vals).to_param()

        assert sql == "`id` BETWEEN ? AND ? OR `id` BETWEEN ? AND ?"
        assert params == [1, 1499, 3000, 3009]


class TestCompactionNotIn:

    def test_contiguous(self) -> None:
        p = Expr("id").not_in(list(range(1, 5001)))
        assert str(p) == "`id` NOT BETWEEN 1 AND 5000"

    def test_multiple_runs_use_and(self) -> None:
        vals = list(range(1, 1500)) + list(range(3000, 3010))
        p = Expr("id").not_in(vals)

        assert str(p) == (
            "`id` NOT BETWEEN 1 AND 1499 AND `id` NOT BETWEEN 3000 AND 3009"
        )

    def test_singleton_run_uses_ne(self) -> None:
        vals = list(range(1, 1500)) + [9999]
        p = Expr("id").not_in(vals)

        assert str(p) == "`id` NOT BETWEEN 1 AND 1499 AND `id` != 9999"


class TestCompactionNullAware:

    def test_in_with_none(self) -> None:
        vals = list(range(1, 1500)) + [None]
        assert (
            str(Expr("id").in_(vals)) == "`id` BETWEEN 1 AND 1499 OR `id` IS NULL"
        )

    def test_not_in_with_none(self) -> None:
        vals = list(range(1, 1500)) + [None]
        assert (
            str(Expr("id").not_in(vals))
            == "`id` NOT BETWEEN 1 AND 1499 AND `id` IS NOT NULL"
        )


class TestCompactionThreshold:
    """Default threshold is 1000; compaction kicks in for > 1000 values."""

    def test_below_threshold_no_compaction(self) -> None:
        p = Expr("id").in_(list(range(1, 1000)))

        assert str(p).startswith("`id` IN (")
        assert " 999)" in str(p)

    def test_at_threshold_no_compaction(self) -> None:
        p = Expr("id").in_(list(range(1, 1001)))

        assert str(p).startswith("`id` IN (")

    def test_just_above_threshold_compacts(self) -> None:
        p = Expr("id").in_(list(range(1, 1002)))
        assert str(p) == "`id` BETWEEN 1 AND 1001"

    def test_per_call_lower_threshold(self) -> None:
        p = Expr("id").in_([1, 2, 3, 4, 5], compact_threshold=3)
        assert str(p) == "`id` BETWEEN 1 AND 5"

    def test_per_call_zero_disables(self) -> None:
        p = Expr("id").in_(list(range(1, 5001)), compact_threshold=0)
        sql, params = p.to_param()

        assert str(p).startswith("`id` IN (")
        assert len(params) == 5000

    def test_too_sparse_falls_back_to_in(self) -> None:
        # 1500 even numbers means 1500 runs — exceeds threshold, so falls back.
        vals = list(range(0, 3000, 2))
        p = Expr("id").in_(vals)

        assert str(p).startswith("`id` IN (")


class TestCompactionTypeRestrictions:
    """Only int and date compact; everything else stays as IN."""

    def test_floats_no_compaction(self) -> None:
        p = Expr("p").in_([float(i) for i in range(1100)], compact_threshold=500)
        assert str(p).startswith("`p` IN (")

    def test_strings_no_compaction(self) -> None:
        p = Expr("s").in_(
            [f"item_{i}" for i in range(1100)], compact_threshold=500
        )
        assert str(p).startswith("`s` IN (")

    def test_decimals_no_compaction(self) -> None:
        p = Expr("p").in_([Decimal(i) for i in range(1100)], compact_threshold=500)
        assert str(p).startswith("`p` IN (")

    def test_datetimes_no_compaction(self) -> None:
        # datetime "next" is ambiguous (microsecond? second?) — no compaction.
        base = datetime(2026, 1, 1)
        vals = [datetime.fromtimestamp(base.timestamp() + i) for i in range(1100)]
        p = Expr("ts").in_(vals, compact_threshold=500)

        assert "IN (" in str(p)

    def test_bools_no_compaction(self) -> None:
        # Bool subclasses int — guard against accidental compaction of T/F.
        vals = [True] * 1100 + [False] * 100
        p = Expr("b").in_(vals, compact_threshold=500)

        assert str(p).startswith("`b` IN (")

    def test_mixed_int_and_date_no_compaction(self) -> None:
        vals = list(range(500)) + [date(2026, i + 1, 1) for i in range(12)] * 50
        p = Expr("c").in_(vals, compact_threshold=100)

        assert str(p).startswith("`c` IN (")


class TestCompactionDateRange:

    def test_contiguous_dates(self) -> None:
        start = date(2026, 1, 1)
        dates = [start + timedelta(days=i) for i in range(1100)]
        end = start + timedelta(days=1099)

        assert str(Expr("d").in_(dates)) == (
            f"`d` BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'"
        )

    def test_two_date_runs(self) -> None:
        start = date(2026, 1, 1)
        run1 = [start + timedelta(days=i) for i in range(1100)]
        run2 = [start + timedelta(days=i) for i in range(2000, 2010)]

        assert str(Expr("d").in_(run1 + run2)) == (
            f"`d` BETWEEN DATE '{run1[0].isoformat()}' AND DATE '{run1[-1].isoformat()}' "
            f"OR `d` BETWEEN DATE '{run2[0].isoformat()}' AND DATE '{run2[-1].isoformat()}'"
        )


class TestCompactionComposition:

    def test_compacted_under_and(self) -> None:
        vals = list(range(1, 1500))
        p = Expr("id").in_(vals).and_(Expr("active").eq(True))

        assert str(p) == "`id` BETWEEN 1 AND 1499 AND `active` = TRUE"

    def test_multi_run_under_and_gets_parens(self) -> None:
        vals = list(range(1, 1500)) + list(range(3000, 3010))
        p = Expr("id").in_(vals).and_(Expr("active").eq(True))

        assert str(p) == (
            "(`id` BETWEEN 1 AND 1499 OR `id` BETWEEN 3000 AND 3009) "
            "AND `active` = TRUE"
        )


class TestCompactionRealLibraries:

    def test_arrow_array_compacts(self) -> None:
        pa = pytest.importorskip("pyarrow")

        arr = pa.array(list(range(1, 5001)))
        assert str(Expr("id").in_(arr)) == "`id` BETWEEN 1 AND 5000"

    def test_polars_series_compacts(self) -> None:
        pl = pytest.importorskip("polars")

        s = pl.Series(list(range(1, 5001)))
        assert str(Expr("id").in_(s)) == "`id` BETWEEN 1 AND 5000"


# ---------------------------------------------------------------------------
# Table alias prefix
# ---------------------------------------------------------------------------


class TestExprTableAlias:

    def test_with_table_alias_method(self) -> None:
        assert (
            str(Expr("price").with_table_alias("t").eq(70))
            == "`t`.`price` = 70"
        )

    def test_alias_kwarg_equivalent(self) -> None:
        assert (
            str(Expr("price", alias="t").eq(70)) == "`t`.`price` = 70"
        )

    def test_alias_applies_to_every_op(self) -> None:
        h = Expr("c", alias="t")

        assert str(h.eq(1)) == "`t`.`c` = 1"
        assert str(h.ne(1)) == "`t`.`c` != 1"
        assert str(h.lt(1)) == "`t`.`c` < 1"
        assert str(h.in_([1, 2])) == "`t`.`c` IN (1, 2)"
        assert str(h.between(1, 5)) == "`t`.`c` BETWEEN 1 AND 5"
        assert str(h.is_null()) == "`t`.`c` IS NULL"
        assert str(h.like("X%")) == "`t`.`c` LIKE 'X%'"


class TestPredicateTableAlias:

    def test_simple(self) -> None:
        p = Expr("price").eq(70)
        assert str(p.with_table_alias("t")) == "`t`.`price` = 70"

    def test_propagates_through_subtree(self) -> None:
        p = Expr("price").ge(70).and_(Expr("region").in_(["EU", "US"]))

        assert str(p.with_table_alias("t")) == (
            "`t`.`price` >= 70 AND `t`.`region` IN ('EU', 'US')"
        )

    def test_propagates_through_or_not(self) -> None:
        p = Expr("a").eq(1).or_(Expr("b").eq(2)).not_()

        assert (
            str(p.with_table_alias("t")) == "NOT (`t`.`a` = 1 OR `t`.`b` = 2)"
        )

    def test_clear_alias_with_none(self) -> None:
        p = Expr("price", alias="t").eq(70)
        assert str(p) == "`t`.`price` = 70"

        cleared = p.with_table_alias(None)
        assert str(cleared) == "`price` = 70"

    def test_repeated_alias_calls_last_wins(self) -> None:
        p = Expr("a").eq(1).and_(Expr("b").eq(2)).with_table_alias("x")
        assert str(p) == "`x`.`a` = 1 AND `x`.`b` = 2"

        p2 = p.with_table_alias("y")
        assert str(p2) == "`y`.`a` = 1 AND `y`.`b` = 2"


class TestAliasPrequalified:

    def test_dotted_column_keeps_user_qualification(self) -> None:
        assert (
            str(Expr("schema.table.col", alias="t").eq(1))
            == "`schema`.`table`.`col` = 1"
        )

    def test_mixed_qualified_and_bare_only_prefixes_bare(self) -> None:
        p = Expr("price").eq(70).and_(Expr("ref.name").eq("X"))

        assert (
            str(p.with_table_alias("t"))
            == "`t`.`price` = 70 AND `ref`.`name` = 'X'"
        )


class TestAliasFlavors:

    def test_postgres(self) -> None:
        assert (
            str(Expr("price", flavor="postgres", alias="t").eq(70))
            == '"t"."price" = 70'
        )

    def test_tsql_brackets(self) -> None:
        assert (
            str(Expr("p", flavor="tsql", alias="t").eq(5)) == "[t].[p] = 5"
        )

    def test_alias_with_quote_char_doubles(self) -> None:
        assert str(Expr("c", alias="we`ird").eq(1)) == "`we``ird`.`c` = 1"

    def test_with_flavor_preserves_alias(self) -> None:
        p = Expr("price", alias="t").eq(70)
        assert str(p.with_flavor("postgres")) == '"t"."price" = 70'

    def test_with_table_alias_preserves_flavor(self) -> None:
        p = Expr("price", flavor="postgres").eq(70)
        assert str(p.with_table_alias("t")) == '"t"."price" = 70'


class TestAliasInteractions:

    def test_alias_in_param_render(self) -> None:
        p = (
            Expr("price", alias="t").ge(70)
            .and_(Expr("region", alias="t").in_(["EU", "US"]))
        )
        sql, params = p.to_param()

        assert sql == "`t`.`price` >= ? AND `t`.`region` IN (?, ?)"
        assert params == [70, "EU", "US"]

    def test_alias_propagates_through_compaction(self) -> None:
        vals = list(range(1, 1500)) + [9999]
        p = Expr("id", alias="t").in_(vals)

        assert str(p) == "`t`.`id` BETWEEN 1 AND 1499 OR `t`.`id` = 9999"

    def test_alias_with_null_aware_in(self) -> None:
        p = Expr("c", alias="t").in_([1, 2, None])
        assert str(p) == "`t`.`c` IN (1, 2) OR `t`.`c` IS NULL"

    def test_alias_with_mysql_ilike_fallback(self) -> None:
        # MySQL has no ILIKE — alias has to apply inside LOWER(...).
        assert (
            str(Expr("n", flavor="mysql", alias="t").ilike("Foo%"))
            == "LOWER(`t`.`n`) LIKE LOWER('Foo%')"
        )


class TestAliasValidation:

    @pytest.mark.parametrize("alias", ["", "   "])
    def test_empty_or_whitespace_alias_raises(self, alias: str) -> None:
        with pytest.raises(ValueError):
            Expr("c").with_table_alias(alias)

    def test_alias_kwarg_validates(self) -> None:
        with pytest.raises(ValueError):
            Expr("c", alias="")

    def test_predicate_empty_alias_raises(self) -> None:
        with pytest.raises(ValueError):
            Expr("c").eq(1).with_table_alias("")


class TestAliasJoinStyle:

    def test_two_aliased_predicates_under_and(self) -> None:
        trades = Expr("price", alias="t").ge(70)
        ref = Expr("country", alias="r").eq("FR")

        sql, params = trades.and_(ref).to_param()

        assert sql == "`t`.`price` >= ? AND `r`.`country` = ?"
        assert params == [70, "FR"]


# ---------------------------------------------------------------------------
# Realistic end-to-end queries
# ---------------------------------------------------------------------------


class TestRealistic:

    def test_commodity_trade_query_databricks_numeric(self) -> None:
        p = (
            Expr("book")
            .eq("PROP_OIL")
            .and_(Expr("trade_dt").between(date(2026, 1, 1), date(2026, 4, 28)))
            .and_(
                # NULL-aware IN expansion ORs with the user's outer .or_().
                Expr("commodity")
                .in_(["WTI", "Brent", None])
                .or_(Expr("commodity").like("Crude%"))
            )
            .and_(Expr("settled_at").is_not_null())
        )

        sql, params = p.to_param(style="numeric")

        assert sql == (
            "`book` = $1 "
            "AND `trade_dt` BETWEEN $2 AND $3 "
            "AND (`commodity` IN ($4, $5) OR `commodity` IS NULL "
            "OR `commodity` LIKE $6) "
            "AND `settled_at` IS NOT NULL"
        )
        assert params == [
            "PROP_OIL",
            date(2026, 1, 1),
            date(2026, 4, 28),
            "WTI",
            "Brent",
            "Crude%",
        ]

    def test_commodity_trade_query_postgres_render(self) -> None:
        p = (
            Expr("book")
            .eq("PROP_OIL")
            .and_(Expr("trade_dt").ge(date(2026, 1, 1)))
        ).with_flavor("postgres")

        assert (
            str(p)
            == "\"book\" = 'PROP_OIL' AND \"trade_dt\" >= DATE '2026-01-01'"
        )
