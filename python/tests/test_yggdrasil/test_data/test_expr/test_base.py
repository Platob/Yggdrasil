"""Unit tests for yggdrasil.data.sql.expr."""

from __future__ import annotations

import unittest
from datetime import date, datetime, time
from decimal import Decimal

from yggdrasil.data.expr.base import Dialect, NULL, Expr, flavor_of


# ---------------------------------------------------------------------------
# Leaf operators (default Databricks flavor)
# ---------------------------------------------------------------------------

class TestLeafOperators(unittest.TestCase):
    def test_eq_int(self):
        self.assertEqual(str(Expr("price").eq(75)), "`price` = 75")

    def test_eq_string_escapes_quote(self):
        self.assertEqual(str(Expr("name").eq("O'Brien")), "`name` = 'O''Brien'")

    def test_ne(self):
        self.assertEqual(str(Expr("status").ne("cancelled")), "`status` != 'cancelled'")

    def test_lt_le_gt_ge(self):
        self.assertEqual(str(Expr("q").lt(10)), "`q` < 10")
        self.assertEqual(str(Expr("q").le(10)), "`q` <= 10")
        self.assertEqual(str(Expr("q").gt(10)), "`q` > 10")
        self.assertEqual(str(Expr("q").ge(10)), "`q` >= 10")

    def test_like_ilike(self):
        self.assertEqual(str(Expr("n").like("Shell%")), "`n` LIKE 'Shell%'")
        # Databricks supports ILIKE natively.
        self.assertEqual(str(Expr("n").ilike("%X%")), "`n` ILIKE '%X%'")

    def test_between(self):
        self.assertEqual(str(Expr("p").between(70, 90)), "`p` BETWEEN 70 AND 90")
        self.assertEqual(str(Expr("p").not_between(70, 90)), "`p` NOT BETWEEN 70 AND 90")

    def test_is_null_is_not_null(self):
        self.assertEqual(str(Expr("c").is_null()), "`c` IS NULL")
        self.assertEqual(str(Expr("c").is_not_null()), "`c` IS NOT NULL")

    def test_eq_none_routes_to_is_null(self):
        self.assertEqual(str(Expr("c").eq(None)), "`c` IS NULL")
        self.assertEqual(str(Expr("c").eq(NULL)), "`c` IS NULL")

    def test_ne_none_routes_to_is_not_null(self):
        self.assertEqual(str(Expr("c").ne(None)), "`c` IS NOT NULL")

    def test_dotted_column(self):
        self.assertEqual(str(Expr("a.b.c").eq(1)), "`a`.`b`.`c` = 1")

    def test_decimal(self):
        self.assertEqual(str(Expr("p").eq(Decimal("1.5"))), "`p` = 1.5")

    def test_bool(self):
        # Default flavor: TRUE/FALSE keywords.
        self.assertEqual(str(Expr("a").eq(True)), "`a` = TRUE")
        self.assertEqual(str(Expr("a").eq(False)), "`a` = FALSE")

    def test_bytes(self):
        self.assertEqual(str(Expr("c").eq(b"\xde\xad")), "`c` = X'dead'")

    def test_identifier_escapes_quote_char(self):
        # A backtick inside an identifier doubles up.
        self.assertEqual(str(Expr("we`ird").eq(1)), "`we``ird` = 1")


# ---------------------------------------------------------------------------
# Datetime / date / time literals
# ---------------------------------------------------------------------------

class TestDatetimeLiterals(unittest.TestCase):
    def test_datetime_databricks_typed(self):
        ts = datetime(2026, 4, 28, 14, 30, 5)
        self.assertEqual(str(Expr("ts").eq(ts)), "`ts` = TIMESTAMP '2026-04-28 14:30:05'")

    def test_date_databricks_typed(self):
        self.assertEqual(str(Expr("d").eq(date(2026, 12, 1))), "`d` = DATE '2026-12-01'")

    def test_time_databricks_typed(self):
        self.assertEqual(str(Expr("t").eq(time(14, 30))), "`t` = TIME '14:30:00'")

    def test_datetime_postgres_typed(self):
        ts = datetime(2026, 1, 2, 3, 4, 5)
        p = Expr("ts", flavor="postgres").eq(ts)
        self.assertEqual(str(p), '"ts" = TIMESTAMP \'2026-01-02 03:04:05\'')

    def test_datetime_mysql_untyped(self):
        ts = datetime(2026, 1, 2, 3, 4, 5)
        p = Expr("ts", flavor="mysql").eq(ts)
        self.assertEqual(str(p), "`ts` = '2026-01-02 03:04:05'")

    def test_datetime_sqlite_untyped(self):
        self.assertEqual(
            str(Expr("d", flavor="sqlite").eq(date(2026, 12, 1))),
            "\"d\" = '2026-12-01'",
        )

    def test_datetime_in_list_databricks(self):
        # Each value gets the typed prefix individually.
        a = datetime(2026, 1, 1)
        b = datetime(2026, 2, 1)
        sql = str(Expr("ts").in_([a, b]))
        self.assertEqual(
            sql,
            "`ts` IN (TIMESTAMP '2026-01-01 00:00:00', TIMESTAMP '2026-02-01 00:00:00')",
        )

    def test_datetime_between_databricks(self):
        sql = str(Expr("d").between(date(2026, 1, 1), date(2026, 12, 31)))
        self.assertEqual(sql, "`d` BETWEEN DATE '2026-01-01' AND DATE '2026-12-31'")


# ---------------------------------------------------------------------------
# IN / NOT IN with NULL
# ---------------------------------------------------------------------------

class TestNullAwareIn(unittest.TestCase):
    def test_in_no_null(self):
        self.assertEqual(
            str(Expr("c").in_([1, 2, 3])),
            "`c` IN (1, 2, 3)",
        )

    def test_in_with_none_expands(self):
        self.assertEqual(
            str(Expr("c").in_([1, 2, None])),
            "`c` IN (1, 2) OR `c` IS NULL",
        )

    def test_in_with_NULL_sentinel_expands(self):
        self.assertEqual(
            str(Expr("c").in_([1, NULL, 2])),
            "`c` IN (1, 2) OR `c` IS NULL",
        )

    def test_in_only_null(self):
        # `in_([None])` collapses to `IS NULL`.
        self.assertEqual(str(Expr("c").in_([None])), "`c` IS NULL")

    def test_in_tuple_with_none(self):
        self.assertEqual(
            str(Expr("c").in_((1, None))),
            "`c` IN (1) OR `c` IS NULL",
        )

    def test_in_with_none_gets_parens_under_and(self):
        # The expansion is an OR; when nested under AND, it must be parenthesized.
        p = Expr("c").in_([1, None]).and_(Expr("d").eq(5))
        self.assertEqual(str(p), "(`c` IN (1) OR `c` IS NULL) AND `d` = 5")

    def test_not_in_no_null(self):
        self.assertEqual(
            str(Expr("c").not_in([1, 2])),
            "`c` NOT IN (1, 2)",
        )

    def test_not_in_with_none_expands_to_and(self):
        # SQL semantics: NOT IN with NULL filters everything out — we expand to
        # (NOT IN (non_nulls) AND IS NOT NULL) so the predicate behaves as expected.
        self.assertEqual(
            str(Expr("c").not_in([1, None])),
            "`c` NOT IN (1) AND `c` IS NOT NULL",
        )

    def test_not_in_only_null(self):
        self.assertEqual(str(Expr("c").not_in([None])), "`c` IS NOT NULL")

    def test_in_param_expansion_no_null(self):
        sql, params = Expr("c").in_([1, 2, 3]).to_param()
        self.assertEqual(sql, "`c` IN (?, ?, ?)")
        self.assertEqual(params, [1, 2, 3])

    def test_in_param_expansion_with_null(self):
        # NULL is expanded into IS NULL and contributes nothing to params.
        sql, params = Expr("c").in_([1, 2, None]).to_param()
        self.assertEqual(sql, "`c` IN (?, ?) OR `c` IS NULL")
        self.assertEqual(params, [1, 2])

    def test_not_in_param_expansion_with_null(self):
        sql, params = Expr("c").not_in([1, None]).to_param()
        self.assertEqual(sql, "`c` NOT IN (?) AND `c` IS NOT NULL")
        self.assertEqual(params, [1])


# ---------------------------------------------------------------------------
# Composition + precedence
# ---------------------------------------------------------------------------

class TestComposition(unittest.TestCase):
    def test_and(self):
        p = Expr("a").eq(1).and_(Expr("b").eq(2))
        self.assertEqual(str(p), "`a` = 1 AND `b` = 2")

    def test_or(self):
        p = Expr("a").eq(1).or_(Expr("b").eq(2))
        self.assertEqual(str(p), "`a` = 1 OR `b` = 2")

    def test_and_chain_flattens(self):
        p = Expr("a").eq(1).and_(Expr("b").eq(2)).and_(Expr("c").eq(3))
        self.assertEqual(str(p), "`a` = 1 AND `b` = 2 AND `c` = 3")

    def test_or_chain_flattens(self):
        p = Expr("a").eq(1).or_(Expr("b").eq(2)).or_(Expr("c").eq(3))
        self.assertEqual(str(p), "`a` = 1 OR `b` = 2 OR `c` = 3")

    def test_or_inside_and_gets_parens(self):
        p = Expr("a").eq(1).and_(Expr("b").eq(2).or_(Expr("c").eq(3)))
        self.assertEqual(str(p), "`a` = 1 AND (`b` = 2 OR `c` = 3)")

    def test_or_left_of_and_gets_parens(self):
        p = Expr("a").eq(1).or_(Expr("b").eq(2)).and_(Expr("c").eq(3))
        self.assertEqual(str(p), "(`a` = 1 OR `b` = 2) AND `c` = 3")

    def test_not_on_leaf(self):
        self.assertEqual(str(Expr("a").eq(1).not_()), "NOT (`a` = 1)")

    def test_not_on_compound(self):
        p = Expr("a").eq(1).and_(Expr("b").eq(2)).not_()
        self.assertEqual(str(p), "NOT (`a` = 1 AND `b` = 2)")

    def test_double_not_collapses(self):
        p = Expr("a").eq(1).not_().not_()
        self.assertEqual(str(p), "`a` = 1")

    def test_op_overloads(self):
        p = (Expr("a") >= 1) & (Expr("b") < 10)
        self.assertEqual(str(p), "`a` >= 1 AND `b` < 10")
        p = (Expr("a") == 1) | (Expr("b") == 2)
        self.assertEqual(str(p), "`a` = 1 OR `b` = 2")
        p = ~((Expr("a") == 1) & (Expr("b") == 2))
        self.assertEqual(str(p), "NOT (`a` = 1 AND `b` = 2)")


# ---------------------------------------------------------------------------
# Parameterized rendering
# ---------------------------------------------------------------------------

class TestParam(unittest.TestCase):
    def test_qmark_default(self):
        sql, params = Expr("a").eq(1).to_param()
        self.assertEqual(sql, "`a` = ?")
        self.assertEqual(params, [1])

    def test_numeric(self):
        sql, params = Expr("a").eq(1).and_(Expr("b").in_([2, 3])).to_param(style="numeric")
        self.assertEqual(sql, "`a` = $1 AND `b` IN ($2, $3)")
        self.assertEqual(params, [1, 2, 3])

    def test_named(self):
        sql, params = Expr("a").eq(1).and_(Expr("b").in_([2, 3])).to_param(style="named")
        self.assertEqual(sql, "`a` = :p0 AND `b` IN (:p1, :p2)")
        self.assertEqual(params, {"p0": 1, "p1": 2, "p2": 3})

    def test_pyformat(self):
        sql, params = Expr("a").eq(1).to_param(style="pyformat")
        self.assertEqual(sql, "`a` = %(p0)s")
        self.assertEqual(params, {"p0": 1})

    def test_start_offset_numeric(self):
        sql, params = Expr("a").eq(1).and_(Expr("b").in_([2, 3])).to_param(style="numeric", start=5)
        self.assertEqual(sql, "`a` = $6 AND `b` IN ($7, $8)")
        self.assertEqual(params, [1, 2, 3])

    def test_start_offset_named(self):
        sql, params = Expr("a").eq(1).to_param(style="named", start=10)
        self.assertEqual(sql, "`a` = :p10")
        self.assertEqual(params, {"p10": 1})

    def test_unary_no_params(self):
        sql, params = Expr("c").is_null().to_param()
        self.assertEqual(sql, "`c` IS NULL")
        self.assertEqual(params, [])

    def test_not_renders_params_once(self):
        # Regression: prior version walked the NOT child twice and double-bound params.
        sql, params = Expr("a").eq(1).and_(Expr("b").eq(2)).not_().to_param()
        self.assertEqual(sql, "NOT (`a` = ? AND `b` = ?)")
        self.assertEqual(params, [1, 2])

    def test_unknown_param_style(self):
        with self.assertRaises(ValueError):
            Expr("a").eq(1).to_param(style="bogus")

    def test_datetime_params_passed_through(self):
        # For parameterized execution, the datetime is passed as-is — the driver handles it.
        ts = datetime(2026, 4, 28, 14, 30)
        sql, params = Expr("ts").eq(ts).to_param()
        self.assertEqual(sql, "`ts` = ?")
        self.assertEqual(params, [ts])


# ---------------------------------------------------------------------------
# Dialect flavors
# ---------------------------------------------------------------------------

class TestFlavors(unittest.TestCase):
    def test_default_is_databricks(self):
        self.assertEqual(str(Expr("c").eq(1)), "`c` = 1")

    def test_postgres(self):
        self.assertEqual(str(Expr("c", flavor="postgres").eq(1)), '"c" = 1')

    def test_mysql(self):
        self.assertEqual(str(Expr("c", flavor="mysql").eq(1)), "`c` = 1")

    def test_tsql_brackets(self):
        self.assertEqual(str(Expr("c", flavor="tsql").eq(1)), "[c] = 1")

    def test_tsql_bool_as_int(self):
        self.assertEqual(str(Expr("a", flavor="tsql").eq(True)), "[a] = 1")
        self.assertEqual(str(Expr("a", flavor="tsql").eq(False)), "[a] = 0")

    def test_tsql_bytes_0x(self):
        self.assertEqual(
            str(Expr("c", flavor="tsql").eq(b"\xde\xad")),
            "[c] = 0xdead",
        )

    def test_ilike_fallback_mysql(self):
        # MySQL has no ILIKE — falls back to LOWER(...) LIKE LOWER(...).
        sql = str(Expr("n", flavor="mysql").ilike("Foo%"))
        self.assertEqual(sql, "LOWER(`n`) LIKE LOWER('Foo%')")

    def test_ilike_fallback_param(self):
        sql, params = Expr("n", flavor="mysql").ilike("Foo%").to_param()
        self.assertEqual(sql, "LOWER(`n`) LIKE LOWER(?)")
        self.assertEqual(params, ["Foo%"])

    def test_ilike_native_databricks(self):
        sql, params = Expr("n").ilike("Foo%").to_param()
        self.assertEqual(sql, "`n` ILIKE ?")
        self.assertEqual(params, ["Foo%"])

    def test_dialect_enum_accepted(self):
        self.assertEqual(str(Expr("c", flavor=Dialect.POSTGRES).eq(1)), '"c" = 1')

    def test_flavor_object_accepted(self):
        f = flavor_of("postgres")
        self.assertEqual(str(Expr("c", flavor=f).eq(1)), '"c" = 1')

    def test_unknown_dialect(self):
        with self.assertRaises(ValueError):
            Expr("c", flavor="oracle").eq(1)

    def test_with_flavor_switches_tree(self):
        # Build with default Databricks, then re-render as Postgres.
        p = Expr("price").ge(70).and_(Expr("ts").eq(datetime(2026, 1, 2, 3, 4, 5)))
        self.assertEqual(str(p), "`price` >= 70 AND `ts` = TIMESTAMP '2026-01-02 03:04:05'")
        p2 = p.with_flavor("postgres")
        self.assertEqual(str(p2), '"price" >= 70 AND "ts" = TIMESTAMP \'2026-01-02 03:04:05\'')

    def test_sqlexpr_with_flavor(self):
        # Expr also has with_flavor (symmetric with with_table_alias).
        e = Expr("price").with_flavor("postgres")
        self.assertIsInstance(e, Expr)
        self.assertEqual(str(e.eq(70)), '"price" = 70')

    def test_sqlexpr_with_flavor_preserves_alias(self):
        e = Expr("price", alias="t").with_flavor("postgres")
        self.assertEqual(str(e.eq(70)), '"t"."price" = 70')

    def test_sqlexpr_class_is_public(self):
        # Sanity: Expr is importable and is what `Expr()` returns.
        e = Expr("price")
        self.assertIsInstance(e, Expr)

    def test_flavor_propagates_through_composition(self):
        a = Expr("a", flavor="postgres").eq(1)
        b = Expr("b", flavor="postgres").eq(2)
        self.assertEqual(str(a.and_(b)), '"a" = 1 AND "b" = 2')

    def test_dialect_aliases(self):
        self.assertIs(Dialect.parse("MSSQL"), Dialect.TSQL)
        self.assertIs(Dialect.parse("SqlServer"), Dialect.TSQL)
        self.assertIs(Dialect.parse("ANSI"), Dialect.STANDARD)
        self.assertIs(Dialect.parse("sparksql"), Dialect.SPARK)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):
    def test_empty_column(self):
        with self.assertRaises(ValueError):
            Expr("")

    def test_whitespace_column(self):
        with self.assertRaises(ValueError):
            Expr("   ")

    def test_eq_rejects_non_scalar(self):
        with self.assertRaises(TypeError):
            Expr("c").eq([1, 2])

    def test_lt_rejects_null(self):
        with self.assertRaises(ValueError):
            Expr("c").lt(None)

    def test_in_empty_list(self):
        with self.assertRaises(ValueError):
            Expr("c").in_([])

    def test_in_none_value(self):
        with self.assertRaises(ValueError):
            Expr("c").in_(None)

    def test_like_non_string(self):
        with self.assertRaises(TypeError):
            Expr("c").like(5)

    def test_and_with_non_predicate(self):
        with self.assertRaises(TypeError):
            Expr("a").eq(1).and_("not a predicate")  # type: ignore[arg-type]

    def test_or_with_non_predicate(self):
        with self.assertRaises(TypeError):
            Expr("a").eq(1).or_(42)  # type: ignore[arg-type]

    def test_dotted_empty_segment(self):
        with self.assertRaises(ValueError):
            Expr("a..b").eq(1)

    def test_between_rejects_null(self):
        with self.assertRaises(ValueError):
            Expr("c").between(None, 5)


# ---------------------------------------------------------------------------
# Realistic end-to-end query
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# between() / not_between() collection form
# ---------------------------------------------------------------------------

class TestBetweenCollection(unittest.TestCase):
    # --- pair form (regression coverage for the original API) ---

    def test_pair_form_still_works(self):
        self.assertEqual(str(Expr("p").between(70, 90)), "`p` BETWEEN 70 AND 90")
        self.assertEqual(str(Expr("p").not_between(70, 90)), "`p` NOT BETWEEN 70 AND 90")

    # --- list / tuple / set ---

    def test_list_min_max(self):
        self.assertEqual(
            str(Expr("p").between([5, 1, 3, 9, 4])),
            "`p` BETWEEN 1 AND 9",
        )

    def test_tuple_min_max(self):
        self.assertEqual(str(Expr("p").between((10, 2, 7))), "`p` BETWEEN 2 AND 10")

    def test_set_min_max(self):
        # Order in a set is unspecified, but min/max are well-defined.
        self.assertEqual(str(Expr("p").between({3, 1, 5, 2})), "`p` BETWEEN 1 AND 5")

    def test_generator_min_max(self):
        gen = (i for i in [4, 1, 7, 2])
        self.assertEqual(str(Expr("p").between(gen)), "`p` BETWEEN 1 AND 7")

    # --- None / NULL filtering ---

    def test_list_with_none_expands(self):
        # NULL in the collection -> OR IS NULL appended.
        self.assertEqual(
            str(Expr("p").between([5, None, 1, 3])),
            "`p` BETWEEN 1 AND 5 OR `p` IS NULL",
        )

    def test_list_with_NULL_sentinel_expands(self):
        self.assertEqual(
            str(Expr("p").between([5, NULL, 1, 3])),
            "`p` BETWEEN 1 AND 5 OR `p` IS NULL",
        )

    def test_no_null_no_expansion(self):
        # Collection without None stays as a plain BETWEEN.
        self.assertEqual(str(Expr("p").between([5, 1, 3])), "`p` BETWEEN 1 AND 5")

    def test_list_only_nones_raises(self):
        with self.assertRaises(ValueError):
            Expr("p").between([None, None, NULL])

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            Expr("p").between([])

    def test_singleton_list_lo_eq_hi(self):
        # Single-element collection: bounds equal, equivalent to `= 3`.
        self.assertEqual(str(Expr("p").between([3])), "`p` BETWEEN 3 AND 3")

    def test_singleton_with_none_expands(self):
        self.assertEqual(
            str(Expr("p").between([None, 7, None])),
            "`p` BETWEEN 7 AND 7 OR `p` IS NULL",
        )

    def test_expansion_under_and_gets_parens(self):
        # The expansion is an OR; nested under AND it must be parenthesized.
        p = Expr("p").between([5, None, 1]).and_(Expr("q").eq(0))
        self.assertEqual(str(p), "(`p` BETWEEN 1 AND 5 OR `p` IS NULL) AND `q` = 0")

    # --- not_between mirror cases ---

    def test_not_between_no_null(self):
        self.assertEqual(
            str(Expr("p").not_between([5, 1, 3])),
            "`p` NOT BETWEEN 1 AND 5",
        )

    def test_not_between_with_none_expands_to_and(self):
        # NOT BETWEEN with NULL would silently filter every row whose col is NULL,
        # so we expand to NOT BETWEEN ... AND IS NOT NULL.
        self.assertEqual(
            str(Expr("p").not_between([5, None, 1, 3])),
            "`p` NOT BETWEEN 1 AND 5 AND `p` IS NOT NULL",
        )

    def test_not_between_only_null_raises(self):
        with self.assertRaises(ValueError):
            Expr("p").not_between([None, NULL])

    # --- types other than int ---

    def test_float_collection(self):
        self.assertEqual(
            str(Expr("p").between([1.5, 0.5, 2.5])),
            "`p` BETWEEN 0.5 AND 2.5",
        )

    def test_string_collection_with_none_expands(self):
        self.assertEqual(
            str(Expr("name").between(["m", "a", "z", None])),
            "`name` BETWEEN 'a' AND 'z' OR `name` IS NULL",
        )

    def test_date_collection_with_none_expands(self):
        d1, d2, d3 = date(2026, 6, 1), date(2026, 1, 1), date(2026, 12, 31)
        self.assertEqual(
            str(Expr("d").between([d1, d2, d3, None])),
            "`d` BETWEEN DATE '2026-01-01' AND DATE '2026-12-31' OR `d` IS NULL",
        )

    def test_datetime_collection_postgres_with_none(self):
        a = datetime(2026, 4, 28, 14, 30)
        b = datetime(2026, 1, 2, 3, 4, 5)
        self.assertEqual(
            str(Expr("ts", flavor="postgres").between([a, None, b])),
            "\"ts\" BETWEEN TIMESTAMP '2026-01-02 03:04:05' "
            "AND TIMESTAMP '2026-04-28 14:30:00' "
            "OR \"ts\" IS NULL",
        )

    # --- parameter rendering ---

    def test_collection_param_no_null(self):
        sql, params = Expr("p").between([5, 1, 3]).to_param()
        self.assertEqual(sql, "`p` BETWEEN ? AND ?")
        self.assertEqual(params, [1, 5])

    def test_collection_param_with_null(self):
        # NULL doesn't bind a placeholder; only the real bounds do.
        sql, params = Expr("p").between([5, None, 1, 3]).to_param()
        self.assertEqual(sql, "`p` BETWEEN ? AND ? OR `p` IS NULL")
        self.assertEqual(params, [1, 5])

    def test_collection_param_numeric_with_null(self):
        sql, params = Expr("p").between([5, None, 1, 3]).to_param(style="numeric")
        self.assertEqual(sql, "`p` BETWEEN $1 AND $2 OR `p` IS NULL")
        self.assertEqual(params, [1, 5])

    def test_not_between_param_with_null(self):
        sql, params = Expr("p").not_between([5, None, 1]).to_param()
        self.assertEqual(sql, "`p` NOT BETWEEN ? AND ? AND `p` IS NOT NULL")
        self.assertEqual(params, [1, 5])

    # --- error paths ---

    def test_scalar_one_arg_raises(self):
        # Scalar with no `high` is ambiguous — caller probably forgot the second bound.
        with self.assertRaises(TypeError):
            Expr("p").between(5)

    def test_string_one_arg_treated_as_scalar(self):
        # A bare string is NOT iterated character-by-character (consistent with in_).
        with self.assertRaises(TypeError):
            Expr("name").between("hello")

    def test_bytes_one_arg_treated_as_scalar(self):
        with self.assertRaises(TypeError):
            Expr("c").between(b"abc")

    def test_mixed_types_raises(self):
        with self.assertRaises(TypeError):
            Expr("p").between([1, "two", 3])

    def test_none_one_arg_raises(self):
        with self.assertRaises(TypeError):
            Expr("p").between(None)

    def test_pair_form_rejects_null(self):
        # Existing behavior preserved: the explicit pair form still rejects NULL bounds.
        with self.assertRaises(ValueError):
            Expr("p").between(None, 5)
        with self.assertRaises(ValueError):
            Expr("p").between(5, None)


# ---------------------------------------------------------------------------
# Array-like sniffing (Arrow / polars / pandas / numpy / duck-typed)
# ---------------------------------------------------------------------------

class _FakeToList:
    """Duck-typed Arrow/polars-style array exposing ``to_list()``."""
    def __init__(self, values):
        self._v = values

    def to_list(self):
        return list(self._v)


class _FakeTolist:
    """Duck-typed pandas/numpy-style array exposing ``tolist()`` (lowercase l)."""
    def __init__(self, values):
        self._v = values

    def tolist(self):
        return list(self._v)


class _FakeBoth:
    """Some arrow versions expose both — to_list() should win."""
    def to_list(self):
        return [1, 2, 3]

    def tolist(self):
        # If sniffer falls through to this, we'd see [9, 9, 9].
        return [9, 9, 9]


class _FakeBadToList:
    """to_list() that raises TypeError (e.g. requires args) — sniffer must fall through."""
    def to_list(self, *args, **kwargs):
        raise TypeError("requires args")

    def tolist(self):
        return [10, 20]


class TestArrayLikeSniffing(unittest.TestCase):
    """Verify in_/not_in/between accept Arrow/polars/pandas/numpy-style arrays."""

    # --- duck-typed (no external library needed) ---

    def test_in_with_to_list(self):
        arr = _FakeToList([1, 2, 3])
        self.assertEqual(str(Expr("c").in_(arr)), "`c` IN (1, 2, 3)")

    def test_in_with_tolist(self):
        arr = _FakeTolist([1, 2, 3])
        self.assertEqual(str(Expr("c").in_(arr)), "`c` IN (1, 2, 3)")

    def test_in_with_to_list_containing_none(self):
        arr = _FakeToList([1, None, 2])
        self.assertEqual(
            str(Expr("c").in_(arr)),
            "`c` IN (1, 2) OR `c` IS NULL",
        )

    def test_to_list_preferred_over_tolist(self):
        # If both methods exist, to_list() (Arrow/polars convention) wins.
        self.assertEqual(str(Expr("c").in_(_FakeBoth())), "`c` IN (1, 2, 3)")

    def test_to_list_typeerror_falls_through_to_tolist(self):
        # If to_list() raises TypeError, the sniffer falls through to tolist().
        self.assertEqual(str(Expr("c").in_(_FakeBadToList())), "`c` IN (10, 20)")

    def test_not_in_with_to_list(self):
        arr = _FakeToList([1, None])
        self.assertEqual(
            str(Expr("c").not_in(arr)),
            "`c` NOT IN (1) AND `c` IS NOT NULL",
        )

    def test_between_with_to_list(self):
        arr = _FakeToList([5, 1, 3])
        self.assertEqual(str(Expr("p").between(arr)), "`p` BETWEEN 1 AND 5")

    def test_between_with_to_list_and_none(self):
        arr = _FakeToList([5, None, 1, 3])
        self.assertEqual(
            str(Expr("p").between(arr)),
            "`p` BETWEEN 1 AND 5 OR `p` IS NULL",
        )

    def test_between_with_tolist(self):
        arr = _FakeTolist([10, 2, 7])
        self.assertEqual(str(Expr("p").between(arr)), "`p` BETWEEN 2 AND 10")

    def test_not_between_with_tolist_and_none(self):
        arr = _FakeTolist([5, None, 1])
        self.assertEqual(
            str(Expr("p").not_between(arr)),
            "`p` NOT BETWEEN 1 AND 5 AND `p` IS NOT NULL",
        )

    def test_in_param_with_to_list(self):
        arr = _FakeToList([1, None, 2])
        sql, params = Expr("c").in_(arr).to_param()
        self.assertEqual(sql, "`c` IN (?, ?) OR `c` IS NULL")
        self.assertEqual(params, [1, 2])

    # --- NaN handling ---

    def test_nan_treated_as_null_in_collection(self):
        # pandas users get NaN for missing numeric values — treat as NULL.
        nan = float("nan")
        self.assertEqual(
            str(Expr("p").between([5.0, nan, 1.0, 3.0])),
            "`p` BETWEEN 1.0 AND 5.0 OR `p` IS NULL",
        )

    def test_nan_in_eq_routes_to_is_null(self):
        nan = float("nan")
        self.assertEqual(str(Expr("c").eq(nan)), "`c` IS NULL")

    def test_nan_in_lt_rejected(self):
        nan = float("nan")
        with self.assertRaises(ValueError):
            Expr("c").lt(nan)

    # --- regular Python sequences still work ---

    def test_native_list_unchanged(self):
        # Sanity check that native containers don't go through the sniffer
        # (and that lists with a fluke `to_list` attr aren't broken — there
        # aren't any in stdlib, but the test pins the behavior).
        self.assertEqual(str(Expr("c").in_([1, 2, 3])), "`c` IN (1, 2, 3)")

    # --- real libraries (skip if not installed) ---

    def test_arrow_array(self):
        try:
            import pyarrow as pa
        except ImportError:
            self.skipTest("pyarrow not installed")
        arr = pa.array([1, 2, None, 3])
        self.assertEqual(
            str(Expr("c").in_(arr)),
            "`c` IN (1, 2, 3) OR `c` IS NULL",
        )

    def test_arrow_chunked_array(self):
        try:
            import pyarrow as pa
        except ImportError:
            self.skipTest("pyarrow not installed")
        arr = pa.chunked_array([[1, 2], [None, 3]])
        self.assertEqual(
            str(Expr("c").in_(arr)),
            "`c` IN (1, 2, 3) OR `c` IS NULL",
        )

    def test_arrow_between(self):
        try:
            import pyarrow as pa
        except ImportError:
            self.skipTest("pyarrow not installed")
        arr = pa.array([5, None, 1, 3])
        self.assertEqual(
            str(Expr("p").between(arr)),
            "`p` BETWEEN 1 AND 5 OR `p` IS NULL",
        )

    def test_polars_series(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        s = pl.Series([1, 2, None, 3])
        self.assertEqual(
            str(Expr("c").in_(s)),
            "`c` IN (1, 2, 3) OR `c` IS NULL",
        )

    def test_polars_between(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        s = pl.Series([5, None, 1, 3])
        self.assertEqual(
            str(Expr("p").between(s)),
            "`p` BETWEEN 1 AND 5 OR `p` IS NULL",
        )

    def test_pandas_series(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        # NB: pandas upcasts [1, 2, None, 3] to float64 because None forces a
        # numeric type that can hold NaN. The sniffer correctly handles the
        # NaN (treating it as NULL), but the surviving values are floats.
        s = pd.Series([1, 2, None, 3])
        self.assertEqual(
            str(Expr("c").in_(s)),
            "`c` IN (1.0, 2.0, 3.0) OR `c` IS NULL",
        )

    def test_pandas_object_series_preserves_int(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        # An explicit object-dtype series keeps None as None and ints as ints.
        s = pd.Series([1, 2, None, 3], dtype=object)
        self.assertEqual(
            str(Expr("c").in_(s)),
            "`c` IN (1, 2, 3) OR `c` IS NULL",
        )

    def test_pandas_index(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        idx = pd.Index([1, 2, 3])
        self.assertEqual(str(Expr("c").in_(idx)), "`c` IN (1, 2, 3)")

    def test_numpy_array(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        arr = np.array([1, 2, 3])
        self.assertEqual(str(Expr("c").in_(arr)), "`c` IN (1, 2, 3)")

    def test_numpy_array_with_nan(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        arr = np.array([1.0, float("nan"), 3.0])
        self.assertEqual(
            str(Expr("c").in_(arr)),
            "`c` IN (1.0, 3.0) OR `c` IS NULL",
        )


# ---------------------------------------------------------------------------
# Compaction of large IN lists into BETWEEN-of-runs
# ---------------------------------------------------------------------------

class TestCompaction(unittest.TestCase):
    """Large `in_()` / `not_in()` values get compacted into BETWEEN clauses
    when the values are integer-step (int or date) and exceed the threshold.
    """

    # --- contiguous-range case (the big win) ---

    def test_contiguous_int_range_one_between(self):
        p = Expr("id").in_(list(range(1, 5001)))
        self.assertEqual(str(p), "`id` BETWEEN 1 AND 5000")

    def test_contiguous_descending_input_still_compacts(self):
        # Input order is irrelevant — compaction sorts internally.
        p = Expr("id").in_(list(range(5000, 0, -1)))
        self.assertEqual(str(p), "`id` BETWEEN 1 AND 5000")

    def test_contiguous_with_duplicates(self):
        # Duplicates are dropped silently before run detection.
        vals = list(range(1, 2001)) + list(range(500, 600))
        p = Expr("id").in_(vals)
        self.assertEqual(str(p), "`id` BETWEEN 1 AND 2000")

    # --- multiple runs ---

    def test_multiple_runs(self):
        vals = list(range(1, 1500)) + list(range(2000, 2010)) + [3000, 5000]
        p = Expr("id").in_(vals)
        self.assertEqual(
            str(p),
            "`id` BETWEEN 1 AND 1499 "
            "OR `id` BETWEEN 2000 AND 2009 "
            "OR `id` = 3000 "
            "OR `id` = 5000",
        )

    def test_two_runs_param(self):
        vals = list(range(1, 1500)) + list(range(3000, 3010))
        sql, params = Expr("id").in_(vals).to_param()
        self.assertEqual(sql, "`id` BETWEEN ? AND ? OR `id` BETWEEN ? AND ?")
        self.assertEqual(params, [1, 1499, 3000, 3009])

    # --- NOT IN ---

    def test_not_in_contiguous_compaction(self):
        p = Expr("id").not_in(list(range(1, 5001)))
        self.assertEqual(str(p), "`id` NOT BETWEEN 1 AND 5000")

    def test_not_in_multiple_runs_uses_AND(self):
        vals = list(range(1, 1500)) + list(range(3000, 3010))
        p = Expr("id").not_in(vals)
        # NOT IN over multiple runs becomes AND of NOT BETWEENs.
        self.assertEqual(
            str(p),
            "`id` NOT BETWEEN 1 AND 1499 "
            "AND `id` NOT BETWEEN 3000 AND 3009",
        )

    def test_not_in_singleton_run_uses_ne(self):
        vals = list(range(1, 1500)) + [9999]
        p = Expr("id").not_in(vals)
        self.assertEqual(
            str(p),
            "`id` NOT BETWEEN 1 AND 1499 AND `id` != 9999",
        )

    # --- NULL-aware on compacted output ---

    def test_compacted_with_none(self):
        vals = list(range(1, 1500)) + [None]
        p = Expr("id").in_(vals)
        self.assertEqual(str(p), "`id` BETWEEN 1 AND 1499 OR `id` IS NULL")

    def test_compacted_not_in_with_none(self):
        vals = list(range(1, 1500)) + [None]
        p = Expr("id").not_in(vals)
        self.assertEqual(
            str(p),
            "`id` NOT BETWEEN 1 AND 1499 AND `id` IS NOT NULL",
        )

    # --- threshold semantics ---

    def test_below_threshold_no_compaction(self):
        # Default threshold is 1000; 999 contiguous values stay as IN.
        vals = list(range(1, 1000))
        p = Expr("id").in_(vals)
        self.assertTrue(str(p).startswith("`id` IN ("))
        self.assertIn(" 999)", str(p))

    def test_at_threshold_no_compaction(self):
        # Boundary: exactly 1000 stays as IN (compaction kicks in for >1000).
        vals = list(range(1, 1001))
        p = Expr("id").in_(vals)
        self.assertTrue(str(p).startswith("`id` IN ("))

    def test_just_above_threshold_compacts(self):
        vals = list(range(1, 1002))   # 1001 elements
        p = Expr("id").in_(vals)
        self.assertEqual(str(p), "`id` BETWEEN 1 AND 1001")

    def test_per_call_threshold_lower(self):
        # User can lower the threshold to compact smaller lists.
        p = Expr("id").in_([1, 2, 3, 4, 5], compact_threshold=3)
        self.assertEqual(str(p), "`id` BETWEEN 1 AND 5")

    def test_per_call_threshold_zero_disables(self):
        vals = list(range(1, 5001))
        p = Expr("id").in_(vals, compact_threshold=0)
        # 5000-element IN with no compaction — sanity-check it's plain IN.
        self.assertTrue(str(p).startswith("`id` IN ("))
        sql, params = p.to_param()
        self.assertEqual(len(params), 5000)

    def test_compaction_falls_back_when_too_sparse(self):
        # 1500 scattered singletons -> 1500 runs, which exceeds threshold.
        # Builder should fall back to plain IN rather than emit 1500 ORs.
        vals = list(range(0, 3000, 2))   # 1500 even numbers, no runs
        p = Expr("id").in_(vals)
        self.assertTrue(str(p).startswith("`id` IN ("))

    # --- types that DON'T compact ---

    def test_floats_no_compaction(self):
        vals = [float(i) for i in range(1100)]
        p = Expr("p").in_(vals, compact_threshold=500)
        self.assertTrue(str(p).startswith("`p` IN ("))

    def test_strings_no_compaction(self):
        vals = [f"item_{i}" for i in range(1100)]
        p = Expr("s").in_(vals, compact_threshold=500)
        self.assertTrue(str(p).startswith("`s` IN ("))

    def test_decimals_no_compaction(self):
        vals = [Decimal(i) for i in range(1100)]
        p = Expr("p").in_(vals, compact_threshold=500)
        self.assertTrue(str(p).startswith("`p` IN ("))

    def test_datetimes_no_compaction(self):
        # datetime "next" is ambiguous (microsecond? second?) — we don't compact.
        base = datetime(2026, 1, 1)
        vals = [datetime.fromtimestamp(base.timestamp() + i) for i in range(1100)]
        p = Expr("ts").in_(vals, compact_threshold=500)
        self.assertTrue("IN (" in str(p))

    def test_bools_no_compaction(self):
        # Edge case: bool subclasses int. Don't try to compact True/False.
        vals = [True] * 1100 + [False] * 100
        p = Expr("b").in_(vals, compact_threshold=500)
        self.assertTrue(str(p).startswith("`b` IN ("))

    def test_mixed_int_and_date_no_compaction(self):
        vals = list(range(500)) + [date(2026, i + 1, 1) for i in range(12)] * 50
        p = Expr("c").in_(vals, compact_threshold=100)
        self.assertTrue(str(p).startswith("`c` IN ("))

    # --- date range ---

    def test_contiguous_date_range_compacts(self):
        from datetime import timedelta
        start = date(2026, 1, 1)
        dates = [start + timedelta(days=i) for i in range(1100)]
        end = start + timedelta(days=1099)
        p = Expr("d").in_(dates)
        self.assertEqual(
            str(p),
            f"`d` BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'",
        )

    def test_sparse_dates_compact_to_runs(self):
        from datetime import timedelta
        start = date(2026, 1, 1)
        # Two runs of consecutive dates.
        run1 = [start + timedelta(days=i) for i in range(1100)]
        run2 = [start + timedelta(days=i) for i in range(2000, 2010)]
        p = Expr("d").in_(run1 + run2)
        self.assertEqual(
            str(p),
            f"`d` BETWEEN DATE '{run1[0].isoformat()}' AND DATE '{run1[-1].isoformat()}' "
            f"OR `d` BETWEEN DATE '{run2[0].isoformat()}' AND DATE '{run2[-1].isoformat()}'",
        )

    # --- composition still works ---

    def test_compacted_in_under_and(self):
        vals = list(range(1, 1500))
        p = Expr("id").in_(vals).and_(Expr("active").eq(True))
        self.assertEqual(
            str(p),
            "`id` BETWEEN 1 AND 1499 AND `active` = TRUE",
        )

    def test_compacted_multiple_runs_under_and_gets_parens(self):
        vals = list(range(1, 1500)) + list(range(3000, 3010))
        p = Expr("id").in_(vals).and_(Expr("active").eq(True))
        self.assertEqual(
            str(p),
            "(`id` BETWEEN 1 AND 1499 OR `id` BETWEEN 3000 AND 3009) "
            "AND `active` = TRUE",
        )

    # --- arrow / polars / pandas array compaction ---

    def test_arrow_array_compacts(self):
        try:
            import pyarrow as pa
        except ImportError:
            self.skipTest("pyarrow not installed")
        arr = pa.array(list(range(1, 5001)))
        self.assertEqual(str(Expr("id").in_(arr)), "`id` BETWEEN 1 AND 5000")

    def test_polars_series_compacts(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")
        s = pl.Series(list(range(1, 5001)))
        self.assertEqual(str(Expr("id").in_(s)), "`id` BETWEEN 1 AND 5000")


# ---------------------------------------------------------------------------
# Table alias prefix
# ---------------------------------------------------------------------------

class TestTableAlias(unittest.TestCase):
    """Bare columns can be prefixed with a table alias (e.g. ``t.price``)."""

    # --- on Expr ---

    def test_expr_with_table_alias(self):
        self.assertEqual(
            str(Expr("price").with_table_alias("t").eq(70)),
            "`t`.`price` = 70",
        )

    def test_expr_alias_kwarg(self):
        # The shorter `Expr("col", alias="t")` form is equivalent.
        self.assertEqual(
            str(Expr("price", alias="t").eq(70)),
            "`t`.`price` = 70",
        )

    def test_alias_applies_across_all_ops(self):
        h = Expr("c", alias="t")
        self.assertEqual(str(h.eq(1)), "`t`.`c` = 1")
        self.assertEqual(str(h.ne(1)), "`t`.`c` != 1")
        self.assertEqual(str(h.lt(1)), "`t`.`c` < 1")
        self.assertEqual(str(h.in_([1, 2])), "`t`.`c` IN (1, 2)")
        self.assertEqual(str(h.between(1, 5)), "`t`.`c` BETWEEN 1 AND 5")
        self.assertEqual(str(h.is_null()), "`t`.`c` IS NULL")
        self.assertEqual(str(h.like("X%")), "`t`.`c` LIKE 'X%'")

    # --- on Predicate ---

    def test_predicate_with_table_alias_simple(self):
        p = Expr("price").eq(70)
        self.assertEqual(str(p.with_table_alias("t")), "`t`.`price` = 70")

    def test_predicate_with_table_alias_propagates_to_subtree(self):
        # All leaves in the tree get the alias applied.
        p = Expr("price").ge(70).and_(Expr("region").in_(["EU", "US"]))
        self.assertEqual(
            str(p.with_table_alias("t")),
            "`t`.`price` >= 70 AND `t`.`region` IN ('EU', 'US')",
        )

    def test_predicate_with_table_alias_through_or_not(self):
        p = Expr("a").eq(1).or_(Expr("b").eq(2)).not_()
        self.assertEqual(
            str(p.with_table_alias("t")),
            "NOT (`t`.`a` = 1 OR `t`.`b` = 2)",
        )

    def test_predicate_alias_clear_with_none(self):
        p = Expr("price", alias="t").eq(70)
        self.assertEqual(str(p), "`t`.`price` = 70")
        cleared = p.with_table_alias(None)
        self.assertEqual(str(cleared), "`price` = 70")

    def test_predicate_alias_override_last_wins(self):
        # Successive calls override; the last call propagates fully through the tree.
        p = Expr("a").eq(1).and_(Expr("b").eq(2)).with_table_alias("x")
        self.assertEqual(str(p), "`x`.`a` = 1 AND `x`.`b` = 2")
        p2 = p.with_table_alias("y")
        self.assertEqual(str(p2), "`y`.`a` = 1 AND `y`.`b` = 2")

    # --- already-qualified columns ---

    def test_alias_skipped_for_dotted_column(self):
        # Caller already qualified — don't double-prefix.
        self.assertEqual(
            str(Expr("schema.table.col", alias="t").eq(1)),
            "`schema`.`table`.`col` = 1",
        )

    def test_alias_mixes_qualified_and_bare(self):
        # In a tree with mixed bare and qualified columns, only bare ones get prefixed.
        p = Expr("price").eq(70).and_(Expr("ref.name").eq("X"))
        self.assertEqual(
            str(p.with_table_alias("t")),
            "`t`.`price` = 70 AND `ref`.`name` = 'X'",
        )

    # --- flavor interaction ---

    def test_alias_with_postgres_flavor(self):
        self.assertEqual(
            str(Expr("price", flavor="postgres", alias="t").eq(70)),
            '"t"."price" = 70',
        )

    def test_alias_with_tsql_brackets(self):
        # T-SQL [bracket] quoting applies to the alias too.
        self.assertEqual(
            str(Expr("p", flavor="tsql", alias="t").eq(5)),
            "[t].[p] = 5",
        )

    def test_alias_with_special_chars_escapes(self):
        # An alias containing the quote char gets escaped via the flavor's rules.
        self.assertEqual(
            str(Expr("c", alias="we`ird").eq(1)),
            "`we``ird`.`c` = 1",
        )

    def test_with_flavor_preserves_alias(self):
        p = Expr("price", alias="t").eq(70)
        self.assertEqual(str(p.with_flavor("postgres")), '"t"."price" = 70')

    def test_with_table_alias_preserves_flavor(self):
        p = Expr("price", flavor="postgres").eq(70)
        self.assertEqual(str(p.with_table_alias("t")), '"t"."price" = 70')

    # --- parameterized rendering ---

    def test_alias_in_param_render(self):
        p = Expr("price", alias="t").ge(70).and_(Expr("region", alias="t").in_(["EU", "US"]))
        sql, params = p.to_param()
        self.assertEqual(sql, "`t`.`price` >= ? AND `t`.`region` IN (?, ?)")
        self.assertEqual(params, [70, "EU", "US"])

    def test_alias_with_compaction(self):
        # Alias applies to every leaf produced by compaction, including
        # the multi-run OR/AND expansions.
        vals = list(range(1, 1500)) + [9999]
        p = Expr("id", alias="t").in_(vals)
        self.assertEqual(
            str(p),
            "`t`.`id` BETWEEN 1 AND 1499 OR `t`.`id` = 9999",
        )

    def test_alias_with_null_aware_in(self):
        p = Expr("c", alias="t").in_([1, 2, None])
        self.assertEqual(str(p), "`t`.`c` IN (1, 2) OR `t`.`c` IS NULL")

    def test_alias_with_ilike_fallback(self):
        # MySQL has no ILIKE -> LOWER(...) LIKE LOWER(...). Alias should apply
        # inside the LOWER call.
        self.assertEqual(
            str(Expr("n", flavor="mysql", alias="t").ilike("Foo%")),
            "LOWER(`t`.`n`) LIKE LOWER('Foo%')",
        )

    # --- validation ---

    def test_empty_alias_raises(self):
        with self.assertRaises(ValueError):
            Expr("c").with_table_alias("")
        with self.assertRaises(ValueError):
            Expr("c").with_table_alias("   ")

    def test_alias_kwarg_validates(self):
        with self.assertRaises(ValueError):
            Expr("c", alias="")

    def test_predicate_empty_alias_raises(self):
        with self.assertRaises(ValueError):
            Expr("c").eq(1).with_table_alias("")

    # --- realistic JOIN-style use ---

    def test_join_style_two_aliased_predicates_and(self):
        # Typical JOIN scenario: `WHERE t.price >= 70 AND r.country = 'FR'`.
        trades = Expr("price", alias="t").ge(70)
        ref = Expr("country", alias="r").eq("FR")
        sql, params = trades.and_(ref).to_param()
        self.assertEqual(sql, "`t`.`price` >= ? AND `r`.`country` = ?")
        self.assertEqual(params, [70, "FR"])


class TestRealistic(unittest.TestCase):
    def test_commodity_trade_query(self):
        p = (
            Expr("book").eq("PROP_OIL")
            .and_(Expr("trade_dt").between(date(2026, 1, 1), date(2026, 4, 28)))
            .and_(
                Expr("commodity").in_(["WTI", "Brent", None])  # NULL-aware -> OR
                .or_(Expr("commodity").like("Crude%"))
            )
            .and_(Expr("settled_at").is_not_null())
        )
        sql, params = p.to_param(style="numeric")
        # The inner OR (from NULL expansion) flattens with the user's outer .or_().
        self.assertEqual(
            sql,
            "`book` = $1 "
            "AND `trade_dt` BETWEEN $2 AND $3 "
            "AND (`commodity` IN ($4, $5) OR `commodity` IS NULL "
            "OR `commodity` LIKE $6) "
            "AND `settled_at` IS NOT NULL",
        )
        self.assertEqual(
            params,
            ["PROP_OIL", date(2026, 1, 1), date(2026, 4, 28), "WTI", "Brent", "Crude%"],
        )

    def test_postgres_render_of_same_query(self):
        p = (
            Expr("book").eq("PROP_OIL")
            .and_(Expr("trade_dt").ge(date(2026, 1, 1)))
        ).with_flavor("postgres")
        self.assertEqual(
            str(p),
            "\"book\" = 'PROP_OIL' AND \"trade_dt\" >= DATE '2026-01-01'",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)