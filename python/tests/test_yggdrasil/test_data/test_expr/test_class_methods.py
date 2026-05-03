"""Tests for the Expression class-level surface.

Covers the ``Expression.from_*`` classmethods (auto-detect plus
per-engine), :meth:`Expression.to_engine` dispatch,
:meth:`Expression.to_arrow` / :meth:`Expression.to_spark`
aliases, and :meth:`Expression.merge_with` semantics.
"""

from __future__ import annotations

import pytest

from yggdrasil.data.expr import Expression, Predicate, col, lit


class TestEngineDispatch:
    def test_to_engine_python_returns_callable(self):
        compiled = (col("x") > 1).to_engine("python")
        assert callable(compiled)
        assert compiled({"x": 2}) is True

    def test_to_engine_sql_renders_string(self):
        sql = (col("x") > 1).to_engine("sql")
        assert sql == "`x` > 1"

    def test_to_engine_arrow_returns_compute_expression(self):
        import pyarrow.compute as pc

        out = (col("x") > 1).to_engine("arrow")
        # pyarrow's expression class is private; assert duck-typed.
        assert hasattr(out, "equals") or isinstance(out, pc.Expression)

    def test_to_engine_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            (col("x") > 1).to_engine("oracle")

    def test_to_engine_passes_kwargs_through(self):
        # ``flavor`` is a SQL-specific kwarg — to_engine should
        # forward it without complaint.
        sql = (col("x") > 1).to_engine("sql", flavor="postgres")
        assert sql == '"x" > 1'


class TestAliases:
    def test_to_arrow_and_to_pyarrow_are_equivalent(self):
        p = col("x") > 1
        a = p.to_arrow()
        b = p.to_pyarrow()
        # Same type / equivalent: pyarrow expressions compare via
        # ``equals``.
        assert a.equals(b)

    def test_to_spark_alias_present(self):
        # We don't have pyspark in this env — assert the method
        # exists and points at to_pyspark (no call).
        assert getattr(Expression, "to_spark", None) is getattr(
            Expression, "to_pyspark"
        )


class TestFromDispatch:
    def test_from_passthrough_on_existing_expression(self):
        p = col("x") > 1
        assert Expression.from_(p) is p

    def test_from_sql_string(self):
        e = Expression.from_("a > 5")
        assert isinstance(e, Predicate)
        # Round-trip: ``a`` becomes a Column, ``5`` a Literal — the
        # comparison structure is preserved.
        assert e.to_sql() == "`a` > 5"

    def test_from_unknown_raises(self):
        with pytest.raises(TypeError, match="does not know how to lift"):
            Expression.from_(object())

    def test_from_pyarrow_alias(self):
        # Both names point at the same lifter.
        assert Expression.from_pyarrow == Expression.from_arrow


class TestMergeWith:
    def test_two_predicates_and_combine(self):
        a = col("x") > 1
        b = col("y") < 10
        merged = a.merge_with(b)
        assert isinstance(merged, Predicate)
        # Compiled SQL should contain both clauses joined by AND.
        assert merged.to_sql() == "`x` > 1 AND `y` < 10"

    def test_identical_scalars_collapse_to_self(self):
        a = lit(5)
        b = lit(5)
        merged = a.merge_with(b)
        assert merged.equals(a)

    def test_mismatched_scalars_raise(self):
        with pytest.raises(TypeError, match="merge_with combines"):
            lit(5).merge_with(lit(7))


class TestSqlFlavorParameter:
    def test_flavor_keyword_works(self):
        sql = (col("x") > 1).to_sql(flavor="postgres")
        assert sql == '"x" > 1'

    def test_dialect_kwarg_alias_for_back_compat(self):
        sql = (col("x") > 1).to_sql(dialect="postgres")
        assert sql == '"x" > 1'

    def test_flavor_takes_priority_when_both_passed(self):
        sql = (col("x") > 1).to_sql(flavor="postgres", dialect="ansi")
        assert sql == '"x" > 1'
