"""Behavioural tests for the Python backend.

Focus is on the *semantics* — three-valued logic, NULL-aware
``IN``, ``LIKE`` translation — rather than re-testing AST shape
(that's covered in ``test_nodes.py``).
"""

from __future__ import annotations

import pytest

from yggdrasil.execution.expr import col
from yggdrasil.execution.expr.backends.python import filter_rows, to_python


class TestComparisonSemantics:
    def test_basic_predicate_filters_rows(self):
        rows = [{"x": 1}, {"x": 2}, {"x": 3}]
        assert list(filter_rows(col("x") > 1, rows)) == [{"x": 2}, {"x": 3}]

    def test_eq_and_ne_with_strings(self):
        rows = [{"side": "buy"}, {"side": "sell"}]
        assert list(filter_rows(col("side") == "buy", rows)) == [{"side": "buy"}]
        assert list(filter_rows(col("side") != "buy", rows)) == [{"side": "sell"}]


class TestThreeValuedLogic:
    def test_missing_column_evaluates_to_none_in_default_mode(self):
        # ``None`` (UNKNOWN) is rejected by the row filter — same
        # as SQL: ``WHERE x = 1`` excludes rows where ``x IS NULL``.
        compiled = to_python(col("x") == 1)
        assert compiled({"y": 1}) is None

    def test_strict_mode_raises_on_missing_column(self):
        compiled = to_python(col("x") == 1, strict=True)
        with pytest.raises(KeyError):
            compiled({"y": 1})

    def test_and_short_circuits_on_false_even_with_null(self):
        # ``False AND UNKNOWN = False`` (SQL) — short-circuit
        # avoids evaluating the right side.
        compiled = to_python((col("x") == 1) & (col("y") > 0))
        assert compiled({"x": 2}) is False  # left is False, skip right

    def test_or_short_circuits_on_true_even_with_null(self):
        compiled = to_python((col("x") == 1) | (col("y") > 0))
        assert compiled({"x": 1}) is True

    def test_not_propagates_null(self):
        compiled = to_python(~(col("x") == 1))
        assert compiled({}) is None  # NOT UNKNOWN = UNKNOWN


class TestNullAwareIn:
    def test_explicit_null_in_set_matches_null_rows(self):
        compiled = to_python(col("x").is_in([1, 2, None]))
        assert compiled({"x": 1}) is True
        assert compiled({"x": None}) is True
        assert compiled({"x": 5}) is False

    def test_not_in_with_null_excludes_null_rows(self):
        compiled = to_python(col("x").not_in([1, None]))
        assert compiled({"x": 5}) is True
        assert compiled({"x": 1}) is False
        # NOT IN with NULL in the set: a NULL row never satisfies.
        assert compiled({"x": None}) is False


class TestBetween:
    def test_inclusive_bounds(self):
        compiled = to_python(col("d").between(1, 10))
        assert compiled({"d": 1}) is True
        assert compiled({"d": 10}) is True
        assert compiled({"d": 11}) is False

    def test_not_between_flips_membership(self):
        compiled = to_python(col("d").not_between(1, 10))
        assert compiled({"d": 11}) is True
        assert compiled({"d": 5}) is False


class TestLike:
    def test_percent_wildcard_anywhere(self):
        compiled = to_python(col("s").like("%foo%"))
        assert compiled({"s": "abcfoodef"}) is True
        assert compiled({"s": "bar"}) is False

    def test_underscore_wildcard_single_char(self):
        compiled = to_python(col("s").like("a_b"))
        assert compiled({"s": "axb"}) is True
        assert compiled({"s": "ab"}) is False
        assert compiled({"s": "axxb"}) is False

    def test_case_insensitive_when_requested(self):
        compiled = to_python(col("s").like("FOO%", case_insensitive=True))
        assert compiled({"s": "foobar"}) is True


class TestArithmetic:
    def test_add_propagates_through_predicate(self):
        # Arithmetic produces a scalar expression; a comparison on
        # top makes it filterable.
        compiled = to_python((col("x") + 1) >= 5)
        assert compiled({"x": 4}) is True
        assert compiled({"x": 3}) is False

    def test_div_with_null_yields_null(self):
        compiled = to_python((col("x") / col("y")) >= 1)
        assert compiled({"x": 5, "y": None}) is None
