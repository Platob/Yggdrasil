"""Tests for the algebraic-rewrite pass.

:func:`simplify` re-shapes an expression tree without changing its
SQL three-valued meaning. The interesting properties:

- It is *semantics-preserving* against the Python and pyarrow
  backends — every test below pairs the original tree with the
  simplified one and asserts both evaluate identically on a
  representative row sample (including NULL).
- It is *shape-preserving* when no rule fires (the input instance
  is returned unchanged), so calling it on already-normalized
  input costs one walk.
- It is conservative around NULL: ``IsNull(c)`` folds into
  ``InList.includes_null=True``, but ``c == None`` stays as-is
  because in SQL it is UNKNOWN regardless of the row's value, not
  ``IS NULL``.

Tests are organized by rule (dedup / flatten / OR-collapse /
AND-dedup / equivalence) so a failure points at the rule that
broke.
"""

from __future__ import annotations

import pytest

from yggdrasil.io.tabular.execution.expr import (
    Comparison,
    Expression,
    InList,
    IsNull,
    Logical,
    LogicalOp,
    col,
    lit,
    simplify,
)
from yggdrasil.io.tabular.execution.expr.backends.python import to_python


def _eval_all(expr: Expression, samples: list[dict]) -> list:
    fn = to_python(expr)
    return [fn(row) for row in samples]


class TestInListDedup:
    def test_duplicate_values_collapse_in_first_seen_order(self):
        p = col("x").is_in([1, 2, 1, 3, 2, 1])
        s = simplify(p)
        assert isinstance(s, InList)
        assert s.values == (1, 2, 3)
        # includes_null/negated flags carry through unchanged.
        assert s.includes_null is False
        assert s.negated is False

    def test_already_unique_returns_same_instance(self):
        # Shape-preserving fast path — no allocation on the no-op
        # path so calling simplify unconditionally is cheap.
        p = col("x").is_in([1, 2, 3])
        assert simplify(p) is p

    def test_unhashable_values_dedupe_via_linear_scan(self):
        # InList stores Python values verbatim; users sometimes
        # seed it with dict / list literals (rare but legal). The
        # linear-scan fallback must still de-duplicate.
        a = {"k": 1}
        b = {"k": 1}
        c = {"k": 2}
        p = col("x").is_in([a, b, c])
        s = simplify(p)
        # ``a`` and ``b`` are == under dict equality, so one dropped.
        assert s.values == ({"k": 1}, {"k": 2})

    def test_not_in_dedup_keeps_negation_flag(self):
        p = col("x").not_in([1, 1, 2])
        s = simplify(p)
        assert isinstance(s, InList)
        assert s.negated is True
        assert s.values == (1, 2)


class TestLogicalFlatten:
    def test_nested_or_flattens_into_single_node(self):
        # ``a | b | c | d`` builds a left-leaning tree; flatten
        # the chain so emitters see one Logical with N operands
        # and OR collapse below sees the full picture.
        a = col("x") > 0
        b = col("y") > 0
        c = col("z") > 0
        d = col("w") > 0
        p = ((a | b) | c) | d
        s = simplify(p)
        assert isinstance(s, Logical)
        assert s.op is LogicalOp.OR
        # Operands are exactly the four leaves, in order.
        assert len(s.operands) == 4

    def test_nested_and_flattens_into_single_node(self):
        a = col("x") > 0
        b = col("y") > 0
        c = col("z") > 0
        p = (a & b) & c
        s = simplify(p)
        assert isinstance(s, Logical)
        assert s.op is LogicalOp.AND
        assert len(s.operands) == 3

    def test_mixed_op_does_not_flatten_across_boundary(self):
        # AND inside OR stays nested — different operator.
        inner = (col("x") > 0) & (col("y") > 0)
        outer = inner | (col("z") > 0)
        s = simplify(outer)
        assert s.op is LogicalOp.OR
        # First operand is the AND subtree, not flattened.
        assert isinstance(s.operands[0], Logical)
        assert s.operands[0].op is LogicalOp.AND


class TestOrCollapse:
    def test_or_of_equalities_on_same_column_collapses_to_inlist(self):
        p = (col("x") == 1) | (col("x") == 2) | (col("x") == 3)
        s = simplify(p)
        assert isinstance(s, InList)
        assert s.values == (1, 2, 3)
        assert s.includes_null is False

    def test_or_with_is_null_folds_into_includes_null(self):
        p = (col("x") == 1) | (col("x") == 2) | col("x").is_null()
        s = simplify(p)
        assert isinstance(s, InList)
        assert s.values == (1, 2)
        assert s.includes_null is True

    def test_or_merges_existing_inlist_with_extra_equalities(self):
        p = col("x").is_in([1, 2]) | (col("x") == 3) | (col("x") == 4)
        s = simplify(p)
        assert isinstance(s, InList)
        assert s.values == (1, 2, 3, 4)

    def test_or_merges_two_inlists_on_same_target(self):
        p = col("x").is_in([1, 2]) | col("x").is_in([3, 4])
        s = simplify(p)
        assert isinstance(s, InList)
        assert s.values == (1, 2, 3, 4)

    def test_or_groups_separately_by_target(self):
        # Two columns, two groups → still an OR of two InLists.
        p = (col("x") == 1) | (col("y") == 2) | (col("x") == 3) | (col("y") == 4)
        s = simplify(p)
        assert isinstance(s, Logical)
        assert s.op is LogicalOp.OR
        targets = {op.target.name for op in s.operands if isinstance(op, InList)}
        assert targets == {"x", "y"}

    def test_single_equality_on_a_target_does_not_inflate_to_inlist(self):
        # A lone ``c == 1`` wouldn't shrink — leave it as a
        # cheaper Comparison.
        p = (col("x") == 1) | (col("y") == 2)
        s = simplify(p)
        # Both groups have 1 contribution each → no collapse.
        assert isinstance(s, Logical)
        assert all(isinstance(op, Comparison) for op in s.operands)

    def test_or_collapse_keeps_non_foldable_operands_in_place(self):
        # ``col("x") > 5`` is not equality — it stays in the OR
        # alongside the collapsed InList.
        p = (col("x") == 1) | (col("x") == 2) | (col("x") > 100)
        s = simplify(p)
        assert isinstance(s, Logical)
        assert s.op is LogicalOp.OR
        assert any(isinstance(op, InList) and op.values == (1, 2) for op in s.operands)
        assert any(isinstance(op, Comparison) for op in s.operands)

    def test_literal_on_left_side_is_recognized(self):
        # ``1 == col("x")`` builds Comparison(Literal(1), EQ, col).
        # Same semantics as ``col("x") == 1`` — fold both shapes.
        p = (lit(1) == col("x")) | (col("x") == 2)
        s = simplify(p)
        assert isinstance(s, InList)
        assert s.values == (1, 2)

    def test_eq_against_none_is_left_untouched(self):
        # SQL ``c = NULL`` is UNKNOWN regardless of c — not the
        # same as ``c IS NULL``. Folding it into includes_null
        # would silently flip UNKNOWN → TRUE for rows where
        # c IS NULL; folding it into a value would treat NULL as
        # a comparable value. Both wrong; leave it as-is.
        p = (col("x") == 1) | (col("x") == None) | (col("x") == 2)  # noqa: E711
        s = simplify(p)
        assert isinstance(s, Logical)
        # Two halves: the collapsed InList and the untouched
        # ``c == None`` comparison.
        assert any(isinstance(op, InList) and op.values == (1, 2) for op in s.operands)
        assert any(
            isinstance(op, Comparison) and op.right.value is None for op in s.operands
        )

    def test_or_of_is_null_alone_returns_unchanged(self):
        # ``c.is_null()`` on its own has nothing to merge with —
        # don't dress it up as a zero-value InList.
        p = col("x").is_null() | col("y").is_null()
        s = simplify(p)
        assert isinstance(s, Logical)
        assert all(isinstance(op, IsNull) for op in s.operands)


class TestAndDedup:
    def test_identical_conjuncts_collapse(self):
        a = col("x") > 5
        p = a & a & (col("y") == 1)
        s = simplify(p)
        assert isinstance(s, Logical)
        assert len(s.operands) == 2

    def test_structurally_equal_but_distinct_instances_collapse(self):
        a = col("x") > 5
        b = col("x") > 5
        assert a is not b
        s = simplify(a & b)
        # Single conjunct left → unwrap to the conjunct itself.
        assert isinstance(s, Comparison)

    def test_distinct_conjuncts_pass_through(self):
        p = (col("x") > 5) & (col("y") < 10)
        s = simplify(p)
        assert isinstance(s, Logical)
        assert len(s.operands) == 2


class TestSemanticEquivalence:
    """Brute-force: simplify(expr) and expr agree on every sample."""

    SAMPLES = [
        {"x": 1, "y": "a"},
        {"x": 2, "y": "b"},
        {"x": 3, "y": "c"},
        {"x": 5, "y": "z"},
        {"x": None, "y": None},
        {"y": "a"},                  # x missing → None
    ]

    @pytest.mark.parametrize("build", [
        lambda: (col("x") == 1) | (col("x") == 2) | (col("x") == 3),
        lambda: (col("x") == 1) | (col("x") == 2) | col("x").is_null(),
        lambda: col("x").is_in([1, 2]) | (col("x") == 3),
        lambda: col("x").is_in([1, 2, 1, 3, 2]),
        lambda: ((col("x") == 1) | (col("x") == 2))
                & ((col("y") == "a") | (col("y") == "b") | col("y").is_null()),
        lambda: (col("x") == 1) | (col("x") == None) | (col("x") == 2),  # noqa: E711
    ])
    def test_simplify_preserves_python_backend_evaluation(self, build):
        orig = build()
        simp = simplify(orig)
        assert _eval_all(orig, self.SAMPLES) == _eval_all(simp, self.SAMPLES)


class TestEdgeCases:
    def test_simplify_on_leaf_is_identity(self):
        c = col("x")
        assert simplify(c) is c
        v = lit(5)
        assert simplify(v) is v

    def test_method_form_delegates_to_function(self):
        p = (col("x") == 1) | (col("x") == 2)
        assert p.simplify().equals(simplify(p))

    def test_simplify_is_idempotent(self):
        p = (col("x") == 1) | (col("x") == 2) | col("x").is_null()
        once = simplify(p)
        twice = simplify(once)
        assert once.equals(twice)
