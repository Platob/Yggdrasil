"""Expression AST node tests — construction-time normalisation,
structural equality, the smart ``cast`` factory and ``merge_with``.

The AST promises cheap normalisation at construction (InList dedupe,
Logical same-op flattening, OR-of-EQ collapse) — these tests pin that
contract so backends can rely on canonical shapes.
"""

from __future__ import annotations

import pytest

from yggdrasil.execution.expr import col, lit
from yggdrasil.execution.expr.nodes import (
    Arithmetic,
    Cast,
    Column,
    Comparison,
    InList,
    Literal,
    Logical,
    Not,
    Predicate,
)
from yggdrasil.execution.expr.operators import ArithmeticOp, CompareOp, LogicalOp


# ---------------------------------------------------------------------------
# Operator overloads build nodes, not booleans
# ---------------------------------------------------------------------------


def test_comparison_overloads_build_nodes():
    cases = {
        CompareOp.EQ: col("x") == 1,
        CompareOp.NE: col("x") != 1,
        CompareOp.LT: col("x") < 1,
        CompareOp.LE: col("x") <= 1,
        CompareOp.GT: col("x") > 1,
        CompareOp.GE: col("x") >= 1,
    }
    for op, node in cases.items():
        assert isinstance(node, Comparison)
        assert node.op is op
        assert isinstance(node.right, Literal)
        assert node.right.value == 1


def test_arithmetic_overloads_and_reflected_forms():
    node = col("x") + 1
    assert isinstance(node, Arithmetic)
    assert node.op is ArithmeticOp.ADD

    reflected = 10 - col("x")
    assert isinstance(reflected, Arithmetic)
    assert reflected.op is ArithmeticOp.SUB
    assert isinstance(reflected.left, Literal)
    assert reflected.left.value == 10

    assert (col("x") * 2).op is ArithmeticOp.MUL
    assert (col("x") / 2).op is ArithmeticOp.DIV
    assert (col("x") % 2).op is ArithmeticOp.MOD


def test_plain_values_coerce_to_literals():
    node = col("x") == 5
    assert isinstance(node.right, Literal)
    pred = (col("a") > 1) & (col("b") < 2)
    assert isinstance(pred, Logical)
    assert pred.op is LogicalOp.AND


def test_predicate_marker():
    assert isinstance(col("x") == 1, Predicate)
    assert isinstance(col("x").is_null(), Predicate)
    assert not isinstance(col("x") + 1, Predicate)
    assert not isinstance(col("x"), Predicate)


# ---------------------------------------------------------------------------
# Structural equality / hashing — ``==`` is the comparison overload,
# so ``equals`` carries structural semantics.
# ---------------------------------------------------------------------------


def test_equals_is_structural():
    a = (col("x") == 1) & (col("y") > 2)
    b = (col("x") == 1) & (col("y") > 2)
    assert a is not b
    assert a.equals(b)
    assert hash(a) == hash(b)
    assert not a.equals((col("x") == 1) & (col("y") > 3))
    assert not a.equals(col("x") == 1)


def test_equals_distinguishes_node_types():
    assert not col("x").is_null().equals(col("x").is_not_null())
    assert not InList(col("x"), (1,)).equals(InList(col("x"), (1,), negated=True))


# ---------------------------------------------------------------------------
# Logical — same-op flattening, single-operand unwrap, OR collapse
# ---------------------------------------------------------------------------


def test_logical_flattens_same_op_chains():
    a, b, c = col("a") > 1, col("b") > 2, col("c") > 3
    left_leaning = (a & b) & c
    right_leaning = a & (b & c)
    assert isinstance(left_leaning, Logical)
    assert len(left_leaning.operands) == 3
    assert left_leaning.equals(right_leaning)


def test_logical_does_not_flatten_across_ops():
    a, b, c = col("a") > 1, col("b") > 2, col("c") > 3
    mixed = (a | b) & c
    assert mixed.op is LogicalOp.AND
    assert len(mixed.operands) == 2
    assert isinstance(mixed.operands[0], Logical)
    assert mixed.operands[0].op is LogicalOp.OR


def test_logical_single_operand_unwraps():
    inner = col("a") > 1
    assert Logical(LogicalOp.AND, (inner,)) is inner


def test_logical_empty_operands_raises():
    with pytest.raises(ValueError):
        Logical(LogicalOp.AND, ())


def test_or_of_eq_collapses_to_inlist():
    merged = (col("x") == 1) | (col("x") == 2) | (col("x") == 3)
    assert isinstance(merged, InList)
    assert set(merged.values) == {1, 2, 3}
    assert not merged.negated
    assert not merged.includes_null


def test_or_collapse_includes_is_null():
    merged = (col("x") == 1) | col("x").is_null()
    assert isinstance(merged, InList)
    assert merged.values == (1,)
    assert merged.includes_null


def test_or_with_different_targets_stays_logical():
    kept = (col("x") == 1) | (col("y") == 2)
    assert isinstance(kept, Logical)
    assert kept.op is LogicalOp.OR


def test_or_with_eq_null_does_not_collapse():
    # ``c == None`` is UNKNOWN under SQL 3VL — folding it into the
    # InList would change semantics from "never matches" to
    # "matches NULL rows".
    kept = (col("x") == 1) | (col("x") == None)  # noqa: E711
    assert isinstance(kept, Logical)


# ---------------------------------------------------------------------------
# InList — dedupe + null routing
# ---------------------------------------------------------------------------


def test_inlist_dedupes_preserving_order():
    node = InList(col("x"), (3, 1, 3, 2, 1))
    assert node.values == (3, 1, 2)


def test_is_in_splits_nulls_into_flag():
    node = col("x").is_in([1, None, 2])
    assert isinstance(node, InList)
    assert node.values == (1, 2)
    assert node.includes_null
    assert not node.negated


def test_not_in_sets_negated():
    node = col("x").not_in([1, 2])
    assert node.negated
    assert not node.includes_null


def test_is_in_rejects_column_values():
    with pytest.raises(TypeError):
        col("x").is_in([col("y")])


def test_is_in_unwraps_literal_values():
    node = col("x").is_in([lit(1), 2])
    assert node.values == (1, 2)


# ---------------------------------------------------------------------------
# Named predicate helpers
# ---------------------------------------------------------------------------


def test_named_helpers_shapes():
    between = col("x").between(1, 5)
    assert (between.low.value, between.high.value, between.negated) == (1, 5, False)
    assert col("x").not_between(1, 5).negated

    assert col("x").is_null().negated is False
    assert col("x").is_not_null().negated is True

    like = col("x").like("a%", case_insensitive=True)
    assert (like.pattern, like.case_insensitive, like.negated) == ("a%", True, False)
    assert col("x").not_like("a%").negated

    inverted = ~(col("x") == 1)
    assert isinstance(inverted, Not)


# ---------------------------------------------------------------------------
# Smart cast factory
# ---------------------------------------------------------------------------


def test_cast_wraps_plain_expression():
    from yggdrasil.data.types.primitive import Int64Type

    node = (col("x") + 1).cast(Int64Type())
    assert isinstance(node, Cast)
    assert isinstance(node.operand, Arithmetic)


def test_cast_on_untyped_column_retypes_in_place():
    from yggdrasil.data.types.primitive import Int64Type

    target = Int64Type()
    node = col("x").cast(target)
    # No Cast wrapper — the column's synthesised field absorbs the dtype.
    assert isinstance(node, Column)
    assert node.dtype == target


def test_cast_is_noop_when_dtype_matches():
    from yggdrasil.data.types.primitive import Int64Type

    typed = col("x", dtype=Int64Type())
    assert typed.cast(Int64Type()) is typed


def test_cast_of_cast_collapses():
    from yggdrasil.data.types.primitive import Int64Type, StringType

    base = col("x") + 1
    twice = base.cast(Int64Type()).cast(StringType())
    assert isinstance(twice, Cast)
    assert twice.operand is base
    assert twice.dtype == StringType()


# ---------------------------------------------------------------------------
# merge_with
# ---------------------------------------------------------------------------


def test_merge_with_ands_predicates():
    merged = (col("a") > 1).merge_with(col("b") > 2)
    assert isinstance(merged, Logical)
    assert merged.op is LogicalOp.AND


def test_merge_with_identical_scalars_is_identity():
    expr = col("a") + 1
    assert expr.merge_with(col("a") + 1) is expr


def test_merge_with_mismatched_scalars_raises():
    with pytest.raises(TypeError):
        (col("a") + 1).merge_with(col("b") + 2)


# ---------------------------------------------------------------------------
# Generic lifters / dispatch
# ---------------------------------------------------------------------------


def test_from_returns_expression_unchanged():
    from yggdrasil.execution.expr import Expression

    pred = col("x") == 1
    assert Expression.from_(pred) is pred


def test_from_rejects_unknown_types():
    from yggdrasil.execution.expr import Expression

    with pytest.raises(TypeError):
        Expression.from_(object())


def test_to_engine_rejects_unknown_engine():
    with pytest.raises(ValueError):
        (col("x") == 1).to_engine("duckdb")
