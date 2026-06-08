"""Unit tests for the AST core (``nodes.py`` + ``builder.py``).

These cover construction, identity (immutability + hashability),
operator overloads, and the boolean composition shortcuts.
Backend-specific behaviour lives in ``test_python.py`` /
``test_sql.py`` / etc.
"""

from __future__ import annotations

import pytest

from yggdrasil.saga.expr import (
    Between,
    Column,
    Comparison,
    CompareOp,
    Expression,
    InList,
    IsNull,
    Like,
    Literal,
    Logical,
    LogicalOp,
    Not,
    Predicate,
    col,
    lit,
)


class TestColFactory:
    def test_col_returns_column_with_operator_surface(self):
        c = col("price")
        assert isinstance(c, Column)
        assert isinstance(c, Expression)
        assert c.name == "price"
        assert c.alias is None

    def test_alias_is_carried_through_unchanged(self):
        c = col("price", alias="t")
        assert c.alias == "t"


class TestComparisonOverloads:
    @pytest.mark.parametrize(
        "build, expected_op",
        [
            (lambda c: c == 1, CompareOp.EQ),
            (lambda c: c != 1, CompareOp.NE),
            (lambda c: c < 1, CompareOp.LT),
            (lambda c: c <= 1, CompareOp.LE),
            (lambda c: c > 1, CompareOp.GT),
            (lambda c: c >= 1, CompareOp.GE),
        ],
    )
    def test_each_overload_produces_matching_compareop(self, build, expected_op):
        cmp = build(col("x"))
        assert isinstance(cmp, Comparison)
        assert isinstance(cmp, Predicate)
        assert cmp.op is expected_op
        assert isinstance(cmp.right, Literal)
        assert cmp.right.value == 1

    def test_plain_value_on_rhs_gets_wrapped_as_literal(self):
        cmp = col("x") == "buy"
        assert cmp.right.equals(Literal(value="buy"))


class TestLogicalComposition:
    def test_and_or_invert_overloads(self):
        a = col("price") > 100
        b = col("side") == "buy"
        combined = a & b
        assert isinstance(combined, Logical)
        assert combined.op is LogicalOp.AND
        assert combined.operands[0].equals(a)
        assert combined.operands[1].equals(b)

        ored = a | b
        assert ored.op is LogicalOp.OR

        negated = ~a
        assert isinstance(negated, Not)
        assert negated.operand.equals(a)

    def test_logical_requires_at_least_one_operand(self):
        with pytest.raises(ValueError, match="at least one operand"):
            Logical(LogicalOp.AND, ())


class TestMembership:
    def test_is_in_carries_values_as_tuple(self):
        p = col("side").is_in(["buy", "sell"])
        assert isinstance(p, InList)
        assert p.values == ("buy", "sell")
        assert p.negated is False
        assert p.includes_null is False

    def test_none_in_values_routes_to_includes_null_flag(self):
        # SQL-aware: when ``None`` shows up in the value set, the
        # backend expands to ``... OR col IS NULL`` rather than
        # treating NULL as a comparable value.
        p = col("x").is_in([1, 2, None])
        assert p.values == (1, 2)
        assert p.includes_null is True

    def test_not_in_flips_negation(self):
        p = col("x").not_in([1, 2])
        assert p.negated is True


class TestBetweenLikeNull:
    def test_between_inclusive_with_optional_negation(self):
        p = col("d").between(1, 10)
        assert isinstance(p, Between)
        assert p.negated is False
        np = col("d").not_between(1, 10)
        assert np.negated is True

    def test_like_pattern_and_case_flag(self):
        p = col("s").like("%foo%", case_insensitive=True)
        assert isinstance(p, Like)
        assert p.pattern == "%foo%"
        assert p.case_insensitive is True

    def test_is_null_and_is_not_null(self):
        assert col("x").is_null().equals(IsNull(target=col("x"), negated=False))
        assert col("x").is_not_null().equals(IsNull(target=col("x"), negated=True))


class TestImmutability:
    def test_nodes_are_structurally_hashable(self):
        # ``__eq__`` on Expression returns a Comparison node (so
        # ``col("x") == 5`` builds an AST), but ``equals()`` does
        # the structural comparison and ``__hash__`` is structural
        # — so equal trees share a bucket and are usable as dict
        # keys / set members.
        a = col("x") > 5
        b = col("x") > 5
        assert a.equals(b)
        assert hash(a) == hash(b)
        assert {a, b} == {a}  # de-duplication via structural hash

    def test_nodes_are_immutable_by_convention(self):
        # Node classes are plain ``__slots__`` types (no frozen-
        # dataclass overhead) so they're immutable by convention,
        # not enforced. Combinators allocate fresh nodes rather
        # than mutate operands; tests / callers that need to assert
        # "predicates round-trip without surprise rewrites" should
        # check ``a.equals(rebuilt)`` rather than identity.
        c = col("x")
        assert isinstance(c, Column)
        assert c.name == "x"


class TestLitFactory:
    def test_lit_wraps_plain_value(self):
        assert lit(5).equals(Literal(value=5))
        assert lit(None).equals(Literal(value=None))
