"""Builder factory tests — ``col`` / ``lit`` / ``neg`` / ``all_of`` / ``any_of``."""

from __future__ import annotations

import pytest

from yggdrasil.execution.expr import all_of, any_of, col, lit, neg
from yggdrasil.execution.expr.nodes import Column, Literal, Logical, Not
from yggdrasil.execution.expr.operators import LogicalOp


def test_col_defaults_to_unbound_field():
    c = col("price")
    assert isinstance(c, Column)
    assert c.name == "price"
    assert c.field is None
    assert c.alias is None
    assert c.qualifier is None


def test_col_with_dtype_synthesises_field():
    from yggdrasil.data.types.primitive import Int64Type

    c = col("price", dtype=Int64Type())
    assert c.field is not None
    assert c.dtype == Int64Type()


def test_col_accepts_prebuilt_field():
    from yggdrasil.data.data_field import Field
    from yggdrasil.data.types.primitive import StringType

    f = Field(name="side", dtype=StringType())
    by_name = col("renamed", field=f)
    assert by_name.name == "renamed"
    assert by_name.field is f

    by_field = col(f)
    assert by_field.name == "side"
    assert by_field.field is f


def test_col_alias_and_qualifier():
    c = col("price", alias="p", qualifier="t")
    assert c.alias == "p"
    assert c.qualifier == "t"


def test_lit_builds_literal():
    node = lit(5)
    assert isinstance(node, Literal)
    assert node.value == 5
    assert node.dtype is None


def test_neg_is_not_node():
    node = neg(col("x") == 1)
    assert isinstance(node, Not)


def test_all_of_builds_flat_and():
    node = all_of(col("a") > 1, col("b") > 2, col("c") > 3)
    assert isinstance(node, Logical)
    assert node.op is LogicalOp.AND
    assert len(node.operands) == 3


def test_any_of_builds_or():
    node = any_of(col("a") > 1, col("b") > 2)
    assert isinstance(node, Logical)
    assert node.op is LogicalOp.OR


def test_all_of_requires_operands():
    with pytest.raises(ValueError):
        all_of()
    with pytest.raises(ValueError):
        any_of()


def test_all_of_rejects_plain_values():
    with pytest.raises(TypeError):
        all_of(col("a") > 1, True)
