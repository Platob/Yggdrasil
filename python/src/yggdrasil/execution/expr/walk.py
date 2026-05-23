"""Pre-order tree walks over the expression AST.

Small visitor primitive every backend reaches for when it needs to
collect schema-side information (referenced column names, free
variables, optimisation flags) without producing a transformed tree.
For rewrites prefer pattern-matching on node types directly.
"""

from __future__ import annotations

from typing import Iterable

from .nodes import (
    Arithmetic,
    Between,
    Cast,
    Column,
    Comparison,
    Expression,
    InList,
    IsNull,
    Like,
    Logical,
    Not,
)


__all__ = ["walk", "free_columns"]


def walk(expr: Expression) -> "Iterable[Expression]":
    """Pre-order walk over every node in *expr*.

    Backends use this for visitors that don't need to produce a
    transformed tree (schema collection, free-variable lookup,
    optimization checks). For tree rewrites prefer pattern-matching
    on node types directly.
    """
    yield expr
    if isinstance(expr, (Comparison, Arithmetic)):
        yield from walk(expr.left)
        yield from walk(expr.right)
    elif isinstance(expr, Logical):
        for op in expr.operands:
            yield from walk(op)
    elif isinstance(expr, Not):
        yield from walk(expr.operand)
    elif isinstance(expr, Between):
        yield from walk(expr.target)
        yield from walk(expr.low)
        yield from walk(expr.high)
    elif isinstance(expr, (InList, IsNull, Like)):
        yield from walk(expr.target)
    elif isinstance(expr, Cast):
        yield from walk(expr.operand)


def free_columns(expr: Expression) -> "tuple[str, ...]":
    """Names of every distinct column referenced by *expr*.

    Order is first-encounter (pre-order walk), de-duplicated. Used
    by the Python backend to build a value-resolution closure and
    by the schema emitter to advertise the predicate's input
    surface.
    """
    seen: "dict[str, None]" = {}
    for node in walk(expr):
        if isinstance(node, Column):
            seen.setdefault(node.name, None)
    return tuple(seen)
