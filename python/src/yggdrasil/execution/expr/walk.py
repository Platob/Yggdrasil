"""Pre-order tree walks over the expression AST.

Small visitor primitive every backend reaches for when it needs to
collect schema-side information (referenced column names, free
variables, optimisation flags) without producing a transformed tree.
For rewrites prefer pattern-matching on node types directly.
"""

from __future__ import annotations

from typing import Iterable

from .nodes import (
    Alias,
    Arithmetic,
    Between,
    CaseWhen,
    Cast,
    Column,
    Comparison,
    Expression,
    FunctionCall,
    InList,
    IsNull,
    Lambda,
    Like,
    Logical,
    Not,
    SortOrder,
    Subscript,
    WindowFunction,
    WindowSpec,
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
    elif isinstance(expr, FunctionCall):
        for arg in expr.args:
            yield from walk(arg)
    elif isinstance(expr, Alias):
        yield from walk(expr.expr)
    elif isinstance(expr, SortOrder):
        yield from walk(expr.expr)
    elif isinstance(expr, WindowSpec):
        for p in expr.partition_by:
            yield from walk(p)
        for o in expr.order_by:
            yield from walk(o)
    elif isinstance(expr, WindowFunction):
        yield from walk(expr.function)
        yield from walk(expr.window)
    elif isinstance(expr, CaseWhen):
        if expr.operand is not None:
            yield from walk(expr.operand)
        for cond, val in expr.branches:
            yield from walk(cond)
            yield from walk(val)
        if expr.else_expr is not None:
            yield from walk(expr.else_expr)
    elif isinstance(expr, Subscript):
        yield from walk(expr.expr)
        yield from walk(expr.index)
    elif isinstance(expr, Lambda):
        yield from walk(expr.body)


def free_columns(expr: Expression) -> "tuple[str, ...]":
    """Names of every distinct column referenced by *expr*.

    Order is first-encounter (pre-order walk), de-duplicated. Used
    by the Python backend to build a value-resolution closure and
    by the schema emitter to advertise the predicate's input
    surface. Lambda parameters are bound locally and excluded.
    """
    seen: "dict[str, None]" = {}
    bound: set[str] = set()
    for node in walk(expr):
        if isinstance(node, Lambda):
            bound.update(node.params)
        elif isinstance(node, Column) and node.name not in bound:
            seen.setdefault(node.name, None)
    return tuple(seen)
