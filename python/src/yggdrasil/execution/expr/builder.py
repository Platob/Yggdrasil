"""Fluent factory functions for the expression AST.

Users build trees with :func:`col` plus operator overloads (the
:class:`Expression` base carries them) and the named methods on
:class:`Expression` for the rare cases where the operator is
spelled with a Python keyword (``is_in``, ``between``, …).

Example
-------

::

    from yggdrasil.execution.expr import col, lit

    p = (col("price") >= 100) & col("side").is_in(["buy", "sell"])
    p.to_sql()       # SQL string
    p.to_python()    # Callable[[Mapping], bool]
    p.to_pyarrow()   # pa.compute.Expression

The factory accepts an optional :class:`Field` so backends that
need typed literals (Spark casts, Arrow scalars) have everything
they need without an extra dtype argument at call sites.

Projections (rename + cast-on-select) live on
:class:`yggdrasil.data.data_field.Field` directly — there's no
separate selector node. Build a Field with the desired output
:attr:`name`, optional :attr:`alias` for the source-side label,
and target :attr:`dtype`, then pass it to ``LazyTabular.select``
or the SQL executor's ``statement.select`` list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .nodes import (
    Column,
    Expression,
    Logical,
    LogicalOp,
    Not,
)

if TYPE_CHECKING:
    from yggdrasil.data.data_field import Field


__all__ = ["col", "neg", "all_of", "any_of"]


def col(
    name: str,
    *,
    field: "Field | None" = None,
    alias: "str | None" = None,
) -> Column:
    """Build a :class:`Column` reference for ``name``.

    ``field`` binds typed metadata so backends can pick the right
    literal cast / engine type without a separate dtype argument.
    ``alias`` qualifies the column for SQL emitters that want
    ``T.col``; the AST node still compares ``equals`` across alias
    differences via the underlying name.
    """
    return Column(name=name, field=field, alias=alias)


def neg(expr: Expression) -> Not:
    """Return ``NOT expr``. Same as ``~expr`` but explicit."""
    return Not(expr)


def all_of(*operands: Any) -> Logical:
    """Conjunction over an arbitrary number of operands.

    Equivalent to chaining ``a & b & c`` but produces a flat
    :class:`Logical` instead of a left-leaning tree — useful when
    callers know the conjunction is commutative and want the
    cheaper round-trip.
    """
    if not operands:
        raise ValueError("all_of needs at least one operand.")
    return Logical(LogicalOp.AND, tuple(_must_be_expression(o) for o in operands))


def any_of(*operands: Any) -> Logical:
    """Disjunction over an arbitrary number of operands."""
    if not operands:
        raise ValueError("any_of needs at least one operand.")
    return Logical(LogicalOp.OR, tuple(_must_be_expression(o) for o in operands))


def _must_be_expression(value: Any) -> Expression:
    if not isinstance(value, Expression):
        raise TypeError(
            f"Expected Expression in all_of/any_of, got "
            f"{type(value).__name__}. Wrap plain values in lit() "
            "or build comparisons with col()."
        )
    return value
