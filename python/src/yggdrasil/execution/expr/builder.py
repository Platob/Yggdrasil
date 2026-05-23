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
    name: "str | Field",
    *,
    field: "Field | None" = None,
    alias: "str | None" = None,
    dtype: "Any | None" = None,
) -> Column:
    """Build a :class:`Column` reference.

    ``name`` is the column identifier; pass a pre-built :class:`Field`
    here to reuse the typed metadata (the bound dtype, nullability,
    children, …). ``dtype`` is the convenience knob for the common
    "I know the column type" case — when omitted the field's dtype
    defaults to :class:`ObjectType` (the Any-shaped sentinel) so
    backends fall back to engine-side inference. ``alias`` qualifies
    the column for SQL emitters that want ``T.col``; it's stored on
    the field as ``table_qualifier`` so the AST node's single slot
    still owns the full lookup state.

    ``field=`` is the legacy entry point: when the caller already has
    a :class:`Field` instance, hand it in directly and the builder
    skips the construction. ``field`` and ``dtype`` are mutually
    exclusive — the field's dtype wins if both are supplied.
    """
    from yggdrasil.data.data_field import Field
    from yggdrasil.data.types.primitive import ObjectType

    if isinstance(name, Field):
        # Caller already prepared the field; honour it.
        bound = name
        caller_owned = True
    elif field is not None:
        bound = field if field.name == name else field.with_name(name, inplace=False)
        caller_owned = field is bound
    else:
        bound = Field(name=name, dtype=dtype if dtype is not None else ObjectType())
        caller_owned = False
    if alias:
        # Caller-owned fields might be shared with other Columns — copy
        # before stamping the qualifier so we don't leak the bolt-on
        # into unrelated trees. The fresh-built path can mutate the
        # locally-owned field directly.
        bound = bound.set_table_qualifier(alias, inplace=not caller_owned)
    return Column(field=bound)


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
