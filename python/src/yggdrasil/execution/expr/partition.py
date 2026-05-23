"""Partition-pruning extractor — over-approximate per-column accepted sets.

Engines that partition by a finite key set (Delta, Iceberg, Hive-style
folder layouts) want a quick "which files can I skip" answer before
any parquet open. The full :func:`Expression.to_python` / ``to_arrow``
evaluator filters *rows*; this extractor walks the predicate once and
returns the set of partition-column values that *could* satisfy it.
The returned dict is consumed by :meth:`Snapshot.prune_files` (and any
other partition-aware reader) — the row-level predicate still runs on
the surviving files, so the extractor is allowed to over-approximate.
"""

from __future__ import annotations

from typing import Any, Iterable

from .nodes import (
    Column,
    Comparison,
    Expression,
    InList,
    IsNull,
    Literal,
    Logical,
)
from .operators import CompareOp, LogicalOp
from .simplify import simplify


__all__ = ["extract_partition_filters"]


def extract_partition_filters(
    expr: Expression,
    columns: "Iterable[str]",
) -> "dict[str, frozenset]":
    """Over-approximate per-column accepted-value sets from a predicate.

    Walks *expr* (after :func:`simplify`) and returns, for each
    column in *columns* that the predicate constrains to a finite
    set, the :class:`frozenset` of values the column *could* take
    in any row the predicate accepts. Columns not in the returned
    dict are unconstrained — the predicate doesn't restrict their
    value to a finite, enumerable set.

    The result is suitable for partition pruning: a file whose
    partition value for ``col`` isn't in the extracted set can be
    skipped. It is *over-approximate* — a file the constraints
    accept may still produce zero matching rows (the row-level
    filter catches the residual), but no row the predicate accepts
    can fall outside the constraints. That makes the extractor
    safe to use as a pre-filter before the row-level scan.

    Supported shapes:

    - ``col == v``: ``{col: {v}}``.
    - ``col.is_in([v1, v2])``: ``{col: {v1, v2}}``.
      ``includes_null=True`` adds ``None`` to the set.
    - ``col.is_null()``: ``{col: {None}}``.
    - ``AND``: per-column intersection of constraints. Columns
      constrained on only one side keep their original set.
    - ``OR``: per-column union, but only for columns constrained
      on *every* operand (one unconstrained operand drops the
      column — the OR could accept any value via that branch).

    Returns ``{}`` for ``NOT``, ranges (``<`` / ``<=`` / ``>`` /
    ``>=`` / ``BETWEEN``), ``LIKE``, ``!=``, arithmetic on column
    references, column-vs-column comparisons, and ``col == NULL``
    (always UNKNOWN in SQL — never accepts a row).

    A returned ``{col: frozenset()}`` means the predicate is
    unsatisfiable on that column — the caller can skip every file
    whose partition value for ``col`` exists.
    """
    allowed = frozenset(columns)
    if not allowed:
        return {}
    return _extract_partition(simplify(expr), allowed)


def _extract_partition(
    expr: Expression,
    allowed: "frozenset[str]",
) -> "dict[str, frozenset]":
    if isinstance(expr, Logical):
        return _extract_logical(expr, allowed)
    if isinstance(expr, Comparison) and expr.op is CompareOp.EQ:
        col, val = _eq_col_and_literal(expr)
        if col is None or col not in allowed:
            return {}
        return {col: frozenset((val,))}
    if isinstance(expr, InList) and not expr.negated and isinstance(expr.target, Column):
        col_name = expr.target.name
        if col_name not in allowed:
            return {}
        if expr.includes_null:
            return {col_name: frozenset(expr.values) | frozenset((None,))}
        return {col_name: frozenset(expr.values)}
    if isinstance(expr, IsNull) and not expr.negated and isinstance(expr.target, Column):
        col_name = expr.target.name
        if col_name not in allowed:
            return {}
        return {col_name: frozenset((None,))}
    # NOT, !=, ranges, LIKE, BETWEEN, arithmetic, col-vs-col EQ,
    # col == NULL (always UNKNOWN) — all fall through to "no constraint".
    return {}


def _extract_logical(
    expr: Logical,
    allowed: "frozenset[str]",
) -> "dict[str, frozenset]":
    parts = [_extract_partition(o, allowed) for o in expr.operands]
    if expr.op is LogicalOp.AND:
        # Intersect per column; union of keys (constraints compose).
        out: "dict[str, frozenset]" = {}
        for d in parts:
            for k, v in d.items():
                if k in out:
                    out[k] = out[k] & v
                else:
                    out[k] = v
        return out
    # OR — per-column union, but only on columns every operand
    # constrained. A single unconstrained branch means the OR
    # could accept any value for that column.
    if not parts:
        return {}
    common = set(parts[0].keys())
    for d in parts[1:]:
        common &= set(d.keys())
    if not common:
        return {}
    out = {}
    for k in common:
        merged: "frozenset" = parts[0][k]
        for d in parts[1:]:
            merged = merged | d[k]
        out[k] = merged
    return out


def _eq_col_and_literal(
    comp: Comparison,
) -> "tuple[str | None, Any]":
    """Return ``(column_name, literal_value)`` for ``col == lit`` or
    ``lit == col``, else ``(None, None)``.

    Drops the ``col == NULL`` shape — SQL evaluates it as UNKNOWN
    for every row, so any value-set we built from it would be a
    lie. The caller's row-level filter still rejects those rows.
    """
    left, right = comp.left, comp.right
    if isinstance(left, Column) and isinstance(right, Literal):
        if right.value is None:
            return None, None
        return left.name, right.value
    if isinstance(right, Column) and isinstance(left, Literal):
        if left.value is None:
            return None, None
        return right.name, left.value
    return None, None
