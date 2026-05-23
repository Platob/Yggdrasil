"""Algebraic rewrites — :func:`simplify` and supporting helpers.

Every rewrite is shape-preserving when no rule fires (the input
instance is returned unchanged), so calling :func:`simplify` is safe
to do unconditionally on any tree — the cost on already-normalized
input is one pre-order walk.

Headline rewrites:

- **InList dedup**: duplicate values are removed in first-seen
  order. ``c.is_in([1, 2, 2, 1])`` → ``c.is_in([1, 2])``.
- **Logical flatten**: a ``Logical`` whose direct child is the
  same operator is inlined. ``(a OR b) OR c`` is the natural
  shape produced by ``a | b | c`` — flattening keeps the OR
  collapse seeing the full operand list.
- **OR collapse**: equality comparisons against the same target
  expression, plus same-target ``InList`` and
  ``IsNull(negated=False)`` operands, are merged into a single
  ``InList``. ``c == 1 | c == 2 | c.is_null()`` →
  ``c.is_in([1, 2], includes_null=True)``. Targets are compared
  structurally via :meth:`Expression.equals`.
- **AND dedup**: structurally identical conjuncts collapse
  (``p AND p → p``). The OR side's dedup falls out of the
  InList merge automatically.
- Single-operand ``Logical`` after dedup unwraps to the operand.

SQL three-valued logic is preserved exactly — ``c.is_null()``
folds into ``includes_null=True``, but ``c == None`` is left
untouched because in SQL it is UNKNOWN regardless of the row's
value, *not* equivalent to ``c IS NULL``. Collapsing it would
silently flip UNKNOWN rows from "rejected by WHERE" to
"accepted" in any non-WHERE evaluation context.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .nodes import (
    Arithmetic,
    Between,
    Cast,
    Comparison,
    Expression,
    InList,
    IsNull,
    Like,
    Literal,
    Logical,
    Not,
)
from .operators import CompareOp, LogicalOp


__all__ = ["simplify"]


def simplify(expr: Expression) -> Expression:
    """Return a logically equivalent (under SQL 3VL) normalized form.

    See the module docstring for the full rewrite list and the 3VL
    caveat on ``c == None``.
    """
    return _simplify(expr)


# ---------------------------------------------------------------------------
# Per-node simplifiers — return ``expr`` unchanged when no rule fires so
# the caller's identity check (``out is expr``) stays meaningful.
# ---------------------------------------------------------------------------


def _simplify_not(expr: "Not") -> Expression:
    inner = _simplify(expr.operand)
    return expr if inner is expr.operand else Not(inner)


def _simplify_comparison(expr: "Comparison") -> Expression:
    left = _simplify(expr.left)
    right = _simplify(expr.right)
    if left is expr.left and right is expr.right:
        return expr
    return Comparison(left, expr.op, right)


def _simplify_between(expr: "Between") -> Expression:
    t = _simplify(expr.target)
    lo = _simplify(expr.low)
    hi = _simplify(expr.high)
    if t is expr.target and lo is expr.low and hi is expr.high:
        return expr
    return Between(t, lo, hi, negated=expr.negated)


def _simplify_isnull(expr: "IsNull") -> Expression:
    t = _simplify(expr.target)
    return expr if t is expr.target else IsNull(t, negated=expr.negated)


def _simplify_like(expr: "Like") -> Expression:
    t = _simplify(expr.target)
    if t is expr.target:
        return expr
    return Like(
        target=t,
        pattern=expr.pattern,
        case_insensitive=expr.case_insensitive,
        negated=expr.negated,
    )


def _simplify_cast(expr: "Cast") -> Expression:
    t = _simplify(expr.operand)
    return expr if t is expr.operand else Cast(t, expr.dtype)


def _simplify_arithmetic(expr: "Arithmetic") -> Expression:
    left = _simplify(expr.left)
    right = _simplify(expr.right)
    if left is expr.left and right is expr.right:
        return expr
    return Arithmetic(expr.op, left, right)


def _simplify_inlist(expr: InList) -> InList:
    target = _simplify(expr.target)
    deduped = _dedupe_preserve_order(expr.values)
    if target is expr.target and deduped == expr.values:
        return expr
    return InList(
        target=target,
        values=deduped,
        negated=expr.negated,
        includes_null=expr.includes_null,
    )


def _simplify_logical(expr: Logical) -> Expression:
    # Flatten same-op nesting before simplifying children. A left-
    # leaning chain ``(((a | b) | c) | d)`` (the shape Python's ``|``
    # builds) is N-1 nested ``Logical(OR)`` nodes — collapsing each
    # level independently would allocate an intermediate ``InList``
    # at every level. Flattening first means one OR collapse pass
    # over the full operand list and one final ``InList``.
    flat: "list[Expression]" = []
    _flatten_same_op(expr, expr.op, flat)
    # Now simplify each non-same-op child individually.
    simplified = [_simplify(o) for o in flat]
    # A child may itself simplify *into* the same op (rare, but
    # possible if a sub-expression rewrote to ``Logical(OR, ...)``)
    # — do one more flatten pass to absorb it.
    needs_reflatten = any(
        isinstance(c, Logical) and c.op is expr.op for c in simplified
    )
    if needs_reflatten:
        reflattened: "list[Expression]" = []
        for c in simplified:
            if isinstance(c, Logical) and c.op is expr.op:
                reflattened.extend(c.operands)
            else:
                reflattened.append(c)
        simplified = reflattened

    if expr.op is LogicalOp.OR:
        return _collapse_or(simplified)
    return _collapse_and(simplified)


# Concrete-type dispatch — one ``type(expr)`` lookup beats an
# ``isinstance`` chain of 8+ checks every visit. Every AST node
# class is concrete (``Predicate`` is a mixin), so identity-keyed
# dispatch is sound. Leaves (``Column`` / ``Literal``) fall through
# to "return as-is".
_SIMPLIFY_DISPATCH: "dict[type, Any]" = {
    InList: _simplify_inlist,
    Logical: _simplify_logical,
    Not: _simplify_not,
    Comparison: _simplify_comparison,
    Between: _simplify_between,
    IsNull: _simplify_isnull,
    Like: _simplify_like,
    Cast: _simplify_cast,
    Arithmetic: _simplify_arithmetic,
}


def _simplify(expr: Expression) -> Expression:
    handler = _SIMPLIFY_DISPATCH.get(type(expr))
    if handler is None:
        return expr  # Column, Literal, or any leaf — already canonical.
    return handler(expr)


# ---------------------------------------------------------------------------
# Helpers used by the dispatchers above.
# ---------------------------------------------------------------------------


def _dedupe_preserve_order(values: "tuple[Any, ...]") -> "tuple[Any, ...]":
    """Drop duplicate values while keeping the first occurrence's position.

    Hashable values use a ``set`` fast path; the unhashable branch
    falls back to a linear ``in out`` scan so dicts / lists land
    in the right slot deterministically. The latter is O(n²) but
    only fires when the caller explicitly seeded the InList with
    unhashable types — uncommon in practice.
    """
    seen: "set[Any]" = set()
    out: "list[Any]" = []
    for v in values:
        try:
            if v in seen:
                continue
            seen.add(v)
        except TypeError:
            if v in out:
                continue
        out.append(v)
    return tuple(out)


def _flatten_same_op(
    expr: Expression,
    op: LogicalOp,
    out: "list[Expression]",
) -> None:
    """Walk ``expr`` and append every non-same-op leaf into ``out``.

    Same-op nested ``Logical`` nodes are descended into; everything
    else (including Logical with a different op) is appended whole.
    """
    if isinstance(expr, Logical) and expr.op is op:
        for child in expr.operands:
            _flatten_same_op(child, op, out)
    else:
        out.append(expr)


def _collapse_or(operands: "list[Expression]") -> Expression:
    """Merge OR-of-(EQ | InList | IsNull) on the same target into one InList.

    The classifier returns a (target, values, includes_null) triple
    for the foldable shapes; everything else passes through
    untouched. We group by structural target equality and rewrite
    only when a group accumulated more than one contribution (a
    single ``c == 1`` stays as-is — folding it into a one-element
    ``InList`` is louder for no win).

    Group lookup keys on a cached ``hash(target)`` per ``_OrGroup``
    — ``Expression.__hash__`` is structural (walks the dataclass
    fields), so without the cache an OR chain of length N pays
    O(N²) hashes during the merge sweep.
    """
    groups: "list[_OrGroup]" = []
    classifications: "list[int | None]" = []  # index into ``groups`` or None.

    for op in operands:
        classified = _classify_or_operand(op)
        if classified is None:
            classifications.append(None)
            continue
        target, values, includes_null = classified
        target_hash = hash(target)
        gidx = _find_group_for_target(groups, target, target_hash)
        if gidx is None:
            groups.append(_OrGroup(
                target=target,
                target_hash=target_hash,
                values=list(values),
                includes_null=includes_null,
            ))
            classifications.append(len(groups) - 1)
        else:
            g = groups[gidx]
            g.values.extend(values)
            g.includes_null = g.includes_null or includes_null
            classifications.append(gidx)

    contributions = [0] * len(groups)
    for c in classifications:
        if c is not None:
            contributions[c] += 1

    # If every group has < 2 contributions, the collapse would be a
    # no-op rename — emit the original (flattened) Logical.
    if not any(n >= 2 for n in contributions):
        return _logical_or_finalize(operands)

    new_ops: "list[Expression]" = []
    placed = [False] * len(groups)
    for idx, op in enumerate(operands):
        gidx = classifications[idx]
        if gidx is None or contributions[gidx] < 2:
            new_ops.append(op)
            continue
        if placed[gidx]:
            continue
        g = groups[gidx]
        new_ops.append(InList(
            target=g.target,
            values=_dedupe_preserve_order(tuple(g.values)),
            negated=False,
            includes_null=g.includes_null,
        ))
        placed[gidx] = True

    return _logical_or_finalize(new_ops)


def _collapse_and(operands: "list[Expression]") -> Expression:
    """Drop structurally duplicate conjuncts (``p AND p → p``).

    Hash buckets give an O(n) pass; structural ``equals`` decides
    inside a bucket so distinct nodes sharing a hash don't get
    merged. No null-aware NE → ``not_in`` collapse here — see the
    module docstring for why it is not safe under SQL 3VL.
    """
    if len(operands) <= 1:
        if not operands:
            return Logical(LogicalOp.AND, tuple(operands))
        return operands[0]
    buckets: "dict[int, list[Expression]]" = {}
    unique: "list[Expression]" = []
    for op in operands:
        h = hash(op)
        bucket = buckets.setdefault(h, [])
        if any(prev.equals(op) for prev in bucket):
            continue
        bucket.append(op)
        unique.append(op)
    if len(unique) == 1:
        return unique[0]
    if len(unique) == len(operands):
        return Logical(LogicalOp.AND, tuple(operands))
    return Logical(LogicalOp.AND, tuple(unique))


def _logical_or_finalize(operands: "list[Expression]") -> Expression:
    if len(operands) == 1:
        return operands[0]
    return Logical(LogicalOp.OR, tuple(operands))


@dataclasses.dataclass(slots=True)
class _OrGroup:
    """Mutable accumulator for one OR-collapse target group.

    ``target_hash`` caches ``hash(target)`` so the linear scan in
    :func:`_find_group_for_target` is one int compare per group
    instead of re-running the structural hash on every probe.
    """

    target: Expression
    target_hash: int
    values: "list[Any]"
    includes_null: bool


def _classify_or_operand(
    op: Expression,
) -> "tuple[Expression, tuple[Any, ...], bool] | None":
    """Return (target, values, includes_null) for an OR-foldable operand.

    Foldable shapes:

    - ``Comparison(target, EQ, Literal(v))`` with ``v is not None``.
      We deliberately *do not* fold ``v is None`` — see the
      :func:`simplify` docstring on the 3VL caveat.
    - ``Comparison(Literal(v), EQ, target)`` (literal-on-left) — same.
    - ``InList(target, values, negated=False, includes_null=…)``.
    - ``IsNull(target, negated=False)`` — contributes the
      ``includes_null=True`` flag with no extra values.

    Anything else returns ``None`` and stays in the OR untouched.
    """
    if isinstance(op, Comparison) and op.op is CompareOp.EQ:
        if isinstance(op.right, Literal):
            v = op.right.value
            if v is None:
                return None
            return (op.left, (v,), False)
        if isinstance(op.left, Literal):
            v = op.left.value
            if v is None:
                return None
            return (op.right, (v,), False)
        return None
    if isinstance(op, InList) and not op.negated:
        return (op.target, op.values, op.includes_null)
    if isinstance(op, IsNull) and not op.negated:
        return (op.target, (), True)
    return None


def _find_group_for_target(
    groups: "list[_OrGroup]",
    target: Expression,
    target_hash: int,
) -> "int | None":
    """Linear lookup keyed by structural equality.

    A dict keyed on the target ``__hash__`` would shave the O(n²)
    worst case, but ``Expression.__eq__`` builds a Comparison node
    instead of returning ``bool`` (the operator-overload trick that
    makes ``col("x") == 5`` work) — using one as a dict key bypasses
    that and would silently collapse hash-colliding distinct
    targets. The linear scan keeps the contract explicit.

    ``target_hash`` is passed in (computed once by the caller)
    because the structural hash walks the dataclass fields —
    re-running it per group would make the merge O(N²) in chain
    length.
    """
    for idx, g in enumerate(groups):
        if g.target_hash == target_hash and g.target.equals(target):
            return idx
    return None
