"""Pure-Python compiler for :class:`Expression` trees.

:func:`to_python` returns a callable that takes a row mapping
(``Mapping[str, Any]``) and evaluates the expression against it.
The result is the natural Python value of the expression — a bool
for predicates, a number / string / etc. for scalar expressions.

Filtering helper
----------------

For row collections, :func:`filter_rows` calls the compiled
predicate against each row and yields the matches:

    >>> from yggdrasil.data.expr import col
    >>> rows = [{"x": 1}, {"x": 2}, {"x": 3}]
    >>> from yggdrasil.data.expr.backends.python import filter_rows
    >>> list(filter_rows(col("x") > 1, rows))
    [{'x': 2}, {'x': 3}]

NULL semantics
--------------

By default, missing columns evaluate to :data:`None` (SQL
three-valued logic): a comparison against ``None`` returns ``None``,
``AND`` of ``None`` and ``False`` is ``False``, ``AND`` of ``None``
and anything else is ``None``. ``Predicate.to_python(strict=True)``
raises :class:`KeyError` instead — useful when the caller wants to
detect schema drift loudly.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Iterator, Mapping

from ..nodes import (
    Arithmetic,
    ArithmeticOp,
    Between,
    Cast,
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
)


__all__ = ["to_python", "filter_rows"]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def to_python(
    expr: Expression,
    *,
    strict: bool = False,
) -> Callable[[Mapping[str, Any]], Any]:
    """Compile *expr* to a callable.

    Returned callable accepts a row mapping and returns the
    expression's value. ``strict=True`` makes missing columns
    raise :class:`KeyError`; the default returns ``None`` for
    them, matching SQL three-valued logic.
    """
    return _compile(expr, strict=strict)


def filter_rows(
    expr: Expression,
    rows: Iterable[Mapping[str, Any]],
    *,
    strict: bool = False,
) -> Iterator[Mapping[str, Any]]:
    """Yield rows from *rows* that satisfy *expr* (treated as a predicate).

    A non-``True`` evaluation result rejects the row — that
    includes ``False`` and SQL's UNKNOWN (``None``), so a
    predicate referencing a missing column rejects the row instead
    of silently passing it.
    """
    pred = _compile(expr, strict=strict)
    for row in rows:
        if pred(row) is True:
            yield row


# ---------------------------------------------------------------------------
# Compilation — recursive descent. Each branch returns a thunk that
# takes the row and produces the node's value.
# ---------------------------------------------------------------------------


def _compile(
    expr: Expression,
    *,
    strict: bool,
) -> Callable[[Mapping[str, Any]], Any]:
    if isinstance(expr, Literal):
        value = expr.value
        return lambda _row, _v=value: _v

    if isinstance(expr, Column):
        name = expr.name
        if strict:
            return lambda row, _n=name: row[_n]
        return lambda row, _n=name: row.get(_n)

    if isinstance(expr, Comparison):
        return _compile_comparison(expr, strict=strict)

    if isinstance(expr, Logical):
        return _compile_logical(expr, strict=strict)

    if isinstance(expr, Not):
        inner = _compile(expr.operand, strict=strict)
        return lambda row, _i=inner: _three_valued_not(_i(row))

    if isinstance(expr, Between):
        return _compile_between(expr, strict=strict)

    if isinstance(expr, InList):
        return _compile_inlist(expr, strict=strict)

    if isinstance(expr, IsNull):
        target = _compile(expr.target, strict=strict)
        if expr.negated:
            return lambda row, _t=target: _t(row) is not None
        return lambda row, _t=target: _t(row) is None

    if isinstance(expr, Like):
        return _compile_like(expr, strict=strict)

    if isinstance(expr, Cast):
        # Pure-Python cast is a no-op for most types — we trust the
        # row already carries the right Python representation.
        # Engines that need a real coercion (Spark) emit a real
        # cast in their backend.
        return _compile(expr.operand, strict=strict)

    if isinstance(expr, Arithmetic):
        return _compile_arithmetic(expr, strict=strict)

    raise NotImplementedError(
        f"Python backend does not implement node type {type(expr).__name__}."
    )


# ---------------------------------------------------------------------------
# Per-node helpers — kept named so a stack trace points at the
# operator that mis-evaluated, not a deeply nested lambda.
# ---------------------------------------------------------------------------


def _compile_comparison(
    expr: Comparison,
    *,
    strict: bool,
) -> Callable[[Mapping[str, Any]], Any]:
    left = _compile(expr.left, strict=strict)
    right = _compile(expr.right, strict=strict)
    op = expr.op

    if op is CompareOp.EQ:
        return lambda row, _l=left, _r=right: _ternary(_l(row), _r(row), _eq)
    if op is CompareOp.NE:
        return lambda row, _l=left, _r=right: _ternary(_l(row), _r(row), _ne)
    if op is CompareOp.LT:
        return lambda row, _l=left, _r=right: _ternary(_l(row), _r(row), _lt)
    if op is CompareOp.LE:
        return lambda row, _l=left, _r=right: _ternary(_l(row), _r(row), _le)
    if op is CompareOp.GT:
        return lambda row, _l=left, _r=right: _ternary(_l(row), _r(row), _gt)
    if op is CompareOp.GE:
        return lambda row, _l=left, _r=right: _ternary(_l(row), _r(row), _ge)
    raise NotImplementedError(f"Comparison op {op!r} not implemented.")


def _compile_logical(
    expr: Logical,
    *,
    strict: bool,
) -> Callable[[Mapping[str, Any]], Any]:
    operands = [_compile(o, strict=strict) for o in expr.operands]
    if expr.op is LogicalOp.AND:
        return lambda row, ops=operands: _three_valued_and(o(row) for o in ops)
    if expr.op is LogicalOp.OR:
        return lambda row, ops=operands: _three_valued_or(o(row) for o in ops)
    raise NotImplementedError(f"Logical op {expr.op!r} not implemented.")


def _compile_between(
    expr: Between,
    *,
    strict: bool,
) -> Callable[[Mapping[str, Any]], Any]:
    target = _compile(expr.target, strict=strict)
    low = _compile(expr.low, strict=strict)
    high = _compile(expr.high, strict=strict)
    negated = expr.negated

    def _eval(row: Mapping[str, Any]) -> Any:
        v = target(row)
        if v is None:
            return None
        lo = low(row)
        hi = high(row)
        if lo is None or hi is None:
            return None
        result = lo <= v <= hi
        return (not result) if negated else result

    return _eval


def _compile_inlist(
    expr: InList,
    *,
    strict: bool,
) -> Callable[[Mapping[str, Any]], Any]:
    target = _compile(expr.target, strict=strict)
    # Hashable values get a set fast-path; everything else falls
    # back to linear lookup.
    try:
        values = frozenset(expr.values)
    except TypeError:
        values = expr.values  # type: ignore[assignment]
    includes_null = expr.includes_null
    negated = expr.negated

    def _eval(row: Mapping[str, Any]) -> Any:
        v = target(row)
        if v is None:
            # SQL semantics: ``NULL IN (...)`` is UNKNOWN unless
            # the user explicitly added NULL to the set, in which
            # case it matches.
            if includes_null:
                return not negated
            return None
        is_in = v in values
        if not is_in and includes_null and v is None:
            is_in = True
        return (not is_in) if negated else is_in

    return _eval


def _compile_like(
    expr: Like,
    *,
    strict: bool,
) -> Callable[[Mapping[str, Any]], Any]:
    pattern = _like_to_regex(expr.pattern)
    flags = re.DOTALL | (re.IGNORECASE if expr.case_insensitive else 0)
    rx = re.compile(pattern, flags)
    target = _compile(expr.target, strict=strict)
    negated = expr.negated

    def _eval(row: Mapping[str, Any]) -> Any:
        v = target(row)
        if v is None:
            return None
        match = rx.fullmatch(str(v)) is not None
        return (not match) if negated else match

    return _eval


def _compile_arithmetic(
    expr: Arithmetic,
    *,
    strict: bool,
) -> Callable[[Mapping[str, Any]], Any]:
    left = _compile(expr.left, strict=strict)
    right = _compile(expr.right, strict=strict)
    op = expr.op

    def _wrap(fn):
        def _eval(row: Mapping[str, Any]) -> Any:
            a = left(row)
            b = right(row)
            if a is None or b is None:
                return None
            return fn(a, b)

        return _eval

    if op is ArithmeticOp.ADD:
        return _wrap(lambda a, b: a + b)
    if op is ArithmeticOp.SUB:
        return _wrap(lambda a, b: a - b)
    if op is ArithmeticOp.MUL:
        return _wrap(lambda a, b: a * b)
    if op is ArithmeticOp.DIV:
        return _wrap(lambda a, b: a / b)
    if op is ArithmeticOp.MOD:
        return _wrap(lambda a, b: a % b)
    raise NotImplementedError(f"Arithmetic op {op!r} not implemented.")


# ---------------------------------------------------------------------------
# Three-valued logic primitives — SQL semantics.
# ---------------------------------------------------------------------------


def _ternary(a: Any, b: Any, fn: Callable[[Any, Any], bool]) -> "bool | None":
    """Run ``fn(a, b)`` with NULL propagation.

    Either operand being ``None`` makes the result ``None``
    (UNKNOWN). Used by every comparison so missing column
    semantics match what SQL would compute.
    """
    if a is None or b is None:
        return None
    return fn(a, b)


def _eq(a: Any, b: Any) -> bool:
    return a == b


def _ne(a: Any, b: Any) -> bool:
    return a != b


def _lt(a: Any, b: Any) -> bool:
    return a < b


def _le(a: Any, b: Any) -> bool:
    return a <= b


def _gt(a: Any, b: Any) -> bool:
    return a > b


def _ge(a: Any, b: Any) -> bool:
    return a >= b


def _three_valued_and(values: Iterable[Any]) -> "bool | None":
    """``AND`` with NULL propagation. ``False`` short-circuits.

    Any ``False`` makes the result ``False`` even when other
    operands are ``None``. Otherwise a ``None`` poisons to
    ``None``. Else ``True``.
    """
    saw_null = False
    for v in values:
        if v is False:
            return False
        if v is None:
            saw_null = True
    return None if saw_null else True


def _three_valued_or(values: Iterable[Any]) -> "bool | None":
    """``OR`` with NULL propagation. ``True`` short-circuits."""
    saw_null = False
    for v in values:
        if v is True:
            return True
        if v is None:
            saw_null = True
    return None if saw_null else False


def _three_valued_not(value: Any) -> "bool | None":
    if value is None:
        return None
    return not value


# ---------------------------------------------------------------------------
# LIKE → regex
# ---------------------------------------------------------------------------


def _like_to_regex(pattern: str) -> str:
    """Translate a SQL ``LIKE`` pattern to a Python regex.

    ``%`` becomes ``.*``, ``_`` becomes ``.``; everything else is
    regex-escaped. Escaped wildcards (``\\%``, ``\\_``) match the
    literal character.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            nxt = pattern[i + 1]
            if nxt in ("%", "_"):
                out.append(re.escape(nxt))
                i += 2
                continue
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)
