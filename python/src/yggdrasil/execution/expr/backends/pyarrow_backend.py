"""pyarrow.compute.Expression emitter + best-effort lifter.

:func:`to_pyarrow` translates the AST to a
:class:`pyarrow.compute.Expression`. Output is suitable as the
``filter=`` argument on :class:`pyarrow.dataset.Dataset.scanner` /
:meth:`pyarrow.Table.filter`, the typical "push the predicate
down to Arrow" surface.

:func:`from_pyarrow` walks a pyarrow expression back into our AST
on a best-effort basis. pyarrow's expression introspection is
limited to a few well-known function names (``equal``,
``less_equal``, ``is_in``, ``and_kleene``, ``or_kleene``,
``invert``, ``is_null``); anything outside that set raises
:class:`NotImplementedError`.
"""

from __future__ import annotations

from typing import Any

from ..nodes import (
    Arithmetic,
    Between,
    Cast,
    Column,
    Comparison,
    Expression,
    InList,
    IsNull,
    Like,
    Literal,
    Logical,
    Not,
)
from ..operators import ArithmeticOp, CompareOp, LogicalOp


__all__ = ["to_pyarrow", "from_pyarrow"]


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------


def to_pyarrow(expr: Expression):
    """Emit *expr* as a :class:`pyarrow.compute.Expression`."""
    import pyarrow.compute as pc

    return _emit(expr, pc)


def _emit(expr: Expression, pc):  # type: ignore[no-untyped-def]
    if isinstance(expr, Column):
        return pc.field(expr.name)

    if isinstance(expr, Literal):
        # pyarrow.compute scalars come from pyarrow.scalar; pyarrow
        # then wraps them transparently inside Expression
        # comparisons. Letting Python objects flow through is
        # equivalent for built-in types but loses dtype intent —
        # use the pinned dtype when present.
        import pyarrow as pa

        if expr.dtype is not None:
            try:
                arrow_dt = expr.dtype.to_arrow()
            except Exception:
                arrow_dt = None
        else:
            arrow_dt = None
        return pa.scalar(expr.value, type=arrow_dt)

    if isinstance(expr, Comparison):
        return _emit_comparison(expr, pc)

    if isinstance(expr, Logical):
        return _emit_logical(expr, pc)

    if isinstance(expr, Not):
        return ~_emit(expr.operand, pc)

    if isinstance(expr, Between):
        target = _emit(expr.target, pc)
        low = _emit(expr.low, pc)
        high = _emit(expr.high, pc)
        between = (target >= low) & (target <= high)
        return ~between if expr.negated else between

    if isinstance(expr, InList):
        return _emit_inlist(expr, pc)

    if isinstance(expr, IsNull):
        target = _emit(expr.target, pc)
        is_null = target.is_null()
        return ~is_null if expr.negated else is_null

    if isinstance(expr, Like):
        return _emit_like(expr, pc)

    if isinstance(expr, Cast):
        target = _emit(expr.operand, pc)
        try:
            arrow_dt = expr.dtype.to_arrow()
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"Cannot translate cast target {expr.dtype!r} to arrow."
            ) from exc
        return target.cast(arrow_dt)

    if isinstance(expr, Arithmetic):
        return _emit_arithmetic(expr, pc)

    raise NotImplementedError(
        f"pyarrow backend does not implement node type {type(expr).__name__}."
    )


def _emit_comparison(expr: Comparison, pc):  # type: ignore[no-untyped-def]
    left = _emit(expr.left, pc)
    right = _emit(expr.right, pc)
    op = expr.op
    if op is CompareOp.EQ:
        return left == right
    if op is CompareOp.NE:
        return left != right
    if op is CompareOp.LT:
        return left < right
    if op is CompareOp.LE:
        return left <= right
    if op is CompareOp.GT:
        return left > right
    if op is CompareOp.GE:
        return left >= right
    raise NotImplementedError(f"Comparison op {op!r} not implemented.")


def _emit_logical(expr: Logical, pc):  # type: ignore[no-untyped-def]
    operands = [_emit(o, pc) for o in expr.operands]
    if expr.op is LogicalOp.AND:
        result = operands[0]
        for o in operands[1:]:
            result = result & o
        return result
    if expr.op is LogicalOp.OR:
        result = operands[0]
        for o in operands[1:]:
            result = result | o
        return result
    raise NotImplementedError(f"Logical op {expr.op!r} not implemented.")


def _emit_inlist(expr: InList, pc):  # type: ignore[no-untyped-def]
    import pyarrow as pa

    target = _emit(expr.target, pc)
    if not expr.values:
        # ``IN ()`` — degenerate; keep the same SQL semantics.
        if expr.includes_null:
            null_clause = target.is_null()
            return ~null_clause if expr.negated else null_clause
        return pc.scalar(False) if not expr.negated else pc.scalar(True)
    values_arr = pa.array(list(expr.values))
    base = target.isin(values_arr)
    if expr.includes_null:
        if expr.negated:
            return ~base & ~target.is_null()
        return base | target.is_null()
    return ~base if expr.negated else base


def _emit_like(expr: Like, pc):  # type: ignore[no-untyped-def]
    target = _emit(expr.target, pc)
    if expr.case_insensitive:
        match = pc.match_like(target, expr.pattern, ignore_case=True)
    else:
        match = pc.match_like(target, expr.pattern)
    return ~match if expr.negated else match


def _emit_arithmetic(expr: Arithmetic, pc):  # type: ignore[no-untyped-def]
    left = _emit(expr.left, pc)
    right = _emit(expr.right, pc)
    if expr.op is ArithmeticOp.ADD:
        return left + right
    if expr.op is ArithmeticOp.SUB:
        return left - right
    if expr.op is ArithmeticOp.MUL:
        return left * right
    if expr.op is ArithmeticOp.DIV:
        return left / right
    if expr.op is ArithmeticOp.MOD:
        return pc.divmod(left, right)[1]
    raise NotImplementedError(f"Arithmetic op {expr.op!r} not implemented.")


# ---------------------------------------------------------------------------
# Lift — best-effort. pyarrow.compute exposes
# ``Expression._call_str`` (in newer versions) and ``__repr__`` but
# the structural surface is private; we walk the underlying
# substrait-style proto by introspecting attributes that exist
# across versions.
# ---------------------------------------------------------------------------


def from_pyarrow(arrow_expr) -> Expression:  # type: ignore[no-untyped-def]
    """Walk a :class:`pyarrow.compute.Expression` back into our AST.

    Limited by what the pyarrow API exposes — and on the pyarrow
    versions this project supports (``>=20``) that is effectively
    nothing: the private per-node accessors older builds leaked
    (``_call_function_name`` / ``_args`` / ``_field_ref``) are gone,
    and the only public serialization (``to_substrait``) requires
    the input schema the lifter doesn't have. Expect
    :class:`NotImplementedError` for any compound expression; keep
    the original yggdrasil :class:`Expression` and emit fresh
    instead of round-tripping through pyarrow.
    """
    return _lift(arrow_expr)


def _lift(node: Any) -> Expression:
    import pyarrow.compute as pc

    # Field reference — pyarrow 16+ exposes ``_field_ref_name``;
    # older builds use ``__repr__``-style introspection.
    name = _try_field_name(node, pc)
    if name is not None:
        from ..builder import col as _col

        return _col(name)

    # Literal — pyarrow scalar wrapped as a constant Expression.
    scalar_value = _try_literal_value(node, pc)
    if scalar_value is not _MISSING:
        return Literal(value=scalar_value)

    # Compound — function-call expression. ``Expression._function`` /
    # ``_call_function_name`` give the kernel name.
    fn_name, args = _try_function(node, pc)
    if fn_name is not None:
        return _lift_function(fn_name, args)

    raise NotImplementedError(
        f"from_pyarrow: cannot lift expression {node!r}. "
        "Add the matching shape to the lifter or carry the original "
        "expression instead."
    )


_MISSING: Any = object()


def _try_field_name(node: Any, pc) -> "str | None":  # type: ignore[no-untyped-def]
    # PyArrow ≥ 14 lets ``Expression`` round-trip through
    # ``Expression._field_ref()``; older builds require parsing
    # ``str(node)`` of the form ``[fieldname]``.
    fr = getattr(node, "_field_ref", None)
    if callable(fr):
        try:
            ref = fr()
            name = getattr(ref, "name", None)
            if name is not None:
                return name
        except Exception:
            pass
    text = str(node)
    if text.startswith("[") and text.endswith("]") and "[" not in text[1:-1]:
        return text[1:-1]
    return None


def _try_literal_value(node: Any, pc) -> Any:  # type: ignore[no-untyped-def]
    scalar = getattr(node, "_scalar", None)
    if scalar is None:
        # ``Expression`` wraps scalars opaquely; the public method
        # is ``Expression._scalar`` / ``Expression.cast``-style
        # property access. Fall back to None when the node isn't
        # a literal — text sniffing is unreliable across versions.
        return _MISSING
    try:
        return scalar.as_py()
    except Exception:
        return _MISSING


def _try_function(node: Any, pc) -> "tuple[str | None, list[Any]]":  # type: ignore[no-untyped-def]
    fn_name = getattr(node, "_call_function_name", None) or getattr(
        node, "_function", None,
    )
    args = getattr(node, "_args", None) or getattr(node, "_call_args", None)
    if not fn_name or not args:
        return None, []
    return fn_name, list(args)


def _lift_function(name: str, args: list[Any]) -> Expression:
    if name in _PA_COMPARE:
        return Comparison(_lift(args[0]), _PA_COMPARE[name], _lift(args[1]))
    if name in ("and", "and_kleene"):
        return Logical(LogicalOp.AND, tuple(_lift(a) for a in args))
    if name in ("or", "or_kleene"):
        return Logical(LogicalOp.OR, tuple(_lift(a) for a in args))
    if name in ("invert", "not"):
        return Not(_lift(args[0]))
    if name == "is_null":
        return IsNull(_lift(args[0]), negated=False)
    if name == "is_valid":
        return IsNull(_lift(args[0]), negated=True)
    if name == "isin":
        target = _lift(args[0])
        # Second arg is the value-set Array — peek at its values.
        try:
            values = list(args[1].as_py())
        except Exception:
            try:
                values = list(args[1])
            except Exception as exc:  # pragma: no cover - defensive
                raise NotImplementedError(
                    f"from_pyarrow: cannot lift isin value-set {args[1]!r}."
                ) from exc
        non_null = tuple(v for v in values if v is not None)
        has_null = len(non_null) != len(values)
        return InList(
            target=target,
            values=non_null,
            negated=False,
            includes_null=has_null,
        )
    if name in ("match_like",):
        return Like(
            target=_lift(args[0]),
            pattern=str(args[1]),
            case_insensitive=False,
            negated=False,
        )

    raise NotImplementedError(
        f"from_pyarrow: function {name!r} not implemented yet."
    )


_PA_COMPARE: dict[str, CompareOp] = {
    "equal": CompareOp.EQ,
    "not_equal": CompareOp.NE,
    "less": CompareOp.LT,
    "less_equal": CompareOp.LE,
    "greater": CompareOp.GT,
    "greater_equal": CompareOp.GE,
}
