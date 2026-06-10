"""polars.Expr emitter + best-effort lifter.

:func:`to_polars` translates the AST to a :class:`polars.Expr` —
suitable as the predicate of :meth:`polars.DataFrame.filter` /
:meth:`polars.LazyFrame.filter`. Output uses native polars
operators so optimization (predicate pushdown, projection
pruning) kicks in.

:func:`from_polars` parses a polars expression's serialized form
back into our AST. Polars exposes ``Expr.meta.serialize()`` /
``Expr.meta.tree_format()`` for introspection — the lifter walks
the tree-format output. Anything beyond comparisons + boolean
combinators raises :class:`NotImplementedError`.
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


__all__ = ["to_polars", "from_polars"]


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------


def to_polars(expr: Expression):
    """Emit *expr* as a :class:`polars.Expr`."""
    import polars as pl

    return _emit(expr, pl)


def _emit(expr: Expression, pl):  # type: ignore[no-untyped-def]
    if isinstance(expr, Column):
        return pl.col(expr.name)

    if isinstance(expr, Literal):
        if expr.dtype is not None:
            try:
                pl_dt = expr.dtype.to_polars()
            except Exception:
                pl_dt = None
            if pl_dt is not None:
                return pl.lit(expr.value, dtype=pl_dt)
        return pl.lit(expr.value)

    if isinstance(expr, Comparison):
        return _emit_comparison(expr, pl)

    if isinstance(expr, Logical):
        return _emit_logical(expr, pl)

    if isinstance(expr, Not):
        return ~_emit(expr.operand, pl)

    if isinstance(expr, Between):
        target = _emit(expr.target, pl)
        low = _emit(expr.low, pl)
        high = _emit(expr.high, pl)
        between = (target >= low) & (target <= high)
        return ~between if expr.negated else between

    if isinstance(expr, InList):
        return _emit_inlist(expr, pl)

    if isinstance(expr, IsNull):
        target = _emit(expr.target, pl)
        return target.is_not_null() if expr.negated else target.is_null()

    if isinstance(expr, Like):
        return _emit_like(expr, pl)

    if isinstance(expr, Cast):
        target = _emit(expr.operand, pl)
        try:
            pl_dt = expr.dtype.to_polars()
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"Cannot translate cast target {expr.dtype!r} to polars."
            ) from exc
        return target.cast(pl_dt)

    if isinstance(expr, Arithmetic):
        return _emit_arithmetic(expr, pl)

    raise NotImplementedError(
        f"polars backend does not implement node type {type(expr).__name__}."
    )


def _emit_comparison(expr: Comparison, pl):  # type: ignore[no-untyped-def]
    left = _emit(expr.left, pl)
    right = _emit(expr.right, pl)
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


def _emit_logical(expr: Logical, pl):  # type: ignore[no-untyped-def]
    operands = [_emit(o, pl) for o in expr.operands]
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


def _emit_inlist(expr: InList, pl):  # type: ignore[no-untyped-def]
    target = _emit(expr.target, pl)
    if not expr.values:
        if expr.includes_null:
            null_clause = target.is_null()
            return ~null_clause if expr.negated else null_clause
        return pl.lit(False) if not expr.negated else pl.lit(True)
    values = list(expr.values)
    base = target.is_in(values)
    if expr.includes_null:
        if expr.negated:
            return (~base) & target.is_not_null()
        return base | target.is_null()
    return ~base if expr.negated else base


def _emit_like(expr: Like, pl):  # type: ignore[no-untyped-def]
    target = _emit(expr.target, pl)
    # polars uses regex via str.contains; convert SQL-style ``%``
    # / ``_`` wildcards to a regex pattern. Mirrors the Python
    # backend's translation so semantics are identical.
    from .python import _like_to_regex

    rx = "^" + _like_to_regex(expr.pattern) + "$"
    if expr.case_insensitive:
        rx = "(?i)" + rx
    match = target.cast(pl.Utf8).str.contains(rx)
    return ~match if expr.negated else match


def _emit_arithmetic(expr: Arithmetic, pl):  # type: ignore[no-untyped-def]
    left = _emit(expr.left, pl)
    right = _emit(expr.right, pl)
    if expr.op is ArithmeticOp.ADD:
        return left + right
    if expr.op is ArithmeticOp.SUB:
        return left - right
    if expr.op is ArithmeticOp.MUL:
        return left * right
    if expr.op is ArithmeticOp.DIV:
        return left / right
    if expr.op is ArithmeticOp.MOD:
        return left % right
    raise NotImplementedError(f"Arithmetic op {expr.op!r} not implemented.")


# ---------------------------------------------------------------------------
# Lift — best-effort. Polars exposes ``Expr.meta.serialize(format="json")``
# in modern builds; we walk that JSON tree.
# ---------------------------------------------------------------------------


def from_polars(pl_expr) -> Expression:  # type: ignore[no-untyped-def]
    """Lift a :class:`polars.Expr` back into our AST.

    Uses ``Expr.meta.serialize(format="json")`` (polars ≥ 0.20).
    Older polars versions raise :class:`NotImplementedError` —
    upgrade or carry the original :class:`Expression` through the
    pipeline.
    """
    meta = getattr(pl_expr, "meta", None)
    if meta is None or not hasattr(meta, "serialize"):
        raise NotImplementedError(
            "from_polars requires polars>=0.20 (Expr.meta.serialize). "
            "Upgrade polars or keep the original yggdrasil Expression."
        )
    try:
        import json

        serialized = meta.serialize(format="json")
    except TypeError:
        # Some intermediate polars versions need ``format=`` as
        # positional, others reject the kwarg entirely. Fall back
        # to the no-arg form (binary) and refuse — JSON is the only
        # form we can walk portably.
        raise NotImplementedError(
            "from_polars: polars Expr.meta.serialize doesn't accept "
            "format='json' on this polars version. Upgrade polars."
        )
    if isinstance(serialized, (bytes, bytearray)):
        serialized = serialized.decode("utf-8")
    tree = json.loads(serialized)
    return _lift_polars_node(tree)


def _lift_polars_node(node: Any) -> Expression:
    """Walk a polars JSON expression tree."""
    if not isinstance(node, dict):
        # Bare scalar: treat as a literal value.
        return Literal(value=node)

    # Polars JSON encodes node types as a single top-level key.
    if "Column" in node:
        from ..builder import col as _col

        return _col(str(node["Column"]))
    if "Literal" in node:
        return _lift_polars_literal(node["Literal"])
    if "BinaryExpr" in node:
        return _lift_polars_binary(node["BinaryExpr"])
    if "Function" in node:
        return _lift_polars_function(node["Function"])
    if "Cast" in node:
        # Cast nodes carry a ``data_type`` we can't fully lift back
        # — yggdrasil doesn't have a polars-DataType→yggdrasil
        # converter on this side. Surface the inner expression as
        # a best-effort.
        inner = node["Cast"].get("expr")
        if inner is None:
            raise NotImplementedError(
                "from_polars: cast node missing inner expression."
            )
        return _lift_polars_node(inner)
    if "Alias" in node:
        return _lift_polars_node(node["Alias"]["expr"])

    raise NotImplementedError(
        f"from_polars: cannot lift polars node {sorted(node.keys())!r}."
    )


def _lift_polars_literal(payload: Any) -> Expression:
    if isinstance(payload, dict) and len(payload) == 1:
        kind, value = next(iter(payload.items()))
        # polars ≥ 1.x wraps every literal in a ``Dyn`` (untyped
        # Python scalar: ``{"Dyn": {"Int": 100}}``) or ``Scalar``
        # (typed: ``{"Scalar": {"String": "buy"}}``) envelope.
        if kind == "Dyn" and isinstance(value, dict) and len(value) == 1:
            return Literal(value=next(iter(value.values())))
        if kind == "Scalar" and isinstance(value, dict) and len(value) == 1:
            scalar_kind, scalar_value = next(iter(value.items()))
            return Literal(value=_decode_polars_scalar(scalar_kind, scalar_value))
        # Legacy (pre-1.x) shape: ``{"<dtype>": <value>}`` — pick the
        # value, drop the dtype.
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        return Literal(value=value)
    return Literal(value=payload)


def _decode_polars_scalar(kind: str, value: Any) -> Any:
    """Decode one ``{"Scalar": {kind: value}}`` payload to a Python value.

    String / Boolean / Int* / UInt* / Float* arrive as plain JSON
    values and pass through. Temporal kinds arrive as epoch offsets;
    ``List`` (the ``is_in`` value-set) arrives as Arrow IPC stream
    bytes. Unknown kinds pass through raw — over-lifting a value we
    don't understand is worse than handing the caller the payload.
    """
    import datetime as dt

    if kind == "Null":
        return None
    if kind == "Date":
        return dt.date(1970, 1, 1) + dt.timedelta(days=value)
    if kind == "Datetime":
        offset, unit, tz = value
        scale_us = {
            "Nanoseconds": 1 / 1000,
            "Microseconds": 1,
            "Milliseconds": 1000,
        }.get(unit)
        if scale_us is None:
            return value
        stamp = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
            microseconds=offset * scale_us,
        )
        if tz is None:
            return stamp.replace(tzinfo=None)
        import zoneinfo

        return stamp.astimezone(zoneinfo.ZoneInfo(tz))
    if kind == "List":
        import pyarrow as pa

        return pa.ipc.open_stream(bytes(value)).read_all().column(0).to_pylist()
    return value


_POLARS_BIN_OPS: dict[str, CompareOp] = {
    "Eq": CompareOp.EQ,
    "NotEq": CompareOp.NE,
    "Lt": CompareOp.LT,
    "LtEq": CompareOp.LE,
    "Gt": CompareOp.GT,
    "GtEq": CompareOp.GE,
}

_POLARS_BIN_ARITH: dict[str, ArithmeticOp] = {
    "Plus": ArithmeticOp.ADD,
    "Minus": ArithmeticOp.SUB,
    "Multiply": ArithmeticOp.MUL,
    "Divide": ArithmeticOp.DIV,
    "Modulus": ArithmeticOp.MOD,
}


def _lift_polars_binary(payload: dict[str, Any]) -> Expression:
    op = payload.get("op")
    left = _lift_polars_node(payload["left"])
    right = _lift_polars_node(payload["right"])
    if op in _POLARS_BIN_OPS:
        return Comparison(left, _POLARS_BIN_OPS[op], right)
    if op in _POLARS_BIN_ARITH:
        return Arithmetic(_POLARS_BIN_ARITH[op], left, right)
    if op == "And":
        return Logical(LogicalOp.AND, (left, right))
    if op == "Or":
        return Logical(LogicalOp.OR, (left, right))
    raise NotImplementedError(
        f"from_polars: binary op {op!r} not implemented."
    )


def _lift_polars_function(payload: dict[str, Any]) -> Expression:
    function = payload.get("function") or {}
    # polars ≥ 1.x wraps the name in a category envelope —
    # ``{"Boolean": "IsNull"}`` for option-free functions,
    # ``{"Boolean": {"IsIn": {"nulls_equal": false}}}`` for ones with
    # options. Older builds used a flat ``{"function": "IsNull"}``.
    name: Any = function
    options: dict[str, Any] = {}
    if isinstance(function, dict):
        if "function" in function:
            name = function["function"]
        elif len(function) == 1:
            inner = next(iter(function.values()))
            if isinstance(inner, dict) and len(inner) == 1:
                name = next(iter(inner.keys()))
                if isinstance(inner[name], dict):
                    options = inner[name]
            else:
                name = inner
    args = [_lift_polars_node(a) for a in payload.get("input", [])]
    if name == "IsNull":
        return IsNull(args[0], negated=False)
    if name == "IsNotNull":
        return IsNull(args[0], negated=True)
    if name == "Not":
        return Not(args[0])
    if name == "IsIn" and args:
        target = args[0]
        # ``IsIn`` second arg is a list literal — recover values.
        if len(args) == 1:
            raise NotImplementedError(
                "from_polars: IsIn missing value-set."
            )
        values_node = args[1]
        if isinstance(values_node, Literal) and isinstance(
            values_node.value, (list, tuple)
        ):
            raw = list(values_node.value)
        else:
            raise NotImplementedError(
                "from_polars: IsIn values must be a list literal."
            )
        non_null = tuple(v for v in raw if v is not None)
        has_null = len(non_null) != len(raw)
        return InList(
            target=target,
            values=non_null,
            negated=False,
            includes_null=has_null,
        )
    if name == "IsBetween" and len(args) == 3:
        if options.get("closed", "Both") != "Both":
            raise NotImplementedError(
                "from_polars: IsBetween with closed="
                f"{options.get('closed')!r} has no AST equivalent — "
                "Between is inclusive on both bounds."
            )
        return Between(args[0], args[1], args[2])

    raise NotImplementedError(
        f"from_polars: function {name!r} not implemented yet."
    )
