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


def _match_literal_tz(pl_dt, value, pl):  # type: ignore[no-untyped-def]
    """Re-stamp a ``pl.Datetime`` literal's tz from the *value*'s own zone key.

    polars treats ``"UTC"`` and ``"Etc/UTC"`` as **distinct** zones for
    comparison and raises ``SchemaError`` when a literal's zone string
    differs from the column's — even though both name the same instant.
    yggdrasil's :class:`~yggdrasil.enums.timezone.Timezone` canonicalises
    every UTC-family spelling to ``"Etc/UTC"``, so a literal that the tz
    pushdown aligned to a column polars spells ``"UTC"`` would emit
    ``"Etc/UTC"`` and blow up the filter.

    The aligned datetime *value* carries the column's actual zone via its
    ``tzinfo.key`` (the pushdown builds it with ``ZoneInfo(column_tz)``), so
    prefer that key over the canonicalised dtype tz. No-op when the value
    has no explicit zone key (fixed-offset / naive) or the strings already
    agree, so non-UTC zones (``"Europe/Paris"`` …) and the steady state are
    untouched. Arrow is unaffected — it treats the two spellings as equal.
    """
    import datetime as _dt

    if not isinstance(pl_dt, pl.Datetime) or pl_dt.time_zone is None:
        return pl_dt
    if not isinstance(value, _dt.datetime) or value.tzinfo is None:
        return pl_dt
    key = getattr(value.tzinfo, "key", None)
    if key and key != pl_dt.time_zone:
        return pl.Datetime(time_unit=pl_dt.time_unit, time_zone=key)
    return pl_dt


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
                return pl.lit(expr.value, dtype=_match_literal_tz(pl_dt, expr.value, pl))
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
    if isinstance(payload, dict):
        # ``{"<dtype>": <value>}`` — pick the value, drop the dtype.
        if len(payload) == 1:
            value = next(iter(payload.values()))
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
            return Literal(value=value)
    return Literal(value=payload)


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
    name = function.get("function") if isinstance(function, dict) else function
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

    raise NotImplementedError(
        f"from_polars: function {name!r} not implemented yet."
    )
