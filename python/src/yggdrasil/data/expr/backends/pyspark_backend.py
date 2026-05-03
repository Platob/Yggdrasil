"""pyspark.sql.Column emitter + best-effort lifter.

:func:`to_pyspark` translates the AST to a
:class:`pyspark.sql.Column`. Output is suitable as the predicate
of :meth:`pyspark.sql.DataFrame.filter` /
:meth:`pyspark.sql.DataFrame.where`. Type-aware literals route
through :func:`pyspark.sql.functions.lit` plus a cast when the
:class:`Literal` carries a pinned :class:`DataType` — keeps Spark
from inferring the wrong dtype on ambiguous Python values
(``date`` becoming ``timestamp``, etc.).

:func:`from_pyspark` falls back to the column's underlying
``Expression``-string representation. Spark only exposes
``Column.expr.json()`` / ``Column.expr.toString()`` reliably; the
lifter parses the SQL produced by ``Column.expr.sql()`` via the
SQL backend's :func:`from_sql`. That reuses one parser and means
this lifter is best-effort but consistent with what the SQL
backend handles.
"""

from __future__ import annotations

from typing import Any

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


__all__ = ["to_pyspark", "from_pyspark"]


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------


def to_pyspark(expr: Expression):
    """Emit *expr* as a :class:`pyspark.sql.Column`."""
    from pyspark.sql import functions as F

    return _emit(expr, F)


def _emit(expr: Expression, F):  # type: ignore[no-untyped-def]
    if isinstance(expr, Column):
        ref = F.col(expr.name)
        if expr.alias:
            return F.col(f"{expr.alias}.{expr.name}")
        return ref

    if isinstance(expr, Literal):
        lit = F.lit(expr.value)
        if expr.dtype is not None:
            try:
                spark_dt = expr.dtype.to_spark()
            except Exception:
                spark_dt = None
            if spark_dt is not None:
                lit = lit.cast(spark_dt)
        return lit

    if isinstance(expr, Comparison):
        return _emit_comparison(expr, F)

    if isinstance(expr, Logical):
        return _emit_logical(expr, F)

    if isinstance(expr, Not):
        return ~_emit(expr.operand, F)

    if isinstance(expr, Between):
        target = _emit(expr.target, F)
        low = _emit(expr.low, F)
        high = _emit(expr.high, F)
        between = target.between(low, high)
        return ~between if expr.negated else between

    if isinstance(expr, InList):
        return _emit_inlist(expr, F)

    if isinstance(expr, IsNull):
        target = _emit(expr.target, F)
        return target.isNotNull() if expr.negated else target.isNull()

    if isinstance(expr, Like):
        return _emit_like(expr, F)

    if isinstance(expr, Cast):
        target = _emit(expr.operand, F)
        try:
            spark_dt = expr.dtype.to_spark()
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"Cannot translate cast target {expr.dtype!r} to spark."
            ) from exc
        return target.cast(spark_dt)

    if isinstance(expr, Arithmetic):
        return _emit_arithmetic(expr, F)

    raise NotImplementedError(
        f"pyspark backend does not implement node type {type(expr).__name__}."
    )


def _emit_comparison(expr: Comparison, F):  # type: ignore[no-untyped-def]
    left = _emit(expr.left, F)
    right = _emit(expr.right, F)
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


def _emit_logical(expr: Logical, F):  # type: ignore[no-untyped-def]
    operands = [_emit(o, F) for o in expr.operands]
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


def _emit_inlist(expr: InList, F):  # type: ignore[no-untyped-def]
    target = _emit(expr.target, F)
    if not expr.values:
        if expr.includes_null:
            null_clause = target.isNull()
            return ~null_clause if expr.negated else null_clause
        return F.lit(False) if not expr.negated else F.lit(True)
    base = target.isin(*list(expr.values))
    if expr.includes_null:
        if expr.negated:
            return (~base) & target.isNotNull()
        return base | target.isNull()
    return ~base if expr.negated else base


def _emit_like(expr: Like, F):  # type: ignore[no-untyped-def]
    target = _emit(expr.target, F)
    if expr.case_insensitive:
        # Spark's Column.ilike is available since 3.3; fall back
        # to UPPER-both-sides for older builds.
        if hasattr(target, "ilike"):
            match = target.ilike(expr.pattern)
        else:
            match = F.upper(target).like(expr.pattern.upper())
    else:
        match = target.like(expr.pattern)
    return ~match if expr.negated else match


def _emit_arithmetic(expr: Arithmetic, F):  # type: ignore[no-untyped-def]
    left = _emit(expr.left, F)
    right = _emit(expr.right, F)
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
# Lift — round-trip through SQL
# ---------------------------------------------------------------------------


def from_pyspark(spark_col) -> Expression:  # type: ignore[no-untyped-def]
    """Lift a :class:`pyspark.sql.Column` back into our AST.

    Strategy: render the column to SQL via Spark's own
    ``Column.expr.sql()`` (or ``str(col)``-style fallback), then
    parse it through the SQL backend's :func:`from_sql`. That keeps
    one parser instead of two and means anything :func:`from_sql`
    handles is supported here too.

    Raises :class:`NotImplementedError` when Spark won't surface
    the SQL representation (e.g. UDFs, ``Column.expr`` private
    APIs that aren't stable across Spark versions).
    """
    from .sql import Dialect, from_sql

    sql = _spark_column_to_sql(spark_col)
    if sql is None:
        raise NotImplementedError(
            "from_pyspark: cannot extract SQL from this Column. "
            "Spark's expr-to-SQL bridge is private; if you hit "
            "this, capture the original yggdrasil Expression instead."
        )
    return from_sql(sql, dialect=Dialect.DATABRICKS)


def _spark_column_to_sql(spark_col: Any) -> "str | None":
    """Best-effort SQL rendering of a Spark Column.

    Tries the documented surface first (``_jc.toString()`` is the
    private but stable path Spark uses for ``__repr__``), then
    falls back to ``str(spark_col)``. Returns ``None`` when
    nothing usable came back.
    """
    jc = getattr(spark_col, "_jc", None)
    if jc is not None:
        try:
            text = jc.expr().sql()
            if text:
                return str(text)
        except Exception:
            pass
        try:
            text = jc.toString()
            if text:
                return str(text)
        except Exception:
            pass
    text = str(spark_col)
    return text if text else None
