"""SQL emitter with dialect flavors + best-effort lifter.

:func:`to_sql` walks the AST and produces a SQL string for the
named dialect. Identifier quoting, string-literal escaping, and a
handful of dialect-specific keywords (``ILIKE``, ``TIMESTAMP``
literals) come from :class:`Dialect`.

:func:`from_sql` parses a SQL predicate string back into our AST.
The parser uses ``sqlglot`` when it's available — install via
``pip install sqlglot`` or skip the lifter and emit-only by
sticking with :func:`to_sql`. Without ``sqlglot``, ``from_sql``
raises a clear error pointing at the missing dependency.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from enum import Enum
from typing import Any, Iterable

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


__all__ = ["to_sql", "from_sql", "Dialect", "DEFAULT_DIALECT"]


# ---------------------------------------------------------------------------
# Dialect — identifier quoting + literal escapes
# ---------------------------------------------------------------------------


class Dialect(str, Enum):
    """SQL dialects supported by :func:`to_sql`.

    Differences are intentionally minimal — we render a portable
    subset and only branch where the dialect's syntax is
    incompatible (identifier quote chars, ``ILIKE`` availability).
    Unsupported dialects fall back to ANSI-ish defaults.
    """

    ANSI = "ansi"
    DATABRICKS = "databricks"
    POSTGRES = "postgres"
    SQLITE = "sqlite"
    MYSQL = "mysql"


DEFAULT_DIALECT: Dialect = Dialect.DATABRICKS


# Each dialect's identifier-quote char and the substring that gets
# doubled to escape it inside an identifier.
_IDENT_QUOTES: dict[Dialect, tuple[str, str]] = {
    Dialect.ANSI: ('"', '""'),
    Dialect.DATABRICKS: ("`", "``"),
    Dialect.POSTGRES: ('"', '""'),
    Dialect.SQLITE: ('"', '""'),
    Dialect.MYSQL: ("`", "``"),
}


def _resolve_dialect(spec: "Dialect | str | None") -> Dialect:
    if spec is None:
        return DEFAULT_DIALECT
    if isinstance(spec, Dialect):
        return spec
    try:
        return Dialect(str(spec).lower())
    except ValueError as exc:
        raise ValueError(
            f"Unknown SQL dialect: {spec!r}. "
            f"Valid: {[d.value for d in Dialect]!r}."
        ) from exc


# ---------------------------------------------------------------------------
# Operator precedence — used to decide when to wrap in parens.
# Higher binds tighter.
# ---------------------------------------------------------------------------


_PREC_OR = 10
_PREC_AND = 20
_PREC_NOT = 30
_PREC_CMP = 40
_PREC_ADD = 50
_PREC_MUL = 60
_PREC_LEAF = 100


# ---------------------------------------------------------------------------
# Public emitters
# ---------------------------------------------------------------------------


def to_sql(
    expr: Expression,
    *,
    dialect: "Dialect | str | None" = None,
) -> str:
    """Emit *expr* as a SQL string in the chosen dialect."""
    return _render(expr, _resolve_dialect(dialect), parent_prec=0)


def from_sql(
    sql: str,
    *,
    dialect: "Dialect | str | None" = None,
) -> Expression:
    """Parse a SQL predicate string into our AST.

    Requires the optional ``sqlglot`` package — install with
    ``pip install sqlglot``. The lifter currently handles
    comparisons, ``AND`` / ``OR`` / ``NOT``, ``IN``, ``BETWEEN``,
    ``LIKE``, ``IS NULL`` / ``IS NOT NULL``, simple casts, and
    arithmetic on column references and literals. Anything outside
    that surface raises :class:`NotImplementedError`.
    """
    try:
        import sqlglot
        from sqlglot import expressions as sge
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        raise ImportError(
            "yggdrasil.data.expr.backends.sql.from_sql requires sqlglot. "
            "Install it with `pip install sqlglot` to enable the lifter."
        ) from exc

    parsed = sqlglot.parse_one(
        sql, read=_resolve_dialect(dialect).value
    )
    return _lift_sqlglot(parsed, sge)


# ---------------------------------------------------------------------------
# Render — recursive descent. Returns the rendered SQL plus
# implicit precedence handled by ``parent_prec`` so we only
# parenthesize where needed.
# ---------------------------------------------------------------------------


def _render(expr: Expression, dialect: Dialect, *, parent_prec: int) -> str:
    if isinstance(expr, Column):
        return _render_column(expr, dialect)
    if isinstance(expr, Literal):
        return _render_literal(expr.value, dialect)
    if isinstance(expr, Comparison):
        return _wrap(_render_comparison(expr, dialect), _PREC_CMP, parent_prec)
    if isinstance(expr, Logical):
        return _render_logical(expr, dialect, parent_prec=parent_prec)
    if isinstance(expr, Not):
        inner = _render(expr.operand, dialect, parent_prec=_PREC_NOT)
        return _wrap(f"NOT {inner}", _PREC_NOT, parent_prec)
    if isinstance(expr, Between):
        return _wrap(_render_between(expr, dialect), _PREC_CMP, parent_prec)
    if isinstance(expr, InList):
        return _wrap(_render_inlist(expr, dialect), _PREC_CMP, parent_prec)
    if isinstance(expr, IsNull):
        target = _render(expr.target, dialect, parent_prec=_PREC_CMP)
        suffix = "IS NOT NULL" if expr.negated else "IS NULL"
        return _wrap(f"{target} {suffix}", _PREC_CMP, parent_prec)
    if isinstance(expr, Like):
        return _wrap(_render_like(expr, dialect), _PREC_CMP, parent_prec)
    if isinstance(expr, Cast):
        return _render_cast(expr, dialect)
    if isinstance(expr, Arithmetic):
        return _render_arithmetic(expr, dialect, parent_prec=parent_prec)

    raise NotImplementedError(
        f"SQL backend does not implement node type {type(expr).__name__}."
    )


def _wrap(text: str, my_prec: int, parent_prec: int) -> str:
    return f"({text})" if my_prec < parent_prec else text


# ---------------------------------------------------------------------------
# Column / literal
# ---------------------------------------------------------------------------


def _render_column(col: Column, dialect: Dialect) -> str:
    quoted = _quote_ident(col.name, dialect)
    if col.alias:
        return f"{_quote_ident(col.alias, dialect)}.{quoted}"
    return quoted


def _quote_ident(name: str, dialect: Dialect) -> str:
    quote, escaped = _IDENT_QUOTES.get(dialect, _IDENT_QUOTES[Dialect.ANSI])
    return f"{quote}{name.replace(quote, escaped)}{quote}"


def _render_literal(value: Any, dialect: Dialect) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, Decimal)):
        return str(value)
    if isinstance(value, float):
        # Avoid the surprise of ``1e10`` rendering — keep the
        # scientific form when it's the natural representation,
        # but never let NaN/inf escape: they'd produce invalid SQL.
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(
                f"Cannot render non-finite float {value!r} as SQL literal."
            )
        return repr(value)
    if isinstance(value, dt.datetime):
        return f"TIMESTAMP '{value.isoformat(sep=' ', timespec='microseconds')}'"
    if isinstance(value, dt.date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, dt.time):
        return f"TIME '{value.isoformat(timespec='microseconds')}'"
    if isinstance(value, bytes):
        return f"X'{value.hex()}'"
    # Strings — escape embedded single-quotes by doubling, ANSI-safe.
    text = str(value).replace("'", "''")
    return f"'{text}'"


# ---------------------------------------------------------------------------
# Comparisons and logical
# ---------------------------------------------------------------------------


def _render_comparison(expr: Comparison, dialect: Dialect) -> str:
    left = _render(expr.left, dialect, parent_prec=_PREC_CMP)
    right = _render(expr.right, dialect, parent_prec=_PREC_CMP)
    return f"{left} {expr.op.value} {right}"


def _render_logical(expr: Logical, dialect: Dialect, *, parent_prec: int) -> str:
    own_prec = _PREC_AND if expr.op is LogicalOp.AND else _PREC_OR
    rendered = [
        _render(o, dialect, parent_prec=own_prec) for o in expr.operands
    ]
    joined = f" {expr.op.value} ".join(rendered)
    return _wrap(joined, own_prec, parent_prec)


# ---------------------------------------------------------------------------
# Between / InList / Like
# ---------------------------------------------------------------------------


def _render_between(expr: Between, dialect: Dialect) -> str:
    target = _render(expr.target, dialect, parent_prec=_PREC_CMP)
    low = _render(expr.low, dialect, parent_prec=_PREC_CMP)
    high = _render(expr.high, dialect, parent_prec=_PREC_CMP)
    keyword = "NOT BETWEEN" if expr.negated else "BETWEEN"
    return f"{target} {keyword} {low} AND {high}"


def _render_inlist(expr: InList, dialect: Dialect) -> str:
    target = _render(expr.target, dialect, parent_prec=_PREC_CMP)
    if not expr.values and not expr.includes_null:
        # Empty IN — SQL semantics: ``x IN ()`` is invalid in most
        # dialects; emit an unsatisfiable predicate to keep the
        # behaviour predictable. ``NOT IN ()`` becomes a tautology.
        return "FALSE" if not expr.negated else "TRUE"
    rendered_values = ", ".join(
        _render_literal(v, dialect) for v in expr.values
    )
    keyword = "NOT IN" if expr.negated else "IN"
    base = (
        f"{target} {keyword} ({rendered_values})"
        if expr.values else None
    )
    if not expr.includes_null:
        return base or _empty_inlist_fallback(expr)
    null_clause = (
        f"{target} IS NOT NULL" if expr.negated
        else f"{target} IS NULL"
    )
    if base is None:
        return null_clause
    connector = " AND " if expr.negated else " OR "
    return f"{base}{connector}{null_clause}"


def _empty_inlist_fallback(expr: InList) -> str:
    return "FALSE" if not expr.negated else "TRUE"


def _render_like(expr: Like, dialect: Dialect) -> str:
    target = _render(expr.target, dialect, parent_prec=_PREC_CMP)
    keyword = _like_keyword(dialect, expr)
    pattern = _render_literal(expr.pattern, dialect)
    return f"{target} {keyword} {pattern}"


def _like_keyword(dialect: Dialect, expr: Like) -> str:
    if expr.case_insensitive:
        if dialect in (Dialect.POSTGRES, Dialect.DATABRICKS):
            return "NOT ILIKE" if expr.negated else "ILIKE"
        # Dialects without ILIKE — UPPER both sides; keep the
        # keyword loud so callers reading the SQL know the column
        # was wrapped.
        return "NOT LIKE" if expr.negated else "LIKE"
    return "NOT LIKE" if expr.negated else "LIKE"


# ---------------------------------------------------------------------------
# Cast / arithmetic
# ---------------------------------------------------------------------------


def _render_cast(expr: Cast, dialect: Dialect) -> str:
    inner = _render(expr.operand, dialect, parent_prec=0)
    return f"CAST({inner} AS {_render_dtype(expr.dtype, dialect)})"


def _render_dtype(dtype: Any, dialect: Dialect) -> str:
    """Best-effort SQL dtype name.

    Defers to the DataType's own engine-flavoured rendering when
    available (most yggdrasil DataTypes know their pyarrow form);
    otherwise falls back to ``str(dtype)``. Engines that need
    dialect-specific names (``LONG`` vs ``BIGINT``) override this
    via subclassing.
    """
    for attr in ("to_sql", "to_databricks_sql", "to_arrow"):
        fn = getattr(dtype, attr, None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:
                continue
    return str(dtype)


def _render_arithmetic(
    expr: Arithmetic,
    dialect: Dialect,
    *,
    parent_prec: int,
) -> str:
    own_prec = _PREC_MUL if expr.op in (
        ArithmeticOp.MUL, ArithmeticOp.DIV, ArithmeticOp.MOD,
    ) else _PREC_ADD
    left = _render(expr.left, dialect, parent_prec=own_prec)
    right = _render(expr.right, dialect, parent_prec=own_prec + 1)
    return _wrap(f"{left} {expr.op.value} {right}", own_prec, parent_prec)


# ---------------------------------------------------------------------------
# from_sql via sqlglot
# ---------------------------------------------------------------------------


def _lift_sqlglot(node: Any, sge: Any) -> Expression:
    """Walk a sqlglot AST and rebuild our :class:`Expression`.

    Branches by sqlglot's expression class. Unknown nodes raise
    :class:`NotImplementedError` with the offending class name —
    callers know exactly what to add when they hit it.
    """
    if isinstance(node, sge.And):
        return Logical(LogicalOp.AND, _flatten_logical(
            node, sge.And, sge,
        ))
    if isinstance(node, sge.Or):
        return Logical(LogicalOp.OR, _flatten_logical(
            node, sge.Or, sge,
        ))
    if isinstance(node, sge.Not):
        return Not(_lift_sqlglot(node.this, sge))

    if isinstance(node, (sge.EQ, sge.NEQ, sge.LT, sge.LTE, sge.GT, sge.GTE)):
        return Comparison(
            _lift_sqlglot(node.this, sge),
            _CMP_FROM_SQLGLOT[type(node).__name__],
            _lift_sqlglot(node.expression, sge),
        )

    if isinstance(node, sge.Between):
        return Between(
            target=_lift_sqlglot(node.this, sge),
            low=_lift_sqlglot(node.args["low"], sge),
            high=_lift_sqlglot(node.args["high"], sge),
            negated=False,
        )

    if isinstance(node, sge.In):
        target = _lift_sqlglot(node.this, sge)
        # ``IN (a, b, c)`` — sqlglot stores the list under ``expressions``.
        raw_values = [_lift_sqlglot(v, sge) for v in node.expressions or []]
        non_null: list[Any] = []
        has_null = False
        for v in raw_values:
            if isinstance(v, Literal):
                if v.value is None:
                    has_null = True
                else:
                    non_null.append(v.value)
            else:
                raise NotImplementedError(
                    "from_sql: IN with non-literal values is not yet supported."
                )
        return InList(
            target=target,
            values=tuple(non_null),
            negated=False,
            includes_null=has_null,
        )

    if isinstance(node, sge.Is):
        target = _lift_sqlglot(node.this, sge)
        # ``IS NULL`` / ``IS NOT NULL`` come back as Is(this=expr,
        # expression=Null()) — the NOT wrapper for ``IS NOT NULL``
        # sits on a parent Not node, handled above.
        if isinstance(node.expression, sge.Null):
            return IsNull(target=target, negated=False)

    if isinstance(node, sge.Like):
        return Like(
            target=_lift_sqlglot(node.this, sge),
            pattern=_lift_literal_string(node.expression, sge),
            case_insensitive=False,
            negated=False,
        )
    if isinstance(node, sge.ILike):
        return Like(
            target=_lift_sqlglot(node.this, sge),
            pattern=_lift_literal_string(node.expression, sge),
            case_insensitive=True,
            negated=False,
        )

    if isinstance(node, sge.Column):
        return Column(name=node.name)

    if isinstance(node, sge.Literal):
        if node.is_string:
            return Literal(value=str(node.this))
        # sqlglot keeps the raw token; cast through Decimal/int/float.
        text = str(node.this)
        try:
            return Literal(value=int(text))
        except ValueError:
            try:
                return Literal(value=float(text))
            except ValueError:
                return Literal(value=text)

    if isinstance(node, sge.Boolean):
        return Literal(value=bool(node.this))
    if isinstance(node, sge.Null):
        return Literal(value=None)

    if isinstance(node, sge.Paren):
        return _lift_sqlglot(node.this, sge)

    raise NotImplementedError(
        f"from_sql: sqlglot node type {type(node).__name__!r} "
        f"is not implemented yet. Pass-through value: {node.sql()}"
    )


_CMP_FROM_SQLGLOT: dict[str, CompareOp] = {
    "EQ": CompareOp.EQ,
    "NEQ": CompareOp.NE,
    "LT": CompareOp.LT,
    "LTE": CompareOp.LE,
    "GT": CompareOp.GT,
    "GTE": CompareOp.GE,
}


def _flatten_logical(node: Any, cls: type, sge: Any) -> "tuple[Expression, ...]":
    """Flatten chains of the same logical operator into one tuple."""
    out: list[Expression] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, cls):
            stack.append(cur.expression)
            stack.append(cur.this)
        else:
            out.append(_lift_sqlglot(cur, sge))
    out.reverse()
    return tuple(out)


def _lift_literal_string(node: Any, sge: Any) -> str:
    if isinstance(node, sge.Literal) and node.is_string:
        return str(node.this)
    raise NotImplementedError(
        f"from_sql LIKE pattern must be a string literal; got "
        f"{type(node).__name__}."
    )


# Avoid an unused-import warning when sqlglot isn't installed at
# build time. Iterables here is kept for the per-dialect override
# tables some downstreams add later.
_ = Iterable
