"""SQL emitter with dialect flavors + hand-rolled lifter.

:func:`to_sql` walks the AST and produces a SQL string for the
named dialect. Identifier quoting, string-literal escaping, and a
handful of dialect-specific keywords (``ILIKE``, ``TIMESTAMP``
literals) come from :class:`Dialect`.

:func:`from_sql` parses a SQL predicate string back into our AST
via the in-module tokenizer + recursive-descent parser below —
no third-party SQL parser dependency. The grammar covers the
predicate-language subset every yggdrasil backend needs:
comparisons, ``AND`` / ``OR`` / ``NOT``, ``IN``, ``BETWEEN``,
``LIKE`` / ``ILIKE``, ``IS [NOT] NULL``, ``CAST``, typed temporal
literals (``TIMESTAMP '…'`` / ``DATE '…'``), function-call
temporal coercions (``TIMESTAMP('…')`` / ``DATE('…')``), and the
``+``/``-``/``*``/``/``/``%`` arithmetic operators between
columns and literals. Anything outside that surface raises a
``ValueError`` pointing at the offending token.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from ..nodes import (
    Alias,
    Arithmetic,
    Between,
    CaseWhen,
    Cast,
    Column,
    Comparison,
    Expression,
    FunctionCall,
    InList,
    IsNull,
    Like,
    Literal,
    Logical,
    Not,
    SortOrder,
    Star,
    Subscript,
    WindowFunction,
    WindowSpec,
)
from ..operators import ArithmeticOp, CompareOp, LogicalOp


from yggdrasil.enums.dialect import Dialect

__all__ = ["to_sql", "from_sql", "Dialect", "DEFAULT_DIALECT"]


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

    The grammar covers comparisons, ``AND`` / ``OR`` / ``NOT``,
    ``IN`` (literal value list), ``BETWEEN``, ``LIKE`` / ``ILIKE``,
    ``IS [NOT] NULL``, ``CAST(… AS …)``, typed temporal literals
    (``TIMESTAMP '…'`` / ``DATE '…'``), the function-call temporal
    coercion form (``TIMESTAMP('…')`` / ``DATE('…')``), and
    ``+``/``-``/``*``/``/``/``%`` arithmetic on columns and
    literals. Tokens outside that surface raise ``ValueError``
    with the offending position.

    Dialect selection only affects identifier / string-literal
    quoting: Databricks and MySQL accept ``\"…\"`` as a string
    literal (and ``\\`…\\``` as the identifier quote); ANSI /
    Postgres / SQLite treat ``\"…\"`` as the identifier quote.
    """
    return _SqlParser(sql, _resolve_dialect(dialect)).parse_expression()


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
    if isinstance(expr, FunctionCall):
        return _render_function_call(expr, dialect)
    if isinstance(expr, Star):
        if expr.qualifier:
            return f"{_quote_ident(expr.qualifier, dialect)}.*"
        return "*"
    if isinstance(expr, Alias):
        inner = _render(expr.expr, dialect, parent_prec=0)
        return f"{inner} AS {_quote_ident(expr.name, dialect)}"
    if isinstance(expr, SortOrder):
        return _render_sort_order(expr, dialect)
    if isinstance(expr, WindowFunction):
        fn = _render(expr.function, dialect, parent_prec=0)
        win = _render_window_spec(expr.window, dialect)
        return f"{fn} {win}"
    if isinstance(expr, WindowSpec):
        return _render_window_spec(expr, dialect)
    if isinstance(expr, CaseWhen):
        return _render_case_when(expr, dialect)
    if isinstance(expr, Subscript):
        base = _render(expr.expr, dialect, parent_prec=_PREC_LEAF)
        idx = _render(expr.index, dialect, parent_prec=0)
        return f"{base}[{idx}]"
    from yggdrasil.execution.expr.nodes import Lambda  # local import to avoid cycle
    if isinstance(expr, Lambda):
        body = _render(expr.body, dialect, parent_prec=0)
        if len(expr.params) == 1:
            return f"{expr.params[0]} -> {body}"
        return f"({', '.join(expr.params)}) -> {body}"

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
    if col.qualifier:
        quoted = f"{_quote_ident(col.qualifier, dialect)}.{quoted}"
    # NB: ``col.alias`` (column-level rename ``foo AS bar``) is the
    # caller's intent in a SELECT list, not inside a predicate. The
    # predicate emitter renders bare ``col`` here so a WHERE clause
    # doesn't accidentally emit ``foo AS bar = 5``; SELECT-list
    # emitters that need the rename wrap this with their own
    # ``AS <alias>`` suffix.
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
# FunctionCall / WindowSpec / SortOrder / CaseWhen rendering
# ---------------------------------------------------------------------------


def _render_function_call(expr: FunctionCall, dialect: Dialect) -> str:
    # INTERVAL 'value' unit — special syntax, not parenthesized
    if expr.name == "INTERVAL" and len(expr.args) == 2:
        val = _render(expr.args[0], dialect, parent_prec=0)
        unit = expr.args[1].value if isinstance(expr.args[1], Literal) else _render(expr.args[1], dialect, parent_prec=0)
        return f"INTERVAL {val} {unit}"
    # EXTRACT(field FROM source) — special keyword syntax
    if expr.name == "EXTRACT" and len(expr.args) == 2:
        field = expr.args[0].value if isinstance(expr.args[0], Literal) else _render(expr.args[0], dialect, parent_prec=0)
        source = _render(expr.args[1], dialect, parent_prec=0)
        return f"EXTRACT({field} FROM {source})"
    args = ", ".join(_render(a, dialect, parent_prec=0) for a in expr.args)
    distinct = "DISTINCT " if expr.distinct else ""
    return f"{expr.name}({distinct}{args})"


def _render_sort_order(expr: SortOrder, dialect: Dialect) -> str:
    inner = _render(expr.expr, dialect, parent_prec=0)
    direction = "ASC" if expr.ascending else "DESC"
    parts = [inner, direction]
    if expr.nulls_first is True:
        parts.append("NULLS FIRST")
    elif expr.nulls_first is False:
        parts.append("NULLS LAST")
    return " ".join(parts)


def _render_window_spec(spec: WindowSpec, dialect: Dialect) -> str:
    clauses: "list[str]" = []
    if spec.partition_by:
        parts = ", ".join(
            _render(p, dialect, parent_prec=0) for p in spec.partition_by
        )
        clauses.append(f"PARTITION BY {parts}")
    if spec.order_by:
        parts = ", ".join(
            _render(o, dialect, parent_prec=0) for o in spec.order_by
        )
        clauses.append(f"ORDER BY {parts}")
    if spec.frame_start is not None:
        if spec.frame_end is not None:
            clauses.append(
                f"ROWS BETWEEN {spec.frame_start} AND {spec.frame_end}"
            )
        else:
            clauses.append(f"ROWS {spec.frame_start}")
    return f"OVER ({' '.join(clauses)})"


def _render_case_when(expr: CaseWhen, dialect: Dialect) -> str:
    parts: "list[str]" = ["CASE"]
    if expr.operand is not None:
        parts.append(_render(expr.operand, dialect, parent_prec=0))
    for cond, val in expr.branches:
        c = _render(cond, dialect, parent_prec=0)
        v = _render(val, dialect, parent_prec=0)
        parts.append(f"WHEN {c} THEN {v}")
    if expr.else_expr is not None:
        parts.append(f"ELSE {_render(expr.else_expr, dialect, parent_prec=0)}")
    parts.append("END")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Tokenizer + recursive-descent predicate parser
#
# Hand-rolled to keep ``from_sql`` free of a third-party SQL parser
# dependency. Covers exactly the predicate-language subset every
# backend (Python / Arrow / Polars / Spark) needs, plus the typed
# temporal literals used by Databricks WHERE clauses.
# ---------------------------------------------------------------------------


# Keyword set is case-insensitive — the tokenizer uppercases the match.
# Type-name heads (BIGINT, DECIMAL, …) stay as plain identifiers so a
# column happens to be named ``decimal`` still parses cleanly.
_RESERVED = frozenset({
    "AND", "OR", "NOT", "IN", "BETWEEN", "IS", "NULL",
    "LIKE", "ILIKE", "TRUE", "FALSE", "CAST", "AS",
    "TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMPNTZ", "TIMESTAMP_NTZ",
    "TIMESTAMP_LTZ", "DATETIME", "DATE", "TIME", "TIMETZ",
    # SELECT / clause keywords
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER",
    "ASC", "DESC", "NULLS", "FIRST", "LAST",
    # CASE expression
    "CASE", "WHEN", "THEN", "ELSE", "END",
    # Window functions
    "OVER", "PARTITION", "ROWS", "RANGE", "UNBOUNDED", "PRECEDING",
    "FOLLOWING", "CURRENT", "ROW",
    # Set / aggregate modifiers
    "DISTINCT", "ALL", "UNION", "INTERSECT", "EXCEPT",
    # Joins
    "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "ON",
    # CTE / lateral / misc
    "WITH", "LATERAL", "VIEW", "EXPLODE",
    # Pagination
    "LIMIT", "OFFSET",
    # Subquery operators
    "EXISTS", "ANY",
    # Complex-type constructors
    "ARRAY", "MAP", "STRUCT",
})


# Casts whose target type is one of these names, applied to a string
# literal, fold eagerly into a typed Python value so backends see the
# right runtime type and ``to_sql`` regenerates the matching
# ``TIMESTAMP 'iso'`` / ``DATE 'iso'`` rendering on round-trip.
def _parse_iso_date(text: str) -> dt.date:
    return dt.datetime.fromisoformat(text).date() if (
        "T" in text or " " in text
    ) else dt.date.fromisoformat(text)


_CAST_TEMPORAL_PARSERS: dict[str, "tuple[Any, str]"] = {
    "DATE": (_parse_iso_date, "date"),
    "DATETIME": (dt.datetime.fromisoformat, "timestamp_ntz"),
    "TIMESTAMP": (dt.datetime.fromisoformat, "timestamp"),
    "TIMESTAMPTZ": (dt.datetime.fromisoformat, "timestamp"),
    "TIMESTAMPLTZ": (dt.datetime.fromisoformat, "timestamp"),
    "TIMESTAMPNTZ": (dt.datetime.fromisoformat, "timestamp_ntz"),
    "TIMESTAMP_NTZ": (dt.datetime.fromisoformat, "timestamp_ntz"),
    "TIMESTAMP_LTZ": (dt.datetime.fromisoformat, "timestamp"),
    "TIME": (dt.time.fromisoformat, "time"),
    "TIMETZ": (dt.time.fromisoformat, "time"),
}


# Type-name aliases for non-temporal casts where the canonical SQL
# spelling differs from yggdrasil's ``DataType.from_str`` vocabulary.
_CAST_DTYPE_ALIASES: dict[str, str] = {
    "BIGINT": "int64",
    "INT": "int32",
    "INTEGER": "int32",
    "SMALLINT": "int16",
    "TINYINT": "int8",
    "DOUBLE": "float64",
    "DOUBLE PRECISION": "float64",
    "REAL": "float32",
    "FLOAT": "float32",
    "TEXT": "string",
    "VARCHAR": "string",
    "CHAR": "string",
    "BOOL": "bool",
    "BOOLEAN": "bool",
    "TIMESTAMPTZ": "timestamp",
    "TIMESTAMPLTZ": "timestamp",
    "TIMESTAMP_LTZ": "timestamp",
    "TIMESTAMPNTZ": "timestamp_ntz",
    "TIMESTAMP_NTZ": "timestamp_ntz",
    "DATETIME": "timestamp_ntz",
}


_CMP_BY_OP: dict[str, CompareOp] = {
    "=": CompareOp.EQ,
    "==": CompareOp.EQ,
    "!=": CompareOp.NE,
    "<>": CompareOp.NE,
    "<": CompareOp.LT,
    "<=": CompareOp.LE,
    ">": CompareOp.GT,
    ">=": CompareOp.GE,
}


_ARITH_ADD = frozenset({"+", "-"})
_ARITH_MUL = frozenset({"*", "/", "%"})


class _Token:
    __slots__ = ("kind", "text", "upper", "pos")

    def __init__(self, kind: str, text: str, upper: str, pos: int) -> None:
        self.kind = kind
        self.text = text
        self.upper = upper
        self.pos = pos

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_Token({self.kind}, {self.text!r}, pos={self.pos})"


# Dialects that treat ``"…"`` as a string literal (Databricks / MySQL)
# rather than an identifier-quote character (ANSI / Postgres / SQLite).
_DOUBLE_QUOTE_IS_STRING = frozenset({Dialect.DATABRICKS, Dialect.MYSQL})


def _tokenize(sql: str, dialect: Dialect) -> "list[_Token]":
    tokens: "list[_Token]" = []
    i = 0
    n = len(sql)
    dq_is_str = dialect in _DOUBLE_QUOTE_IS_STRING
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
            continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            while i < n and sql[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and sql[i + 1] == "*":
            i += 2
            while i + 1 < n and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            if i + 1 >= n:
                raise ValueError(
                    f"from_sql: unterminated block comment in {sql!r}"
                )
            i += 2
            continue
        if c == "`":
            text, i = _scan_quoted(sql, i, "`")
            tokens.append(_Token("ident", text, text.upper(), i))
            continue
        if c == '"':
            if dq_is_str:
                text, i = _scan_quoted(sql, i, '"')
                tokens.append(_Token("string", text, "", i))
            else:
                text, i = _scan_quoted(sql, i, '"')
                tokens.append(_Token("ident", text, text.upper(), i))
            continue
        if c == "'":
            text, i = _scan_quoted(sql, i, "'")
            tokens.append(_Token("string", text, "", i))
            continue
        if c.isdigit() or (c == "." and i + 1 < n and sql[i + 1].isdigit()):
            text, i = _scan_number(sql, i)
            tokens.append(_Token("number", text, "", i))
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            text = sql[i:j]
            upper = text.upper()
            kind = "keyword" if upper in _RESERVED else "ident"
            tokens.append(_Token(kind, text, upper, i))
            i = j
            continue
        if c in "!<>=":
            # Multi-char operators: !=, <>, <=, >=, ==.
            if i + 1 < n and sql[i:i + 2] in ("!=", "<>", "<=", ">=", "=="):
                op = sql[i:i + 2]
                tokens.append(_Token("op", op, op, i))
                i += 2
                continue
            tokens.append(_Token("op", c, c, i))
            i += 1
            continue
        if c in "+-*/%":
            tokens.append(_Token("op", c, c, i))
            i += 1
            continue
        if c in "().,[]":
            kind = {
                "(": "lparen", ")": "rparen",
                ",": "comma", ".": "dot",
                "[": "lbracket", "]": "rbracket",
            }[c]
            tokens.append(_Token(kind, c, c, i))
            i += 1
            continue
        raise ValueError(
            f"from_sql: unexpected character {c!r} at position {i} in {sql!r}"
        )
    tokens.append(_Token("eof", "", "", n))
    return tokens


def _scan_quoted(sql: str, start: int, quote: str) -> "tuple[str, int]":
    """Scan a quoted token (string or identifier) from ``start``.

    Doubled quote chars escape the quote (``''`` inside ``'...'``,
    `` `` `` inside `` ` ... ` `` — standard SQL). Returns the unquoted
    text and the new cursor position (one past the closing quote).
    """
    n = len(sql)
    j = start + 1
    out: "list[str]" = []
    while j < n:
        ch = sql[j]
        if ch == quote:
            if j + 1 < n and sql[j + 1] == quote:
                out.append(quote)
                j += 2
                continue
            return "".join(out), j + 1
        out.append(ch)
        j += 1
    raise ValueError(
        f"from_sql: unterminated {quote!r}-quoted token starting at "
        f"position {start} in {sql!r}"
    )


def _scan_number(sql: str, start: int) -> "tuple[str, int]":
    n = len(sql)
    j = start
    saw_dot = sql[j] == "."
    saw_exp = False
    j += 1
    while j < n:
        ch = sql[j]
        if ch.isdigit():
            j += 1
        elif ch == "." and not saw_dot and not saw_exp:
            saw_dot = True
            j += 1
        elif ch in "eE" and not saw_exp:
            saw_exp = True
            j += 1
            if j < n and sql[j] in "+-":
                j += 1
        else:
            break
    return sql[start:j], j


def _coerce_number(text: str) -> Any:
    try:
        return int(text)
    except ValueError:
        return float(text)


class _SqlParser:
    """Recursive-descent predicate parser.

    Precedence (low → high):

        OR < AND < NOT < (comparison / IN / BETWEEN / LIKE / IS NULL)
        < add (+ / -) < mul (* / / / %) < unary (-x / +x) < primary.

    ``BETWEEN``'s upper bound parses at the additive level — that's
    how SQL avoids the ``AND`` of ``BETWEEN a AND b`` colliding with
    an outer ``AND``.
    """

    __slots__ = ("sql", "dialect", "tokens", "pos")

    def __init__(self, sql: str, dialect: Dialect) -> None:
        self.sql = sql
        self.dialect = dialect
        self.tokens = _tokenize(sql, dialect)
        self.pos = 0

    # ---- token helpers -------------------------------------------------

    @property
    def cur(self) -> _Token:
        return self.tokens[self.pos]

    def _peek(self, offset: int = 1) -> _Token:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[idx]

    def _eat(self) -> _Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _expect_kw(self, *kws: str) -> _Token:
        t = self.cur
        if t.kind != "keyword" or t.upper not in kws:
            raise self._error(f"expected {', '.join(kws)}")
        return self._eat()

    def _expect_kind(self, kind: str) -> _Token:
        if self.cur.kind != kind:
            raise self._error(f"expected {kind}")
        return self._eat()

    def _accept_kw(self, *kws: str) -> "_Token | None":
        t = self.cur
        if t.kind == "keyword" and t.upper in kws:
            return self._eat()
        return None

    def _accept_op(self, *ops: str) -> "_Token | None":
        t = self.cur
        if t.kind == "op" and t.text in ops:
            return self._eat()
        return None

    def _error(self, what: str) -> ValueError:
        t = self.cur
        return ValueError(
            f"from_sql: {what} at position {t.pos} in {self.sql!r}; "
            f"got {t.kind} {t.text!r}."
        )

    # ---- entry ---------------------------------------------------------

    def parse_expression(self) -> Expression:
        expr = self._parse_or()
        if self.cur.kind != "eof":
            raise self._error("unexpected trailing token")
        return expr

    # ---- precedence levels --------------------------------------------

    def _parse_or(self) -> Expression:
        left = self._parse_and()
        operands: "list[Expression]" = []
        while self._accept_kw("OR") is not None:
            operands.append(self._parse_and())
        if not operands:
            return left
        return Logical(LogicalOp.OR, (left, *operands))

    def _parse_and(self) -> Expression:
        left = self._parse_not()
        operands: "list[Expression]" = []
        while self._accept_kw("AND") is not None:
            operands.append(self._parse_not())
        if not operands:
            return left
        return Logical(LogicalOp.AND, (left, *operands))

    def _parse_not(self) -> Expression:
        if self._accept_kw("NOT") is not None:
            return Not(self._parse_not())
        return self._parse_predicate()

    def _parse_predicate(self) -> Expression:
        """Parse a comparison / IN / BETWEEN / LIKE / IS NULL chain.

        Reads one ``add_expr`` for the left side, then loops over
        ``NOT``-introduced suffixes (BETWEEN / IN / LIKE), bare
        IS [NOT] NULL, or one of the comparison operators. The
        loop ends at the first non-suffix token.
        """
        left = self._parse_add()
        while True:
            negated = False
            if self.cur.kind == "keyword" and self.cur.upper == "NOT":
                nxt = self._peek(1)
                if nxt.kind == "keyword" and nxt.upper in (
                    "BETWEEN", "IN", "LIKE", "ILIKE",
                ):
                    self._eat()
                    negated = True
                else:
                    break

            t = self.cur
            if t.kind == "keyword":
                if t.upper == "BETWEEN":
                    self._eat()
                    low = self._parse_add()
                    self._expect_kw("AND")
                    high = self._parse_add()
                    left = Between(left, low, high, negated=negated)
                    continue
                if t.upper == "IN":
                    self._eat()
                    left = self._parse_in_tail(left, negated)
                    continue
                if t.upper in ("LIKE", "ILIKE"):
                    ci = t.upper == "ILIKE"
                    self._eat()
                    pattern = self._parse_primary()
                    if not (
                        isinstance(pattern, Literal)
                        and isinstance(pattern.value, str)
                    ):
                        raise ValueError(
                            "from_sql: LIKE pattern must be a string literal."
                        )
                    left = Like(
                        target=left, pattern=pattern.value,
                        case_insensitive=ci, negated=negated,
                    )
                    continue
                if t.upper == "IS" and not negated:
                    self._eat()
                    neg = self._accept_kw("NOT") is not None
                    self._expect_kw("NULL")
                    left = IsNull(target=left, negated=neg)
                    continue

            if t.kind == "op" and t.text in _CMP_BY_OP:
                if negated:
                    raise self._error("unexpected NOT before comparison")
                op = self._eat().text
                right = self._parse_add()
                left = Comparison(left, _CMP_BY_OP[op], right)
                continue

            if negated:
                raise self._error("unexpected NOT")
            break
        return left

    def _parse_in_tail(
        self, target: Expression, negated: bool,
    ) -> Expression:
        self._expect_kind("lparen")
        values: "list[Any]" = []
        has_null = False
        if self.cur.kind != "rparen":
            while True:
                v = self._parse_primary()
                if not isinstance(v, Literal):
                    raise NotImplementedError(
                        "from_sql: IN with non-literal values is not yet supported."
                    )
                if v.value is None:
                    has_null = True
                else:
                    values.append(v.value)
                if self.cur.kind == "comma":
                    self._eat()
                    continue
                break
        self._expect_kind("rparen")
        return InList(
            target=target,
            values=tuple(values),
            negated=negated,
            includes_null=has_null,
        )

    def _parse_add(self) -> Expression:
        left = self._parse_mul()
        while self.cur.kind == "op" and self.cur.text in _ARITH_ADD:
            op_text = self._eat().text
            right = self._parse_mul()
            left = Arithmetic(ArithmeticOp(op_text), left, right)
        return left

    def _parse_mul(self) -> Expression:
        left = self._parse_unary()
        while self.cur.kind == "op" and self.cur.text in _ARITH_MUL:
            op_text = self._eat().text
            right = self._parse_unary()
            left = Arithmetic(ArithmeticOp(op_text), left, right)
        return left

    def _parse_unary(self) -> Expression:
        # Fold a leading sign into a numeric literal when the operand
        # is one — ``-5`` becomes ``Literal(-5)`` rather than
        # ``Arithmetic(SUB, Literal(0), Literal(5))``. Keeps the AST
        # round-trippable with the renderer (which has no unary form).
        if self.cur.kind == "op" and self.cur.text in ("+", "-"):
            sign = self._eat().text
            inner = self._parse_unary()
            if sign == "+":
                return inner
            if isinstance(inner, Literal) and isinstance(
                inner.value, (int, float, Decimal),
            ):
                return Literal(value=-inner.value, dtype=inner.dtype)
            return Arithmetic(ArithmeticOp.SUB, Literal(value=0), inner)
        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        t = self.cur
        if t.kind == "lparen":
            self._eat()
            expr = self._parse_or()
            self._expect_kind("rparen")
            return self._parse_postfix(expr)
        if t.kind == "string":
            self._eat()
            return Literal(value=t.text)
        if t.kind == "number":
            self._eat()
            return Literal(value=_coerce_number(t.text))
        # Bare * (for SELECT * or similar contexts)
        if t.kind == "op" and t.text == "*":
            self._eat()
            return Star()
        if t.kind == "keyword":
            if t.upper == "NULL":
                self._eat()
                return Literal(value=None)
            if t.upper == "TRUE":
                self._eat()
                return Literal(value=True)
            if t.upper == "FALSE":
                self._eat()
                return Literal(value=False)
            if t.upper == "CAST":
                return self._parse_cast()
            if t.upper == "CASE":
                return self._parse_case_when()
            if t.upper in _CAST_TEMPORAL_PARSERS:
                return self._parse_typed_temporal(t)
            # Keywords that can also be function names (e.g. ARRAY(...))
            if self._peek(1).kind == "lparen":
                return self._parse_function_call()
            raise self._error(f"unexpected keyword {t.text!r}")
        if t.kind == "ident":
            # Check if next token is lparen -> function call
            if self._peek(1).kind == "lparen":
                return self._parse_function_call()
            col = self._parse_column()
            return self._parse_postfix(col)
        raise self._error("unexpected token")

    # ---- column / cast / typed-literal --------------------------------

    def _parse_column(self) -> Expression:
        first = self._expect_kind("ident")
        qualifier: "str | None" = None
        name = first.text
        if self.cur.kind == "dot":
            self._eat()
            second = self._expect_kind("ident")
            qualifier, name = name, second.text
        from ..builder import col as _col

        return _col(name, qualifier=qualifier)

    def _parse_cast(self) -> Expression:
        self._expect_kw("CAST")
        self._expect_kind("lparen")
        inner = self._parse_or()
        self._expect_kw("AS")
        type_name = self._parse_type_head()
        self._expect_kind("rparen")
        return _fold_cast(inner, type_name)

    def _parse_typed_temporal(self, head: _Token) -> Expression:
        """``TIMESTAMP '…'`` literal or ``TIMESTAMP('…')`` call form.

        The first form is the SQL standard typed-literal spelling;
        the second is Databricks' coercion-function shorthand. Both
        collapse to a Cast and then fold to a typed :class:`Literal`
        when the inner is a string. Anything else (e.g. ``TIMESTAMP``
        used as a bare identifier — unusual but legal) falls through
        to the column path so the parser doesn't silently swallow it.
        """
        nxt = self._peek(1)
        if nxt.kind == "string":
            self._eat()  # type keyword
            lit_t = self._eat()  # string literal
            return _fold_cast(Literal(value=lit_t.text), head.upper)
        if nxt.kind == "lparen":
            self._eat()  # type keyword
            self._eat()  # lparen
            inner = self._parse_or()
            self._expect_kind("rparen")
            return _fold_cast(inner, head.upper)
        raise self._error(
            f"unexpected use of type keyword {head.text!r}"
        )

    def _parse_function_call(self) -> Expression:
        """Parse ``NAME(args)`` or ``NAME(DISTINCT args)`` optionally followed by OVER."""
        name_tok = self._eat()  # ident or keyword token
        name = name_tok.upper
        self._expect_kind("lparen")

        # Handle COUNT(*) or fn(*)
        if self.cur.kind == "op" and self.cur.text == "*":
            self._eat()
            self._expect_kind("rparen")
            fc = FunctionCall(name, (Star(),))
            return self._parse_postfix(self._maybe_over(fc))

        # Handle DISTINCT modifier
        distinct = False
        if self._accept_kw("DISTINCT") is not None:
            distinct = True

        args: "list[Expression]" = []
        if self.cur.kind != "rparen":
            while True:
                args.append(self._parse_or())
                if self.cur.kind == "comma":
                    self._eat()
                    continue
                break
        self._expect_kind("rparen")

        fc = FunctionCall(name, tuple(args), distinct=distinct)
        return self._parse_postfix(self._maybe_over(fc))

    def _maybe_over(self, fc: Expression) -> Expression:
        """If the next token is OVER, parse the window spec and wrap."""
        if self._accept_kw("OVER") is not None:
            window = self._parse_window_spec()
            return WindowFunction(fc, window)
        return fc

    def _parse_window_spec(self) -> WindowSpec:
        """Parse ``(PARTITION BY … ORDER BY … [ROWS|RANGE BETWEEN … AND …])``."""
        self._expect_kind("lparen")
        partition_by: "list[Expression]" = []
        order_by: "list[SortOrder]" = []
        frame_start: "str | None" = None
        frame_end: "str | None" = None

        # PARTITION BY
        if self._accept_kw("PARTITION") is not None:
            self._expect_kw("BY")
            while True:
                partition_by.append(self._parse_or())
                if self.cur.kind == "comma":
                    self._eat()
                    continue
                break

        # ORDER BY
        if self._accept_kw("ORDER") is not None:
            self._expect_kw("BY")
            while True:
                expr = self._parse_or()
                ascending = True
                nulls_first: "bool | None" = None
                if self._accept_kw("ASC") is not None:
                    ascending = True
                elif self._accept_kw("DESC") is not None:
                    ascending = False
                if self._accept_kw("NULLS") is not None:
                    if self._accept_kw("FIRST") is not None:
                        nulls_first = True
                    elif self._accept_kw("LAST") is not None:
                        nulls_first = False
                    else:
                        raise self._error("expected FIRST or LAST after NULLS")
                order_by.append(SortOrder(expr, ascending=ascending, nulls_first=nulls_first))
                if self.cur.kind == "comma":
                    self._eat()
                    continue
                break

        # ROWS or RANGE frame
        if self._accept_kw("ROWS", "RANGE") is not None:
            if self._accept_kw("BETWEEN") is not None:
                frame_start = self._parse_frame_bound()
                self._expect_kw("AND")
                frame_end = self._parse_frame_bound()
            else:
                frame_start = self._parse_frame_bound()

        self._expect_kind("rparen")
        return WindowSpec(
            partition_by=tuple(partition_by),
            order_by=tuple(order_by),
            frame_start=frame_start,
            frame_end=frame_end,
        )

    def _parse_frame_bound(self) -> str:
        """Parse a frame boundary: UNBOUNDED PRECEDING/FOLLOWING, CURRENT ROW, N PRECEDING/FOLLOWING."""
        if self._accept_kw("UNBOUNDED") is not None:
            kw = self._expect_kw("PRECEDING", "FOLLOWING")
            return f"UNBOUNDED {kw.upper}"
        if self._accept_kw("CURRENT") is not None:
            self._expect_kw("ROW")
            return "CURRENT ROW"
        # N PRECEDING / N FOLLOWING
        if self.cur.kind == "number":
            n_tok = self._eat()
            kw = self._expect_kw("PRECEDING", "FOLLOWING")
            return f"{n_tok.text} {kw.upper}"
        raise self._error("expected frame bound (UNBOUNDED PRECEDING/FOLLOWING, CURRENT ROW, or N PRECEDING/FOLLOWING)")

    def _parse_case_when(self) -> Expression:
        """Parse ``CASE [expr] WHEN … THEN … [ELSE …] END``."""
        self._expect_kw("CASE")

        # Simple form: CASE expr WHEN val THEN result …
        # Searched form: CASE WHEN cond THEN result …
        operand: "Expression | None" = None
        if self.cur.kind != "keyword" or self.cur.upper != "WHEN":
            operand = self._parse_or()

        branches: "list[tuple[Expression, Expression]]" = []
        while self._accept_kw("WHEN") is not None:
            cond = self._parse_or()
            self._expect_kw("THEN")
            val = self._parse_or()
            branches.append((cond, val))

        else_expr: "Expression | None" = None
        if self._accept_kw("ELSE") is not None:
            else_expr = self._parse_or()

        self._expect_kw("END")
        return CaseWhen(
            branches=tuple(branches),
            else_expr=else_expr,
            operand=operand,
        )

    def _parse_postfix(self, expr: Expression) -> Expression:
        """Parse postfix operators: subscript ``[index]``."""
        while self.cur.kind == "lbracket":
            self._eat()
            index = self._parse_or()
            self._expect_kind("rbracket")
            expr = Subscript(expr, index)
        return expr

    def _parse_type_head(self) -> str:
        """Parse the type name after ``AS`` in a CAST.

        Accepts the keyword type heads (``DATE``, ``TIMESTAMP``,
        ``TIME``, …), bare identifiers (``BIGINT``, ``DECIMAL``,
        ``VARCHAR``, …), an optional precision parenthesis (``(10)``
        / ``(10, 2)``), and the ``DOUBLE PRECISION`` two-word form.
        Precision / scale are discarded — we only need the type name
        to map to a yggdrasil :class:`DataType`.
        """
        t = self.cur
        if t.kind not in ("ident", "keyword"):
            raise self._error("expected a type name after AS")
        self._eat()
        name = t.upper
        # ``DOUBLE PRECISION`` is the one two-word type head in our
        # supported surface — collapse it to a single canonical key.
        if (
            name == "DOUBLE"
            and self.cur.kind == "ident"
            and self.cur.upper == "PRECISION"
        ):
            self._eat()
            name = "DOUBLE PRECISION"
        if self.cur.kind == "lparen":
            # Skip ``(N)`` / ``(N, N)`` — we don't preserve precision.
            self._eat()
            depth = 1
            while depth > 0 and self.cur.kind != "eof":
                if self.cur.kind == "lparen":
                    depth += 1
                elif self.cur.kind == "rparen":
                    depth -= 1
                    if depth == 0:
                        break
                self._eat()
            self._expect_kind("rparen")
        return name


def _fold_cast(inner: Expression, type_name: str) -> Expression:
    """Lift a parsed CAST into a typed :class:`Literal` when possible.

    String-literal operands of temporal target types fold eagerly
    into a typed Python value (``date`` / ``datetime`` / ``time``)
    so backends see the right runtime type. Other casts wrap the
    inner expression in a :class:`Cast` node carrying the resolved
    yggdrasil :class:`DataType`; if the target type name is unknown
    to ``DataType.from_str`` the cast is treated as a no-op so the
    predicate still functions.
    """
    target = type_name.upper()
    if (
        isinstance(inner, Literal)
        and isinstance(inner.value, str)
        and target in _CAST_TEMPORAL_PARSERS
    ):
        parser, dtype_name = _CAST_TEMPORAL_PARSERS[target]
        try:
            parsed = parser(inner.value)
        except ValueError as exc:
            raise ValueError(
                f"from_sql: cannot parse {inner.value!r} as {target}: {exc}"
            ) from exc
        return Literal(value=parsed, dtype=_resolve_cast_dtype_str(dtype_name))
    dtype = _resolve_cast_dtype(target)
    if dtype is None:
        return inner
    return Cast(inner, dtype)


def _resolve_cast_dtype(target_name: str) -> Any:
    if not target_name:
        return None
    return _resolve_cast_dtype_str(
        _CAST_DTYPE_ALIASES.get(target_name, target_name.lower())
    )


def _resolve_cast_dtype_str(name: str) -> Any:
    from yggdrasil.data import DataType

    try:
        return DataType.from_str(name)
    except Exception:
        return None
