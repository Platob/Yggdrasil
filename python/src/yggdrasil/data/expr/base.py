"""Fluent SQL expression and predicate builder with dialect flavors.

    >>> from yggdrasil.data.expr import Expr
    >>> p = Expr("price").ge(70).and_(Expr("commodity").in_(["WTI", "Brent"]))
    >>> str(p)
    "`price` >= 70 AND `commodity` IN ('WTI', 'Brent')"

The two main types are :class:`Expr` (a column expression) and
:class:`Predicate` (a boolean expression tree returned by operator methods
like ``.eq()``, ``.in_()``, ``.between()``).

Default flavor is :attr:`Dialect.DATABRICKS` (backtick identifiers, ``TIMESTAMP '...'``
literals, native ``ILIKE``). Pass ``flavor=`` to switch:

    >>> Expr("c", flavor="postgres").eq(1).to_sql()
    '"c" = 1'

Table aliases prefix bare columns at render time (already-qualified columns
are left alone):

    >>> Expr("price", alias="t").eq(70).to_sql()
    '`t`.`price` = 70'

NULL-aware IN: when ``None``/``NULL`` appears in the value list, the predicate
expands so SQL semantics match Python intuition:

    >>> Expr("c").in_([1, 2, None]).to_sql()
    "`c` IN (1, 2) OR `c` IS NULL"
    >>> Expr("c").not_in([1, None]).to_sql()
    "`c` NOT IN (1) AND `c` IS NOT NULL"

Large IN lists of integer- or date-typed values are automatically compacted
into ``BETWEEN`` clauses when contiguous runs are detected (threshold:
:data:`COMPACT_IN_THRESHOLD`, default 1000):

    >>> Expr("id").in_(range(1, 5001)).to_sql()
    '`id` BETWEEN 1 AND 5000'
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar, Final, Iterable

__all__ = ["Expr", "Predicate", "SQLOp", "Dialect", "Flavor", "NULL", "flavor_of"]


# --- NULL sentinel ----------------------------------------------------------

class _Null:
    _instance: ClassVar["_Null | None"] = None

    def __new__(cls) -> "_Null":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NULL"

    def __bool__(self) -> bool:
        return False


NULL: Final = _Null()


# Sentinel used to distinguish "argument not provided" from "argument is None"
# in dual-form methods like Expr.between(low, high) vs. Expr.between(values).
_UNSET: Final = object()


def _is_null(v: Any) -> bool:
    """True for SQL-NULL-equivalent values: None, the NULL sentinel, or NaN.

    NaN is treated as NULL because pandas/numpy use it to mean 'missing' for
    numeric dtypes — and that's almost always the intent in commodity-data
    pipelines where missing observations sneak in via float columns.
    """
    if v is None or v is NULL:
        return True
    # NaN is the only float that isn't equal to itself; cheap and robust.
    if isinstance(v, float) and v != v:
        return True
    return False


def _as_python_list(values: Any) -> list[Any] | None:
    """Sniff for array-like objects (Arrow, polars, pandas, numpy) and return
    a plain Python list of their elements, with library-specific nulls
    normalized to ``None``. Returns ``None`` if the input isn't a recognized
    array-like — caller should fall back to plain iteration.

    Detection prefers ``to_list()`` (Arrow, polars) over ``tolist()`` (pandas,
    numpy). Bare strings/bytes are NOT treated as array-like even though they
    happen to be iterable.
    """
    if values is None or isinstance(values, (str, bytes, bytearray, memoryview)):
        return None
    # Native Python sequences/sets/dicts/etc — let the caller iterate them
    # directly so we don't pay for a copy and we preserve any custom types.
    if isinstance(values, (list, tuple, set, frozenset)):
        return None

    # Arrow / polars: zero-copy-ish .to_list() returning Python objects with
    # nulls already mapped to None. Arrow's ChunkedArray uses to_pylist instead.
    for method_name in ("to_list", "to_pylist", "tolist"):
        method = getattr(values, method_name, None)
        if callable(method):
            try:
                return list(method())
            except TypeError:
                # Some unrelated class might have a same-named method that takes
                # required args; fall through to the next candidate.
                continue

    return None


# --- operators --------------------------------------------------------------

class SQLOp(str, Enum):
    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IN = "IN"
    NOT_IN = "NOT IN"
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    ILIKE = "ILIKE"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    BETWEEN = "BETWEEN"
    NOT_BETWEEN = "NOT BETWEEN"


# --- dialect flavors --------------------------------------------------------

class Dialect(str, Enum):
    DATABRICKS = "databricks"  # also: spark
    SPARK = "spark"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    TSQL = "tsql"              # SQL Server
    STANDARD = "standard"      # ANSI-ish fallback

    @classmethod
    def parse(cls, v: Dialect | str) -> Dialect:
        if isinstance(v, cls):
            return v
        key = v.strip().lower()
        aliases = {"sparksql": cls.SPARK, "mssql": cls.TSQL, "sqlserver": cls.TSQL, "ansi": cls.STANDARD}
        if key in aliases:
            return aliases[key]
        for m in cls:
            if m.value == key:
                return m
        raise ValueError(f"unknown dialect: {v!r}")


@dataclass(frozen=True, slots=True)
class Flavor:
    """Per-dialect rendering rules. Not normally constructed by users — use :func:`flavor_of`."""
    dialect: Dialect
    ident_quote: str            # character used to quote identifiers
    typed_datetime: bool        # emit `TIMESTAMP '...'` / `DATE '...'` / `TIME '...'`
    boolean_as_int: bool        # render bools as 1/0 instead of TRUE/FALSE
    bytes_hex_prefix: str       # "X'" for standard, "0x" for T-SQL (no closing quote needed)
    supports_ilike: bool        # native ILIKE support; falls back to LOWER(...) LIKE LOWER(...)


_FLAVORS: dict[Dialect, Flavor] = {
    Dialect.DATABRICKS: Flavor(Dialect.DATABRICKS, "`",  True,  False, "X'", True),
    Dialect.SPARK:      Flavor(Dialect.SPARK,      "`",  True,  False, "X'", True),
    Dialect.POSTGRES:   Flavor(Dialect.POSTGRES,   '"',  True,  False, "X'", True),
    Dialect.MYSQL:      Flavor(Dialect.MYSQL,      "`",  False, False, "X'", False),
    Dialect.SQLITE:     Flavor(Dialect.SQLITE,     '"',  False, False, "X'", False),
    Dialect.TSQL:       Flavor(Dialect.TSQL,       "[",  False, True,  "0x", False),
    Dialect.STANDARD:   Flavor(Dialect.STANDARD,   '"',  True,  False, "X'", False),
}


def flavor_of(spec: Dialect | str | Flavor | None) -> Flavor:
    if spec is None:
        return _FLAVORS[Dialect.DATABRICKS]
    if isinstance(spec, Flavor):
        return spec
    return _FLAVORS[Dialect.parse(spec)]


_DEFAULT_FLAVOR: Final = _FLAVORS[Dialect.DATABRICKS]


# When `in_()` / `not_in()` receive more than this many non-null values AND
# those values are integer-step orderable (int / date / datetime), the builder
# attempts to compact contiguous runs into BETWEEN clauses. Override per-call
# via the `compact_threshold` kwarg, or globally by reassigning this module
# attribute. Set to 0 to disable.
COMPACT_IN_THRESHOLD: int = 1000


# --- identifier / literal rendering -----------------------------------------

def _quote_ident(name: str, flavor: Flavor) -> str:
    if not name:
        raise ValueError("empty identifier")
    q = flavor.ident_quote
    # T-SQL uses [name], everything else uses paired quotes.
    if q == "[":
        open_q, close_q, escape = "[", "]", lambda s: s.replace("]", "]]")
    else:
        open_q = close_q = q
        escape = lambda s: s.replace(q, q * 2)  # noqa: E731
    out = []
    for part in name.split("."):
        part = part.strip()
        if not part:
            raise ValueError(f"empty segment in identifier: {name!r}")
        if part == "*":
            out.append(part)
        else:
            out.append(open_q + escape(part) + close_q)
    return ".".join(out)


def _qualify_column(name: str, flavor: Flavor, alias: str | None) -> str:
    """Quote ``name``, prefixing it with ``alias.`` if provided AND the column
    is bare (contains no ``.``). Already-qualified columns pass through unchanged.
    """
    if alias and "." not in name:
        return _quote_ident(alias, flavor) + "." + _quote_ident(name, flavor)
    return _quote_ident(name, flavor)


def _render_literal(value: Any, flavor: Flavor) -> str:
    if _is_null(value):
        return "NULL"
    # bool BEFORE int (bool is a subclass of int)
    if isinstance(value, bool):
        if flavor.boolean_as_int:
            return "1" if value else "0"
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, datetime):
        iso = value.isoformat(sep=" ")  # SQL-style space, not T
        return f"TIMESTAMP '{iso}'" if flavor.typed_datetime else f"'{iso}'"
    if isinstance(value, date):
        return f"DATE '{value.isoformat()}'" if flavor.typed_datetime else f"'{value.isoformat()}'"
    if isinstance(value, time):
        return f"TIME '{value.isoformat()}'" if flavor.typed_datetime else f"'{value.isoformat()}'"
    if isinstance(value, bytes):
        prefix = flavor.bytes_hex_prefix
        if prefix == "0x":
            return "0x" + value.hex()
        return prefix + value.hex() + "'"
    return "'" + str(value).replace("'", "''") + "'"


# --- precedence -------------------------------------------------------------

_PREC_LEAF = 3
_PREC_NOT = 2
_PREC_AND = 1
_PREC_OR = 0


# --- predicate tree ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Predicate:
    """A node in a SQL WHERE expression tree.

    Build leaves via :func:`where`; combine via ``.and_()`` / ``.or_()`` /
    ``.not_()`` or operators ``&`` / ``|`` / ``~``.
    """

    kind: str       # "leaf" | "and" | "or" | "not"
    payload: Any    # leaf: (col, op, value); and/or: tuple[Predicate]; not: Predicate
    flavor: Flavor = _DEFAULT_FLAVOR
    table_alias: str | None = None   # prefix bare columns with this alias at render time

    PARAM_STYLE: ClassVar[str] = "qmark"

    # ---- composition (fluent) ----

    def and_(self, other: "Predicate") -> "Predicate":
        if not isinstance(other, Predicate):
            raise TypeError(f"and_() expects a Predicate, got {type(other).__name__}")
        left = self.payload if self.kind == "and" else (self,)
        right = other.payload if other.kind == "and" else (other,)
        return Predicate("and", tuple(left) + tuple(right), self.flavor, self.table_alias)

    def or_(self, other: "Predicate") -> "Predicate":
        if not isinstance(other, Predicate):
            raise TypeError(f"or_() expects a Predicate, got {type(other).__name__}")
        left = self.payload if self.kind == "or" else (self,)
        right = other.payload if other.kind == "or" else (other,)
        return Predicate("or", tuple(left) + tuple(right), self.flavor, self.table_alias)

    def not_(self) -> "Predicate":
        if self.kind == "not":
            return self.payload   # double-negation collapse
        return Predicate("not", self, self.flavor, self.table_alias)

    # ---- composition (operators) ----

    def __and__(self, other: "Predicate") -> "Predicate":
        return self.and_(other)

    def __or__(self, other: "Predicate") -> "Predicate":
        return self.or_(other)

    def __invert__(self) -> "Predicate":
        return self.not_()

    # ---- flavor switching ----

    def with_flavor(self, flavor: Dialect | str | Flavor) -> "Predicate":
        """Return a copy of this tree rendered with a different dialect."""
        f = flavor_of(flavor)
        if self.kind == "leaf":
            return Predicate(self.kind, self.payload, f, self.table_alias)
        if self.kind == "not":
            return Predicate("not", self.payload.with_flavor(f), f, self.table_alias)
        return Predicate(
            self.kind,
            tuple(c.with_flavor(f) for c in self.payload),
            f,
            self.table_alias,
        )

    # ---- table alias ----

    def with_table_alias(self, alias: str | None) -> "Predicate":
        """Return a copy of this tree where bare (undotted) columns are
        prefixed with ``alias.`` at render time. Pass ``None`` to clear.

        Columns that already contain a dot (e.g. ``"trades.book"``) are left
        alone — the user has qualified them deliberately. The alias overrides
        any previously-set alias on the tree (last call wins).
        """
        if alias is not None:
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("with_table_alias() requires a non-empty string or None")
            alias = alias.strip()
        if self.kind == "leaf":
            return Predicate(self.kind, self.payload, self.flavor, alias)
        if self.kind == "not":
            return Predicate("not", self.payload.with_table_alias(alias), self.flavor, alias)
        return Predicate(
            self.kind,
            tuple(c.with_table_alias(alias) for c in self.payload),
            self.flavor,
            alias,
        )

    # ---- rendering ----

    @property
    def _precedence(self) -> int:
        return {"leaf": _PREC_LEAF, "not": _PREC_NOT, "and": _PREC_AND, "or": _PREC_OR}[self.kind]

    def to_sql(self) -> str:
        """Render as inline SQL (literals expanded). Debug only — use :meth:`to_param` for execution."""
        return self._render_sql(parent_prec=-1)

    def to_param(
        self,
        style: str | None = None,
        start: int = 0,
    ) -> tuple[str, list[Any] | dict[str, Any]]:
        """Render with placeholders. Returns ``(sql, params)``.

        Styles: ``"qmark"`` (``?``), ``"numeric"`` (``$1``), ``"named"`` (``:p0``),
        ``"pyformat"`` (``%(p0)s``). ``start`` shifts the placeholder index for
        composition with other parameterized fragments.
        """
        style = style or self.PARAM_STYLE
        named = style in ("named", "pyformat")
        params: list[Any] | dict[str, Any] = {} if named else []
        sql = self._render_param(parent_prec=-1, style=style, params=params, start=start)
        return sql, params

    def __str__(self) -> str:
        return self.to_sql()

    # ---- internals ----

    def _render_sql(self, parent_prec: int) -> str:
        if self.kind == "leaf":
            return _render_leaf_sql(*self.payload, flavor=self.flavor, alias=self.table_alias)
        if self.kind == "not":
            text = f"NOT ({self.payload._render_sql(-1)})"
            return _maybe_paren(text, _PREC_NOT, parent_prec)
        sep = " AND " if self.kind == "and" else " OR "
        my_prec = self._precedence
        text = sep.join(c._render_sql(my_prec) for c in self.payload)
        return _maybe_paren(text, my_prec, parent_prec)

    def _render_param(
        self,
        parent_prec: int,
        style: str,
        params: list[Any] | dict[str, Any],
        start: int,
    ) -> str:
        if self.kind == "leaf":
            return _render_leaf_param(
                *self.payload,
                flavor=self.flavor,
                alias=self.table_alias,
                style=style,
                params=params,
                start=start,
            )
        if self.kind == "not":
            text = f"NOT ({self.payload._render_param(-1, style, params, start)})"
            return _maybe_paren(text, _PREC_NOT, parent_prec)
        sep = " AND " if self.kind == "and" else " OR "
        my_prec = self._precedence
        text = sep.join(c._render_param(my_prec, style, params, start) for c in self.payload)
        return _maybe_paren(text, my_prec, parent_prec)


def _maybe_paren(text: str, my_prec: int, parent_prec: int) -> str:
    return f"({text})" if my_prec < parent_prec else text


# --- leaf rendering ---------------------------------------------------------

def _is_unary(op: SQLOp) -> bool:
    return op in (SQLOp.IS_NULL, SQLOp.IS_NOT_NULL)


def _is_collection(op: SQLOp) -> bool:
    return op in (SQLOp.IN, SQLOp.NOT_IN)


def _is_range(op: SQLOp) -> bool:
    return op in (SQLOp.BETWEEN, SQLOp.NOT_BETWEEN)


def _render_leaf_sql(col: str, op: SQLOp, value: Any, *, flavor: Flavor, alias: str | None = None) -> str:
    qcol = _qualify_column(col, flavor, alias)
    if _is_unary(op):
        return f"{qcol} {op.value}"
    if _is_collection(op):
        return f"{qcol} {op.value} ({', '.join(_render_literal(v, flavor) for v in value)})"
    if _is_range(op):
        lo, hi = value
        return f"{qcol} {op.value} {_render_literal(lo, flavor)} AND {_render_literal(hi, flavor)}"
    if op is SQLOp.ILIKE and not flavor.supports_ilike:
        # Portable ILIKE fallback: LOWER(col) LIKE LOWER(pattern)
        return f"LOWER({qcol}) LIKE LOWER({_render_literal(value, flavor)})"
    return f"{qcol} {op.value} {_render_literal(value, flavor)}"


def _placeholder(style: str, idx: int) -> str:
    if style == "qmark":
        return "?"
    if style == "numeric":
        return f"${idx + 1}"
    if style == "named":
        return f":p{idx}"
    if style == "pyformat":
        return f"%(p{idx})s"
    raise ValueError(f"unknown param style: {style!r}")


def _render_leaf_param(
    col: str,
    op: SQLOp,
    value: Any,
    *,
    flavor: Flavor,
    alias: str | None = None,
    style: str,
    params: list[Any] | dict[str, Any],
    start: int,
) -> str:
    qcol = _qualify_column(col, flavor, alias)
    if _is_unary(op):
        return f"{qcol} {op.value}"

    named = style in ("named", "pyformat")

    def bind(v: Any) -> str:
        idx = start + len(params)
        ph = _placeholder(style, idx)
        if named:
            assert isinstance(params, dict)
            params[f"p{idx}"] = v
        else:
            assert isinstance(params, list)
            params.append(v)
        return ph

    if _is_collection(op):
        phs = ", ".join(bind(v) for v in value)
        return f"{qcol} {op.value} ({phs})"
    if _is_range(op):
        lo, hi = value
        return f"{qcol} {op.value} {bind(lo)} AND {bind(hi)}"
    if op is SQLOp.ILIKE and not flavor.supports_ilike:
        return f"LOWER({qcol}) LIKE LOWER({bind(value)})"
    return f"{qcol} {op.value} {bind(value)}"


# --- column handle ----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Expr:
    """A SQL column expression.

    Call an operator method (``.eq()``, ``.in_()``, ``.between()``, …) to
    produce a :class:`Predicate`. Carries the column name plus the dialect
    flavor and optional table alias used at render time. Immutable —
    :meth:`with_flavor` / :meth:`with_table_alias` return new instances.

    Construct directly:

        >>> Expr("price").eq(70)
        >>> Expr("price", flavor="postgres", alias="t").ge(70)

    The ``flavor`` argument accepts a :class:`Dialect`, a string like
    ``"postgres"``, or a :class:`Flavor`.
    """

    column: str
    flavor: Flavor = _DEFAULT_FLAVOR
    table_alias: str | None = None

    def __init__(
        self,
        column: str,
        flavor: "Dialect | str | Flavor | None" = None,
        alias: str | None = None,
    ) -> None:
        if not isinstance(column, str) or not column.strip():
            raise ValueError("Expr requires a non-empty column name")
        column = column.strip()
        for part in column.split("."):
            if not part.strip():
                raise ValueError(f"empty segment in identifier: {column!r}")

        resolved_flavor = flavor_of(flavor) if not isinstance(flavor, Flavor) else flavor

        if alias is not None:
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("alias must be a non-empty string or None")
            alias = alias.strip()

        # frozen=True: bypass the auto-generated __setattr__.
        object.__setattr__(self, "column", column)
        object.__setattr__(self, "flavor", resolved_flavor)
        object.__setattr__(self, "table_alias", alias)

    # ---- table alias ----

    def with_table_alias(self, alias: str | None) -> "Expr":
        """Return a copy of this expression with ``alias.`` prefixing the
        column at render time. Pass ``None`` to clear.

        If the column is already qualified (contains a ``.``), the alias is
        recorded but won't be applied — explicit qualification wins.
        """
        return Expr(self.column, self.flavor, alias)

    def with_flavor(self, flavor: Dialect | str | Flavor) -> "Expr":
        """Return a copy of this expression bound to a different dialect."""
        return Expr(self.column, flavor_of(flavor), self.table_alias)

    # ---- comparison ----

    def eq(self, value: Any) -> Predicate:
        if _is_null(value):
            return self.is_null()
        return self._leaf(SQLOp.EQ, _check_scalar(value, "eq"))

    def ne(self, value: Any) -> Predicate:
        if _is_null(value):
            return self.is_not_null()
        return self._leaf(SQLOp.NE, _check_scalar(value, "ne"))

    def lt(self, value: Any) -> Predicate:
        return self._leaf(SQLOp.LT, _check_scalar(value, "lt", allow_null=False))

    def le(self, value: Any) -> Predicate:
        return self._leaf(SQLOp.LE, _check_scalar(value, "le", allow_null=False))

    def gt(self, value: Any) -> Predicate:
        return self._leaf(SQLOp.GT, _check_scalar(value, "gt", allow_null=False))

    def ge(self, value: Any) -> Predicate:
        return self._leaf(SQLOp.GE, _check_scalar(value, "ge", allow_null=False))

    # ---- collection (NULL-aware) ----

    def in_(
        self,
        values: Iterable[Any],
        *,
        compact_threshold: int | None = None,
    ) -> Predicate:
        """``col IN (...)``. If the values contain ``None``/``NULL``, expand to
        ``(col IN (non_nulls...) OR col IS NULL)`` so SQL semantics match the
        natural Python reading.

        For large inputs (more than ``compact_threshold`` values, default
        :data:`COMPACT_IN_THRESHOLD` = 1000) of int- or date-typed values, the
        builder attempts to compact contiguous runs into ``BETWEEN`` clauses.
        E.g. a 5,000-element ``IN`` over consecutive ids becomes a single
        ``BETWEEN low AND high``. Pass ``compact_threshold=0`` to disable.
        """
        non_null, has_null = _split_nulls(values, "in_")
        threshold = COMPACT_IN_THRESHOLD if compact_threshold is None else compact_threshold

        if not non_null:
            return self.is_null() if has_null else self._leaf(SQLOp.IN, non_null)

        compacted = _try_compact_in(self, non_null, threshold, negate=False)
        base = compacted if compacted is not None else self._leaf(SQLOp.IN, non_null)

        if has_null:
            return base.or_(self.is_null())
        return base

    def not_in(
        self,
        values: Iterable[Any],
        *,
        compact_threshold: int | None = None,
    ) -> Predicate:
        """``col NOT IN (...)``. If the values contain ``None``/``NULL``, expand to
        ``(col NOT IN (non_nulls...) AND col IS NOT NULL)``. This is necessary
        because SQL ``NOT IN`` with a NULL element silently filters out every row
        (NULL propagation), which is rarely what the caller wants.

        Same compaction policy as :meth:`in_`: large int/date lists with
        contiguous runs are emitted as ``NOT BETWEEN ... AND NOT BETWEEN ...``.
        """
        non_null, has_null = _split_nulls(values, "not_in")
        threshold = COMPACT_IN_THRESHOLD if compact_threshold is None else compact_threshold

        if not non_null:
            return self.is_not_null() if has_null else self._leaf(SQLOp.NOT_IN, non_null)

        compacted = _try_compact_in(self, non_null, threshold, negate=True)
        base = compacted if compacted is not None else self._leaf(SQLOp.NOT_IN, non_null)

        if has_null:
            return base.and_(self.is_not_null())
        return base

    # ---- pattern ----

    def like(self, pattern: str) -> Predicate:
        return self._leaf(SQLOp.LIKE, _check_pattern(pattern, "like"))

    def not_like(self, pattern: str) -> Predicate:
        return self._leaf(SQLOp.NOT_LIKE, _check_pattern(pattern, "not_like"))

    def ilike(self, pattern: str) -> Predicate:
        return self._leaf(SQLOp.ILIKE, _check_pattern(pattern, "ilike"))

    # ---- range ----

    def between(self, low: Any, high: Any = _UNSET) -> Predicate:
        """``col BETWEEN low AND high``.

        Two call shapes:

        * ``between(low, high)`` — the explicit pair (NULL bounds rejected).
        * ``between(values)`` — a list/tuple/set; the predicate uses
          ``min(values)`` and ``max(values)`` after dropping ``None``/``NULL``.
          If the collection contained any ``None``/``NULL``, the predicate
          expands to ``(col BETWEEN lo AND hi OR col IS NULL)`` so SQL
          semantics match the natural Python reading.
        """
        lo, hi, has_null = _resolve_range_args(low, high, "between")
        leaf = self._leaf(SQLOp.BETWEEN, (lo, hi))
        return leaf.or_(self.is_null()) if has_null else leaf

    def not_between(self, low: Any, high: Any = _UNSET) -> Predicate:
        """``col NOT BETWEEN low AND high``. Same dual call shape as
        :meth:`between`. If the collection contained any ``None``/``NULL``,
        the predicate expands to ``(col NOT BETWEEN lo AND hi AND col IS NOT NULL)``
        — without this, SQL's NULL-propagation would silently filter every
        row whose ``col`` is NULL.
        """
        lo, hi, has_null = _resolve_range_args(low, high, "not_between")
        leaf = self._leaf(SQLOp.NOT_BETWEEN, (lo, hi))
        return leaf.and_(self.is_not_null()) if has_null else leaf

    # ---- null ----

    def is_null(self) -> Predicate:
        return self._leaf(SQLOp.IS_NULL, None)

    def is_not_null(self) -> Predicate:
        return self._leaf(SQLOp.IS_NOT_NULL, None)

    # ---- operator overloads ----

    def __eq__(self, value: Any) -> Predicate:  # type: ignore[override]
        return self.eq(value)

    def __ne__(self, value: Any) -> Predicate:  # type: ignore[override]
        return self.ne(value)

    def __lt__(self, value: Any) -> Predicate:
        return self.lt(value)

    def __le__(self, value: Any) -> Predicate:
        return self.le(value)

    def __gt__(self, value: Any) -> Predicate:
        return self.gt(value)

    def __ge__(self, value: Any) -> Predicate:
        return self.ge(value)

    # ---- internal ----

    def _leaf(self, op: SQLOp, value: Any) -> Predicate:
        return Predicate("leaf", (self.column, op, value), self.flavor, self.table_alias)


# --- value validation -------------------------------------------------------

_SCALAR_TYPES = (str, bytes, bool, int, float, Decimal, datetime, date, time)


def _check_scalar(value: Any, op_name: str, *, allow_null: bool = True) -> Any:
    if _is_null(value):
        if allow_null:
            return value
        raise ValueError(f"{op_name}() does not accept NULL; use is_null()/is_not_null()")
    if not isinstance(value, _SCALAR_TYPES):
        raise TypeError(f"{op_name}() requires a scalar, got {type(value).__name__}")
    return value


def _split_nulls(values: Any, op_name: str) -> tuple[tuple[Any, ...], bool]:
    """Partition values into ``(non_null_values, has_null)``.

    Recognizes array-like objects with ``to_list()`` (Arrow, polars) or
    ``tolist()`` (pandas, numpy) and converts them to plain Python lists
    before splitting. Bare strings/bytes are treated as a single value, not
    iterated character-by-character.
    """
    if values is None:
        raise ValueError(f"{op_name}() requires an iterable")
    if isinstance(values, (str, bytes)):
        values = (values,)
    sniffed = _as_python_list(values)
    if sniffed is not None:
        materialized = tuple(sniffed)
    else:
        materialized = tuple(values)
    if not materialized:
        raise ValueError(f"{op_name}() requires at least one value")
    non_null = tuple(v for v in materialized if not _is_null(v))
    has_null = len(non_null) != len(materialized)
    return non_null, has_null


def _check_pattern(pattern: Any, op_name: str) -> str:
    if not isinstance(pattern, str):
        raise TypeError(f"{op_name}() requires a string pattern, got {type(pattern).__name__}")
    return pattern


def _resolve_range_args(low: Any, high: Any, op_name: str) -> tuple[Any, Any, bool]:
    """Resolve the dual call shape of ``between`` / ``not_between``.

    Returns ``(low, high, has_null)``. ``has_null`` is only ever True when the
    one-arg collection form was used and the collection contained ``None`` or
    ``NULL``; the pair form rejects NULL bounds outright.

    * If ``high is _UNSET``, ``low`` must be a non-string iterable; the bounds
      are derived from ``min``/``max`` after dropping ``None``/``NULL``.
    * Otherwise both ``low`` and ``high`` must be non-NULL scalars.
    """
    if high is _UNSET:
        # Try array-like sniffing first; fall back to plain iteration.
        sniffed = _as_python_list(low)
        if sniffed is not None:
            materialized = tuple(sniffed)
        elif low is None or isinstance(low, (str, bytes)) or not _is_iterable(low):
            raise TypeError(
                f"{op_name}() with one argument requires an iterable of values; "
                f"got {type(low).__name__}"
            )
        else:
            materialized = tuple(low)
        non_null = tuple(v for v in materialized if not _is_null(v))
        has_null = len(non_null) != len(materialized)
        if not non_null:
            raise ValueError(f"{op_name}() requires at least one non-NULL value")
        for v in non_null:
            if not isinstance(v, _SCALAR_TYPES):
                raise TypeError(
                    f"{op_name}() values must be scalars; got {type(v).__name__}"
                )
        try:
            lo = min(non_null)
            hi = max(non_null)
        except TypeError as e:
            raise TypeError(f"{op_name}() values are not mutually comparable: {e}") from e
        return lo, hi, has_null

    return (
        _check_scalar(low, op_name, allow_null=False),
        _check_scalar(high, op_name, allow_null=False),
        False,
    )


def _is_iterable(v: Any) -> bool:
    try:
        iter(v)
        return True
    except TypeError:
        return False


# ---------------------------------------------------------------------------
# IN compaction: collapse large value lists into BETWEEN-of-runs
# ---------------------------------------------------------------------------

# Type -> step function. We compact only types where "next value" is
# unambiguously defined, so contiguity has a precise meaning.
def _step_int(v: int) -> int:
    return v + 1


def _step_date(v: date) -> date:
    from datetime import timedelta
    return v + timedelta(days=1)


def _compact_step(values: tuple[Any, ...]) -> Any | None:
    """Return the step function for these values, or None if compaction
    doesn't apply (mixed types, floats, strings, decimals, etc.)."""
    if not values:
        return None
    t = type(values[0])
    if not all(type(v) is t for v in values):
        return None
    # bool is a subclass of int — don't compact a list of bools.
    if t is bool:
        return None
    if t is int:
        return _step_int
    # datetime is a subclass of date, so check datetime FIRST.
    if t is datetime:
        # datetime "next" is ambiguous (microsecond? second?) — skip compaction.
        return None
    if t is date:
        return _step_date
    return None


def _find_runs(sorted_values: tuple[Any, ...], step: Any) -> list[tuple[Any, Any]]:
    """Group sorted values into contiguous runs. Returns ``[(low, high), ...]``
    where each tuple represents a maximal run of step-consecutive values.
    Singletons appear as ``(v, v)``.
    """
    runs: list[tuple[Any, Any]] = []
    run_lo = sorted_values[0]
    run_hi = sorted_values[0]
    for v in sorted_values[1:]:
        if v == run_hi:
            continue   # drop duplicates silently
        if v == step(run_hi):
            run_hi = v
        else:
            runs.append((run_lo, run_hi))
            run_lo = run_hi = v
    runs.append((run_lo, run_hi))
    return runs


def _try_compact_in(
    self_handle: "Expr",
    non_null: tuple[Any, ...],
    threshold: int,
    *,
    negate: bool,
) -> "Predicate | None":
    """Attempt to compact ``non_null`` into a runs-based predicate.

    Returns the compacted predicate, or ``None`` if compaction either doesn't
    apply (incompatible types) or would not reduce the predicate below the
    threshold (e.g. lots of singletons).

    For ``negate=False``: emits ``(BETWEEN .. OR BETWEEN .. OR = ..)``.
    For ``negate=True``:  emits ``(NOT BETWEEN .. AND NOT BETWEEN .. AND != ..)``.
    """
    if threshold <= 0 or len(non_null) <= threshold:
        return None
    step = _compact_step(non_null)
    if step is None:
        return None

    sorted_vals = tuple(sorted(set(non_null)))
    runs = _find_runs(sorted_vals, step)

    # If the compacted form has more nodes than the threshold, it's not a win
    # (lots of scattered singletons). Fall back to plain IN.
    if len(runs) > threshold:
        return None

    # Build the compacted predicate from the runs.
    leaves: list[Predicate] = []
    if negate:
        for lo, hi in runs:
            if lo == hi:
                leaves.append(self_handle.ne(lo))
            else:
                leaves.append(self_handle._leaf(SQLOp.NOT_BETWEEN, (lo, hi)))
        # NOT IN -> all-of-the-NOTs == AND chain.
        result = leaves[0]
        for leaf in leaves[1:]:
            result = result.and_(leaf)
        return result
    else:
        for lo, hi in runs:
            if lo == hi:
                leaves.append(self_handle.eq(lo))
            else:
                leaves.append(self_handle._leaf(SQLOp.BETWEEN, (lo, hi)))
        # IN -> any-of == OR chain.
        result = leaves[0]
        for leaf in leaves[1:]:
            result = result.or_(leaf)
        return result