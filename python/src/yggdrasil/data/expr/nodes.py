"""Abstract expression AST.

The expression module is built around three layers:

- **Nodes** (this file): immutable dataclasses describing a tree of
  expressions. Every node inherits from :class:`Expression`. Boolean
  nodes additionally inherit the :class:`Predicate` marker so the
  type system can distinguish a filterable expression from an
  arbitrary scalar one.
- **Builder** (``builder.py``): the fluent factory + operator
  overloads users actually call (``col("price") > 100``).
- **Backends** (``backends/``): one module per target — Python,
  SQL, pyarrow, polars, pyspark. Each backend exposes a ``to_*``
  emitter and (where feasible) a ``from_*`` lifter that walks the
  foreign expression and rebuilds our AST. Methods on the base
  :class:`Expression` (``to_python``, ``to_sql``, …) dispatch to
  those backend modules.

Why this shape
--------------

A single AST means:

- One source of truth for predicate semantics. Backends are
  emitters, not behaviour redefinitions.
- Round-trip (``from_X(to_X(p))``) is well-defined for the node
  shapes the AST covers, so callers can move predicates between
  engines without losing intent.
- Field/DataType integration lives on the AST nodes (``Column.field``
  / ``Literal.dtype``). Backends consult those tags when the
  target engine needs typed literals (e.g. SQL ``TIMESTAMP '...'``,
  Spark casts).

The class hierarchy is intentionally narrow — :class:`Expression`,
:class:`Predicate`, plus a handful of leaf and combinator types.
Adding a new operator means adding a single dataclass plus a case in
each backend's emitter, not subclassing N abstract operator classes.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Union

if TYPE_CHECKING:
    from yggdrasil.data.data_field import Field
    from yggdrasil.data.types.base import DataType


__all__ = [
    "Expression",
    "Predicate",
    "Column",
    "Literal",
    "Comparison",
    "Logical",
    "Not",
    "Between",
    "InList",
    "IsNull",
    "Like",
    "Cast",
    "Arithmetic",
    "CompareOp",
    "LogicalOp",
    "ArithmeticOp",
    "ExpressionLike",
    "lit",
]


# ---------------------------------------------------------------------------
# Operator enums — one shared set across every backend
# ---------------------------------------------------------------------------


class CompareOp(str, enum.Enum):
    """Binary comparison operator, target-engine-agnostic.

    Backends translate to their own dialect: ``EQ`` becomes ``=`` in
    SQL, ``__eq__`` in Python, ``pa.compute.equal`` in pyarrow,
    etc.
    """

    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


class LogicalOp(str, enum.Enum):
    AND = "AND"
    OR = "OR"


class ArithmeticOp(str, enum.Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Expression:
    """Abstract base for every node in the AST.

    Expressions are immutable dataclasses. Combinators don't mutate
    operands — they wrap them in a new node — so an expression
    tree can be safely cached, reused, or shared across threads.

    The dataclass is built with ``eq=False`` because the operator
    overloads (``==``, ``<``, …) below produce :class:`Comparison`
    nodes instead of returning a bool — that's what makes
    ``col("price") >= 100`` work. Structural equality uses
    :meth:`equals`; identity-based hashing keeps nodes usable as
    dict keys.

    Subclasses override nothing; this class is a marker plus the
    operator surface and ``to_*`` dispatchers. Backend-specific
    compilation lives in the matching
    ``yggdrasil.data.expr.backends.*`` module — kept off the node
    so a build that excludes (say) pyspark doesn't import the
    optional dependency.
    """

    #: Set on Boolean-valued subclasses so `isinstance(x, Predicate)`
    #: is the cheap typing test. The base class stays scalar.
    _IS_PREDICATE: ClassVar[bool] = False

    # ------------------------------------------------------------------
    # Identity / structural equality
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        # Structural hash — every field tuple-folded so equal trees
        # share the same bucket. ``equals`` does the matching
        # comparison; ``__eq__`` is reserved for the operator
        # overload below.
        return hash((type(self),) + tuple(
            _hashable(getattr(self, f.name))
            for f in dataclasses.fields(self)
        ))

    def equals(self, other: Any) -> bool:
        """Structural equality.

        ``a.equals(b)`` is True when *a* and *b* are the same
        concrete node type with field-by-field equality. The plain
        ``==`` operator on Expressions builds a :class:`Comparison`
        node instead, so use :meth:`equals` whenever the test
        target is an Expression you'd otherwise be comparing for
        identity.
        """
        if type(self) is not type(other):
            return False
        for f in dataclasses.fields(self):
            a = getattr(self, f.name)
            b = getattr(other, f.name)
            if isinstance(a, Expression) and isinstance(b, Expression):
                if not a.equals(b):
                    return False
            elif isinstance(a, tuple) and isinstance(b, tuple):
                if len(a) != len(b):
                    return False
                for x, y in zip(a, b):
                    if isinstance(x, Expression) and isinstance(y, Expression):
                        if not x.equals(y):
                            return False
                    elif x != y:
                        return False
            elif a != b:
                return False
        return True

    # ------------------------------------------------------------------
    # Comparison overloads — produce :class:`Comparison` predicates.
    # ``==`` and ``!=`` shadow the dataclass-generated equality
    # (disabled via eq=False above); the rest are new.
    # ------------------------------------------------------------------

    def __eq__(self, other: "ExpressionLike") -> "Comparison":  # type: ignore[override]
        return Comparison(self, CompareOp.EQ, _coerce(other))

    def __ne__(self, other: "ExpressionLike") -> "Comparison":  # type: ignore[override]
        return Comparison(self, CompareOp.NE, _coerce(other))

    def __lt__(self, other: "ExpressionLike") -> "Comparison":
        return Comparison(self, CompareOp.LT, _coerce(other))

    def __le__(self, other: "ExpressionLike") -> "Comparison":
        return Comparison(self, CompareOp.LE, _coerce(other))

    def __gt__(self, other: "ExpressionLike") -> "Comparison":
        return Comparison(self, CompareOp.GT, _coerce(other))

    def __ge__(self, other: "ExpressionLike") -> "Comparison":
        return Comparison(self, CompareOp.GE, _coerce(other))

    # ------------------------------------------------------------------
    # Arithmetic — chained scalar expressions; result still has the
    # whole operator surface.
    # ------------------------------------------------------------------

    def __add__(self, other: "ExpressionLike") -> "Arithmetic":
        return Arithmetic(ArithmeticOp.ADD, self, _coerce(other))

    def __radd__(self, other: "ExpressionLike") -> "Arithmetic":
        return Arithmetic(ArithmeticOp.ADD, _coerce(other), self)

    def __sub__(self, other: "ExpressionLike") -> "Arithmetic":
        return Arithmetic(ArithmeticOp.SUB, self, _coerce(other))

    def __rsub__(self, other: "ExpressionLike") -> "Arithmetic":
        return Arithmetic(ArithmeticOp.SUB, _coerce(other), self)

    def __mul__(self, other: "ExpressionLike") -> "Arithmetic":
        return Arithmetic(ArithmeticOp.MUL, self, _coerce(other))

    def __rmul__(self, other: "ExpressionLike") -> "Arithmetic":
        return Arithmetic(ArithmeticOp.MUL, _coerce(other), self)

    def __truediv__(self, other: "ExpressionLike") -> "Arithmetic":
        return Arithmetic(ArithmeticOp.DIV, self, _coerce(other))

    def __rtruediv__(self, other: "ExpressionLike") -> "Arithmetic":
        return Arithmetic(ArithmeticOp.DIV, _coerce(other), self)

    def __mod__(self, other: "ExpressionLike") -> "Arithmetic":
        return Arithmetic(ArithmeticOp.MOD, self, _coerce(other))

    def __rmod__(self, other: "ExpressionLike") -> "Arithmetic":
        return Arithmetic(ArithmeticOp.MOD, _coerce(other), self)

    # ------------------------------------------------------------------
    # Boolean composition — works on any expression that's also a
    # Predicate. The ``__and__``/``__or__``/``__invert__`` overloads
    # below let users write ``a & b | ~c`` instead of named methods.
    # ------------------------------------------------------------------

    def __and__(self, other: "ExpressionLike") -> "Logical":
        return Logical(LogicalOp.AND, (self, _coerce(other)))

    def __rand__(self, other: "ExpressionLike") -> "Logical":
        return Logical(LogicalOp.AND, (_coerce(other), self))

    def __or__(self, other: "ExpressionLike") -> "Logical":
        return Logical(LogicalOp.OR, (self, _coerce(other)))

    def __ror__(self, other: "ExpressionLike") -> "Logical":
        return Logical(LogicalOp.OR, (_coerce(other), self))

    def __invert__(self) -> "Not":
        return Not(self)

    # ------------------------------------------------------------------
    # Named membership / null / cast / alias helpers — lifted up to
    # the base so chained expressions (``(col("x") + 1).is_in(...)``,
    # ``col("x").cast(int_t).is_null()``) are free of the
    # ``ColumnExpr``-only subclass.
    # ------------------------------------------------------------------

    def is_in(self, values: "Iterable[ExpressionLike]") -> "InList":
        materialized = _coerce_iter(values)
        non_null, has_null = _split_nulls(materialized)
        return InList(
            target=self, values=non_null,
            negated=False, includes_null=has_null,
        )

    def not_in(self, values: "Iterable[ExpressionLike]") -> "InList":
        materialized = _coerce_iter(values)
        non_null, has_null = _split_nulls(materialized)
        return InList(
            target=self, values=non_null,
            negated=True, includes_null=has_null,
        )

    def between(self, low: "ExpressionLike", high: "ExpressionLike") -> "Between":
        return Between(self, _coerce(low), _coerce(high), negated=False)

    def not_between(self, low: "ExpressionLike", high: "ExpressionLike") -> "Between":
        return Between(self, _coerce(low), _coerce(high), negated=True)

    def is_null(self) -> "IsNull":
        return IsNull(self, negated=False)

    def is_not_null(self) -> "IsNull":
        return IsNull(self, negated=True)

    def like(self, pattern: str, *, case_insensitive: bool = False) -> "Like":
        return Like(
            target=self, pattern=str(pattern),
            case_insensitive=case_insensitive, negated=False,
        )

    def not_like(self, pattern: str, *, case_insensitive: bool = False) -> "Like":
        return Like(
            target=self, pattern=str(pattern),
            case_insensitive=case_insensitive, negated=True,
        )

    def cast(self, dtype: "DataType") -> "Cast":
        return Cast(self, dtype)

    # ------------------------------------------------------------------
    # Backend dispatch — kept thin. Each backend is a module that
    # exposes ``to_<target>(expr)``; we just delegate.
    # ------------------------------------------------------------------

    def to_python(self, *, strict: bool = False):
        """Compile to a ``Callable[[Mapping[str, Any]], Any]``.

        With ``strict=True``, missing columns raise ``KeyError`` on
        evaluation; the default treats them as ``None`` (matching
        SQL three-valued logic).
        """
        from .backends.python import to_python

        return to_python(self, strict=strict)

    def to_sql(self, *, dialect: "str | None" = None) -> str:
        """Render to a SQL string for the named dialect.

        Default dialect mirrors yggdrasil's primary target
        (Databricks). Pass ``"postgres"`` / ``"sqlite"`` /
        ``"ansi"`` / ``"databricks"`` to switch quoting and
        literal-rendering rules.
        """
        from .backends.sql import to_sql

        return to_sql(self, dialect=dialect)

    def to_pyarrow(self):
        """Lift to a :class:`pyarrow.compute.Expression`."""
        from .backends.pyarrow_backend import to_pyarrow

        return to_pyarrow(self)

    def to_polars(self):
        """Lift to a :class:`polars.Expr`."""
        from .backends.polars_backend import to_polars

        return to_polars(self)

    def to_pyspark(self):
        """Lift to a :class:`pyspark.sql.Column`."""
        from .backends.pyspark_backend import to_pyspark

        return to_pyspark(self)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Predicate(Expression):
    """Marker mix-in for boolean-valued expressions.

    Every comparison / logical / membership / null-check node
    inherits this so callers can guard a ``where=`` argument with
    ``isinstance(x, Predicate)``. Doesn't add behaviour beyond the
    type tag.
    """

    _IS_PREDICATE: ClassVar[bool] = True


# ---------------------------------------------------------------------------
# Leaf nodes — Column, Literal
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Column(Expression):
    """A reference to a column in the source frame.

    ``field`` carries the typed :class:`Field` when known — backends
    that need engine-flavoured types (Spark casts, Arrow scalars)
    use it to build correctly-typed literals on comparison without
    asking the caller for a separate dtype.
    """

    name: str
    field: "Field | None" = None
    alias: "str | None" = None  # Optional table alias (e.g. ``T`` in ``T.col``).

    @property
    def dtype(self) -> "DataType | None":
        return self.field.dtype if self.field is not None else None


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Literal(Expression):
    """A scalar literal value.

    ``dtype`` is optional — when left ``None`` the backend infers
    from the Python value. Pinning a dtype is useful when the
    inferred type would be wrong (e.g. naive ``datetime`` you want
    rendered as ``DATE`` instead of ``TIMESTAMP``).
    """

    value: Any
    dtype: "DataType | None" = None


def lit(value: Any, dtype: "DataType | None" = None) -> Literal:
    """Build a :class:`Literal`. Convenience for the builder API."""
    return Literal(value=value, dtype=dtype)


# ---------------------------------------------------------------------------
# Boolean operators — every node here inherits ``Predicate``
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Comparison(Predicate):
    left: Expression
    op: CompareOp
    right: Expression


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Logical(Predicate):
    op: LogicalOp
    operands: "tuple[Expression, ...]" = ()

    def __post_init__(self) -> None:
        # Defensive: keep the operand tuple immutable even if the
        # caller handed in a list. Frozen dataclasses use
        # object.__setattr__ for post-init normalization.
        object.__setattr__(self, "operands", tuple(self.operands))
        if not self.operands:
            raise ValueError(
                f"Logical {self.op.value} needs at least one operand."
            )


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Not(Predicate):
    operand: Expression


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Between(Predicate):
    """``column BETWEEN low AND high`` — inclusive on both bounds.

    ``negated=True`` flips to ``NOT BETWEEN``. The Python and
    pyarrow emitters honour the inclusive contract; SQL renders
    natively.
    """

    target: Expression
    low: Expression
    high: Expression
    negated: bool = False


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class InList(Predicate):
    """``column IN (...)`` against a finite literal value list.

    Values are stored as tuples of literal Python objects so the
    node remains hashable. The :attr:`includes_null` flag carries
    forward through round-trips — backends that don't natively
    handle ``NULL`` inside ``IN`` (most SQL dialects) expand to
    ``... OR col IS NULL``.
    """

    target: Expression
    values: "tuple[Any, ...]" = ()
    negated: bool = False
    includes_null: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class IsNull(Predicate):
    target: Expression
    negated: bool = False  # ``IS NOT NULL`` when True.


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Like(Predicate):
    """SQL-style ``LIKE`` / ``ILIKE``.

    Pattern uses ``%`` and ``_`` as wildcards. Set ``case_insensitive``
    for ``ILIKE`` semantics; ``negated`` for ``NOT LIKE``.
    """

    target: Expression
    pattern: str
    case_insensitive: bool = False
    negated: bool = False


# ---------------------------------------------------------------------------
# Scalar combinators — Cast, Arithmetic
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Cast(Expression):
    """Explicit type cast. Returns a typed scalar expression."""

    operand: Expression
    dtype: "DataType"


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Arithmetic(Expression):
    """Two-operand arithmetic. Result type is the widened operand type."""

    op: ArithmeticOp
    left: Expression
    right: Expression


# ---------------------------------------------------------------------------
# Coercion — let users pass plain Python values where an Expression
# is expected (most common case is ``col("x") == 5``).
# ---------------------------------------------------------------------------


ExpressionLike = Union[Expression, Any]


def _coerce(value: "ExpressionLike") -> Expression:
    """Wrap a plain value in :class:`Literal` if it isn't already an Expression.

    Used by every operator overload so callers don't have to spell
    out ``lit(5)`` for the right-hand side of ``col("x") == 5``.
    Returns the input unchanged when it's already an Expression.
    """
    if isinstance(value, Expression):
        return value
    return Literal(value=value)


def _coerce_iter(values: "Iterable[ExpressionLike]") -> "tuple[Any, ...]":
    """Materialize a value iterable into a tuple of plain Python values.

    Strips :class:`Literal` wrappers — :class:`InList` stores the
    raw value tuple so equality / hashability stay simple. Anything
    that's already an :class:`Expression` other than a Literal
    triggers a ``TypeError`` because variadic membership against
    column references would have to compile to a join, not an
    ``IN`` predicate.
    """
    out: list[Any] = []
    for v in values:
        if isinstance(v, Literal):
            out.append(v.value)
        elif isinstance(v, Expression):
            raise TypeError(
                f"InList values must be plain literals; got {type(v).__name__}. "
                "For column-vs-column membership use a join."
            )
        else:
            out.append(v)
    return tuple(out)


def _split_nulls(values: "tuple[Any, ...]") -> "tuple[tuple[Any, ...], bool]":
    """Pull None / NaN out of ``values``, return ``(rest, had_null)``.

    Used by :meth:`Expression.is_in` / :meth:`Expression.not_in` so
    SQL-aware backends can route ``NULL`` through ``includes_null``
    rather than mixing it into the ``IN`` value set (where SQL would
    silently treat it as UNKNOWN).
    """
    has_null = False
    rest: list[Any] = []
    for v in values:
        if v is None or (isinstance(v, float) and v != v):
            has_null = True
        else:
            rest.append(v)
    return tuple(rest), has_null


def _hashable(value: Any) -> Any:
    """Best-effort hashable representation for :meth:`Expression.__hash__`.

    Lists / dicts get tuple-folded; anything that's already
    hashable passes through. Non-hashable leaves fall back to
    ``id``-based hashing — a structural hash is still preferred,
    so passing unhashable user objects as literals is a fast
    way to lose dedup.
    """
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(
            sorted((k, _hashable(v)) for k, v in value.items())
        )
    try:
        hash(value)
    except TypeError:
        return id(value)
    return value


# ---------------------------------------------------------------------------
# Tree walk — small visitor used by every backend
# ---------------------------------------------------------------------------


def walk(expr: Expression) -> "Iterable[Expression]":
    """Pre-order walk over every node in *expr*.

    Backends use this for visitors that don't need to produce a
    transformed tree (schema collection, free-variable lookup,
    optimization checks). For tree rewrites prefer pattern-matching
    on node types directly.
    """
    yield expr
    if isinstance(expr, (Comparison, Arithmetic)):
        yield from walk(expr.left)
        yield from walk(expr.right)
    elif isinstance(expr, Logical):
        for op in expr.operands:
            yield from walk(op)
    elif isinstance(expr, Not):
        yield from walk(expr.operand)
    elif isinstance(expr, (Between,)):
        yield from walk(expr.target)
        yield from walk(expr.low)
        yield from walk(expr.high)
    elif isinstance(expr, (InList, IsNull, Like)):
        yield from walk(expr.target)
    elif isinstance(expr, Cast):
        yield from walk(expr.operand)


def free_columns(expr: Expression) -> "tuple[str, ...]":
    """Names of every distinct column referenced by *expr*.

    Order is first-encounter (pre-order walk), de-duplicated. Used
    by the Python backend to build a value-resolution closure and
    by the schema emitter to advertise the predicate's input
    surface.
    """
    seen: dict[str, None] = {}
    for node in walk(expr):
        if isinstance(node, Column):
            seen.setdefault(node.name, None)
    return tuple(seen)
