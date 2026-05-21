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
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Union

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
    "simplify",
    "extract_partition_filters",
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
    ``yggdrasil.io.tabular.execution.expr.backends.*`` module — kept off the node
    so a build that excludes (say) pyspark doesn't import the
    optional dependency.
    """

    #: Set on Boolean-valued subclasses so `isinstance(x, Predicate)`
    #: is the cheap typing test. The base class stays scalar.
    _IS_PREDICATE: ClassVar[bool] = False

    #: Cached per-class field-name tuple. ``dataclasses.fields(self)``
    #: walks the ``__dataclass_fields__`` mapping and filters by field
    #: kind on every call — fine for cold paths, painful on the
    #: structural ``__hash__`` / :meth:`equals` hot loop. We populate
    #: this slot once per concrete subclass via :meth:`__init_subclass__`
    #: so the hot loop becomes a single attribute lookup.
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Skip ``super().__init_subclass__`` — the dataclass decorator
        # wraps the class with a closure that breaks the implicit
        # ``__class__`` cell ``super()`` relies on. ``object``'s
        # default ``__init_subclass__`` is a no-op anyway, so calling
        # it explicitly costs nothing here.
        #
        # ``dataclasses.fields`` is safe to call once at class-creation
        # time. We deliberately don't recurse into base classes — the
        # dataclass decorator already flattens inherited fields onto
        # the subclass's ``__dataclass_fields__``.
        if dataclasses.is_dataclass(cls):
            cls._FIELD_NAMES = tuple(
                f.name for f in dataclasses.fields(cls)
            )

    # ------------------------------------------------------------------
    # Identity / structural equality
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        # Structural hash — every field tuple-folded so equal trees
        # share the same bucket. ``equals`` does the matching
        # comparison; ``__eq__`` is reserved for the operator
        # overload below.
        #
        # Walks the cached ``_FIELD_NAMES`` tuple instead of calling
        # ``dataclasses.fields(self)`` per invocation; on a structural
        # hash of a 16-EQ OR chain this knocks ~30% off the cost.
        names = type(self)._FIELD_NAMES
        return hash((type(self),) + tuple(
            _hashable(getattr(self, n)) for n in names
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
        # Same-instance fast path — when the caller already holds
        # a reference to the live node (column re-use across an
        # OR chain, ``InList.target`` matching the merged group's
        # cached target), skip the field walk.
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        for n in type(self)._FIELD_NAMES:
            a = getattr(self, n)
            b = getattr(other, n)
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
    # Algebraic rewrites — equivalent tree, fewer / cheaper nodes.
    # ------------------------------------------------------------------

    def simplify(self) -> "Expression":
        """Return a logically equivalent but normalized form.

        Convenience method that delegates to the module-level
        :func:`simplify`. See its docstring for the exact rewrites
        applied — the headline ones are nested-Logical flattening,
        ``InList`` value de-duplication, and OR-of-equalities
        collapse (``c == a | c == b | c.is_null() → c.is_in([a, b])``
        with ``includes_null=True``).
        """
        return simplify(self)

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

    def to_sql(
        self,
        flavor: "str | None" = None,
        *,
        dialect: "str | None" = None,
    ) -> str:
        """Render to a SQL string for the named flavor / dialect.

        ``flavor`` is the canonical parameter name and accepts
        ``"databricks"`` / ``"postgres"`` / ``"sqlite"`` / ``"mysql"``
        / ``"ansi"``. ``dialect`` is a deprecated alias kept for
        callers that already use the older keyword. Default mirrors
        yggdrasil's primary target (Databricks).
        """
        from .backends.sql import to_sql

        chosen = flavor if flavor is not None else dialect
        return to_sql(self, dialect=chosen)

    def to_arrow(self):
        """Lift to a :class:`pyarrow.compute.Expression`.

        Canonical name; :meth:`to_pyarrow` is kept as an alias so
        callers don't have to update on rename.
        """
        from .backends.pyarrow_backend import to_pyarrow

        return to_pyarrow(self)

    #: Alias — :meth:`to_arrow` is the canonical name; ``to_pyarrow``
    #: is kept for code that already imports the long form.
    to_pyarrow = to_arrow

    def to_polars(self):
        """Lift to a :class:`polars.Expr`."""
        from .backends.polars_backend import to_polars

        return to_polars(self)

    def to_pyspark(self):
        """Lift to a :class:`pyspark.sql.Column`."""
        from .backends.pyspark_backend import to_pyspark

        return to_pyspark(self)

    #: Alias — Spark callers usually spell this ``to_spark``.
    to_spark = to_pyspark

    def to_engine(self, engine: str, **kwargs: Any) -> Any:
        """Dispatch by backend name.

        ``engine`` ∈ ``{"python", "sql", "arrow", "polars", "spark"}``
        — the same set :meth:`from_` accepts. ``**kwargs`` are
        forwarded to the matching ``to_*`` method (e.g. SQL takes
        ``flavor`` / ``strict`` for Python). Useful for code that
        picks the target at runtime (configuration-driven
        emitters, dispatch tables, …).
        """
        key = engine.strip().lower()
        if key == "python":
            return self.to_python(**kwargs)
        if key == "sql":
            return self.to_sql(**kwargs)
        if key in ("arrow", "pyarrow"):
            return self.to_arrow()
        if key == "polars":
            return self.to_polars()
        if key in ("spark", "pyspark"):
            return self.to_spark()
        raise ValueError(
            f"Unknown engine {engine!r}. Valid: "
            "python, sql, arrow, polars, spark."
        )

    # ------------------------------------------------------------------
    # Combination — AND-merge two predicates, identity-merge two equal
    # scalar expressions. Used by callers layering predicates
    # incrementally (cache-config filters, schema-side validators, …).
    # ------------------------------------------------------------------

    def merge_with(self, other: "Expression") -> "Expression":
        """Combine *self* with *other* into a single expression.

        Both sides predicates → conjunction (``self AND other``).
        Both sides structurally equal → return *self* (idempotent).
        Anything else raises :class:`TypeError` — a "merge" between
        a scalar and a different scalar isn't a well-defined
        operation; callers needing arithmetic combination should
        spell out the operator (``self + other``, etc.).
        """
        if isinstance(self, Predicate) and isinstance(other, Predicate):
            return Logical(LogicalOp.AND, (self, other))
        if self.equals(other):
            return self
        raise TypeError(
            "merge_with combines predicates (via AND) or identical "
            "expressions; got mismatched scalar expressions "
            f"{type(self).__name__} vs {type(other).__name__}. "
            "For arithmetic combinations use the explicit operator."
        )

    # ------------------------------------------------------------------
    # Class-method lifters — every backend's ``from_*`` rolls up here.
    # The generic :meth:`from_` sniffs the source's runtime type so
    # callers that don't know which engine produced an expression can
    # still hand it to us.
    # ------------------------------------------------------------------

    @classmethod
    def from_(cls, source: Any, **kwargs: Any) -> "Expression":
        """Auto-detect lifter.

        Routes to the matching ``from_*`` based on the source's
        runtime type:

        - ``str`` → :meth:`from_sql`
        - ``pyarrow.compute.Expression`` → :meth:`from_arrow`
        - ``polars.Expr`` → :meth:`from_polars`
        - ``pyspark.sql.Column`` → :meth:`from_spark`
        - already an :class:`Expression` → returned unchanged

        ``**kwargs`` are forwarded to the chosen lifter (e.g. SQL
        takes ``flavor=`` / ``dialect=``).
        """
        if isinstance(source, Expression):
            return source
        if isinstance(source, str):
            return cls.from_sql(source, **kwargs)

        # Module-name sniffing keeps the optional dependencies
        # truly optional — we never import polars / pyarrow / spark
        # here, just check what the object claims to be.
        module = (type(source).__module__ or "").split(".", 1)[0]
        if module == "pyarrow":
            return cls.from_arrow(source)
        if module == "polars":
            return cls.from_polars(source)
        if module == "pyspark":
            return cls.from_spark(source)

        raise TypeError(
            f"Expression.from_ does not know how to lift "
            f"{type(source).__module__}.{type(source).__name__}. "
            "Pass a string (SQL), pyarrow / polars / pyspark "
            "expression, or an existing yggdrasil Expression."
        )

    @classmethod
    def from_sql(
        cls,
        sql: str,
        flavor: "str | None" = None,
        *,
        dialect: "str | None" = None,
    ) -> "Expression":
        """Parse a SQL predicate string into our AST.

        Uses the in-tree tokenizer + recursive-descent parser in
        ``backends.sql`` — no third-party SQL parser dependency.
        See :func:`backends.sql.from_sql` for the supported grammar.
        """
        from .backends.sql import from_sql

        chosen = flavor if flavor is not None else dialect
        return from_sql(sql, dialect=chosen)

    @classmethod
    def from_arrow(cls, expr: Any) -> "Expression":
        """Lift a :class:`pyarrow.compute.Expression`."""
        from .backends.pyarrow_backend import from_pyarrow

        return from_pyarrow(expr)

    #: Alias — ``from_pyarrow`` matches ``to_pyarrow`` for callers
    #: that prefer the longer name.
    from_pyarrow = from_arrow

    @classmethod
    def from_polars(cls, expr: Any) -> "Expression":
        """Lift a :class:`polars.Expr`."""
        from .backends.polars_backend import from_polars

        return from_polars(expr)

    @classmethod
    def from_spark(cls, expr: Any) -> "Expression":
        """Lift a :class:`pyspark.sql.Column`."""
        from .backends.pyspark_backend import from_pyspark

        return from_pyspark(expr)

    #: Alias — ``from_pyspark`` matches ``to_pyspark``.
    from_pyspark = from_spark


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Predicate(Expression):
    """Marker mix-in for boolean-valued expressions.

    Every comparison / logical / membership / null-check node
    inherits this so callers can guard a ``where=`` argument with
    ``isinstance(x, Predicate)``. Adds two convenience filters
    (:meth:`filter_arrow_batch`, :meth:`filter_arrow_table`) that
    compile the predicate to a :class:`pyarrow.compute.Expression`
    and run the row-level filter natively in C++ — no Python row
    iteration. Used by :meth:`yggdrasil.io.tabular.base.Tabular.delete`
    on every leaf rewrite.
    """

    _IS_PREDICATE: ClassVar[bool] = True

    def filter_arrow_batch(self, batch: "Any") -> "Any":
        """Return *batch* keeping only rows where the predicate holds.

        Wraps the input :class:`pyarrow.RecordBatch` in a single-row-
        group :class:`pyarrow.dataset.InMemoryDataset` and runs the
        filter through ``Dataset.to_table(filter=...)``, which executes
        in pyarrow's C++ kernels. Returns a fresh ``RecordBatch``;
        when the filter drops every row, the result is a zero-row
        batch with the same schema. Pass-through on a zero-row input.
        """
        import pyarrow as pa
        import pyarrow.dataset as pds

        if batch.num_rows == 0:
            return batch
        table = pds.dataset(pa.Table.from_batches([batch])).to_table(
            filter=self.to_arrow(),
        )
        out = table.combine_chunks().to_batches()
        if out:
            return out[0]
        return pa.RecordBatch.from_pylist([], schema=batch.schema)

    def filter_arrow_table(self, table: "Any") -> "Any":
        """Return *table* keeping only rows where the predicate holds.

        Same shape as :meth:`filter_arrow_batch` but for
        :class:`pyarrow.Table`. Empty input returns the input
        unchanged so callers don't have to guard the zero-row case.
        """
        import pyarrow.dataset as pds

        if table.num_rows == 0:
            return table
        return pds.dataset(table).to_table(filter=self.to_arrow())

    def filter_arrow_batches(
        self, batches: "Iterable[Any]",
    ) -> "Iterator[Any]":
        """Streaming filter — yield surviving batches one at a time.

        Compiles the predicate to a pyarrow expression once and
        reuses it across the stream. Empty / fully-dropped batches
        are skipped (no zero-row pass-through), so consumers can
        treat the output as "non-empty rows that match" without an
        extra guard.

        Routes through :meth:`pa.RecordBatch.filter` directly so
        the per-batch cost stays C++-native — the previous shape
        wrapped each batch in a fresh
        :class:`pa.dataset.Dataset` + Table.from_batches before
        the kernel ran, which dominated on small batches.
        """
        expr = self.to_arrow()
        for batch in batches:
            if batch.num_rows == 0:
                continue
            kept = batch.filter(expr)
            if kept.num_rows > 0:
                yield kept


# ---------------------------------------------------------------------------
# Leaf nodes — Column, Literal
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Column(Expression):
    """A reference to a column inside an expression tree.

    Used as the leaf node for in-expression references
    (``col("price") > 100``); projection / rename / cast-on-select
    is the job of :class:`yggdrasil.data.data_field.Field`, which
    is the canonical "selector" for the tabular API.

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


# ---------------------------------------------------------------------------
# Algebraic rewrites — ``simplify(expr)``
#
# Every rewrite is shape-preserving when no rule fires (the input
# instance is returned unchanged), so calling ``simplify`` is safe to
# do unconditionally on any tree — the cost on already-normalized
# input is one pre-order walk.
# ---------------------------------------------------------------------------


def simplify(expr: Expression) -> Expression:
    """Return a logically equivalent (under SQL 3VL) normalized form.

    Rewrites applied bottom-up:

    - **InList dedup**: duplicate values are removed in first-seen
      order. ``c.is_in([1, 2, 2, 1])`` → ``c.is_in([1, 2])``.
    - **Logical flatten**: a ``Logical`` whose direct child is the
      same operator is inlined. ``(a OR b) OR c`` is the natural
      shape produced by ``a | b | c`` — flattening keeps the OR
      collapse (next bullet) seeing the full operand list.
    - **OR collapse**: equality comparisons against the same target
      expression, plus same-target ``InList`` and
      ``IsNull(negated=False)`` operands, are merged into a single
      ``InList``. ``c == 1 | c == 2 | c.is_null()`` →
      ``c.is_in([1, 2], includes_null=True)``. Targets are
      compared structurally via :meth:`Expression.equals`, so this
      handles ``col("x")`` *and* projections (``col("x").cast(...)``,
      ``col("x") + 1``) consistently.
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
    return _simplify(expr)


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


# Concrete-type dispatch — one ``type(expr)`` lookup beats an
# ``isinstance`` chain of 8+ checks every visit. Every AST node
# class is concrete (``Predicate`` is a mixin), so identity-keyed
# dispatch is sound. Leaves (``Column`` / ``Literal``) fall through
# to "return as-is".
_SIMPLIFY_DISPATCH: "dict[type, Any]" = {}


def _simplify(expr: Expression) -> Expression:
    handler = _SIMPLIFY_DISPATCH.get(type(expr))
    if handler is None:
        return expr  # Column, Literal, or any leaf — already canonical.
    return handler(expr)


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


def _dedupe_preserve_order(values: "tuple[Any, ...]") -> "tuple[Any, ...]":
    """Drop duplicate values while keeping the first occurrence's position.

    Hashable values use a ``set`` fast path; the unhashable branch
    falls back to a linear ``in out`` scan so dicts / lists land
    in the right slot deterministically. The latter is O(n²) but
    only fires when the caller explicitly seeded the InList with
    unhashable types — uncommon in practice.
    """
    seen: set[Any] = set()
    out: list[Any] = []
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


def _simplify_logical(expr: Logical) -> Expression:
    # Flatten same-op nesting before simplifying children. A left-
    # leaning chain ``(((a | b) | c) | d)`` (the shape Python's ``|``
    # builds) is N-1 nested ``Logical(OR)`` nodes — collapsing each
    # level independently would allocate an intermediate ``InList``
    # at every level. Flattening first means one OR collapse pass
    # over the full operand list and one final ``InList``.
    flat: list[Expression] = []
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
        reflattened: list[Expression] = []
        for c in simplified:
            if isinstance(c, Logical) and c.op is expr.op:
                reflattened.extend(c.operands)
            else:
                reflattened.append(c)
        simplified = reflattened

    if expr.op is LogicalOp.OR:
        return _collapse_or(simplified)
    return _collapse_and(simplified)


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
    groups: list[_OrGroup] = []
    classifications: list["int | None"] = []  # index into ``groups`` or None.

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

    new_ops: list[Expression] = []
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
    buckets: dict[int, list[Expression]] = {}
    unique: list[Expression] = []
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


# ---------------------------------------------------------------------------
# Partition-pruning extractor — over-approximate per-column accepted sets
#
# Engines that partition by a finite key set (Delta, Iceberg, Hive-style
# folder layouts) want a quick "which files can I skip" answer before any
# parquet open. The full :func:`Expression.to_python` / ``to_arrow``
# evaluator filters *rows*; this extractor walks the predicate once and
# returns the set of partition-column values that *could* satisfy it. The
# returned dict is consumed by :meth:`Snapshot.prune_files` (and any other
# partition-aware reader) — the row-level predicate still runs on the
# surviving files, so the extractor is allowed to over-approximate.
# ---------------------------------------------------------------------------


def extract_partition_filters(
    expr: Expression,
    columns: "Iterable[str]",
) -> "dict[str, frozenset]":
    """Over-approximate per-column accepted-value sets from a predicate.

    Walks *expr* (after :func:`simplify`) and returns, for each
    column in *columns* that the predicate constrains to a finite
    set, the :class:`frozenset` of values the column *could* take
    in any row the predicate accepts. Columns not in the returned
    dict are unconstrained — the predicate doesn't restrict their
    value to a finite, enumerable set.

    The result is suitable for partition pruning: a file whose
    partition value for ``col`` isn't in the extracted set can be
    skipped. It is *over-approximate* — a file the constraints
    accept may still produce zero matching rows (the row-level
    filter catches the residual), but no row the predicate accepts
    can fall outside the constraints. That makes the extractor
    safe to use as a pre-filter before the row-level scan.

    Supported shapes:

    - ``col == v``: ``{col: {v}}``.
    - ``col.is_in([v1, v2])``: ``{col: {v1, v2}}``.
      ``includes_null=True`` adds ``None`` to the set.
    - ``col.is_null()``: ``{col: {None}}``.
    - ``AND``: per-column intersection of constraints. Columns
      constrained on only one side keep their original set.
    - ``OR``: per-column union, but only for columns constrained
      on *every* operand (one unconstrained operand drops the
      column — the OR could accept any value via that branch).

    Returns ``{}`` for ``NOT``, ranges (``<`` / ``<=`` / ``>`` /
    ``>=`` / ``BETWEEN``), ``LIKE``, ``!=``, arithmetic on column
    references, column-vs-column comparisons, and ``col == NULL``
    (always UNKNOWN in SQL — never accepts a row).

    A returned ``{col: frozenset()}`` means the predicate is
    unsatisfiable on that column — the caller can skip every file
    whose partition value for ``col`` exists.
    """
    allowed = frozenset(columns)
    if not allowed:
        return {}
    return _extract_partition(simplify(expr), allowed)


def _extract_partition(
    expr: Expression,
    allowed: "frozenset[str]",
) -> "dict[str, frozenset]":
    if isinstance(expr, Logical):
        return _extract_logical(expr, allowed)
    if isinstance(expr, Comparison) and expr.op is CompareOp.EQ:
        col, val = _eq_col_and_literal(expr)
        if col is None or col not in allowed:
            return {}
        return {col: frozenset((val,))}
    if isinstance(expr, InList) and not expr.negated and isinstance(expr.target, Column):
        col_name = expr.target.name
        if col_name not in allowed:
            return {}
        if expr.includes_null:
            return {col_name: frozenset(expr.values) | frozenset((None,))}
        return {col_name: frozenset(expr.values)}
    if isinstance(expr, IsNull) and not expr.negated and isinstance(expr.target, Column):
        col_name = expr.target.name
        if col_name not in allowed:
            return {}
        return {col_name: frozenset((None,))}
    # NOT, !=, ranges, LIKE, BETWEEN, arithmetic, col-vs-col EQ,
    # col == NULL (always UNKNOWN) — all fall through to "no constraint".
    return {}


def _extract_logical(
    expr: Logical,
    allowed: "frozenset[str]",
) -> "dict[str, frozenset]":
    parts = [_extract_partition(o, allowed) for o in expr.operands]
    if expr.op is LogicalOp.AND:
        # Intersect per column; union of keys (constraints compose).
        out: "dict[str, frozenset]" = {}
        for d in parts:
            for k, v in d.items():
                if k in out:
                    out[k] = out[k] & v
                else:
                    out[k] = v
        return out
    # OR — per-column union, but only on columns every operand
    # constrained. A single unconstrained branch means the OR
    # could accept any value for that column.
    if not parts:
        return {}
    common = set(parts[0].keys())
    for d in parts[1:]:
        common &= set(d.keys())
    if not common:
        return {}
    out = {}
    for k in common:
        merged: "frozenset" = parts[0][k]
        for d in parts[1:]:
            merged = merged | d[k]
        out[k] = merged
    return out


def _eq_col_and_literal(
    comp: Comparison,
) -> "tuple[str | None, Any]":
    """Return ``(column_name, literal_value)`` for ``col == lit`` or
    ``lit == col``, else ``(None, None)``.

    Drops the ``col == NULL`` shape — SQL evaluates it as UNKNOWN
    for every row, so any value-set we built from it would be a
    lie. The caller's row-level filter still rejects those rows.
    """
    left, right = comp.left, comp.right
    if isinstance(left, Column) and isinstance(right, Literal):
        if right.value is None:
            return None, None
        return left.name, right.value
    if isinstance(right, Column) and isinstance(left, Literal):
        if left.value is None:
            return None, None
        return right.name, left.value
    return None, None


# ---------------------------------------------------------------------------
# Dispatch tables — populated after every node class + handler is in scope.
# Concrete-type dict lookup beats an ``isinstance`` chain on the hot
# simplify / extract paths.
# ---------------------------------------------------------------------------


_SIMPLIFY_DISPATCH.update({
    InList: _simplify_inlist,
    Logical: _simplify_logical,
    Not: _simplify_not,
    Comparison: _simplify_comparison,
    Between: _simplify_between,
    IsNull: _simplify_isnull,
    Like: _simplify_like,
    Cast: _simplify_cast,
    Arithmetic: _simplify_arithmetic,
})
