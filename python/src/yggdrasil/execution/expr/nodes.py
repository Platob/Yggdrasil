"""Abstract expression AST — node classes only.

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

Algorithmic transforms over the AST live in sibling modules so
this file stays focused on the dataclass shapes:

- ``operators.py`` — operator enums (``CompareOp``, ``LogicalOp``,
  ``ArithmeticOp``).
- ``walk.py`` — :func:`walk` / :func:`free_columns` visitors.
- ``simplify.py`` — :func:`simplify` and the OR/AND collapse rules.
- ``partition.py`` — :func:`extract_partition_filters` over-approx
  pruner.

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
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Union

from .operators import ArithmeticOp, CompareOp, LogicalOp

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
    "ExpressionLike",
    "lit",
]


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
    ``yggdrasil.execution.expr.backends.*`` module — kept off the node
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

        Convenience method that delegates to
        :func:`yggdrasil.execution.expr.simplify.simplify`.
        See its docstring for the exact rewrites — the headline
        ones are nested-Logical flattening, ``InList`` value
        de-duplication, and OR-of-equalities collapse
        (``c == a | c == b | c.is_null() → c.is_in([a, b])``
        with ``includes_null=True``).
        """
        from .simplify import simplify

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
        """Filter *batch* — hashset shortcut for ``InList`` / ``AND(InList)``,
        :meth:`pa.RecordBatch.filter` otherwise.

        The hashset shortcut probes each row's column value against a
        :class:`frozenset` and skips pyarrow's per-call compile + scan
        — ~30x faster than the kernel on the 1-row-per-leaf cache
        shape. Empty input is returned unchanged; an all-drop result
        is a zero-row slice with the source schema. Null semantics
        match :func:`pyarrow.compute.is_in` (``includes_null=True``
        keeps null rows, ``False`` drops them).
        """
        return self._filter(batch) if batch.num_rows else batch

    def filter_arrow_table(self, table: "Any") -> "Any":
        """Filter *table* — same dispatch as :meth:`filter_arrow_batch`."""
        return self._filter(table) if table.num_rows else table

    def filter_arrow_batches(
        self, batches: "Iterable[Any]",
    ) -> "Iterator[Any]":
        """Streaming filter — yield surviving batches one at a time.

        Decomposition / pyarrow compile happens once outside the
        loop; per-batch is then either a hashset probe or a single
        :meth:`pa.RecordBatch.filter` call. Empty / fully-dropped
        batches are skipped so consumers see only "non-empty rows
        that match".
        """
        clauses = _to_inlist_clauses(self)
        apply = (
            (lambda b: _apply_inlist(b, clauses))
            if clauses is not None
            else (lambda b, _expr=self.to_arrow(): b.filter(_expr))
        )
        for batch in batches:
            if batch.num_rows == 0:
                continue
            kept = apply(batch)
            if kept.num_rows > 0:
                yield kept

    def _filter(self, target: "Any") -> "Any":
        """One-shot filter dispatch shared by ``filter_arrow_batch`` / ``_table``.

        ``target`` may be a :class:`pa.RecordBatch` or :class:`pa.Table`
        — both expose ``num_rows`` / ``schema`` / ``column(name)`` /
        ``slice`` / ``take``, so one method covers both.
        """
        clauses = _to_inlist_clauses(self)
        if clauses is not None:
            return _apply_inlist(target, clauses)
        return target.filter(self.to_arrow())


def _to_inlist_clauses(
    predicate: "Predicate",
) -> "list[tuple[str, frozenset, bool]] | None":
    """Decompose into ``[(col, value_set, includes_null), …]`` or ``None``.

    Recognises ``InList(Column, [literals])`` and AND-of-InList.
    Returns ``None`` for everything else (non-``Column`` targets,
    negated InList, unhashable values, or any other AST shape) —
    the caller falls back to the pyarrow filter on ``None``.
    """
    if isinstance(predicate, InList):
        target = predicate.target
        if not isinstance(target, Column) or predicate.negated:
            return None
        try:
            return [(target.name, frozenset(predicate.values), predicate.includes_null)]
        except TypeError:
            return None
    if isinstance(predicate, Logical) and predicate.op is LogicalOp.AND:
        out: "list[tuple[str, frozenset, bool]]" = []
        for clause in predicate.operands:
            sub = _to_inlist_clauses(clause)
            if sub is None:
                return None
            out.extend(sub)
        return out
    return None


def _apply_inlist(
    target: "Any",
    clauses: "list[tuple[str, frozenset, bool]]",
) -> "Any":
    """Apply IN-list clauses to a :class:`pa.RecordBatch` or :class:`pa.Table`.

    Hashset row filter — per clause, materialise the column once,
    keep indices whose value is in ``value_set`` (plus nulls when
    ``includes_null``). Identity-returns *target* when every row
    matches, a zero-row slice when none do. Both Arrow types speak
    the same ``num_rows`` / ``schema`` / ``column(name).to_pylist``
    / ``slice`` / ``take`` surface, so the helper is duck-typed.
    """
    import pyarrow as pa

    n = target.num_rows
    schema_names = target.schema.names
    surviving: "list[int] | None" = None
    for column, value_set, includes_null in clauses:
        if column not in schema_names:
            # Defer the missing-column verdict to pyarrow so the
            # caller sees a uniform error rather than a silently-
            # empty answer.
            return target.filter(pa.compute.field(column).isin(list(value_set)))
        col_values = target.column(column).to_pylist()
        iterator = range(n) if surviving is None else surviving
        if includes_null:
            surviving = [
                i for i in iterator
                if (v := col_values[i]) is None or v in value_set
            ]
        else:
            surviving = [
                i for i in iterator
                if (v := col_values[i]) is not None and v in value_set
            ]
        if not surviving:
            return target.slice(0, 0)
    if surviving is None or len(surviving) == n:
        return target
    return target.take(pa.array(surviving, type=pa.int64()))


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
    out: "list[Any]" = []
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
    rest: "list[Any]" = []
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
