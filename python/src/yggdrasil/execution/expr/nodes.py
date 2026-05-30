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
    "FunctionCall",
    "Star",
    "Alias",
    "SortOrder",
    "WindowSpec",
    "WindowFunction",
    "CaseWhen",
    "Subscript",
    "ExpressionLike",
    "PredicateLike",
    "lit",
]


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class Expression:
    """Abstract base for every node in the AST.

    Expressions are immutable by convention — combinators don't
    mutate operands but wrap them in a new node, so an expression
    tree can be safely cached, reused, or shared across threads.

    Concrete subclasses opt out of :class:`dataclasses.dataclass`:
    a manual ``__slots__`` + explicit ``__init__`` is roughly 3× the
    construction throughput of ``@dataclass(frozen=True, slots=True)``
    on the per-node hot path, and the AST builds enough nodes per
    predicate (one ``Comparison`` per ``c == v``, one ``Logical`` per
    ``|`` / ``&``) that the savings show up in
    ``benchmarks/io/tabular/bench_predicate.py``.

    The ``__eq__`` / ``<`` / ``>`` / ``>=`` / ``<=`` / ``!=`` operator
    overloads on this base produce :class:`Comparison` nodes instead
    of returning a bool — that's what makes ``col("price") >= 100``
    work. Structural equality uses :meth:`equals`; identity-based
    hashing keeps nodes usable as dict keys.

    Subclasses override nothing structural; this class is a marker
    plus the operator surface and ``to_*`` dispatchers. Backend-
    specific compilation lives in the matching
    ``yggdrasil.execution.expr.backends.*`` module — kept off the
    node so a build that excludes (say) pyspark doesn't import the
    optional dependency.
    """

    __slots__ = ()

    #: Set on Boolean-valued subclasses so `isinstance(x, Predicate)`
    #: is the cheap typing test. The base class stays scalar.
    _IS_PREDICATE: ClassVar[bool] = False

    #: Per-class field-name tuple. Set explicitly on every concrete
    #: subclass alongside ``__slots__`` so the structural ``__hash__``
    #: / :meth:`equals` hot loop is a single ClassVar lookup — no
    #: ``dataclasses.fields(...)`` walk, no ``__slots__`` filtering
    #: per call.
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Fallback for subclasses that forgot to declare _FIELD_NAMES
        # explicitly: read from __slots__. Cheap once per class.
        super().__init_subclass__(**kwargs)
        if "_FIELD_NAMES" not in cls.__dict__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            cls._FIELD_NAMES = tuple(slots)

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
        return _smart_comparison(self, CompareOp.EQ, _coerce(other))

    def __ne__(self, other: "ExpressionLike") -> "Comparison":  # type: ignore[override]
        return _smart_comparison(self, CompareOp.NE, _coerce(other))

    def __lt__(self, other: "ExpressionLike") -> "Comparison":
        return _smart_comparison(self, CompareOp.LT, _coerce(other))

    def __le__(self, other: "ExpressionLike") -> "Comparison":
        return _smart_comparison(self, CompareOp.LE, _coerce(other))

    def __gt__(self, other: "ExpressionLike") -> "Comparison":
        return _smart_comparison(self, CompareOp.GT, _coerce(other))

    def __ge__(self, other: "ExpressionLike") -> "Comparison":
        return _smart_comparison(self, CompareOp.GE, _coerce(other))

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

    and_ = __and__

    def __rand__(self, other: "ExpressionLike") -> "Logical":
        return Logical(LogicalOp.AND, (_coerce(other), self))

    def __or__(self, other: "ExpressionLike") -> "Logical":
        return Logical(LogicalOp.OR, (self, _coerce(other)))

    or_ = __or__

    def __ror__(self, other: "ExpressionLike") -> "Logical":
        return Logical(LogicalOp.OR, (_coerce(other), self))

    def __invert__(self) -> "Not":
        return Not(self)

    invert = __invert__

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

    def cast(self, dtype: "DataType") -> "Expression":
        """Coerce *self* to ``dtype``.

        Smart factory — not all calls return a fresh :class:`Cast` node:

        - When *self* already advertises ``dtype`` (a :class:`Column`
          whose ``field.dtype`` matches, or a :class:`Cast` already
          targeting ``dtype``), :meth:`cast` is a no-op and returns
          *self* unchanged.
        - When *self* is an Object- or Null-typed :class:`Column`,
          the column carries no real type information yet — its
          ``field.dtype`` is replaced with ``dtype`` in a new
          :class:`Column` (no :class:`Cast` wrapper needed: the
          downstream filter / SQL emitter reads the field's dtype
          directly).
        - When *self* is a :class:`Cast` already, the inner cast
          collapses: ``Cast(x, A).cast(B)`` becomes
          ``Cast(x, B)`` so backends emit one cast, not two.
        - Otherwise, wrap *self* in :class:`Cast`.

        Construct :class:`Cast` directly to opt out of these
        rewrites (e.g. when a test wants the raw wrapper).
        """
        from yggdrasil.data.types.id import DataTypeId

        if isinstance(self, Column):
            current = self.field.dtype if self.field is not None else None
            if current is dtype or current == dtype:
                return self
            if current is None or current.type_id in (
                DataTypeId.OBJECT, DataTypeId.NULL,
            ):
                from yggdrasil.data.data_field import Field

                if self.field is None:
                    new_field = Field(name=self.name, dtype=dtype)
                else:
                    new_field = self.field.with_dtype(dtype, inplace=False)
                return Column(
                    name=self.name,
                    field=new_field,
                    alias=self.alias,
                    qualifier=self.qualifier,
                )
        if isinstance(self, Cast):
            if self.dtype is dtype or self.dtype == dtype:
                return self
            return Cast(self.operand, dtype)
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

    def __repr__(self) -> str:
        # Generic ``Node(field=value, ...)`` rendering keyed on the
        # ClassVar ``_FIELD_NAMES`` tuple. Subclasses that want a
        # custom shape (or want to hide internal slots) override
        # this; the default is fine for the AST nodes whose fields
        # all carry useful repr.
        parts = [
            f"{name}={getattr(self, name, None)!r}"
            for name in type(self)._FIELD_NAMES
        ]
        return f"{type(self).__name__}({', '.join(parts)})"


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

    __slots__ = ()
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

        Decomposition happens once outside the loop. ``InList`` /
        ``AND(InList)`` shapes route through :func:`_apply_inlist`
        per-batch (one ``pc.is_in`` kernel per clause); everything
        else compiles a single ``pa.Expression`` on the first
        non-empty batch (so a stream of empty batches doesn't pay
        the compile cost) and reuses it across the stream. Empty /
        fully-dropped batches are skipped so consumers see only
        "non-empty rows that match".
        """
        clauses = _to_inlist_clauses(self)
        effective_arrow: "Any" = None
        for batch in batches:
            if batch.num_rows == 0:
                continue
            if clauses is not None:
                kept = _apply_inlist(batch, clauses)
            else:
                if effective_arrow is None:
                    effective = _rewrite_with_target_lookup(
                        self, _arrow_field_lookup(batch.schema),
                    )
                    effective_arrow = effective.to_arrow()
                kept = _materialize_view_columns(batch).filter(effective_arrow)
            if kept.num_rows > 0:
                yield kept

    def _filter(self, target: "Any") -> "Any":
        """One-shot filter dispatch shared by ``filter_arrow_batch`` / ``_table``.

        ``target`` may be a :class:`pa.RecordBatch` or :class:`pa.Table`
        — both expose ``num_rows`` / ``schema`` / ``column(name)`` /
        ``filter`` (mask-shaped), so one method covers both.

        Two paths:

        * ``InList(Column, …)`` / ``AND(InList, …)`` shapes route
          through :func:`_apply_inlist` — one ``pc.is_in`` kernel
          call per clause, then ``and_kleene`` to combine. Skips the
          ``pa.Expression`` compile that ``target.filter(self.to_arrow())``
          would pay. Fastest at every batch size for these shapes
          and bypasses the target-schema rewrite (the rewrites
          don't touch naked InLists).
        * Everything else runs :func:`_rewrite_with_target_lookup`
          first (so the timezone pushdown / literal coercion fire
          against the target's actual schema) and then the standard
          ``target.filter(pa.Expression)`` path.
        """
        clauses = _to_inlist_clauses(self)
        if clauses is not None:
            return _apply_inlist(target, clauses)
        effective = _rewrite_with_target_lookup(self, _arrow_field_lookup(target.schema))
        return _materialize_view_columns(target).filter(effective.to_arrow())

    # ------------------------------------------------------------------
    # Engine filters — mirrors the read_X / write_X surface every
    # ``Tabular`` already exposes. Each method runs the predicate on
    # the matching engine's native representation, applying the
    # cast-column optimisation when present (see :func:`_cast_columns`
    # and the per-backend fallback in :meth:`_apply_engine_filter`).
    # ------------------------------------------------------------------

    def filter_polars_frame(self, frame: "Any") -> "Any":
        """Filter a :class:`polars.DataFrame` / :class:`polars.LazyFrame`.

        Both shapes expose ``.filter(expr)`` accepting a
        :class:`polars.Expr``, so one method covers both. The target
        schema feeds the same temporal-tz rewrite as the Arrow
        filter — a Polars frame whose ``ts`` column is UTC and a
        predicate built with a Paris-tz Cast lands with the literal
        already in UTC by the time polars sees the expression.
        """
        effective = _rewrite_with_target_lookup(self, _polars_field_lookup(frame.schema))
        return frame.filter(effective.to_polars())

    def filter_pandas_frame(self, frame: "Any") -> "Any":
        """Filter a :class:`pandas.DataFrame` via the pyarrow kernel.

        Pandas has no first-class predicate kernel and ``df.apply``
        row-wise costs ~20 ms / 5 k rows (the slowest path in
        ``bench_predicate.py``). Round-trip through Arrow instead:
        ``pa.Table.from_pandas`` → :meth:`filter_arrow_table` (which
        already auto-tunes between the hashset shortcut and the
        pyarrow filter kernel) → ``Table.to_pandas`` — the zero-copy
        legs land at sub-millisecond range on the same fixture, and
        downstream behaves identically (same row order, same dtype
        per column) to the original frame.
        """
        import pyarrow as pa

        table = pa.Table.from_pandas(frame, preserve_index=True)
        filtered = self.filter_arrow_table(table)
        return filtered.to_pandas()

    def filter_spark_frame(self, frame: "Any") -> "Any":
        """Filter a :class:`pyspark.sql.DataFrame`."""
        return frame.filter(self.to_pyspark())

    def filter_pylist(
        self, rows: "Iterable[Any]",
    ) -> "list[Any]":
        """Filter a list / iterable of ``Mapping`` rows (the
        :meth:`Tabular.read_pylist` shape).

        Uses the same ``InList`` / ``AND(InList, ...)`` hashset
        shortcut as the arrow filter: when the predicate decomposes
        into ``(col, value_set, includes_null)`` clauses, each row
        becomes a dict-lookup + set-probe instead of compiling the
        full AST to a callable. Speeds up the cache-key lookup
        shape by ~2× on bench-sized inputs.
        """
        clauses = _to_inlist_clauses(self)
        if clauses is not None:
            return _apply_inlist_pylist(rows, clauses)
        from .backends.python import to_python

        fn = to_python(self)
        return [row for row in rows if fn(row)]

    def filter_pydict(self, data: "Mapping[str, Any]") -> "dict[str, list]":
        """Filter a column-oriented ``{col: [values, ...]}`` dict.

        Mirrors :meth:`Tabular.read_pydict`: builds a temporary Arrow
        table, runs the same hashset / pyarrow filter as
        :meth:`filter_arrow_table`, and unzips the survivors back into
        a fresh ``{col: list}`` dict. Empty input or no surviving rows
        return a dict with the same key set and empty lists.
        """
        import pyarrow as pa

        if not data:
            return {}
        table = pa.table(data)
        filtered = self.filter_arrow_table(table)
        return {name: filtered.column(name).to_pylist() for name in filtered.schema.names}

    def filter_iterable(
        self, items: "Iterable[Any]",
        *,
        key: "Any | None" = None,
    ) -> "Iterator[Any]":
        """Filter an arbitrary iterable. Items default to dict-shaped
        rows (``Mapping[str, Any]``); pass ``key=`` to project the
        comparison target (e.g. ``key=attrgetter('payload')``) when
        the predicate columns address a sub-shape of each item.
        Yields each item the predicate accepts.

        Uses the same hashset shortcut as :meth:`filter_pylist` when
        the predicate decomposes into ``InList`` clauses — same
        speedup, just lazy (yields one at a time) instead of
        materialising into a list.
        """
        clauses = _to_inlist_clauses(self)
        if clauses is not None:
            cols = [c for c, _, _ in clauses]
            sets = [s for _, s, _ in clauses]
            nulls = [n for _, _, n in clauses]
            n = len(clauses)

            def _probe(row: "Any", _c=cols, _s=sets, _nl=nulls, _n=n) -> bool:
                for i in range(_n):
                    value = row.get(_c[i])
                    if value is None:
                        if not _nl[i]:
                            return False
                    elif value not in _s[i]:
                        return False
                return True

            if key is None:
                for item in items:
                    if _probe(item):
                        yield item
                return
            for item in items:
                if _probe(key(item)):
                    yield item
            return

        from .backends.python import to_python

        fn = to_python(self)
        if key is None:
            for item in items:
                if fn(item):
                    yield item
            return
        for item in items:
            if fn(key(item)):
                yield item

    def filter(self, target: "Any") -> "Any":
        """Generic dispatch — pick the right ``filter_*`` for *target*'s type.

        Recognised shapes:

        - :class:`pyarrow.Table` → :meth:`filter_arrow_table`.
        - :class:`pyarrow.RecordBatch` → :meth:`filter_arrow_batch`.
        - :class:`pandas.DataFrame` → :meth:`filter_pandas_frame`.
        - :class:`polars.DataFrame` / :class:`polars.LazyFrame` →
          :meth:`filter_polars_frame`.
        - :class:`pyspark.sql.DataFrame` → :meth:`filter_spark_frame`.
        - :class:`dict` of columns → :meth:`filter_pydict`.
        - :class:`list` / :class:`tuple` → :meth:`filter_pylist`.
        - Anything else iterable → :meth:`filter_iterable`.

        Optional dependencies (polars / pandas / pyspark) are detected
        by module-name sniffing on ``type(target)`` so this method
        never imports them just to dispatch — the relevant
        ``filter_*`` does the import when called.
        """
        import pyarrow as pa

        if isinstance(target, pa.Table):
            return self.filter_arrow_table(target)
        if isinstance(target, pa.RecordBatch):
            return self.filter_arrow_batch(target)
        module = (type(target).__module__ or "").split(".", 1)[0]
        if module == "pandas":
            return self.filter_pandas_frame(target)
        if module == "polars":
            return self.filter_polars_frame(target)
        if module == "pyspark":
            return self.filter_spark_frame(target)
        if isinstance(target, dict):
            return self.filter_pydict(target)
        if isinstance(target, (list, tuple)):
            return self.filter_pylist(target)
        return self.filter_iterable(target)


def _smart_comparison(
    left: "Expression",
    op: "CompareOp",
    right: "Expression",
) -> "Comparison":
    """Build a :class:`Comparison`, applying construction-time rewrites.

    Currently only the temporal-timezone pushdown
    (:func:`_try_timezone_pushdown`) — keeping the dispatch table
    open here means any future "rewrite at operator time" rule
    (Cast-of-Literal constant-folding, NOT-EQ shortcut for null
    literals, …) lands at this one site instead of leaking branches
    through every overload on :class:`Expression`.
    """
    rewrite = _try_timezone_pushdown(left, op, right)
    if rewrite is not None:
        return rewrite
    return Comparison(left, op, right)


def _try_timezone_pushdown(
    left: "Any",
    op: "CompareOp",
    right: "Any",
    column_field: "Callable[[Column], Any | None] | None" = None,
) -> "Expression | None":
    """Rewrite ``Cast(col_tz_X, ts_tz_Y) op lit_tz_Y`` to native filter.

    When a comparison casts a tz-aware timestamp column into a
    different timezone just to match a tz-aware literal, the result
    is N-row tz arithmetic that an engine has to run before it can
    filter. The arithmetic is reversible and constant on the literal
    side: converting the literal to the column's native tz once
    means the filter can compare native column values directly,
    which Arrow / Polars / Spark all push down into their column
    pruning + parquet predicate pushdown layer.

    Returns the rewritten :class:`Comparison` (without the Cast wrap)
    when the rewrite is safe, or ``None`` to fall back to the
    literal-shape comparison.

    ``column_tz_iana`` lets a filter-time caller hand in a lookup
    against the *target's* schema — pyarrow Table / RecordBatch
    schemas, polars frame schemas, etc. — so the rewrite fires even
    when the predicate's :class:`Column` carries no bound
    :class:`Field`. Default is to read from ``column.field.dtype``
    so construction-time ``__eq__`` / ``__lt__`` overloads still
    pick up the rewrite when the caller already provided a typed
    Field.

    Triggers only when ALL of these hold:

    * Exactly one side is a :class:`Cast` over a :class:`Column`;
    * The Cast targets a :class:`TimestampType` whose tz is not
      naive;
    * The source column's actual tz (from ``column_tz_iana`` or
      the bound Field) is tz-aware;
    * The opposing side is a :class:`Literal` carrying a
      tz-aware ``datetime.datetime`` value.
    """
    # Light, lazy imports — temporal pushdown is the one place the
    # expression AST has to peek at concrete DataType subclasses, and
    # we don't want to drag :mod:`yggdrasil.data.types` into base AST
    # construction unconditionally.
    from yggdrasil.data.types.primitive.temporal import TimestampType

    cast_side, lit_side = None, None
    if isinstance(left, Cast) and isinstance(right, Literal):
        cast_side, lit_side = left, right
        flip = False
    elif isinstance(right, Cast) and isinstance(left, Literal):
        cast_side, lit_side = right, left
        flip = True
    else:
        return None

    if not isinstance(cast_side.operand, Column):
        return None
    cast_target = cast_side.dtype
    if not isinstance(cast_target, TimestampType) or cast_target.tz.is_naive():
        return None
    source_tz: "str | None" = None
    if column_field is not None:
        field = column_field(cast_side.operand)
        if field is not None:
            import pyarrow as pa

            if pa.types.is_timestamp(field.type) and field.type.tz:
                source_tz = field.type.tz
    if source_tz is None:
        # Fall back to the Column's bound field — the construction-time
        # path (no target schema available) relies on this.
        source_field = cast_side.operand.field
        if source_field is None:
            return None
        source_dtype = source_field.dtype
        if not isinstance(source_dtype, TimestampType):
            return None
        if source_dtype.tz.is_naive():
            return None
        source_tz = source_dtype.tz.iana
    if source_tz == cast_target.tz.iana:
        # Cast to the same tz is a no-op the smart ``cast`` factory
        # already swallows; but defensively short-circuit here too.
        if flip:
            return Comparison(lit_side, op, cast_side.operand)
        return Comparison(cast_side.operand, op, lit_side)
    value = lit_side.value
    import datetime as _dt

    if not isinstance(value, _dt.datetime):
        return None
    if value.tzinfo is None:
        # Naive literal — caller didn't tell us which tz; refuse to
        # guess. Keep the comparison as-is (the Cast renders at the
        # engine).
        return None
    try:
        import zoneinfo
    except ImportError:
        return None
    try:
        source_zone = zoneinfo.ZoneInfo(source_tz)
    except Exception:
        return None
    converted = value.astimezone(source_zone)
    source_dtype_lit = TimestampType(unit="us", tz=source_tz)
    new_literal = Literal(value=converted, dtype=source_dtype_lit)
    if flip:
        return Comparison(new_literal, op, cast_side.operand)
    return Comparison(cast_side.operand, op, new_literal)


def _rewrite_with_target_lookup(
    expr: "Expression",
    column_field: "Callable[[Column], Any | None]",
) -> "Expression":
    """Walk *expr* and apply target-schema-aware rewrites.

    ``column_field(column)`` returns a :class:`pyarrow.Field` for the
    matching column in the target (or ``None`` if absent). The field
    carries both the Arrow type (used for literal coercion + tz
    pushdown) and ``nullable`` (used for null-skip preemption).

    Rewrites fired per :class:`Comparison`, in order:

    1. **Cast-of-Column tz pushdown** —
       ``Cast(col_X, ts_tz_Y) op lit_Y`` with target column X in
       tz_Z: convert the literal to tz_Z and drop the Cast wrap.
    2. **Bare-column tz pushdown** — ``Column op tz_aware_lit``
       where the literal's tz differs from the target column's tz.
       Polars in particular refuses to compare two tz-aware
       Datetime columns with different time zones, so this is the
       difference between a working filter and a ``SchemaError``.
    3. **Literal coercion** — ``Column op Literal(value)`` where
       ``value``'s Python type doesn't match the target column's
       Arrow type and ``pa.scalar(value).cast(target_type,
       safe=True)`` succeeds: replace the literal with the cast
       result. Covers string→int, int→string, naive date→timestamp
       and any other lossless conversion ``pyarrow`` knows about.

    Returns *expr* unchanged when no rewrite triggers, so callers
    can use identity (``new is expr``) to skip downstream
    allocations on the steady-state "predicate already aligned with
    target" path.
    """
    if isinstance(expr, Comparison):
        new_left = _rewrite_with_target_lookup(expr.left, column_field)
        new_right = _rewrite_with_target_lookup(expr.right, column_field)
        rewrite = _try_timezone_pushdown(
            new_left, expr.op, new_right, column_field=column_field,
        )
        if rewrite is not None:
            return rewrite
        rewrite = _try_bare_tz_rewrite(
            new_left, expr.op, new_right, column_field,
        )
        if rewrite is not None:
            return rewrite
        rewrite = _try_literal_coercion(
            new_left, expr.op, new_right, column_field,
        )
        if rewrite is not None:
            return rewrite
        if new_left is expr.left and new_right is expr.right:
            return expr
        return Comparison(new_left, expr.op, new_right)
    if isinstance(expr, Logical):
        new_ops = tuple(
            _rewrite_with_target_lookup(o, column_field)
            for o in expr.operands
        )
        if all(a is b for a, b in zip(new_ops, expr.operands)):
            return expr
        return Logical(expr.op, new_ops)
    if isinstance(expr, Not):
        new_inner = _rewrite_with_target_lookup(expr.operand, column_field)
        if new_inner is expr.operand:
            return expr
        return Not(new_inner)
    if isinstance(expr, Between):
        new_target = _rewrite_with_target_lookup(expr.target, column_field)
        new_low = _rewrite_with_target_lookup(expr.low, column_field)
        new_high = _rewrite_with_target_lookup(expr.high, column_field)
        if (
            new_target is expr.target
            and new_low is expr.low
            and new_high is expr.high
        ):
            return expr
        return Between(new_target, new_low, new_high, negated=expr.negated)
    if isinstance(expr, InList):
        rewrite = _try_inlist_coercion(expr, column_field)
        if rewrite is not None:
            return rewrite
        return expr
    if isinstance(expr, IsNull):
        new_target = _rewrite_with_target_lookup(expr.target, column_field)
        if new_target is expr.target:
            return expr
        return IsNull(new_target, negated=expr.negated)
    return expr


def _try_bare_tz_rewrite(
    left: "Any",
    op: "CompareOp",
    right: "Any",
    column_field: "Callable[[Column], Any | None]",
) -> "Expression | None":
    """Rewrite ``Column op tz_aware_literal`` to the target column's tz.

    Mirrors :func:`_try_timezone_pushdown` for the no-Cast shape:
    when a comparison's column claims one tz (or none) and the
    target schema reports a different tz, convert the literal once
    to the target tz so the engine compares in storage-native units.
    Polars in particular refuses to compare two tz-aware Datetime
    columns with different time zones, so this rewrite is the
    difference between a working filter and a ``SchemaError`` for
    polars filters built against a mis-typed predicate.
    """
    if isinstance(left, Column) and isinstance(right, Literal):
        column, lit_side, flip = left, right, False
    elif isinstance(right, Column) and isinstance(left, Literal):
        column, lit_side, flip = right, left, True
    else:
        return None
    value = lit_side.value
    import datetime as _dt

    if not isinstance(value, _dt.datetime) or value.tzinfo is None:
        return None
    target_field = column_field(column)
    if target_field is None:
        return None
    import pyarrow as pa

    if not pa.types.is_timestamp(target_field.type) or not target_field.type.tz:
        return None
    target_tz = target_field.type.tz
    # Already aligned — keep the comparison shape unchanged so the
    # outer walk's identity check can skip the rebuild.
    try:
        import zoneinfo
    except ImportError:
        return None
    try:
        target_zone = zoneinfo.ZoneInfo(target_tz)
    except Exception:
        return None
    if value.tzinfo is target_zone or value.utcoffset() == target_zone.utcoffset(value):
        # Same physical zone — no conversion needed. Still strip a
        # stale ``Literal.dtype`` if it disagrees with target, so the
        # downstream emitter doesn't re-tag.
        if lit_side.dtype is None:
            return None
    converted = value.astimezone(target_zone)
    from yggdrasil.data.types.primitive.temporal import TimestampType

    new_literal = Literal(value=converted, dtype=TimestampType(unit="us", tz=target_tz))
    if flip:
        return Comparison(new_literal, op, column)
    return Comparison(column, op, new_literal)


def _try_literal_coercion(
    left: "Any",
    op: "CompareOp",
    right: "Any",
    column_field: "Callable[[Column], Any | None]",
) -> "Expression | None":
    """Rewrite ``Column op Literal(value)`` to match the target column's dtype.

    When the literal's Python type doesn't match the target's Arrow
    type and ``pyarrow.scalar(value).cast(target_type, safe=True)``
    succeeds, replace the literal with the cast result. ``safe=True``
    is what keeps the rewrite honest: pyarrow refuses lossy casts
    (float → int truncation, out-of-range to narrow int, …) and we
    keep the original predicate on refusal.

    Headline shapes this rewrites:

    * ``col(int_col) == "5"``  → ``col(int_col) == 5``
    * ``col(str_col) == 5``    → ``col(str_col) == "5"``
    * ``col(date_col) == "2026-01-01"`` → ``col(date_col) == date(2026,1,1)``
    * ``col(float_col) == 5``  → ``col(float_col) == 5.0``

    No-op when literal already matches (pyarrow scalar type ==
    target type) or when the predicate's other side isn't a bare
    :class:`Column`.
    """
    if isinstance(left, Column) and isinstance(right, Literal):
        column, lit_side, flip = left, right, False
    elif isinstance(right, Column) and isinstance(left, Literal):
        column, lit_side, flip = right, left, True
    else:
        return None
    field = column_field(column)
    if field is None:
        return None
    target_type = field.type
    converted = _safe_cast_literal(lit_side.value, target_type)
    if converted is None:
        return None
    if converted is lit_side.value:
        # Literal was already the right type — short-circuit so the
        # outer walk's identity check skips the rebuild.
        return None
    new_literal = Literal(value=converted, dtype=lit_side.dtype)
    if flip:
        return Comparison(new_literal, op, column)
    return Comparison(column, op, new_literal)


def _try_inlist_coercion(
    expr: "InList",
    column_field: "Callable[[Column], Any | None]",
) -> "InList | None":
    """Rewrite ``col.is_in([v1, v2, ...])`` to match the target column's dtype.

    Same logic as :func:`_try_literal_coercion` but applied per value
    in the :class:`InList`. Refuses to rewrite when *any* value fails
    to safe-cast — half-converted IN lists would change membership
    semantics depending on row dtype, which is worse than leaving the
    raw list and letting the engine coerce per row.
    """
    if not isinstance(expr.target, Column):
        return None
    field = column_field(expr.target)
    if field is None:
        return None
    target_type = field.type
    new_values: "list[Any]" = []
    any_changed = False
    for value in expr.values:
        converted = _safe_cast_literal(value, target_type)
        if converted is None:
            return None
        if converted is not value:
            any_changed = True
        new_values.append(converted)
    if not any_changed:
        return None
    return InList(
        target=expr.target,
        values=tuple(new_values),
        negated=expr.negated,
        includes_null=expr.includes_null,
    )


def _safe_cast_literal(value: "Any", target_type: "Any") -> "Any | None":
    """Try to cast *value* into *target_type* losslessly.

    Returns the cast value when pyarrow accepts the conversion under
    ``safe=True`` (no truncation, no out-of-range), the original
    *value* unchanged when the value already matches the target type
    (cheap short-circuit), or ``None`` when the conversion is unsafe
    / unsupported. The caller treats ``None`` as "keep the original
    predicate" and a returned value as "rewrite is safe".

    ``None`` (Python ``None``) is never rewritten — its semantics
    are SQL 3VL and depend on whether the comparison is EQ / IS NULL,
    so we leave that to the engine and the upstream null-skip rule.
    """
    if value is None:
        return None
    import pyarrow as pa

    try:
        scalar = pa.scalar(value)
    except (pa.ArrowTypeError, pa.ArrowInvalid, TypeError, ValueError):
        return None
    if scalar.type == target_type:
        return value  # identity short-circuit
    try:
        cast_scalar = scalar.cast(target_type, safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        return None
    return cast_scalar.as_py()


def _arrow_field_lookup(schema: "Any") -> "Callable[[Column], Any | None]":
    """Return a column → :class:`pyarrow.Field` lookup over a :class:`pyarrow.Schema`.

    The Field carries both the Arrow type (used by tz pushdown +
    literal coercion) and the ``nullable`` flag (reserved for the
    null-skip preemption when a target column is non-nullable).
    """

    def lookup(column: Column) -> "Any | None":
        try:
            return schema.field(column.name)
        except (KeyError, ValueError):
            return None

    return lookup


def _polars_field_lookup(schema: "Any") -> "Callable[[Column], Any | None]":
    """Return a column → :class:`pyarrow.Field` lookup over a :class:`polars.Schema`.

    Maps each polars dtype to its pyarrow equivalent via polars's own
    ``to_arrow`` machinery — pyarrow is the common currency the
    rewrite reasoning is written against.
    """
    import pyarrow as pa
    try:
        import polars as pl
    except ImportError:
        # No polars available — empty lookup is the safe fallback.
        return lambda _column: None

    # Cache a per-schema mapping so the walk doesn't re-translate
    # the polars dtype on every Column it visits.
    cache: "dict[str, Any]" = {}

    def lookup(column: Column) -> "Any | None":
        cached = cache.get(column.name, _MISSING)
        if cached is not _MISSING:
            return cached
        try:
            pl_dtype = schema[column.name]
        except (KeyError, TypeError):
            cache[column.name] = None
            return None
        try:
            arrow_type = pl.DataFrame({column.name: []}, schema={column.name: pl_dtype})\
                .to_arrow().schema.field(column.name).type
        except Exception:
            cache[column.name] = None
            return None
        field = pa.field(column.name, arrow_type)
        cache[column.name] = field
        return field

    return lookup


_MISSING = object()


def _try_collapse_or(
    operands: "Iterable[Expression]",
) -> "Expression | None":
    """Detect OR-of-(EQ | InList | IsNull) on the same target.

    Walks *operands* (descending into nested ``Logical(OR)`` so the
    left-leaning chain ``a | b | c`` is treated as one flat
    disjunction) and tries to consolidate every operand into a
    single :class:`InList`. Returns the merged InList when every
    operand was foldable AND shared the same structural target —
    that's the predicate-AST normalization the caller used to do by
    hand via the old ``simplify`` helper.

    Returns ``None`` for any shape that can't fully collapse:
    different targets, negated InList / IsNull, comparisons other
    than EQ, unhashable literal values. The caller then keeps the
    raw :class:`Logical(OR)` as-is.

    Null-aware: ``IsNull(c)`` contributes ``includes_null=True`` to
    the merged InList; ``c == None`` is *not* collapsible because
    SQL 3VL makes it UNKNOWN regardless of row value (rejected by
    WHERE), which differs from "matches NULL rows".
    """
    target: "Expression | None" = None
    target_hash: int = 0
    values: "list[Any]" = []
    includes_null = False
    seen_values: "set[Any]" = set()

    def _ingest(values_iter: "Iterable[Any]") -> bool:
        # Hashable-only collapse — an unhashable literal escapes the
        # frozenset path :meth:`Predicate._filter` later relies on.
        for v in values_iter:
            try:
                if v in seen_values:
                    continue
                seen_values.add(v)
            except TypeError:
                return False
            values.append(v)
        return True

    def _walk(items: "Iterable[Expression]") -> bool:
        nonlocal target, target_hash, includes_null
        for op in items:
            # Descend through nested same-op OR so the flat operand
            # list is what we collapse against.
            if isinstance(op, Logical) and op.op is LogicalOp.OR:
                if not _walk(op.operands):
                    return False
                continue
            this_target, this_values, this_null = _classify_or_operand(op)
            if this_target is None:
                return False
            if target is None:
                target = this_target
                target_hash = hash(target)
            else:
                # Structural target equality — same column / cast /
                # arithmetic shape. Hash short-circuit avoids the
                # field walk on the common case.
                if hash(this_target) != target_hash or not target.equals(this_target):
                    return False
            if this_null:
                includes_null = True
            if this_values and not _ingest(this_values):
                return False
        return True

    if not _walk(operands):
        return None
    if target is None:
        return None
    # Need at least two contributions to be a meaningful collapse —
    # a single EQ collapsing to a 1-element InList loses signal.
    if not values and not includes_null:
        return None
    if len(values) + (1 if includes_null else 0) < 2:
        return None
    return InList(
        target=target,
        values=tuple(values),
        negated=False,
        includes_null=includes_null,
    )


def _classify_or_operand(
    op: "Expression",
) -> "tuple[Expression | None, tuple[Any, ...], bool]":
    """Return ``(target, values, includes_null)`` for an OR-foldable operand.

    Foldable shapes:

    - ``Comparison(target, EQ, Literal(v))`` with ``v is not None``.
      ``Comparison(Literal(v), EQ, target)`` (literal-on-left) works
      the same way. We deliberately *do not* fold ``v is None`` —
      SQL 3VL makes ``c == None`` UNKNOWN regardless of row value.
    - ``InList(target, values, negated=False, includes_null=…)``.
    - ``IsNull(target, negated=False)`` — contributes only the
      ``includes_null=True`` flag.

    Anything else returns ``(None, (), False)`` and the caller
    bails out.
    """
    if isinstance(op, Comparison) and op.op is CompareOp.EQ:
        if isinstance(op.right, Literal):
            v = op.right.value
            if v is None:
                return None, (), False
            return op.left, (v,), False
        if isinstance(op.left, Literal):
            v = op.left.value
            if v is None:
                return None, (), False
            return op.right, (v,), False
        return None, (), False
    if isinstance(op, InList) and not op.negated:
        return op.target, op.values, op.includes_null
    if isinstance(op, IsNull) and not op.negated:
        return op.target, (), True
    return None, (), False


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


def _apply_inlist_pylist(
    rows: "Iterable[Any]",
    clauses: "list[tuple[str, frozenset, bool]]",
) -> "list[Any]":
    """Apply IN-list clauses to a stream of ``Mapping`` rows.

    Per row, every clause is one ``dict.get`` + one ``frozenset``
    membership check (or a null guard for ``includes_null=True``);
    skips compiling the full AST to a callable, so dispatch +
    closure overhead drops on tight ingest loops. Returns a list
    in input order so the caller can index / slice it.
    """
    out: "list[Any]" = []
    # Hoist clause tuples into parallel arrays so the per-row hot
    # loop avoids unpacking three values N times.
    cols = [c for c, _, _ in clauses]
    sets = [s for _, s, _ in clauses]
    nulls = [n for _, _, n in clauses]
    n = len(clauses)
    for row in rows:
        keep = True
        for i in range(n):
            value = row.get(cols[i])
            if value is None:
                if not nulls[i]:
                    keep = False
                    break
            elif value not in sets[i]:
                keep = False
                break
        if keep:
            out.append(row)
    return out


def _materialize_view_columns(target: "Any") -> "Any":
    """Replace Arrow ``string_view`` / ``binary_view`` columns with their
    materialized (``utf8`` / ``binary``) equivalents.

    PyArrow's ``filter`` / ``take`` kernels have no ``*_view`` support in the
    runtimes bundled with some platforms (e.g. Databricks), so applying a
    boolean mask over a ``string_view`` column raises
    ``ArrowNotImplementedError: Function 'array_take' has no kernel matching
    input types (string_view, ...)``. Materializing the offending columns
    first keeps :meth:`filter` on the supported path. *target* (a
    :class:`pa.Table` or :class:`pa.RecordBatch`) is returned unchanged when
    it carries no view columns. Detection is by type name so it works even on
    pyarrow builds that predate the ``pa.types.is_string_view`` predicate.
    """
    import pyarrow as pa

    plain_for = {"string_view": pa.string(), "binary_view": pa.binary()}
    schema = target.schema
    replacements: "dict[int, Any]" = {}
    for index, field in enumerate(schema):
        plain = plain_for.get(str(field.type))
        if plain is not None:
            replacements[index] = field.with_type(plain)
    if not replacements:
        return target

    new_schema = pa.schema(
        [replacements.get(i, f) for i, f in enumerate(schema)],
        metadata=schema.metadata,
    )
    if hasattr(target, "cast"):  # pa.Table.cast(schema)
        return target.cast(new_schema)
    # pa.RecordBatch has no .cast(schema) — rebuild from per-column casts.
    columns = [
        target.column(i).cast(new_schema.field(i).type) if i in replacements
        else target.column(i)
        for i in range(target.num_columns)
    ]
    return pa.RecordBatch.from_arrays(columns, schema=new_schema)


def _apply_inlist(
    target: "Any",
    clauses: "list[tuple[str, frozenset, bool]]",
) -> "Any":
    """Apply IN-list clauses to a :class:`pa.RecordBatch` or :class:`pa.Table`.

    Two paths, chosen by row count:

    * **≤ 4 rows** — per-row Python probe. One ``column[i].as_py()``
      + one ``frozenset`` membership check per clause per row. The
      constant is tuned so cache-lookup-style 1-row batches pay
      ~1.5 µs instead of the ~16 µs ``pc.is_in`` kernel dispatch
      overhead.
    * **> 4 rows** — vectorised ``pyarrow.compute.is_in`` + ``and_kleene``.
      One kernel call per clause, no ``pa.Expression`` compile.
      Faster than ``target.filter(pa.Expression)`` across all sizes.

    Both :class:`pa.Table` and :class:`pa.RecordBatch` accept a
    boolean mask in :meth:`filter`, so the helper is duck-typed.
    """
    if not clauses:
        return target
    n = target.num_rows
    if n <= 4:
        return _apply_inlist_small(target, clauses, n)
    import pyarrow as pa
    import pyarrow.compute as pc

    mask: "Any | None" = None
    for column, value_set, includes_null in clauses:
        col_arr = target.column(column)
        m = pc.is_in(col_arr, value_set=pa.array(list(value_set)))
        if includes_null:
            m = pc.or_kleene(m, pc.is_null(col_arr))
        mask = m if mask is None else pc.and_kleene(mask, m)
    return _materialize_view_columns(target).filter(mask)


def _apply_inlist_small(
    target: "Any",
    clauses: "list[tuple[str, frozenset, bool]]",
    n: int,
) -> "Any":
    """Fast path for tiny batches (≤ 4 rows).

    One ``column[i].as_py()`` + ``frozenset`` probe per clause per
    row. For a 1-row cache-lookup batch this runs in ~1.5 µs — the
    ``pc.is_in`` kernel dispatch alone costs ~16 µs.
    """
    import pyarrow as pa

    surviving: "list[int] | None" = None
    for col_name, value_set, includes_null in clauses:
        try:
            col_arr = target.column(col_name)
        except KeyError:
            return target
        keep: "list[int]" = []
        for i in (range(n) if surviving is None else surviving):
            v = col_arr[i].as_py()
            if v is None:
                if includes_null:
                    keep.append(i)
            elif v in value_set:
                keep.append(i)
        surviving = keep
        if not surviving:
            return target.slice(0, 0)
    if surviving is None or len(surviving) == n:
        return target
    return target.take(pa.array(surviving, type=pa.int64()))


# ---------------------------------------------------------------------------
# Leaf nodes — Column, Literal
# ---------------------------------------------------------------------------


class Column(Expression):
    """A reference to a column inside an expression tree.

    Owns the *expression-side* lookup state: the column ``name`` the
    backend resolves on the frame, an optional SQL ``alias`` rename
    (so ``col("foo").with_alias("bar")`` renders as ``foo AS bar``),
    and the table-level ``qualifier`` for ``T.col`` style addressing
    inside a MERGE / aliased SQL query.

    ``field`` is the *origin* metadata — the typed :class:`Field`
    the column was sourced from (dtype, nullability, source-schema
    children). Backends consult it for typed-literal casts and
    engine-flavoured dtypes, but it deliberately does **not** carry
    expression-side knobs like the table qualifier — those would
    bleed back into Field consumers (schema diffs, cast registry,
    pickle round-trips) that have no business with them.
    """

    __slots__ = ("name", "field", "alias", "qualifier")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = (
        "name", "field", "alias", "qualifier",
    )

    def __init__(
        self,
        name: str,
        field: "Field | None" = None,
        alias: "str | None" = None,
        qualifier: "str | None" = None,
    ) -> None:
        self.name = name
        self.field = field
        self.alias = alias
        self.qualifier = qualifier

    @property
    def dtype(self) -> "DataType | None":
        return self.field.dtype if self.field is not None else None


class Literal(Expression):
    """A scalar literal value.

    ``dtype`` is optional — when left ``None`` the backend infers
    from the Python value. Pinning a dtype is useful when the
    inferred type would be wrong (e.g. naive ``datetime`` you want
    rendered as ``DATE`` instead of ``TIMESTAMP``).
    """

    __slots__ = ("value", "dtype")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("value", "dtype")

    def __init__(
        self,
        value: Any,
        dtype: "DataType | None" = None,
    ) -> None:
        self.value = value
        self.dtype = dtype


def lit(value: Any, dtype: "DataType | None" = None) -> Literal:
    """Build a :class:`Literal`. Convenience for the builder API."""
    return Literal(value=value, dtype=dtype)


# ---------------------------------------------------------------------------
# Boolean operators — every node here inherits ``Predicate``
# ---------------------------------------------------------------------------


class Comparison(Predicate):
    __slots__ = ("left", "op", "right")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("left", "op", "right")

    def __init__(
        self,
        left: Expression,
        op: CompareOp,
        right: Expression,
    ) -> None:
        self.left = left
        self.op = op
        self.right = right


class Logical(Predicate):
    __slots__ = ("op", "operands")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("op", "operands")

    def __new__(
        cls,
        op: LogicalOp = LogicalOp.AND,
        operands: "Iterable[Expression]" = (),
    ):
        # Smart constructor — absorbs the cheap rewrites callers used
        # to remember to apply by hand:
        #
        # * OR-of-(EQ | InList | IsNull) on the same target merges
        #   into one :class:`InList` so emitters never see a wide
        #   ``c = v1 OR c = v2 OR ...`` chain.
        # * Single-operand Logical unwraps to its only operand.
        #
        # Returning a non-:class:`Logical` from ``__new__`` skips
        # Python's normal ``__init__`` dispatch on the result, so the
        # collapse / unwrap path stays a single allocation.
        if op is LogicalOp.OR:
            collapsed = _try_collapse_or(operands)
            if collapsed is not None:
                return collapsed
        operands_tuple = operands if isinstance(operands, tuple) else tuple(operands)
        if len(operands_tuple) == 1:
            return operands_tuple[0]
        return object.__new__(cls)

    def __init__(
        self,
        op: LogicalOp = LogicalOp.AND,
        operands: "Iterable[Expression]" = (),
    ) -> None:
        # Same-op flatten — inline a child ``Logical`` with the same
        # operator so the tree stays right-leaning regardless of how
        # Python's left-associative ``|`` / ``&`` built it.
        # ``(a | b) | c`` and ``a | (b | c)`` both land as
        # ``Logical(OR, (a, b, c))``. Skip the rebuild when the
        # operands are already canonical: the common case (every
        # operand is a non-Logical leaf, the caller already handed us
        # a tuple) costs one type check.
        operands_tuple = operands if isinstance(operands, tuple) else tuple(operands)
        if not operands_tuple:
            raise ValueError(f"Logical {op.value} needs at least one operand.")
        needs_flatten = False
        for operand in operands_tuple:
            if isinstance(operand, Logical) and operand.op is op:
                needs_flatten = True
                break
        if needs_flatten:
            flat: "list[Expression]" = []
            for operand in operands_tuple:
                if isinstance(operand, Logical) and operand.op is op:
                    flat.extend(operand.operands)
                else:
                    flat.append(operand)
            operands_tuple = tuple(flat)
        self.op = op
        self.operands = operands_tuple


class Not(Predicate):
    __slots__ = ("operand",)
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("operand",)

    def __init__(self, operand: Expression) -> None:
        self.operand = operand


class Between(Predicate):
    """``column BETWEEN low AND high`` — inclusive on both bounds.

    ``negated=True`` flips to ``NOT BETWEEN``. The Python and
    pyarrow emitters honour the inclusive contract; SQL renders
    natively.
    """

    __slots__ = ("target", "low", "high", "negated")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = (
        "target", "low", "high", "negated",
    )

    def __init__(
        self,
        target: Expression,
        low: Expression,
        high: Expression,
        negated: bool = False,
    ) -> None:
        self.target = target
        self.low = low
        self.high = high
        self.negated = negated


class InList(Predicate):
    """``column IN (...)`` against a finite literal value list.

    Values are stored as tuples of literal Python objects so the
    node remains hashable. The :attr:`includes_null` flag carries
    forward through round-trips — backends that don't natively
    handle ``NULL`` inside ``IN`` (most SQL dialects) expand to
    ``... OR col IS NULL``.
    """

    __slots__ = ("target", "values", "negated", "includes_null")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = (
        "target", "values", "negated", "includes_null",
    )

    def __init__(
        self,
        target: Expression,
        values: "Iterable[Any]" = (),
        negated: bool = False,
        includes_null: bool = False,
    ) -> None:
        # Dedupe in first-seen order — hashable values use a ``set``
        # fast path; the unhashable branch falls back to a linear
        # ``in out`` scan so dicts / lists still land deterministically.
        # The latter is O(n²) but only fires when the caller seeded
        # the InList with unhashable types — uncommon in practice.
        if not isinstance(values, tuple):
            values = tuple(values)
        seen: "set[Any]" = set()
        deduped: "list[Any] | None" = None
        for i, v in enumerate(values):
            try:
                if v in seen:
                    if deduped is None:
                        deduped = list(values[:i])
                    continue
                seen.add(v)
            except TypeError:
                if deduped is None:
                    deduped = list(values[:i])
                if v in deduped:
                    continue
                deduped.append(v)
                continue
            if deduped is not None:
                deduped.append(v)
        self.target = target
        self.values = tuple(deduped) if deduped is not None else values
        self.negated = negated
        self.includes_null = includes_null


class IsNull(Predicate):
    __slots__ = ("target", "negated")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("target", "negated")

    def __init__(self, target: Expression, negated: bool = False) -> None:
        self.target = target
        self.negated = negated  # ``IS NOT NULL`` when True.


class Like(Predicate):
    """SQL-style ``LIKE`` / ``ILIKE``.

    Pattern uses ``%`` and ``_`` as wildcards. Set ``case_insensitive``
    for ``ILIKE`` semantics; ``negated`` for ``NOT LIKE``.
    """

    __slots__ = ("target", "pattern", "case_insensitive", "negated")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = (
        "target", "pattern", "case_insensitive", "negated",
    )

    def __init__(
        self,
        target: Expression,
        pattern: str,
        case_insensitive: bool = False,
        negated: bool = False,
    ) -> None:
        self.target = target
        self.pattern = pattern
        self.case_insensitive = case_insensitive
        self.negated = negated


# ---------------------------------------------------------------------------
# Scalar combinators — Cast, Arithmetic
# ---------------------------------------------------------------------------


class Cast(Expression):
    """Explicit type cast. Returns a typed scalar expression."""

    __slots__ = ("operand", "dtype")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("operand", "dtype")

    def __init__(self, operand: Expression, dtype: "DataType") -> None:
        self.operand = operand
        self.dtype = dtype


class Arithmetic(Expression):
    """Two-operand arithmetic. Result type is the widened operand type."""

    __slots__ = ("op", "left", "right")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("op", "left", "right")

    def __init__(
        self,
        op: ArithmeticOp,
        left: Expression,
        right: Expression,
    ) -> None:
        self.op = op
        self.left = left
        self.right = right


# ---------------------------------------------------------------------------
# Extended expression nodes — function calls, aliasing, ordering,
# windowing, conditional (CASE), subscript.
# ---------------------------------------------------------------------------


class FunctionCall(Expression):
    """Arbitrary function call — ``UPPER(col)``, ``COUNT(DISTINCT col)``, etc.

    ``name`` is stored upper-cased so comparisons are case-insensitive.
    ``args`` are the positional arguments as Expression nodes.
    ``distinct`` renders ``DISTINCT`` inside the parentheses (aggregate
    functions like ``COUNT(DISTINCT …)``).
    """

    __slots__ = ("name", "args", "distinct")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("name", "args", "distinct")

    def __init__(
        self,
        name: str,
        args: "tuple[Expression, ...]" = (),
        distinct: bool = False,
    ) -> None:
        self.name = name.upper()
        self.args = args if isinstance(args, tuple) else tuple(args)
        self.distinct = distinct


class Star(Expression):
    """``SELECT *`` or ``COUNT(*)`` — a bare wildcard reference.

    ``qualifier`` handles ``table.*`` style addressing:
    ``Star(qualifier="t")`` renders as ``t.*``.
    """

    __slots__ = ("qualifier",)
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("qualifier",)

    def __init__(self, qualifier: "str | None" = None) -> None:
        self.qualifier = qualifier


class Alias(Expression):
    """``expr AS name`` — wraps any expression with a user-visible alias.

    Used in SELECT lists, sub-expressions, and CTE column naming.
    The inner ``expr`` carries the computation; ``name`` is the
    output label.
    """

    __slots__ = ("expr", "name")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("expr", "name")

    def __init__(self, expr: Expression, name: str) -> None:
        self.expr = expr
        self.name = name


class SortOrder(Expression):
    """``ORDER BY expr [ASC|DESC] [NULLS FIRST|LAST]``.

    ``ascending=True`` for ASC, ``False`` for DESC.
    ``nulls_first=None`` leaves the null ordering to the engine
    default; explicit ``True`` / ``False`` renders ``NULLS FIRST``
    / ``NULLS LAST``.
    """

    __slots__ = ("expr", "ascending", "nulls_first")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("expr", "ascending", "nulls_first")

    def __init__(
        self,
        expr: Expression,
        ascending: bool = True,
        nulls_first: "bool | None" = None,
    ) -> None:
        self.expr = expr
        self.ascending = ascending
        self.nulls_first = nulls_first


class WindowSpec(Expression):
    """``OVER (PARTITION BY … ORDER BY … ROWS BETWEEN … AND …)``.

    ``partition_by`` and ``order_by`` are tuples of Expression /
    SortOrder nodes respectively. ``frame_start`` / ``frame_end``
    are raw SQL frame-boundary strings (``"UNBOUNDED PRECEDING"``,
    ``"CURRENT ROW"``, ``"3 PRECEDING"``, etc.) — kept as strings
    because the combinatorial space of frame specs is huge and
    rarely manipulated programmatically.
    """

    __slots__ = ("partition_by", "order_by", "frame_start", "frame_end")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = (
        "partition_by", "order_by", "frame_start", "frame_end",
    )

    def __init__(
        self,
        partition_by: "tuple[Expression, ...]" = (),
        order_by: "tuple[SortOrder, ...]" = (),
        frame_start: "str | None" = None,
        frame_end: "str | None" = None,
    ) -> None:
        self.partition_by = partition_by if isinstance(partition_by, tuple) else tuple(partition_by)
        self.order_by = order_by if isinstance(order_by, tuple) else tuple(order_by)
        self.frame_start = frame_start
        self.frame_end = frame_end


class WindowFunction(Expression):
    """``function OVER window_spec`` — a windowed aggregate or ranking call.

    ``function`` is typically a :class:`FunctionCall` (e.g.
    ``ROW_NUMBER()``, ``SUM(col)``). ``window`` carries the
    ``OVER (…)`` specification.
    """

    __slots__ = ("function", "window")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("function", "window")

    def __init__(self, function: Expression, window: WindowSpec) -> None:
        self.function = function
        self.window = window


class CaseWhen(Expression):
    """``CASE [operand] WHEN cond THEN val … [ELSE val] END``.

    Two forms:

    * **Searched** (``operand=None``): each branch condition is a
      full boolean expression.
    * **Simple** (``operand=expr``): each branch condition is
      compared against ``operand`` with implicit equality.

    ``branches`` is a tuple of ``(condition, result)`` pairs.
    ``else_expr`` is the fallback value (``None`` ⇒ implicit
    ``NULL``).
    """

    __slots__ = ("operand", "branches", "else_expr")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("operand", "branches", "else_expr")

    def __init__(
        self,
        branches: "tuple[tuple[Expression, Expression], ...]",
        else_expr: "Expression | None" = None,
        operand: "Expression | None" = None,
    ) -> None:
        self.operand = operand
        self.branches = branches if isinstance(branches, tuple) else tuple(branches)
        self.else_expr = else_expr


class Subscript(Expression):
    """``expr[index]`` — array element access or map key lookup.

    ``expr`` is the collection expression; ``index`` is the key /
    offset expression. SQL renders as ``expr[index]``.
    """

    __slots__ = ("expr", "index")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("expr", "index")

    def __init__(self, expr: Expression, index: Expression) -> None:
        self.expr = expr
        self.index = index


class Lambda(Expression):
    """``(p1, p2, ...) -> body`` — a lambda for higher-order functions.

    Databricks higher-order functions (TRANSFORM, FILTER, AGGREGATE,
    EXISTS, FORALL, ZIP_WITH, REDUCE) take a lambda as one of their
    arguments. ``params`` are the parameter names; ``body`` is the
    expression evaluated per element. SQL renders as
    ``(p1, p2) -> body`` (parens omitted for single-param lambdas).
    """

    __slots__ = ("params", "body")
    _FIELD_NAMES: ClassVar["tuple[str, ...]"] = ("params", "body")

    def __init__(self, params: "tuple[str, ...]", body: Expression) -> None:
        self.params = params if isinstance(params, tuple) else tuple(params)
        self.body = body


# ---------------------------------------------------------------------------
# Coercion — let users pass plain Python values where an Expression
# is expected (most common case is ``col("x") == 5``).
# ---------------------------------------------------------------------------


ExpressionLike = Union[Expression, Any]

#: Anything :meth:`Expression.from_` accepts as a predicate input —
#: a yggdrasil :class:`Predicate`, a yggdrasil :class:`Expression`,
#: a SQL string, or a foreign engine expression
#: (``pyarrow.compute.Expression`` / ``polars.Expr`` /
#: ``pyspark.sql.Column``). Use this on public surfaces that accept
#: "anything that can be lifted to a predicate" — internal call
#: sites that have already routed through :meth:`Expression.from_`
#: should narrow to :class:`Predicate`.
PredicateLike = Union["Predicate", Expression, str, Any]


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
