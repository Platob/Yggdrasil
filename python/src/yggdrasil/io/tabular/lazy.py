"""Lazy :class:`Tabular` wrapper backed by an :class:`ExecutionPlan`.

:class:`LazyTabular` wraps a *source* :class:`Tabular` and a *plan* —
an :class:`ExecutionPlan` of pending transformations
(:meth:`select`, :meth:`filter` / :meth:`where`, :meth:`group_by`,
:meth:`apply`). Nothing runs until something pulls data
(``read_arrow_batches``, ``read_arrow_table``, ``collect_schema``,
``read_polars_frame``, …). Each builder call returns a new
:class:`LazyTabular` carrying an extended plan; the original instance
is never mutated, so chains like ``lt.where(...).where(...).select(...)``
are safe.

The plan is the single intermediate representation. Predicates are
canonicalized through :class:`yggdrasil.io.tabular.execution.expr.Expression`
when the input is a SQL string or a yggdrasil node, and kept
backend-native otherwise (round-tripping a polars ``Expr`` through
the AST loses dtype information). Adjacent :class:`Filter` nodes fuse
inside :meth:`ExecutionPlan.append`, so two stacked ``where`` calls
share one op and the merged form is what introspecting ``.plan``
shows.

Pushdown
--------

Execution routes through a polars :class:`LazyFrame` scan of the
source. Polars' query planner pushes projections and predicates into
the underlying pyarrow dataset whenever the source exposes one (the
default :meth:`Tabular._scan_polars_frame` returns
``pl.scan_pyarrow_dataset(...)``).

Writes are forwarded to the source as-is. The plan describes a *view*
of the source; it does not apply on the write path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Union

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.io.tabular.execution.expr import Expression, Predicate
from yggdrasil.io.tabular.execution.plan import (
    Apply,
    ExecutionPlan,
    Filter,
    GroupByAgg,
    Join,
    PlanOp,
    Select,
)
from yggdrasil.lazy_imports import polars_module

if TYPE_CHECKING:
    import polars as pl


__all__ = ["LazyTabular", "lazy_for"]


# Process-wide registry mapping a source :class:`Tabular` subclass to
# the :class:`LazyTabular` that specializes for it. Populated by
# :meth:`LazyTabular.__init_subclass__` whenever a subclass declares
# :attr:`LazyTabular.source_cls`. Mirror of
# :data:`yggdrasil.io.tabular.base._TABULAR_REGISTRY` — same shape,
# same rules: declarative class-level discriminator, populated at
# import time, looked up by :func:`lazy_for`.
_LAZY_REGISTRY: "dict[type[Tabular], type[LazyTabular]]" = {}


def lazy_for(source: Tabular, plan: "ExecutionPlan") -> "LazyTabular":
    """Build the most specific :class:`LazyTabular` subclass for *source*.

    Walks ``type(source).__mro__`` looking for an entry in
    :data:`_LAZY_REGISTRY`; falls back to the plain :class:`LazyTabular`
    when nothing matches. Format-specific Lazy IO subclasses
    (``LazyParquetFile`` / ``LazyArrowIPCFile`` / ``LazyFolder``
    / …) live next to their source class and register themselves
    on import — so any *source* a caller can construct has its
    matching Lazy subclass loaded by the time this is called.
    """
    for cls in type(source).__mro__:
        hit = _LAZY_REGISTRY.get(cls)
        if hit is not None:
            return hit(source, plan=plan)
    return LazyTabular(source, plan=plan)


# A "selector" is anything that fronts a column / column-expression: a
# bare column name, a yggdrasil :class:`Expression`, or a backend-
# native expression that ``Expression.from_`` could lift. We keep
# strings as strings (column names, not SQL fragments) and yggdrasil
# nodes as nodes; native exprs pass through.
_SelectorIn = Union[str, Expression, Any]


def _coerce_keys(value: Any) -> "tuple[Any, ...]":
    """Normalize a join-keys argument to a tuple.

    Accepts ``None`` / a single column name / an iterable of column
    names or expressions. A bare string is one key (not a sequence
    of single-character keys).
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _normalize_filter(value: Any) -> Any:
    """Normalize a filter input to its canonical stored form.

    - :class:`Expression`: kept as-is (must be a :class:`Predicate`).
    - ``str``: parsed as a SQL predicate via :meth:`Expression.from_sql`.
    - Anything else: kept backend-native (polars / pyarrow / pyspark).
    """
    if isinstance(value, Expression):
        if not isinstance(value, Predicate):
            raise TypeError(
                f"LazyTabular.filter expected a boolean expression "
                f"(predicate); got {type(value).__name__}: {value!r}. "
                "Use a comparison (col('x') > 1), is_in / between / "
                "is_null, or a SQL predicate string."
            )
        return value
    if isinstance(value, str):
        expr = Expression.from_sql(value)
        if not isinstance(expr, Predicate):
            raise TypeError(
                f"LazyTabular.filter expected a SQL predicate string; "
                f"{value!r} parses to a non-predicate "
                f"{type(expr).__name__}."
            )
        return expr
    return value


class LazyTabular(Tabular[CastOptions]):
    """:class:`Tabular` view over a source plus an :class:`ExecutionPlan`.

    The source is the I/O truth — schema, batches, and writes all
    originate from it. The plan is replayed on every read against a
    polars LazyFrame scan of the source, so engine-side pushdown
    happens automatically as long as the source exposes a pyarrow
    dataset.

    Builder methods (:meth:`select`, :meth:`filter` / :meth:`where`,
    :meth:`group_by`, :meth:`apply`) return new instances carrying
    extended plans. They preserve the concrete subclass via
    :meth:`_clone`, so subclasses like :class:`UnionTabular` keep
    their state across chained calls.
    """

    #: Source :class:`Tabular` subclass this Lazy specializes for.
    #: Concrete subclasses set to a registered leaf (``ParquetFile`` /
    #: ``Folder`` / …) so :func:`lazy_for` can pick this class when
    #: wrapping a source of the matching type. ``None`` (the default)
    #: opts out of registration — :class:`LazyTabular` itself stays
    #: out so it remains the universal fallback.
    source_cls: ClassVar["type[Tabular] | None"] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        sc = cls.source_cls
        if sc is None:
            return
        existing = _LAZY_REGISTRY.get(sc)
        if existing is not None and existing is not cls:
            raise RuntimeError(
                f"Duplicate LazyTabular registration for {sc.__name__}: "
                f"{cls.__name__} clashes with {existing.__name__}. If "
                f"the override is intentional, clear the slot first via "
                f"_LAZY_REGISTRY.pop(...) at module-load time."
            )
        _LAZY_REGISTRY[sc] = cls

    def __init__(
        self,
        source: Tabular,
        *,
        plan: "ExecutionPlan | Iterable[PlanOp] | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._source: Tabular = source
        self._plan: ExecutionPlan = (
            plan if isinstance(plan, ExecutionPlan)
            else ExecutionPlan(tuple(plan)) if plan is not None
            else ExecutionPlan.empty()
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(source={self._source!r}, "
            f"plan={self._plan!r})"
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def source(self) -> Tabular:
        return self._source

    @property
    def plan(self) -> ExecutionPlan:
        return self._plan

    @classmethod
    def options_class(cls) -> "type[CastOptions]":
        return CastOptions

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def _clone(self, plan: ExecutionPlan) -> "LazyTabular":
        """Build a sibling instance carrying *plan*.

        Hook for subclasses (:class:`UnionTabular`, …) that need to
        preserve their own state when chaining methods. Default
        rebuilds the same kind of :class:`LazyTabular` over the same
        source.
        """
        return type(self)(self._source, plan=plan)

    def execute_plan(
        self,
        plan: Any,
        *,
        options: "CastOptions | None" = None,
        **kwargs: Any,
    ) -> "LazyTabular":
        """Append *plan*'s ops onto this LazyTabular's plan.

        Stacking two :class:`LazyTabular` wrappers is wasteful when we
        can just fold the second plan's ops into the first — adjacent
        :class:`Filter` nodes still fuse via :meth:`ExecutionPlan.append`.
        """
        coerced = (
            plan if isinstance(plan, ExecutionPlan)
            else ExecutionPlan(tuple(plan)) if plan is not None
            else ExecutionPlan.empty()
        )
        if coerced.is_empty():
            return self
        del options, kwargs
        return self._clone(self._plan.extend(coerced))

    def _append_op(self, op: PlanOp) -> "LazyTabular":
        return self._clone(self._plan.append(op))

    def lazy(self) -> "LazyTabular":
        """Already lazy — return ``self`` instead of re-wrapping.

        Overrides :meth:`Tabular.lazy`, which would otherwise stack a
        second :class:`LazyTabular` carrying a ``SELECT *`` plan on top
        of this one.
        """
        return self

    def select(self, *columns: _SelectorIn) -> "LazyTabular":
        """Project to *columns*. Accepts column names or expressions."""
        if not columns:
            raise ValueError(
                "LazyTabular.select needs at least one column; got none. "
                "Pass column names ('a', 'b'), yggdrasil expressions "
                "(col('a').alias('x')), or polars / pyarrow expressions."
            )
        return self._append_op(Select(tuple(columns)))

    def filter(self, *predicates: Any) -> "LazyTabular":
        """Row filter. Adjacent :class:`Filter` ops fuse via
        :meth:`Filter.extend` (yggdrasil predicates AND-merge into one
        :class:`Logical`; native predicates stack)."""
        if not predicates:
            raise ValueError(
                "LazyTabular.filter needs at least one predicate; got none. "
                "Pass a SQL string ('x > 0'), a yggdrasil predicate "
                "(col('x') > 0), or a polars / pyarrow expression."
            )
        normalized = tuple(_normalize_filter(p) for p in predicates)
        return self._append_op(Filter(normalized))

    where = filter

    def group_by(self, *by: _SelectorIn) -> "_LazyGroupBy":
        """Start a group-by. Finalize with :meth:`_LazyGroupBy.agg`."""
        if not by:
            raise ValueError(
                "LazyTabular.group_by needs at least one key; got none. "
                "Pass column names or expressions."
            )
        return _LazyGroupBy(self, tuple(by))

    groupby = group_by

    def join(
        self,
        right: Any,
        on: "str | Iterable[Any] | None" = None,
        how: str = "inner",
        *,
        left_on: "str | Iterable[Any] | None" = None,
        right_on: "str | Iterable[Any] | None" = None,
        suffix: str = "_right",
    ) -> "LazyTabular":
        """Join with *right*.

        *right* accepts a :class:`Tabular`, a polars ``DataFrame`` /
        ``LazyFrame``, or a string name to resolve against
        :data:`yggdrasil.io.tabular.engine.SYSTEM_ENGINE` at apply
        time. ``on`` (or the symmetric ``left_on`` / ``right_on``)
        gives the join keys; ``how`` is any value polars accepts
        (``inner``, ``left``, ``right``, ``full``, ``cross``,
        ``semi``, ``anti``).
        """
        on_keys = _coerce_keys(on)
        left_keys = _coerce_keys(left_on)
        right_keys = _coerce_keys(right_on)
        if how != "cross" and not (on_keys or (left_keys and right_keys)):
            raise ValueError(
                "LazyTabular.join needs ``on`` or both ``left_on`` / "
                f"``right_on`` unless how='cross'; got on={on!r}, "
                f"left_on={left_on!r}, right_on={right_on!r}."
            )
        return self._append_op(
            Join(
                right=right,
                on=on_keys,
                how=how,
                left_on=left_keys,
                right_on=right_keys,
                suffix=suffix,
            )
        )

    def apply(self, fn: Any) -> "LazyTabular":
        """Escape hatch: ``fn(LazyFrame) -> LazyFrame``.

        Schema-changing and non-commutative — anything after this
        op runs post-union in :class:`UnionTabular`.
        """
        if not callable(fn):
            raise TypeError(
                f"LazyTabular.apply expected a callable LazyFrame -> "
                f"LazyFrame; got {type(fn).__name__}: {fn!r}."
            )
        return self._append_op(Apply(fn))

    # ------------------------------------------------------------------
    # Native-arrow pushdown helpers — used by format-specific subclasses
    # ------------------------------------------------------------------

    def _local_path_str(self) -> "str | None":
        """Backend-native local-path string for the source's holder, else ``None``.

        Looks for the conventional :meth:`is_local_path` /
        :meth:`full_path` pair on the source's ``_holder``. Format
        Lazy subclasses (``LazyParquetFile`` / ``LazyArrowIPCFile``
        / ``LazyCsvFile``) use this to decide whether the
        :mod:`pyarrow.dataset` scanner can take the path directly
        — that's what enables column / predicate pushdown into the
        format reader. Returns ``None`` for in-memory holders or
        anything that doesn't expose a path; subclasses then route
        through their in-memory fallback.
        """
        src = self._source
        holder = getattr(src, "_holder", None)
        if holder is None or not getattr(holder, "is_local_path", False):
            return None
        full_path = getattr(holder, "full_path", None)
        return full_path() if callable(full_path) else None

    def _arrow_pushdown_spec(
        self,
    ) -> "tuple[list[str] | None, Any | None] | None":
        """Compile the plan to ``(columns, filter_expr)`` for pyarrow.

        Walks :attr:`plan` accumulating column projection from
        :class:`Select` ops (bare column names only) and an
        AND-combined :class:`pyarrow.compute.Expression` from
        :class:`Filter` ops with yggdrasil :class:`Predicate`
        bodies. Returns ``None`` (== "not pushdownable") on the
        first op that breaks the pattern: a ``Select`` carrying an
        expression instead of a bare name, ``SELECT *``, a
        ``Filter`` with a backend-native predicate that can't lift
        to pyarrow, or any of :class:`GroupByAgg` /
        :class:`Apply` / :class:`Join`.

        Format-specific Lazy subclasses call this from their
        :meth:`_read_arrow_batches` override — when it returns
        ``None`` they fall back to the base polars-LazyFrame path.
        """
        columns: "list[str] | None" = None
        filter_expr: Any = None
        for op in self._plan.ops:
            if isinstance(op, Select):
                cols: "list[str]" = []
                for c in op.columns:
                    if not isinstance(c, str) or c == "*":
                        return None
                    cols.append(c)
                # A later Select narrows further; the final projection wins.
                columns = cols
            elif isinstance(op, Filter):
                for p in op.predicates:
                    if not isinstance(p, Predicate):
                        return None
                    try:
                        e = p.to_arrow()
                    except Exception:
                        return None
                    filter_expr = e if filter_expr is None else filter_expr & e
            else:
                return None
        return columns, filter_expr

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------

    def _build_lazy(self, options: CastOptions) -> "pl.LazyFrame":
        lf = self._source._scan_polars_frame(options)
        return self._plan.apply_polars(lf)

    # ------------------------------------------------------------------
    # Tabular contract — read hooks
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self, options: CastOptions,
    ) -> Iterator[pa.RecordBatch]:
        if self._plan.is_empty():
            yield from self._source._read_arrow_batches(options)
            return

        lf = self._build_lazy(options)
        table: pa.Table = lf.collect().to_arrow()
        row_size = getattr(options, "row_size", None) or None
        for batch in table.to_batches(max_chunksize=row_size):
            yield options.cast_arrow_tabular(batch)

    def _scan_polars_frame(self, options: CastOptions) -> "pl.LazyFrame":
        return self._build_lazy(options)

    def _read_polars_frame(self, options: CastOptions) -> "pl.DataFrame":
        if self._plan.is_empty():
            return self._source._read_polars_frame(options)
        return self._build_lazy(options).collect()

    def _collect_schema(self, options: CastOptions) -> Schema:
        if self._plan.is_empty():
            return self._source._collect_schema(options)
        if self._plan.is_schema_preserving():
            # Nothing in the plan reshapes columns — answer comes
            # straight from the source's own canonical Schema.
            return self._source._collect_schema(options)
        # Reshape op present (select / group_by / apply) — derive the
        # post-plan schema from polars, since the AST doesn't model
        # all output column shapes (especially aggregations).
        pl = polars_module()
        lf = self._build_lazy(options)
        pl_schema = lf.collect_schema()
        if isinstance(pl_schema, pl.Schema):
            return Schema.from_polars(pl_schema)
        return Schema.from_polars(lf)

    # ------------------------------------------------------------------
    # Tabular contract — write hooks (forward to source)
    # ------------------------------------------------------------------

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        self._source._write_arrow_batches(batches, options)


class _LazyGroupBy:
    """Builder returned by :meth:`LazyTabular.group_by`.

    Calling :meth:`agg` finalizes the group-by and returns a new
    :class:`LazyTabular` with a :class:`GroupByAgg` op appended. Kept
    as a distinct type rather than inlining onto :class:`LazyTabular`
    so typos like ``lt.group_by('x').where(...)`` fail loudly instead
    of silently dropping the grouping.
    """

    __slots__ = ("_parent", "_by")

    def __init__(
        self, parent: LazyTabular, by: "tuple[_SelectorIn, ...]",
    ) -> None:
        self._parent = parent
        self._by = by

    def __repr__(self) -> str:
        return f"_LazyGroupBy(by={list(self._by)!r})"

    def agg(self, *aggs: Any) -> LazyTabular:
        """Finalize the group-by with zero or more aggregations.

        Aggregation expressions stay backend-native — the AST
        doesn't model aggregations, and rebuilding that here would
        duplicate planner logic polars already does well. Zero aggs
        is the polars "distinct keys" behavior.
        """
        return self._parent._append_op(GroupByAgg(self._by, tuple(aggs)))
