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
canonicalized through :class:`yggdrasil.execution.expr.Expression`
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

import logging
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Union

import pyarrow as pa

from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.execution.expr import Expression, Predicate
from yggdrasil.execution.plan import (
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


__all__ = ["LazyTabular"]


logger = logging.getLogger(__name__)


# A "selector" is anything that fronts a column / column-expression. The
# canonical type is :class:`yggdrasil.data.data_field.Field` — its
# :attr:`name` is the projected output name and :attr:`alias` (when
# different) is the source-side label, so a single Field carries
# rename + cast-on-select intent. For ergonomic call sites we also
# accept a bare column-name string, a yggdrasil :class:`Expression`,
# or a backend-native expression that ``Expression.from_`` could
# lift. Strings stay strings (column names, not SQL fragments) and
# yggdrasil nodes stay nodes; native exprs pass through.
_SelectorIn = Union[str, Field, Expression, Any]


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
        """Project to *columns*.

        The canonical selector type is
        :class:`yggdrasil.data.data_field.Field` — its
        :attr:`name` is the projected output name and
        :attr:`alias` (when set) is the source-side label, so a
        single Field carries rename + cast-on-select. Bare column
        names, yggdrasil :class:`Expression` nodes, and backend-
        native (polars / pyarrow / pyspark) expressions are also
        accepted.
        """
        if not columns:
            raise ValueError(
                "LazyTabular.select needs at least one column; got none. "
                "Pass Fields (the canonical selector), column names "
                "('a', 'b'), yggdrasil expressions (col('a').alias('x')), "
                "or polars / pyarrow expressions."
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "LazyTabular plan(%d ops) over %s -> %d rows",
                len(self._plan),
                type(self._source).__name__,
                table.num_rows,
            )
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
