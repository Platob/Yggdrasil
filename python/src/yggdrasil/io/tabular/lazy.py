"""Lazy :class:`Tabular` wrapper that defers transformations to read time.

:class:`LazyTabular` wraps an inner :class:`Tabular` with a chain of
pending transformations — :meth:`select`, :meth:`filter` (alias
:meth:`where`), :meth:`group_by` — and only runs them when something
pulls data (``read_arrow_batches``, ``read_arrow_table``, ``collect_schema``,
``read_polars_frame``, …). Each call returns a new :class:`LazyTabular`;
the original is never mutated, so chains like
``lt.where(...).where(...).select(...)`` are safe.

Pushdown
--------

Execution routes through a polars :class:`LazyFrame` scan of the inner
Tabular. Polars' query planner pushes projections and predicates into
the underlying pyarrow dataset (the inner's :meth:`_scan_polars_frame`
returns ``pl.scan_pyarrow_dataset(...)`` by default), so filters and
column selections never read the columns / rows they don't need.
Stacked filters are conjoined into a single predicate before scan,
which is also what polars would produce after optimization — explicit
here so the plan is obvious from the call site.

Writes are forwarded to the inner Tabular as-is. The lazy ops describe
a *view* of the source; they don't apply on the write path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, Tuple

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.lazy_imports import polars_module

if TYPE_CHECKING:
    import polars as pl


__all__ = ["LazyTabular"]


# Each pending operation is a small immutable tuple. Keeping them as
# data (instead of as already-bound polars closures) lets ``__repr__``,
# equality checks, and the conjoin-filters pass inspect them without
# evaluating any polars expression — important because polars exprs
# don't have a stable ``==`` and constructing them eagerly would defeat
# the lazy contract.
#
#   ("select",   tuple_of_columns_or_exprs)
#   ("filter",   tuple_of_predicate_exprs)        # AND-combined
#   ("group_by", tuple_of_keys, tuple_of_aggs)
#   ("apply",    callable_lazyframe_to_lazyframe) # escape hatch
_Op = Tuple[Any, ...]


class LazyTabular(Tabular[CastOptions]):
    """Tabular view over an inner Tabular plus a chain of lazy ops.

    The inner Tabular is the source of truth — schema, batches, and
    writes all originate from it. The op chain is replayed on every
    read against a polars LazyFrame scan, which means engine-side
    pushdown happens automatically as long as the inner exposes a
    pyarrow dataset (the default :meth:`Tabular._scan_polars_frame`
    does).

    Operations
    ~~~~~~~~~~

    - :meth:`select` — column projection (str names or ``pl.Expr``).
    - :meth:`filter` / :meth:`where` — row filtering. Multiple calls
      AND-combine into one predicate before the scan.
    - :meth:`group_by` — returns a small builder whose :meth:`agg`
      finalizes back into a new :class:`LazyTabular`.
    - :meth:`apply` — escape hatch for one-off polars transforms that
      don't fit the dedicated methods.
    """

    def __init__(
        self,
        inner: Tabular,
        *,
        ops: Iterable[_Op] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._inner: Tabular = inner
        self._ops: Tuple[_Op, ...] = tuple(ops)

    def __repr__(self) -> str:
        return (
            f"LazyTabular(inner={self._inner!r}, "
            f"ops={[op[0] for op in self._ops]!r})"
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def inner(self) -> Tabular:
        return self._inner

    @property
    def ops(self) -> Tuple[_Op, ...]:
        return self._ops

    @classmethod
    def options_class(cls) -> "type[CastOptions]":
        return CastOptions

    # ------------------------------------------------------------------
    # Builder methods — return a new LazyTabular each time
    # ------------------------------------------------------------------

    def _with(self, op: _Op) -> "LazyTabular":
        return LazyTabular(self._inner, ops=(*self._ops, op))

    def select(self, *columns: Any) -> "LazyTabular":
        """Project to *columns*. Accepts column names or ``pl.Expr``."""
        if not columns:
            raise ValueError(
                "LazyTabular.select needs at least one column; got none. "
                "Pass column names ('a', 'b') or polars expressions "
                "(pl.col('a').alias('x'))."
            )
        return self._with(("select", tuple(columns)))

    def filter(self, *predicates: Any) -> "LazyTabular":
        """Row filter. Multiple predicates AND-combine."""
        if not predicates:
            raise ValueError(
                "LazyTabular.filter needs at least one predicate; got none. "
                "Pass polars expressions, e.g. pl.col('x') > 0."
            )
        return self._with(("filter", tuple(predicates)))

    where = filter

    def group_by(self, *by: Any) -> "_LazyGroupBy":
        """Start a group-by. Finalize with :meth:`_LazyGroupBy.agg`."""
        if not by:
            raise ValueError(
                "LazyTabular.group_by needs at least one key; got none. "
                "Pass column names or polars expressions."
            )
        return _LazyGroupBy(self, tuple(by))

    groupby = group_by

    def apply(self, fn: Any) -> "LazyTabular":
        """Escape hatch: ``fn(LazyFrame) -> LazyFrame``.

        Use only when the dedicated methods don't fit — e.g. ``join``,
        ``with_columns``, ``sort``. The callable runs once per execution
        with the in-flight :class:`pl.LazyFrame`.
        """
        if not callable(fn):
            raise TypeError(
                f"LazyTabular.apply expected a callable LazyFrame -> "
                f"LazyFrame; got {type(fn).__name__}: {fn!r}."
            )
        return self._with(("apply", fn))

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------

    def _build_lazy(self, options: CastOptions) -> "pl.LazyFrame":
        """Compose the polars LazyFrame: scan inner, replay ops in order.

        Adjacent filters are conjoined into a single ``filter`` call so
        the scan sees one combined predicate instead of N. Polars'
        optimizer would do the same, but doing it here keeps the
        explained plan tight and makes pushdown obvious to anyone
        printing the LazyFrame.
        """
        lf = self._inner._scan_polars_frame(options)
        return self._apply_ops(lf, self._ops)

    @staticmethod
    def _apply_ops(
        lf: "pl.LazyFrame",
        ops: Tuple[_Op, ...],
    ) -> "pl.LazyFrame":
        # Pre-pass: collapse runs of consecutive filters. Order is
        # preserved across non-filter ops because filter-vs-select
        # placement matters once you start reordering exprs.
        merged: list[_Op] = []
        for op in ops:
            if op[0] == "filter" and merged and merged[-1][0] == "filter":
                merged[-1] = ("filter", (*merged[-1][1], *op[1]))
            else:
                merged.append(op)

        for op in merged:
            kind = op[0]
            if kind == "select":
                lf = lf.select(*op[1])
            elif kind == "filter":
                lf = lf.filter(*op[1])
            elif kind == "group_by":
                _, by, aggs = op
                gb = lf.group_by(*by)
                lf = gb.agg(*aggs) if aggs else gb.agg()
            elif kind == "apply":
                lf = op[1](lf)
            else:
                raise ValueError(f"Unknown LazyTabular op kind: {kind!r}")
        return lf

    # ------------------------------------------------------------------
    # Tabular contract — read hooks
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self, options: CastOptions,
    ) -> Iterator[pa.RecordBatch]:
        if not self._ops:
            yield from self._inner._read_arrow_batches(options)
            return

        lf = self._build_lazy(options)
        table: pa.Table = lf.collect().to_arrow()
        row_size = getattr(options, "row_size", None) or None
        for batch in table.to_batches(max_chunksize=row_size):
            yield options.cast_arrow_tabular(batch)

    def _scan_polars_frame(self, options: CastOptions) -> "pl.LazyFrame":
        # Stacking another LazyTabular on top of this one composes
        # cleanly because scan_polars_frame returns an already-planned
        # LazyFrame — polars folds the two plans together.
        return self._build_lazy(options)

    def _read_polars_frame(self, options: CastOptions) -> "pl.DataFrame":
        if not self._ops:
            return self._inner._read_polars_frame(options)
        return self._build_lazy(options).collect()

    def _collect_schema(self, options: CastOptions) -> Schema:
        if not self._ops:
            return self._inner._collect_schema(options)
        # ``LazyFrame.collect_schema`` returns a polars Schema without
        # materializing rows — that's the whole point of routing through
        # polars for the lazy plan.
        pl = polars_module()
        lf = self._build_lazy(options)
        pl_schema = lf.collect_schema()
        if isinstance(pl_schema, pl.Schema):
            return Schema.from_polars(pl_schema)
        return Schema.from_polars(lf)

    # ------------------------------------------------------------------
    # Tabular contract — write hooks (forward to inner)
    # ------------------------------------------------------------------

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        self._inner._write_arrow_batches(batches, options)


class _LazyGroupBy:
    """Builder returned by :meth:`LazyTabular.group_by`.

    Calling :meth:`agg` finalizes the group-by and returns a new
    :class:`LazyTabular` with the operation appended. Kept as a
    distinct type rather than inlining onto :class:`LazyTabular` so
    typos like ``lt.group_by('x').where(...)`` fail loudly instead
    of silently dropping the grouping.
    """

    __slots__ = ("_parent", "_by")

    def __init__(self, parent: LazyTabular, by: Tuple[Any, ...]) -> None:
        self._parent = parent
        self._by = by

    def __repr__(self) -> str:
        return f"_LazyGroupBy(by={list(self._by)!r})"

    def agg(self, *aggs: Any) -> LazyTabular:
        """Finalize the group-by with zero or more aggregations.

        Zero aggs is the polars "distinct keys" behavior — equivalent
        to ``lf.group_by(*by).agg()``.
        """
        return self._parent._with(("group_by", self._by, tuple(aggs)))
