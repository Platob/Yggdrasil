"""Lazy :class:`Tabular` wrapper that defers transformations to read time.

:class:`LazyTabular` wraps an inner :class:`Tabular` with a chain of
pending transformations — :meth:`select`, :meth:`filter` (alias
:meth:`where`), :meth:`group_by` — and only runs them when something
pulls data (``read_arrow_batches``, ``read_arrow_table``, ``collect_schema``,
``read_polars_frame``, …). Each call returns a new :class:`LazyTabular`;
the original is never mutated, so chains like
``lt.where(...).where(...).select(...)`` are safe.

Predicate canonicalization
--------------------------

Filters and column expressions are canonicalized through the
:mod:`yggdrasil.data.expr` AST so the same predicate can target any
backend without each engine getting its own builder. Anything
:meth:`Expression.from_` accepts is fair game — SQL strings, pyarrow
``compute.Expression`` nodes, polars ``Expr`` nodes, pyspark ``Column``,
or already-built :class:`Expression` trees. They're stored as
:class:`Expression` and compiled to the target engine (``to_polars`` for
the LazyFrame plan, ``to_arrow`` if a caller wants the dataset filter
directly) at execution time.

Stacked filters AND-merge into a single :class:`Logical` node via
:meth:`Predicate.merge_with` as soon as they're recorded — both for
plan tightness and so the canonical form survives a print-the-ops
debug session.

Pushdown
--------

Execution routes through a polars :class:`LazyFrame` scan of the inner
Tabular. Polars' query planner pushes projections and predicates into
the underlying pyarrow dataset (the inner's :meth:`_scan_polars_frame`
returns ``pl.scan_pyarrow_dataset(...)`` by default), so filters and
column selections never read the columns / rows they don't need.

Writes are forwarded to the inner Tabular as-is. The lazy ops describe
a *view* of the source; they don't apply on the write path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Iterator, Tuple, Union

import pyarrow as pa

from yggdrasil.data.expr import Expression, Predicate
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.lazy_imports import polars_module

if TYPE_CHECKING:
    import polars as pl


__all__ = ["LazyTabular"]


# A "selector" in this module is anything that fronts a column /
# column-expression: a bare column name, a yggdrasil :class:`Expression`,
# or a backend-native expression (polars ``Expr``, pyarrow
# ``compute.Expression``, …) that :meth:`Expression.from_` can lift.
# We keep strings as strings (they're column names, not SQL fragments —
# polars / pyarrow both accept them directly) and lift everything else
# through :meth:`Expression.from_` so the stored form is canonical.
_SelectorIn = Union[str, Expression, Any]


# Each pending operation is a small immutable tuple. Keeping them as
# data (instead of as already-bound polars closures) lets ``__repr__``,
# equality checks, and the merge-filters pass inspect them without
# evaluating any engine-side expression. Filters are stored as a single
# canonical :class:`Predicate` — :meth:`filter` AND-merges with the
# trailing filter op so two stacked ``where`` calls share one node.
#
#   ("select",   tuple_of_stored_selectors)
#   ("filter",   single_predicate_expression)
#   ("group_by", tuple_of_stored_selectors, tuple_of_aggs)
#   ("apply",    callable_lazyframe_to_lazyframe)   # escape hatch
_Op = Tuple[Any, ...]


def _normalize_filter(value: Any) -> Any:
    """Normalize a filter input to its stored form.

    - Yggdrasil :class:`Expression`: kept as-is (must be a
      :class:`Predicate`).
    - ``str``: parsed as a SQL predicate via :meth:`Expression.from_sql`.
    - Anything else (polars / pyarrow / pyspark expression): kept
      backend-native — round-tripping a polars ``Expr`` through the AST
      can lose type information (literal precision, struct dtype),
      so we don't try to canonicalize what the engine already has.

    The result either *is* a :class:`Predicate` or is opaquely
    "engine-native"; :meth:`LazyTabular._apply_ops` compiles it to the
    LazyFrame at execution time without further translation.
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


def _compile_filter(value: Any) -> Any:
    """Compile a stored filter to a polars-friendly value."""
    if isinstance(value, Predicate):
        return value.to_polars()
    return value


def _compile_selector(value: _SelectorIn) -> Any:
    """Compile a stored selector to a polars-friendly value.

    Strings (column names) and backend-native expressions pass
    through untouched; yggdrasil :class:`Expression` nodes compile
    via :meth:`Expression.to_polars`.
    """
    if isinstance(value, Expression):
        return value.to_polars()
    return value


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

    - :meth:`select` — column projection. Accepts column names or any
      :meth:`Expression.from_`-coercible expression.
    - :meth:`filter` / :meth:`where` — row filtering. Accepts SQL
      predicate strings, yggdrasil predicates, or backend-native
      expressions. Multiple calls AND-merge into one canonical
      :class:`Predicate` before the scan.
    - :meth:`group_by` — returns a small builder whose :meth:`agg`
      finalizes back into a new :class:`LazyTabular`. Aggregation
      expressions stay polars-native (the AST doesn't model
      aggregations).
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

    def select(self, *columns: _SelectorIn) -> "LazyTabular":
        """Project to *columns*. Accepts column names or expressions."""
        if not columns:
            raise ValueError(
                "LazyTabular.select needs at least one column; got none. "
                "Pass column names ('a', 'b'), yggdrasil expressions "
                "(col('a').alias('x')), or polars / pyarrow expressions."
            )
        return self._with(("select", tuple(columns)))

    def filter(self, *predicates: Any) -> "LazyTabular":
        """Row filter. Multiple predicates AND-merge.

        Predicates expressed via the yggdrasil AST (or SQL strings,
        which we parse to AST) are folded into a single canonical
        :class:`Logical` node via :meth:`Predicate.merge_with` —
        adjacent calls fuse their predicates so the stored plan
        stays tight. Backend-native expressions (polars ``Expr``,
        pyarrow ``compute.Expression``) are kept native and stacked
        alongside; polars' planner ANDs the resulting filter calls
        on its own.
        """
        if not predicates:
            raise ValueError(
                "LazyTabular.filter needs at least one predicate; got none. "
                "Pass a SQL string ('x > 0'), a yggdrasil predicate "
                "(col('x') > 0), or a polars / pyarrow expression."
            )

        ops = list(self._ops)
        items: list[Any] = list(ops[-1][1]) if (
            ops and ops[-1][0] == "filter"
        ) else []
        for raw in predicates:
            normalized = _normalize_filter(raw)
            # Fuse adjacent yggdrasil predicates into one canonical
            # Logical(AND, ...) so introspecting ``.ops`` shows the
            # merged tree the AST guarantees.
            if (
                items
                and isinstance(items[-1], Predicate)
                and isinstance(normalized, Predicate)
            ):
                items[-1] = items[-1].merge_with(normalized)
            else:
                items.append(normalized)

        new_op: _Op = ("filter", tuple(items))
        if ops and ops[-1][0] == "filter":
            ops[-1] = new_op
        else:
            ops.append(new_op)
        return LazyTabular(self._inner, ops=tuple(ops))

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
        lf = self._inner._scan_polars_frame(options)
        return self._apply_ops(lf, self._ops)

    @staticmethod
    def _apply_ops(
        lf: "pl.LazyFrame",
        ops: Tuple[_Op, ...],
    ) -> "pl.LazyFrame":
        for op in ops:
            kind = op[0]
            if kind == "select":
                lf = lf.select(*(_compile_selector(c) for c in op[1]))
            elif kind == "filter":
                lf = lf.filter(*(_compile_filter(p) for p in op[1]))
            elif kind == "group_by":
                _, keys, aggs = op
                gb = lf.group_by(*(_compile_selector(k) for k in keys))
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

    def __init__(
        self, parent: LazyTabular, by: Tuple[_SelectorIn, ...],
    ) -> None:
        self._parent = parent
        self._by = by

    def __repr__(self) -> str:
        return f"_LazyGroupBy(by={list(self._by)!r})"

    def agg(self, *aggs: Any) -> LazyTabular:
        """Finalize the group-by with zero or more aggregations.

        Aggregation expressions stay polars-native — the yggdrasil
        AST doesn't model aggregations yet, and rebuilding that here
        would duplicate planner logic polars already does well.
        Zero aggs is the polars "distinct keys" behavior — equivalent
        to ``lf.group_by(*by).agg()``.
        """
        return self._parent._with(("group_by", self._by, tuple(aggs)))
