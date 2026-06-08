"""Autonomous, mutable execution plans for lazy Tabular transformations.

The plan tree is two-tier:

- :class:`ExecutionPlan` — abstract base, also a :class:`Tabular`.
  Subclasses carry every object they need to run standalone (target
  / source Tabulars, predicates, transformations) so
  :meth:`execute` takes only ``wait`` / ``raise_error``.
- :class:`SelectPlan` — builder for SELECT-like reads. Returns the
  read result as a :class:`Tabular`.
- :class:`InsertPlan` — builder for ``INSERT INTO target (...) ...``.
  Returns an :class:`OperationResult` with row counts + post-write target.
- :class:`MergePlan` — builder for ``MERGE INTO target USING source
  ON ... WHEN MATCHED ... WHEN NOT MATCHED ...``. Returns
  :class:`OperationResult`.

Because each plan IS a :class:`Tabular`, callers can pass a plan
anywhere a Tabular is expected; reading the plan triggers
:meth:`execute` and exposes the result (for SELECT) or the operation
metadata as a one-row table (for INSERT / MERGE).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import pyarrow as pa

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.data.options import CastOptions
from yggdrasil.enums import JoinType, Mode
from yggdrasil.io.tabular.base import (
    Tabular,
    _coerce_column_keys,
    _coerce_predicate,
    _coerce_sampling_seconds,
    _flatten_column_args,
)

from .operation_result import OperationResult
from .ops import CTE, GroupByOp, JoinOp, OrderByOp, ResampleOp, UnionOp

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema
    from yggdrasil.enums import Dialect
    from yggdrasil.saga.expr import Predicate, PredicateLike
    from .nodes import InsertNode, MergeNode
    from .execution_result import ExecutionResult

O = TypeVar("O", bound=CastOptions)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ExecutionPlan(Tabular[O], Generic[O]):
    """Abstract base — a plan IS a Tabular.

    Subclasses carry every object they need to execute standalone
    (source/target Tabulars, predicates, transformations). The
    :meth:`execute` entry point takes only ``wait`` and
    ``raise_error`` — same calling convention every async-aware
    backend in yggdrasil uses.

    Reading the plan (``plan.read_arrow_table()`` /
    ``plan.read_arrow_batches()`` / …) executes the plan once and
    streams the result. Writing into the plan is a type error —
    plans are read-only as Tabulars.
    """

    @classmethod
    def options_class(cls) -> "type[O]":
        return CastOptions  # type: ignore[return-value]

    # -- SQL round-trip --------------------------------------------------

    @classmethod
    def from_sql(
        cls,
        sql: str,
        dialect: "Dialect | str | None" = None,
        default: Any = ...,
    ) -> "ExecutionPlan | Any":
        """Parse *sql* into a concrete plan (SelectPlan / InsertPlan / MergePlan).

        Returns *default* on parse failure when *default* is not
        ``...``; raises otherwise. The returned plan is *not* yet
        bound to concrete Tabulars — bind via the constructor
        keywords (``source=``, ``target=``) before calling
        :meth:`execute`.
        """
        from .sql_parser import parse_sql
        from .nodes import InsertNode, MergeNode, SelectNode

        node = parse_sql(sql, dialect=dialect, default=default)
        if isinstance(node, SelectNode):
            plan = SelectPlan()
            plan._apply_select_node(node)
            return plan
        if isinstance(node, InsertNode):
            plan = InsertPlan()
            plan._apply_insert_node(node)
            return plan
        if isinstance(node, MergeNode):
            plan = MergePlan()
            plan._apply_merge_node(node)
            return plan
        if default is not ...:
            return default
        return SelectPlan()

    def to_sql(self, dialect: "Dialect | str | None" = None) -> str:
        from .sql_emitter import emit_sql
        return emit_sql(self.to_plan_node(), dialect=dialect)

    def to_plan_node(self):
        raise NotImplementedError(
            f"{type(self).__name__} cannot be rendered as a PlanNode."
        )

    # -- Execution -------------------------------------------------------

    @abstractmethod
    def execute(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "Tabular | OperationResult":
        """Run the plan; returns a :class:`Tabular` (SELECT) or
        :class:`OperationResult` (INSERT / MERGE / …)."""

    def submit(self) -> "ExecutionResult[O]":
        """Wrap this plan in a lazy, awaitable :class:`ExecutionResult`.

        The plan is bound (carries its own source / target), so nothing
        runs until the returned handle is read / awaited / started."""
        from .execution_result import ExecutionResult
        return ExecutionResult(self)

    @abstractmethod
    def copy(self) -> "ExecutionPlan[O]": ...

    # -- Tabular contract ------------------------------------------------

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        result = self.execute()
        if isinstance(result, OperationResult):
            result = result.to_arrow_tabular()
        yield from result.read_arrow_batches(options)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        raise TypeError(f"{type(self).__name__} is read-only as a Tabular.")

    def _delete(self, predicate: Any = None, *, wait: Any = True,
                missing_ok: bool = False, delete_staging: bool = True,
                **kwargs: Any) -> int:
        raise NotImplementedError(f"{type(self).__name__} is read-only as a Tabular.")

    def _collect_schema(self, options: O) -> "Schema":
        result = self.execute()
        if isinstance(result, OperationResult):
            result = result.to_arrow_tabular()
        return result.collect_schema(options)


# ---------------------------------------------------------------------------
# SelectPlan
# ---------------------------------------------------------------------------


class SelectPlan(ExecutionPlan[O]):
    """Mutable SELECT-like plan: projection, filter, join, union,
    dedup, resample, cast, limit.

    Bind a source at construction (``SelectPlan(source=tabular)``) or
    later via :meth:`bind` to make the plan autonomous —
    :meth:`execute` then runs without any extra arguments.
    """

    __slots__ = (
        "_source", "_select", "_drop", "_predicate", "_joins", "_unions",
        "_unique_by", "_resample", "_cast_options", "_limit",
        "_group_by", "_having", "_order_by", "_ctes", "_offset",
    )

    def __init__(
        self,
        source: "Tabular | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._source: "Tabular | None" = source
        self._select: list[str] | None = None
        self._drop: list[str] | None = None
        self._predicate: "Predicate | None" = None
        self._joins: list[JoinOp] = []
        self._unions: list[UnionOp] = []
        self._unique_by: list[str] | None = None
        self._resample: ResampleOp | None = None
        self._cast_options: CastOptions | None = None
        self._limit: int | None = None
        self._group_by: GroupByOp | None = None
        self._having: "Predicate | None" = None
        self._order_by: OrderByOp | None = None
        self._ctes: list[CTE] | None = None
        self._offset: int | None = None

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        # Plans are mutable builders — every construction is a fresh
        # instance regardless of args.  Stamping the key with ``id``
        # keeps every constructor call distinct even when the same
        # ``(source, ...)`` tuple is passed.
        return (cls, id(object()))

    # -- Introspection ---------------------------------------------------

    @property
    def is_identity(self) -> bool:
        return (self._select is None and self._drop is None
                and self._predicate is None and not self._joins
                and not self._unions and self._unique_by is None
                and self._resample is None and self._cast_options is None
                and self._limit is None and self._group_by is None
                and self._having is None and self._order_by is None
                and self._ctes is None and self._offset is None)

    @property
    def is_schema_preserving(self) -> bool:
        """True iff the plan can't reshape the source's column set/types.

        Filter, limit, offset and order-by only drop or reorder rows; the
        output schema is identical to the source. Anything that projects,
        aggregates, joins, unions, resamples or casts can change it. Callers
        (e.g. :meth:`LazyTabular._collect_schema`) use this to skip a full
        plan execution when they only need column names/types.
        """
        return (self._select is None and self._drop is None
                and not self._joins and not self._unions
                and self._unique_by is None and self._resample is None
                and self._cast_options is None and self._group_by is None
                and self._ctes is None)

    @property
    def source(self) -> "Tabular | None": return self._source
    @property
    def columns(self) -> list[str] | None: return self._select
    @property
    def predicate(self) -> "Predicate | None": return self._predicate
    @property
    def joins(self) -> list[JoinOp]: return self._joins
    @property
    def unions(self) -> list[UnionOp]: return self._unions
    @property
    def limit_rows(self) -> int | None: return self._limit
    @property
    def group_by_op(self) -> GroupByOp | None: return self._group_by
    @property
    def having_pred(self) -> "Predicate | None": return self._having
    @property
    def order_by_op(self) -> OrderByOp | None: return self._order_by
    @property
    def ctes(self) -> list[CTE] | None: return self._ctes
    @property
    def offset_rows(self) -> int | None: return self._offset

    # -- Binding ---------------------------------------------------------

    def bind(self, source: "Tabular") -> "SelectPlan[O]":
        """Attach *source* to this plan so :meth:`execute` runs standalone."""
        self._source = source
        return self

    # -- Builders (mutate + return self) ---------------------------------

    def select(self, *columns: "str | Any") -> "SelectPlan[O]":
        cols = _flatten_column_args(columns)
        if not cols:
            raise ValueError("select needs at least one column.")
        self._select = cols
        self._drop = None
        return self

    def drop(self, *columns: "str | Any") -> "SelectPlan[O]":
        cols = _flatten_column_args(columns)
        if cols:
            self._drop = cols
        return self

    def filter(self, predicate: "PredicateLike") -> "SelectPlan[O]":
        pred = _coerce_predicate(predicate)
        self._predicate = (self._predicate & pred) if self._predicate else pred
        return self

    def clear_filter(self) -> "SelectPlan[O]":
        self._predicate = None
        return self

    def join(self, right: "Tabular", on: "str | list[str] | Any",
             how: "str | JoinType" = "inner", *, suffix: str = "_right") -> "SelectPlan[O]":
        keys = _coerce_column_keys(on)
        if not keys:
            raise ValueError("join needs at least one key column.")
        self._joins.append(JoinOp(right=right, on=keys,
                                   how=JoinType.from_(how), right_suffix=suffix))
        return self

    def union(self, other: "Tabular", *, mode: "str | Mode | None" = None) -> "SelectPlan[O]":
        self._unions.append(UnionOp(other=other, mode=Mode.from_(mode, default=Mode.IGNORE)))
        return self

    def unique(self, by: "str | Any | list[Any]") -> "SelectPlan[O]":
        self._unique_by = _coerce_column_keys(by) or None
        return self

    def resample(self, on: "str | Any", sampling: "int | float | Any", *,
                 partition_by: "str | Any | list[Any] | None" = None,
                 fill_strategy: "str | None" = "ffill") -> "SelectPlan[O]":
        from yggdrasil.io.tabular.base import _coerce_column_name
        self._resample = ResampleOp(
            time_column=_coerce_column_name(on),
            sampling_seconds=_coerce_sampling_seconds(sampling),
            partition_by=_coerce_column_keys(partition_by) if partition_by else [],
            fill_strategy=fill_strategy)
        return self

    def cast(self, options: "CastOptions | None" = None, **kwargs: Any) -> "SelectPlan[O]":
        self._cast_options = CastOptions.check(options, **kwargs)
        return self

    def limit(self, n: int | None) -> "SelectPlan[O]":
        self._limit = n
        return self

    def offset(self, n: int | None) -> "SelectPlan[O]":
        self._offset = n
        return self

    def group_by(self, *keys: "str | Any",
                 aggregations: "dict[str, str] | None" = None) -> "SelectPlan[O]":
        kl = _flatten_column_args(keys)
        if not kl:
            raise ValueError("group_by needs at least one key column.")
        self._group_by = GroupByOp(keys=kl, aggregations=aggregations or {})
        return self

    def having(self, predicate: "PredicateLike") -> "SelectPlan[O]":
        pred = _coerce_predicate(predicate)
        self._having = (self._having & pred) if self._having else pred
        return self

    def order_by(self, *keys: "str | tuple[str, bool] | Any") -> "SelectPlan[O]":
        parsed: list[tuple[str, bool]] = []
        for k in keys:
            if isinstance(k, tuple):
                parsed.append((str(k[0]), bool(k[1])))
            elif isinstance(k, str):
                if k.startswith("-"):
                    parsed.append((k[1:], False))
                else:
                    parsed.append((k.lstrip("+"), True))
            else:
                parsed.append((str(k), True))
        if not parsed:
            raise ValueError("order_by needs at least one key.")
        self._order_by = OrderByOp(keys=parsed)
        return self

    def with_cte(self, name: str, plan: "SelectPlan | Any") -> "SelectPlan[O]":
        if self._ctes is None:
            self._ctes = []
        self._ctes.append(CTE(name=name, plan=plan))
        return self

    # -- Copy / clear ----------------------------------------------------

    def copy(self) -> "SelectPlan[O]":
        c: SelectPlan[O] = SelectPlan(source=self._source)
        c._select = list(self._select) if self._select else None
        c._drop = list(self._drop) if self._drop else None
        c._predicate = self._predicate
        c._joins = list(self._joins)
        c._unions = list(self._unions)
        c._unique_by = list(self._unique_by) if self._unique_by else None
        c._resample = self._resample
        c._cast_options = self._cast_options
        c._limit = self._limit
        c._group_by = self._group_by
        c._having = self._having
        c._order_by = self._order_by
        c._ctes = list(self._ctes) if self._ctes else None
        c._offset = self._offset
        return c

    def clear(self) -> "SelectPlan[O]":
        self._select = self._drop = self._predicate = None
        self._joins.clear(); self._unions.clear()
        self._unique_by = self._resample = self._cast_options = self._limit = None
        self._group_by = self._having = self._order_by = self._ctes = self._offset = None
        return self

    # -- Execution -------------------------------------------------------

    def execute(
        self,
        source: "Tabular | WaitingConfigArg | None" = None,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "Tabular":
        """Run the SELECT plan against the bound or supplied source.

        Accepts a positional :class:`Tabular` as the source for
        ergonomics (and backward compat with the previous
        ``execute(source)`` shape). Otherwise uses the source bound
        at construction / via :meth:`bind`.

        ``wait`` and ``raise_error`` are routed through to async
        sources (warehouse statements, remote runs); fully-local
        Arrow sources ignore them — there's no work to wait for.
        """
        if isinstance(source, Tabular):
            src = source
        elif source is None:
            src = self._source
        else:
            # First positional was a WaitingConfigArg — promote to ``wait``.
            wait = source
            src = self._source

        if src is None:
            raise ValueError(
                "SelectPlan has no source; pass source positionally, "
                "construct with SelectPlan(source=...), or call .bind(source) first."
            )

        return self._execute_against(src, wait=wait, raise_error=raise_error)

    def _execute_against(
        self,
        source: "Tabular[O]",
        *,
        wait: WaitingConfigArg,
        raise_error: bool,
    ) -> "Tabular":
        if self.is_identity:
            return source

        from yggdrasil.arrow.tabular import ArrowTabular
        import pyarrow.compute as pc

        result: "Tabular" = source

        # Predicate pushdown into CastOptions for format-level readers
        pushdown: dict[str, Any] = {}
        if self._predicate is not None and not self._joins:
            pushdown["predicate"] = self._predicate
        if self._cast_options is not None:
            opts = self._cast_options.copy(**pushdown) if pushdown else self._cast_options
            result = ArrowTabular(result.read_arrow_table(opts))
        elif pushdown:
            result = ArrowTabular(result.read_arrow_table(
                source.check_options(None, overrides=pushdown)))

        # Filter (before joins when possible)
        if self._predicate is not None and not self._joins:
            result = result.filter(self._predicate)

        # Joins
        if self._joins:
            left_table = result.read_arrow_table()
            for jop in self._joins:
                right = (jop.right.execute(wait=wait, raise_error=raise_error)
                         if isinstance(jop.right, ExecutionPlan)
                         else (jop.right._execute_plan() if hasattr(jop.right, "_execute_plan") else jop.right))
                left_table = left_table.join(
                    right.read_arrow_table(),
                    keys=jop.on, join_type=jop.how.arrow, right_suffix=jop.right_suffix,
                )
            result = ArrowTabular(left_table)

        # Unions
        for uop in self._unions:
            other = (uop.other.execute(wait=wait, raise_error=raise_error)
                     if isinstance(uop.other, ExecutionPlan)
                     else (uop.other._execute_plan() if hasattr(uop.other, "_execute_plan") else uop.other))
            result = result.union(other, mode=uop.mode)

        # Filter after joins
        if self._predicate is not None and self._joins:
            result = result.filter(self._predicate)

        # GROUP BY with aggregations
        if self._group_by is not None:
            table = result.read_arrow_table()
            gb = self._group_by
            if gb.aggregations and table.num_columns > 0:
                available_cols = set(table.column_names)
                agg_specs: list[tuple[str, str]] = []
                rename_map: dict[str, str] = {}
                for out_name, agg_expr in gb.aggregations.items():
                    expr_str = agg_expr.strip()
                    if "(" in expr_str:
                        fn = expr_str[:expr_str.index("(")].strip().upper()
                        col_part = expr_str[expr_str.index("(") + 1:expr_str.rindex(")")].strip()
                    else:
                        fn, col_part = expr_str.upper(), gb.keys[0]
                    _agg_map = {"COUNT": "count", "SUM": "sum", "AVG": "mean",
                                "MIN": "min", "MAX": "max", "MEAN": "mean"}
                    pa_func = _agg_map.get(fn, fn.lower())
                    if col_part == "*":
                        col_part = gb.keys[0]
                    elif col_part not in available_cols:
                        for ac in available_cols:
                            if ac.lower() == col_part.lower():
                                col_part = ac
                                break
                    agg_specs.append((col_part, pa_func))
                    rename_map[f"{col_part}_{pa_func}"] = out_name
                result_table = table.group_by(gb.keys).aggregate(agg_specs)
                result_table = result_table.rename_columns(
                    [rename_map.get(c, c) for c in result_table.column_names])
                result = ArrowTabular(result_table)
            else:
                result = result.unique(gb.keys)

        # HAVING
        if self._having is not None:
            result = result.filter(self._having)

        # Projection
        if self._select is not None:
            result = result.select(*self._select)
        elif self._drop is not None:
            result = result.drop(*self._drop)

        # Unique / resample
        if self._unique_by is not None:
            result = result.unique(self._unique_by)
        if self._resample is not None:
            r = self._resample
            result = result.resample(r.time_column, r.sampling_seconds,
                partition_by=r.partition_by or None, fill_strategy=r.fill_strategy)

        # Cast
        if self._cast_options is not None:
            result = result.cast(self._cast_options)

        # ORDER BY
        if self._order_by is not None:
            table = result.read_arrow_table()
            sort_keys = [(col, "ascending" if asc else "descending")
                         for col, asc in self._order_by.keys]
            result = ArrowTabular(table.take(pc.sort_indices(table, sort_keys=sort_keys)))

        # Limit / offset
        if self._limit is not None or self._offset is not None:
            table = result.read_arrow_table()
            off = self._offset or 0
            lim = self._limit
            if lim is not None:
                table = table.slice(off, lim)
            elif off > 0:
                table = table.slice(off)
            result = ArrowTabular(table)

        return result

    # -- Plan node conversion --------------------------------------------

    def to_plan_node(self):
        from .nodes import SelectNode
        from yggdrasil.saga.expr.nodes import Column, SortOrder, Star
        projs = [Column(name=c) for c in self._select] if self._select else [Star()]
        gb = [Column(name=k) for k in self._group_by.keys] if self._group_by else None
        ob = None
        if self._order_by:
            ob = [SortOrder(expr=Column(name=k), ascending=asc)
                  for k, asc in self._order_by.keys]
        ctes = None
        if self._ctes:
            ctes = [CTE(name=c.name, plan=c.plan.to_plan_node()
                        if hasattr(c.plan, "to_plan_node") else c.plan)
                    for c in self._ctes]
        return SelectNode(
            projections=projs, where=self._predicate,
            group_by=gb, having=self._having, order_by=ob,
            limit=self._limit, offset=self._offset,
            distinct=bool(self._unique_by), ctes=ctes)

    def _apply_select_node(self, node) -> None:
        from yggdrasil.saga.expr.nodes import Alias, Column, Star
        from yggdrasil.saga.expr import Predicate
        if node.projections:
            cols = []
            for p in node.projections:
                if isinstance(p, Star): cols = None; break
                elif isinstance(p, Column): cols.append(p.alias or p.name)
                elif isinstance(p, Alias) and isinstance(p.expr, Column): cols.append(p.name)
                else: cols = None; break
            if cols:
                self._select = cols
        if isinstance(node.where, Predicate):
            self._predicate = node.where
        if node.limit is not None:
            self._limit = node.limit
        if node.offset is not None:
            self._offset = node.offset
        if node.distinct and self._select:
            self._unique_by = list(self._select)
        if node.group_by:
            gk = [g.name for g in node.group_by if isinstance(g, Column)]
            if gk:
                self._group_by = GroupByOp(keys=gk, aggregations={})
        if isinstance(node.having, Predicate):
            self._having = node.having
        if node.order_by:
            from yggdrasil.saga.expr.nodes import SortOrder
            ob = []
            for o in node.order_by:
                if isinstance(o, SortOrder) and isinstance(o.expr, Column):
                    ob.append((o.expr.name, o.ascending))
            if ob:
                self._order_by = OrderByOp(keys=ob)

    def __repr__(self) -> str:
        parts = []
        if self._source is not None: parts.append(f"source={type(self._source).__name__}")
        if self._ctes: parts.append(f"ctes={len(self._ctes)}")
        if self._predicate: parts.append(f"filter={self._predicate!r}")
        if self._joins: parts.append(f"joins={len(self._joins)}")
        if self._unions: parts.append(f"unions={len(self._unions)}")
        if self._select: parts.append(f"select={self._select!r}")
        if self._drop: parts.append(f"drop={self._drop!r}")
        if self._group_by: parts.append(f"group_by={self._group_by.keys!r}")
        if self._having: parts.append(f"having={self._having!r}")
        if self._unique_by: parts.append(f"unique_by={self._unique_by!r}")
        if self._resample: parts.append(f"resample={self._resample.time_column!r}")
        if self._cast_options: parts.append("cast=True")
        if self._order_by: parts.append(f"order_by={self._order_by.keys!r}")
        if self._limit is not None: parts.append(f"limit={self._limit}")
        if self._offset is not None: parts.append(f"offset={self._offset}")
        return f"SelectPlan({', '.join(parts) or 'identity'})"


# ---------------------------------------------------------------------------
# InsertPlan
# ---------------------------------------------------------------------------


class InsertPlan(ExecutionPlan[O]):
    """Autonomous ``INSERT INTO target ...`` plan.

    Carries the concrete target Tabular plus the source — either
    another :class:`Tabular`, an :class:`ExecutionPlan` (a SELECT plan
    feeding the insert), or an inline ``VALUES`` list.
    """

    __slots__ = ("_target", "_source", "_columns", "_values", "_mode")

    def __init__(
        self,
        *,
        target: "Tabular | None" = None,
        source: "Tabular | ExecutionPlan | None" = None,
        columns: "list[str] | None" = None,
        values: "list[list[Any]] | None" = None,
        mode: "Mode | str | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._target: "Tabular | None" = target
        self._source: "Tabular | ExecutionPlan | None" = source
        self._columns: "list[str] | None" = list(columns) if columns else None
        self._values: "list[list[Any]] | None" = (
            [list(r) for r in values] if values else None
        )
        self._mode: "Mode | None" = Mode.from_(mode, default=None) if mode is not None else None

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        return (cls, id(object()))

    # -- Introspection ---------------------------------------------------

    @property
    def target(self) -> "Tabular | None": return self._target
    @property
    def source(self) -> "Tabular | ExecutionPlan | None": return self._source
    @property
    def columns(self) -> "list[str] | None": return self._columns
    @property
    def values(self) -> "list[list[Any]] | None": return self._values
    @property
    def mode(self) -> "Mode | None": return self._mode

    # -- Builders --------------------------------------------------------

    def into(self, target: "Tabular") -> "InsertPlan[O]":
        self._target = target
        return self

    def select_from(self, source: "Tabular | ExecutionPlan") -> "InsertPlan[O]":
        self._source = source
        self._values = None
        return self

    def add_values(self, *rows: list[Any]) -> "InsertPlan[O]":
        if self._values is None:
            self._values = []
        for r in rows:
            self._values.append(list(r))
        self._source = None
        return self

    def with_columns(self, *columns: str) -> "InsertPlan[O]":
        self._columns = list(columns)
        return self

    def with_mode(self, mode: "Mode | str | None") -> "InsertPlan[O]":
        self._mode = Mode.from_(mode, default=None) if mode is not None else None
        return self

    # -- Copy ------------------------------------------------------------

    def copy(self) -> "InsertPlan[O]":
        return InsertPlan(
            target=self._target,
            source=(self._source.copy() if isinstance(self._source, ExecutionPlan)
                    else self._source),
            columns=list(self._columns) if self._columns else None,
            values=[list(r) for r in self._values] if self._values else None,
            mode=self._mode,
        )

    # -- Execution -------------------------------------------------------

    def execute(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> OperationResult:
        if self._target is None:
            raise ValueError("InsertPlan has no target; call .into(target) first.")

        target = self._target

        if self._values is not None:
            cols = self._columns or [f"col{i}" for i in range(len(self._values[0]))]
            pydict = _values_to_pydict(self._values, cols)
            data = pa.table(pydict)
            row_count = data.num_rows
            target.write_table(data, mode=self._mode) if self._mode is not None \
                else target.write_table(data)
            return OperationResult(
                operation="INSERT",
                rows_inserted=row_count,
                target=target,
            )

        if self._source is None:
            raise ValueError("InsertPlan has neither source nor values.")

        # Resolve source: ExecutionPlan → execute (handles wait/raise_error);
        # Tabular → use as-is.
        if isinstance(self._source, ExecutionPlan):
            src_result = self._source.execute(wait=wait, raise_error=raise_error)
            if isinstance(src_result, OperationResult):
                src_tab = src_result.to_arrow_tabular()
            else:
                src_tab = src_result
        else:
            src_tab = self._source

        # Project to the requested columns if specified.
        if self._columns:
            src_tab = src_tab.select(*self._columns)

        row_count = src_tab.read_arrow_table().num_rows
        if self._mode is not None:
            target.write_table(src_tab, mode=self._mode)
        else:
            target.write_table(src_tab)

        return OperationResult(
            operation="INSERT",
            rows_inserted=row_count,
            target=target,
        )

    # -- Plan node conversion --------------------------------------------

    def to_plan_node(self) -> "InsertNode":
        from .nodes import InsertNode
        from .ops import TableRef
        target_ref = None
        if isinstance(self._target, Tabular):
            target_ref = TableRef(name=getattr(self._target, "name", "?"))
        elif self._target is not None:
            target_ref = self._target
        src_node = None
        if isinstance(self._source, ExecutionPlan):
            src_node = self._source.to_plan_node()
        elif self._source is not None:
            from .nodes import ScanNode
            src_node = ScanNode(tabular=self._source)
        return InsertNode(
            target=target_ref,
            columns=list(self._columns) if self._columns else None,
            source=src_node,
            values=([list(r) for r in self._values] if self._values else None),
        )

    def _apply_insert_node(self, node: "InsertNode") -> None:
        from .nodes import SelectNode
        if node.columns:
            self._columns = list(node.columns)
        if node.values is not None:
            self._values = [list(r) for r in node.values]
        if isinstance(node.source, SelectNode):
            sp = SelectPlan()
            sp._apply_select_node(node.source)
            self._source = sp

    def __repr__(self) -> str:
        parts = []
        if self._target is not None:
            parts.append(f"target={type(self._target).__name__}")
        if self._columns:
            parts.append(f"columns={self._columns!r}")
        if self._values is not None:
            parts.append(f"values={len(self._values)} rows")
        if self._source is not None:
            parts.append(f"source={type(self._source).__name__}")
        if self._mode is not None:
            parts.append(f"mode={self._mode.name}")
        return f"InsertPlan({', '.join(parts) or 'unbound'})"


# ---------------------------------------------------------------------------
# MergePlan
# ---------------------------------------------------------------------------


class MergePlan(ExecutionPlan[O]):
    """Autonomous ``MERGE INTO target USING source ON ...`` plan.

    Carries the target Tabular, the source (Tabular or
    :class:`ExecutionPlan`), the ON predicate (or simple key list),
    and the WHEN MATCHED / WHEN NOT MATCHED action dicts.

    The Arrow execution path handles the common in-memory patterns:

    - ``ON t.<key> = s.<key>`` (equality on one or more columns).
    - ``WHEN MATCHED THEN UPDATE SET <col> = <s.col | literal>``
    - ``WHEN MATCHED THEN DELETE``
    - ``WHEN NOT MATCHED THEN INSERT (cols) VALUES (s.cols | literals)``
    """

    __slots__ = ("_target", "_source", "_on", "_on_keys",
                 "_when_matched", "_when_not_matched")

    def __init__(
        self,
        *,
        target: "Tabular | None" = None,
        source: "Tabular | ExecutionPlan | None" = None,
        on: "Predicate | None" = None,
        on_keys: "list[str] | None" = None,
        when_matched: "list[dict] | None" = None,
        when_not_matched: "list[dict] | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._target: "Tabular | None" = target
        self._source: "Tabular | ExecutionPlan | None" = source
        self._on: "Predicate | None" = on
        self._on_keys: "list[str] | None" = list(on_keys) if on_keys else None
        self._when_matched: list[dict] = list(when_matched) if when_matched else []
        self._when_not_matched: list[dict] = (
            list(when_not_matched) if when_not_matched else []
        )

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        return (cls, id(object()))

    # -- Introspection ---------------------------------------------------

    @property
    def target(self) -> "Tabular | None": return self._target
    @property
    def source(self) -> "Tabular | ExecutionPlan | None": return self._source
    @property
    def on_predicate(self) -> "Predicate | None": return self._on
    @property
    def on_keys(self) -> "list[str] | None": return self._on_keys
    @property
    def when_matched(self) -> list[dict]: return self._when_matched
    @property
    def when_not_matched(self) -> list[dict]: return self._when_not_matched

    # -- Builders --------------------------------------------------------

    def into(self, target: "Tabular") -> "MergePlan[O]":
        self._target = target
        return self

    def using(self, source: "Tabular | ExecutionPlan") -> "MergePlan[O]":
        self._source = source
        return self

    def on(self, predicate: "PredicateLike | list[str] | str") -> "MergePlan[O]":
        if isinstance(predicate, list):
            self._on_keys = list(predicate)
            self._on = None
            return self
        if isinstance(predicate, str) and "=" not in predicate and " " not in predicate:
            self._on_keys = [predicate]
            self._on = None
            return self
        self._on = _coerce_predicate(predicate)
        self._on_keys = None
        return self

    def when_matched_update(
        self,
        assignments: "dict[str, Any] | None" = None,
        *,
        condition: "PredicateLike | None" = None,
    ) -> "MergePlan[O]":
        self._when_matched.append({
            "condition": _coerce_predicate(condition) if condition is not None else None,
            "action": {"type": "UPDATE", "set": dict(assignments or {})},
        })
        return self

    def when_matched_delete(
        self,
        *,
        condition: "PredicateLike | None" = None,
    ) -> "MergePlan[O]":
        self._when_matched.append({
            "condition": _coerce_predicate(condition) if condition is not None else None,
            "action": {"type": "DELETE"},
        })
        return self

    def when_not_matched_insert(
        self,
        values: "dict[str, Any] | None" = None,
        *,
        columns: "list[str] | None" = None,
        condition: "PredicateLike | None" = None,
    ) -> "MergePlan[O]":
        action: dict[str, Any] = {"type": "INSERT"}
        if columns is not None:
            action["columns"] = list(columns)
        if values is not None:
            action["values"] = values
        self._when_not_matched.append({
            "condition": _coerce_predicate(condition) if condition is not None else None,
            "action": action,
        })
        return self

    # -- Copy ------------------------------------------------------------

    def copy(self) -> "MergePlan[O]":
        return MergePlan(
            target=self._target,
            source=(self._source.copy() if isinstance(self._source, ExecutionPlan)
                    else self._source),
            on=self._on,
            on_keys=list(self._on_keys) if self._on_keys else None,
            when_matched=[dict(w) for w in self._when_matched],
            when_not_matched=[dict(w) for w in self._when_not_matched],
        )

    # -- Execution -------------------------------------------------------

    def execute(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> OperationResult:
        from yggdrasil.arrow.tabular import ArrowTabular

        if self._target is None:
            raise ValueError("MergePlan has no target; call .into(target) first.")
        if self._source is None:
            raise ValueError("MergePlan has no source; call .using(source) first.")

        # Resolve source through the plan layer so wait/raise_error
        # propagate to async sources.
        if isinstance(self._source, ExecutionPlan):
            src_result = self._source.execute(wait=wait, raise_error=raise_error)
            if isinstance(src_result, OperationResult):
                src_tab = src_result.to_arrow_tabular()
            else:
                src_tab = src_result
        else:
            src_tab = self._source

        target_table = self._target.read_arrow_table()
        source_table = src_tab.read_arrow_table()

        keys = self._resolve_match_keys(target_table, source_table)
        if not keys:
            raise ValueError(
                "MergePlan needs a column-equality ON clause to execute "
                "in-memory. Pass on=['id'] or on='t.id = s.id'."
            )

        new_target, inserted, updated, deleted = _merge_arrow_tables(
            target_table,
            source_table,
            keys=keys,
            when_matched=self._when_matched,
            when_not_matched=self._when_not_matched,
        )

        # Write the merged result back to the target.  Use OVERWRITE
        # so the in-place merge replaces the old rows.
        self._target.write_table(ArrowTabular(new_target), mode=Mode.OVERWRITE)

        return OperationResult(
            operation="MERGE",
            rows_inserted=inserted,
            rows_updated=updated,
            rows_deleted=deleted,
            target=self._target,
        )

    def _resolve_match_keys(
        self,
        target_table: pa.Table,
        source_table: pa.Table,
    ) -> list[tuple[str, str]]:
        """Return [(target_col, source_col), ...] derived from the
        match condition. Handles explicit ``on_keys`` lists plus
        :class:`Predicate` ON expressions of the
        ``target.k = source.k [AND ...]`` shape."""
        if self._on_keys:
            return [(k, k) for k in self._on_keys
                    if k in target_table.column_names and k in source_table.column_names]

        if self._on is None:
            return []

        from yggdrasil.saga.expr.nodes import Column, Comparison, Logical
        from yggdrasil.saga.expr.operators import CompareOp

        comparisons: list[Comparison] = []
        if isinstance(self._on, Comparison) and self._on.op == CompareOp.EQ:
            comparisons.append(self._on)
        elif isinstance(self._on, Logical):
            for o in getattr(self._on, "operands", ()):
                if isinstance(o, Comparison) and o.op == CompareOp.EQ:
                    comparisons.append(o)

        keys: list[tuple[str, str]] = []
        tcols, scols = set(target_table.column_names), set(source_table.column_names)
        for cmp in comparisons:
            if not (isinstance(cmp.left, Column) and isinstance(cmp.right, Column)):
                continue
            ln, rn = cmp.left.name, cmp.right.name
            if ln in tcols and rn in scols:
                keys.append((ln, rn))
            elif ln in scols and rn in tcols:
                keys.append((rn, ln))
            elif ln in tcols and ln in scols:
                keys.append((ln, ln))
        return keys

    # -- Plan node conversion --------------------------------------------

    def to_plan_node(self) -> "MergeNode":
        from .nodes import MergeNode, ScanNode
        from .ops import TableRef
        target_ref = TableRef(name=getattr(self._target, "name", "?")) if self._target else None
        src_node: Any = None
        if isinstance(self._source, ExecutionPlan):
            src_node = self._source.to_plan_node()
        elif isinstance(self._source, Tabular):
            src_node = ScanNode(tabular=self._source)
        return MergeNode(
            target=target_ref,
            source=src_node,
            on=self._on,
            when_matched=list(self._when_matched) or None,
            when_not_matched=list(self._when_not_matched) or None,
        )

    def _apply_merge_node(self, node: "MergeNode") -> None:
        from .nodes import SelectNode
        self._on = node.on
        if node.when_matched:
            self._when_matched = list(node.when_matched)
        if node.when_not_matched:
            self._when_not_matched = list(node.when_not_matched)
        if isinstance(node.source, SelectNode):
            sp = SelectPlan()
            sp._apply_select_node(node.source)
            self._source = sp

    def __repr__(self) -> str:
        parts = []
        if self._target is not None:
            parts.append(f"target={type(self._target).__name__}")
        if self._source is not None:
            parts.append(f"source={type(self._source).__name__}")
        if self._on_keys:
            parts.append(f"on_keys={self._on_keys!r}")
        elif self._on is not None:
            parts.append(f"on={self._on!r}")
        if self._when_matched:
            parts.append(f"when_matched={len(self._when_matched)}")
        if self._when_not_matched:
            parts.append(f"when_not_matched={len(self._when_not_matched)}")
        return f"MergePlan({', '.join(parts) or 'unbound'})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _values_to_pydict(values: list[list[Any]], cols: list[str]) -> dict[str, list]:
    """Render ``[[Literal|raw, ...], ...]`` rows as a column-major dict."""
    from yggdrasil.saga.expr.nodes import Literal
    out: dict[str, list] = {c: [] for c in cols}
    for row in values:
        for i, c in enumerate(cols):
            cell = row[i] if i < len(row) else None
            out[c].append(cell.value if isinstance(cell, Literal) else cell)
    return out


def _resolve_action_value(
    cell: Any,
    source_row: dict[str, Any],
) -> Any:
    """Map a MERGE action RHS value to a concrete cell.

    Accepts:

    * :class:`Literal` — its ``.value``.
    * :class:`Column` — looked up in ``source_row`` (already aliased
      down to bare names by the caller).
    * Anything else — passed through (callers using the plan builder
      hand raw Python values directly).
    """
    from yggdrasil.saga.expr.nodes import Column, Literal
    if isinstance(cell, Literal):
        return cell.value
    if isinstance(cell, Column):
        return source_row.get(cell.name)
    return cell


def _merge_arrow_tables(
    target: pa.Table,
    source: pa.Table,
    *,
    keys: list[tuple[str, str]],
    when_matched: list[dict],
    when_not_matched: list[dict],
) -> tuple[pa.Table, int, int, int]:
    """Apply MERGE actions row by row, return ``(table, ins, upd, del)``.

    Index by match-key tuple on both sides so each source row visits
    its matching target row exactly once.  Actions run in declaration
    order — the first MATCHED clause whose optional condition holds
    wins, same for NOT MATCHED.  Conditions are unsupported in this
    pure-Arrow path and treated as always-true.
    """
    tkeys = [k[0] for k in keys]
    skeys = [k[1] for k in keys]

    # Build a position index on the source by its match keys.
    source_index: dict[tuple, int] = {}
    source_rows = source.to_pylist()
    for idx, row in enumerate(source_rows):
        source_index[tuple(row[c] for c in skeys)] = idx

    target_rows = target.to_pylist()
    matched_source_idxs: set[int] = set()

    inserted = updated = deleted = 0
    new_target: list[dict] = []

    for trow in target_rows:
        key = tuple(trow[c] for c in tkeys)
        sidx = source_index.get(key)
        if sidx is None:
            new_target.append(trow)
            continue
        matched_source_idxs.add(sidx)
        srow = source_rows[sidx]
        action = _pick_action(when_matched)
        if action is None:
            new_target.append(trow)
            continue
        atype = action["type"]
        if atype == "DELETE":
            deleted += 1
            continue
        if atype == "UPDATE":
            updated += 1
            assignments = action.get("set") or {}
            new_row = dict(trow)
            for col, rhs in assignments.items():
                new_row[col] = _resolve_action_value(rhs, srow)
            new_target.append(new_row)
            continue
        # Unknown action — keep the target row untouched.
        new_target.append(trow)

    # NOT MATCHED — apply INSERT to every source row we haven't yet matched.
    insert_action = _pick_action(when_not_matched)
    if insert_action and insert_action.get("type") == "INSERT":
        target_cols = target.column_names
        for sidx, srow in enumerate(source_rows):
            if sidx in matched_source_idxs:
                continue
            cols = insert_action.get("columns") or target_cols
            vals = insert_action.get("values")
            new_row: dict[str, Any] = {c: None for c in target_cols}
            if isinstance(vals, dict):
                # Builder path — column → raw value mapping.
                for c in target_cols:
                    if c in vals:
                        new_row[c] = _resolve_action_value(vals[c], srow)
                    elif c in srow:
                        new_row[c] = srow[c]
            elif isinstance(vals, list):
                # AST path — positional list aligned with ``cols``.
                for i, c in enumerate(cols):
                    if c not in target_cols:
                        continue
                    cell = vals[i] if i < len(vals) else None
                    new_row[c] = _resolve_action_value(cell, srow)
            else:
                # No explicit values — copy aligned source columns.
                for c in target_cols:
                    if c in srow:
                        new_row[c] = srow[c]
            new_target.append(new_row)
            inserted += 1

    return pa.Table.from_pylist(new_target, schema=target.schema), inserted, updated, deleted


def _pick_action(clauses: list[dict]) -> "dict | None":
    """Return the first clause's action (conditions ignored for now)."""
    for clause in clauses:
        # Future: evaluate clause["condition"] against the row.
        return clause.get("action")
    return None
