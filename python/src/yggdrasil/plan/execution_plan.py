"""Mutable execution plan for lazy Tabular transformations.

:class:`ExecutionPlan` is the abstract base (with ``from_sql`` /
``to_sql``).  :class:`SelectPlan` is the concrete builder for
SELECT-like queries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from yggdrasil.data.options import CastOptions
from yggdrasil.enums import JoinType, Mode
from yggdrasil.io.tabular.base import (
    _coerce_column_keys,
    _coerce_predicate,
    _coerce_sampling_seconds,
    _flatten_column_args,
)

from .ops import CTE, GroupByOp, JoinOp, OrderByOp, ResampleOp, UnionOp

if TYPE_CHECKING:
    from yggdrasil.enums import Dialect
    from yggdrasil.execution.expr import Predicate, PredicateLike
    from yggdrasil.io.tabular import Tabular

O = TypeVar("O", bound=CastOptions)


class ExecutionPlan(ABC, Generic[O]):
    """Abstract base — provides ``from_sql`` / ``to_sql`` / ``execute``."""

    __slots__ = ()

    @classmethod
    def from_sql(cls, sql: str, dialect: "Dialect | str | None" = None,
                 default: Any = ...) -> "ExecutionPlan | Any":
        from .sql_parser import parse_sql
        from .nodes import SelectNode
        node = parse_sql(sql, dialect=dialect, default=default)
        if default is not ... and not isinstance(node, SelectNode):
            return default if node is default else SelectPlan()
        plan = SelectPlan()
        if isinstance(node, SelectNode):
            plan._apply_select_node(node)
        return plan

    def to_sql(self, dialect: "Dialect | str | None" = None) -> str:
        from .sql_emitter import emit_sql
        return emit_sql(self.to_plan_node(), dialect=dialect)

    def to_plan_node(self):
        raise NotImplementedError

    @abstractmethod
    def execute(self, source: "Tabular[O]") -> "Tabular": ...

    @abstractmethod
    def copy(self) -> "ExecutionPlan[O]": ...


class SelectPlan(ExecutionPlan[O]):
    """Mutable SELECT-like plan: projection, filter, join, union,
    dedup, resample, cast, limit."""

    __slots__ = ("_select", "_drop", "_predicate", "_joins", "_unions",
                 "_unique_by", "_resample", "_cast_options", "_limit",
                 "_group_by", "_having", "_order_by", "_ctes", "_offset")

    def __init__(self) -> None:
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
        c: SelectPlan[O] = object.__new__(type(self))
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

    def execute(self, source: "Tabular[O]") -> "Tabular":
        if self.is_identity:
            return source
        from yggdrasil.arrow.tabular import ArrowTabular
        import pyarrow as pa
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
                right = jop.right._execute_plan() if hasattr(jop.right, "_execute_plan") else jop.right
                left_table = left_table.join(right.read_arrow_table(),
                    keys=jop.on, join_type=jop.how.arrow, right_suffix=jop.right_suffix)
            result = ArrowTabular(left_table)

        # Unions
        for uop in self._unions:
            other = uop.other._execute_plan() if hasattr(uop.other, "_execute_plan") else uop.other
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
                    # Parse "func(col)" or "func(*)"
                    if "(" in expr_str:
                        fn = expr_str[:expr_str.index("(")].strip().upper()
                        col_part = expr_str[expr_str.index("(") + 1:expr_str.rindex(")")].strip()
                    else:
                        fn, col_part = expr_str.upper(), gb.keys[0]
                    _agg_map = {"COUNT": "count", "SUM": "sum", "AVG": "mean",
                                "MIN": "min", "MAX": "max", "MEAN": "mean"}
                    pa_func = _agg_map.get(fn, fn.lower())
                    # Resolve column: use original case from table schema
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
        from yggdrasil.execution.expr.nodes import Column, SortOrder, Star
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
        from yggdrasil.execution.expr.nodes import Alias, Column, Star
        from yggdrasil.execution.expr import Predicate
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
            from yggdrasil.execution.expr.nodes import SortOrder
            ob = []
            for o in node.order_by:
                if isinstance(o, SortOrder) and isinstance(o.expr, Column):
                    ob.append((o.expr.name, o.ascending))
            if ob:
                self._order_by = OrderByOp(keys=ob)

    def __repr__(self) -> str:
        parts = []
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
