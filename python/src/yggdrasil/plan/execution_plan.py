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

from .ops import JoinOp, ResampleOp, UnionOp

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
                 "_unique_by", "_resample", "_cast_options", "_limit")

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

    # -- Introspection ---------------------------------------------------

    @property
    def is_identity(self) -> bool:
        return (self._select is None and self._drop is None
                and self._predicate is None and not self._joins
                and not self._unions and self._unique_by is None
                and self._resample is None and self._cast_options is None
                and self._limit is None)

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
        return c

    def clear(self) -> "SelectPlan[O]":
        self._select = self._drop = self._predicate = None
        self._joins.clear(); self._unions.clear()
        self._unique_by = self._resample = self._cast_options = self._limit = None
        return self

    # -- Execution -------------------------------------------------------

    def execute(self, source: "Tabular[O]") -> "Tabular":
        if self.is_identity:
            return source
        from yggdrasil.arrow.tabular import ArrowTabular

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

        # Limit
        if self._limit is not None:
            table = result.read_arrow_table()
            if table.num_rows > self._limit:
                table = table.slice(0, self._limit)
            result = ArrowTabular(table)

        return result

    # -- Plan node conversion --------------------------------------------

    def to_plan_node(self):
        from .nodes import SelectNode
        from yggdrasil.execution.expr.nodes import Column, Star
        return SelectNode(
            projections=[Column(name=c) for c in self._select] if self._select else [Star()],
            where=self._predicate, limit=self._limit, distinct=bool(self._unique_by))

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
        if node.distinct and self._select:
            self._unique_by = list(self._select)

    def __repr__(self) -> str:
        parts = []
        if self._predicate: parts.append(f"filter={self._predicate!r}")
        if self._joins: parts.append(f"joins={len(self._joins)}")
        if self._unions: parts.append(f"unions={len(self._unions)}")
        if self._select: parts.append(f"select={self._select!r}")
        if self._drop: parts.append(f"drop={self._drop!r}")
        if self._unique_by: parts.append(f"unique_by={self._unique_by!r}")
        if self._resample: parts.append(f"resample={self._resample.time_column!r}")
        if self._cast_options: parts.append("cast=True")
        if self._limit is not None: parts.append(f"limit={self._limit}")
        return f"SelectPlan({', '.join(parts) or 'identity'})"
