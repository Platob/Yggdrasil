"""Mutable execution plan that accumulates lazy transformations.

:class:`ExecutionPlan` is the abstract base for all plan types.
:class:`SelectPlan` is the concrete implementation for
SELECT-like queries (column projection, filter, join, union,
dedup, resample, cast, limit).

The plan is **mutable**: every builder method mutates in-place and
returns ``self`` for chaining.  Call :meth:`copy` to fork a plan
before branching.

Generic over ``O`` (the :class:`CastOptions` subtype) so that
format-specific options survive through the pipeline.
"""

from __future__ import annotations

import copy as _copy
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterator,
    TypeVar,
)

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.enums import JoinType, Mode

from .ops import JoinOp, ResampleOp, UnionOp

if TYPE_CHECKING:
    from yggdrasil.enums import Dialect
    from yggdrasil.execution.expr import Predicate, PredicateLike
    from yggdrasil.io.tabular import Tabular

O = TypeVar("O", bound=CastOptions)


def _flatten_column_args(args: tuple[Any, ...]) -> list[str]:
    from yggdrasil.io.tabular.base import _flatten_column_args as _flat
    return _flat(args)


def _coerce_predicate(value: Any) -> "Predicate":
    from yggdrasil.io.tabular.base import _coerce_predicate as _coerce
    return _coerce(value)


def _coerce_column_keys(value: Any) -> list[str]:
    from yggdrasil.io.tabular.base import _coerce_column_keys as _coerce
    return _coerce(value)


def _coerce_sampling_seconds(value: Any) -> int:
    from yggdrasil.io.tabular.base import _coerce_sampling_seconds as _coerce
    return _coerce(value)


class ExecutionPlan(ABC, Generic[O]):
    """Abstract base for all execution plan types.

    Concrete subclasses: :class:`SelectPlan`, and the SQL-level
    plans in :mod:`yggdrasil.plan.nodes` (SelectNode, InsertNode,
    MergeNode).

    Provides :meth:`from_sql` / :meth:`to_sql` for SQL round-trip
    and :meth:`execute` for materialisation against a Tabular source.
    """

    __slots__ = ()

    @classmethod
    def from_sql(
        cls,
        sql: str,
        dialect: "Dialect | str | None" = None,
    ) -> "ExecutionPlan":
        """Parse a SQL query into a :class:`SelectPlan`."""
        from .sql_parser import parse_sql
        from .nodes import SelectNode
        node = parse_sql(sql, dialect=dialect)
        plan = SelectPlan()
        if isinstance(node, SelectNode):
            plan._apply_select_node(node)
        return plan

    def to_sql(self, dialect: "Dialect | str | None" = None) -> str:
        """Emit this plan as a SQL query string."""
        node = self.to_plan_node()
        from .sql_emitter import emit_sql
        return emit_sql(node, dialect=dialect)

    def to_plan_node(self) -> "PlanNode":
        """Convert this mutable plan to an immutable plan node tree."""
        from .nodes import PlanNode
        raise NotImplementedError(
            f"{type(self).__name__}.to_plan_node is not implemented."
        )

    @abstractmethod
    def execute(self, source: "Tabular[O]") -> "Tabular":
        """Materialise the plan against *source*."""

    @abstractmethod
    def copy(self) -> "ExecutionPlan[O]":
        """Return a deep copy."""


class SelectPlan(ExecutionPlan[O]):
    """Mutable SELECT-like plan: projection, filter, join, union,
    dedup, resample, cast, limit.

    Builder methods mutate in-place and return ``self``::

        plan = SelectPlan()
        plan.select("a", "b").filter(col("a") > 10).limit(100)

    Call :meth:`execute` to materialise::

        result: Tabular = plan.execute(source_tabular)
    """

    __slots__ = (
        "_select",
        "_drop",
        "_predicate",
        "_joins",
        "_unions",
        "_unique_by",
        "_resample",
        "_cast_options",
        "_limit",
    )

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

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_identity(self) -> bool:
        """``True`` when the plan would return the source unchanged."""
        return (
            self._select is None
            and self._drop is None
            and self._predicate is None
            and not self._joins
            and not self._unions
            and self._unique_by is None
            and self._resample is None
            and self._cast_options is None
            and self._limit is None
        )

    @property
    def columns(self) -> list[str] | None:
        return self._select

    @property
    def predicate(self) -> "Predicate | None":
        return self._predicate

    @property
    def joins(self) -> list[JoinOp]:
        return self._joins

    @property
    def unions(self) -> list[UnionOp]:
        return self._unions

    @property
    def limit_rows(self) -> int | None:
        return self._limit

    # ------------------------------------------------------------------
    # Column projection (replaces on each call)
    # ------------------------------------------------------------------

    def select(self, *columns: "str | Any") -> "ExecutionPlan[O]":
        """Set (or replace) the column projection."""
        cols = _flatten_column_args(columns)
        if not cols:
            raise ValueError("select needs at least one column.")
        self._select = cols
        self._drop = None
        return self

    def drop(self, *columns: "str | Any") -> "ExecutionPlan[O]":
        """Set (or replace) columns to drop."""
        cols = _flatten_column_args(columns)
        if cols:
            self._drop = cols
        return self

    # ------------------------------------------------------------------
    # Pushdown predicate (ANDs with existing)
    # ------------------------------------------------------------------

    def filter(self, predicate: "PredicateLike") -> "ExecutionPlan[O]":
        """Add a row filter. Multiple calls are ANDed together."""
        pred = _coerce_predicate(predicate)
        if self._predicate is not None:
            self._predicate = self._predicate & pred
        else:
            self._predicate = pred
        return self

    def clear_filter(self) -> "ExecutionPlan[O]":
        """Remove the accumulated predicate."""
        self._predicate = None
        return self

    # ------------------------------------------------------------------
    # Joins
    # ------------------------------------------------------------------

    def join(
        self,
        right: "Tabular",
        on: "str | list[str] | Any",
        how: "str | JoinType" = "inner",
        *,
        suffix: str = "_right",
    ) -> "ExecutionPlan[O]":
        """Append a join with *right* on the given key columns."""
        keys = _coerce_column_keys(on) if isinstance(on, str) else _coerce_column_keys(on)
        if not keys:
            raise ValueError("join needs at least one key column.")
        self._joins.append(JoinOp(
            right=right,
            on=keys,
            how=JoinType.from_(how),
            right_suffix=suffix,
        ))
        return self

    # ------------------------------------------------------------------
    # Unions
    # ------------------------------------------------------------------

    def union(
        self,
        other: "Tabular",
        *,
        mode: "str | Mode | None" = None,
    ) -> "ExecutionPlan[O]":
        """Append a UNION ALL with *other*."""
        self._unions.append(UnionOp(
            other=other,
            mode=Mode.from_(mode, default=Mode.IGNORE),
        ))
        return self

    # ------------------------------------------------------------------
    # Dedup / resample
    # ------------------------------------------------------------------

    def unique(self, by: "str | Any | list[Any]") -> "ExecutionPlan[O]":
        """Set dedup keys. ``None`` clears."""
        self._unique_by = _coerce_column_keys(by) or None
        return self

    def resample(
        self,
        on: "str | Any",
        sampling: "int | float | Any",
        *,
        partition_by: "str | Any | list[Any] | None" = None,
        fill_strategy: "str | None" = "ffill",
    ) -> "ExecutionPlan[O]":
        """Set a time-grid resample operation."""
        from yggdrasil.io.tabular.base import _coerce_column_name
        self._resample = ResampleOp(
            time_column=_coerce_column_name(on),
            sampling_seconds=_coerce_sampling_seconds(sampling),
            partition_by=_coerce_column_keys(partition_by) if partition_by else [],
            fill_strategy=fill_strategy,
        )
        return self

    # ------------------------------------------------------------------
    # Cast / options override
    # ------------------------------------------------------------------

    def cast(self, options: "CastOptions | None" = None, **kwargs: Any) -> "ExecutionPlan[O]":
        """Set a :class:`CastOptions` to apply at read time."""
        self._cast_options = CastOptions.check(options, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Row limit
    # ------------------------------------------------------------------

    def limit(self, n: int | None) -> "ExecutionPlan[O]":
        """Cap the result to *n* rows.  ``None`` removes the cap."""
        self._limit = n
        return self

    # ------------------------------------------------------------------
    # Copy / clear
    # ------------------------------------------------------------------

    def copy(self) -> "ExecutionPlan[O]":
        """Return a deep copy so mutations don't alias."""
        clone: ExecutionPlan[O] = object.__new__(type(self))
        clone._select = list(self._select) if self._select else None
        clone._drop = list(self._drop) if self._drop else None
        clone._predicate = self._predicate
        clone._joins = list(self._joins)
        clone._unions = list(self._unions)
        clone._unique_by = list(self._unique_by) if self._unique_by else None
        clone._resample = self._resample
        clone._cast_options = self._cast_options
        clone._limit = self._limit
        return clone

    def clear(self) -> "ExecutionPlan[O]":
        """Reset the plan to the identity transformation."""
        self._select = None
        self._drop = None
        self._predicate = None
        self._joins.clear()
        self._unions.clear()
        self._unique_by = None
        self._resample = None
        self._cast_options = None
        self._limit = None
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _build_read_options(self, source: "Tabular[O]") -> "O | None":
        """Build :class:`CastOptions` for the source read.

        Pushes the predicate into CastOptions as a hint for format-level
        readers that support pushdown (Parquet row-group pruning, etc.).
        The predicate is *also* applied explicitly in :meth:`execute` so
        in-memory sources that ignore the CastOptions hint still filter.
        """
        kwargs: dict[str, Any] = {}
        if self._predicate is not None and not self._joins:
            kwargs["predicate"] = self._predicate
        if self._cast_options is not None:
            return self._cast_options.copy(**kwargs) if kwargs else self._cast_options
        if kwargs:
            return source.check_options(None, overrides=kwargs)
        return None

    def execute(self, source: "Tabular[O]") -> "Tabular":
        """Materialise the plan against *source*.

        Execution order:

        1. Read from *source* (predicate hint pushed into CastOptions
           for format-level readers that support it).
        2. Apply predicate as explicit row filter (before joins when
           no joins are pending, after joins otherwise).
        3. Apply joins (left-to-right).
        4. Apply unions.
        5. Apply column projection (select / drop).
        6. Apply unique / resample.
        7. Apply cast options.
        8. Apply row limit.
        """
        if self.is_identity:
            return source

        from yggdrasil.arrow.tabular import ArrowTabular

        options = self._build_read_options(source)
        result: "Tabular" = source

        if options is not None:
            result = ArrowTabular(result.read_arrow_table(options))

        # Filter before joins when possible (reduces data volume).
        if self._predicate is not None and not self._joins:
            result = result.filter(self._predicate)

        # Joins.
        if self._joins:
            result = self._apply_joins(result)

        # Unions.
        for uop in self._unions:
            other = uop.other
            if hasattr(other, "_execute_plan"):
                other = other._execute_plan()
            result = result.union(other, mode=uop.mode)

        # Filter after joins (predicate may reference join columns).
        if self._predicate is not None and self._joins:
            result = result.filter(self._predicate)

        # Projection.
        if self._select is not None:
            result = result.select(*self._select)
        elif self._drop is not None:
            result = result.drop(*self._drop)

        # Unique / resample.
        if self._unique_by is not None:
            result = result.unique(self._unique_by)
        if self._resample is not None:
            r = self._resample
            result = result.resample(
                r.time_column,
                r.sampling_seconds,
                partition_by=r.partition_by or None,
                fill_strategy=r.fill_strategy,
            )

        # Cast.
        if self._cast_options is not None:
            result = result.cast(self._cast_options)

        # Limit.
        if self._limit is not None:
            result = self._apply_limit(result, self._limit)

        return result

    def execute_arrow_batches(
        self,
        source: "Tabular[O]",
        options: "O | None" = None,
    ) -> "Iterator[pa.RecordBatch]":
        """Execute the plan and yield Arrow record batches."""
        result = self.execute(source)
        yield from result.read_arrow_batches(options)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_joins(self, left: "Tabular") -> "Tabular":
        """Apply the join chain, materialising each step as Arrow."""
        from yggdrasil.arrow.tabular import ArrowTabular

        left_table = left.read_arrow_table()

        for jop in self._joins:
            right = jop.right
            if hasattr(right, "_execute_plan"):
                right = right._execute_plan()
            right_table = right.read_arrow_table()
            left_table = left_table.join(
                right_table,
                keys=jop.on,
                join_type=jop.how.arrow,
                right_suffix=jop.right_suffix,
            )

        return ArrowTabular(left_table)

    @staticmethod
    def _apply_limit(result: "Tabular", n: int) -> "Tabular":
        from yggdrasil.arrow.tabular import ArrowTabular

        table = result.read_arrow_table()
        if table.num_rows > n:
            table = table.slice(0, n)
        return ArrowTabular(table)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts: list[str] = []
        if self._predicate is not None:
            parts.append(f"filter={self._predicate!r}")
        if self._joins:
            parts.append(f"joins={len(self._joins)}")
        if self._unions:
            parts.append(f"unions={len(self._unions)}")
        if self._select is not None:
            parts.append(f"select={self._select!r}")
        if self._drop is not None:
            parts.append(f"drop={self._drop!r}")
        if self._unique_by is not None:
            parts.append(f"unique_by={self._unique_by!r}")
        if self._resample is not None:
            parts.append(f"resample={self._resample.time_column!r}")
        if self._cast_options is not None:
            parts.append("cast=True")
        if self._limit is not None:
            parts.append(f"limit={self._limit}")
        body = ", ".join(parts) if parts else "identity"
        return f"SelectPlan({body})"

    # ------------------------------------------------------------------
    # Plan node conversion
    # ------------------------------------------------------------------

    def to_plan_node(self) -> "PlanNode":
        from .nodes import SelectNode
        from yggdrasil.execution.expr.nodes import Column, Star

        projections = []
        if self._select:
            projections = [Column(name=c) for c in self._select]
        else:
            projections = [Star()]

        return SelectNode(
            projections=projections,
            where=self._predicate,
            limit=self._limit,
            distinct=bool(self._unique_by),
        )

    def _apply_select_node(self, node: "SelectNode") -> None:
        """Populate this plan from a parsed SelectNode."""
        from yggdrasil.execution.expr.nodes import Alias, Column, Star

        if node.projections:
            cols = []
            all_columns = True
            for p in node.projections:
                if isinstance(p, Star):
                    all_columns = False
                    break
                if isinstance(p, Column):
                    cols.append(p.alias or p.name)
                elif isinstance(p, Alias) and isinstance(p.expr, Column):
                    cols.append(p.name)
                else:
                    all_columns = False
                    break
            if all_columns and cols:
                self._select = cols

        if node.where is not None:
            from yggdrasil.execution.expr import Predicate
            if isinstance(node.where, Predicate):
                self._predicate = node.where

        if node.limit is not None:
            self._limit = node.limit

        if node.distinct:
            if self._select:
                self._unique_by = list(self._select)
