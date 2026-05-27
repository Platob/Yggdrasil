"""Lightweight operation descriptors for execution plans.

Each dataclass is a value-only record — no behavior, no engine
imports. The execution engine reads these and dispatches to the
appropriate Tabular / Arrow / Spark surface.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yggdrasil.enums import JoinType, Mode
    from yggdrasil.execution.expr import Expression, Predicate
    from yggdrasil.io.tabular import Tabular


# ---------------------------------------------------------------------------
# Tabular-level operations (from the original plan module)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(slots=True, frozen=True)
class JoinOp:
    """A pending join between the current result and *right*."""
    right: Tabular
    on: list[str]
    how: JoinType
    right_suffix: str = "_right"


@dataclasses.dataclass(slots=True, frozen=True)
class UnionOp:
    """A pending UNION ALL with *other*."""
    other: Tabular
    mode: Mode


@dataclasses.dataclass(slots=True, frozen=True)
class ResampleOp:
    """A pending time-grid resample."""
    time_column: str
    sampling_seconds: int
    partition_by: list[str]
    fill_strategy: str | None = "ffill"


# ---------------------------------------------------------------------------
# SQL-level from-clause items
# ---------------------------------------------------------------------------

@dataclasses.dataclass(slots=True)
class TableRef:
    """A concrete table reference in a FROM clause."""
    name: str
    alias: str | None = None
    schema: str | None = None
    catalog: str | None = None


@dataclasses.dataclass(slots=True)
class SubqueryRef:
    """A subquery in a FROM clause: ``(SELECT ...) AS alias``."""
    plan: Any  # forward ref to SelectPlan
    alias: str


@dataclasses.dataclass(slots=True)
class JoinClause:
    """A SQL JOIN between two from-items."""
    left: Any   # TableRef | SubqueryRef | JoinClause | LateralViewItem
    right: Any
    join_type: JoinType
    on: Predicate | None = None


@dataclasses.dataclass(slots=True)
class LateralViewItem:
    """A LATERAL VIEW clause: ``LATERAL VIEW func(expr) alias AS col1, col2``."""
    function: Expression
    table_alias: str
    column_aliases: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True, frozen=True)
class CTE:
    """A WITH clause entry: ``name AS (plan)``."""
    name: str
    plan: Any  # forward ref to SelectPlan


@dataclasses.dataclass(slots=True, frozen=True)
class SetOp:
    """A set operation between two queries."""
    kind: str   # "UNION ALL", "UNION", "INTERSECT", "EXCEPT"
    plan: Any   # forward ref to SelectPlan


# Type alias for anything that can appear in a FROM clause
FromItem = TableRef | SubqueryRef | JoinClause | LateralViewItem
