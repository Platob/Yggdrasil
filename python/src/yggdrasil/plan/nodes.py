"""Immutable plan-node tree for SQL-like execution plans.

Each node is a frozen dataclass describing one relational operation.
Composite plans (CTEs, subqueries, joins) compose by nesting nodes.

The tree is pure data — execution lives in :mod:`execute`, SQL
emission in :mod:`emitters.sql`, parsing in :mod:`parsers`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yggdrasil.enums import Dialect
    from yggdrasil.execution.expr import Predicate
    from yggdrasil.io.tabular import Tabular

from .ops import CTE, FromItem, LateralViewItem, SetOp, TableRef


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PlanNode:
    """Abstract base for all plan nodes.

    Subclasses are frozen dataclasses. The base provides
    :meth:`from_sql` and :meth:`to_sql` hooks so every node
    can round-trip through SQL.
    """

    __slots__ = ()

    @classmethod
    def from_sql(
        cls,
        sql: str,
        dialect: "Dialect | str | None" = None,
        default: "Any" = ...,
    ) -> "PlanNode | Any":
        """Parse SQL into a plan node. Returns *default* on failure
        when *default* is not ``...``; raises otherwise."""
        from .sql_parser import parse_sql
        return parse_sql(sql, dialect=dialect, default=default)

    def to_sql(self, dialect: "Dialect | str | None" = None) -> str:
        from .sql_emitter import emit_sql
        return emit_sql(self, dialect=dialect)

    def execute(
        self,
        tables: "dict[str, Tabular] | None" = None,
    ) -> "Tabular":
        from .execute import execute_plan
        return execute_plan(self, tables=tables or {})


# ---------------------------------------------------------------------------
# Concrete nodes
# ---------------------------------------------------------------------------

@dataclasses.dataclass(slots=True)
class ScanNode(PlanNode):
    """Leaf — references a named table or a concrete Tabular."""
    name: str | None = None
    tabular: "Tabular | None" = None
    alias: str | None = None


@dataclasses.dataclass(slots=True)
class SelectNode(PlanNode):
    """``SELECT [DISTINCT] projections FROM source [WHERE ...] [GROUP BY ...] ...``"""
    projections: list[Any] = dataclasses.field(default_factory=list)
    from_clause: FromItem | PlanNode | None = None
    where: "Predicate | None" = None
    group_by: list[Any] | None = None
    having: "Predicate | None" = None
    qualify: "Predicate | None" = None
    order_by: list[Any] | None = None
    limit: int | None = None
    offset: int | None = None
    distinct: bool = False
    ctes: list[CTE] | None = None
    set_ops: list[SetOp] | None = None
    lateral_views: list[LateralViewItem] | None = None


@dataclasses.dataclass(slots=True)
class InsertNode(PlanNode):
    """``INSERT INTO target [(columns)] source_plan | VALUES (...)``"""
    target: TableRef | None = None
    columns: list[str] | None = None
    source: PlanNode | None = None
    values: list[list[Any]] | None = None


@dataclasses.dataclass(slots=True)
class MergeNode(PlanNode):
    """``MERGE INTO target USING source ON condition WHEN ...``"""
    target: TableRef | None = None
    source: PlanNode | FromItem | None = None
    on: "Predicate | None" = None
    when_matched: list[Any] | None = None
    when_not_matched: list[Any] | None = None
