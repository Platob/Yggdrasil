"""Lazy execution plan for :class:`Tabular` transformations.

Public surface:

- :class:`ExecutionPlan` — abstract base for all plan types, with
  :meth:`from_sql` / :meth:`to_sql` for SQL round-trip.
- :class:`SelectPlan` — mutable plan accumulating select, filter,
  join, union, unique, resample, cast, and limit operations.
- :class:`LazyTabular` — a :class:`Tabular` subclass that defers
  execution until a ``read_*`` method is called.
- :class:`SelectNode`, :class:`InsertNode`, :class:`MergeNode`,
  :class:`ScanNode` — immutable SQL plan nodes.
- :class:`JoinOp`, :class:`UnionOp`, :class:`ResampleOp` — immutable
  operation descriptors.
- :func:`parse_sql` — parse a SQL string into a plan node tree.
"""

from .execution_plan import ExecutionPlan, SelectPlan
from .lazy import LazyTabular
from .nodes import InsertNode, MergeNode, PlanNode, ScanNode, SelectNode
from .ops import CTE, JoinOp, ResampleOp, SetOp, UnionOp

__all__ = [
    "CTE",
    "ExecutionPlan",
    "InsertNode",
    "JoinOp",
    "LazyTabular",
    "MergeNode",
    "PlanNode",
    "ResampleOp",
    "ScanNode",
    "SelectNode",
    "SelectPlan",
    "SetOp",
    "UnionOp",
]


def parse_sql(sql: str, *, dialect=None):
    """Parse a SQL query string into a :class:`PlanNode` tree."""
    from .sql_parser import parse_sql as _parse
    return _parse(sql, dialect=dialect)
