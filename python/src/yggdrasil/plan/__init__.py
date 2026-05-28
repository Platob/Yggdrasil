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
from .func_registry import BUILTIN_REGISTRY, FunctionMeta, FunctionRegistry
from .lazy import LazyTabular
from .nodes import InsertNode, MergeNode, PlanNode, ScanNode, SelectNode
from .ops import CTE, GroupByOp, JoinOp, OrderByOp, ResampleOp, SetOp, UnionOp

__all__ = [
    "BUILTIN_REGISTRY",
    "CTE",
    "ExecutionPlan",
    "FunctionMeta",
    "FunctionRegistry",
    "GroupByOp",
    "InsertNode",
    "JoinOp",
    "LazyTabular",
    "MergeNode",
    "OrderByOp",
    "PlanNode",
    "ResampleOp",
    "ScanNode",
    "SelectNode",
    "SelectPlan",
    "SetOp",
    "UnionOp",
]


def parse_sql(sql: str, *, dialect=None, default=...):
    """Parse a SQL query string into a :class:`PlanNode` tree.

    Returns *default* on parse failure when *default* is not ``...``.
    """
    from .sql_parser import parse_sql as _parse
    return _parse(sql, dialect=dialect, default=default)
