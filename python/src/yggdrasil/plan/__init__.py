"""Lazy execution plan for :class:`Tabular` transformations.

Public surface:

- :class:`ExecutionPlan` — abstract base for all plan types (itself a
  :class:`Tabular`), with :meth:`from_sql` / :meth:`to_sql` for SQL
  round-trip and an :meth:`execute(wait, raise_error)` entry point.
- :class:`SelectPlan` — mutable plan accumulating select, filter,
  join, union, unique, resample, cast, and limit operations.
- :class:`InsertPlan` — mutable plan for ``INSERT INTO target ...``;
  carries target + source so it runs standalone.
- :class:`MergePlan` — mutable plan for ``MERGE INTO target USING
  source ON ... WHEN ...``; carries target + source + actions.
- :class:`OperationResult` — metadata (rows inserted/updated/deleted
  + target Tabular) returned by INSERT / MERGE / UPDATE / DELETE
  plan executions.
- :class:`LazyTabular` — a :class:`Tabular` subclass that defers
  execution until a ``read_*`` method is called.
- :class:`SelectNode`, :class:`InsertNode`, :class:`MergeNode`,
  :class:`ScanNode` — immutable SQL plan nodes.
- :class:`JoinOp`, :class:`UnionOp`, :class:`ResampleOp` — immutable
  operation descriptors.
- :func:`parse_sql` — parse a SQL string into a plan node tree.
"""

from .execution_plan import ExecutionPlan, InsertPlan, MergePlan, SelectPlan
from .func_registry import BUILTIN_REGISTRY, FunctionMeta, FunctionRegistry
from .lazy import LazyTabular
from .nodes import InsertNode, MergeNode, PlanNode, ScanNode, SelectNode
from .operation_result import OperationResult
from .ops import CTE, GroupByOp, JoinOp, OrderByOp, ResampleOp, SetOp, UnionOp

__all__ = [
    "BUILTIN_REGISTRY",
    "CTE",
    "ExecutionPlan",
    "FunctionMeta",
    "FunctionRegistry",
    "GroupByOp",
    "InsertNode",
    "InsertPlan",
    "JoinOp",
    "LazyTabular",
    "MergeNode",
    "MergePlan",
    "OperationResult",
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
