"""Lazy execution plan for :class:`Tabular` transformations.

Public surface:

- :class:`ExecutionPlan` — mutable plan accumulating select, filter,
  join, union, unique, resample, cast, and limit operations.
- :class:`LazyTabular` — a :class:`Tabular` subclass that defers
  execution until a ``read_*`` method is called.
- :class:`JoinOp`, :class:`UnionOp`, :class:`ResampleOp` — immutable
  operation descriptors inspectable on the plan.
"""

from .execution_plan import ExecutionPlan
from .lazy import LazyTabular
from .ops import JoinOp, ResampleOp, UnionOp

__all__ = [
    "ExecutionPlan",
    "JoinOp",
    "LazyTabular",
    "ResampleOp",
    "UnionOp",
]
