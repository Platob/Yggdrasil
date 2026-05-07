"""Lazy, pushable execution plans for :class:`Tabular`.

This package consolidates the cross-engine expression AST
(:mod:`.expr`) and the SQL surface (:mod:`.sql`) under one roof, and
adds the :class:`ExecutionPlan` abstraction
:class:`yggdrasil.io.tabular.lazy.LazyTabular` consumes.

Design
------

- :mod:`.expr` is the canonical predicate / projection AST. Every
  backend (polars, pyarrow, pyspark, SQL, plain Python) compiles to
  and from it via the lifters on :class:`Expression`.
- :mod:`.sql` is the SQL parser + planner that emits this same AST
  (plus the higher-level plan-node tree in :mod:`.sql.plan`) so a
  textual query and a builder-side ``where(col(...) > 0)`` end up
  speaking the same intermediate representation.
- :mod:`.plan` defines :class:`ExecutionPlan`, the small, immutable
  op-list LazyTabular threads. Each plan node compiles down to
  whatever backend the source Tabular exposes a hook for — polars
  LazyFrame is the default, with pyarrow / SQL pushdown layered on
  top when the source supports it.
"""

from yggdrasil.io.tabular.execution.expr import (
    Arithmetic,
    ArithmeticOp,
    Between,
    Cast,
    Column,
    Comparison,
    CompareOp,
    Expression,
    InList,
    IsNull,
    Like,
    Literal,
    Logical,
    LogicalOp,
    Not,
    Predicate,
    Selector,
    all_of,
    any_of,
    col,
    lit,
    neg,
    select,
)
from yggdrasil.io.tabular.execution.plan import (
    ExecutionPlan,
    PlanOp,
    Filter as FilterOp,
    Select as SelectOp,
    GroupByAgg as GroupByAggOp,
    Apply as ApplyOp,
)


__all__ = [
    # expr
    "Arithmetic",
    "ArithmeticOp",
    "Between",
    "Cast",
    "Column",
    "Comparison",
    "CompareOp",
    "Expression",
    "InList",
    "IsNull",
    "Like",
    "Literal",
    "Logical",
    "LogicalOp",
    "Not",
    "Predicate",
    "Selector",
    "all_of",
    "any_of",
    "col",
    "lit",
    "neg",
    "select",
    # plan
    "ExecutionPlan",
    "PlanOp",
    "FilterOp",
    "SelectOp",
    "GroupByAggOp",
    "ApplyOp",
]
