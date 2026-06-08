"""Saga — yggdrasil's unified, autonomous, lazy data engine.

Saga is the single home for schema-aware lazy compute over
:class:`~yggdrasil.io.tabular.base.Tabular` data. It leverages the core
``Tabular`` / ``Field`` / ``DataType`` model and layers three things on top:

- :mod:`yggdrasil.saga.expr` — the expression / predicate AST with
  multi-backend emitters (python / arrow / polars / spark / sql).
- :mod:`yggdrasil.saga.plan` — mutable, autonomous execution plans
  (:class:`SelectPlan` / :class:`InsertPlan` / :class:`MergePlan`), the
  immutable plan-node tree, the lazy :class:`LazyTabular`, SQL
  parse/emit across dialects, and the Arrow-native UDF registry.
- :class:`Saga` — the engine facade that ties them together: register
  tables, parse SQL from many dialects, build/execute autonomous lazy
  plans against a catalog.

The top-level surface re-exports the most-used names so callers reach for
one import::

    from yggdrasil.saga import Saga, parse_sql, col, lit, SelectPlan

> PARITY: the JS/TS port (``packages/yggdrasil``) does not yet mirror
> ``saga`` — when it gains a lazy/plan layer, mirror this package's layout
> (``expr/`` + ``plan/`` + the engine facade) file-for-file.
"""

from .engine import Saga, SagaSession
from .expr import (
    Alias,
    Arithmetic,
    ArithmeticOp,
    Between,
    CaseWhen,
    Cast,
    Column,
    Comparison,
    CompareOp,
    Expression,
    ExpressionLike,
    FunctionCall,
    InList,
    IsNull,
    Like,
    Literal,
    Logical,
    LogicalOp,
    Not,
    Predicate,
    PredicateLike,
    SortOrder,
    Star,
    Subscript,
    WindowFunction,
    WindowSpec,
    all_of,
    any_of,
    col,
    extract_partition_filters,
    free_columns,
    lit,
    neg,
    walk,
)
from .plan import (
    BUILTIN_REGISTRY,
    CTE,
    ExecutionPlan,
    ExecutionResult,
    FunctionMeta,
    FunctionRegistry,
    GroupByOp,
    InsertNode,
    InsertPlan,
    JoinOp,
    LazyTabular,
    MergeNode,
    MergePlan,
    OperationResult,
    OrderByOp,
    PlanNode,
    ResampleOp,
    ScanNode,
    SelectNode,
    SelectPlan,
    SetOp,
    UnionOp,
    parse_sql,
)

__all__ = [
    "BUILTIN_REGISTRY",
    "CTE",
    "Alias",
    "Arithmetic",
    "ArithmeticOp",
    "Between",
    "CaseWhen",
    "Cast",
    "Column",
    "Comparison",
    "CompareOp",
    "ExecutionPlan",
    "ExecutionResult",
    "Expression",
    "ExpressionLike",
    "FunctionCall",
    "FunctionMeta",
    "FunctionRegistry",
    "GroupByOp",
    "InList",
    "InsertNode",
    "InsertPlan",
    "IsNull",
    "JoinOp",
    "LazyTabular",
    "Like",
    "Literal",
    "Logical",
    "LogicalOp",
    "MergeNode",
    "MergePlan",
    "Not",
    "OperationResult",
    "OrderByOp",
    "PlanNode",
    "Predicate",
    "PredicateLike",
    "ResampleOp",
    "Saga",
    "SagaSession",
    "ScanNode",
    "SelectNode",
    "SelectPlan",
    "SetOp",
    "SortOrder",
    "Star",
    "Subscript",
    "UnionOp",
    "WindowFunction",
    "WindowSpec",
    "all_of",
    "any_of",
    "col",
    "extract_partition_filters",
    "free_columns",
    "lit",
    "neg",
    "parse_sql",
    "walk",
]
