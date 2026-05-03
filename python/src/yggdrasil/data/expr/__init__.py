"""Expression / Predicate AST with multi-backend emitters.

Public surface:

- :class:`Expression`, :class:`Predicate` — abstract bases.
- :class:`Column`, :class:`Literal`, :class:`Comparison`,
  :class:`Logical`, :class:`Not`, :class:`Between`,
  :class:`InList`, :class:`IsNull`, :class:`Like`, :class:`Cast`,
  :class:`Arithmetic` — concrete node types.
- :func:`col`, :func:`lit` — fluent factories.
- :class:`CompareOp`, :class:`LogicalOp`, :class:`ArithmeticOp` —
  shared operator enums.

Backends live under ``backends/``: ``to_python``, ``to_sql``,
``to_pyarrow``, ``to_polars``, ``to_pyspark`` (plus matching
``from_*`` lifters for SQL / pyarrow / polars / pyspark, where
feasible). Methods on :class:`Expression` dispatch to the matching
backend module so the optional dependencies stay optional.
"""

from .builder import col
from .nodes import (
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
    lit,
)

__all__ = [
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
    "col",
    "lit",
]
