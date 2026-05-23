"""Expression / Predicate AST with multi-backend emitters.

Public surface:

- :class:`Expression`, :class:`Predicate` — abstract bases.
- :class:`Column`, :class:`Literal`, :class:`Comparison`,
  :class:`Logical`, :class:`Not`, :class:`Between`,
  :class:`InList`, :class:`IsNull`, :class:`Like`,
  :class:`Cast`, :class:`Arithmetic` — concrete node types.
- :func:`col`, :func:`lit`, :func:`all_of`, :func:`any_of`,
  :func:`neg` — fluent factories.
- :class:`CompareOp`, :class:`LogicalOp`, :class:`ArithmeticOp` —
  shared operator enums.
- :func:`simplify` — algebraic normaliser (also exposed as
  :meth:`Expression.simplify`).
- :func:`walk`, :func:`free_columns` — pre-order visitors.
- :func:`extract_partition_filters` — over-approximate partition
  pruner.

The implementation is split across sibling modules to keep each
file focused: ``nodes.py`` for the AST dataclasses, ``operators.py``
for the operator enums, ``simplify.py`` / ``partition.py`` / ``walk.py``
for the algorithms. Callers import everything from here.

Projections live on :class:`yggdrasil.data.data_field.Field`,
which is the single canonical "selector" the tabular API
accepts — no separate selector node lives here. Build a Field
with the output :attr:`name`, optional :attr:`alias` for the
source-side label, and target :attr:`dtype`, then pass it to
``LazyTabular.select`` or a SQL ``statement.select`` list.

Per-engine compilation lives under :mod:`yggdrasil.execution.expr.backends`:
each backend ships ``to_<target>`` and (where introspection is
feasible) ``from_<target>``. The :class:`Expression` base
exposes them as instance and class methods (``to_python`` /
``to_sql`` / ``to_arrow`` / ``to_polars`` / ``to_spark`` /
``to_engine``; ``Expression.from_`` / ``from_sql`` / ``from_arrow``
/ etc.) so callers don't have to import the backend modules
directly.
"""

from .builder import all_of, any_of, col, neg
from .nodes import (
    Arithmetic,
    Between,
    Cast,
    Column,
    Comparison,
    Expression,
    InList,
    IsNull,
    Like,
    Literal,
    Logical,
    Not,
    Predicate,
    lit,
)
from .operators import ArithmeticOp, CompareOp, LogicalOp
from .partition import extract_partition_filters
from .simplify import simplify
from .walk import free_columns, walk

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
    "all_of",
    "any_of",
    "col",
    "extract_partition_filters",
    "free_columns",
    "lit",
    "neg",
    "simplify",
    "walk",
]
