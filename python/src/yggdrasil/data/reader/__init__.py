"""Reader implementations for Yggdrasil data."""

from .base import (
    DataReader,
    ReadOptions,
    ReaderPredicate,
    ColumnPredicate,
    AndPredicate,
    OrPredicate,
    NotPredicate,
    # Helper functions for predicates
    eq, gt, lt, gte, lte, ne, is_in, not_in,
    and_, or_, not_,
)

# Import Delta reader if available
try:
    from .delta import DeltaReader, DeltaReaderConfig
except ImportError:
    pass

__all__ = [
    # Abstract reader
    "DataReader",
    "ReadOptions",
    "ReaderPredicate",
    "ColumnPredicate",
    "AndPredicate",
    "OrPredicate",
    "NotPredicate",

    # Predicate helpers
    "eq", "gt", "lt", "gte", "lte", "ne", "is_in", "not_in",
    "and_", "or_", "not_",

    # Implementations
    "DeltaReader",
    "DeltaReaderConfig",
]