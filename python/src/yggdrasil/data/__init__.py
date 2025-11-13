"""Data interoperability helpers for Yggdrasil."""

from .data_cast import (
    DataCaster,
    DataCastRegistry,
    DataUtility,
    DATA_CAST_REGISTRY,
    DataType,
    SeriesLike,
)

# Re-export reader and writer modules
from .reader import (
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
    # Implementations
    DeltaReader,
    DeltaReaderConfig,
)

from .writer import (
    DataWriter,
    WriteOptions,
    WriteMode,
    # Implementations
    DeltaWriter,
    DeltaWriterConfig,
)

# Determine if Delta Lake is available
try:
    from .reader.delta import HAS_DELTA
except ImportError:
    HAS_DELTA = False

__all__ = [
    # Data casting
    "DataCaster",
    "DataCastRegistry",
    "DataUtility",
    "DATA_CAST_REGISTRY",
    "DataType",
    "SeriesLike",

    # Abstract reader/writer
    "DataReader",
    "ReadOptions",
    "ReaderPredicate",
    "ColumnPredicate",
    "AndPredicate",
    "OrPredicate",
    "NotPredicate",
    "DataWriter",
    "WriteOptions",
    "WriteMode",

    # Predicate helpers
    "eq", "gt", "lt", "gte", "lte", "ne", "is_in", "not_in",
    "and_", "or_", "not_",

    # Delta support
    "DeltaReader",
    "DeltaWriter",
    "DeltaReaderConfig",
    "DeltaWriterConfig",
    "HAS_DELTA",
]