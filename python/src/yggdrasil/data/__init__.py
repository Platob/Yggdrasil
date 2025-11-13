"""Data interoperability helpers for Yggdrasil."""

from .data_cast import (
    DataCaster,
    DataCastRegistry,
    DataUtility,
    DATA_CAST_REGISTRY,
    DataType,
    SeriesLike,
)

# Import abstract reader/writer classes
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
)

from .writer import (
    DataWriter,
    WriteOptions,
    WriteMode,
)

# Conditionally import Delta classes if available
try:
    from .delta_common import HAS_DELTA
    from .delta_reader import DeltaReader, DeltaReaderConfig
    from .delta_writer import DeltaWriter, DeltaWriterConfig
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