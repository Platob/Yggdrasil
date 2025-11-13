"""Data interoperability helpers for Yggdrasil."""

from .data_cast import (
    DataCaster,
    DataCastRegistry,
    DataUtility,
    DATA_CAST_REGISTRY,
    DataType,
    SeriesLike,
)

# Conditionally import Delta classes if available
try:
    from .delta_io import DeltaReader, DeltaWriter, HAS_DELTA
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

    # Delta support
    "DeltaReader",
    "DeltaWriter",
    "HAS_DELTA",
]