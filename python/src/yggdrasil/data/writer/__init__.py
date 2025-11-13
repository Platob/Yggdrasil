"""Writer implementations for Yggdrasil data."""

from .base import DataWriter, WriteOptions, WriteMode

# Import Delta writer if available
try:
    from .delta import DeltaWriter, DeltaWriterConfig
except ImportError:
    pass

__all__ = [
    # Abstract writer
    "DataWriter",
    "WriteOptions",
    "WriteMode",

    # Implementations
    "DeltaWriter",
    "DeltaWriterConfig",
]