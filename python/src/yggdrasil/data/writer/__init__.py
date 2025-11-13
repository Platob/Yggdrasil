"""Writer implementations for Yggdrasil data."""

from ...logging import get_logger

# Create module-level logger
logger = get_logger(__name__)

from .base import DataWriter, WriteOptions, WriteMode

# Import Delta writer if available
try:
    from .delta import DeltaWriter, DeltaWriterConfig
    logger.debug("Successfully imported Delta writer")
except ImportError:
    logger.debug("Delta writer not available")
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