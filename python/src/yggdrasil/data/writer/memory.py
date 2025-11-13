"""In-memory data writer implementation for Yggdrasil."""

from __future__ import annotations

from typing import Dict, Any, Optional

import polars as pl

from .base import DataWriter, WriteOptions, WriteMode
from ...logging import get_logger

# Create module-level logger
logger = get_logger(__name__)


class MemoryWriterConfig:
    """Configuration for in-memory writer."""
    pass


class MemoryWriter(DataWriter[MemoryWriterConfig]):
    """In-memory writer for storing Polars DataFrames.

    This writer is primarily intended for testing purposes or
    as a simple in-memory data store for small datasets.
    """

    def __init__(self, config: MemoryWriterConfig):
        """Initialize the writer.

        Args:
            config: Configuration for the writer.
        """
        self.config = config
        self._data: Optional[pl.DataFrame] = None
        self._write_count = 0
        self._schema = None
        logger.debug("Initialized MemoryWriter")

    def write_polars(
        self, df: pl.DataFrame, options: Optional[WriteOptions] = None
    ) -> Dict[str, Any]:
        """Write a Polars DataFrame to memory.

        Args:
            df: The DataFrame to write.
            options: Options controlling how to write the data.

        Returns:
            Dict[str, Any]: Dictionary with metadata about the operation.
        """
        logger.debug(f"Writing DataFrame with {len(df)} rows and options: {options}")

        # Use default options if none provided
        if options is None:
            options = WriteOptions()
            logger.debug("Using default write options")

        if self.exists() and options.mode == WriteMode.ERROR:
            error_msg = "Data already exists and mode is 'error'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if self.exists() and options.mode == WriteMode.IGNORE:
            logger.info("Skipping write operation: data exists and mode is 'ignore'")
            return {"status": "skipped", "reason": "Data exists and mode is 'ignore'"}

        # Apply column mapping if provided
        if options.column_mapping:
            logger.debug(f"Applying column mapping: {options.column_mapping}")
            df = df.rename(options.column_mapping)

        # Apply type coercion if requested
        if options.coerce_to_schema and self.exists() and self._schema:
            logger.debug("Coercing DataFrame to match schema")
            df = self._coerce_dataframe_to_schema(df, self._schema)

        # Handle different write modes
        operation = options.mode.value
        if self.exists() and options.mode == WriteMode.APPEND:
            # Append to existing data
            logger.debug("Appending data")
            self._data = pl.concat([self._data, df], how="diagonal_relaxed")
        else:  # Overwrite or first write
            logger.debug("Setting data (overwrite or first write)")
            # Set the data
            self._data = df
            # Store the schema
            self._schema = {col: dtype for col, dtype in zip(df.columns, df.dtypes)}

        self._write_count += 1
        logger.info(f"Write operation completed: {operation}, total rows: {len(self._data)}")

        return {
            "status": "success",
            "operation": operation,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "total_rows": len(self._data),
            "write_count": self._write_count
        }

    def exists(self) -> bool:
        """Check if data exists.

        Returns:
            bool: True if data exists, False otherwise.
        """
        return self._data is not None

    def get_schema(self) -> Optional[Dict[str, pl.DataType]]:
        """Get the schema of the data if it exists.

        Returns:
            Optional[Dict[str, pl.DataType]]: Dictionary mapping column names to data types,
                or None if data doesn't exist.
        """
        if not self.exists():
            return None
        return self._schema

    def get_data(self) -> Optional[pl.DataFrame]:
        """Get the stored data (test helper method).

        Returns:
            Optional[pl.DataFrame]: The stored DataFrame, or None if no data.
        """
        return self._data