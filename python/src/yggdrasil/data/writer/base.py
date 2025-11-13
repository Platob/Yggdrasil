"""Abstract base classes for data writers in Yggdrasil."""

from __future__ import annotations

import abc
import dataclasses
import enum
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union, Type

import polars as pl

from ..data_cast import DataCaster, DataUtility, DATA_CAST_REGISTRY

# Import logging
from ...logging import get_logger

# Create module-level logger
logger = get_logger(__name__)


# Type variable for implementing generic writer with specific config type
ConfigT = TypeVar('ConfigT')


class WriteMode(enum.Enum):
    """Enumeration of write modes."""

    ERROR = "error"
    """Raise an exception if the target already exists."""

    APPEND = "append"
    """Append data to the target if it exists."""

    OVERWRITE = "overwrite"
    """Replace the target if it exists."""

    IGNORE = "ignore"
    """Do nothing if the target already exists."""


@dataclasses.dataclass
class WriteOptions:
    """Common options for all writers."""

    mode: WriteMode = WriteMode.ERROR
    """Write mode."""

    schema_evolution: bool = False
    """Whether to allow schema evolution."""

    column_mapping: Optional[Dict[str, str]] = None
    """Mapping from source column names to target column names."""

    partition_by: Optional[List[str]] = None
    """List of columns to partition the data by."""

    coerce_to_schema: bool = False
    """Whether to coerce data types to match the target schema."""


class DataWriter(Generic[ConfigT], abc.ABC):
    """Abstract base class for data writers.

    This class provides a common interface for writing Polars DataFrames to various
    destinations with support for schema evolution and type coercion.
    """

    @abc.abstractmethod
    def from_polars(
        self, df: pl.DataFrame, options: Optional[WriteOptions] = None
    ) -> Dict[str, Any]:
        """Write a Polars DataFrame to the destination.

        Args:
            df: The DataFrame to write.
            options: Options controlling how to write the data.

        Returns:
            Dict[str, Any]: Dictionary with metadata about the operation.
        """
        pass

    @abc.abstractmethod
    def get_schema(self) -> Optional[Dict[str, pl.DataType]]:
        """Get the schema of the destination if it exists.

        Returns:
            Optional[Dict[str, pl.DataType]]: Dictionary mapping column names to Polars data types,
                or None if the destination doesn't exist or schema cannot be determined.
        """
        pass

    @abc.abstractmethod
    def exists(self) -> bool:
        """Check if the destination exists.

        Returns:
            bool: True if the destination exists, False otherwise.
        """
        pass

    @classmethod
    def create(cls, config: ConfigT) -> DataWriter:
        """Factory method to create a writer from a configuration.

        Args:
            config: Configuration for the writer.

        Returns:
            DataWriter: A writer instance.
        """
        return cls(config)

    def _coerce_dataframe_to_schema(
        self, df: pl.DataFrame, schema: Dict[str, pl.DataType]
    ) -> pl.DataFrame:
        """Coerce DataFrame to match the schema.

        Args:
            df: DataFrame to coerce.
            schema: Target schema as a dictionary mapping column names to data types.

        Returns:
            pl.DataFrame: Coerced DataFrame.
        """
        logger.debug(f"Coercing DataFrame with {len(df)} rows to match schema with {len(schema)} columns")
        result = df

        # Process column mappings and type conversions
        for col_name, target_dtype in schema.items():
            if col_name not in result.columns:
                # Skip columns in the schema that aren't in the DataFrame
                logger.debug(f"Column {col_name} in schema but not in DataFrame, skipping")
                continue

            # Get the source data type
            source_dtype = result.schema[col_name]

            if source_dtype == target_dtype:
                # No conversion needed
                logger.debug(f"Column {col_name} already has correct type {target_dtype}, skipping")
                continue

            logger.debug(f"Converting column {col_name} from {source_dtype} to {target_dtype}")
            # Get or build a caster
            try:
                caster = DATA_CAST_REGISTRY.get_or_build(
                    source_dtype=source_dtype,
                    target_dtype=target_dtype,
                    source_name=col_name,
                    target_name=col_name
                )

                # Apply the cast
                result = result.with_columns(
                    caster.cast_series(result.get_column(col_name))
                )
                logger.debug(f"Successfully converted column {col_name}")

            except ValueError as e:
                # If we can't safely cast, we'll leave the column as-is and log a warning
                logger.warning(f"Failed to convert column {col_name} from {source_dtype} to {target_dtype}: {e}")
                pass

        logger.info(f"Finished coercing DataFrame to schema, resulted in {len(result)} rows")
        return result