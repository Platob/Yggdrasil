"""Delta Lake writer implementation for Yggdrasil."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Dict, List, Optional, Union

# Make Polars the primary dependency
import polars as pl

# Import abstract writer class
from .base import DataWriter, WriteOptions, WriteMode

# Import logging
from ...logging import get_logger

# Create module-level logger
logger = get_logger(__name__)

# Conditionally import delta and pyarrow for interoperability
try:
    import pyarrow as pa
    import deltalake
    from deltalake import DeltaTable, write_deltalake
    HAS_DELTA = True
    logger.info("Delta Lake is available, enabling Delta table writer support")
except ImportError:
    HAS_DELTA = False
    logger.warning("Delta Lake not available, Delta table writer support disabled")
    pa = Any  # type: ignore
    deltalake = Any  # type: ignore
    DeltaTable = Any  # type: ignore
    write_deltalake = Any  # type: ignore


@dataclasses.dataclass
class DeltaWriterConfig:
    """Configuration for Delta table writers."""

    table_path: Union[str, pathlib.Path]
    """Path to the Delta table."""

    overwrite_schema: bool = False
    """Whether to overwrite the schema when using overwrite mode."""


class DeltaWriter(DataWriter[DeltaWriterConfig]):
    """Writer for saving Polars DataFrames to Delta tables.

    This class provides functionality to write Polars DataFrames to Delta tables,
    supporting various options like partitioning, mode selection, and schema evolution.
    """

    def __init__(self, config: DeltaWriterConfig):
        """Initialize the Delta writer with a configuration.

        Args:
            config: Configuration for the writer.
        """
        self.config = config
        logger.debug(f"Initializing DeltaWriter for table at {config.table_path}")

        if not HAS_DELTA:
            error_msg = (
                "The 'deltalake' and 'pyarrow' packages are required for Delta Lake support. "
                "Install them with 'pip install yggdrasil[delta]'"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

    def from_polars(
        self, df: pl.DataFrame, options: Optional[WriteOptions] = None
    ) -> Dict[str, Any]:
        """Write a Polars DataFrame to a Delta table.

        Args:
            df: The DataFrame to write.
            options: Options controlling how to write the data.

        Returns:
            Dict[str, Any]: Dictionary with metadata about the operation.
        """
        # Use default options if none provided
        if options is None:
            options = WriteOptions()

        logger.info(f"Writing DataFrame with {len(df)} rows, {len(df.columns)} columns to Delta table at {self.config.table_path}")
        logger.debug(f"Write options: {options}")

        # Check if table exists and handle according to mode
        table_exists = self.exists()
        logger.debug(f"Table exists: {table_exists}")

        if table_exists and options.mode == WriteMode.ERROR:
            error_msg = f"Table already exists at {self.config.table_path} and mode is 'error'"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if table_exists and options.mode == WriteMode.IGNORE:
            logger.info(f"Skipping write operation: Table exists at {self.config.table_path} and mode is 'ignore'")
            return {"status": "skipped", "reason": "Table exists and mode is 'ignore'"}

        # Apply column mapping if provided
        if options.column_mapping:
            logger.debug(f"Applying column mapping: {options.column_mapping}")
            df = df.rename(options.column_mapping)

        # Apply type coercion if requested
        if options.coerce_to_schema and table_exists:
            logger.debug("Coercing DataFrame to match table schema")
            schema = self.get_schema()
            if schema:
                df = self._coerce_dataframe_to_schema(df, schema)
            else:
                logger.warning("Failed to get schema for coercion")

        # Determine if this is a replace operation
        is_replace = table_exists and options.mode == WriteMode.OVERWRITE

        # Convert to pyarrow table
        logger.debug("Converting DataFrame to PyArrow table")
        arrow_table = df.to_arrow()

        # Prepare write options
        write_options = {}

        if options.partition_by:
            write_options["partition_by"] = options.partition_by
            logger.debug(f"Partitioning by columns: {options.partition_by}")

        if options.schema_evolution:
            write_options["schema_mode"] = "merge"
            logger.debug("Enabling schema evolution")

        if is_replace and self.config.overwrite_schema:
            write_options["mode"] = "overwrite"
            # Force schema overwrite in replace mode if requested
            if "schema_mode" not in write_options:
                write_options["schema_mode"] = "overwrite"
                logger.debug("Using overwrite mode with schema overwrite")
        elif is_replace:
            write_options["mode"] = "overwrite"
            logger.debug("Using overwrite mode without schema overwrite")
        elif table_exists:  # append mode
            write_options["mode"] = "append"
            logger.debug("Using append mode")
        else:
            logger.debug("Creating new table")

        # Write the table
        logger.debug(f"Writing PyArrow table with options: {write_options}")
        write_deltalake(self.config.table_path, arrow_table, **write_options)
        logger.info(f"Successfully wrote data to Delta table at {self.config.table_path}")

        # Return metadata about the operation
        result = {
            "status": "success",
            "operation": write_options.get("mode", "create"),
            "schema_mode": write_options.get("schema_mode", "none"),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "partition_columns": options.partition_by or []
        }
        logger.debug(f"Write operation completed: {result}")
        return result

    def exists(self) -> bool:
        """Check if the table exists.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        logger.debug(f"Checking if Delta table exists at {self.config.table_path}")
        try:
            # Try to open the table
            DeltaTable(self.config.table_path)
            logger.debug(f"Delta table found at {self.config.table_path}")
            return True
        except Exception as e:
            logger.debug(f"Delta table not found at {self.config.table_path}: {e}")
            return False

    def get_schema(self) -> Optional[Dict[str, pl.DataType]]:
        """Get the schema of the destination if it exists.

        Returns:
            Optional[Dict[str, pl.DataType]]: Dictionary mapping column names to Polars data types,
                or None if the destination doesn't exist or schema cannot be determined.
        """
        logger.debug(f"Getting schema for Delta table at {self.config.table_path}")

        if not self.exists():
            logger.debug("Table does not exist, cannot get schema")
            return None

        try:
            # Open the table and get its schema
            table = DeltaTable(self.config.table_path)
            arrow_schema = table.schema()
            logger.debug(f"Retrieved Arrow schema with {len(arrow_schema)} fields")

            # Convert Arrow schema to Polars schema
            schema_dict = {}
            for field in arrow_schema:
                try:
                    # Try to convert Arrow type to Polars type
                    polars_type = self._arrow_to_polars_type(field.type)
                    if polars_type:
                        schema_dict[field.name] = polars_type
                        logger.debug(f"Mapped field '{field.name}' from Arrow type {field.type} to Polars type {polars_type}")
                    else:
                        logger.warning(f"No Polars type mapping for Arrow type {field.type} in field '{field.name}'")
                except Exception as e:
                    # If conversion fails, skip this field
                    logger.warning(f"Failed to convert type for field '{field.name}': {e}")
                    pass

            logger.info(f"Converted Delta table schema with {len(schema_dict)} fields")
            return schema_dict
        except Exception as e:
            logger.error(f"Failed to get schema from Delta table: {e}")
            return None

    def _arrow_to_polars_type(self, arrow_type: pa.DataType) -> Optional[pl.DataType]:
        """Convert an Arrow data type to a Polars data type.

        Args:
            arrow_type: Arrow data type.

        Returns:
            Optional[pl.DataType]: Corresponding Polars data type, or None if no match.
        """
        logger.debug(f"Converting Arrow type {arrow_type} to Polars type")

        # This is a simplified mapping and may need expansion for more complex types
        if pa.types.is_boolean(arrow_type):
            logger.debug(f"Mapped Arrow boolean to Polars Boolean")
            return pl.Boolean
        elif pa.types.is_integer(arrow_type):
            if arrow_type.bit_width <= 8:
                result = pl.Int8 if arrow_type.is_signed else pl.UInt8
                logger.debug(f"Mapped Arrow integer (width: {arrow_type.bit_width}, signed: {arrow_type.is_signed}) to {result}")
                return result
            elif arrow_type.bit_width <= 16:
                result = pl.Int16 if arrow_type.is_signed else pl.UInt16
                logger.debug(f"Mapped Arrow integer (width: {arrow_type.bit_width}, signed: {arrow_type.is_signed}) to {result}")
                return result
            elif arrow_type.bit_width <= 32:
                result = pl.Int32 if arrow_type.is_signed else pl.UInt32
                logger.debug(f"Mapped Arrow integer (width: {arrow_type.bit_width}, signed: {arrow_type.is_signed}) to {result}")
                return result
            else:
                result = pl.Int64 if arrow_type.is_signed else pl.UInt64
                logger.debug(f"Mapped Arrow integer (width: {arrow_type.bit_width}, signed: {arrow_type.is_signed}) to {result}")
                return result
        elif pa.types.is_floating(arrow_type):
            result = pl.Float32 if arrow_type.bit_width == 32 else pl.Float64
            logger.debug(f"Mapped Arrow float (width: {arrow_type.bit_width}) to {result}")
            return result
        elif pa.types.is_string(arrow_type):
            logger.debug("Mapped Arrow string to Polars Utf8")
            return pl.Utf8
        elif pa.types.is_binary(arrow_type):
            logger.debug("Mapped Arrow binary to Polars Binary")
            return pl.Binary
        elif pa.types.is_date(arrow_type):
            logger.debug("Mapped Arrow date to Polars Date")
            return pl.Date
        elif pa.types.is_timestamp(arrow_type):
            logger.debug("Mapped Arrow timestamp to Polars Datetime")
            return pl.Datetime
        elif pa.types.is_list(arrow_type):
            logger.debug(f"Processing Arrow List type with value_type: {arrow_type.value_type}")
            inner_type = self._arrow_to_polars_type(arrow_type.value_type)
            if inner_type:
                result = pl.List(inner_type)
                logger.debug(f"Mapped Arrow List({arrow_type.value_type}) to Polars List({inner_type})")
                return result
            else:
                logger.warning(f"Could not map inner type {arrow_type.value_type} of Arrow List")

        # No mapping found
        logger.warning(f"No Polars type mapping found for Arrow type {arrow_type}")
        return None