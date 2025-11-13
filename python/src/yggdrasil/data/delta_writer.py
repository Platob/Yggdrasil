"""Delta Lake writer implementation for Yggdrasil."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Dict, List, Optional, Union

# Make Polars the primary dependency
import polars as pl

# Import abstract writer class
from .writer import DataWriter, WriteOptions, WriteMode

# Conditionally import delta and pyarrow for interoperability
try:
    import pyarrow as pa
    import deltalake
    from deltalake import DeltaTable, write_deltalake
    HAS_DELTA = True
except ImportError:
    HAS_DELTA = False
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

        if not HAS_DELTA:
            raise ImportError(
                "The 'deltalake' and 'pyarrow' packages are required for Delta Lake support. "
                "Install them with 'pip install yggdrasil[delta]'"
            )

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

        # Check if table exists and handle according to mode
        table_exists = self.exists()

        if table_exists and options.mode == WriteMode.ERROR:
            raise ValueError(f"Table already exists at {self.config.table_path} and mode is 'error'")

        if table_exists and options.mode == WriteMode.IGNORE:
            return {"status": "skipped", "reason": "Table exists and mode is 'ignore'"}

        # Apply column mapping if provided
        if options.column_mapping:
            df = df.rename(options.column_mapping)

        # Apply type coercion if requested
        if options.coerce_to_schema and table_exists:
            schema = self.get_schema()
            if schema:
                df = self._coerce_dataframe_to_schema(df, schema)

        # Determine if this is a replace operation
        is_replace = table_exists and options.mode == WriteMode.OVERWRITE

        # Convert to pyarrow table
        arrow_table = df.to_arrow()

        # Prepare write options
        write_options = {}

        if options.partition_by:
            write_options["partition_by"] = options.partition_by

        if options.schema_evolution:
            write_options["schema_mode"] = "merge"

        if is_replace and self.config.overwrite_schema:
            write_options["mode"] = "overwrite"
            # Force schema overwrite in replace mode if requested
            if "schema_mode" not in write_options:
                write_options["schema_mode"] = "overwrite"
        elif is_replace:
            write_options["mode"] = "overwrite"
        elif table_exists:  # append mode
            write_options["mode"] = "append"

        # Write the table
        write_deltalake(self.config.table_path, arrow_table, **write_options)

        # Return metadata about the operation
        return {
            "status": "success",
            "operation": write_options.get("mode", "create"),
            "schema_mode": write_options.get("schema_mode", "none"),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "partition_columns": options.partition_by or []
        }

    def exists(self) -> bool:
        """Check if the table exists.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        try:
            # Try to open the table
            DeltaTable(self.config.table_path)
            return True
        except Exception:
            return False

    def get_schema(self) -> Optional[Dict[str, pl.DataType]]:
        """Get the schema of the destination if it exists.

        Returns:
            Optional[Dict[str, pl.DataType]]: Dictionary mapping column names to Polars data types,
                or None if the destination doesn't exist or schema cannot be determined.
        """
        if not self.exists():
            return None

        try:
            # Open the table and get its schema
            table = DeltaTable(self.config.table_path)
            arrow_schema = table.schema()

            # Convert Arrow schema to Polars schema
            schema_dict = {}
            for field in arrow_schema:
                try:
                    # Try to convert Arrow type to Polars type
                    # This is a simplified conversion and may need expansion for more complex types
                    polars_type = self._arrow_to_polars_type(field.type)
                    if polars_type:
                        schema_dict[field.name] = polars_type
                except Exception:
                    # If conversion fails, skip this field
                    pass

            return schema_dict
        except Exception:
            return None

    def _arrow_to_polars_type(self, arrow_type: pa.DataType) -> Optional[pl.DataType]:
        """Convert an Arrow data type to a Polars data type.

        Args:
            arrow_type: Arrow data type.

        Returns:
            Optional[pl.DataType]: Corresponding Polars data type, or None if no match.
        """
        # This is a simplified mapping and may need expansion for more complex types
        if pa.types.is_boolean(arrow_type):
            return pl.Boolean
        elif pa.types.is_integer(arrow_type):
            if arrow_type.bit_width <= 8:
                return pl.Int8 if arrow_type.is_signed else pl.UInt8
            elif arrow_type.bit_width <= 16:
                return pl.Int16 if arrow_type.is_signed else pl.UInt16
            elif arrow_type.bit_width <= 32:
                return pl.Int32 if arrow_type.is_signed else pl.UInt32
            else:
                return pl.Int64 if arrow_type.is_signed else pl.UInt64
        elif pa.types.is_floating(arrow_type):
            return pl.Float32 if arrow_type.bit_width == 32 else pl.Float64
        elif pa.types.is_string(arrow_type):
            return pl.Utf8
        elif pa.types.is_binary(arrow_type):
            return pl.Binary
        elif pa.types.is_date(arrow_type):
            return pl.Date
        elif pa.types.is_timestamp(arrow_type):
            return pl.Datetime
        elif pa.types.is_list(arrow_type):
            inner_type = self._arrow_to_polars_type(arrow_type.value_type)
            if inner_type:
                return pl.List(inner_type)
        # Add more type conversions as needed
        return None