"""Delta Lake integration for reading and writing Polars DataFrames."""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Any, Dict, List, Optional, Union, ClassVar, Literal

# Make Polars the primary dependency
import polars as pl

# Conditionally import delta and pyarrow for interoperability
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import deltalake
    from deltalake import DeltaTable, write_deltalake
    HAS_DELTA = True
except ImportError:
    HAS_DELTA = False
    pa = Any  # type: ignore
    pc = Any  # type: ignore
    deltalake = Any  # type: ignore
    DeltaTable = Any  # type: ignore
    write_deltalake = Any  # type: ignore


@dataclasses.dataclass
class DeltaReader:
    """Reader for Delta tables that loads them into Polars DataFrames.

    This class provides functionality to read Delta tables into Polars DataFrames,
    supporting various options like filters, partition selection, and versioning.
    """

    table_path: Union[str, pathlib.Path]
    """Path to the Delta table."""

    version: Optional[int] = None
    """Specific version of the Delta table to read. If None, reads the latest version."""

    filter_column: Optional[str] = None
    """Column name to filter on."""

    filter_value: Optional[Any] = None
    """Value to filter by."""

    columns: Optional[List[str]] = None
    """List of columns to select. If None, selects all columns."""

    limit: Optional[int] = None
    """Maximum number of rows to read."""

    _delta_table: Optional[DeltaTable] = dataclasses.field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate initialization parameters and check dependencies."""
        if not HAS_DELTA:
            raise ImportError(
                "The 'deltalake' and 'pyarrow' packages are required for Delta Lake support. "
                "Install them with 'pip install yggdrasil[delta]'"
            )

    @property
    def delta_table(self) -> DeltaTable:
        """Get or load the Delta table."""
        if self._delta_table is None:
            load_args = {}
            if self.version is not None:
                load_args["version"] = self.version

            self._delta_table = DeltaTable(self.table_path, **load_args)
        return self._delta_table

    def read_to_polars(self) -> pl.DataFrame:
        """Read the Delta table into a Polars DataFrame.

        Returns:
            pl.DataFrame: The loaded data as a Polars DataFrame.
        """
        # Prepare scan arguments
        scan_args = {}

        # Add columns if specified
        if self.columns:
            scan_args["columns"] = self.columns

        # Create a scan object
        arrow_scanner = self.delta_table.to_pyarrow_scanner(**scan_args)

        # Apply filtering if specified
        if self.filter_column and self.filter_value is not None:
            # Create a filter expression
            filter_expr = pc.field(self.filter_column) == pc.scalar(self.filter_value)
            arrow_scanner = arrow_scanner.filter(filter_expr)

        # Apply limit if specified
        if self.limit:
            arrow_scanner = arrow_scanner.head(self.limit)

        # Convert to Arrow table and then to Polars
        arrow_table = arrow_scanner.to_table()
        return pl.from_arrow(arrow_table)

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the Delta table.

        Returns:
            Dict[str, Any]: Dictionary with table metadata.
        """
        metadata = {
            "num_files": len(self.delta_table.files()),
            "table_uri": str(self.table_path),
            "version": self.delta_table.version(),
        }

        # Add schema information
        schema = self.delta_table.schema()
        metadata["schema"] = {
            "fields": [{
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable
            } for field in schema.fields]
        }

        # Add partition information if available
        try:
            metadata["partition_columns"] = self.delta_table.metadata().partition_columns
        except Exception:
            metadata["partition_columns"] = []

        return metadata


@dataclasses.dataclass
class DeltaWriter:
    """Writer for saving Polars DataFrames to Delta tables.

    This class provides functionality to write Polars DataFrames to Delta tables,
    supporting various options like partitioning, mode selection, and schema evolution.
    """

    table_path: Union[str, pathlib.Path]
    """Path to the Delta table."""

    mode: Literal["error", "append", "overwrite", "ignore"] = "error"
    """Write mode.

    - "error": Raise exception if table exists (default)
    - "append": Append to table if exists
    - "overwrite": Replace table if exists
    - "ignore": Do nothing if table exists
    """

    partition_by: Optional[List[str]] = None
    """List of columns to partition the table by."""

    schema_evolution: bool = False
    """Whether to allow schema evolution."""

    overwrite_schema: bool = False
    """Whether to overwrite the schema when using overwrite mode."""

    def __post_init__(self) -> None:
        """Validate initialization parameters and check dependencies."""
        if not HAS_DELTA:
            raise ImportError(
                "The 'deltalake' and 'pyarrow' packages are required for Delta Lake support. "
                "Install them with 'pip install yggdrasil[delta]'"
            )

        valid_modes = ["error", "append", "overwrite", "ignore"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {valid_modes}")

    def table_exists(self) -> bool:
        """Check if the table exists.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        try:
            # Try to open the table
            DeltaTable(self.table_path)
            return True
        except Exception:
            return False

    def write_from_polars(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Write a Polars DataFrame to a Delta table.

        Args:
            df: Polars DataFrame to write.

        Returns:
            Dict[str, Any]: Dictionary with metadata about the write operation.
        """
        # Check if table exists and handle according to mode
        table_exists = self.table_exists()

        if table_exists and self.mode == "error":
            raise ValueError(f"Table already exists at {self.table_path} and mode is 'error'")

        if table_exists and self.mode == "ignore":
            return {"status": "skipped", "reason": "Table exists and mode is 'ignore'"}

        # Determine if this is a replace operation
        is_replace = table_exists and self.mode == "overwrite"

        # Convert to pyarrow table
        arrow_table = df.to_arrow()

        # Prepare write options
        write_options = {}

        if self.partition_by:
            write_options["partition_by"] = self.partition_by

        if self.schema_evolution:
            write_options["schema_mode"] = "merge"

        if is_replace and self.overwrite_schema:
            write_options["mode"] = "overwrite"
            # Force schema overwrite in replace mode if requested
            if "schema_mode" not in write_options:
                write_options["schema_mode"] = "overwrite"
        elif is_replace:
            write_options["mode"] = "overwrite"
        elif table_exists:  # append mode
            write_options["mode"] = "append"

        # Write the table
        write_deltalake(self.table_path, arrow_table, **write_options)

        # Return metadata about the operation
        return {
            "status": "success",
            "operation": write_options.get("mode", "create"),
            "schema_mode": write_options.get("schema_mode", "none"),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "partition_columns": self.partition_by or []
        }