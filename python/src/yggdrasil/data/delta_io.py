"""Delta Lake integration for reading and writing Polars DataFrames."""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Any, Dict, List, Optional, Union, ClassVar, Type, get_type_hints

# Make Polars the primary dependency
import polars as pl

# Import abstract base classes
from .reader import DataReader, ReadOptions, ReaderPredicate, ColumnPredicate
from .writer import DataWriter, WriteOptions, WriteMode

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
class DeltaReaderConfig:
    """Configuration for Delta table readers."""

    table_path: Union[str, pathlib.Path]
    """Path to the Delta table."""

    version: Optional[int] = None
    """Specific version of the Delta table to read. If None, reads the latest version."""


@dataclasses.dataclass
class DeltaWriterConfig:
    """Configuration for Delta table writers."""

    table_path: Union[str, pathlib.Path]
    """Path to the Delta table."""

    overwrite_schema: bool = False
    """Whether to overwrite the schema when using overwrite mode."""


class DeltaReader(DataReader[DeltaReaderConfig]):
    """Reader for Delta tables that loads them into Polars DataFrames.

    This class provides functionality to read Delta tables into Polars DataFrames,
    supporting various options like filters, partition selection, and versioning.
    """

    def __init__(self, config: DeltaReaderConfig):
        """Initialize the Delta reader with a configuration.

        Args:
            config: Configuration for the reader.
        """
        self.config = config
        self._delta_table: Optional[DeltaTable] = None

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
            if self.config.version is not None:
                load_args["version"] = self.config.version

            self._delta_table = DeltaTable(self.config.table_path, **load_args)
        return self._delta_table

    def to_polars(self, options: Optional[ReadOptions] = None) -> pl.DataFrame:
        """Read the Delta table into a Polars DataFrame.

        Args:
            options: Options controlling which data to read.

        Returns:
            pl.DataFrame: The loaded data as a Polars DataFrame.
        """
        # Use default options if none provided
        if options is None:
            options = ReadOptions()

        # Prepare scan arguments
        scan_args = {}

        # Add columns if specified
        if options.columns:
            scan_args["columns"] = options.columns

        # Create a scan object
        arrow_scanner = self.delta_table.to_pyarrow_scanner(**scan_args)

        # Apply filtering if specified
        if options.predicate:
            filter_expr = self._convert_predicate_to_arrow(options.predicate)
            if filter_expr is not None:
                arrow_scanner = arrow_scanner.filter(filter_expr)

        # Apply limit if specified
        if options.limit:
            arrow_scanner = arrow_scanner.head(options.limit)

        # Convert to Arrow table and then to Polars
        arrow_table = arrow_scanner.to_table()
        return pl.from_arrow(arrow_table)

    def _convert_predicate_to_arrow(self, predicate: ReaderPredicate) -> Optional[pc.Expression]:
        """Convert a ReaderPredicate to a PyArrow expression.

        Args:
            predicate: The predicate to convert.

        Returns:
            Optional[pc.Expression]: PyArrow expression, or None if conversion failed.
        """
        if not HAS_DELTA:
            return None

        expr_dict = predicate.to_expression()

        # Handle column predicates
        if expr_dict["type"] == "column_predicate":
            column = expr_dict["column"]
            op = expr_dict["op"]
            value = expr_dict["value"]

            field = pc.field(column)

            if op == "eq":
                return field == pc.scalar(value)
            elif op == "gt":
                return field > pc.scalar(value)
            elif op == "lt":
                return field < pc.scalar(value)
            elif op == "gte":
                return field >= pc.scalar(value)
            elif op == "lte":
                return field <= pc.scalar(value)
            elif op == "ne":
                return field != pc.scalar(value)
            elif op == "in":
                # For IN operator, we need to create a list of OR conditions
                # This is a simplified implementation
                return pc.is_in(field, pa.array(value))
            elif op == "not_in":
                return ~pc.is_in(field, pa.array(value))

        # Handle logical operators
        elif expr_dict["type"] == "and":
            sub_exprs = [
                self._convert_predicate_to_arrow(ReaderPredicate.from_expression(pred))
                for pred in expr_dict["predicates"]
            ]
            # Remove any None values
            sub_exprs = [expr for expr in sub_exprs if expr is not None]
            if not sub_exprs:
                return None

            result = sub_exprs[0]
            for expr in sub_exprs[1:]:
                result = result & expr
            return result

        elif expr_dict["type"] == "or":
            sub_exprs = [
                self._convert_predicate_to_arrow(ReaderPredicate.from_expression(pred))
                for pred in expr_dict["predicates"]
            ]
            # Remove any None values
            sub_exprs = [expr for expr in sub_exprs if expr is not None]
            if not sub_exprs:
                return None

            result = sub_exprs[0]
            for expr in sub_exprs[1:]:
                result = result | expr
            return result

        elif expr_dict["type"] == "not":
            sub_expr = self._convert_predicate_to_arrow(
                ReaderPredicate.from_expression(expr_dict["predicate"])
            )
            if sub_expr is None:
                return None
            return ~sub_expr

        # Default case: unable to convert
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the Delta table.

        Returns:
            Dict[str, Any]: Dictionary with table metadata.
        """
        metadata = {
            "num_files": len(self.delta_table.files()),
            "table_uri": str(self.config.table_path),
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


# Backward compatibility functions
def create_delta_reader(
    table_path: Union[str, pathlib.Path],
    version: Optional[int] = None,
    filter_column: Optional[str] = None,
    filter_value: Optional[Any] = None,
    columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> DeltaReader:
    """Create a DeltaReader with the given parameters.

    This function provides backward compatibility with the previous API.

    Returns:
        DeltaReader: A configured Delta reader.
    """
    config = DeltaReaderConfig(table_path=table_path, version=version)
    reader = DeltaReader(config)

    # Create read options with any provided filters
    options = ReadOptions(columns=columns, limit=limit)
    if filter_column is not None and filter_value is not None:
        options.predicate = ColumnPredicate(filter_column, "eq", filter_value)

    # Attach the options to the reader for backward compatibility
    reader._options = options

    # Add backward compatibility properties
    reader.table_path = table_path
    reader.version = version
    reader.filter_column = filter_column
    reader.filter_value = filter_value
    reader.columns = columns
    reader.limit = limit

    # Add backward compatibility method
    reader.read_to_polars = lambda: reader.to_polars(options)

    return reader


def create_delta_writer(
    table_path: Union[str, pathlib.Path],
    mode: str = "error",
    partition_by: Optional[List[str]] = None,
    schema_evolution: bool = False,
    overwrite_schema: bool = False,
) -> DeltaWriter:
    """Create a DeltaWriter with the given parameters.

    This function provides backward compatibility with the previous API.

    Returns:
        DeltaWriter: A configured Delta writer.
    """
    config = DeltaWriterConfig(table_path=table_path, overwrite_schema=overwrite_schema)
    writer = DeltaWriter(config)

    # Create write options
    options = WriteOptions(
        mode=WriteMode(mode),
        partition_by=partition_by,
        schema_evolution=schema_evolution
    )

    # Attach the options to the writer for backward compatibility
    writer._options = options

    # Add backward compatibility properties
    writer.table_path = table_path
    writer.mode = mode
    writer.partition_by = partition_by
    writer.schema_evolution = schema_evolution
    writer.overwrite_schema = overwrite_schema

    # Add backward compatibility method
    writer.write_from_polars = lambda df: writer.from_polars(df, options)
    writer.table_exists = writer.exists

    return writer