"""Delta Lake reader implementation for Yggdrasil."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Dict, List, Optional, Union

# Make Polars the primary dependency
import polars as pl

# Import abstract reader class
from .base import DataReader, ReadOptions, ReaderPredicate

# Conditionally import delta and pyarrow for interoperability
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import deltalake
    from deltalake import DeltaTable
    HAS_DELTA = True
except ImportError:
    HAS_DELTA = False
    pa = Any  # type: ignore
    pc = Any  # type: ignore
    deltalake = Any  # type: ignore
    DeltaTable = Any  # type: ignore


@dataclasses.dataclass
class DeltaReaderConfig:
    """Configuration for Delta table readers."""

    table_path: Union[str, pathlib.Path]
    """Path to the Delta table."""

    version: Optional[int] = None
    """Specific version of the Delta table to read. If None, reads the latest version."""


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