"""In-memory data reader implementation for Yggdrasil."""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Set

import polars as pl

from .base import DataReader, ReadOptions, FilterPredicate
from ...logging import get_logger

# Create module-level logger
logger = get_logger(__name__)


class MemoryReaderConfig:
    """Configuration for in-memory reader."""
    pass


class MemoryReader(DataReader[MemoryReaderConfig]):
    """In-memory reader for reading Polars DataFrames.

    This reader is primarily intended for testing purposes or
    as a simple in-memory data store for small datasets.
    """

    def __init__(self, config: MemoryReaderConfig, data: Optional[pl.DataFrame] = None):
        """Initialize with optional test data.

        Args:
            config: Configuration for the reader.
            data: Optional initial data to store in the reader.
        """
        self.config = config
        self._data = data if data is not None else pl.DataFrame()
        logger.debug(f"Initialized MemoryReader with {len(self._data)} rows")

    def read_polars(self, options: Optional[ReadOptions] = None) -> pl.DataFrame:
        """Read data into a Polars DataFrame.

        Args:
            options: Options controlling which data to read.

        Returns:
            pl.DataFrame: The filtered data as a Polars DataFrame.
        """
        logger.debug(f"Reading data with options: {options}")
        df = self._data

        # Handle options if provided
        if options:
            # Apply filtering
            if options.predicate:
                logger.debug(f"Applying filter predicate")
                df = self._apply_predicate(df, options.predicate)

            # Apply column selection
            if options.columns:
                logger.debug(f"Selecting columns: {options.columns}")
                df = df.select(options.columns)

            # Apply limit
            if options.limit is not None:
                logger.debug(f"Applying limit: {options.limit}")
                df = df.head(options.limit)

        logger.debug(f"Returning {len(df)} rows")
        return df

    def _apply_predicate(self, df: pl.DataFrame, predicate: FilterPredicate) -> pl.DataFrame:
        """Apply a predicate to filter a DataFrame.

        Args:
            df: DataFrame to filter.
            predicate: Filter predicate to apply.

        Returns:
            pl.DataFrame: Filtered DataFrame.
        """
        logger.debug(f"Applying predicate to DataFrame")
        expr_dict = predicate.to_expression()

        if expr_dict["type"] == "column_predicate":
            col = expr_dict["column"]
            op = expr_dict["op"]
            val = expr_dict["value"]
            logger.debug(f"Applying column predicate: {col} {op} {val}")

            if op == "eq":
                return df.filter(pl.col(col) == val)
            elif op == "gt":
                return df.filter(pl.col(col) > val)
            elif op == "lt":
                return df.filter(pl.col(col) < val)
            elif op == "gte":
                return df.filter(pl.col(col) >= val)
            elif op == "lte":
                return df.filter(pl.col(col) <= val)
            elif op == "ne":
                return df.filter(pl.col(col) != val)
            elif op == "in":
                return df.filter(pl.col(col).is_in(val))
            elif op == "not_in":
                return df.filter(~pl.col(col).is_in(val))
            else:
                logger.warning(f"Unsupported operation: {op}, returning unfiltered DataFrame")
                return df

        elif expr_dict["type"] == "and":
            logger.debug(f"Applying AND predicate with {len(expr_dict['predicates'])} conditions")
            result = df
            for pred_dict in expr_dict["predicates"]:
                pred = FilterPredicate.from_expression(pred_dict)
                result = self._apply_predicate(result, pred)
            return result

        elif expr_dict["type"] == "or":
            logger.debug(f"Applying OR predicate with {len(expr_dict['predicates'])} conditions")
            # For OR, directly build expressions and combine them
            filters = []
            for pred_dict in expr_dict["predicates"]:
                pred = FilterPredicate.from_expression(pred_dict)
                if pred_dict["type"] == "column_predicate":
                    col = pred_dict["column"]
                    op = pred_dict["op"]
                    val = pred_dict["value"]

                    if op == "eq":
                        filters.append(pl.col(col) == val)
                    elif op == "gt":
                        filters.append(pl.col(col) > val)
                    elif op == "lt":
                        filters.append(pl.col(col) < val)
                    elif op == "gte":
                        filters.append(pl.col(col) >= val)
                    elif op == "lte":
                        filters.append(pl.col(col) <= val)
                    elif op == "ne":
                        filters.append(pl.col(col) != val)
                    elif op == "in":
                        filters.append(pl.col(col).is_in(val))
                    elif op == "not_in":
                        filters.append(~pl.col(col).is_in(val))

            # Combine all expressions with OR
            combined_filter = None
            for f in filters:
                if combined_filter is None:
                    combined_filter = f
                else:
                    combined_filter = combined_filter | f

            # Apply the combined filter
            if combined_filter is not None:
                return df.filter(combined_filter)
            return df

        elif expr_dict["type"] == "not":
            logger.debug("Applying NOT predicate")
            pred = FilterPredicate.from_expression(expr_dict["predicate"])
            if expr_dict["predicate"]["type"] == "column_predicate":
                col = expr_dict["predicate"]["column"]
                op = expr_dict["predicate"]["op"]
                val = expr_dict["predicate"]["value"]

                # Negate the condition directly
                if op == "eq":
                    return df.filter(pl.col(col) != val)
                elif op == "gt":
                    return df.filter(pl.col(col) <= val)
                elif op == "lt":
                    return df.filter(pl.col(col) >= val)
                elif op == "gte":
                    return df.filter(pl.col(col) < val)
                elif op == "lte":
                    return df.filter(pl.col(col) > val)
                elif op == "ne":
                    return df.filter(pl.col(col) == val)
                elif op == "in":
                    return df.filter(~pl.col(col).is_in(val))
                elif op == "not_in":
                    return df.filter(pl.col(col).is_in(val))

            # For complex conditions, we can use the full DataFrame approach
            # but avoid using row index methods
            filtered_df = self._apply_predicate(df, pred)
            # Get a unique identifier for each row (for complex cases)
            # Use all columns to identify rows uniquely
            columns_to_join = df.columns
            all_columns_expr = [pl.col(col) for col in columns_to_join]

            # Create unique identifiers for rows in filtered and original dataframes
            filtered_keys = filtered_df.select(all_columns_expr).hash_rows()
            all_keys = df.select(all_columns_expr).hash_rows()

            # Find rows in original dataframe that are not in filtered dataframe
            filtered_keys_set = set(filtered_keys.to_list())
            result = df.with_columns(pl.lit(all_keys).alias("__tmp_hash"))
            result = result.filter(~pl.col("__tmp_hash").is_in(filtered_keys_set))
            return result.drop("__tmp_hash")

        logger.warning(f"Unsupported predicate type: {expr_dict['type']}, returning unfiltered DataFrame")
        return df

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data source.

        Returns:
            Dict[str, Any]: Dictionary with metadata.
        """
        logger.debug("Getting metadata")
        return {
            "num_rows": len(self._data),
            "num_columns": len(self._data.columns),
            "columns": self._data.columns,
            "schema": {col: dtype for col, dtype in zip(self._data.columns, self._data.dtypes)}
        }