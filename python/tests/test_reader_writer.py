from __future__ import annotations

import pathlib
import sys
from typing import Dict, Any, Optional, List

import polars as pl
import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from yggdrasil.data import (
    DataReader,
    ReadOptions,
    FilterPredicate,
    ColumnPredicate,
    AndPredicate,
    OrPredicate,
    NotPredicate,
    DataWriter,
    WriteOptions,
    WriteMode,
    # Predicate helper functions
    eq, gt, lt, gte, lte, ne, is_in, not_in,
    and_, or_, not_
)
from yggdrasil.data.reader.base import ConfigT


# Test implementation of DataReader for testing
class MemoryReaderConfig:
    """Configuration for in-memory test reader."""
    pass


class MemoryReader(DataReader[MemoryReaderConfig]):
    """In-memory reader for testing."""

    def __init__(self, config: MemoryReaderConfig, data: Optional[pl.DataFrame] = None):
        """Initialize with test data."""
        self.config = config
        self._data = data if data is not None else pl.DataFrame()

    def read_polars(self, options: Optional[ReadOptions] = None) -> pl.DataFrame:
        """Read data into a Polars DataFrame."""
        df = self._data

        # Handle options if provided
        if options:
            # Apply column selection
            if options.columns:
                df = df.select(options.columns)

            # Apply filtering
            if options.predicate:
                df = self._apply_predicate(df, options.predicate)

            # Apply limit
            if options.limit is not None:
                df = df.head(options.limit)

        return df

    def _apply_predicate(self, df: pl.DataFrame, predicate: FilterPredicate) -> pl.DataFrame:
        """Apply a predicate to filter a DataFrame."""
        expr_dict = predicate.to_expression()

        if expr_dict["type"] == "column_predicate":
            col = expr_dict["column"]
            op = expr_dict["op"]
            val = expr_dict["value"]

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

        elif expr_dict["type"] == "and":
            result = df
            for pred_dict in expr_dict["predicates"]:
                pred = FilterPredicate.from_expression(pred_dict)
                result = self._apply_predicate(result, pred)
            return result

        elif expr_dict["type"] == "or":
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

        return df

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data source."""
        return {
            "num_rows": len(self._data),
            "num_columns": len(self._data.columns),
            "columns": self._data.columns,
            "schema": {col: dtype for col, dtype in zip(self._data.columns, self._data.dtypes)}
        }


# Test implementation of DataWriter for testing
class MemoryWriterConfig:
    """Configuration for in-memory test writer."""
    pass


class MemoryWriter(DataWriter[MemoryWriterConfig]):
    """In-memory writer for testing."""

    def __init__(self, config: MemoryWriterConfig):
        """Initialize the writer."""
        self.config = config
        self._data: Optional[pl.DataFrame] = None
        self._write_count = 0
        self._schema = None

    def write_polars(
        self, df: pl.DataFrame, options: Optional[WriteOptions] = None
    ) -> Dict[str, Any]:
        """Write a Polars DataFrame to memory."""
        # Use default options if none provided
        if options is None:
            options = WriteOptions()

        if self.exists() and options.mode == WriteMode.ERROR:
            raise ValueError("Data already exists and mode is 'error'")

        if self.exists() and options.mode == WriteMode.IGNORE:
            return {"status": "skipped", "reason": "Data exists and mode is 'ignore'"}

        # Apply column mapping if provided
        if options.column_mapping:
            df = df.rename(options.column_mapping)

        # Apply type coercion if requested
        if options.coerce_to_schema and self.exists() and self._schema:
            df = self._coerce_dataframe_to_schema(df, self._schema)

        # Handle different write modes
        if self.exists() and options.mode == WriteMode.APPEND:
            # Append to existing data
            self._data = pl.concat([self._data, df], how="diagonal_relaxed")
        else:  # Overwrite or first write
            # Set the data
            self._data = df
            # Store the schema
            self._schema = {col: dtype for col, dtype in zip(df.columns, df.dtypes)}

        self._write_count += 1

        return {
            "status": "success",
            "operation": options.mode.value,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "total_rows": len(self._data),
            "write_count": self._write_count
        }

    def exists(self) -> bool:
        """Check if data exists."""
        return self._data is not None

    def get_schema(self) -> Optional[Dict[str, pl.DataType]]:
        """Get the schema of the data if it exists."""
        if not self.exists():
            return None
        return self._schema

    def get_data(self) -> Optional[pl.DataFrame]:
        """Get the stored data (test helper method)."""
        return self._data


def test_reader_predicate_expression():
    """Test converting predicates to and from expressions."""
    # Create a column predicate
    pred1 = eq("age", 30)
    expr1 = pred1.to_expression()

    assert expr1["type"] == "column_predicate"
    assert expr1["column"] == "age"
    assert expr1["op"] == "eq"
    assert expr1["value"] == 30

    # Convert back to predicate
    pred1_recovered = FilterPredicate.from_expression(expr1)
    assert isinstance(pred1_recovered, ColumnPredicate)
    assert pred1_recovered.column == "age"
    assert pred1_recovered.op == "eq"
    assert pred1_recovered.value == 30

    # Test AND predicate
    pred2 = gt("salary", 50000)
    and_pred = and_(pred1, pred2)
    and_expr = and_pred.to_expression()

    assert and_expr["type"] == "and"
    assert len(and_expr["predicates"]) == 2

    # Test OR predicate
    or_pred = or_(pred1, pred2)
    or_expr = or_pred.to_expression()

    assert or_expr["type"] == "or"
    assert len(or_expr["predicates"]) == 2

    # Test NOT predicate
    not_pred = not_(pred1)
    not_expr = not_pred.to_expression()

    assert not_expr["type"] == "not"
    assert not_expr["predicate"]["type"] == "column_predicate"


def test_memory_reader():
    """Test the memory reader implementation."""
    # Create test data
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "salary": [50000, 60000, 70000, 80000, 90000],
    })

    # Create a reader with the test data
    reader = MemoryReader(MemoryReaderConfig(), df)

    # Test reading without options
    result = reader.read_polars()
    assert len(result) == 5
    assert result.columns == ["id", "name", "age", "salary"]

    # Test reading with column selection
    options = ReadOptions(columns=["id", "name"])
    result = reader.read_polars(options)
    assert len(result) == 5
    assert result.columns == ["id", "name"]

    # Test reading with limit
    options = ReadOptions(limit=2)
    result = reader.read_polars(options)
    assert len(result) == 2

    # Test reading with filtering (equals)
    options = ReadOptions(predicate=eq("name", "Bob"))
    result = reader.read_polars(options)
    assert len(result) == 1
    assert result["name"][0] == "Bob"

    # Test reading with filtering (greater than)
    options = ReadOptions(predicate=gt("age", 35))
    result = reader.read_polars(options)
    assert len(result) == 2
    assert all(age > 35 for age in result["age"])

    # Test reading with filtering (AND)
    options = ReadOptions(predicate=and_(gt("age", 30), lt("salary", 80000)))
    result = reader.read_polars(options)
    assert len(result) == 1
    assert result["name"][0] == "Charlie"

    # Test reading with filtering (OR)
    options = ReadOptions(predicate=or_(eq("name", "Alice"), eq("name", "Eve")))
    result = reader.read_polars(options)
    assert len(result) == 2
    assert set(result["name"].to_list()) == {"Alice", "Eve"}

    # Test reading with filtering (NOT)
    options = ReadOptions(predicate=not_(eq("name", "Bob")))
    result = reader.read_polars(options)
    assert len(result) == 4
    assert "Bob" not in result["name"].to_list()

    # Test reading with filtering (IN)
    options = ReadOptions(predicate=is_in("name", ["Alice", "Bob"]))
    result = reader.read_polars(options)
    assert len(result) == 2
    assert set(result["name"].to_list()) == {"Alice", "Bob"}

    # Test reading with combined options
    options = ReadOptions(
        columns=["id", "name", "age"],
        limit=2,
        predicate=gt("age", 30)
    )
    result = reader.read_polars(options)
    assert len(result) == 2
    assert result.columns == ["id", "name", "age"]
    assert all(age > 30 for age in result["age"])


def test_memory_writer():
    """Test the memory writer implementation."""
    # Create test data
    df1 = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
    })

    df2 = pl.DataFrame({
        "id": [4, 5],
        "name": ["Dave", "Eve"],
        "age": [40, 45],
    })

    # Create a writer
    writer = MemoryWriter(MemoryWriterConfig())

    # Test initial state
    assert not writer.exists()
    assert writer.get_schema() is None

    # Test writing data
    result = writer.write_polars(df1)
    assert result["status"] == "success"
    assert result["num_rows"] == 3
    assert writer.exists()

    # Test schema
    schema = writer.get_schema()
    assert schema is not None
    assert "id" in schema
    assert "name" in schema
    assert "age" in schema

    # Test error mode
    with pytest.raises(ValueError):
        writer.write_polars(df2, WriteOptions(mode=WriteMode.ERROR))

    # Test ignore mode
    result = writer.write_polars(df2, WriteOptions(mode=WriteMode.IGNORE))
    assert result["status"] == "skipped"
    assert writer.get_data().shape == (3, 3)  # Still the original data

    # Test append mode
    result = writer.write_polars(df2, WriteOptions(mode=WriteMode.APPEND))
    assert result["status"] == "success"
    assert result["operation"] == "append"
    assert writer.get_data().shape == (5, 3)  # Combined data

    # Test overwrite mode
    result = writer.write_polars(df2, WriteOptions(mode=WriteMode.OVERWRITE))
    assert result["status"] == "success"
    assert result["operation"] == "overwrite"
    assert writer.get_data().shape == (2, 3)  # Only the new data

    # Test column mapping
    df3 = pl.DataFrame({
        "user_id": [6, 7],
        "user_name": ["Frank", "Grace"],
        "user_age": [50, 55],
    })

    result = writer.write_polars(
        df3,
        WriteOptions(
            mode=WriteMode.APPEND,
            column_mapping={"user_id": "id", "user_name": "name", "user_age": "age"}
        )
    )
    assert result["status"] == "success"
    assert writer.get_data().shape == (4, 3)  # Combined with mapped columns
    assert "Frank" in writer.get_data()["name"].to_list()


def test_predicate_helper_functions():
    """Test the predicate helper functions."""
    # Test equals
    pred = eq("name", "Alice")
    assert isinstance(pred, ColumnPredicate)
    assert pred.column == "name"
    assert pred.op == "eq"
    assert pred.value == "Alice"

    # Test greater than
    pred = gt("age", 30)
    assert pred.op == "gt"

    # Test less than
    pred = lt("age", 40)
    assert pred.op == "lt"

    # Test greater than or equal
    pred = gte("age", 25)
    assert pred.op == "gte"

    # Test less than or equal
    pred = lte("age", 50)
    assert pred.op == "lte"

    # Test not equal
    pred = ne("name", "Bob")
    assert pred.op == "ne"

    # Test in
    pred = is_in("name", ["Alice", "Bob"])
    assert pred.op == "in"
    assert pred.value == ["Alice", "Bob"]

    # Test not in
    pred = not_in("name", ["Charlie", "Dave"])
    assert pred.op == "not_in"

    # Test and
    pred = and_(eq("name", "Alice"), gt("age", 20))
    assert isinstance(pred, AndPredicate)
    assert len(pred.predicates) == 2

    # Test or
    pred = or_(eq("name", "Alice"), eq("name", "Bob"))
    assert isinstance(pred, OrPredicate)
    assert len(pred.predicates) == 2

    # Test not
    pred = not_(eq("name", "Alice"))
    assert isinstance(pred, NotPredicate)
    assert pred.predicate.column == "name"


def test_write_options():
    """Test the WriteOptions class."""
    # Test default values
    options = WriteOptions()
    assert options.mode == WriteMode.ERROR
    assert not options.schema_evolution
    assert options.column_mapping is None
    assert options.partition_by is None
    assert not options.coerce_to_schema

    # Test custom values
    options = WriteOptions(
        mode=WriteMode.APPEND,
        schema_evolution=True,
        column_mapping={"a": "b"},
        partition_by=["date"],
        coerce_to_schema=True
    )
    assert options.mode == WriteMode.APPEND
    assert options.schema_evolution
    assert options.column_mapping == {"a": "b"}
    assert options.partition_by == ["date"]
    assert options.coerce_to_schema


def test_read_options():
    """Test the ReadOptions class."""
    # Test default values
    options = ReadOptions()
    assert options.columns is None
    assert options.limit is None
    assert options.predicate is None

    # Test custom values
    pred = eq("name", "Alice")
    options = ReadOptions(
        columns=["id", "name"],
        limit=10,
        predicate=pred
    )
    assert options.columns == ["id", "name"]
    assert options.limit == 10
    assert options.predicate == pred