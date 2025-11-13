"""Abstract base classes for data readers in Yggdrasil."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Dict, List, Optional, TypeVar, Union, Callable, Generic

import polars as pl

# Type variable for implementing generic reader with specific config type
ConfigT = TypeVar('ConfigT')


@dataclasses.dataclass
class ReaderPredicate:
    """Base class for reader predicates used for filter pushdown."""

    def to_expression(self) -> Dict[str, Any]:
        """Convert predicate to a dictionary expression format.

        Returns:
            Dict[str, Any]: Dictionary representation of the predicate.
        """
        raise NotImplementedError("Predicate must implement to_expression")

    @classmethod
    def from_expression(cls, expr: Dict[str, Any]) -> "ReaderPredicate":
        """Create a predicate from a dictionary expression.

        Args:
            expr: Dictionary representation of a predicate.

        Returns:
            ReaderPredicate: The corresponding predicate instance.
        """
        pred_type = expr.get("type")

        if pred_type == "column_predicate":
            return ColumnPredicate(
                column=expr["column"],
                op=expr["op"],
                value=expr["value"]
            )
        elif pred_type == "and":
            return AndPredicate([
                cls.from_expression(pred) for pred in expr["predicates"]
            ])
        elif pred_type == "or":
            return OrPredicate([
                cls.from_expression(pred) for pred in expr["predicates"]
            ])
        elif pred_type == "not":
            return NotPredicate(
                cls.from_expression(expr["predicate"])
            )
        else:
            raise ValueError(f"Unknown predicate type: {pred_type}")


@dataclasses.dataclass
class ColumnPredicate(ReaderPredicate):
    """Predicate for filtering by column values.

    This predicate represents simple column-value comparisons
    that can be pushed down to the source system.
    """

    column: str
    """Name of the column to filter on."""

    op: str
    """Comparison operator: 'eq', 'gt', 'lt', 'gte', 'lte', 'ne', 'in', 'not_in'."""

    value: Any
    """Value to compare against."""

    def to_expression(self) -> Dict[str, Any]:
        """Convert predicate to a dictionary expression format.

        Returns:
            Dict[str, Any]: Dictionary representation of the predicate.
        """
        return {
            "type": "column_predicate",
            "column": self.column,
            "op": self.op,
            "value": self.value
        }


@dataclasses.dataclass
class AndPredicate(ReaderPredicate):
    """Logical AND of multiple predicates."""

    predicates: List[ReaderPredicate]
    """List of predicates to combine with AND logic."""

    def to_expression(self) -> Dict[str, Any]:
        """Convert predicate to a dictionary expression format.

        Returns:
            Dict[str, Any]: Dictionary representation of the predicate.
        """
        return {
            "type": "and",
            "predicates": [pred.to_expression() for pred in self.predicates]
        }


@dataclasses.dataclass
class OrPredicate(ReaderPredicate):
    """Logical OR of multiple predicates."""

    predicates: List[ReaderPredicate]
    """List of predicates to combine with OR logic."""

    def to_expression(self) -> Dict[str, Any]:
        """Convert predicate to a dictionary expression format.

        Returns:
            Dict[str, Any]: Dictionary representation of the predicate.
        """
        return {
            "type": "or",
            "predicates": [pred.to_expression() for pred in self.predicates]
        }


@dataclasses.dataclass
class NotPredicate(ReaderPredicate):
    """Logical NOT of a predicate."""

    predicate: ReaderPredicate
    """Predicate to negate."""

    def to_expression(self) -> Dict[str, Any]:
        """Convert predicate to a dictionary expression format.

        Returns:
            Dict[str, Any]: Dictionary representation of the predicate.
        """
        return {
            "type": "not",
            "predicate": self.predicate.to_expression()
        }


@dataclasses.dataclass
class ReadOptions:
    """Common options for all readers."""

    columns: Optional[List[str]] = None
    """List of columns to select. If None, selects all columns."""

    limit: Optional[int] = None
    """Maximum number of rows to read."""

    predicate: Optional[ReaderPredicate] = None
    """Filter predicate to push down to the source."""


class DataReader(Generic[ConfigT], abc.ABC):
    """Abstract base class for data readers.

    This class provides a common interface for reading data from various sources
    into Polars DataFrames with support for predicate pushdown.
    """

    @abc.abstractmethod
    def to_polars(self, options: Optional[ReadOptions] = None) -> pl.DataFrame:
        """Read data into a Polars DataFrame.

        Args:
            options: Options controlling which data to read.

        Returns:
            pl.DataFrame: The data as a Polars DataFrame.
        """
        pass

    @abc.abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data source.

        Returns:
            Dict[str, Any]: Dictionary with metadata.
        """
        pass

    @classmethod
    def create(cls, config: ConfigT) -> DataReader:
        """Factory method to create a reader from a configuration.

        Args:
            config: Configuration for the reader.

        Returns:
            DataReader: A reader instance.
        """
        return cls(config)


# Helper functions to create predicates
def eq(column: str, value: Any) -> ColumnPredicate:
    """Create an equals predicate."""
    return ColumnPredicate(column, "eq", value)


def gt(column: str, value: Any) -> ColumnPredicate:
    """Create a greater-than predicate."""
    return ColumnPredicate(column, "gt", value)


def lt(column: str, value: Any) -> ColumnPredicate:
    """Create a less-than predicate."""
    return ColumnPredicate(column, "lt", value)


def gte(column: str, value: Any) -> ColumnPredicate:
    """Create a greater-than-or-equals predicate."""
    return ColumnPredicate(column, "gte", value)


def lte(column: str, value: Any) -> ColumnPredicate:
    """Create a less-than-or-equals predicate."""
    return ColumnPredicate(column, "lte", value)


def ne(column: str, value: Any) -> ColumnPredicate:
    """Create a not-equals predicate."""
    return ColumnPredicate(column, "ne", value)


def is_in(column: str, values: List[Any]) -> ColumnPredicate:
    """Create an IN predicate."""
    return ColumnPredicate(column, "in", values)


def not_in(column: str, values: List[Any]) -> ColumnPredicate:
    """Create a NOT IN predicate."""
    return ColumnPredicate(column, "not_in", values)


def and_(*predicates: ReaderPredicate) -> AndPredicate:
    """Create an AND predicate combining multiple predicates."""
    return AndPredicate(list(predicates))


def or_(*predicates: ReaderPredicate) -> OrPredicate:
    """Create an OR predicate combining multiple predicates."""
    return OrPredicate(list(predicates))


def not_(predicate: ReaderPredicate) -> NotPredicate:
    """Create a NOT predicate that negates the input predicate."""
    return NotPredicate(predicate)