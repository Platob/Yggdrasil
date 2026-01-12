"""Databricks SQL helpers and engine wrappers."""

from .engine import SQLEngine, StatementResult

# Backwards compatibility
DBXSQL = SQLEngine
DBXStatementResult = StatementResult

__all__ = ["SQLEngine", "StatementResult"]
