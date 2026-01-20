"""Databricks SQL helpers and engine wrappers."""

from .engine import SQLEngine, StatementResult
from .exceptions import SqlStatementError

# Backwards compatibility
DBXSQL = SQLEngine
DBXStatementResult = StatementResult
