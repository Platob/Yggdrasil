"""Backward-compatible alias for the merged :class:`Statement` handler.

``StatementResult`` was historically a separate class wrapping execution state.
It is now an alias of :class:`~yggdrasil.databricks.sql.statement.Statement`,
which unifies pre-execution and post-execution state into a single handler.
"""

from .statement import (
    DONE_STATES,
    FAILED_STATES,
    Statement,
)

StatementResult = Statement

__all__ = ["StatementResult", "Statement", "DONE_STATES", "FAILED_STATES"]
