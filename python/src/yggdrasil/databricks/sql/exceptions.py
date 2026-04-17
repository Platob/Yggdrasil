"""Custom exceptions for Databricks SQL helpers."""
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from databricks.sdk.service.sql import ServiceErrorCode, StatementState

from ..lib import DatabricksError

if TYPE_CHECKING:
    from .statement_result import StatementResult

__all__ = [
    "SQLError",
    "SqlStatementError"
]


@dataclass(frozen=True)
class SQLError(DatabricksError):
    statement_id: str
    state: StatementState
    message: str
    error_code: ServiceErrorCode
    url: str | None = None

    def __str__(self) -> str:
        return f"[%s][%s][%s]: %s" % (
            self.url,
            self.state.value,
            self.error_code.value,
            self.message
        )

    def __repr__(self):
        return f"SqlStatementError(url={self.url!r}, message={self.message!r})"

    @classmethod
    def from_statement(cls, stmt: "StatementResult") -> "SQLError":
        statement_id = stmt.statement_id or "<unknown>"
        state = stmt.state
        status = stmt.status

        if status and status.error:
            message = status.error.message or "Unknown"
            error_code = status.error.error_code or ServiceErrorCode.INTERNAL_ERROR
        else:
            message, error_code = "Unknown", ServiceErrorCode.INTERNAL_ERROR

        url = str(stmt.monitoring_url)

        return cls(
            statement_id=statement_id,
            state=state,
            message=message,
            error_code=error_code,
            url=url
        )


SqlStatementError = SQLError