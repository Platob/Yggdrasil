"""Custom exceptions for Databricks SQL helpers."""
from dataclasses import dataclass
from typing import Optional, Any

__all__ = [
    "SqlStatementError"
]


@dataclass(frozen=True)
class SqlStatementError(RuntimeError):
    statement_id: str
    state: str
    message: str
    error_code: Optional[str] = None
    sql_state: Optional[str] = None

    def __str__(self) -> str:
        meta = []

        if self.error_code:
            meta.append(f"code={self.error_code}")
        if self.sql_state:
            meta.append(f"state={self.sql_state}")

        meta_str = f" ({', '.join(meta)})" if meta else ""

        return f"SQL statement {self.statement_id!r} failed [{self.state}]: {self.message}{meta_str}"

    @classmethod
    def from_statement(cls, stmt: Any) -> "SqlStatementError":
        statement_id = getattr(stmt, "statement_id", "<unknown>")
        state = getattr(stmt, "state", "<unknown>")

        err = getattr(getattr(stmt, "status", None), "error", None)

        message = getattr(err, "message", None) or "Unknown SQL error"
        error_code = getattr(err, "error_code", None)
        sql_state = getattr(err, "sql_state", None)

        return cls(
            statement_id=str(statement_id),
            state=str(state),
            message=str(message),
            error_code=str(error_code) if error_code is not None else None,
            sql_state=str(sql_state) if sql_state is not None else None,
        )
