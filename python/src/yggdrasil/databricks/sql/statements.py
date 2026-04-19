"""
Collection-level service for Databricks SQL statements.

The :class:`Statements` service wraps the workspace
``statement_execution`` SDK for single-statement lookups and reads
``system.query.history`` for listing historical executions.

Per-execution lifecycle operations (``start``, ``cancel``, ``wait``,
``to_arrow``…) live on the
:class:`~yggdrasil.databricks.sql.statement.StatementResult` handler
returned by this service.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional, TYPE_CHECKING

from databricks.sdk.errors import DatabricksError, ResourceDoesNotExist
from databricks.sdk.service.sql import StatementResponse, StatementState

from yggdrasil.data import any_to_datetime
from yggdrasil.data.statement import Statement
from yggdrasil.databricks.client import DatabricksService

from .sql_utils import escape_sql_string

if TYPE_CHECKING:
    from .statement import StatementResult

__all__ = ["Statements"]

logger = logging.getLogger(__name__)

# system.query.history columns we care about.  Ordered for a stable SELECT.
_HISTORY_COLUMNS: tuple[str, ...] = (
    "statement_id",
    "workspace_id",
    "executed_by",
    "executed_by_user_id",
    "statement_text",
    "statement_type",
    "execution_status",
    "error_message",
    "warehouse_id",
    "client_application",
    "start_time",
    "end_time",
    "execution_duration_ms",
    "compilation_duration_ms",
    "total_duration_ms",
    "read_rows",
    "produced_rows",
)

_SYSTEM_QUERY_HISTORY = "system.query.history"


def _sql_quote(value: str) -> str:
    return "'" + escape_sql_string(value) + "'"


def _to_timestamp_literal(value: Any) -> str:
    parsed = any_to_datetime(value, tz=_dt.timezone.utc)
    return f"TIMESTAMP {_sql_quote(parsed.isoformat(sep=' '))}"


@dataclass(frozen=True)
class Statements(DatabricksService):
    """Collection-level SQL statement execution service.

    Provides ``list_statements`` / ``find_statement`` operations plus
    :meth:`statement` to build an unstarted
    :class:`~yggdrasil.databricks.sql.statement.Statement` resource bound
    to this service.

    Listing queries the Unity Catalog ``system.query.history`` table.
    The Databricks REST API has no list endpoint for statement
    executions, so historical rows from the system table are the only
    reliable source of truth for completed queries.
    """

    warehouse_id: str | None = None
    executed_by: str | None = None

    @classmethod
    def service_name(cls) -> str:
        return "statements"

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #

    def statement(
        self,
        text: "str | Statement | StatementResult" = "",
        *,
        parameters: Optional[dict] = None,
        temporary_tables: Optional[dict] = None,
        statement_id: str | None = None,
        warehouse_id: str | None = None,
    ) -> "StatementResult":
        """Return an unstarted :class:`StatementResult` bound to this service."""
        from .statement import StatementResult

        if isinstance(text, StatementResult):
            prepared = StatementResult.prepare(
                text,
                parameters=parameters,
                temporary_tables=temporary_tables,
            )
            object.__setattr__(prepared, "service", self)
            if statement_id is not None:
                object.__setattr__(prepared, "statement_id", statement_id)
            if warehouse_id is not None:
                object.__setattr__(prepared, "warehouse_id", warehouse_id)
            return prepared

        cfg = Statement.prepare(
            text,
            parameters=parameters,
            temporary_tables=temporary_tables,
        )
        return StatementResult(
            statement=cfg,
            service=self,
            statement_id=statement_id,
            warehouse_id=warehouse_id or self.warehouse_id,
        )

    # ------------------------------------------------------------------ #
    # Single-statement lookup
    # ------------------------------------------------------------------ #

    def get_statement_response(self, statement_id: str) -> StatementResponse:
        """Raw ``GetStatement`` response for *statement_id*."""
        return (
            self.client
            .workspace_client()
            .statement_execution
            .get_statement(statement_id)
        )

    def find_statement(
        self,
        statement_id: str,
        *,
        raise_error: bool = True,
    ) -> Optional["StatementResult"]:
        """Resolve a :class:`StatementResult` by ``statement_id``.

        Uses the workspace ``GetStatement`` endpoint to fetch live status
        and binds the response onto a result handle.  When the id is
        not found and ``raise_error`` is False, returns ``None``.
        """
        from .statement import StatementResult

        try:
            response = self.get_statement_response(statement_id)
        except ResourceDoesNotExist:
            if raise_error:
                raise
            return None
        except DatabricksError as exc:
            if raise_error:
                raise ResourceDoesNotExist(
                    f"Failed to fetch statement {statement_id!r}: {exc}"
                ) from exc
            return None

        warehouse_id = None
        manifest = getattr(response, "manifest", None)
        if manifest is not None:
            warehouse_id = getattr(manifest, "warehouse_id", None) or warehouse_id

        stmt = StatementResult(
            service=self,
            statement_id=response.statement_id,
            warehouse_id=warehouse_id or self.warehouse_id,
        )
        object.__setattr__(stmt, "_response", response)
        return stmt

    # ------------------------------------------------------------------ #
    # Listing (system.query.history)
    # ------------------------------------------------------------------ #

    def _build_list_query(
        self,
        *,
        warehouse_id: Optional[str],
        executed_by: Optional[str],
        executed_by_user_id: Optional[str],
        statement_id: Optional[str],
        statement_type: Optional[str],
        status: Any,
        start_time_from: Any,
        start_time_to: Any,
        text_contains: Optional[str],
        limit: Optional[int],
        order_by: str,
    ) -> str:
        filters: list[str] = []

        if warehouse_id:
            filters.append(f"warehouse_id = {_sql_quote(warehouse_id)}")

        if executed_by:
            filters.append(f"executed_by = {_sql_quote(executed_by)}")

        if executed_by_user_id:
            filters.append(
                f"cast(executed_by_user_id as string) = {_sql_quote(str(executed_by_user_id))}"
            )

        if statement_id:
            filters.append(f"statement_id = {_sql_quote(statement_id)}")

        if statement_type:
            filters.append(f"statement_type = {_sql_quote(statement_type)}")

        if status is not None:
            state_value = status.value if isinstance(status, StatementState) else str(status)
            filters.append(f"execution_status = {_sql_quote(state_value.upper())}")

        if start_time_from is not None:
            filters.append(f"start_time >= {_to_timestamp_literal(start_time_from)}")

        if start_time_to is not None:
            filters.append(f"start_time < {_to_timestamp_literal(start_time_to)}")

        if text_contains:
            filters.append(
                f"lower(statement_text) like {_sql_quote('%' + text_contains.lower() + '%')}"
            )

        where = ("\nWHERE " + "\n  AND ".join(filters)) if filters else ""
        columns = ", ".join(_HISTORY_COLUMNS)

        query = (
            f"SELECT {columns}\nFROM {_SYSTEM_QUERY_HISTORY}{where}\n"
            f"ORDER BY {order_by}"
        )
        if limit is not None:
            query += f"\nLIMIT {int(limit)}"
        return query

    def list_statements(
        self,
        *,
        warehouse_id: str | None = None,
        executed_by: str | None = None,
        executed_by_user_id: str | None = None,
        statement_id: str | None = None,
        statement_type: str | None = None,
        status: StatementState | str | None = None,
        start_time_from: Any = "default",
        start_time_to: Any = "default",
        text_contains: str | None = None,
        limit: int | None = 1000,
        order_by: str = "start_time DESC",
        fetch_response: bool = False,
    ) -> Iterator["StatementResult"]:
        """Yield :class:`StatementResult` handlers from ``system.query.history``.

        Defaults inherited from the service (``warehouse_id``,
        ``executed_by``) are applied when the per-call argument is not
        supplied.

        Parameters
        ----------
        warehouse_id, executed_by, executed_by_user_id, statement_id,
        statement_type, status:
            Equality filters applied in the SQL ``WHERE`` clause.
        start_time_from, start_time_to:
            Half-open time range on the ``start_time`` column.  Parsed
            via :func:`yggdrasil.data.any_to_datetime`, so ISO strings,
            ``datetime`` / ``date`` objects, and epoch numerics are all
            accepted; aware values are normalised to UTC.  The sentinel
            ``"default"`` (the default) resolves ``start_time_to`` to
            the current UTC time and ``start_time_from`` to
            ``start_time_to - 7 days``.  Pass ``None`` to drop the
            corresponding bound from the query.
        text_contains:
            Case-insensitive substring match on ``statement_text``.
        limit:
            Row cap applied via SQL ``LIMIT``.  ``None`` means no cap.
        order_by:
            Raw ``ORDER BY`` clause (default: most recent first).
        fetch_response:
            When ``True``, resolve each row's live ``StatementResponse``
            via the SDK so that ``state`` / ``status`` reflect current
            values.  Slower but accurate.  When ``False`` (default),
            resources carry only the values from the history snapshot.
        """
        from .statement import StatementResult

        effective_warehouse = warehouse_id or self.warehouse_id
        effective_executed_by = executed_by or self.executed_by

        if start_time_to == "default":
            start_time_to = _dt.datetime.now(_dt.timezone.utc)
        if start_time_from == "default":
            anchor = (
                any_to_datetime(start_time_to, tz=_dt.timezone.utc)
                if start_time_to is not None
                else _dt.datetime.now(_dt.timezone.utc)
            )
            start_time_from = anchor - _dt.timedelta(days=7)

        query = self._build_list_query(
            warehouse_id=effective_warehouse,
            executed_by=effective_executed_by,
            executed_by_user_id=executed_by_user_id,
            statement_id=statement_id,
            statement_type=statement_type,
            status=status,
            start_time_from=start_time_from,
            start_time_to=start_time_to,
            text_contains=text_contains,
            limit=limit,
            order_by=order_by,
        )

        logger.debug("Listing statements via system.query.history:\n%s", query)

        result = self.client.sql.execute(query, engine="api")

        for row in result.to_pylist():
            row_stmt_id = row.get("statement_id")
            if not row_stmt_id:
                continue

            if fetch_response:
                stmt = self.find_statement(row_stmt_id, raise_error=False)
                if stmt is None:
                    continue
                object.__setattr__(stmt, "_history", row)
                yield stmt
                continue

            stmt = StatementResult(
                statement=Statement(text=row.get("statement_text") or ""),
                service=self,
                statement_id=row_stmt_id,
                warehouse_id=row.get("warehouse_id") or effective_warehouse,
            )
            object.__setattr__(stmt, "_history", row)
            yield stmt

    # ------------------------------------------------------------------ #
    # Dict-like access by statement_id
    # ------------------------------------------------------------------ #

    def __getitem__(self, statement_id: str) -> "StatementResult":
        stmt = self.find_statement(statement_id, raise_error=True)
        assert stmt is not None  # find_statement raises when not found
        return stmt

    def __contains__(self, statement_id: Any) -> bool:
        if not isinstance(statement_id, str):
            return False
        return self.find_statement(statement_id, raise_error=False) is not None
