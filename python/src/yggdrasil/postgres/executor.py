""":class:`PostgresExecutor` — :class:`StatementExecutor` for Postgres.

A thin shim around :meth:`PostgresConnection.psycopg_cursor` /
:meth:`PostgresConnection.adbc_cursor` that pins the
:class:`PostgresPreparedStatement` / :class:`PostgresStatementResult`
classes onto the base :class:`StatementExecutor`.

Postgres execution is synchronous, so the executor's only job is:

1. Coerce the incoming statement into the typed
   :class:`PostgresPreparedStatement`.
2. Build a :class:`PostgresStatementResult` bound to the connection.
3. Call :meth:`PostgresStatementResult.start` (with ``raise_error=False``
   so the base executor's policy decides what to do with failures).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

from yggdrasil.data.executor import StatementExecutor

from .connection import PostgresConnection
from .statement import (
    PostgresPreparedStatement,
    PostgresStatementBatch,
    PostgresStatementResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PostgresExecutor",
]


class PostgresExecutor(
    StatementExecutor[
        PostgresPreparedStatement,
        PostgresStatementResult,
        PostgresStatementBatch,
    ]
):
    """Run statements against a :class:`PostgresConnection`."""

    _PREPARED_STATEMENT_CLASS: ClassVar[type[PostgresPreparedStatement]] = PostgresPreparedStatement
    _STATEMENT_RESULT_CLASS: ClassVar[type[PostgresStatementResult]] = PostgresStatementResult
    _STATEMENT_BATCH_CLASS: ClassVar[type[PostgresStatementBatch]] = PostgresStatementBatch

    def __init__(
        self,
        connection: "PostgresConnection | str | None" = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.connection: PostgresConnection = PostgresConnection.from_(connection)

    # ------------------------------------------------------------------
    # Sub-services — placed here so the engine's catalog/schemas/tables
    # are reachable from any executor that owns a connection. Keeping
    # these as lazy properties means a pure-DDL workflow doesn't pay
    # the import cost of the resource hierarchy.
    # ------------------------------------------------------------------

    @property
    def catalogs(self):
        """Catalog (database) collection navigated through this connection."""
        from .catalogs import Catalogs
        return Catalogs(executor=self)

    def catalog(self, name: Optional[str] = None):
        """Resolve a single :class:`Catalog` (Postgres database) by name."""
        return self.catalogs.catalog(name)

    @property
    def schemas(self):
        from .schemas import Schemas
        return Schemas(executor=self)

    def schema(self, name: Optional[str] = None, *, catalog_name: Optional[str] = None):
        return self.schemas.schema(name, catalog_name=catalog_name)

    @property
    def tables(self):
        from .tables import Tables
        return Tables(executor=self)

    def table(
        self,
        location: Optional[str] = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ):
        return self.tables.table(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )

    # ------------------------------------------------------------------
    # Executor contract
    # ------------------------------------------------------------------

    def _submit_statement(
        self,
        statement: PostgresPreparedStatement,
        start: bool = True
    ) -> PostgresStatementResult:
        """Build a :class:`PostgresStatementResult` and run it eagerly."""
        result = self._STATEMENT_RESULT_CLASS(
            statement=statement,
            executor=self,
            connection=self.connection,
        )
        # raise_error=False: the base executor's _execute calls
        # raise_for_status afterwards if the caller's options say so.
        result.start(wait=False, raise_error=False)
        return result

    # ------------------------------------------------------------------
    # Conveniences
    # ------------------------------------------------------------------

    def sql(
        self,
        text: str,
        *,
        parameters: Any = None,
        prefer_arrow: bool = True,
        fetch_size: Optional[int] = None,
    ) -> PostgresStatementResult:
        """Run raw SQL and return the terminal result."""
        stmt = PostgresPreparedStatement(
            text=text,
            parameters=parameters,
            prefer_arrow=prefer_arrow,
            fetch_size=fetch_size,
        )
        return self.execute(stmt, wait=False, raise_error=True)

    # ------------------------------------------------------------------
    # Disposable
    # ------------------------------------------------------------------

    def _release(self, committed: bool = False) -> None:
        """Close the underlying connection on dispose."""
        super()._release(committed=committed)
        try:
            self.connection.close()
        except Exception:
            logger.exception("Closing PostgresConnection failed; continuing.")
