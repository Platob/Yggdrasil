"""Per-schema (Postgres ``schema``, not yggdrasil) resource: lifecycle + tables."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from .sql_utils import DEFAULT_SCHEMA, quote_ident, sql_literal

if TYPE_CHECKING:
    from .catalog import Catalog
    from .executor import PostgresExecutor
    from .schemas import Schemas
    from .table import Table

logger = logging.getLogger(__name__)

__all__ = ["Schema"]


class Schema:
    """A single Postgres schema bound to an executor.

    Postgres "schema" is the second tier under "database" (which the
    yggdrasil model surfaces as a :class:`Catalog`). Tables live
    inside a schema; cross-schema queries are fine on a single
    connection, cross-database queries require a fresh connection.

    Navigation
    ----------
    ``schema["users"]`` resolves to a :class:`Table`; iterating a
    schema yields tables. ``schema.catalog`` walks back up to the
    parent.
    """

    def __init__(
        self,
        executor: Optional["PostgresExecutor"] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        *,
        service: Optional["Schemas"] = None,
    ):
        if executor is None and service is not None:
            executor = service.executor
        if executor is None:
            raise ValueError("Schema requires an executor (or a service that carries one).")
        self.executor = executor
        self.service = service
        self.catalog_name = catalog_name
        self.schema_name = schema_name or DEFAULT_SCHEMA

    # ── identity ──────────────────────────────────────────────────────────

    def full_name(self, safe: bool = False) -> str:
        if self.catalog_name:
            parts = [self.catalog_name, self.schema_name]
        else:
            parts = [self.schema_name]
        if safe:
            return ".".join(quote_ident(p) for p in parts)
        return ".".join(parts)

    def __repr__(self) -> str:
        return f"PostgresSchema<{self.full_name()!r}>"

    def __str__(self) -> str:
        return self.full_name()

    def __getitem__(self, name: str) -> "Table":
        return self.table(name)

    def __setitem__(self, name: str, new_name: str) -> None:
        self.table(name).rename(new_name)

    def __iter__(self) -> Iterator["Table"]:
        return self.tables()

    # ── existence / lifecycle ─────────────────────────────────────────────

    @property
    def exists(self) -> bool:
        cursor = self.executor.connection.psycopg_cursor()
        try:
            cursor.execute(
                "SELECT 1 FROM information_schema.schemata "
                "WHERE schema_name = %s LIMIT 1",
                (self.schema_name,),
            )
            return cursor.fetchone() is not None
        finally:
            cursor.close()

    def create(
        self,
        *,
        missing_ok: bool = True,
        owner: Optional[str] = None,
    ) -> "Schema":
        head = "CREATE SCHEMA IF NOT EXISTS" if missing_ok else "CREATE SCHEMA"
        ddl = f"{head} {quote_ident(self.schema_name)}"
        if owner:
            ddl += f" AUTHORIZATION {quote_ident(owner)}"
        self.executor.sql(ddl, prefer_arrow=False)
        return self

    def ensure_created(self, *, owner: Optional[str] = None) -> "Schema":
        if not self.exists:
            self.create(missing_ok=True, owner=owner)
        return self

    def delete(self, *, if_exists: bool = True, cascade: bool = False) -> "Schema":
        head = "DROP SCHEMA IF EXISTS" if if_exists else "DROP SCHEMA"
        tail = " CASCADE" if cascade else ""
        self.executor.sql(
            f"{head} {quote_ident(self.schema_name)}{tail}", prefer_arrow=False,
        )
        return self

    drop = delete

    def rename(self, new_name: str) -> "Schema":
        new_name = (new_name or "").strip().strip('"')
        if not new_name:
            raise ValueError("Cannot rename schema to an empty name")
        if new_name == self.schema_name:
            return self
        self.executor.sql(
            f"ALTER SCHEMA {quote_ident(self.schema_name)} "
            f"RENAME TO {quote_ident(new_name)}",
            prefer_arrow=False,
        )
        self.schema_name = new_name
        return self

    def set_comment(self, comment: Optional[str]) -> "Schema":
        value = "NULL" if comment is None else sql_literal(comment)
        self.executor.sql(
            f"COMMENT ON SCHEMA {quote_ident(self.schema_name)} IS {value}",
            prefer_arrow=False,
        )
        return self

    # ── navigation ────────────────────────────────────────────────────────

    @property
    def catalog(self) -> "Catalog":
        from .catalog import Catalog as _Catalog
        return _Catalog(executor=self.executor, catalog_name=self.catalog_name)

    def table(self, name: str) -> "Table":
        from .table import Table as _Table
        return _Table(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=name,
            executor=self.executor,
        )

    def tables(self, *, name_pattern: Optional[str] = None) -> Iterator["Table"]:
        """Iterate over user tables (excluding views / system tables)."""
        from .table import Table as _Table
        cursor = self.executor.connection.psycopg_cursor()
        try:
            sql = (
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s AND table_type = 'BASE TABLE'"
            )
            params: list[Any] = [self.schema_name]
            if name_pattern:
                sql += " AND table_name LIKE %s"
                params.append(name_pattern)
            sql += " ORDER BY table_name"
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        finally:
            cursor.close()
        for (name,) in rows:
            yield _Table(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                table_name=name,
                executor=self.executor,
            )
