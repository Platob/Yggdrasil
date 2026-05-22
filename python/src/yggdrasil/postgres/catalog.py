"""Per-catalog (Postgres ``database``) resource: lifecycle + schemas.

Postgres calls the top-level container a *database*. SQL-standard
parlance — and the rest of yggdrasil — calls it a *catalog*. We
keep the yggdrasil naming so the cross-backend API stays uniform
with :mod:`yggdrasil.databricks.catalog.catalog`, but every DDL emitted
here is the Postgres ``DATABASE`` form.

Cross-database queries
----------------------
Postgres connections are bound to a single database; switching
catalogs requires a fresh :class:`PostgresConnection`. The
``catalog_name`` on a :class:`Catalog` is therefore an *identity*,
not a runtime knob — calls to :meth:`schemas` walk
``information_schema.schemata`` on the *connection's* current
database, which must match ``self.catalog_name`` for the listing
to make sense. The mismatch is logged once.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from .sql_utils import quote_ident, sql_literal

if TYPE_CHECKING:
    from .catalogs import Catalogs
    from .executor import PostgresExecutor
    from .schema import Schema
    from .table import Table

logger = logging.getLogger(__name__)

__all__ = ["Catalog"]


class Catalog:
    """A Postgres database, addressed by name.

    Lifecycle DDL (``CREATE DATABASE`` / ``DROP DATABASE``) is
    emitted via the bound executor's connection — which itself must
    be connected to a database with permission to issue those
    commands (typically ``postgres`` or a dedicated admin DB).
    """

    def __init__(
        self,
        executor: Optional["PostgresExecutor"] = None,
        catalog_name: Optional[str] = None,
        *,
        service: Optional["Catalogs"] = None,
    ):
        if executor is None and service is not None:
            executor = service.executor
        if executor is None:
            raise ValueError("Catalog requires an executor (or a service that carries one).")
        self.executor = executor
        self.service = service
        self.catalog_name = catalog_name or self._current_database()

    # ── identity ──────────────────────────────────────────────────────────

    def full_name(self, safe: bool = False) -> str:
        return quote_ident(self.catalog_name) if safe else (self.catalog_name or "")

    def __repr__(self) -> str:
        return f"PostgresCatalog<{self.full_name()!r}>"

    def __str__(self) -> str:
        return self.catalog_name or ""

    def __getitem__(self, name: str) -> "Schema":
        return self.schema(name)

    def __setitem__(self, name: str, new_name: str) -> None:
        self.schema(name).rename(new_name)

    def __iter__(self) -> Iterator["Schema"]:
        return self.schemas()

    # ── existence / lifecycle ─────────────────────────────────────────────

    @property
    def exists(self) -> bool:
        cursor = self.executor.connection.psycopg_cursor()
        try:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s LIMIT 1",
                (self.catalog_name,),
            )
            return cursor.fetchone() is not None
        finally:
            cursor.close()

    def create(
        self,
        *,
        missing_ok: bool = True,
        owner: Optional[str] = None,
        encoding: Optional[str] = None,
        template: Optional[str] = None,
    ) -> "Catalog":
        """``CREATE DATABASE`` — must run outside a transaction.

        Postgres rejects ``CREATE DATABASE`` inside a transaction
        block, so this temporarily flips the underlying psycopg
        connection to autocommit. The flip is reverted on the way
        out.
        """
        if missing_ok and self.exists:
            return self
        clauses: list[str] = []
        if owner:
            clauses.append(f"OWNER {quote_ident(owner)}")
        if encoding:
            clauses.append(f"ENCODING {sql_literal(encoding)}")
        if template:
            clauses.append(f"TEMPLATE {quote_ident(template)}")
        ddl = f"CREATE DATABASE {quote_ident(self.catalog_name)}"
        if clauses:
            ddl += " WITH " + " ".join(clauses)
        self._exec_outside_transaction(ddl)
        return self

    def ensure_created(self, **kwargs: Any) -> "Catalog":
        if not self.exists:
            self.create(missing_ok=True, **kwargs)
        return self

    def delete(self, *, if_exists: bool = True) -> "Catalog":
        """``DROP DATABASE`` — also requires autocommit."""
        head = "DROP DATABASE IF EXISTS" if if_exists else "DROP DATABASE"
        self._exec_outside_transaction(f"{head} {quote_ident(self.catalog_name)}")
        return self

    drop = delete

    def rename(self, new_name: str) -> "Catalog":
        new_name = (new_name or "").strip().strip('"')
        if not new_name:
            raise ValueError("Cannot rename database to an empty name")
        if new_name == self.catalog_name:
            return self
        self._exec_outside_transaction(
            f"ALTER DATABASE {quote_ident(self.catalog_name)} "
            f"RENAME TO {quote_ident(new_name)}"
        )
        self.catalog_name = new_name
        return self

    def set_comment(self, comment: Optional[str]) -> "Catalog":
        value = "NULL" if comment is None else sql_literal(comment)
        self.executor.sql(
            f"COMMENT ON DATABASE {quote_ident(self.catalog_name)} IS {value}",
            prefer_arrow=False,
        )
        return self

    # ── navigation ────────────────────────────────────────────────────────

    def schema(self, name: str) -> "Schema":
        from .schema import Schema as _Schema
        return _Schema(
            executor=self.executor,
            catalog_name=self.catalog_name,
            schema_name=name,
        )

    def schemas(self) -> Iterator["Schema"]:
        """Iterate over user schemas (excludes ``pg_*`` and ``information_schema``)."""
        from .schema import Schema as _Schema
        self._warn_on_catalog_mismatch()
        cursor = self.executor.connection.psycopg_cursor()
        try:
            cursor.execute(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT LIKE 'pg\\_%%' "
                "AND schema_name <> 'information_schema' "
                "ORDER BY schema_name"
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
        for (schema_name,) in rows:
            yield _Schema(
                executor=self.executor,
                catalog_name=self.catalog_name,
                schema_name=schema_name,
            )

    def table(
        self,
        location: Optional[str] = None,
        *,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> "Table":
        return self.executor.tables.table(
            location=location,
            catalog_name=self.catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _current_database(self) -> Optional[str]:
        cursor = self.executor.connection.psycopg_cursor()
        try:
            cursor.execute("SELECT current_database()")
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            cursor.close()

    def _warn_on_catalog_mismatch(self) -> None:
        current = self._current_database()
        if (
            self.catalog_name
            and current
            and current != self.catalog_name
        ):
            logger.warning(
                "PostgresCatalog.catalog_name=%r does not match the connected "
                "database %r; schema listings reflect %r. Use a separate "
                "PostgresConnection to traverse other databases.",
                self.catalog_name, current, current,
            )

    def _exec_outside_transaction(self, ddl: str) -> None:
        """Run *ddl* with autocommit forced on, restoring the prior state."""
        conn = self.executor.connection.psycopg_conn
        previous = getattr(conn, "autocommit", False)
        if not previous:
            conn.autocommit = True
        try:
            cursor = conn.cursor()
            try:
                cursor.execute(ddl)
            finally:
                cursor.close()
        finally:
            if not previous:
                conn.autocommit = False
