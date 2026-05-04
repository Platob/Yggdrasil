"""Catalog (Postgres database) collection — list / lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional

from .catalog import Catalog

if TYPE_CHECKING:
    from .executor import PostgresExecutor

__all__ = ["Catalogs"]


class Catalogs:
    """Collection-level operations across Postgres databases.

    Lifecycle for a single database lives on :class:`Catalog`; this
    class only handles enumeration and lookup. The bound executor's
    connection must point at a database with permission to read
    ``pg_database`` (every login user has it by default).
    """

    def __init__(self, executor: "PostgresExecutor"):
        self.executor = executor

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def catalog(self, name: Optional[str] = None) -> Catalog:
        """Resolve a single catalog by name.

        ``None`` (the default) resolves to the connection's current
        database via :meth:`Catalog._current_database`.
        """
        return Catalog(executor=self.executor, catalog_name=name)

    def __getitem__(self, name: str) -> Catalog:
        return self.catalog(name)

    def __iter__(self) -> Iterator[Catalog]:
        return self.list()

    def list(self, *, include_template: bool = False) -> Iterator[Catalog]:
        """Iterate over Postgres databases.

        Template databases (``template0``, ``template1``) are
        excluded by default — they're not user-managed targets.
        """
        cursor = self.executor.connection.psycopg_cursor()
        try:
            sql = "SELECT datname FROM pg_database WHERE datallowconn"
            if not include_template:
                sql += " AND datistemplate = false"
            sql += " ORDER BY datname"
            cursor.execute(sql)
            rows = cursor.fetchall()
        finally:
            cursor.close()
        for (name,) in rows:
            yield Catalog(executor=self.executor, catalog_name=name)
