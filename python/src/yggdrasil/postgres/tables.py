"""Table collection — list / lookup across schemas in a database."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional

from .sql_utils import DEFAULT_SCHEMA
from .table import Table

if TYPE_CHECKING:
    from .executor import PostgresExecutor

__all__ = ["Tables"]


class Tables:
    """Collection-level operations on Postgres tables.

    Provides ``find_table`` / ``list_tables`` / dotted-name lookup;
    per-table operations live on :class:`Table` itself.
    """

    def __init__(
        self,
        executor: "PostgresExecutor",
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
    ):
        self.executor = executor
        self.catalog_name = catalog_name
        self.schema_name = schema_name

    def table(
        self,
        location: Optional[str] = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> Table:
        """Resolve a :class:`Table` from a dotted name + per-call overrides."""
        return Table.from_(
            obj=location,
            catalog_name=catalog_name or self.catalog_name,
            schema_name=schema_name or self.schema_name or DEFAULT_SCHEMA,
            table_name=table_name,
            service=self,
            executor=self.executor,
        )

    def __getitem__(self, location: str) -> Table:
        return self.table(location)

    def __iter__(self) -> Iterator[Table]:
        return self.list_tables()

    def list_tables(
        self,
        *,
        schema_name: Optional[str] = None,
        name_pattern: Optional[str] = None,
    ) -> Iterator[Table]:
        """Iterate over tables, optionally scoped to a single schema.

        ``schema_name=None`` walks every non-system schema in the
        connected database. ``name_pattern`` is a SQL ``LIKE`` pattern
        applied server-side.
        """
        cursor = self.executor.connection.psycopg_cursor()
        try:
            sql = (
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_type = 'BASE TABLE' "
                "AND table_schema NOT LIKE 'pg\\_%%' "
                "AND table_schema <> 'information_schema'"
            )
            params: list = []
            target_schema = schema_name or self.schema_name
            if target_schema:
                sql += " AND table_schema = %s"
                params.append(target_schema)
            if name_pattern:
                sql += " AND table_name LIKE %s"
                params.append(name_pattern)
            sql += " ORDER BY table_schema, table_name"
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        finally:
            cursor.close()
        for table_schema, table_name in rows:
            yield Table(
                catalog_name=self.catalog_name,
                schema_name=table_schema,
                table_name=table_name,
                executor=self.executor,
            )
