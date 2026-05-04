"""Schema collection — list / lookup across schemas in a database."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Optional

from .schema import Schema
from .sql_utils import DEFAULT_SCHEMA

if TYPE_CHECKING:
    from .executor import PostgresExecutor

__all__ = ["Schemas"]


class Schemas:
    """Collection-level operations on Postgres schemas (single database)."""

    def __init__(
        self,
        executor: "PostgresExecutor",
        catalog_name: Optional[str] = None,
    ):
        self.executor = executor
        self.catalog_name = catalog_name

    def schema(
        self,
        name: Optional[str] = None,
        *,
        catalog_name: Optional[str] = None,
    ) -> Schema:
        """Resolve a single schema; ``None`` falls back to ``public``."""
        return Schema(
            executor=self.executor,
            catalog_name=catalog_name or self.catalog_name,
            schema_name=name or DEFAULT_SCHEMA,
        )

    def __getitem__(self, name: str) -> Schema:
        return self.schema(name)

    def __iter__(self) -> Iterator[Schema]:
        return self.list()

    def list(self, *, include_system: bool = False) -> Iterator[Schema]:
        """Iterate over schemas in the current database.

        System schemas (``pg_*`` + ``information_schema``) are
        excluded unless ``include_system=True`` — they're rarely
        useful targets for application code, and including them
        bloats simple ``for s in engine.schemas`` loops.
        """
        cursor = self.executor.connection.psycopg_cursor()
        try:
            sql = "SELECT schema_name FROM information_schema.schemata"
            if not include_system:
                sql += (
                    " WHERE schema_name NOT LIKE 'pg\\_%%' "
                    "AND schema_name <> 'information_schema'"
                )
            sql += " ORDER BY schema_name"
            cursor.execute(sql)
            rows = cursor.fetchall()
        finally:
            cursor.close()
        for (name,) in rows:
            yield Schema(
                executor=self.executor,
                catalog_name=self.catalog_name,
                schema_name=name,
            )
