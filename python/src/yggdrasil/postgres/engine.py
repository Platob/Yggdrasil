""":class:`PostgresEngine` — top-level façade around the Postgres backend.

The engine composes the catalog/schema/table hierarchy with the
:class:`PostgresExecutor` so end users have a single entry point::

    from yggdrasil.postgres import PostgresEngine

    with PostgresEngine("postgresql://localhost/mydb") as eng:
        tbl = eng.table("public.users")
        df = tbl.read_polars_frame()
        eng.execute("INSERT INTO public.users(name) VALUES ('alice')")

The engine itself is a :class:`StatementExecutor` — every SQL call
funnels through :meth:`PostgresExecutor.execute` / ``execute_many``,
inheriting the cross-backend statement / batch / retry contract. The
hierarchy properties (``catalogs`` / ``schemas`` / ``tables``) are
re-exposed at the engine level for convenience; rebinding scope
without forking the connection is done with :meth:`__call__`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.statement import StatementResult
from yggdrasil.data.enums import Mode

from .connection import PostgresConnection
from .executor import PostgresExecutor
from .statement import PostgresPreparedStatement
from .sql_utils import DEFAULT_SCHEMA

if TYPE_CHECKING:
    from .catalog import Catalog
    from .catalogs import Catalogs
    from .schema import Schema
    from .schemas import Schemas
    from .table import Table
    from .tables import Tables

logger = logging.getLogger(__name__)

__all__ = ["PostgresEngine"]


class PostgresEngine(PostgresExecutor):
    """Top-level Postgres backend façade.

    Inherits :class:`PostgresExecutor` (which itself is a
    :class:`StatementExecutor`), so every cross-backend statement
    helper — ``execute``, ``execute_many``, ``batch`` — works
    out-of-the-box. The hierarchy navigation (``engine.catalogs``,
    ``engine.schemas``, ``engine.tables``) shadows the executor's
    own properties to thread default scope (catalog + schema)
    through.

    Construction
    ------------
    Pass a URI, an existing :class:`PostgresConnection`, or rely on
    the ``POSTGRES_URI`` environment variable (handled by
    :class:`PostgresConnection`).
    """

    def __init__(
        self,
        connection: "PostgresConnection | str | None" = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(connection=connection, **kwargs)
        self.catalog_name = catalog_name
        self.schema_name = schema_name or DEFAULT_SCHEMA

    # ------------------------------------------------------------------
    # Scope rebind
    # ------------------------------------------------------------------

    def __call__(
        self,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> "PostgresEngine":
        """Return a re-scoped engine sharing the same connection."""
        if catalog_name is None and schema_name is None:
            return self
        if (
            catalog_name == self.catalog_name
            and schema_name == self.schema_name
        ):
            return self
        eng = PostgresEngine.__new__(PostgresEngine)
        # Bypass __init__ so we don't open a second connection — the
        # rebind reuses the live one verbatim.
        PostgresExecutor.__init__(eng, connection=self.connection)
        eng.catalog_name = catalog_name if catalog_name is not None else self.catalog_name
        eng.schema_name = schema_name if schema_name is not None else self.schema_name
        return eng

    # ------------------------------------------------------------------
    # Hierarchy navigation
    # ------------------------------------------------------------------

    @property
    def catalogs(self) -> "Catalogs":
        from .catalogs import Catalogs as _Catalogs
        return _Catalogs(executor=self)

    def catalog(self, name: Optional[str] = None) -> "Catalog":
        from .catalog import Catalog as _Catalog
        return _Catalog(executor=self, catalog_name=name or self.catalog_name)

    @property
    def schemas(self) -> "Schemas":
        from .schemas import Schemas as _Schemas
        return _Schemas(executor=self, catalog_name=self.catalog_name)

    def schema(
        self,
        name: Optional[str] = None,
        *,
        catalog_name: Optional[str] = None,
    ) -> "Schema":
        from .schema import Schema as _Schema
        return _Schema(
            executor=self,
            catalog_name=catalog_name or self.catalog_name,
            schema_name=name or self.schema_name,
        )

    @property
    def tables(self) -> "Tables":
        from .tables import Tables as _Tables
        return _Tables(
            executor=self,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )

    def table(
        self,
        location: Optional[str] = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> "Table":
        return self.tables.table(
            location=location,
            catalog_name=catalog_name or self.catalog_name,
            schema_name=schema_name or self.schema_name,
            table_name=table_name,
        )

    # ------------------------------------------------------------------
    # High-level conveniences
    # ------------------------------------------------------------------

    def insert_into(
        self,
        data: Any,
        *,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        mode: Mode | str | None = None,
        match_by: Optional[Iterable[str]] = None,
        update_column_names: Optional[Iterable[str]] = None,
        cast_options: Optional[CastOptions] = None,
        table: Optional["Table"] = None,
    ) -> "Table":
        """Resolve target + delegate to :meth:`Table.insert_into`."""
        if table is None:
            table = self.table(
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
            )
        return table.insert_into(
            data,
            mode=mode,
            match_by=list(match_by) if match_by else None,
            update_column_names=list(update_column_names) if update_column_names else None,
            cast_options=cast_options,
        )

    def create_table(
        self,
        definition: Any,
        *,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "Table":
        target = self.table(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )
        return target.create(definition=definition, **kwargs)

    def drop_table(
        self,
        location: Optional[str] = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        if_exists: bool = True,
        cascade: bool = False,
    ) -> None:
        target = self.table(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )
        target.delete(if_exists=if_exists, cascade=cascade)

    # ------------------------------------------------------------------
    # Public execute — fold per-call routing knobs into the typed stmt.
    # ------------------------------------------------------------------

    def execute(
        self,
        statement: "PostgresPreparedStatement | StatementResult | str",
        *,
        parameters: Optional[Any] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        prefer_arrow: bool = True,
        fetch_size: Optional[int] = None,
        wait: Any = True,
        raise_error: bool = True,
        retry: Any = None,
    ) -> StatementResult:
        """Execute a SQL statement with per-call routing kwargs.

        Already-started results pass through untouched (matches the
        Databricks engine's behaviour). String / bare statement
        inputs are coerced through
        :meth:`PostgresPreparedStatement.prepare`.
        """
        if isinstance(statement, StatementResult):
            already_running = (
                parameters is None
                and getattr(statement, "started", statement.done)
            )
            if already_running:
                return statement.wait(wait=wait, raise_error=raise_error)
            statement = statement.statement

        prepared = PostgresPreparedStatement.prepare(
            statement,
            parameters=parameters,
            catalog_name=catalog_name or self.catalog_name,
            schema_name=schema_name or self.schema_name,
            prefer_arrow=prefer_arrow,
            fetch_size=fetch_size,
            retry=retry,
        )
        return super().execute(prepared, wait=wait, raise_error=raise_error)
