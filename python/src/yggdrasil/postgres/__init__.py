"""Postgres backend — Arrow-native catalog/schema/table hierarchy + executor.

Public surface
--------------

::

    from yggdrasil.postgres import (
        PostgresEngine,        # top-level façade
        PostgresConnection,    # paired psycopg + ADBC handles
        PostgresExecutor,      # StatementExecutor implementation
        Catalog, Schema, Table,
        Catalogs, Schemas, Tables,
        Column,
        PostgresPreparedStatement, PostgresStatementResult, PostgresStatementBatch,
    )

The engine is a :class:`yggdrasil.data.executor.StatementExecutor`
that composes the resource hierarchy with an ADBC-fast Arrow
read/write path. :class:`Table` implements
:class:`yggdrasil.io.buffer.base.TabularIO`, so every cross-engine
conversion (Arrow / Polars / pandas / Spark / records) lights up
out-of-the-box.

Optional dependencies live behind :mod:`yggdrasil.postgres.lib`.
"""

from .catalog import Catalog
from .catalogs import Catalogs
from .column import Column
from .connection import PostgresConnection, normalize_postgres_uri
from .engine import PostgresEngine
from .executor import PostgresExecutor
from .schema import Schema
from .schemas import Schemas
from .sql_utils import (
    DEFAULT_SCHEMA,
    escape_sql_string,
    parse_dotted_name,
    quote_ident,
    quote_qualified_ident,
    sql_literal,
    split_qualified_ident,
)
from .statement import (
    POSTGRES_STATEMENT_MIME,
    POSTGRES_TABLE_MIME,
    PostgresPreparedStatement,
    PostgresStatementBatch,
    PostgresStatementResult,
)
from .table import Table
from .tables import Tables
from .types import (
    arrow_schema_to_postgres_columns,
    arrow_to_postgres_field,
    arrow_to_postgres_type,
    postgres_to_arrow_field,
    postgres_to_arrow_type,
)

__all__ = [
    # Engine + executor + connection
    "PostgresEngine",
    "PostgresExecutor",
    "PostgresConnection",
    "normalize_postgres_uri",
    # Hierarchy
    "Catalog",
    "Catalogs",
    "Schema",
    "Schemas",
    "Table",
    "Tables",
    "Column",
    # Statement + batch
    "PostgresPreparedStatement",
    "PostgresStatementResult",
    "PostgresStatementBatch",
    "POSTGRES_STATEMENT_MIME",
    "POSTGRES_TABLE_MIME",
    # Types
    "arrow_to_postgres_type",
    "postgres_to_arrow_type",
    "arrow_to_postgres_field",
    "postgres_to_arrow_field",
    "arrow_schema_to_postgres_columns",
    # SQL utilities
    "DEFAULT_SCHEMA",
    "quote_ident",
    "quote_qualified_ident",
    "split_qualified_ident",
    "parse_dotted_name",
    "escape_sql_string",
    "sql_literal",
]
