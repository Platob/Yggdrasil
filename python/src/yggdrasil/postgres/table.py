"""Per-table resource: DDL, DML, schema introspection, and Arrow I/O.

The :class:`Table` is both a navigation object (rename / drop /
list columns) **and** a :class:`Tabular` — it implements the two
hooks (:meth:`_read_arrow_batches` / :meth:`_write_arrow_batches`)
that drive every Arrow / Polars / pandas conversion in
:class:`yggdrasil.io.buffer.base.Tabular`. That means a caller can
treat a :class:`Table` like any other tabular IO::

    tbl = engine.table("public.users")
    df  = tbl.read_polars_frame()      # SELECT * FROM ... → Polars
    tbl.write_arrow_table(arrow_tbl)   # adbc_ingest fast-path

The actual round-trips are split between two driver paths:

* ADBC (preferred when the driver is installed): Arrow goes in and
  out without ever touching Python rows. Reads use
  ``cursor.fetch_arrow_table``; writes use ``cursor.adbc_ingest``
  with the matching :class:`Mode`.
* psycopg fallback: rows are materialised through the cursor and
  lifted into Arrow. Functional, but slower for large tables.

Save modes
----------
:class:`Mode` resolves to ADBC's ``mode=`` argument:

* ``APPEND`` / ``AUTO`` → ``"append"``
* ``OVERWRITE`` → ``"replace"`` (DROP+CREATE; loses constraints)
* ``TRUNCATE`` → ``TRUNCATE`` + ``"append"``
* ``ERROR_IF_EXISTS`` → ``"create"``
* ``IGNORE`` → ``"create_append"`` (no-op when target exists)
* ``UPSERT`` / ``MERGE`` → temp-table + ``INSERT ... ON CONFLICT``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

import pyarrow as pa

from yggdrasil.data import Schema as DataSchema
from yggdrasil.data.options import CastOptions
from yggdrasil.io.tabular.base import O, Tabular
from yggdrasil.data.enums import MimeType, MimeTypes, Mode

from .column import Column
from .sql_utils import (
    DEFAULT_SCHEMA,
    parse_dotted_name,
    quote_ident,
    quote_qualified_ident,
)
from .statement import POSTGRES_TABLE_MIME, PostgresPreparedStatement
from .types import arrow_schema_to_postgres_columns

if TYPE_CHECKING:
    from .connection import PostgresConnection
    from .executor import PostgresExecutor
    from .schema import Schema as PgSchema
    from .tables import Tables

logger = logging.getLogger(__name__)

__all__ = ["Table"]


# ADBC ingest mode names — pinned here so callers don't rely on
# pyarrow's enum (the wire constants are the same across driver
# versions). Mapping is opinionated: AUTO/APPEND both pick "append"
# since "append" is the safe default for every ingest, and OVERWRITE
# uses ADBC's "replace" so the table is rebuilt from the Arrow
# schema.

_ADBC_MODE_MAP: dict[Mode, str] = {
    Mode.AUTO: "append",
    Mode.APPEND: "append",
    Mode.OVERWRITE: "replace",
    Mode.TRUNCATE: "append",  # we issue TRUNCATE ourselves first
    Mode.ERROR_IF_EXISTS: "create",
    Mode.IGNORE: "create_append",
    Mode.UPSERT: "append",  # handled via temp-table in _write_upsert
    Mode.MERGE: "append",
}


class Table(Tabular):
    """A single Postgres table — DDL, DML, and Arrow IO."""

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls) -> "MimeType | None":
        return POSTGRES_TABLE_MIME

    def __init__(
        self,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        *,
        service: Optional["Tables"] = None,
        executor: Optional["PostgresExecutor"] = None,
        connection: Optional["PostgresConnection"] = None,
        **kwargs: Any,
    ):
        # Tabular.__init__ wires the persist cache slot, the
        # ``_media_type``, and other shared bookkeeping.
        super().__init__(**kwargs)
        if executor is None and service is not None:
            executor = service.executor
        if executor is None:
            raise ValueError(
                "Table requires an executor (or a service that carries one)."
            )
        self.service = service
        self.executor = executor
        self._connection = connection
        if not table_name:
            raise ValueError("Table requires a non-empty table_name")
        self.catalog_name = catalog_name
        self.schema_name = schema_name or DEFAULT_SCHEMA
        self.table_name = table_name
        self._cached_columns: Optional[List[Column]] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_(
        cls,
        obj: "Table | str",
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        service: Optional["Tables"] = None,
        executor: Optional["PostgresExecutor"] = None,
    ) -> "Table":
        """Coerce a dotted name / existing Table into a :class:`Table`."""
        if isinstance(obj, cls):
            if (catalog_name is None and schema_name is None and table_name is None):
                return obj
            return cls(
                catalog_name=catalog_name or obj.catalog_name,
                schema_name=schema_name or obj.schema_name,
                table_name=table_name or obj.table_name,
                service=service or obj.service,
                executor=executor or obj.executor,
                connection=obj._connection,
            )
        location = obj if isinstance(obj, str) else None
        c, s, t = parse_dotted_name(
            location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )
        if not t:
            raise ValueError(
                f"Cannot resolve a table name from {obj!r}; pass "
                "table_name= or a dotted ``schema.table`` location."
            )
        return cls(
            catalog_name=c,
            schema_name=s or DEFAULT_SCHEMA,
            table_name=t,
            service=service,
            executor=executor,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def full_name(self, safe: bool = False) -> str:
        """Identity name including catalog when set.

        Used for logging / repr / cross-backend identity. **Not** used
        for SQL emission — Postgres can't qualify across databases in
        a single statement, so DDL/DML uses :meth:`qualified_name`
        (just ``schema.table``).
        """
        parts = [self.schema_name, self.table_name]
        if self.catalog_name:
            parts = [self.catalog_name, *parts]
        if safe:
            return quote_qualified_ident(parts)
        return ".".join(parts)

    def qualified_name(self, safe: bool = True) -> str:
        """``schema.table`` — the SQL-emission form for DDL/DML.

        Postgres connections are bound to a single database, so
        cross-catalog qualification in a single statement isn't
        possible. Catalog routing happens at the
        :class:`PostgresConnection` level instead.
        """
        parts = [self.schema_name, self.table_name]
        if safe:
            return quote_qualified_ident(parts)
        return ".".join(parts)

    def __repr__(self) -> str:
        return f"PostgresTable<{self.full_name()!r}>"

    def __str__(self) -> str:
        return self.full_name()

    # ------------------------------------------------------------------
    # Connection routing
    # ------------------------------------------------------------------

    @property
    def connection(self) -> "PostgresConnection":
        if self._connection is not None:
            return self._connection
        return self.executor.connection

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    @property
    def schema(self) -> "PgSchema":
        """Navigate up to the parent :class:`Schema`."""
        from .schema import Schema as _Schema
        return _Schema(
            executor=self.executor,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )

    # ------------------------------------------------------------------
    # Existence / lifecycle
    # ------------------------------------------------------------------

    @property
    def exists(self) -> bool:
        """``True`` iff the table is reachable via ``information_schema``."""
        cursor = self.connection.psycopg_cursor()
        try:
            cursor.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = %s LIMIT 1",
                (self.schema_name, self.table_name),
            )
            return cursor.fetchone() is not None
        finally:
            cursor.close()

    def create(
        self,
        definition: pa.Schema | pa.Field | DataSchema | Any,
        *,
        if_not_exists: bool = True,
        primary_key: Optional[Sequence[str]] = None,
        comment: Optional[str] = None,
    ) -> "Table":
        """``CREATE TABLE`` from a :class:`pyarrow.Schema` / yggdrasil schema.

        Existing tables are left alone when ``if_not_exists=True``;
        otherwise the call raises if the target is taken.
        """
        if isinstance(definition, DataSchema):
            arrow_schema = definition.to_arrow_schema()
        elif isinstance(definition, pa.Schema):
            arrow_schema = definition
        elif isinstance(definition, pa.Field):
            arrow_schema = pa.schema([definition])
        else:
            from yggdrasil.arrow.cast import any_to_arrow_schema
            arrow_schema = any_to_arrow_schema(definition)

        columns = arrow_schema_to_postgres_columns(arrow_schema)
        column_clauses = list(columns)
        if primary_key:
            pk_cols = ", ".join(quote_ident(c) for c in primary_key)
            column_clauses.append(f"PRIMARY KEY ({pk_cols})")

        head = "CREATE TABLE IF NOT EXISTS" if if_not_exists else "CREATE TABLE"
        ddl = (
            f"{head} {self.qualified_name()} (\n  "
            + ",\n  ".join(column_clauses)
            + "\n)"
        )
        self.executor.sql(ddl, prefer_arrow=False)
        if comment:
            self.set_comment(comment)
        self._cached_columns = None
        return self

    def ensure_created(
        self,
        definition: Any,
        *,
        primary_key: Optional[Sequence[str]] = None,
        comment: Optional[str] = None,
    ) -> "Table":
        """Create the table if it does not already exist."""
        if not self.exists:
            self.create(
                definition=definition,
                if_not_exists=True,
                primary_key=primary_key,
                comment=comment,
            )
        return self

    def delete(
        self,
        *,
        if_exists: bool = True,
        cascade: bool = False,
    ) -> "Table":
        """``DROP TABLE`` — idempotent by default."""
        head = "DROP TABLE IF EXISTS" if if_exists else "DROP TABLE"
        tail = " CASCADE" if cascade else ""
        self.executor.sql(f"{head} {self.qualified_name()}{tail}", prefer_arrow=False)
        self._cached_columns = None
        return self

    drop = delete

    def truncate(self, *, cascade: bool = False, restart_identity: bool = False) -> "Table":
        """``TRUNCATE TABLE`` — wipe rows in place."""
        parts = [f"TRUNCATE TABLE {self.qualified_name()}"]
        if restart_identity:
            parts.append("RESTART IDENTITY")
        if cascade:
            parts.append("CASCADE")
        self.executor.sql(" ".join(parts), prefer_arrow=False)
        return self

    def rename(self, new_name: str) -> "Table":
        """``ALTER TABLE … RENAME TO …`` — schema-local rename."""
        new_name = (new_name or "").strip().strip('"')
        if not new_name:
            raise ValueError("Cannot rename table to an empty name")
        if new_name == self.table_name:
            return self
        self.executor.sql(
            f"ALTER TABLE {self.qualified_name()} RENAME TO {quote_ident(new_name)}",
            prefer_arrow=False,
        )
        self.table_name = new_name
        self._cached_columns = None
        return self

    def set_comment(self, comment: Optional[str]) -> "Table":
        """``COMMENT ON TABLE … IS …`` — set or clear the table comment."""
        from .sql_utils import sql_literal
        value = "NULL" if comment is None else sql_literal(comment)
        self.executor.sql(
            f"COMMENT ON TABLE {self.qualified_name()} IS {value}",
            prefer_arrow=False,
        )
        return self

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def columns(self, *, refresh: bool = False) -> List[Column]:
        """Return the column list from ``information_schema.columns``.

        Cached on the instance until :attr:`_cached_columns` is reset
        (rename / create / explicit ``refresh=True``).
        """
        if self._cached_columns is not None and not refresh:
            return self._cached_columns
        cursor = self.connection.psycopg_cursor()
        try:
            cursor.execute(
                """
                SELECT column_name, data_type, is_nullable, column_default,
                       ordinal_position, udt_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (self.schema_name, self.table_name),
            )
            rows = cursor.fetchall()
        finally:
            cursor.close()
        out: list[Column] = []
        for name, data_type, is_nullable, default, ordinal, udt_name in rows:
            # ``data_type`` returns "USER-DEFINED" for enum / domain
            # types — fall back to the udt_name for those.
            declared = data_type if data_type and data_type != "USER-DEFINED" else udt_name
            out.append(
                Column(
                    name=name,
                    data_type=declared,
                    nullable=(is_nullable == "YES"),
                    default=default,
                    ordinal_position=ordinal,
                )
            )
        self._cached_columns = out
        return out

    def _collect_schema(self, options: O) -> DataSchema:
        """Build a yggdrasil :class:`Schema` from :meth:`columns`."""
        if self._cached_schema is not None:
            return self._cached_schema
        fields = [c.to_arrow_field() for c in self.columns()]
        if not fields:
            self._cached_schema = DataSchema.empty()
        else:
            self._cached_schema = DataSchema.from_arrow(pa.schema(fields))
        return self._cached_schema

    # ------------------------------------------------------------------
    # Tabular — read
    # ------------------------------------------------------------------

    def _select_text(
        self,
        *,
        column_names: Optional[Sequence[str]] = None,
        row_limit: Optional[int] = None,
    ) -> str:
        if column_names:
            cols_sql = ", ".join(quote_ident(c) for c in column_names)
        else:
            cols_sql = "*"
        sql = f"SELECT {cols_sql} FROM {self.qualified_name()}"
        if row_limit:
            sql += f" LIMIT {int(row_limit)}"
        return sql

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Stream ``SELECT *`` as Arrow record batches.

        Honors :attr:`CastOptions.column_names` to select a subset
        (avoids over-fetching) and :attr:`CastOptions.row_size` to
        rechunk the resulting table — the wire returns a single Arrow
        table from ADBC; rechunking is purely for downstream
        consumers.
        """
        column_names = options.select_source_column_names() or None
        row_limit = None
        # ``row_size`` is the *batch* size, not a limit — keep it
        # separate from any caller-supplied LIMIT clause inside
        # ``options.text`` (which we don't take here).
        sql = self._select_text(column_names=column_names, row_limit=row_limit)
        prepared = PostgresPreparedStatement(text=sql)
        result = self.executor.execute(prepared, wait=True, raise_error=True)
        try:
            row_size = getattr(options, "row_size", None) or None
            for batch in result.read_arrow_batches(options=options):
                if row_size:
                    # ``read_arrow_batches`` already honors row_size
                    # via the inherited rechunker; this fallback is
                    # defensive against drivers that hand us a single
                    # giant batch.
                    yield from _rechunk_batch(batch, row_size)
                else:
                    yield batch
        finally:
            result.close()

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Bulk-write Arrow batches into this table.

        Picks the matching ADBC ingest mode from
        :attr:`CastOptions.mode`. UPSERT / MERGE go through a temp-
        table dance because ADBC ingest doesn't natively support
        ``ON CONFLICT``.
        """
        mode = options.mode
        if mode in (Mode.UPSERT, Mode.MERGE):
            return self._write_upsert(batches, options)

        if mode == Mode.TRUNCATE:
            if self.exists:
                self.truncate()
            ingest_mode = "append"
        else:
            ingest_mode = _ADBC_MODE_MAP.get(mode, "append")

        # Collect into an Arrow table — ADBC ingest takes a single
        # table or RecordBatchReader. RecordBatchReader streams the
        # batches without materialising the whole table.
        reader = _to_arrow_reader(batches, options)
        self._adbc_ingest(reader, mode=ingest_mode)
        self._cached_columns = None

    # ------------------------------------------------------------------
    # ADBC ingest
    # ------------------------------------------------------------------

    def _adbc_ingest(self, reader: pa.RecordBatchReader, *, mode: str) -> None:
        """Run ``cursor.adbc_ingest`` against this table.

        Falls back to a row-by-row INSERT if the ADBC driver isn't
        available — slower, but keeps a pure-psycopg install
        functional. The fallback is deliberately noisy at INFO level
        so deploys notice when they've forgotten to install
        ``adbc_driver_postgresql``.
        """
        if not self.connection.has_adbc:
            logger.info(
                "ADBC driver unavailable; falling back to psycopg row "
                "insert into %s. Install ``adbc-driver-postgresql`` for "
                "the Arrow-native fast path.",
                self.full_name(),
            )
            return self._psycopg_insert(reader, mode=mode)

        cursor = self.connection.adbc_cursor()
        try:
            cursor.adbc_ingest(
                table_name=self.table_name,
                data=reader,
                mode=mode,
                db_schema_name=self.schema_name,
            )
        finally:
            cursor.close()

    def _psycopg_insert(self, reader: pa.RecordBatchReader, *, mode: str) -> None:
        """Row-fallback for environments without the ADBC driver."""
        if mode == "replace":
            self.delete(if_exists=True)
            # Caller is responsible for create; replace expects the
            # ADBC path to recreate from the Arrow schema, but in the
            # fallback we expect the table to exist already.
        elif mode == "create" and self.exists:
            raise RuntimeError(
                f"Table {self.full_name()} already exists and mode='create'."
            )
        elif mode == "create_append" and not self.exists:
            # Same problem — fallback can't create from Arrow schema
            # alone without a yggdrasil-side schema build. Surface
            # clearly rather than silently doing the wrong thing.
            raise RuntimeError(
                f"Table {self.full_name()} does not exist; create() it first or "
                "install adbc-driver-postgresql for create-on-ingest support."
            )

        cursor = self.connection.psycopg_cursor()
        try:
            for batch in reader:
                rows = batch.to_pylist()
                if not rows:
                    continue
                columns = batch.schema.names
                cols_sql = ", ".join(quote_ident(c) for c in columns)
                placeholders = ", ".join(["%s"] * len(columns))
                sql = (
                    f"INSERT INTO {self.qualified_name()} ({cols_sql}) "
                    f"VALUES ({placeholders})"
                )
                cursor.executemany(sql, [tuple(r[c] for c in columns) for r in rows])
        finally:
            cursor.close()

    def _write_upsert(self, batches: Iterable[pa.RecordBatch], options: O) -> None:
        """UPSERT via temp-table + ``INSERT … ON CONFLICT … DO UPDATE``.

        ``options.match_by`` (or the target's primary-key columns)
        define the conflict target. Update columns default to all
        non-key columns; override with ``CastOptions.update_column_names``.
        """
        match_by = options.match_by_keys or self._primary_key_columns()
        if not match_by:
            raise ValueError(
                f"UPSERT into {self.full_name()} requires match_by or a "
                "primary key on the target table."
            )

        reader = _to_arrow_reader(batches, options)
        # Stage into a real (unlogged) temp table — cheaper than an
        # ON COMMIT DROP temp because ADBC ingest can't target temp
        # tables in some driver versions, and a unique stage name
        # keeps parallel calls safe.
        import os as _os
        stage_name = f"_ygg_pg_stage_{_os.urandom(6).hex()}"
        stage_qualified = f"{quote_ident(self.schema_name)}.{quote_ident(stage_name)}"
        target_qualified = self.qualified_name()

        self.executor.sql(
            f"CREATE UNLOGGED TABLE {stage_qualified} (LIKE {target_qualified} "
            "INCLUDING DEFAULTS)",
            prefer_arrow=False,
        )
        try:
            stage = Table(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                table_name=stage_name,
                executor=self.executor,
            )
            stage._adbc_ingest(reader, mode="append")
            update_cols = options.update_column_names or [
                c.name for c in self.columns() if c.name not in match_by
            ]
            on_conflict = ", ".join(quote_ident(c) for c in match_by)
            target_columns = [c.name for c in self.columns()]
            cols_sql = ", ".join(quote_ident(c) for c in target_columns)
            select_sql = ", ".join(quote_ident(c) for c in target_columns)
            if update_cols:
                set_clause = ", ".join(
                    f"{quote_ident(c)} = EXCLUDED.{quote_ident(c)}" for c in update_cols
                )
                conflict_clause = f"DO UPDATE SET {set_clause}"
            else:
                conflict_clause = "DO NOTHING"
            self.executor.sql(
                f"INSERT INTO {target_qualified} ({cols_sql}) "
                f"SELECT {select_sql} FROM {stage_qualified} "
                f"ON CONFLICT ({on_conflict}) {conflict_clause}",
                prefer_arrow=False,
            )
        finally:
            self.executor.sql(
                f"DROP TABLE IF EXISTS {stage_qualified}", prefer_arrow=False,
            )
        self._cached_columns = None

    def _primary_key_columns(self) -> list[str]:
        cursor = self.connection.psycopg_cursor()
        try:
            cursor.execute(
                """
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = (
                    SELECT format('%I.%I', %s, %s)::regclass
                )
                AND i.indisprimary
                ORDER BY array_position(i.indkey, a.attnum)
                """,
                (self.schema_name, self.table_name),
            )
            return [r[0] for r in cursor.fetchall()]
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # Insert convenience
    # ------------------------------------------------------------------

    def insert_into(
        self,
        data: Any,
        *,
        mode: Mode | str | None = None,
        match_by: Optional[Sequence[str]] = None,
        update_column_names: Optional[Sequence[str]] = None,
        cast_options: Optional[CastOptions] = None,
    ) -> "Table":
        """High-level insert: arrow / polars / pandas / dict / list → table.

        Lifts the input through :func:`yggdrasil.arrow.cast.any_to_arrow_table`
        and routes to :meth:`write_arrow_table` with the resolved
        :class:`Mode`.
        """
        from yggdrasil.arrow.cast import any_to_arrow_table

        options = self.check_options(
            cast_options,
            mode=mode if mode is not None else Mode.AUTO,
            match_by=list(match_by) if match_by else None,
            update_column_names=list(update_column_names) if update_column_names else None,
        )
        table = any_to_arrow_table(data, options)
        self.write_arrow_table(table, options=options)
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_arrow_reader(
    batches: Iterable[pa.RecordBatch],
    options: O,
) -> pa.RecordBatchReader:
    """Coerce ``batches`` into a :class:`pyarrow.RecordBatchReader`.

    ADBC ingest is happiest with a streaming reader — it pulls
    batches lazily without materialising the full table. We peek
    the first batch to discover the schema, then chain the
    remainder back through a generator.
    """
    iterator = iter(batches)
    try:
        first = next(iterator)
    except StopIteration:
        # Empty input — emit a zero-batch reader against whatever
        # target schema we can find. Fall back to a pa.null() schema
        # if even that's missing; ADBC ingest will reject the call,
        # which is the right answer (caller wrote nothing into a
        # table whose shape can't be inferred).
        target = (
            options.target_schema.to_arrow_schema()
            if options and options.target_schema is not None
            else pa.schema([])
        )
        return pa.RecordBatchReader.from_batches(target, iter(()))

    schema = first.schema

    def gen():
        yield first
        for b in iterator:
            yield b

    return pa.RecordBatchReader.from_batches(schema, gen())


def _rechunk_batch(batch: pa.RecordBatch, row_size: int) -> Iterator[pa.RecordBatch]:
    if batch.num_rows <= row_size:
        yield batch
        return
    offset = 0
    while offset < batch.num_rows:
        yield batch.slice(offset, row_size)
        offset += row_size
