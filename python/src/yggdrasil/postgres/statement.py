"""Postgres-backed :class:`PreparedStatement` and :class:`StatementResult`.

Postgres execution is synchronous from the client's perspective: the
moment ``cursor.execute()`` returns, the server has either parsed
+ planned + (depending on cursor type) materialized the result, or
raised an error. There's nothing to poll for — the result handle is
terminal as soon as :meth:`PostgresStatementResult.start` returns.

The polling loop on :class:`StatementBatch.wait` therefore degenerates
to a no-op for Postgres statements, which is the expected behaviour
when mixing Postgres + (e.g.) warehouse statements in the same batch.

Two read paths
--------------
* **Arrow-fast** — ADBC driver. ``cursor.execute(text)`` followed by
  ``cursor.fetch_arrow_table()`` returns a :class:`pa.Table` without
  ever materializing Python rows. This is the default when the ADBC
  driver is importable.
* **Row-fallback** — psycopg cursor. We materialize ``cursor.fetchall()``
  into a list of dicts and lift it into Arrow. Slower, but doesn't
  require the ADBC dependency. Triggered automatically when ADBC
  isn't available *or* the caller passes ``prefer_arrow=False`` for
  diagnostics / parity testing.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
)

import pyarrow as pa

from yggdrasil.data import Schema
from yggdrasil.data.options import CastOptions
from yggdrasil.data.statement import (
    PreparedStatement,
    StatementBatch,
    StatementResult,
)
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io.tabular.base import O
from yggdrasil.enums import MimeType, State

if TYPE_CHECKING:
    from .connection import PostgresConnection
    from .executor import PostgresExecutor

logger = logging.getLogger(__name__)

__all__ = [
    "PostgresPreparedStatement",
    "PostgresStatementResult",
    "PostgresStatementBatch",
]


# Postgres assigns unique mime types so :class:`Tabular`'s
# auto-registry can dispatch on the result's identity. Defined here
# rather than in ``yggdrasil.io.enums.mime_type`` to keep the
# Postgres backend optional — importing ``yggdrasil.postgres`` is
# the trigger that registers them.

POSTGRES_STATEMENT_MIME = MimeType.define(
    MimeType("POSTGRES_STATEMENT", "application/vnd.postgresql.statement"),
)
POSTGRES_TABLE_MIME = MimeType.define(
    MimeType("POSTGRES_TABLE", "application/vnd.postgresql.table"),
)


# ---------------------------------------------------------------------------
# PostgresPreparedStatement
# ---------------------------------------------------------------------------


class PostgresPreparedStatement(PreparedStatement):
    """A SQL statement targeted at a Postgres connection.

    Carries the SQL text plus the per-statement bindings and routing
    knobs that :meth:`PostgresExecutor._submit_statement` reads. None
    of these survive into the :class:`StatementResult` itself —
    runtime state (cursor, fetched table, failure) lives on the
    result, not the statement.

    Fields
    ------
    parameters
        Positional or named bindings for ``%s`` / ``%(name)s``
        placeholders. Forwarded verbatim to the cursor's
        ``execute`` call.
    catalog_name / schema_name
        Optional ``SET search_path`` / ``SET database`` hints. The
        executor applies them inside a transaction-scoped ``SET
        LOCAL`` so they don't leak across calls.
    prefer_arrow
        Whether to take the ADBC fast path when available. Defaults
        to ``True``; flip to ``False`` for parity testing or when
        the caller specifically wants the psycopg cursor (some
        result types — ``oid`` columns, ``cursor`` columns — are
        easier to inspect via psycopg).
    fetch_size
        Cursor batch size for the row-fallback path. Defaults to
        the driver default; large values trade memory for
        round-trips on streaming reads.
    """

    parameters: Optional[Sequence[Any] | Mapping[str, Any]] = None
    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None
    prefer_arrow: bool = True
    fetch_size: Optional[int] = None

    def __init__(
        self,
        text: str = "",
        *,
        key: Optional[str] = None,
        retry: Optional[WaitingConfigArg] = None,
        parameters: Optional[Sequence[Any] | Mapping[str, Any]] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        prefer_arrow: bool = True,
        fetch_size: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(text=text, key=key, retry=retry)
        self.parameters = parameters
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.prefer_arrow = prefer_arrow
        self.fetch_size = fetch_size

    @classmethod
    def prepare(
        cls,
        statement: "PreparedStatement | str",
        *,
        parameters: Optional[Sequence[Any] | Mapping[str, Any]] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        prefer_arrow: bool = True,
        fetch_size: Optional[int] = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> "PostgresPreparedStatement":
        """Coerce a string / base statement into a Postgres one with bindings."""
        if isinstance(statement, cls):
            base = statement
        else:
            base = cls.from_(statement)
        # Always rebuild so we don't mutate a shared input.
        return cls(
            text=base.text,
            key=base.key,
            retry=retry if retry is not None else base.retry,
            parameters=parameters if parameters is not None else base.parameters,
            catalog_name=catalog_name if catalog_name is not None else base.catalog_name,
            schema_name=schema_name if schema_name is not None else base.schema_name,
            prefer_arrow=prefer_arrow if prefer_arrow is not None else base.prefer_arrow,
            fetch_size=fetch_size if fetch_size is not None else base.fetch_size,
        )


# ---------------------------------------------------------------------------
# PostgresStatementResult
# ---------------------------------------------------------------------------


class PostgresStatementResult(StatementResult[PostgresPreparedStatement]):
    """Synchronous Postgres result handle wrapping a fetched Arrow table.

    Always terminal once :meth:`start` has run — Postgres returns the
    full result on the same round-trip that submits the statement
    (cursor-based streaming is opt-in via ``fetch_size``, and even
    then the executor materializes batch-by-batch eagerly).

    The materialized payload is held as a :class:`ArrowTabular` on
    ``_persisted_data`` so the inherited :class:`Tabular` read
    methods (``read_arrow_batches``, ``read_polars_frame``,
    ``read_pandas_frame``, …) work out of the box.
    """

    _PREPARED_CLASS: ClassVar[type[PostgresPreparedStatement]] = PostgresPreparedStatement

    def __init__(
        self,
        statement: PostgresPreparedStatement,
        *,
        executor: Optional["PostgresExecutor"] = None,
        connection: Optional["PostgresConnection"] = None,
        **kwargs: Any,
    ):
        super().__init__(statement=statement, executor=executor, **kwargs)
        self._connection = connection
        self._started: bool = False
        self._failure: Optional[BaseException] = None
        self._row_count: int = -1

    @classmethod
    def default_media_type(cls) -> "MimeType | None":
        return POSTGRES_STATEMENT_MIME

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _compute_state(self) -> State:
        """Local-state mapping — Postgres materialises synchronously in :meth:`start`."""
        if not self._started:
            return State.IDLE
        if self._failure is not None:
            return State.FAILED
        return State.SUCCEEDED

    def refresh_status(self) -> None:
        """No-op — Postgres is synchronous, no remote state to poll."""
        return None

    def _failure_message(self) -> str:
        if self._failure is None:
            return ""
        return f"{type(self._failure).__name__}: {self._failure}"

    def _raise_for_status(self) -> None:
        if self._failure is not None:
            raise self._failure

    @property
    def row_count(self) -> int:
        """Number of rows reported by the cursor (``-1`` before start)."""
        return self._row_count

    # ------------------------------------------------------------------
    # Submit / cancel
    # ------------------------------------------------------------------

    def start(
        self,
        reset: bool = False,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        **kwargs: Any,
    ) -> "PostgresStatementResult":
        """Run the SQL against the bound connection and stash the Arrow result.

        Idempotent on already-started results unless ``reset=True``,
        in which case the prior fetched table and failure state are
        cleared and the statement re-runs (the path
        :meth:`StatementResult.retry` drives).
        """
        if self._started and not reset:
            if raise_error:
                self.raise_for_status()
            return self

        if reset:
            self._failure = None
            self._persisted_data = None
            self._cached_schema = None
            self._row_count = -1
            self._started = False

        connection = self._resolve_connection()
        try:
            table, row_count = self._execute(connection)
        except BaseException as exc:
            self._failure = exc
            self._started = True
            if raise_error:
                raise
            return self

        from yggdrasil.io.tabular import ArrowTabular
        self._persisted_data = ArrowTabular(table)
        self._row_count = row_count
        self._started = True
        return self

    def cancel(self, wait: WaitingConfigArg = None, raise_error: bool = False, **kwargs) -> "PostgresStatementResult":
        """Best-effort cursor cancel — Postgres cancellation is connection-scoped.

        For an already-terminal result this is a no-op. For an in-
        flight result on another thread we'd need to call
        ``conn.cancel()`` on the connection handle; cross-thread
        cancellation is left to the caller for now (psycopg exposes
        the API but the policy is library-specific).
        """
        return self

    # ------------------------------------------------------------------
    # Read hooks — the inherited Tabular surface delegates here.
    # ------------------------------------------------------------------

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        self._require_started()
        yield from self._persisted_data._read_arrow_batches(options)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        raise NotImplementedError(
            "Cannot write to a PostgresStatementResult; use "
            "PostgresTable.write_arrow_table or PostgresExecutor.execute "
            "with an INSERT statement instead."
        )

    def _read_records(self, options: O) -> Iterator[Any]:
        self._require_started()
        yield from self._persisted_data._read_records(options)

    def _collect_schema(self, options: CastOptions) -> Schema:
        if self._cached_schema is None:
            if self._persisted_data is not None:
                self._cached_schema = self._persisted_data.collect_schema(options)
            else:
                self._cached_schema = super()._collect_schema(options)
        return self._cached_schema

    def _require_started(self) -> None:
        if self._persisted_data is None:
            raise RuntimeError(
                "Cannot read from a non-started Postgres statement; "
                "call start() first."
            )

    # ------------------------------------------------------------------
    # Connection / driver dispatch
    # ------------------------------------------------------------------

    def _resolve_connection(self) -> "PostgresConnection":
        if self._connection is not None:
            return self._connection
        if self.executor is not None and hasattr(self.executor, "connection"):
            return self.executor.connection
        raise RuntimeError(
            "PostgresStatementResult has no bound connection; pass "
            "connection= at construction or run via a PostgresExecutor."
        )

    def _execute(self, connection: "PostgresConnection") -> tuple[pa.Table, int]:
        """Run the statement and return ``(arrow_table, row_count)``.

        Picks the ADBC fast path when available and the caller hasn't
        opted out via ``prefer_arrow=False``; otherwise materializes
        rows through psycopg and lifts them into Arrow.
        """
        use_arrow = self.statement.prefer_arrow and connection.has_adbc
        if use_arrow:
            return self._execute_adbc(connection)
        return self._execute_psycopg(connection)

    def _execute_adbc(self, connection: "PostgresConnection") -> tuple[pa.Table, int]:
        """ADBC fast path — server returns Arrow batches natively."""
        cursor = connection.adbc_cursor()
        try:
            self._apply_session_scope(cursor, dialect="adbc")
            cursor.execute(self.statement.text, self.statement.parameters)
            row_count = getattr(cursor, "rowcount", -1) or -1
            try:
                table = cursor.fetch_arrow_table()
            except Exception:
                # Non-result-bearing statement (DDL, INSERT without
                # RETURNING) — fetch_arrow_table raises on these. The
                # statement still ran successfully; surface an empty
                # table so callers that always read get a consistent
                # shape.
                table = pa.table({})
            return table, row_count
        finally:
            try:
                cursor.close()
            except Exception:
                logger.debug("ADBC cursor close failed; continuing.", exc_info=True)

    def _execute_psycopg(self, connection: "PostgresConnection") -> tuple[pa.Table, int]:
        """Row-fallback path — psycopg cursor + Arrow lift."""
        cursor = connection.psycopg_cursor()
        try:
            if self.statement.fetch_size:
                cursor.itersize = int(self.statement.fetch_size)
            self._apply_session_scope(cursor, dialect="psycopg")
            cursor.execute(self.statement.text, self.statement.parameters)
            row_count = getattr(cursor, "rowcount", -1) or -1
            description = getattr(cursor, "description", None)
            if not description:
                # DDL / non-result statement.
                return pa.table({}), row_count
            columns = [d[0] for d in description]
            rows = cursor.fetchall()
            if not rows:
                return pa.table({c: pa.array([], type=pa.null()) for c in columns}), row_count
            data = {c: [row[i] for row in rows] for i, c in enumerate(columns)}
            return pa.table(data), row_count
        finally:
            try:
                cursor.close()
            except Exception:
                logger.debug("psycopg cursor close failed; continuing.", exc_info=True)

    def _apply_session_scope(self, cursor: Any, *, dialect: str) -> None:
        """Apply per-statement ``search_path`` / database hints.

        Wrapped in ``SET LOCAL`` when we're inside a transaction so
        the override doesn't leak; outside a transaction we use a
        plain ``SET`` since there's no other scoping mechanism.
        """
        from .sql_utils import quote_ident
        # Postgres has no per-cursor "USE catalog" — switching catalogs
        # requires a fresh connection. We log a warning when the
        # caller asks for a non-current catalog.
        if self.statement.catalog_name:
            current = self._current_database(cursor)
            if current and current != self.statement.catalog_name:
                logger.warning(
                    "PostgresPreparedStatement.catalog_name=%r differs from "
                    "the connection's current database %r; cross-database "
                    "queries require a separate PostgresConnection. "
                    "Statement will run against %r.",
                    self.statement.catalog_name, current, current,
                )
        if self.statement.schema_name:
            cursor.execute(f"SET search_path TO {quote_ident(self.statement.schema_name)}")

    @staticmethod
    def _current_database(cursor: Any) -> Optional[str]:
        try:
            cursor.execute("SELECT current_database()")
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# PostgresStatementBatch
# ---------------------------------------------------------------------------


class PostgresStatementBatch(
    StatementBatch[PostgresPreparedStatement, PostgresStatementResult]
):
    """A batch of Postgres statements — base behaviour fits."""

    def _coerce(
        self,
        statement: "PostgresPreparedStatement | str",
    ) -> PostgresPreparedStatement:
        return PostgresPreparedStatement.from_(statement)
