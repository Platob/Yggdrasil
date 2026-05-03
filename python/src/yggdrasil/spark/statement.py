"""Spark-side :class:`PreparedStatement` and :class:`StatementResult`.

Spark execution is synchronous from the SQL-submission perspective:
``session.sql(text)`` returns a :class:`pyspark.sql.DataFrame` immediately,
and any heavy work is deferred to the lazy frame.  That means the result
handle is *always* terminal after :meth:`SparkStatementResult.start`:
``done`` is ``True``, ``failed`` is ``True`` only if ``session.sql``
itself raised.

The polling loop on :class:`StatementBatch.wait` therefore degenerates to
a no-op for Spark statements — exactly what we want when mixing Spark +
warehouse statements in the same batch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Optional

import pyarrow as pa

from yggdrasil.data import Schema
from yggdrasil.data.options import CastOptions
from yggdrasil.data.record import Record
from yggdrasil.data.statement import (
    PreparedStatement,
    StatementResult,
    StatementBatch,
)
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.environ import PyEnv
from yggdrasil.io.buffer.base import O
from yggdrasil.io.enums import MimeType, MimeTypes

if TYPE_CHECKING:
    from yggdrasil.spark.executor import SparkStatementExecutor
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

__all__ = [
    "SparkPreparedStatement",
    "SparkStatementResult",
    "SparkStatementBatch",
]


# ---------------------------------------------------------------------------
# SparkPreparedStatement
# ---------------------------------------------------------------------------


class SparkPreparedStatement(PreparedStatement):
    """A :class:`PreparedStatement` that may carry an explicit SparkSession.

    Pinning the session lets a caller force a specific session for one
    call without rebinding the executor.  When ``spark_session`` is
    ``None`` the executor / result resolves one through
    :class:`yggdrasil.environ.PyEnv`.

    ``row_limit`` is applied via ``df.limit(...)`` after ``session.sql``.

    Inherits ``retry`` (a :class:`WaitingConfig`) from
    :class:`PreparedStatement`.  Default is ``None`` (not retryable).
    Spark errors raised inside ``session.sql`` get captured on the
    result; if they match
    :attr:`SparkStatementResult._TRANSIENT_ERROR_PATTERNS`, the statement
    is auto-promoted (``self.statement.retry`` is filled in by
    :meth:`SparkStatementResult.default_retry`) so
    :meth:`StatementResult.retry` will re-run it.
    """

    spark_session: Optional["SparkSession"] = None
    row_limit: Optional[int] = None

    def __init__(
        self,
        text: str = "",
        *,
        key: Optional[str] = None,
        retry: Optional[WaitingConfigArg] = None,
        spark_session: Optional["SparkSession"] = None,
        row_limit: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(text=text, key=key, retry=retry)
        self.spark_session = spark_session
        self.row_limit = row_limit

    def get_or_create_spark_session(self) -> "SparkSession":
        """Resolve the bound session, creating one via PyEnv if absent."""
        if self.spark_session is None:
            self.spark_session = PyEnv.spark_session(create=True, import_error=True)
        return self.spark_session


# ---------------------------------------------------------------------------
# SparkStatementResult
# ---------------------------------------------------------------------------


class SparkStatementResult(StatementResult[SparkPreparedStatement]):
    """Synchronous result wrapping a Spark :class:`DataFrame`.

    Always terminal once :meth:`start` has run.  ``session.sql`` errors
    propagate to the caller before a result is constructed (when
    ``raise_error=True``); a downstream Spark action that fails later
    surfaces directly when the caller touches the persisted frame.
    """

    _PREPARED_STATEMENT_CLASS = SparkPreparedStatement

    def __init__(
        self,
        statement: SparkPreparedStatement,
        executor: Optional["SparkStatementExecutor"] = None,
        **kwargs: Any,
    ):
        # Importing here keeps the module importable without pyspark.
        if executor is None:
            from yggdrasil.spark.executor import SparkStatementExecutor as _Exec
            executor = _Exec()
        super().__init__(statement=statement, executor=executor, **kwargs)
        # Result-local flags — assigned through plain attribute access; the
        # base class is not frozen so this is fine.
        self._started: bool = False
        self._failure: Optional[BaseException] = None

    def _collect_schema(self, options: CastOptions) -> Schema:
        if self._cached_schema is None:
            self._cached_schema = Schema.from_(self.spark_dataframe)
        return self._cached_schema

    @classmethod
    def default_mime_type(cls) -> "MimeType | None":
        return MimeTypes.SPARK_SQL_STATEMENT

    # -------------------------------------------------------------------------
    # Transient-error patterns (Spark-specific)
    # -------------------------------------------------------------------------
    #
    # Spark surfaces Delta concurrent-append etc. as
    # ``pyspark.sql.utils.AnalysisException`` (or
    # ``pyspark.errors.exceptions.captured.AnalysisException`` on newer
    # PySpark) whose ``str(exc)`` carries the same DELTA_* error codes
    # the warehouse path sees.  We also include ``Py4JJavaError`` markers
    # since Spark wraps Java-side exceptions through Py4J.

    _TRANSIENT_ERROR_PATTERNS = (
        # Delta concurrency family — same codes as the warehouse path.
        r"DELTA_CONCURRENT_APPEND",
        r"ConcurrentAppendException",
        r"DELTA_CONCURRENT_DELETE_READ",
        r"DELTA_CONCURRENT_DELETE_DELETE",
        r"DELTA_CONCURRENT_WRITE",
        r"DELTA_METADATA_CHANGED",
        # Py4J-wrapped Java-side exception classes for the same family.
        r"ConcurrentDeleteReadException",
        r"ConcurrentDeleteDeleteException",
        r"ConcurrentWriteException",
        r"MetadataChangedException",
        # Sentinel string Databricks/Delta append to several transient errors.
        r"Please retry the operation",
    )

    def _failure_message(self) -> str:
        """Stringify ``self._failure`` for transient-pattern matching.

        Walks the cause chain (``__cause__`` + ``__context__``) so a
        Py4JJavaError wrapping a Spark exception still surfaces the
        underlying message.  Returns ``""`` when no failure is recorded.
        """
        if self._failure is None:
            return ""
        parts: list[str] = []
        seen: set[int] = set()
        exc: Optional[BaseException] = self._failure
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            parts.append(f"{type(exc).__name__}: {exc}")
            # Some PySpark wrappers expose .desc / .stackTrace via Py4J
            # gateway that aren't surfaced by str(exc); try a couple of
            # common attributes defensively.
            for attr in ("desc", "java_message", "errorClass"):
                v = getattr(exc, attr, None)
                if v:
                    parts.append(str(v))
            exc = exc.__cause__ or exc.__context__
        return " | ".join(parts)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    @property
    def started(self) -> bool:
        """True once :meth:`start` has run (terminal in either direction)."""
        return self._started

    @property
    def done(self) -> bool:
        # Once start() has run we're terminal: either a frame is persisted
        # or _failure is set.  Pre-start, done=False so wait() will loop —
        # refresh_status() is a no-op so wait() will return after the first
        # call to start() that the loop induces (start is idempotent, but
        # nothing in the base contract calls start from wait()).  Callers
        # are expected to call start() before wait() — same as the warehouse
        # path.
        return self._started

    @property
    def failed(self) -> bool:
        return self._failure is not None

    def _raise_for_status(self) -> None:
        # Auto-promote happens in the base raise_for_status() before this
        # hook fires; we just re-raise the captured exception.
        if self._failure is not None:
            raise self._failure

    def refresh_status(self) -> None:
        """No-op — Spark execution is synchronous, no remote state to poll."""
        return None

    # -------------------------------------------------------------------------
    # Persisted DataFrame
    # -------------------------------------------------------------------------

    @property
    def spark_dataframe(self) -> Optional["DataFrame"]:
        """The materialized Spark frame, or ``None`` if not yet started."""
        return self._persisted_data

    @property
    def cached(self) -> bool:
        """True once a frame has been persisted (= post-start, success)."""
        return self._persisted_data is not None

    def persist(
        self,
        engine: Literal["arrow", "polars", "spark", "auto"] = "auto",
        *,
        data: Any | None = None,
    ) -> "SparkStatementResult":
        """Stash a materialized frame on this result.

        ``data`` overrides the persisted frame; otherwise this is a no-op
        (Spark caches lazily on the frame itself, not on this handle).
        """
        if data is not None:
            from yggdrasil.spark.cast import any_to_spark_dataframe
            self._persisted_data = any_to_spark_dataframe(data)
        return self

    def unpersist(self) -> None:
        """Drop the cached frame reference; Spark's cache is unaffected."""
        self._persisted_data = None

    # -------------------------------------------------------------------------
    # Spark
    # -------------------------------------------------------------------------

    def _read_spark_frame(self, options: CastOptions) -> "DataFrame":
        # Persisted data is already a Spark frame — return it under the
        # caller's cast options without round-tripping through Arrow.
        if self._persisted_data is None:
            raise RuntimeError(
                "Cannot read Spark frame from a non-started Spark statement; "
                "call start() first."
            )
        return options.cast_spark_tabular(self._persisted_data)

    # -------------------------------------------------------------------------
    # Arrow
    # -------------------------------------------------------------------------

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        if self._persisted_data is None:
            raise RuntimeError(
                "Cannot read Arrow batches from a non-started Spark statement; "
                "call start() first."
            )
        return (
            options.cast_spark_tabular(self._persisted_data)
            .toArrow()
            .to_batches(max_chunksize=options.row_size)
        )

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        raise NotImplementedError("Cannot write to Spark via this interface")

    def _read_records(self, options: O) -> "Iterator[Any]":
        schema = self._collect_schema(options)

        yield from Record.from_spark_frame(
            options.cast_spark_tabular(self._persisted_data),
            schema=schema,
        )

    # -------------------------------------------------------------------------
    # Submit / cancel
    # -------------------------------------------------------------------------

    def start(
        self,
        reset: bool = False,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        spark_session: Optional["SparkSession"] = None,
        **kwargs: Any,
    ) -> "SparkStatementResult":
        """Run ``session.sql(text)`` and stash the resulting DataFrame.

        Idempotent on already-started results: returns ``self`` without
        re-running unless ``reset=True``, in which case the prior
        DataFrame and failure state are cleared and the statement re-runs.
        ``reset=True`` is the path :meth:`StatementResult.retry` drives.

        ``wait`` is accepted for signature compatibility but is irrelevant —
        Spark is already synchronous.
        """
        if self._started and not reset:
            if raise_error:
                self.raise_for_status()
            return self

        if reset:
            # Clear prior submission state before re-running.  Note: we
            # don't drop a *successful* DataFrame from a prior attempt
            # because retry() only reaches here on failure (the loop
            # short-circuits when done and not failed), but the
            # invariant of reset=True is "as if start was never called".
            self._failure = None
            self._persisted_data = None
            self._cached_schema = None
            self._started = False

        session = spark_session or self.statement.spark_session
        if session is None:
            session = PyEnv.spark_session(create=True, import_error=True)

        try:
            df = session.sql(self.statement.text)
            row_limit = self.statement.row_limit
            if row_limit:
                df = df.limit(row_limit)
            self.persist(data=df)
        except BaseException as exc:
            self._failure = exc
            self._started = True
            if raise_error:
                raise
            return self

        self._started = True
        return self

    def cancel(self) -> "SparkStatementResult":
        """No-op: Spark SQL is synchronous, there is nothing to cancel.

        Cancelling a running Spark *job* needs the caller's
        :class:`SparkContext` and is out of scope here.
        """
        return self


# ---------------------------------------------------------------------------
# SparkStatementBatch
# ---------------------------------------------------------------------------


class SparkStatementBatch(StatementBatch[SparkPreparedStatement, SparkStatementResult]):
    """A batch of Spark statements.

    The base contract just works for Spark — every result is terminal as
    soon as it's submitted, so :meth:`StatementBatch.wait` is effectively
    a no-op.  Override :meth:`_coerce` to coerce strings into
    :class:`SparkPreparedStatement`.
    """

    def __init__(
        self,
        executor: Optional["SparkStatementExecutor"] = None,
        statements: Optional[Iterable["SparkPreparedStatement | str"]] = None,
        *,
        parallel: int = 1,
    ):
        if executor is None:
            from yggdrasil.spark.executor import SparkStatementExecutor as _Exec
            executor = _Exec.current()
        super().__init__(executor=executor, statements=statements, parallel=parallel)

    def _coerce(
        self,
        statement: "SparkPreparedStatement | str",
    ) -> SparkPreparedStatement:
        return SparkPreparedStatement.from_(statement)