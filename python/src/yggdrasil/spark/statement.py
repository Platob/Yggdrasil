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
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Mapping, Optional

import pyarrow as pa

from yggdrasil.data import Schema
from yggdrasil.data.options import CastOptions
from yggdrasil.data.statement import (
    ExternalStatementData,
    PreparedStatement,
    StatementResult,
    StatementBatch,
)
from yggdrasil.io.tabular import Tabular
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.environ import PyEnv
from yggdrasil.io.tabular.base import O
from yggdrasil.data.enums import MimeType, MimeTypes

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
        external_data: Optional[
            Mapping[str, "ExternalStatementData | Tabular | str | tuple"]
        ] = None,
        **kwargs: Any,
    ):
        super().__init__(text=text, key=key, retry=retry, external_data=external_data)
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
        # Tracks (session, view_name) tuples for every temp view registered
        # by ``_register_external_views`` so ``clear_temporary_resources``
        # can drop them on the same session.  Keyed by the
        # ``ExternalStatementData.text_key`` so re-registration on retry
        # replaces rather than leaks.
        self._temp_views: dict[str, tuple["SparkSession", str]] = {}

    def _collect_schema(self, options: CastOptions) -> Schema:
        if self._cached_schema is None:
            self._cached_schema = Schema.from_(self.spark_dataframe)
        return self._cached_schema

    @classmethod
    def default_media_type(cls) -> "MimeType | None":
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
    #
    # ``cached`` / ``unpersist`` come from :class:`Tabular` —
    # ``_persisted_data`` (a :class:`SparkTabular` wrapper) is the
    # single source of truth for "was this statement materialised".
    # The :class:`Tabular` read paths delegate to it before the
    # private ``_read_*`` hooks fire, so we only need to fill in
    # ``persist`` and the convenience ``spark_dataframe`` accessor.

    @property
    def spark_dataframe(self) -> Optional["DataFrame"]:
        """The materialized Spark frame, or ``None`` if not yet started."""
        if self._persisted_data is None:
            return None
        return getattr(self._persisted_data, "frame", None)

    def persist(
        self,
        engine: Literal["arrow", "polars", "spark", "auto"] = "auto",
        *,
        data: Any | None = None,
    ) -> "SparkStatementResult":
        """Stash a materialised frame on this result as a :class:`SparkTabular`.

        ``data`` overrides the persisted frame; otherwise this is a no-op
        (Spark caches lazily on the frame itself, not on this handle).
        """
        if data is not None:
            from yggdrasil.io.tabular import SparkTabular
            from yggdrasil.spark.cast import any_to_spark_dataframe

            self._persisted_data = SparkTabular(any_to_spark_dataframe(data))
        return self

    # -------------------------------------------------------------------------
    # Read hooks — forward to the held SparkTabular when set so callers
    # that bypass the public API (or hit ``_read_records`` through the
    # base default) still see the materialised frame.
    # -------------------------------------------------------------------------

    def _read_spark_frame(self, options: CastOptions) -> "DataFrame":
        self._require_started()
        return self._persisted_data._read_spark_frame(options)

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        self._require_started()
        yield from self._persisted_data._read_arrow_batches(options)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        raise NotImplementedError("Cannot write to Spark via this interface")

    def _read_records(self, options: O) -> "Iterator[Any]":
        self._require_started()
        yield from self._persisted_data._read_records(options)

    def _require_started(self) -> None:
        if self._persisted_data is None:
            raise RuntimeError(
                "Cannot read from a non-started Spark statement; "
                "call start() first."
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
            self._register_external_views(session)
            text = PreparedStatement.apply_external_substitution(
                self.statement.text, self.statement.external_data,
            )
            df = session.sql(text)
            row_limit = self.statement.row_limit
            if row_limit:
                df = df.limit(row_limit)
            self.persist(data=df)
        except BaseException as exc:
            self._failure = exc
            self._started = True
            # Drop any views we registered before the failure so a retry
            # can re-register cleanly without leaking session-scoped state.
            self.clear_temporary_resources()
            if raise_error:
                raise
            return self

        self._started = True
        return self

    # -------------------------------------------------------------------------
    # External-data temp views
    # -------------------------------------------------------------------------

    def _register_external_views(self, session: "SparkSession") -> None:
        """Materialize each :class:`ExternalStatementData` as a temp view.

        ``text_value`` is taken as the view name when set; otherwise we
        derive one from ``text_key`` and a per-result suffix to avoid
        clashing across concurrent statements in the same session.  The
        substituted SQL references the view by name, and
        :meth:`clear_temporary_resources` drops every view registered here
        from the same session.

        Tabulars whose ``read_spark_frame`` returns nothing useful raise
        loudly — silent skips would leave ``{key}`` placeholders in the
        SQL and Spark would complain about undefined relations.
        """
        external_data = self.statement.external_data
        if not external_data:
            return

        for key, entry in external_data.items():
            if entry.tabular is None:
                # Caller already pinned ``text_value`` to a concrete
                # SQL fragment (existing table, file path, ...); nothing
                # to register.  Substitution still fires from
                # apply_external_substitution.
                if not entry.text_value:
                    raise ValueError(
                        f"ExternalStatementData[{key!r}]: tabular is None and "
                        f"text_value is empty; cannot materialize a temp view"
                    )
                continue

            view_name = entry.text_value or self._mint_temp_view_name(key)
            # Route through the registered converter so non-Spark tabulars
            # (ArrowTabular, polars/pandas wrappers, ...) get a real
            # DataFrame back instead of falling through the base
            # ``read_spark_frame`` Arrow short-circuit, which doesn't pass
            # a schema to ``createDataFrame`` and chokes on Arrow tables.
            from yggdrasil.spark.cast import any_to_spark_dataframe
            df = any_to_spark_dataframe(entry.tabular, CastOptions())
            df.createOrReplaceTempView(view_name)
            entry.text_value = view_name
            self._temp_views[key] = (session, view_name)

    def _mint_temp_view_name(self, key: str) -> str:
        """Build a session-unique temp view name for ``key``.

        Spark temp views live in the session catalog, so concurrent
        statements that register the same alias would clobber each other.
        Suffixing with the statement key keeps them disjoint.
        """
        # Statement keys come from ``_new_key``: ``<usec>-<hex>``; the
        # hyphen would break unquoted identifiers, so swap it out.
        suffix = self.key.replace("-", "_") if self.key else ""
        return f"_ygg_ext_{key}_{suffix}" if suffix else f"_ygg_ext_{key}"

    def clear_temporary_resources(self) -> None:
        """Drop every temp view registered by :meth:`_register_external_views`.

        Idempotent — safe to call from ``start``'s failure path, from the
        batch teardown loop, or from a caller that wants to release
        session-side scratch eagerly.  Failures while dropping a view are
        logged and swallowed so a transient session disconnect doesn't
        mask the real error.
        """
        if self._temp_views:
            for key, (session, view_name) in list(self._temp_views.items()):
                try:
                    session.catalog.dropTempView(view_name)
                except Exception:
                    logger.debug(
                        "Failed to drop temp view %r (key=%r); continuing.",
                        view_name, key, exc_info=True,
                    )
            self._temp_views.clear()
        # Reset materialized text_value so a subsequent start() re-registers
        # under a fresh view name (and a fresh session if the caller swapped
        # one in via ``start(spark_session=...)``).
        external_data = self.statement.external_data
        if external_data:
            for entry in external_data.values():
                if entry.tabular is not None:
                    entry.text_value = None
        super().clear_temporary_resources()

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