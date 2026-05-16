"""
:class:`SparkStatementExecutor` — minimal :class:`StatementExecutor` for
running SQL through a :class:`pyspark.sql.SparkSession`.

Spark is synchronous from the SQL submission perspective, so this
executor is essentially a thin shim around ``session.sql(text)`` that
returns a terminal :class:`SparkStatementResult`.  It plugs into
:class:`StatementBatch` like any other executor; the wait phase is a
no-op because each result is already terminal.

This module is kept independent of any Databricks-specific code so it can
be used standalone — open-source Spark, local PySpark, or composed into
:class:`SQLEngine` alongside a Databricks warehouse executor.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar, Optional, Any, TypeVar, Callable, Iterator

from yggdrasil.data.executor import StatementExecutor
from .statement import SparkPreparedStatement, SparkStatementResult, SparkStatementBatch
from ..data.options import CastOptions

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

__all__ = [
    "SparkStatementExecutor",
]


IN = TypeVar("IN", bound=Any)
OUT = TypeVar("OUT", bound=Any)


class SparkStatementExecutor(
    StatementExecutor[SparkPreparedStatement, SparkStatementResult, SparkStatementBatch]
):
    """Run statements through a SparkSession.

    The session is resolved in priority order:

    1. The session attached to the incoming :class:`SparkPreparedStatement`.
    2. The session pinned on this executor (``spark_session`` field).
    3. :meth:`PyEnv.spark_session` — creates one if necessary.

    Lazy resolution means the executor is cheap to construct even in
    environments where pyspark isn't installed; the import only fires
    when a statement actually runs.

    Singleton-cached for the process lifetime — Spark is a per-JVM
    singleton already, so two callers asking for a Spark executor
    share one instance. The pinned ``spark_session`` is rebindable in
    place; it doesn't participate in singleton identity (raw
    ``SparkSession`` objects aren't reliably hashable across processes
    anyway).
    """

    # Pin the concrete types so the base executor's coercion produces the
    # right subclass and `result.statement` always has the expected shape.
    _PREPARED_CLASS: ClassVar[type[SparkPreparedStatement]] = SparkPreparedStatement
    _RESPONSE_CLASS: ClassVar[type[SparkStatementResult]] = SparkStatementResult
    _BATCH_CLASS: ClassVar[type[SparkStatementBatch]] = SparkStatementBatch

    _SINGLETON_TTL: ClassVar[Any] = None

    # ``spark_session`` is rebindable live state, not identity. The
    # SparkSession itself is a per-JVM singleton already.
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({"spark_session"})

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        # One executor per (sub)class — Spark sessions are JVM-scoped
        # singletons, so multiple Python-side executors for the same
        # process would just share state anyway.
        return (cls,)

    def __init__(
        self,
        spark_session: Optional["SparkSession"] = None,
        *args,
        **kwargs,
    ):
        if getattr(self, "_initialized", False):
            # Re-pin the session if the caller passed one; otherwise
            # leave whatever's already cached intact.
            if spark_session is not None:
                self.spark_session = spark_session
            return
        super().__init__(*args, **kwargs)
        self.spark_session = spark_session
        self._initialized = True

    @classmethod
    def default(cls) -> "SparkStatementExecutor":
        """Return a default executor — no shared registry, just a fresh handle."""
        return cls()

    # Back-compat alias; prefer :meth:`default` going forward.
    current = default

    # -------------------------------------------------------------------------
    # Session resolution
    # -------------------------------------------------------------------------

    def resolve_session(
        self,
        statement: Optional[SparkPreparedStatement] = None,
        *,
        create: bool = True,
    ) -> Optional["SparkSession"]:
        """Resolve the SparkSession used to execute ``statement``.

        Precedence: per-statement → executor-pinned → environment.  When
        ``create=True`` (the default) a missing session is materialized
        via :meth:`PyEnv.spark_session`; with ``create=False`` and no
        session reachable, ``None`` is returned.
        """
        if statement is not None and statement.spark_session is not None:
            return statement.spark_session
        if self.spark_session is not None:
            return self.spark_session

        from yggdrasil.environ import PyEnv  # local import keeps pyspark optional

        return PyEnv.spark_session(
            create=create,
            install_spark=False,
            import_error=create,
        )

    def has_session(self) -> bool:
        """Whether a SparkSession is reachable without creating a new one.

        Useful for engines that compose this executor and want to fall
        back to a different backend when Spark isn't available.
        """
        if self.spark_session is not None:
            return True
        from yggdrasil.environ import PyEnv

        return (
            PyEnv.spark_session(create=False, install_spark=False, import_error=False)
            is not None
        )

    # -------------------------------------------------------------------------
    # Executor contract
    # -------------------------------------------------------------------------

    def _submit_statement(self, statement: SparkPreparedStatement, start: bool = True) -> SparkStatementResult:
        """Build a :class:`SparkStatementResult` and optionally run it.

        Spark execution is synchronous; when *start* is True the result
        handle comes back terminal — errors during ``session.sql`` are
        captured on the result (``result.failed`` becomes True) and the
        base executor's ``_execute`` decides whether to re-raise. When
        *start* is False the result is returned in its idle state so
        the caller (typically :class:`StatementBatch`) can drive the
        ``start`` / ``wait`` lifecycle itself.

        Plumbs a :class:`SparkSession` onto the statement up front via
        :meth:`resolve_session` so subclasses that build their session
        lazily (e.g. :class:`ServerlessClusterStatementExecutor` going
        through ``client.spark()``) get a chance to install their own
        before :meth:`SparkStatementResult.start` falls back to
        :meth:`PyEnv.spark_session`.
        """
        if statement.spark_session is None:
            session = self.resolve_session(statement, create=False)
            if session is None and self.spark_session is not None:
                session = self.spark_session
            if session is not None:
                statement.spark_session = session

        result = self._RESPONSE_CLASS(statement=statement, executor=self)
        if not start:
            return result

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Spark executing:\n%s", statement.text)
        # raise_error=False: errors are recorded on the result; the base
        # executor's _execute calls raise_for_status afterwards if needed.
        result.start(wait=False, raise_error=False)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Spark executed: failed=%s",
                getattr(result, "failed", "?"),
            )
        return result

    # -------------------------------------------------------------------------
    # Convenience
    # -------------------------------------------------------------------------

    def sql(self, text: str, *, row_limit: Optional[int] = None) -> SparkStatementResult:
        """Shortcut: run a raw SQL string and return the terminal result.

        Equivalent to ``execute(SparkPreparedStatement(text, row_limit=row_limit))``
        but skips the coercion round-trip.
        """
        stmt = SparkPreparedStatement(
            text=text,
            spark_session=self.spark_session,
            row_limit=row_limit,
        )
        return self.execute(stmt, wait=False, raise_error=True)

    def scoped_spark_conf(
        self,
        session: "SparkSession | None" = None,
        conf: dict[str, str] | None = None,
    ):
        session = self.resolve_session() if session is None else session
        conf = dict(conf or {})
        return _scoped_spark_conf(session, conf)
    
    def parallelize(
        self,
        inputs: Any,
        *,
        options: CastOptions | None = None,
        transformer: Callable[[IN], OUT] | None = None,
    ):
        from .cast import any_to_spark_dataframe

        options = CastOptions.check(options)
        df = any_to_spark_dataframe(
            inputs,
            options.with_target(options.source).with_source(None)
        )

        def within_partitions(iterator: Iterator[IN]):
            for item in iterator:
                yield transformer(item) if transformer is not None else item

        # TODO: complete
        raise NotImplementedError

class _scoped_spark_conf:
    """Context manager: stash & set Spark session conf keys; restore on exit."""

    def __init__(self, session: Any, conf: dict[str, str]):
        self._session = session
        self._conf = conf
        self._saved: dict[str, Optional[str]] = {}

    def __enter__(self):
        for k, v in self._conf.items():
            try:
                self._saved[k] = self._session.conf.get(k)
            except Exception:
                self._saved[k] = None
            self._session.conf.set(k, v)
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, prev in self._saved.items():
            try:
                if prev is None:
                    self._session.conf.unset(k)
                else:
                    self._session.conf.set(k, prev)
            except Exception:
                logger.debug("Failed to restore Spark conf %r; continuing.", k, exc_info=True)
        return False