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
    """

    # Pin the concrete types so the base executor's coercion produces the
    # right subclass and `result.statement` always has the expected shape.
    _PREPARED_STATEMENT_CLASS: ClassVar[type[SparkPreparedStatement]] = SparkPreparedStatement
    _STATEMENT_RESULT_CLASS: ClassVar[type[SparkStatementResult]] = SparkStatementResult
    _STATEMENT_BATCH_CLASS: ClassVar[type[SparkStatementBatch]] = SparkStatementBatch

    def __init__(
        self,
        spark_session: Optional["SparkSession"] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.spark_session = spark_session

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

    def _submit_statement(self, statement: SparkPreparedStatement) -> SparkStatementResult:
        """Build a :class:`SparkStatementResult` and run it eagerly.

        Spark execution is synchronous; the result handle is terminal by
        the time it returns.  Errors during ``session.sql`` are stored on
        the result (``result.failed`` becomes True); the base executor's
        ``_execute`` decides whether to raise based on its options.
        """
        # If the statement didn't carry a session and the executor has one,
        # plumb it onto the statement so SparkStatementResult.start can
        # read it without re-running session resolution.
        if statement.spark_session is None and self.spark_session is not None:
            statement.spark_session = self.spark_session

        result = self._STATEMENT_RESULT_CLASS(statement=statement, executor=self)
        # raise_error=False: errors are recorded on the result; the base
        # executor's _execute calls raise_for_status afterwards if needed.
        result.start(wait=False, raise_error=False)
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