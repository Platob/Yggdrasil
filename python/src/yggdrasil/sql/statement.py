"""SQL :class:`PreparedStatement` and :class:`StatementResult`.

Two value types backing :func:`yggdrasil.sql.sql`:

- :class:`SqlPreparedStatement` — the SQL text plus a frozen view
  of the source bindings (resolved at submit time so a later
  ``register`` doesn't change what the statement runs against),
  the optional ``where=`` predicate added in Python (composed
  with the parsed ``WHERE`` via AND), and the persistence target
  (``"memory"``, ``"path"``, or ``None`` to skip persistence).
- :class:`SqlStatementResult` — the live :class:`StatementResult`
  the executor returns. Inherits the full :class:`Tabular`
  surface (``read_arrow_table`` / ``read_polars_frame`` /
  ``read_pandas_frame`` / ``write_*``) from
  :class:`yggdrasil.data.statement.StatementResult`, persists the
  materialized result through ``_persisted_data`` so the standard
  cache path serves repeat reads from memory or disk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Mapping, Optional

import pyarrow as pa

from yggdrasil.data.expr import Expression
from yggdrasil.data.options import CastOptions
from yggdrasil.data.statement import PreparedStatement, StatementResult
from yggdrasil.data.enums import MimeType
from yggdrasil.io.tabular import Tabular

from .catalog import SqlContext, default_context
from .dialect import Dialect, resolve_dialect

if TYPE_CHECKING:
    from .executor import SqlExecutor


__all__ = [
    "SqlPreparedStatement",
    "SqlStatementResult",
    "PersistTarget",
]


PersistTarget = Optional[str]
"""Where to land the result.

``"memory"`` (default) → :class:`MemoryArrowIO`. ``"path"`` →
spill to a parquet folder under ``path``. ``None`` → don't persist
(the result is consumed once on read; subsequent reads re-execute
the underlying engine, so callers who plan to drain the result
multiple times should leave the default).
"""


class SqlPreparedStatement(PreparedStatement):
    """Frozen SQL plan: text + dialect + bound sources + composed predicate.

    Sources are snapshotted from the :class:`SqlContext` at
    construction so concurrent ``register`` / ``deregister`` calls
    don't pull the rug out from under an in-flight statement.

    The ``predicate`` slot accepts any :class:`Expression`
    (typically a :class:`Predicate`) added in Python — composed
    with the parsed ``WHERE`` via ``AND`` at execution time. That's
    how ``sql("SELECT * FROM t", where=col("x") > 5)`` works
    without re-stringifying the predicate.

    The ``select`` slot accepts a list of column names or
    :class:`Selector` projections to apply after the SQL ``SELECT``
    runs — useful for renaming / casting on the way out without
    mutating the SQL text.
    """

    def __init__(
        self,
        text: str = "",
        *,
        dialect: "Dialect | str | None" = None,
        sources: "Mapping[str, Tabular] | None" = None,
        predicate: "Expression | None" = None,
        select: "Iterable[Any] | None" = None,
        persist: PersistTarget = "memory",
        path: "str | None" = None,
        key: "str | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(text=text, key=key, **kwargs)
        self.dialect: Dialect = resolve_dialect(dialect)
        self.sources: "dict[str, Tabular]" = dict(sources) if sources else {}
        self.predicate: "Expression | None" = predicate
        self.select: "tuple[Any, ...] | None" = (
            tuple(select) if select is not None else None
        )
        self.persist: PersistTarget = persist
        self.path: "str | None" = path

    def with_predicate(
        self,
        predicate: "Expression | None",
        *,
        inplace: bool = False,
    ) -> "SqlPreparedStatement":
        """Return (or update in place) a copy with ``predicate`` AND-merged in.

        ``predicate=None`` clears the slot. A predicate already on
        the statement merges with the new one via
        :meth:`Expression.merge_with` (AND for two predicates,
        identity-merge for two equal expressions).
        """
        target = self if inplace else _shallow_copy(self)
        if predicate is None:
            target.predicate = None
        elif target.predicate is None:
            target.predicate = predicate
        else:
            target.predicate = target.predicate.merge_with(predicate)
        return target

    @classmethod
    def from_(
        cls,
        statement: "PreparedStatement | StatementResult | str",
    ) -> "SqlPreparedStatement":
        if isinstance(statement, cls):
            return statement
        if isinstance(statement, str):
            return cls(statement)
        if isinstance(statement, StatementResult):
            return cls.from_(statement.statement)
        if isinstance(statement, PreparedStatement):
            return cls(statement.text, key=statement.key)
        raise TypeError(
            f"Cannot prepare {type(statement).__module__}.{type(statement).__name__} "
            f"as {cls.__name__}. Pass a SQL string or another SqlPreparedStatement."
        )


def _shallow_copy(stmt: SqlPreparedStatement) -> SqlPreparedStatement:
    """Build a sibling statement with shared mutable slots.

    A real ``copy.copy`` would tear off the underlying bindings;
    we want predicate replacement to be cheap but the source dict
    to remain shared (it's frozen by convention after construction).
    """
    out = SqlPreparedStatement.__new__(SqlPreparedStatement)
    out.__dict__.update(stmt.__dict__)
    return out


class SqlStatementResult(StatementResult[SqlPreparedStatement]):
    """Result handle for an in-process SQL execution.

    Lifecycle is synchronous: :meth:`start` runs the executor and
    stashes the materialized payload on ``_persisted_data`` (a
    :class:`MemoryArrowIO` for ``persist="memory"``, a
    :class:`ParquetIO` folder for ``persist="path"``). Once
    started, every :class:`Tabular` read method on this object
    serves from the cache — ``read_arrow_table`` /
    ``read_polars_frame`` / ``read_pandas_frame`` /
    ``read_spark_frame`` / ``read_pylist`` / ``to_records`` all
    go through the standard funnel.

    Call :meth:`unpersist` to drop the cache and force the next
    read to re-execute. Useful when the underlying source
    :class:`Tabular` has been mutated and you want a fresh view.
    """

    _PREPARED_STATEMENT_CLASS: ClassVar[type[SqlPreparedStatement]] = SqlPreparedStatement

    @classmethod
    def default_media_type(cls) -> "MimeType | None":
        # In-process SQL results don't claim a wire mime type — we
        # don't want to override the base StatementResult registration
        # in the IO factory dispatch table. Returning None keeps us
        # out of the registry while the inherited Tabular surface
        # still works fine.
        return None

    def __init__(
        self,
        statement: "SqlPreparedStatement | str",
        *,
        executor: "SqlExecutor | None" = None,
        context: "SqlContext | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(statement=statement, executor=executor, **kwargs)
        # Bind a context for resolve-time. When the prepared statement
        # already carries snapshotted sources we use those directly;
        # otherwise we fall back to the supplied (or default) context
        # at submit time.
        self._context: SqlContext = context or default_context
        self._started: bool = False
        self._failure: "BaseException | None" = None
        self._row_count: int = -1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._started

    @property
    def done(self) -> bool:
        # In-process execution is synchronous — there's no remote
        # state to poll, so "started" is "done" for our purposes.
        return self._started

    @property
    def failed(self) -> bool:
        return self._failure is not None

    def refresh_status(self) -> None:
        """No-op — in-process SQL has no remote state to refresh."""
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
        """Materialized row count, or ``-1`` before :meth:`start`."""
        return self._row_count

    def start(
        self,
        reset: bool = False,
        *,
        wait: Any = True,
        raise_error: bool = True,
        **kwargs: Any,
    ) -> "SqlStatementResult":
        """Execute the SQL and stash the materialized payload.

        Idempotent unless ``reset=True``. On failure the exception
        is recorded on ``_failure`` and re-raised when
        ``raise_error`` is True.
        """
        if self._started and not reset:
            if raise_error:
                self.raise_for_status()
            return self

        if reset:
            self._failure = None
            self._cached_schema = None
            self._row_count = -1
            self._started = False
            if self._persisted_data is not None:
                try:
                    self._persisted_data.close()
                except Exception:
                    pass
                self._persisted_data = None

        # Lazy import to avoid the executor pulling in optional
        # backends just because someone touched this module.
        from .executor import resolve_executor

        executor = self.executor or resolve_executor(self._context)
        try:
            holder, row_count = executor.run(self.statement, context=self._context)
        except BaseException as exc:
            self._failure = exc
            self._started = True
            if raise_error:
                raise
            return self

        self._persisted_data = holder
        self._row_count = row_count
        self._started = True
        return self

    def cancel(self) -> "SqlStatementResult":
        """No-op for in-process execution."""
        return self

    # ------------------------------------------------------------------
    # Tabular hooks — delegate through the cache
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self, options: CastOptions,
    ) -> Iterator[pa.RecordBatch]:
        if not self._started:
            self.start()
        if self._persisted_data is None:
            # Failure branch: start() recorded an exception but the
            # caller said raise_error=False. Surface it now so the
            # iteration doesn't silently yield nothing.
            self._raise_for_status()
            return iter(())
        # Route through the holder's *public* method so its
        # options_class (e.g. :class:`ParquetOptions` for the
        # disk-spilled path) gets a chance to promote the incoming
        # :class:`CastOptions` to its richer subclass. The holder
        # itself never has a nested ``_persisted_data`` set, so
        # there's no loop risk in delegating up here.
        yield from self._persisted_data.read_arrow_batches(options=options)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        raise NotImplementedError(
            "SqlStatementResult is read-only — it represents the output "
            "of a SQL query, not a writable target. Build a Tabular "
            "(MemoryArrowIO, ParquetIO, ...) and pipe the result into "
            "it via `result.read_arrow_batches()` if you need to land "
            "it somewhere new."
        )
