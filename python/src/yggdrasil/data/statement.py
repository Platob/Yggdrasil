from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Union

import pyarrow as pa
from yggdrasil.data import Schema
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg

logger = logging.getLogger(__name__)

DEFAULT_PARALLEL = 4

if TYPE_CHECKING:
    import pandas
    import polars
    import pyarrow.dataset as ds
    import pyspark.sql
    from databricks.sdk.service.sql import StatementParameterListItem
    from yggdrasil.databricks.sql.engine import SQLEngine

BatchConcatMode = Literal[
    "vertical",
    "vertical_relaxed",
    "diagonal",
    "diagonal_relaxed",
]

__all__ = [
    "BatchConcatMode",
    "PreparedStatement",
    "StatementBatch",
    "StatementResult",
]


_SQL_COMMENT_OR_WS_RE = re.compile(
    r"\A(?:\s+|--[^\n]*\n|--[^\n]*\Z|/\*.*?\*/)+",
    re.DOTALL,
)
_SQL_QUERY_LEAD_RE = re.compile(
    r"(?:SELECT|WITH|VALUES|TABLE|FROM)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PreparedStatement:
    """Configuration for a single statement execution.

    ``PreparedStatement`` is a plain value object — it carries the SQL text, any
    named parameters, and a map of external-table aliases that the engine
    should substitute before submission.  Runtime/execution state
    (backend handle, response, cached results) lives on
    :class:`StatementResult`.

    Instances are frozen: every mutator (``bind``, ``with_external_tables``,
    ``clear``) returns a new ``PreparedStatement``.
    """

    text: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    external_tables: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def looks_like_query(text: Any) -> bool:
        """Fast heuristic: return ``True`` when ``text`` looks like a SQL query.

        Leading whitespace and SQL comments are skipped; a string is treated
        as a query when its first keyword is ``SELECT``, ``WITH``, ``VALUES``,
        ``TABLE``, or ``FROM``.  Non-string inputs return ``False``.
        """
        if not isinstance(text, str) or not text:
            return False
        stripped = text.lstrip()
        if not stripped:
            return False
        while True:
            match = _SQL_COMMENT_OR_WS_RE.match(stripped)
            if not match:
                break
            stripped = stripped[match.end():]
        return bool(_SQL_QUERY_LEAD_RE.match(stripped))

    @classmethod
    def prepare(
        cls,
        statement: "PreparedStatement | str",
        *,
        parameters: Mapping[str, Any] | None = None,
        external_tables: Mapping[str, Any] | None = None,
    ) -> "PreparedStatement":
        """Coerce ``statement`` into a :class:`PreparedStatement`, merging extra args."""
        if isinstance(statement, cls):
            prepared = statement
            if parameters:
                prepared = prepared.bind(**parameters)
            if external_tables:
                prepared = prepared.with_external_tables(**external_tables)
            return prepared

        return cls(
            text=str(statement),
            parameters=dict(parameters) if parameters else {},
            external_tables=dict(external_tables) if external_tables else {},
        )

    def bind(self, **parameters: Any) -> "PreparedStatement":
        """Return a new ``PreparedStatement`` with additional named parameters merged."""
        if not parameters:
            return self
        return replace(self, parameters={**self.parameters, **parameters})

    def with_external_tables(self, **tables: Any) -> "PreparedStatement":
        """Return a new ``PreparedStatement`` with additional external-table aliases."""
        if not tables:
            return self
        return replace(self, external_tables={**self.external_tables, **tables})

    def clear(self) -> "PreparedStatement":
        """Return a new ``PreparedStatement`` with text and all bound arguments cleared."""
        return replace(self, text="", parameters={}, external_tables={})

    def with_text(self, text: str) -> "PreparedStatement":
        """Return a new ``PreparedStatement`` with ``text`` replaced."""
        if text == self.text:
            return self
        return replace(self, text=text)

    def to_parameter_list(self) -> Optional[List["StatementParameterListItem"]]:
        """Render bound parameters as Databricks ``StatementParameterListItem``.

        Lazy-imports the Databricks SDK type so base installs without the
        ``databricks`` extra keep working.
        """
        if not self.parameters:
            return None
        from databricks.sdk.service.sql import StatementParameterListItem
        return [
            StatementParameterListItem(
                name=str(name),
                value=None if value is None else str(value),
            )
            for name, value in self.parameters.items()
        ]


@dataclass
class StatementResult(ABC):
    """Arrow-first result handler for a :class:`PreparedStatement`.

    This class defines a small execution contract plus a rich set of conversion helpers.
    Concrete implementations only need to provide status handling and a way to expose
    results as an Arrow ``RecordBatchReader``.

    Design goals
    ------------
    - Use Apache Arrow as the primary interchange format.
    - Make common conversions cheap and predictable.
    - Allow optional local caching of materialized Arrow or Spark data.

    Subclasses must implement
    -------------------------
    - ``done``: whether execution has reached a terminal state
    - ``failed``: whether execution failed or was canceled
    - ``raise_for_status()``: raise on failure or cancellation
    - ``refresh_status()``: pull fresh execution state from the backend
    - ``start()`` / ``cancel()``: submit / cancel on the backend
    - ``collect_schema()``: schema for the result
    - ``to_arrow_reader()``: stream the result as Arrow record batches

    Notes
    -----
    - ``stream=True`` means "prefer lazy / streaming consumers where possible".
      It does not guarantee that the backend itself is fully streaming.
    - Some conversions materialize data locally and may collect all rows to the driver.
      Those cases are called out in method docs.
    """

    statement: PreparedStatement = field(default_factory=PreparedStatement)

    _data_schema: Optional[Schema] = field(init=False, default=None, repr=False, compare=False)
    _arrow_table: Optional[pa.Table] = field(init=False, default=None, repr=False, compare=False)
    _spark_df: Optional["pyspark.sql.DataFrame"] = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _external_tables: tuple[Any, ...] = field(
        init=False,
        default=(),
        repr=False,
        compare=False,
    )
    _external_tables_cleaned: bool = field(
        init=False,
        default=False,
        repr=False,
        compare=False,
    )

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        """Iterate over result batches as Arrow ``RecordBatch`` objects."""
        return self.to_arrow_batches()

    # -------------------------------------------------------------------------
    # Pickling
    # -------------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Return pickle-safe instance state.

        Spark DataFrames are not pickleable. If a Spark DataFrame is attached, it is
        converted to a local Arrow table before serializing state.

        Warning
        -------
        Converting Spark to Arrow collects data to the driver. Avoid pickling very large
        Spark-backed results unless that is explicitly intended.
        """
        state = {
            "_data_schema": self._data_schema,
            "_arrow_table": self._arrow_table,
            "_spark_df": None,
        }

        if self._spark_df is not None:
            from yggdrasil.arrow.cast import any_to_arrow_table

            state["_arrow_table"] = any_to_arrow_table(self._spark_df, None)

        return state

    def __setstate__(self, state: dict) -> None:
        """Restore pickle-safe instance state."""
        for name in ("_data_schema", "_arrow_table", "_spark_df"):
            object.__setattr__(self, name, state.get(name))

    # -------------------------------------------------------------------------
    # Core state / caching
    # -------------------------------------------------------------------------

    @property
    def is_spark_sql(self) -> bool:
        """Whether this result currently has a cached Spark DataFrame."""
        return self._spark_df is not None

    @property
    def persisted(self) -> bool:
        """Whether this result has a cached local representation."""
        return self._arrow_table is not None or self._spark_df is not None

    def persist(
        self,
        mode: Literal["arrow", "spark", "auto"] = "auto",
        *,
        data: Optional[Union[pa.Table, "pyspark.sql.DataFrame"]] = None,
    ) -> StatementResult:
        """Materialize and cache the result.

        Parameters
        ----------
        mode:
            Cache target:
            - ``"arrow"``: materialize to a local ``pyarrow.Table``
            - ``"spark"``: materialize to a Spark DataFrame
            - ``"auto"``: currently the same as ``"arrow"``
        data:
            Optional precomputed representation to attach directly.

        Returns
        -------
        StatementResult
            ``self`` with cache fields updated.

        Notes
        -----
        - Providing ``data`` bypasses materialization.
        - Persisting as Spark may still require first collecting a local Arrow table.
        - Persisting as Arrow materializes all rows locally.
        """
        if data is not None:
            object.__setattr__(self, "_data_schema", Schema.from_(data))

            if isinstance(data, pa.Table):
                object.__setattr__(self, "_arrow_table", data)
                object.__setattr__(self, "_spark_df", None)
                return self

            if _is_spark_dataframe(data):
                object.__setattr__(self, "_spark_df", data)
                object.__setattr__(self, "_arrow_table", None)
                return self

            raise TypeError(
                f"Unsupported data type for persist(): {type(data)!r}. "
                "Expected pyarrow.Table or pyspark.sql.DataFrame."
            )

        if self.persisted:
            return self

        if mode in {"auto", "arrow"}:
            return self.persist(data=self.to_arrow_table())

        if mode == "spark":
            return self.persist(data=self.to_spark())

        raise ValueError(
            f"Unknown persist mode: {mode!r}. Expected 'auto', 'arrow', or 'spark'."
        )

    # -------------------------------------------------------------------------
    # Execution lifecycle contract
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the statement is in a terminal state."""
        raise NotImplementedError

    @property
    @abstractmethod
    def failed(self) -> bool:
        """Whether the statement failed or was canceled."""
        raise NotImplementedError

    @abstractmethod
    def raise_for_status(self) -> None:
        """Raise an exception if the statement failed or was canceled."""
        raise NotImplementedError

    @abstractmethod
    def refresh_status(self) -> None:
        """Refresh execution state from the backend."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Config shortcuts
    # -------------------------------------------------------------------------

    @property
    def text(self) -> str:
        return self.statement.text

    @property
    def parameters(self) -> Mapping[str, Any]:
        return self.statement.parameters

    @property
    def external_tables(self) -> Mapping[str, Any]:
        return self.statement.external_tables

    @abstractmethod
    def start(
        self,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        **kwargs: Any,
    ) -> StatementResult:
        """Submit the statement for execution.

        Implementations must be idempotent: calling ``start()`` on an already
        started statement returns ``self`` unchanged.  Subclasses are free to
        accept additional backend-specific keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def cancel(self) -> StatementResult:
        """Request cancellation of a running statement.

        Implementations must be idempotent and a no-op when the statement has
        not been started or has already reached a terminal state.
        """
        raise NotImplementedError

    def wait(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> StatementResult:
        """Wait until execution reaches a terminal state.

        Parameters
        ----------
        wait:
            Waiting policy.
            - ``True``: use default polling behavior
            - ``False``: do not wait
            - ``WaitingConfig``: custom polling / timeout behavior
        raise_error:
            Whether to raise if execution finishes in a failed state.

        Returns
        -------
        StatementResult
            ``self``

        Notes
        -----
        This method polls by repeatedly calling ``refresh_status()`` until ``done`` is true.
        """
        wait = WaitingConfig.check_arg(wait)

        if not wait:
            if raise_error:
                self.raise_for_status()
            self._maybe_cleanup_external_tables()
            return self

        iteration = 0
        start = time.time()

        self.refresh_status()
        while not self.done:
            wait.sleep(iteration=iteration, start=start)
            iteration += 1
            self.refresh_status()

        if raise_error:
            self.raise_for_status()

        self._maybe_cleanup_external_tables()

        return self

    # -------------------------------------------------------------------------
    # External table cleanup
    # -------------------------------------------------------------------------

    def attach_external_tables(self, tables: Iterable[Any]) -> StatementResult:
        """Attach external staging resources to be cleaned up when ``done``.

        Each entry must expose ``cleanup(allow_not_found: bool = True)``.
        Cleanup is best-effort and idempotent; it runs lazily the first time
        the statement reaches a terminal state (see ``_maybe_cleanup_external_tables``).
        """
        items = tuple(tables)
        if not items:
            return self
        object.__setattr__(
            self,
            "_external_tables",
            tuple(self._external_tables) + items,
        )
        object.__setattr__(self, "_external_tables_cleaned", False)
        return self

    def _maybe_cleanup_external_tables(self) -> None:
        if self._external_tables_cleaned or not self._external_tables:
            return
        try:
            is_done = self.done
        except Exception:
            return
        if not is_done:
            return

        for resource in self._external_tables:
            try:
                resource.cleanup(allow_not_found=True)
            except Exception:
                logger.debug(
                    "Failed to cleanup external staging resource %r",
                    resource,
                    exc_info=True,
                )
        object.__setattr__(self, "_external_tables_cleaned", True)
        object.__setattr__(self, "_external_tables", ())

    # -------------------------------------------------------------------------
    # Arrow contract
    # -------------------------------------------------------------------------

    @property
    def data_schema(self) -> Schema:
        if self._data_schema is None:
            schema = self.collect_schema()
            object.__setattr__(self, "_data_schema", schema)
        return self._data_schema

    @property
    def arrow_schema(self) -> pa.Schema:
        return self.data_schema.to_arrow_schema()

    @abstractmethod
    def collect_schema(self, full: bool = False) -> Schema:
        """Generate and cache the result schema."""
        raise NotImplementedError

    @abstractmethod
    def to_arrow_reader(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> pa.RecordBatchReader:
        """Return an Arrow ``RecordBatchReader`` for the result."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Arrow / Dataset conversions
    # -------------------------------------------------------------------------

    def to_arrow_batches(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> Iterator[pa.RecordBatch]:
        """Yield the result as Arrow record batches."""
        reader = self.to_arrow_reader(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schema=schema or self.arrow_schema,
            maintain_order=maintain_order,
            stream=stream,
        )
        yield from reader

    def to_arrow_dataset(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> "ds.Dataset":
        """Return an in-memory ``pyarrow.dataset.Dataset`` view of the result."""
        import pyarrow.dataset as pds

        resolved_schema = schema or self.arrow_schema

        if self._arrow_table is not None:
            return pds.dataset(self._arrow_table, schema=resolved_schema)

        if self._spark_df is not None:
            table = self._spark_df.toArrow()
            batches = table.to_batches(max_chunksize=batch_size) if batch_size else table.to_batches()
            return pds.dataset(batches, schema=resolved_schema)

        reader = self.to_arrow_reader(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schema=resolved_schema,
            maintain_order=maintain_order,
            stream=stream,
        )
        return pds.dataset(reader, schema=reader.schema)

    def to_arrow_table(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> pa.Table:
        """Materialize the full result as a local Arrow table."""
        if self._arrow_table is not None:
            return self._arrow_table

        if self._spark_df is not None:
            return self._spark_df.toArrow()

        reader = self.to_arrow_reader(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schema=schema or self.arrow_schema,
            maintain_order=maintain_order,
            stream=stream,
        )
        return reader.read_all()

    # -------------------------------------------------------------------------
    # pandas / polars
    # -------------------------------------------------------------------------

    def to_pandas(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> "pandas.DataFrame":
        """Materialize the result as a pandas DataFrame via Arrow."""
        return self.to_arrow_table(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schema=schema,
            maintain_order=maintain_order,
            stream=stream,
        ).to_pandas()

    def to_polars(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> Union["polars.LazyFrame", "polars.DataFrame"]:
        """Convert the result to Polars."""
        from ..polars.lib import polars

        dataset = self.to_arrow_dataset(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schema=schema,
            maintain_order=maintain_order,
            stream=stream,
        )

        lazy_frame = polars.scan_pyarrow_dataset(dataset)
        return lazy_frame if stream else lazy_frame.collect()

    # -------------------------------------------------------------------------
    # Spark
    # -------------------------------------------------------------------------

    def to_spark(
        self,
        *,
        spark: Optional["pyspark.sql.SparkSession"] = None,
        prefer_cached: bool = True,
        cache_result: bool = True,
    ) -> "pyspark.sql.DataFrame":
        """Convert the result to a Spark DataFrame."""
        if prefer_cached and self._spark_df is not None:
            return self._spark_df

        from yggdrasil.spark.cast import any_to_spark_dataframe

        spark_df = any_to_spark_dataframe(self.to_arrow_table(), spark)

        if cache_result:
            object.__setattr__(self, "_spark_df", spark_df)

        return spark_df


@dataclass(frozen=True)
class StatementBatch(Mapping[str, "StatementResult"]):
    """Ordered batch wrapper around multiple :class:`StatementResult` handlers.

    Construct via :meth:`from_results` (already-built handlers) or
    :meth:`from_statements` (configs + a factory callable that turns each
    :class:`PreparedStatement` config into an unstarted :class:`StatementResult`).

    By default, materialized conversions concatenate inner tabular results using
    Polars ``how="diagonal_relaxed"`` semantics.

    Use ``concat=None`` to preserve per-statement outputs.

    Lifecycle
    ---------
    :meth:`start`, :meth:`wait` and :meth:`cancel` orchestrate execution across
    every result in the batch.  When ``parallel`` is ``True`` or an integer
    greater than ``1``, :meth:`start` keeps at most ``parallel`` statements
    in flight on the backend at a time by driving each result's own
    :meth:`StatementResult.start` / :meth:`StatementResult.refresh_status` /
    :meth:`StatementResult.cancel` — no extra threads are created.  When any
    in-flight statement fails, every remaining submission is cancelled
    before the exception propagates.
    """

    results: OrderedDict[str, StatementResult]
    engine: Optional["SQLEngine"] = field(
        default=None, repr=False, compare=False,
    )

    _in_flight: OrderedDict[str, StatementResult] = field(
        default_factory=OrderedDict, init=False, repr=False, compare=False,
    )
    _pending_queue: deque = field(
        default_factory=deque, init=False, repr=False, compare=False,
    )
    _pool_runner: Optional[Callable[[StatementResult], Any]] = field(
        default=None, init=False, repr=False, compare=False,
    )
    _pool_raise_error: bool = field(
        default=True, init=False, repr=False, compare=False,
    )

    @classmethod
    def from_results(
        cls,
        results: Iterable[StatementResult] | Mapping[str, StatementResult],
        *,
        engine: Optional["SQLEngine"] = None,
    ) -> StatementBatch:
        """Build a batch from already-constructed :class:`StatementResult` handlers.

        When ``engine`` is provided, :meth:`start` uses
        ``engine.execute(result, wait=False)`` as its default submission
        callable — callers no longer need to pass a custom ``runner``.
        """
        if isinstance(results, Mapping):
            return cls(results=OrderedDict(results.items()), engine=engine)

        return cls(
            results=OrderedDict(
                (str(i), result)
                for i, result in enumerate(results)
            ),
            engine=engine,
        )

    @classmethod
    def from_statements(
        cls,
        statements: Iterable[PreparedStatement | str] | Mapping[str, PreparedStatement | str],
        *,
        engine: Optional["SQLEngine"] = None,
    ) -> StatementBatch:
        """Build a batch from :class:`PreparedStatement` configs (or strings).

        Each entry is normalized via :meth:`PreparedStatement.prepare` and
        handed to :meth:`factory` to build the corresponding
        :class:`StatementResult`.  The default factory delegates to
        ``engine.statement_result`` — subclasses may override :meth:`factory`
        for custom construction.
        """
        items = (
            statements.items()
            if isinstance(statements, Mapping)
            else enumerate(statements)
        )

        # Seed with an empty batch so ``factory`` can run as an instance
        # method and pick up the bound engine.
        batch = cls(results=OrderedDict(), engine=engine)
        results: OrderedDict[str, StatementResult] = OrderedDict()
        for key, raw in items:
            cfg = PreparedStatement.prepare(raw)
            results[str(key)] = batch.factory(cfg)
        object.__setattr__(batch, "results", results)
        return batch

    def factory(self, statement: PreparedStatement) -> StatementResult:
        """Build an unstarted :class:`StatementResult` for ``statement``.

        Default implementation delegates to the bound engine's
        ``statement_result`` — that's the cleanest way to build a
        backend-specific result from a generic config.  Subclasses may
        override for custom construction; callers with no engine to hand
        over must either attach one at construction time or provide a
        subclass.
        """
        if self.engine is None:
            raise NotImplementedError(
                "StatementBatch.factory needs an engine bound to the batch "
                "(``engine=`` kwarg on from_statements / from_results) or a "
                "subclass override."
            )
        return self.engine.statement_result(statement)

    def __iter__(self) -> Iterator[str]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, key: str) -> StatementResult:
        return self.results[key]

    @property
    def statements(self) -> OrderedDict[str, PreparedStatement]:
        """Config view of the batch: each result's :class:`PreparedStatement`."""
        return OrderedDict(
            (key, result.statement)
            for key, result in self.results.items()
        )

    @property
    def first(self) -> StatementResult | None:
        for result in self.results.values():
            return result
        return None

    @property
    def last(self) -> StatementResult | None:
        if not self.results:
            return None
        return next(reversed(self.results.values()))

    @property
    def done(self) -> bool:
        return all(result.done for result in self.results.values())

    @property
    def failed(self) -> bool:
        return any(result.failed for result in self.results.values())

    @property
    def persisted(self) -> bool:
        return all(result.persisted for result in self.results.values())

    @property
    def data_schemas(self) -> OrderedDict[str, Schema]:
        return OrderedDict(
            (key, result.data_schema)
            for key, result in self.results.items()
        )

    @property
    def arrow_schemas(self) -> OrderedDict[str, pa.Schema]:
        return OrderedDict(
            (key, result.arrow_schema)
            for key, result in self.results.items()
        )

    def refresh_status(self) -> StatementBatch:
        for result in self.results.values():
            result.refresh_status()
        return self

    def raise_for_status(self) -> StatementBatch:
        for key, result in self.results.items():
            try:
                result.raise_for_status()
            except Exception as exc:
                raise RuntimeError(f"PreparedStatement batch item {key!r} failed.") from exc
        return self

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_parallel(parallel: int | bool | None) -> int:
        """Return the effective pool size.

        - ``False`` / ``None`` / ``0`` → ``1`` (sequential).
        - ``True`` → ``DEFAULT_PARALLEL`` (4).
        - ``int >= 1`` → the integer itself.
        """
        if parallel is True:
            return DEFAULT_PARALLEL
        if not parallel:
            return 1
        if isinstance(parallel, int):
            return max(1, parallel)
        raise TypeError(
            "parallel must be a bool or non-negative int, "
            f"got {type(parallel).__name__}: {parallel!r}"
        )

    def _cancel_siblings(self, after_key: str | None) -> None:
        """Cancel every statement that appears after ``after_key``.

        When ``after_key`` is ``None`` every statement is cancelled.
        """
        seen = after_key is None
        for key, stmt in self.results.items():
            if not seen:
                if key == after_key:
                    seen = True
                continue
            try:
                stmt.cancel()
            except Exception:
                logger.debug(
                    "Failed to cancel sibling statement %r after %r",
                    key, after_key, exc_info=True,
                )

    def _cancel_in_flight(self) -> None:
        """Cancel every statement currently submitted to the backend."""
        for key, stmt in self._in_flight.items():
            try:
                stmt.cancel()
            except Exception:
                logger.debug(
                    "Failed to cancel in-flight statement %r", key, exc_info=True,
                )

    def _discard_pending(self) -> None:
        object.__setattr__(self, "_pending_queue", deque())

    def _clear_pool_state(self) -> None:
        object.__setattr__(self, "_in_flight", OrderedDict())
        object.__setattr__(self, "_pending_queue", deque())
        object.__setattr__(self, "_pool_runner", None)

    @staticmethod
    def _refresh_any_done(
        in_flight: Mapping[str, StatementResult],
        wait_cfg: WaitingConfig,
    ) -> tuple[str, StatementResult]:
        """Poll ``in_flight`` statements until at least one reaches a terminal state.

        Statements are refreshed in submission order; the first ``done`` one
        is returned as ``(key, statement)``.  Between full passes the
        configured backoff is applied via ``wait_cfg.sleep``.
        """
        iteration = 0
        start_ts = time.time()
        while True:
            for key, stmt in in_flight.items():
                try:
                    stmt.refresh_status()
                except Exception:
                    # Surface refresh errors as terminal for this statement so
                    # the caller can decide whether to re-raise.
                    return key, stmt
                if stmt.done:
                    return key, stmt
            wait_cfg.sleep(iteration=iteration, start=start_ts)
            iteration += 1

    def _submit(
        self,
        key: str,
        result: StatementResult,
        runner: Callable[[StatementResult], Any],
    ) -> None:
        runner(result)
        self._in_flight[key] = result

    def start(
        self,
        parallel: int | bool = True,
        *,
        wait: WaitingConfigArg = False,
        raise_error: bool = True,
        runner: Optional[Callable[[StatementResult], Any]] = None,
    ) -> StatementBatch:
        """Submit every statement result for execution.

        Parameters
        ----------
        parallel
            Concurrency window.  ``False`` (or ``0``) runs sequentially and
            waits on each statement before moving to the next.  ``True``
            defaults to a window of ``DEFAULT_PARALLEL`` (4).  An ``int >= 2``
            caps the number of statements that may be in flight on the
            backend at any one time.
        wait
            When falsy (default), ``start`` returns as soon as the window is
            filled.  The remaining statements stay queued and are submitted
            as slots free up inside :meth:`wait`.  When truthy, ``start``
            also drains every pending statement before returning.
        raise_error
            Whether to raise on per-statement failure.  When ``True`` and
            any statement fails, every in-flight and queued result is
            cancelled before the exception propagates.
        runner
            Optional callable invoked once per result instead of
            ``result.start(wait=False, raise_error=raise_error)``.  Useful for
            engines that need to resolve execution context (warehouse,
            catalog, …) before delegating to the backend.  Runners should
            submit without blocking — the batch handles polling.

        Notes
        -----
        ``start`` is not re-entrant: calling it twice on the same batch
        without first draining (:meth:`wait`) or tearing down (:meth:`cancel`)
        the active window raises ``RuntimeError``.
        """
        if self._in_flight or self._pending_queue:
            raise RuntimeError(
                "StatementBatch.start() was already called; call wait() or "
                "cancel() before starting again."
            )

        workers = self._resolve_parallel(parallel)

        def _default_runner(result: StatementResult) -> Any:
            # Prefer the bound engine when one is attached — lets callers
            # hand off execution context (catalog, warehouse, caching, etc.)
            # without wiring a custom runner.
            if self.engine is not None:
                return self.engine.execute(
                    result, wait=False, raise_error=raise_error,
                )
            return result.start(wait=False, raise_error=raise_error)

        effective_runner = runner or _default_runner

        try:
            if workers == 1 or len(self.results) <= 1:
                # Sequential submission: each statement is fully drained
                # before the next is submitted so later statements can
                # observe earlier writes.  The caller's ``wait`` only
                # applies to the final statement — every preceding one is
                # forced to ``wait=True``.
                items = list(self.results.items())
                last_index = len(items) - 1
                for idx, (key, stmt) in enumerate(items):
                    stmt_wait = wait if idx == last_index else True
                    effective_runner(stmt)
                    stmt.wait(wait=stmt_wait, raise_error=raise_error)
                return self

            object.__setattr__(self, "_pool_runner", effective_runner)
            object.__setattr__(self, "_pool_raise_error", raise_error)
            object.__setattr__(
                self,
                "_pending_queue",
                deque(self.results.items()),
            )

            # Fill the initial window.
            while self._pending_queue and len(self._in_flight) < workers:
                key, stmt = self._pending_queue.popleft()
                self._submit(key, stmt, effective_runner)

            if wait:
                self.wait(wait=wait, raise_error=raise_error)

            return self
        except Exception:
            # Any submission or wait error: tear everything down.  Cancel
            # in-flight statements, drop the pending queue, then run
            # :meth:`cancel` to cover every result in the batch (covers
            # the sequential path where earlier statements already
            # completed and are untracked by the pool).
            try:
                self.cancel()
            except Exception:
                logger.debug(
                    "StatementBatch.cancel() after start() failure raised",
                    exc_info=True,
                )
            if raise_error:
                raise
            return self

    def wait(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> StatementBatch:
        """Block until every statement reaches a terminal state.

        When a pool is active (``start`` was called without ``wait``), this
        drains it: as each in-flight statement completes, the next queued
        submission takes its slot.  A failure causes every remaining
        submission and every still-running statement to be cancelled before
        the original exception is re-raised (when ``raise_error`` is ``True``).

        When no pool is active, statements are waited on in submission order
        via their own ``wait()``.  A failure in one statement cancels every
        later statement before the exception propagates.
        """
        if self._in_flight or self._pending_queue:
            self._drain(wait=wait, raise_error=raise_error)
            return self

        for key, result in self.results.items():
            try:
                result.wait(wait=wait, raise_error=raise_error)
            except Exception:
                self._cancel_siblings(after_key=key)
                if raise_error:
                    raise
        return self

    def _drain(
        self,
        *,
        wait: WaitingConfigArg,
        raise_error: bool,
    ) -> None:
        wait_cfg = WaitingConfig.check_arg(wait) if wait else WaitingConfig.default()
        effective_runner = self._pool_runner

        first_exc: Optional[BaseException] = None
        failed_key: Optional[str] = None

        try:
            while self._in_flight:
                key, stmt = self._refresh_any_done(self._in_flight, wait_cfg)
                self._in_flight.pop(key, None)

                try:
                    stmt.raise_for_status()
                except Exception as exc:
                    first_exc = exc
                    failed_key = key
                    break

                if self._pending_queue and effective_runner is not None:
                    next_key, next_stmt = self._pending_queue.popleft()
                    try:
                        self._submit(next_key, next_stmt, effective_runner)
                    except Exception as exc:
                        first_exc = exc
                        failed_key = next_key
                        break

            if first_exc is not None:
                self._cancel_in_flight()
                for pkey, pstmt in self._pending_queue:
                    try:
                        pstmt.cancel()
                    except Exception:
                        logger.debug(
                            "Failed to cancel queued statement %r",
                            pkey, exc_info=True,
                        )
        finally:
            self._clear_pool_state()

        if first_exc is not None and raise_error:
            raise RuntimeError(
                f"PreparedStatement batch item {failed_key!r} failed."
            ) from first_exc

    def cancel(self) -> StatementBatch:
        """Cancel the active window and every underlying statement.

        Safe to call whether or not :meth:`start` has been invoked.  Queued
        statements are dropped, every in-flight statement is cancelled on the
        backend, and every remaining statement in the batch has its own
        ``cancel()`` invoked so backend handles are released even if the
        batch was started eagerly with ``parallel=False``.
        """
        self._cancel_in_flight()
        self._discard_pending()

        for key, stmt in self.results.items():
            try:
                stmt.cancel()
            except Exception:
                logger.debug(
                    "Failed to cancel statement %r", key, exc_info=True,
                )

        self._clear_pool_state()
        return self

    def persist(
        self,
        mode: Literal["arrow", "spark", "auto"] = "auto",
    ) -> StatementBatch:
        for result in self.results.values():
            result.persist(mode=mode)
        return self

    def to_arrow_readers(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schemas: Mapping[str, pa.Schema] | None = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> OrderedDict[str, pa.RecordBatchReader]:
        return OrderedDict(
            (
                key,
                result.to_arrow_reader(
                    max_workers=max_workers,
                    max_in_flight=max_in_flight,
                    batch_size=batch_size,
                    schema=None if schemas is None else schemas.get(key),
                    maintain_order=maintain_order,
                    stream=stream,
                ),
            )
            for key, result in self.results.items()
        )

    def to_arrow_batches(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schemas: Mapping[str, pa.Schema] | None = None,
        maintain_order: bool = False,
        stream: bool = True,
        concat: BatchConcatMode | None = "diagonal_relaxed",
    ) -> OrderedDict[str, list[pa.RecordBatch]] | list[pa.RecordBatch]:
        if concat is None:
            return OrderedDict(
                (
                    key,
                    list(
                        result.to_arrow_batches(
                            max_workers=max_workers,
                            max_in_flight=max_in_flight,
                            batch_size=batch_size,
                            schema=None if schemas is None else schemas.get(key),
                            maintain_order=maintain_order,
                            stream=stream,
                        )
                    ),
                )
                for key, result in self.results.items()
            )

        table = self.to_arrow_table(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schemas=schemas,
            maintain_order=maintain_order,
            stream=stream,
            concat=concat,
        )
        return table.to_batches(max_chunksize=batch_size) if batch_size else table.to_batches()

    def to_arrow_datasets(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schemas: Mapping[str, pa.Schema] | None = None,
        maintain_order: bool = False,
        stream: bool = True,
        concat: BatchConcatMode | None = "diagonal_relaxed",
    ) -> OrderedDict[str, "ds.Dataset"]:
        import pyarrow.dataset as pds

        if concat is None:
            return OrderedDict(
                (
                    key,
                    result.to_arrow_dataset(
                        max_workers=max_workers,
                        max_in_flight=max_in_flight,
                        batch_size=batch_size,
                        schema=None if schemas is None else schemas.get(key),
                        maintain_order=maintain_order,
                        stream=stream,
                    ),
                )
                for key, result in self.results.items()
            )

        table = self.to_arrow_table(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schemas=schemas,
            maintain_order=maintain_order,
            stream=stream,
            concat=concat,
        )
        return pds.dataset(table, schema=table.schema)

    def to_arrow_tables(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schemas: Mapping[str, pa.Schema] | None = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> OrderedDict[str, pa.Table]:
        return OrderedDict(
            (
                key,
                result.to_arrow_table(
                    max_workers=max_workers,
                    max_in_flight=max_in_flight,
                    batch_size=batch_size,
                    schema=None if schemas is None else schemas.get(key),
                    maintain_order=maintain_order,
                    stream=stream,
                ),
            )
            for key, result in self.results.items()
        )

    def to_arrow_table(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schemas: Mapping[str, pa.Schema] | None = None,
        maintain_order: bool = False,
        stream: bool = True,
        concat: BatchConcatMode | None = "diagonal_relaxed",
    ) -> OrderedDict[str, pa.Table] | pa.Table:
        tables = self.to_arrow_tables(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schemas=schemas,
            maintain_order=maintain_order,
            stream=stream,
        )

        if concat is None:
            return tables

        return _concat_arrow_tables(tables.values(), how=concat)

    def to_pandas(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schemas: Mapping[str, pa.Schema] | None = None,
        maintain_order: bool = False,
        stream: bool = True,
        concat: BatchConcatMode | None = "diagonal_relaxed",
    ) -> "OrderedDict[str, pandas.DataFrame] | pandas.DataFrame":
        result = self.to_arrow_table(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schemas=schemas,
            maintain_order=maintain_order,
            stream=stream,
            concat=concat,
        )

        if isinstance(result, OrderedDict):
            return OrderedDict(
                (key, table.to_pandas())
                for key, table in result.items()
            )

        return result.to_pandas()

    def to_polars(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        schemas: Mapping[str, pa.Schema] | None = None,
        maintain_order: bool = False,
        stream: bool = True,
        concat: BatchConcatMode | None = "diagonal_relaxed",
    ) -> "OrderedDict[str, polars.LazyFrame | polars.DataFrame] | polars.LazyFrame | polars.DataFrame":
        from ..polars.lib import polars

        if concat is None:
            return OrderedDict(
                (
                    key,
                    result.to_polars(
                        max_workers=max_workers,
                        max_in_flight=max_in_flight,
                        batch_size=batch_size,
                        schema=None if schemas is None else schemas.get(key),
                        maintain_order=maintain_order,
                        stream=stream,
                    ),
                )
                for key, result in self.results.items()
            )

        frames = [
            result.to_polars(
                max_workers=max_workers,
                max_in_flight=max_in_flight,
                batch_size=batch_size,
                schema=None if schemas is None else schemas.get(key),
                maintain_order=maintain_order,
                stream=False,
            )
            for key, result in self.results.items()
        ]

        if not frames:
            return polars.LazyFrame() if stream else polars.DataFrame()

        df = polars.concat(frames, how=concat)
        return df.lazy() if stream else df

    def to_spark(
        self,
        *,
        spark: Optional["pyspark.sql.SparkSession"] = None,
        prefer_cached: bool = True,
        cache_result: bool = True,
        concat: BatchConcatMode | None = "diagonal_relaxed",
    ) -> "OrderedDict[str, pyspark.sql.DataFrame] | pyspark.sql.DataFrame":
        if concat is None:
            return OrderedDict(
                (
                    key,
                    result.to_spark(
                        spark=spark,
                        prefer_cached=prefer_cached,
                        cache_result=cache_result,
                    ),
                )
                for key, result in self.results.items()
            )

        from yggdrasil.spark.cast import any_to_spark_dataframe

        table = self.to_arrow_table(concat=concat)
        return any_to_spark_dataframe(table, spark)


def _concat_arrow_tables(
    tables: Iterable[pa.Table],
    *,
    how: BatchConcatMode = "diagonal_relaxed",
) -> pa.Table:
    from ..polars.lib import polars

    table_list = [table for table in tables]
    if not table_list:
        return pa.table({})

    if len(table_list) == 1:
        return table_list[0]

    frames = [polars.from_arrow(table) for table in table_list]
    return polars.concat(frames, how=how).to_arrow()


def _is_spark_dataframe(value: object) -> bool:
    """Return True when ``value`` looks like a PySpark DataFrame without importing Spark eagerly."""
    return type(value).__module__.startswith("pyspark.sql")