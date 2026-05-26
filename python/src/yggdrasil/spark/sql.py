"""Lightweight Spark SQL statement — :class:`SparkSQL`.

Holds a SQL query string and lazily executes it against a Spark
session on :meth:`start` or first data access. Implements
:class:`Tabular` (read arrow/spark) and :class:`Awaitable`
(start/wait with state tracking).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

import pyarrow as pa

from yggdrasil.dataclasses.waiting import Awaitable, WaitingConfig
from yggdrasil.enums import State
from yggdrasil.io.tabular.base import Tabular

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
    from yggdrasil.data.schema import Schema

LOGGER = logging.getLogger(__name__)

__all__ = ["SparkSQL"]


class SparkSQL(Tabular, Awaitable):
    """Lazy Spark SQL statement with state tracking.

    Holds a SQL query string. Execution is deferred until
    :meth:`start` is called (or any read triggers it).
    The result DataFrame is cached after first execution.

    States: IDLE → RUNNING → SUCCEEDED / FAILED
    """

    __slots__ = ("_query", "_spark", "_frame", "_state", "_error")

    def __init__(
        self,
        query: str,
        spark: "SparkSession | None" = None,
    ) -> None:
        super().__init__()
        self._query = query
        self._spark = spark
        self._frame: Optional["SparkDataFrame"] = None
        self._state: State = State.IDLE
        self._error: Optional[Exception] = None

    @property
    def query(self) -> str:
        return self._query

    @property
    def state(self) -> State:
        return self._state

    @property
    def spark(self) -> "SparkSession":
        if self._spark is None:
            from yggdrasil.environ import PyEnv
            self._spark = PyEnv.spark_session()
        return self._spark

    @property
    def frame(self) -> "SparkDataFrame | None":
        self._ensure_started()
        return self._frame

    @property
    def done(self) -> bool:
        return self._state.is_done

    @property
    def failed(self) -> bool:
        return self._state.is_failed

    @property
    def ok(self) -> bool:
        return self._state == State.SUCCEEDED

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    def __repr__(self) -> str:
        return f"SparkSQL(state={self._state.name}, query={self._query[:60]!r})"

    # ------------------------------------------------------------------
    # Awaitable
    # ------------------------------------------------------------------

    def _start(self) -> "SparkSQL":
        if self._state != State.IDLE:
            return self
        self._state = State.RUNNING
        try:
            LOGGER.debug("Executing Spark SQL: %s", self._query[:200])
            self._frame = self.spark.sql(self._query)
            self._state = State.SUCCEEDED
        except Exception as exc:
            self._state = State.FAILED
            self._error = exc
            LOGGER.error("Spark SQL failed: %s", exc)
        return self

    def _wait(self, config: WaitingConfig, raise_error: bool = True) -> "SparkSQL":
        self._ensure_started()
        if raise_error and self._error is not None:
            raise self._error
        return self

    def _ensure_started(self) -> None:
        if self._state == State.IDLE:
            self._start()

    # ------------------------------------------------------------------
    # Tabular
    # ------------------------------------------------------------------

    def _collect_schema(self, options=None) -> "Schema":
        from yggdrasil.data.schema import Schema
        self._ensure_started()
        if self._frame is not None:
            return Schema.from_arrow(
                self._frame.schema if hasattr(self._frame.schema, 'names')
                else self._frame.toPandas().dtypes
            )
        return Schema.empty()

    def _read_arrow_batches(self, options=None) -> Iterator[pa.RecordBatch]:
        self._ensure_started()
        if self._frame is None:
            return
        table = self._frame.toArrow() if hasattr(self._frame, 'toArrow') else pa.Table.from_pandas(self._frame.toPandas())
        yield from table.to_batches()

    def _write_arrow_batches(self, batches, options=None):
        raise NotImplementedError("SparkSQL is read-only")

    def _read_spark_frame(self, options=None) -> "SparkDataFrame":
        self._ensure_started()
        if self._frame is None:
            return self.spark.createDataFrame([], schema="")
        return self._frame

    def _count(self, options=None) -> int:
        self._ensure_started()
        if self._frame is None:
            return 0
        if options is not None:
            return options.cast_spark_tabular(self._frame).count()
        return self._frame.count()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def collect(self) -> pa.Table:
        """Execute and collect to a pyarrow Table."""
        return self.read_arrow_table()

    def show(self, n: int = 20) -> None:
        """Print first *n* rows (delegates to Spark's show)."""
        self._ensure_started()
        if self._frame is not None:
            self._frame.show(n)
