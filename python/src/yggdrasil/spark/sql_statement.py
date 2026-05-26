from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.dataclasses.awaitable import Awaitable
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.enums import MimeTypes, State
from yggdrasil.io.tabular.base import Tabular

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

__all__ = ["SparkSQLStatement"]


class SparkSQLStatement(Tabular, Awaitable):
    def __init__(
        self,
        text: str,
        *,
        spark_session: Optional["SparkSession"] = None,
        row_limit: Optional[int] = None,
    ):
        self.text = text
        self.spark_session = spark_session
        self.row_limit = row_limit
        self._dataframe: Optional["DataFrame"] = None
        self._failure: Optional[BaseException] = None
        self._cached_schema: Optional[Schema] = None

    @classmethod
    def default_media_type(cls):
        return MimeTypes.SPARK_SQL_STATEMENT

    def _resolve_session(self) -> "SparkSession":
        if self.spark_session is not None:
            return self.spark_session
        from yggdrasil.environ import PyEnv
        return PyEnv.spark_session(create=True, import_error=True)

    # -- Awaitable hooks -----------------------------------------------------

    def _start(self) -> None:
        session = self._resolve_session()
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("session.sql:\n%s", self.text)
            df = session.sql(self.text)
            if self.row_limit:
                df = df.limit(self.row_limit)
            self._dataframe = df
            self._failure = None
            self._state = State.SUCCEEDED
        except BaseException as exc:
            self._dataframe = None
            self._failure = exc
            self._state = State.FAILED

    def _poll(self) -> None:
        pass

    def _error_for_status(self) -> BaseException | None:
        return self._failure

    @property
    def retryable(self) -> bool:
        return False

    def cancel(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "SparkSQLStatement":
        return self

    # -- Tabular hooks -------------------------------------------------------

    @property
    def dataframe(self) -> Optional["DataFrame"]:
        return self._dataframe

    def _native_spark_frame(self) -> Optional["DataFrame"]:
        return self._dataframe

    def _collect_schema(self, options: CastOptions) -> Schema:
        if self._cached_schema is None:
            if self._dataframe is None:
                raise RuntimeError("Cannot collect schema before start()")
            self._cached_schema = Schema.from_(self._dataframe)
        return self._cached_schema

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        if self._dataframe is None:
            raise RuntimeError("Cannot read before start()")
        from yggdrasil.spark.cast import spark_dataframe_to_arrow
        yield from spark_dataframe_to_arrow(self._dataframe).to_batches()

    def _read_spark_frame(self, options: CastOptions) -> "DataFrame":
        if self._dataframe is None:
            raise RuntimeError("Cannot read before start()")
        return self._dataframe

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        raise NotImplementedError("SparkSQLStatement is read-only")

    def __repr__(self) -> str:
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"<SparkSQLStatement state={self._state} sql={preview!r}>"
