from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import pyarrow as pa
from yggdrasil.data import Schema
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas
    import polars
    import pyarrow.dataset as ds
    import pyspark.sql

BatchConcatMode = Literal[
    "vertical",
    "vertical_relaxed",
    "diagonal",
    "diagonal_relaxed",
]

__all__ = [
    "BatchConcatMode",
    "StatementResult",
    "StatementResultBatch",
]


@dataclass
class StatementResult(ABC):
    """Arrow-first wrapper around a statement execution result.

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
    - ``make_data_schema()``: schema for the result
    - ``to_arrow_reader()``: stream the result as Arrow record batches

    Notes
    -----
    - ``stream=True`` means "prefer lazy / streaming consumers where possible".
      It does not guarantee that the backend itself is fully streaming.
    - Some conversions materialize data locally and may collect all rows to the driver.
      Those cases are called out in method docs.
    """

    _data_schema: Optional[Schema] = field(init=False, default=None, repr=False, compare=False)
    _arrow_table: Optional[pa.Table] = field(init=False, default=None, repr=False, compare=False)
    _spark_df: Optional["pyspark.sql.DataFrame"] = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _temporary_tables: tuple[Any, ...] = field(
        init=False,
        default=(),
        repr=False,
        compare=False,
    )
    _temporary_tables_cleaned: bool = field(
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
            self._maybe_cleanup_temporary_tables()
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

        self._maybe_cleanup_temporary_tables()

        return self

    # -------------------------------------------------------------------------
    # Temporary table cleanup
    # -------------------------------------------------------------------------

    def attach_temporary_tables(self, tables: Iterable[Any]) -> StatementResult:
        """Attach temporary staging resources to be cleaned up when ``done``.

        Each entry must expose ``cleanup(allow_not_found: bool = True)``.
        Cleanup is best-effort and idempotent; it runs lazily the first time
        the statement reaches a terminal state (see ``_maybe_cleanup_temporary_tables``).
        """
        items = tuple(tables)
        if not items:
            return self
        object.__setattr__(
            self,
            "_temporary_tables",
            tuple(self._temporary_tables) + items,
        )
        object.__setattr__(self, "_temporary_tables_cleaned", False)
        return self

    def _maybe_cleanup_temporary_tables(self) -> None:
        if self._temporary_tables_cleaned or not self._temporary_tables:
            return
        try:
            is_done = self.done
        except Exception:
            return
        if not is_done:
            return

        for resource in self._temporary_tables:
            try:
                resource.cleanup(allow_not_found=True)
            except Exception:
                logger.debug(
                    "Failed to cleanup temporary staging resource %r",
                    resource,
                    exc_info=True,
                )
        object.__setattr__(self, "_temporary_tables_cleaned", True)
        object.__setattr__(self, "_temporary_tables", ())

    # -------------------------------------------------------------------------
    # Arrow contract
    # -------------------------------------------------------------------------

    @property
    def data_schema(self) -> Schema:
        if self._data_schema is None:
            schema = self.make_data_schema()
            object.__setattr__(self, "_data_schema", schema)
        return self._data_schema

    @property
    def arrow_schema(self) -> pa.Schema:
        return self.data_schema.to_arrow_schema()

    @abstractmethod
    def make_data_schema(self) -> Schema:
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
class StatementResultBatch(Mapping[str, StatementResult]):
    """Ordered batch wrapper around multiple statement results.

    By default, materialized conversions concatenate inner tabular results using
    Polars ``how="diagonal_relaxed"`` semantics.

    Use ``concat=None`` to preserve per-statement outputs.
    """

    results: OrderedDict[str, StatementResult]

    @classmethod
    def from_results(
        cls,
        results: Iterable[StatementResult] | Mapping[str, StatementResult],
    ) -> StatementResultBatch:
        if isinstance(results, Mapping):
            return cls(results=OrderedDict(results.items()))

        return cls(
            results=OrderedDict(
                (str(i), result)
                for i, result in enumerate(results)
            )
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, key: str) -> StatementResult:
        return self.results[key]

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

    def refresh_status(self) -> StatementResultBatch:
        for result in self.results.values():
            result.refresh_status()
        return self

    def raise_for_status(self) -> StatementResultBatch:
        for key, result in self.results.items():
            try:
                result.raise_for_status()
            except Exception as exc:
                raise RuntimeError(f"Statement batch item {key!r} failed.") from exc
        return self

    def wait(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> StatementResultBatch:
        for result in self.results.values():
            result.wait(wait=wait, raise_error=raise_error)
        return self

    def persist(
        self,
        mode: Literal["arrow", "spark", "auto"] = "auto",
    ) -> StatementResultBatch:
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
    ) -> OrderedDict[str, "pandas.DataFrame"] | "pandas.DataFrame":
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
    ) -> OrderedDict[str, "polars.LazyFrame | polars.DataFrame"] | "polars.LazyFrame | polars.DataFrame":
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
    ) -> OrderedDict[str, "pyspark.sql.DataFrame"] | "pyspark.sql.DataFrame":
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