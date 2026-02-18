from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Literal, Optional, Union

import pyarrow as pa
from yggdrasil.pyutils.waiting_config import WaitingConfig, WaitingConfigArg

if TYPE_CHECKING:
    import pandas
    import polars
    import pyarrow.dataset as ds
    import pyspark.sql


__all__ = ["StatementResult"]


@dataclass
class StatementResult(ABC):
    """Arrow-first result wrapper with fast, ergonomic conversions.

    This base class models a "statement-like" execution result whose best interchange
    format is Apache Arrow (RecordBatches / Tables).

    Implementations MUST provide:
    - `done`: terminal-state indicator
    - `failed`: failure/cancellation indicator
    - `raise_for_status()`: raise an exception on failure/cancellation (no-op on success)
    - `refresh_status()`: update status from the backend
    - `arrow_schema`: result schema
    - `to_arrow_reader()`: produce a `pyarrow.RecordBatchReader` over result batches

    Everything else (batches / Dataset / Table / pandas / Polars / Spark) is built on top.

    Notes
    -----
    - This class maintains optional, internal caches: `_arrow_table` and `_spark_df`.
      They are NOT dataclass fields on purpose, to avoid them appearing in repr/equality,
      and to make subclass dataclass layouts frictionless.
    - If `_spark_df` is present, conversions that go through Arrow (`toArrow()`) collect
      to the driver.
    - `stream` in this API means "prefer streaming / lazy execution" for consumers that
      support it (notably Polars). For Arrow itself, streaming behavior is determined by
      the concrete implementation of `to_arrow_reader()`.
    """

    # -------------------------------------------------------------------------
    # Lifecycle / internal caches (NOT dataclass fields)
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        # These are intentionally not dataclass fields.
        self._arrow_table: Optional[pa.Table] = None
        self._spark_df: Optional["pyspark.sql.DataFrame"] = None

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        """Iterate over Arrow record batches.

        Equivalent to calling `to_arrow_batches()` with default parameters.
        """
        return self.to_arrow_batches()

    # -------------------------------------------------------------------------
    # Pickling
    # -------------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Prepare pickle-ready state.

        Spark DataFrames are not pickleable. If `_spark_df` exists, it is converted to a
        local Arrow table and stored into `_arrow_table` in the pickled state.

        Returns
        -------
        dict
            Copy of instance state with Spark payload converted to Arrow when needed.

        Notes
        -----
        This conversion collects to the driver, so do not pickle instances holding huge
        Spark DataFrames unless that's explicitly intended.
        """
        state = self.__dict__.copy()
        spark_df = state.pop("_spark_df", None)

        if spark_df is not None:
            from ..spark.cast import spark_dataframe_to_arrow_table
            state["_arrow_table"] = spark_dataframe_to_arrow_table(spark_df, None)

        # Ensure stable unpickle shape.
        state.setdefault("_arrow_table", None)
        state.setdefault("_spark_df", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore instance state from pickle."""
        self.__dict__.update(state)
        # Spark DF cannot be restored; keep it None unless caller manually reattaches.
        self.__dict__.setdefault("_arrow_table", None)
        self.__dict__.setdefault("_spark_df", None)

    # -------------------------------------------------------------------------
    # Core state / caching
    # -------------------------------------------------------------------------

    @property
    def is_spark_sql(self) -> bool:
        """True if this result currently has a Spark DataFrame attached."""
        return self._spark_df is not None

    @property
    def persisted(self) -> bool:
        """Whether results are already materialized locally."""
        return self._arrow_table is not None or self._spark_df is not None

    def persist(self, mode: Literal["arrow", "spark"] = "arrow") -> "StatementResult":
        """Materialize and cache this result for cheaper subsequent conversions.

        Parameters
        ----------
        mode : {"arrow", "spark"}
            - "arrow": materialize to a local `pyarrow.Table` and store in `_arrow_table`.
            - "spark": materialize to a Spark DataFrame and store in `_spark_df`.

        Returns
        -------
        StatementResult
            Self (mutated in-place by setting `_arrow_table` or `_spark_df`).

        Notes
        -----
        Persisting to Spark may still collect to the driver first if the only available
        representation is Arrow.
        """
        if self.persisted:
            return self

        if mode == "arrow":
            self._arrow_table = self.to_arrow_table()
            return self

        if mode == "spark":
            self._spark_df = self.to_spark(prefer_cached=False, cache_result=False)
            return self

        raise ValueError(f"Unknown persist mode: {mode!r}. Expected 'arrow' or 'spark'.")

    # -------------------------------------------------------------------------
    # Execution lifecycle contract
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def done(self) -> bool:
        """True when the statement is in a terminal state."""
        raise NotImplementedError

    @property
    @abstractmethod
    def failed(self) -> bool:
        """True when the statement failed or was cancelled."""
        raise NotImplementedError

    @abstractmethod
    def raise_for_status(self) -> None:
        """Raise an exception if the statement failed/cancelled; no-op on success."""
        raise NotImplementedError

    @abstractmethod
    def refresh_status(self) -> None:
        """Refresh the statement status from the backend."""
        raise NotImplementedError

    def wait(self, wait: WaitingConfigArg = True, raise_error: bool = True) -> "StatementResult":
        """Block until the statement reaches a terminal state or timeout.

        Parameters
        ----------
        wait : WaitingConfigArg
            Waiting configuration.
            - True: use default waiting behavior (poll + timeout from defaults)
            - False: do not wait
            - WaitingConfig: custom waiting behavior (sleep/backoff/timeout)
        raise_error : bool
            If True, raise after reaching terminal state if failed/cancelled.

        Returns
        -------
        StatementResult
            Self.
        """
        wait_cfg = WaitingConfig.check_arg(wait)

        # Convention: falsy timeout => don't wait (or waiting disabled)
        if not wait_cfg.timeout:
            if raise_error:
                self.raise_for_status()
            return self

        iteration, start = 0, time.time()

        while not self.done:
            wait_cfg.sleep(iteration=iteration, start=start)
            iteration += 1
            self.refresh_status()

        if raise_error:
            self.raise_for_status()

        return self

    # -------------------------------------------------------------------------
    # Arrow contract
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def arrow_schema(self) -> pa.Schema:
        """Arrow schema for the result set."""
        raise NotImplementedError

    @abstractmethod
    def to_arrow_reader(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> pa.RecordBatchReader:
        """Return a streaming Arrow `RecordBatchReader` over the results.

        Parameters
        ----------
        max_workers : int
            Maximum parallel workers used for remote chunk downloads / decoding.
        max_in_flight : Optional[int]
            Backpressure limit for concurrently in-flight remote requests/batches.
        batch_size : Optional[int]
            Target batch size for client-side batching/coalescing. Hint only.
        schema : Optional[pyarrow.Schema]
            Override schema. Defaults to `self.arrow_schema`.
        maintain_order : bool
            Preserve original ordering of remote chunks/batches when streaming.
        stream : bool
            Prefer a streaming reader implementation (if the backend supports it).

        Returns
        -------
        pyarrow.RecordBatchReader
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Arrow / Dataset conversions
    # -------------------------------------------------------------------------

    def to_arrow_batches(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> Iterator[pa.RecordBatch]:
        """Iterate over result batches as `pyarrow.RecordBatch`."""
        resolved_schema = schema or self.arrow_schema

        reader = self.to_arrow_reader(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schema=resolved_schema,
            maintain_order=maintain_order,
            stream=stream,
        )

        yield from reader

    def to_arrow_dataset(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> "ds.Dataset":
        """Return an in-memory `pyarrow.dataset.Dataset` view of the results.

        Best bridge into lazy engines, especially Polars:
        `polars.scan_pyarrow_dataset(result.to_arrow_dataset(...))`.

        Notes
        -----
        If backed by Spark, `toArrow()` collects to the driver.
        """
        import pyarrow.dataset as pds

        resolved_schema = schema or self.arrow_schema

        if self.persisted:
            if self._arrow_table is not None:
                return pds.dataset(self._arrow_table, schema=resolved_schema)

            if self._spark_df is not None:
                tbl = self._spark_df.toArrow()
                batches = tbl.to_batches(max_chunksize=batch_size) if batch_size else tbl.to_batches()
                return pds.dataset(batches, schema=resolved_schema)

            raise NotImplementedError("Persisted without Arrow table or Spark DF")

        reader = self.to_arrow_reader(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schema=resolved_schema,
            maintain_order=maintain_order,
            stream=stream,
        )
        return pds.dataset(reader)

    def to_arrow_table(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> pa.Table:
        """Materialize results into a `pyarrow.Table` (collects all batches)."""
        if self.persisted:
            if self._arrow_table is not None:
                return self._arrow_table
            if self._spark_df is not None:
                return self._spark_df.toArrow()
            raise NotImplementedError("Persisted without Arrow table or Spark DF")

        resolved_schema = schema or self.arrow_schema

        reader = self.to_arrow_reader(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schema=resolved_schema,
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
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> "pandas.DataFrame":
        """Materialize into a pandas DataFrame via Arrow."""
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
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> Union["polars.LazyFrame", "polars.DataFrame"]:
        """Convert results to Polars.

        - stream=True  -> returns `polars.LazyFrame` scanning an Arrow Dataset
        - stream=False -> returns `polars.DataFrame` by collecting immediately
        """
        from ..polars.lib import polars

        arrow_dataset = self.to_arrow_dataset(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schema=schema,
            maintain_order=maintain_order,
            stream=stream,
        )

        lf = polars.scan_pyarrow_dataset(arrow_dataset)
        return lf if stream else lf.collect()

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
        """Convert results to a Spark DataFrame.

        Notes
        -----
        If `_spark_df` is not present, this materializes an Arrow table on the driver and
        then creates a Spark DataFrame. This is driver-memory-bound.
        """
        if prefer_cached and self._spark_df is not None:
            return self._spark_df

        from ..spark.cast import arrow_table_to_spark_dataframe

        sdf = arrow_table_to_spark_dataframe(self.to_arrow_table(), spark)

        if cache_result:
            self._spark_df = sdf

        return sdf
