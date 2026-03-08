from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Literal, Optional, Union

import pyarrow as pa

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg

if TYPE_CHECKING:
    import pandas
    import polars
    import pyarrow.dataset as ds
    import pyspark.sql


__all__ = ["StatementResult"]


@dataclass(frozen=True)
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
    - ``arrow_schema``: Arrow schema for the result
    - ``to_arrow_reader()``: stream the result as Arrow record batches

    Notes
    -----
    - ``stream=True`` means "prefer lazy / streaming consumers where possible".
      It does not guarantee that the backend itself is fully streaming.
    - Some conversions materialize data locally and may collect all rows to the driver.
      Those cases are called out in method docs.
    """

    _arrow_schema: Optional[pa.Schema] = field(init=False, default=None, repr=False, compare=False)
    _arrow_table: Optional[pa.Table] = field(init=False, default=None, repr=False, compare=False)
    _spark_df: Optional["pyspark.sql.DataFrame"] = field(init=False, default=None, repr=False, compare=False)

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
            "_arrow_schema": self._arrow_schema,
            "_arrow_table": self._arrow_table,
            "_spark_df": None,
        }

        if self._spark_df is not None:
            from ..spark.cast import spark_dataframe_to_arrow_table

            state["_arrow_table"] = spark_dataframe_to_arrow_table(self._spark_df, None)

        return state

    def __setstate__(self, state: dict) -> None:
        """Restore pickle-safe instance state."""
        for name in ("_arrow_schema", "_arrow_table", "_spark_df"):
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
    ) -> "StatementResult":
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
            if isinstance(data, pa.Table):
                object.__setattr__(self, "_arrow_table", data)
                object.__setattr__(self, "_spark_df", None)
                object.__setattr__(self, "_arrow_schema", data.schema)
                return self

            if _is_spark_dataframe(data):
                from yggdrasil.spark.cast import spark_schema_to_arrow_schema

                object.__setattr__(self, "_spark_df", data)
                object.__setattr__(self, "_arrow_table", None)
                object.__setattr__(self, "_arrow_schema", spark_schema_to_arrow_schema(data.schema))
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

        raise ValueError(f"Unknown persist mode: {mode!r}. Expected 'auto', 'arrow', or 'spark'.")

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

    def wait(self, wait: WaitingConfigArg = True, raise_error: bool = True) -> "StatementResult":
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

        return self

    # -------------------------------------------------------------------------
    # Arrow contract
    # -------------------------------------------------------------------------

    @property
    def arrow_schema(self) -> pa.Schema:
        """Arrow schema for the result."""
        if self._arrow_schema is None:
            schema = self.make_arrow_schema()
            object.__setattr__(self, "_arrow_schema", schema)
        return self._arrow_schema

    @abstractmethod
    def make_arrow_schema(self):
        """
        Generate and cache the Arrow schema for the result.
        """
        pass

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
        """Return an Arrow ``RecordBatchReader`` for the result.

        Parameters
        ----------
        max_workers:
            Maximum number of workers for remote fetch / decode.
        max_in_flight:
            Optional backpressure limit for concurrent in-flight work.
        batch_size:
            Optional target batch size for client-side batching.
        schema:
            Optional schema override. Defaults to ``self.arrow_schema``.
        maintain_order:
            Whether to preserve remote batch order.
        stream:
            Whether to prefer a streaming implementation where supported.
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
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> "ds.Dataset":
        """Return an in-memory ``pyarrow.dataset.Dataset`` view of the result.

        This is a handy bridge into lazy engines such as Polars via
        ``polars.scan_pyarrow_dataset(...)``.

        Notes
        -----
        - If backed by Arrow, this wraps the local Arrow table.
        - If backed by Spark, this collects data to the driver through ``toArrow()``.
        """
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
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> pa.Table:
        """Materialize the full result as a local Arrow table.

        Warning
        -------
        This collects all rows into memory.
        """
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
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
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
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> Union["polars.LazyFrame", "polars.DataFrame"]:
        """Convert the result to Polars.

        Returns
        -------
        polars.LazyFrame | polars.DataFrame
            - ``stream=True`` returns a ``LazyFrame``
            - ``stream=False`` collects immediately and returns a ``DataFrame``
        """
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
        """Convert the result to a Spark DataFrame.

        Parameters
        ----------
        spark:
            Optional Spark session to use when creating the DataFrame.
        prefer_cached:
            Whether to return a cached Spark DataFrame if present.
        cache_result:
            Whether to store the created Spark DataFrame in ``_spark_df``.

        Notes
        -----
        If no cached Spark DataFrame exists, this first materializes a local Arrow table
        and then creates a Spark DataFrame from it. That path is driver-memory-bound.
        """
        if prefer_cached and self._spark_df is not None:
            return self._spark_df

        from ..spark.cast import arrow_table_to_spark_dataframe

        spark_df = arrow_table_to_spark_dataframe(self.to_arrow_table(), spark)

        if cache_result:
            object.__setattr__(self, "_spark_df", spark_df)

        return spark_df


def _is_spark_dataframe(value: object) -> bool:
    """Return True when ``value`` looks like a PySpark DataFrame without importing Spark eagerly."""
    return type(value).__module__.startswith("pyspark.sql")