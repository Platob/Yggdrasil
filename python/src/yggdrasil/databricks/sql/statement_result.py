"""Result wrapper for Databricks SQL statement execution.

This module provides a `StatementResult` object that encapsulates:
- Databricks SQL statement execution status tracking
- Retrieval of results via Databricks "EXTERNAL_LINKS" disposition (Arrow IPC streams)
- Conversion into Arrow, pandas, Polars, and Spark representations

Design goals:
- Fast path for cached results (persisted Arrow table or Spark DataFrame)
- Streaming-friendly APIs (RecordBatch iterator / RecordBatchReader / Dataset)
- Parallel download of external result chunks with bounded in-flight work
"""

import dataclasses
import time
from typing import Optional, Iterator, TYPE_CHECKING, Iterable, List

import pyarrow as pa
import pyarrow.ipc as pipc
import urllib3
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    StatementState,
    StatementResponse,
    Disposition,
    StatementStatus,
    ExternalLink,
)

from .exceptions import SqlStatementError
from .types import column_info_to_arrow_field
from ...concurrent.threading import JobThreadPoolExecutor, Job
from ...pyutils.waiting_config import WaitingConfigArg, WaitingConfig
from ...sql.statement_result import StatementResult as BaseStatementResult

if TYPE_CHECKING:
    import polars
    import pandas
    import pyspark
    import pyarrow.dataset as pds


DONE_STATES = {
    StatementState.CANCELED,
    StatementState.CLOSED,
    StatementState.FAILED,
    StatementState.SUCCEEDED,
}

FAILED_STATES = {
    StatementState.FAILED,
    StatementState.CANCELED,
}

__all__ = ["StatementResult"]


@dataclasses.dataclass
class StatementResult(BaseStatementResult):
    """Wrapper around Databricks SQL statement execution result.

    This class models a statement execution (identified by `statement_id`) and provides:

    - Status polling (`response`, `status`, `state`, `wait`, `raise_for_status`)
    - Result retrieval (`external_links`, `to_arrow_batches`)
    - Conversion helpers:
        - `to_arrow_table` (materialize)
        - `to_arrow_reader` (streaming reader)
        - `to_arrow_dataset` (scanable dataset)
        - `to_pandas`, `to_polars_lazy`, `to_polars`, `to_spark`

    Caching / persistence:
    - If `_spark_df` is set, this instance represents Spark SQL output.
    - If `_arrow_table` is set, this instance represents a cached Arrow materialization.
    - `persist()` materializes and caches Arrow locally for reuse.

    Notes on ordering:
    - Unless your SQL query has an ORDER BY, the concept of "row order" is not meaningful.
    - For throughput, external chunks are fetched concurrently and may be yielded out of order.
      Use `maintain_order=True` only when you must preserve external link ordering.
    """

    workspace_client: WorkspaceClient
    warehouse_id: str
    statement_id: str
    disposition: Disposition

    _response: Optional[StatementResponse] = dataclasses.field(default=None, repr=False)

    _spark_df: Optional["pyspark.sql.DataFrame"] = dataclasses.field(default=None, repr=False)
    _arrow_table: Optional[pa.Table] = dataclasses.field(default=None, repr=False)

    # ----------------------------
    # Pickling
    # ----------------------------

    def __getstate__(self):
        """Prepare pickle-ready state.

        Spark DataFrames are not pickleable; when `_spark_df` exists, it is converted to an Arrow table.

        Returns
        -------
        dict
            Copy of instance state with Spark payload converted to Arrow.
        """
        state = self.__dict__.copy()
        spark_df = state.pop("_spark_df", None)

        if spark_df is not None:
            from ...spark.cast import spark_dataframe_to_arrow_table
            state["_arrow_table"] = spark_dataframe_to_arrow_table(spark_df, None)

        return state

    def __setstate__(self, state):
        """Restore instance state from pickle.

        Parameters
        ----------
        state : dict
            The serialized state dictionary.
        """
        # Ensure missing fields are reintroduced if needed
        self.__dict__.update(state)
        # `_spark_df` cannot be restored; it remains None after unpickling
        self.__dict__.setdefault("_spark_df", None)

    # ----------------------------
    # Dunder / convenience
    # ----------------------------

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        """Iterate over Arrow record batches.

        Equivalent to calling `to_arrow_batches()` with default parameters.
        """
        return self.to_arrow_batches()

    def __repr__(self) -> str:
        return "StatementResult(url='%s')" % self.monitoring_url

    def __str__(self) -> str:
        return self.monitoring_url

    @property
    def monitoring_url(self) -> str:
        """Databricks UI monitoring URL for this statement execution."""
        return "%s/sql/warehouses/%s/monitoring?queryId=%s" % (
            self.workspace_client.config.host,
            self.warehouse_id,
            self.statement_id,
        )

    # ----------------------------
    # State / status
    # ----------------------------

    @property
    def is_spark_sql(self) -> bool:
        """True if this result was produced by Spark SQL (local Spark DataFrame attached)."""
        return self._spark_df is not None

    @property
    def response(self) -> StatementResponse:
        """Latest statement response, auto-refreshing until terminal.

        For Spark SQL results, returns a synthetic SUCCEEDED response.

        Returns
        -------
        StatementResponse
            The current response payload (possibly refreshed).
        """
        if self.is_spark_sql:
            return StatementResponse(
                statement_id=self.statement_id or "sparksql",
                status=StatementStatus(state=StatementState.SUCCEEDED),
            )

        if not self.statement_id:
            return StatementResponse(
                statement_id="unknown",
                status=StatementStatus(state=StatementState.PENDING),
            )

        statement_execution = self.workspace_client.statement_execution

        if self._response is None:
            self._response = statement_execution.get_statement(self.statement_id)
        elif self._response.status.state not in DONE_STATES:
            self._response = statement_execution.get_statement(self.statement_id)

        return self._response

    def api_result_data_at_index(self, chunk_index: int):
        """Fetch a specific result chunk by chunk index via the Databricks SDK."""
        return self.workspace_client.statement_execution.get_statement_result_chunk_n(
            statement_id=self.statement_id,
            chunk_index=chunk_index,
        )

    @property
    def status(self) -> StatementStatus:
        """Statement status object."""
        return self.response.status

    @property
    def state(self) -> StatementState:
        """Current statement state enum."""
        return self.status.state

    @property
    def done(self) -> bool:
        """True when the statement is in a terminal state."""
        return self.state in DONE_STATES

    @property
    def failed(self) -> bool:
        """True when the statement failed or was cancelled."""
        return self.state in FAILED_STATES

    def raise_for_status(self) -> "StatementResult":
        """Raise `SqlStatementError` if the statement is in a failed state."""
        if self.failed:
            raise SqlStatementError.from_statement(self)
        return self

    def wait(self, wait: WaitingConfigArg = True, raise_error: bool = True) -> "StatementResult":
        """Block until the statement reaches a terminal state or timeout.

        Parameters
        ----------
        wait : WaitingConfigArg
            Waiting configuration. Use `True` for defaults, `False` for no waiting,
            or a `WaitingConfig` instance for custom behavior.
        raise_error : bool
            If True, raise when execution is terminal and failed/cancelled.

        Returns
        -------
        StatementResult
            Self.
        """
        wait_cfg = WaitingConfig.check_arg(wait)
        if not wait_cfg.timeout:
            if raise_error:
                self.raise_for_status()
            return self

        iteration, start = 0, time.time()

        while not self.done:
            wait_cfg.sleep(iteration=iteration, start=start)
            iteration += 1
            # Trigger refresh
            _ = self.response

        if raise_error:
            self.raise_for_status()

        return self

    # ----------------------------
    # Manifest / schema
    # ----------------------------

    @property
    def manifest(self):
        """SQL result manifest (Databricks API), if available.

        Returns None for Spark SQL results.
        """
        self.wait()
        return self.response.manifest

    @property
    def result(self):
        """Raw statement result payload from the Databricks API."""
        self.wait()
        return self.response.result

    @property
    def persisted(self) -> bool:
        """True when this result is cached locally (Arrow table or Spark DataFrame)."""
        return self._spark_df is not None or self._arrow_table is not None

    def persist(self) -> "StatementResult":
        """Materialize and cache this result as a local Arrow table.

        Returns
        -------
        StatementResult
            Self (with `_arrow_table` set).
        """
        if not self.persisted:
            self._arrow_table = self.to_arrow_table()
        return self

    def arrow_schema(self) -> pa.Schema:
        """Return the Arrow schema for the result.

        Strategy:
        - If cached Arrow table exists, reuse its schema.
        - If Spark DF exists, convert Spark schema → Arrow schema.
        - Else, derive schema from the Databricks SQL manifest.

        Returns
        -------
        pyarrow.Schema
            Arrow schema with metadata including source and statement id.
        """
        if self.persisted:
            if self._arrow_table is not None:
                return self._arrow_table.schema
            if self._spark_df is not None:
                from ...spark.cast import spark_schema_to_arrow_schema
                return spark_schema_to_arrow_schema(self._spark_df.schema, None)
            raise NotImplementedError("Persisted without Arrow table or Spark DF")

        manifest = self.manifest
        metadata = {"source": "databricks-sql", "statement_id": self.statement_id or ""}

        if manifest is None:
            return pa.schema([], metadata=metadata)

        fields = [column_info_to_arrow_field(c) for c in manifest.schema.columns]
        return pa.schema(fields, metadata=metadata)

    # ----------------------------
    # External links
    # ----------------------------

    def external_links(self) -> Iterator[ExternalLink]:
        """Yield external result links for `Disposition.EXTERNAL_LINKS`.

        Databricks may paginate external links; this generator follows the internal
        `next_chunk_internal_link` chain until exhausted.

        Yields
        ------
        ExternalLink
            External links in API chunk order.
        """
        assert self.disposition == Disposition.EXTERNAL_LINKS, (
            "Cannot get external links from %s, disposition %s != %s"
            % (self, self.disposition, Disposition.EXTERNAL_LINKS)
        )

        result_data = self.result
        wsdk = self.workspace_client

        while True:
            links = result_data.external_links or []
            if not links:
                return

            for link in links:
                yield link

            next_internal = getattr(links[-1], "next_chunk_internal_link", None)
            if not next_internal:
                return

            try:
                chunk_index = int(next_internal.rstrip("/").split("/")[-1])
            except Exception as e:
                raise ValueError(f"Bad next_chunk_internal_link {next_internal!r}: {e}")

            try:
                result_data = wsdk.statement_execution.get_statement_result_chunk_n(
                    statement_id=self.statement_id,
                    chunk_index=chunk_index,
                )
            except Exception as e:
                raise ValueError(f"Cannot retrieve data batch from {next_internal!r}: {e}")

    # ----------------------------
    # Arrow conversions
    # ----------------------------

    def to_arrow_table(self, max_workers: int = 4) -> pa.Table:
        """Materialize the full result set into a single Arrow Table.

        This loads all batches into memory. Prefer `to_arrow_reader()` or
        `to_arrow_dataset()` if you want to keep a streaming or lazy pipeline.

        Parameters
        ----------
        max_workers : int
            Maximum number of parallel workers used to fetch external chunks.

        Returns
        -------
        pyarrow.Table
            Fully materialized table.
        """
        if self.persisted:
            if self._arrow_table is not None:
                return self._arrow_table
            return self._spark_df.toArrow()

        return self.to_arrow_reader(
            max_workers=max_workers
        ).read_all()

    def to_arrow_batches(
        self,
        max_workers: int = 4,
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        maintain_order: bool = False,
    ) -> Iterator[pa.RecordBatch]:
        """Stream results as Arrow RecordBatches.

        Parameters
        ----------
        max_workers : int
            Maximum number of parallel workers for external chunk downloads.
        max_in_flight : Optional[int]
            Max number of futures allowed in-flight (backpressure). If None, executor default.
        batch_size : Optional[int]
            Optional chunk size (rows) when splitting cached tables into batches.
            Does not change batch sizing of remote chunks (those are server-defined).
        maintain_order : bool
            If True, preserve external link completion order (lower throughput).
            If False, yield as chunks complete (higher throughput).

        Yields
        ------
        pyarrow.RecordBatch
            Record batches of result rows.
        """
        if self.persisted:
            if self._arrow_table is not None:
                yield from self._arrow_table.to_batches(max_chunksize=batch_size)
                return
            if self._spark_df is not None:
                yield from self._spark_df.toArrow().to_batches(max_chunksize=batch_size)
                return
            raise NotImplementedError("Persisted without Arrow table or Spark DF")

        # HTTP pool + retries tuned for many small-ish GETs
        retry = urllib3.Retry(
            total=3,
            backoff_factor=0.2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )

        http = urllib3.PoolManager(
            num_pools=64,
            maxsize=64,
            retries=retry,
            timeout=urllib3.Timeout(connect=2.0, read=30.0),
            headers={"Accept-Encoding": "gzip,deflate"},
            cert_reqs="CERT_REQUIRED",
        )

        def extract_batches(url: str) -> List[pa.RecordBatch]:
            """Download one Arrow IPC stream from `url` and return its batches.

            Implementation is streaming-friendly: we iterate the IPC reader and collect batches.
            We still keep the response bytes in memory because urllib3 returns `resp.data`
            unless you do chunked streaming; if blobs get huge, we can switch to `preload_content=False`
            and stream into a buffer output stream.
            """
            resp = http.request("GET", url, preload_content=True)
            try:
                if resp.status >= 400:
                    raise RuntimeError(f"GET {url} failed: {resp.status}")
                buf = memoryview(resp.data)
            finally:
                resp.release_conn()

            with pa.input_stream(buf) as src:
                reader = pipc.open_stream(src).read_all()
                return reader.to_batches()

        def jobs() -> Iterable[Job]:
            for link in self.external_links():
                if link.external_link:
                    yield Job.make(extract_batches, link.external_link)

        with JobThreadPoolExecutor(max_workers=max_workers or 4) as ex:
            for fut in ex.as_completed(
                jobs(),
                ordered=maintain_order,
                max_in_flight=max_in_flight,
                cancel_on_exit=True,
                shutdown_on_exit=True,
                shutdown_wait=False,
            ):
                for rb in fut.result():
                    yield rb

    def to_arrow_reader(
        self,
        max_workers: int = 4,
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
    ) -> pa.RecordBatchReader:
        """Return a streaming Arrow RecordBatchReader for this result.

        This is the most "Arrow-native" streaming interface and avoids building a full table.

        Parameters
        ----------
        max_workers : int
            Max parallel workers for remote chunk downloads.
        max_in_flight : Optional[int]
            Backpressure for futures in-flight.
        batch_size : Optional[int]
            Batch sizing for cached tables (Arrow table / Spark Arrow table).
        schema : Optional[pyarrow.Schema]
            Override schema. Defaults to `self.arrow_schema()`.
        maintain_order : bool
            Preserve external link ordering when streaming remote chunks.

        Returns
        -------
        pyarrow.RecordBatchReader
            Single-pass reader. Iterating exhausts the underlying stream.
        """
        schema = schema or self.arrow_schema()

        if self.persisted:
            if self._arrow_table is not None:
                return self._arrow_table.to_reader(max_chunksize=batch_size)
            if self._spark_df is not None:
                return self._spark_df.toArrow().to_reader(max_chunksize=batch_size)
            raise NotImplementedError("Persisted without Arrow table or Spark DF")

        batches_iter = self.to_arrow_batches(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            maintain_order=maintain_order,
        )
        return pa.RecordBatchReader.from_batches(schema, batches_iter)  # type: ignore[arg-type]

    def to_arrow_dataset(
        self,
        max_workers: int = 4,
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
    ) -> "pds.Dataset":
        """Return a PyArrow Dataset view of the results (in-memory).

        This avoids materializing a full `pa.Table` and is the best bridge into lazy engines,
        especially Polars: `polars.scan_pyarrow_dataset(...)`.

        Parameters
        ----------
        max_workers : int
            Max parallel workers for remote chunk downloads.
        max_in_flight : Optional[int]
            Backpressure for in-flight remote downloads.
        batch_size : Optional[int]
            Batch sizing for cached tables (does not affect server-side chunking).
        schema : Optional[pyarrow.Schema]
            Override schema. Defaults to `self.arrow_schema()`.
        maintain_order : bool
            Preserve external link ordering when streaming remote chunks.

        Returns
        -------
        pyarrow.dataset.Dataset
            An in-memory dataset backed by record batches.
        """
        import pyarrow.dataset as ds
        schema = schema or self.arrow_schema()

        if self.persisted:
            if self._arrow_table is not None:
                return ds.dataset(self._arrow_table, schema=schema)
            if self._spark_df is not None:
                tbl = self._spark_df.toArrow()
                return ds.dataset(tbl.to_batches(max_chunksize=batch_size), schema=schema)
            raise NotImplementedError("Persisted without Arrow table or Spark DF")

        reader = self.to_arrow_reader(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            schema=schema,
            maintain_order=maintain_order,
        )

        # Public API: dataset() can take a RecordBatchReader or iterable of RecordBatch
        return ds.dataset(reader)

    # ----------------------------
    # pandas / Polars / Spark
    # ----------------------------

    def to_pandas(self, max_workers: int = 4) -> "pandas.DataFrame":
        """Materialize results into a pandas DataFrame.

        Parameters
        ----------
        max_workers : int
            Parallelism for remote downloads.

        Returns
        -------
        pandas.DataFrame
        """
        return self.to_arrow_table(max_workers=max_workers).to_pandas()

    def to_polars_lazy(
        self,
        max_workers: int = 4,
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        allow_pyarrow_filter: bool = True,
        maintain_order: bool = False,
    ) -> "polars.LazyFrame":
        """Return results as a Polars LazyFrame without materializing a full Arrow table.

        This builds a PyArrow Dataset and scans it lazily in Polars.

        Parameters
        ----------
        max_workers : int
            Parallelism for remote downloads.
        max_in_flight : Optional[int]
            Backpressure for remote downloads.
        batch_size : Optional[int]
            Batch size hint for scanning.
        allow_pyarrow_filter : bool
            Allow filter pushdown into PyArrow where supported.
        maintain_order : bool
            Preserve external link ordering (lower throughput).

        Returns
        -------
        polars.LazyFrame
            A lazy query plan; call `.collect()` to materialize.
        """
        from ...polars.lib import polars

        dataset = self.to_arrow_dataset(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            maintain_order=maintain_order,
        )

        return polars.scan_pyarrow_dataset(
            dataset,
            batch_size=batch_size,
            allow_pyarrow_filter=allow_pyarrow_filter,
        )

    def to_polars(
        self,
        max_workers: int = 4,
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        streaming: bool = True,
    ) -> "polars.DataFrame":
        """Materialize results into a Polars DataFrame via the lazy pipeline.

        Parameters
        ----------
        max_workers : int
            Parallelism for remote downloads.
        max_in_flight : Optional[int]
            Backpressure for remote downloads.
        batch_size : Optional[int]
            Scan batch size hint.
        streaming : bool
            Ask Polars to use streaming execution where supported.

        Returns
        -------
        polars.DataFrame
        """
        lf = self.to_polars_lazy(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
        )
        return lf.collect()

    def to_spark(self) -> "pyspark.sql.DataFrame":
        """Convert results to a Spark DataFrame.

        If `_spark_df` is already present, it is returned directly. Otherwise, results are
        materialized to Arrow and converted to Spark via the local helper.

        Returns
        -------
        pyspark.sql.DataFrame
        """
        if self._spark_df is not None:
            return self._spark_df

        from ...spark.cast import arrow_table_to_spark_dataframe
        return arrow_table_to_spark_dataframe(self.to_arrow_table(), None)
