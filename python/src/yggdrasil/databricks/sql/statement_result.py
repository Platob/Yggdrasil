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
from ...sql.statement_result import StatementResult as BaseStatementResult
from ...types import cast_arrow_tabular
from ...types.cast.cast_options import CastOptions

if TYPE_CHECKING:
    pass

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
    workspace_client: WorkspaceClient
    warehouse_id: str
    statement_id: str
    disposition: Disposition

    _response: Optional[StatementResponse] = dataclasses.field(default=None, repr=False)

    # ----------------------------
    # Dunder / convenience
    # ----------------------------

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
    def response(self) -> StatementResponse:
        """Latest statement response, auto-refreshing until terminal.

        For Spark SQL results, returns a synthetic SUCCEEDED response.

        Returns
        -------
        StatementResponse
            The current response payload (possibly refreshed).
        """
        self.refresh_status()
        return self._response

    def api_result_data_at_index(self, chunk_index: int):
        """Fetch a specific result chunk by chunk index via the Databricks SDK."""
        return self.workspace_client.statement_execution.get_statement_result_chunk_n(
            statement_id=self.statement_id,
            chunk_index=chunk_index,
        )

    def refresh_status(self):
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

        return self

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
    def _to_arrow_batches(
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

        cast_options = CastOptions.safe_init(
            safe=True,
            target_field=self.arrow_schema
        )

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
                tb = pipc.open_stream(src).read_all()
                casted = cast_arrow_tabular(tb, cast_options)
                return casted.to_batches()

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
        *,
        max_workers: int = 4,
        max_in_flight: Optional[int] = None,
        batch_size: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        maintain_order: bool = False,
        stream: bool = True,
    ) -> pa.RecordBatchReader:
        schema = schema or self.arrow_schema

        if self.persisted:
            if self._arrow_table is not None:
                return self._arrow_table.to_reader(max_chunksize=batch_size)
            if self._spark_df is not None:
                return self._spark_df.toArrow().to_reader(max_chunksize=batch_size)
            raise NotImplementedError("Persisted without Arrow table or Spark DF")

        batches_iter = self._to_arrow_batches(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            maintain_order=maintain_order,
        )

        return pa.RecordBatchReader.from_batches(schema, batches_iter)  # type: ignore[arg-type]
