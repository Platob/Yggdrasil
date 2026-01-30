"""Result wrapper for Databricks SQL statement execution."""

import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait as concurrent_wait
from typing import Optional, Iterator, TYPE_CHECKING

import pyarrow as pa
import pyarrow.ipc as pipc

from .exceptions import SqlStatementError
from .types import column_info_to_arrow_field
from ...libs.databrickslib import databricks_sdk, WorkspaceClient, DatabricksDummyClass
from ...libs.pandaslib import PandasDataFrame
from ...libs.polarslib import polars
from ...libs.sparklib import SparkDataFrame
from ...pyutils.waiting_config import WaitingConfigArg, WaitingConfig
from ...requests.session import YGGSession
from ...types import spark_dataframe_to_arrow_table, \
    spark_schema_to_arrow_schema, arrow_table_to_spark_dataframe

if databricks_sdk is not None:
    from databricks.sdk.service.sql import (
        StatementState, StatementResponse, Disposition, StatementStatus
    )
else:
    StatementState = DatabricksDummyClass
    StatementResponse = DatabricksDummyClass
    Disposition = DatabricksDummyClass
    StatementStatus = DatabricksDummyClass


if TYPE_CHECKING:
    pass


DONE_STATES = {
    StatementState.CANCELED, StatementState.CLOSED, StatementState.FAILED,
    StatementState.SUCCEEDED
}

FAILED_STATES = {
    StatementState.FAILED, StatementState.CANCELED
}

__all__ = [
    "StatementResult"
]


@dataclasses.dataclass
class StatementResult:
    """Container for statement responses, data extraction, and conversions."""
    workspace_client: WorkspaceClient
    warehouse_id: str
    statement_id: str
    disposition: Disposition

    _response: Optional[StatementResponse] = dataclasses.field(default=None, repr=False)

    _spark_df: Optional[SparkDataFrame] = dataclasses.field(default=None, repr=False)
    _arrow_table: Optional[pa.Table] = dataclasses.field(default=None, repr=False)

    def __getstate__(self):
        """Serialize statement results, converting Spark dataframes to Arrow.

        Returns:
            A pickle-ready state dictionary.
        """
        state = self.__dict__.copy()

        _spark_df = state.pop("_spark_df", None)

        if _spark_df is not None:
            state["_arrow_table"] = spark_dataframe_to_arrow_table(_spark_df)

        return state

    def __setstate__(self, state):
        """Restore statement result state, rehydrating cached data.

        Args:
            state: Serialized state dictionary.
        """
        _spark_df = state.pop("_spark_df")

    def __iter__(self):
        """Iterate over Arrow record batches."""
        return self.to_arrow_batches()

    def __repr__(self):
        return "StatementResult(url='%s')" % self.monitoring_url

    def __str__(self):
        return self.monitoring_url

    @property
    def monitoring_url(self):
        return "%s/sql/warehouses/%s/monitoring?queryId=%s" % (
            self.workspace_client.config.host,
            self.warehouse_id,
            self.statement_id
        )

    @property
    def is_spark_sql(self):
        """Return True when this result was produced by Spark SQL."""
        return self._spark_df is not None

    @property
    def response(self):
        """Return the latest statement response, refreshing when needed.

        Returns:
            The current StatementResponse object.
        """
        if self.is_spark_sql:
            return StatementResponse(
                statement_id=self.statement_id or "sparksql",
                status=StatementStatus(
                    state=StatementState.SUCCEEDED
                )
            )
        elif not self.statement_id:
            return StatementResponse(
                statement_id="unknown",
                status=StatementStatus(
                    state=StatementState.PENDING
                )
            )

        statement_execution = self.workspace_client.statement_execution

        if self._response is None:
            # Initialize
            self._response = statement_execution.get_statement(self.statement_id)
        elif self._response.status.state not in DONE_STATES:
            # Refresh
            self._response = statement_execution.get_statement(self.statement_id)

        return self._response

    def api_result_data_at_index(self, chunk_index: int):
        """Fetch a specific result chunk by index.

        Args:
            chunk_index: Result chunk index to retrieve.

        Returns:
            The SDK result chunk response.
        """
        sdk = self.workspace_client

        return sdk.statement_execution.get_statement_result_chunk_n(
            statement_id=self.statement_id,
            chunk_index=chunk_index,
        )

    @property
    def status(self):
        """Return the statement status, handling persisted data.

        Returns:
            A StatementStatus object.
        """
        return self.response.status

    @property
    def state(self):
        """Return the statement state.

        Returns:
            The StatementState enum value.
        """
        return self.status.state

    @property
    def manifest(self):
        """Return the SQL result manifest, if available.

        Returns:
            The result manifest or None for Spark SQL results.
        """
        self.wait()
        return self.response.manifest

    @property
    def result(self):
        """Return the raw statement result object.

        Returns:
            The statement result payload from the API.
        """
        self.wait()
        return self.response.result

    @property
    def done(self):
        """Return True when the statement is in a terminal state.

        Returns:
            True if the statement is done, otherwise False.
        """
        return self.state in DONE_STATES

    @property
    def failed(self):
        """Return True when the statement failed or was cancelled.

        Returns:
            True if the statement failed or was cancelled.
        """
        return self.state in FAILED_STATES

    @property
    def persisted(self):
        """Return True when data is cached locally.

        Returns:
            True when cached Arrow or Spark data is present.
        """
        return self._spark_df is not None or self._arrow_table is not None

    def persist(self):
        """Cache the statement result locally as Arrow data.

        Returns:
            The current StatementResult instance.
        """
        if not self.persisted:
            self._arrow_table = self.to_arrow_table()
        return self

    def external_links(self):
        """Yield external result links for EXTERNAL_LINKS dispositions.

        Yields:
            External link objects in result order.
        """
        assert self.disposition == Disposition.EXTERNAL_LINKS, "Cannot get from %s, disposition %s != %s" % (
            self, self.disposition, Disposition.EXTERNAL_LINKS
        )

        result_data = self.result
        wsdk = self.workspace_client

        seen_chunk_indexes = set()

        while True:
            links = getattr(result_data, "external_links", None) or []
            if not links:
                return

            # yield all links in the current chunk/page
            for link in links:
                yield link

            # follow the next chunk (usually only present/meaningful on the last link)
            next_internal = getattr(links[-1], "next_chunk_internal_link", None)
            if not next_internal:
                return

            try:
                chunk_index = int(next_internal.rstrip("/").split("/")[-1])
            except Exception as e:
                raise ValueError(
                    f"Bad next_chunk_internal_link {next_internal!r}: {e}"
                )

            # cycle guard
            if chunk_index in seen_chunk_indexes:
                raise ValueError(
                    f"Detected chunk cycle at index {chunk_index} from {next_internal!r}"
                )
            seen_chunk_indexes.add(chunk_index)

            try:
                result_data = wsdk.statement_execution.get_statement_result_chunk_n(
                    statement_id=self.statement_id,
                    chunk_index=chunk_index,
                )
            except Exception as e:
                raise ValueError(
                    f"Cannot retrieve data batch from {next_internal!r}: {e}"
                )

    def raise_for_status(self):
        if self.failed:
            raise SqlStatementError.from_statement(self)
        return self

    def wait(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True
    ):
        """Wait for statement completion with optional timeout.

        Args:
            wait: Waiting config
            raise_error: Raise error if failed

        Returns:
            The current StatementResult instance.
        """
        wait = WaitingConfig.check_arg(wait)
        iteration, start = 0, time.time()

        if not self.done:
            wait.sleep(iteration=iteration, start=start)
            iteration += 1

        if raise_error:
            self.raise_for_status()

        return self

    def arrow_schema(self):
        """Return the Arrow schema for the result.

        Returns:
            An Arrow Schema instance.
        """
        if self.persisted:
            if self._arrow_table is not None:
                return self._arrow_table.schema
            elif self._spark_df is not None:
                return spark_schema_to_arrow_schema(self._spark_df.schema)
            else:
                raise NotImplementedError("")

        manifest = self.manifest

        metadata = {
            "source": "databricks-sql",
            "sid": self.statement_id or ""
        }

        if manifest is None:
            return pa.schema([], metadata=metadata)

        fields = [
            column_info_to_arrow_field(_) for _ in manifest.schema.columns
        ]

        return pa.schema(
            fields,
            metadata=metadata
        )

    def to_arrow_table(
        self,
        parallel_pool: int = 4
    ) -> pa.Table:
        """Collect the statement result into a single Arrow table.

        Args:
            parallel_pool: Maximum parallel fetch workers.

        Returns:
            An Arrow Table containing all rows.
        """
        if self.persisted:
            if self._arrow_table is not None:
                return self._arrow_table
            else:
                return self._spark_df.toArrow()

        batches = list(self.to_arrow_batches(parallel_pool=parallel_pool))

        if not batches:
            return pa.Table.from_batches([], schema=self.arrow_schema())

        return pa.Table.from_batches(batches)

    def to_arrow_batches(
        self,
        parallel_pool: int = 4,
        batch_size: Optional[int] = None
    ) -> Iterator[pa.RecordBatch]:
        """Stream the result as Arrow record batches.

        Args:
            parallel_pool: Maximum parallel fetch workers.
            batch_size: Fetch batch size

        Yields:
            Arrow RecordBatch objects.
        """
        if self.persisted:
            if self._arrow_table is not None:
                for batch in self._arrow_table.to_batches(max_chunksize=batch_size):
                    yield batch
            elif self._spark_df is not None:
                for batch in self._spark_df.toArrow().to_batches(max_chunksize=batch_size):
                    yield batch
            else:
                raise NotImplementedError("")
        else:
            _tls = threading.local()

            def _get_session():
                # requests.Session-style objects are not reliably thread-safe, so keep one per thread
                s = getattr(_tls, "session", None)
                if s is None:
                    s = YGGSession()
                    _tls.session = s
                return s

            def _fetch_bytes(link):
                s = _get_session()
                resp = s.get(link.external_link, verify=False, timeout=10)
                resp.raise_for_status()
                return resp.content

            max_workers = max(1, int(parallel_pool) if parallel_pool else 4)
            max_in_flight = max_workers * 2  # keeps pipeline full without exploding memory

            links_iter = enumerate(self.external_links())
            pending = {}  # future -> idx
            ready = {}  # idx -> bytes
            next_idx = 0

            def submit_more(ex):
                while len(pending) < max_in_flight:
                    try:
                        idx, link = next(links_iter)
                    except StopIteration:
                        break
                    fut = ex.submit(_fetch_bytes, link)
                    pending[fut] = idx

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                submit_more(ex)

                while pending:
                    done, _ = concurrent_wait(pending, return_when=FIRST_COMPLETED)

                    # collect completed downloads
                    for fut in done:
                        idx = pending.pop(fut)
                        ready[idx] = fut.result()  # raises here if the GET failed

                    # yield strictly in-order
                    while next_idx in ready:
                        content = ready.pop(next_idx)

                        buf = pa.BufferReader(content)

                        # IPC stream (your current format)
                        reader = pipc.open_stream(buf)

                        # if it’s IPC file instead:
                        # reader = pipc.open_file(buf)

                        for batch in reader:
                            yield batch

                        next_idx += 1

                    submit_more(ex)

    def to_pandas(
        self,
        parallel_pool: Optional[int] = 4
    ) -> PandasDataFrame:
        """Return the result as a pandas DataFrame.

        Args:
            parallel_pool: Maximum parallel fetch workers.

        Returns:
            A pandas DataFrame with the result rows.
        """
        return self.to_arrow_table(parallel_pool=parallel_pool).to_pandas()

    def to_polars(
        self,
        parallel_pool: int = 4
    ) -> "polars.DataFrame":
        """Return the result as a polars DataFrame.

        Args:
            parallel_pool: Maximum parallel fetch workers.

        Returns:
            A polars DataFrame with the result rows.
        """
        arrow_table = self.to_arrow_table(parallel_pool=parallel_pool)

        return polars.from_arrow(arrow_table)

    def to_spark(self) -> SparkDataFrame:
        """Return the result as a Spark DataFrame, caching it locally.

        Returns:
            A Spark DataFrame with the result rows.
        """
        if self._spark_df is not None:
            return self._spark_df

        return arrow_table_to_spark_dataframe(self.to_arrow_table())
