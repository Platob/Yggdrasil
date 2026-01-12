"""Result wrapper for Databricks SQL statement execution."""

import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from typing import Optional, Iterator, TYPE_CHECKING

import pyarrow as pa
import pyarrow.ipc as pipc

from .types import column_info_to_arrow_field
from ...libs.databrickslib import databricks_sdk
from ...libs.pandaslib import pandas
from ...libs.polarslib import polars
from ...libs.sparklib import SparkDataFrame
from ...requests.session import YGGSession
from ...types import spark_dataframe_to_arrow_table, \
    spark_schema_to_arrow_schema, arrow_table_to_spark_dataframe

try:
    from delta.tables import DeltaTable as SparkDeltaTable
except ImportError:
    class SparkDeltaTable:
        @classmethod
        def forName(cls, *args, **kwargs):
            from delta.tables import DeltaTable

            return DeltaTable.forName(*args, **kwargs)


if databricks_sdk is not None:
    from databricks.sdk.service.sql import (
        StatementState, StatementResponse, Disposition, StatementStatus
)

    StatementResponse = StatementResponse
else:
    class StatementResponse:
        pass


if TYPE_CHECKING:
    from .engine import SQLEngine


__all__ = [
    "StatementResult"
]


@dataclasses.dataclass
class StatementResult:
    """Container for statement responses, data extraction, and conversions."""
    engine: "SQLEngine"
    statement_id: str
    disposition: "Disposition"

    _response: Optional[StatementResponse] = dataclasses.field(default=None, repr=False)
    _response_refresh_time: float = dataclasses.field(default=0, repr=False)

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
        if self._response is None and not self.is_spark_sql:
            self.response = self.workspace.sdk().statement_execution.get_statement(self.statement_id)
        return self._response

    @response.setter
    def response(self, value: "StatementResponse"):
        """Update the cached response and refresh timestamp.

        Args:
            value: StatementResponse to cache.
        """
        self._response = value
        self._response_refresh_time = time.time()

        self.statement_id = self._response.statement_id

    def fresh_response(self, delay: float):
        """Refresh the response if it is older than ``delay`` seconds.

        Args:
            delay: Minimum age in seconds before refreshing.

        Returns:
            The refreshed StatementResponse object.
        """
        if self.is_spark_sql:
            return self._response

        if self.statement_id and not self.done and time.time() - self._response_refresh_time > delay:
            self.response = self.workspace.sdk().statement_execution.get_statement(self.statement_id)

        return self._response

    def result_data_at(self, chunk_index: int):
        """Fetch a specific result chunk by index.

        Args:
            chunk_index: Result chunk index to retrieve.

        Returns:
            The SDK result chunk response.
        """
        sdk = self.workspace.sdk()

        return sdk.statement_execution.get_statement_result_chunk_n(
            statement_id=self.statement_id,
            chunk_index=chunk_index,
        )

    @property
    def workspace(self):
        """Expose the underlying workspace from the engine.

        Returns:
            The Workspace instance backing this statement.
        """
        return self.engine.workspace

    @property
    def status(self):
        """Return the statement status, handling persisted data.

        Returns:
            A StatementStatus object.
        """
        if self.persisted:
            return StatementStatus(
                state=StatementState.SUCCEEDED
            )

        if not self.statement_id:
            return StatementStatus(
                state=StatementState.PENDING
            )

        return self.fresh_response(delay=1).status

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
        if self.is_spark_sql:
            return None
        return self.response.manifest

    @property
    def result(self):
        """Return the raw statement result object.

        Returns:
            The statement result payload from the API.
        """
        return self.response.result

    @property
    def done(self):
        """Return True when the statement is in a terminal state.

        Returns:
            True if the statement is done, otherwise False.
        """
        if self.persisted:
            return True

        if self._response is None:
            return False

        return self._response.status.state in [
            StatementState.CANCELED, StatementState.CLOSED, StatementState.FAILED, StatementState.SUCCEEDED
        ]

    @property
    def failed(self):
        """Return True when the statement failed or was cancelled.

        Returns:
            True if the statement failed or was cancelled.
        """
        if self.persisted:
            return True

        if self._response is None:
            return False

        return self._response.status.state in [StatementState.CANCELED, StatementState.FAILED]

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

        self.wait()
        result_data = self.result
        wsdk = self.workspace.sdk()

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
        """Raise a ValueError if the statement failed.

        Returns:
            None.
        """
        if self.failed:
            # grab error info if present
            err = self.status.error
            message = err.message or "Unknown SQL error"
            error_code = err.error_code
            sql_state = getattr(err, "sql_state", None)

            parts = [message]
            if error_code:
                parts.append(f"error_code={error_code}")
            if sql_state:
                parts.append(f"sql_state={sql_state}")

            raise ValueError(
                f"Statement {self.statement_id} {self.state}: " + " | ".join(parts)
            )

    def wait(
        self,
        timeout: Optional[int] = None,
        poll_interval: Optional[float] = None
    ):
        """Wait for statement completion with optional timeout.

        Args:
            timeout: Maximum seconds to wait.
            poll_interval: Initial poll interval in seconds.

        Returns:
            The current StatementResult instance.
        """
        if self.done:
            return self

        start = time.time()
        poll_interval = poll_interval or 1
        current = self

        while not self.done:
            # still running / queued / pending
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(
                    f"Statement {current.statement_id} did not finish within {timeout} seconds "
                    f"(last state={current})"
                )

            poll_interval = max(10, poll_interval * 1.2)
            time.sleep(poll_interval)

        return current

    def arrow_schema(self):
        """Return the Arrow schema for the result.

        Returns:
            An Arrow Schema instance.
        """
        if self.persisted:
            if self._arrow_table is not None:
                return self._arrow_table.schema
            return spark_schema_to_arrow_schema(self._spark_df.schema)

        fields = [
            column_info_to_arrow_field(_) for _ in self.manifest.schema.columns
        ]

        return pa.schema(fields)

    def to_arrow_table(self, parallel_pool: Optional[int] = 4) -> pa.Table:
        """Collect the statement result into a single Arrow table.

        Args:
            parallel_pool: Maximum parallel fetch workers.

        Returns:
            An Arrow Table containing all rows.
        """
        if self.persisted:
            if self._arrow_table:
                return self._arrow_table
            else:
                return self._spark_df.toArrow()

        batches = list(self.to_arrow_batches(parallel_pool=parallel_pool))

        if not batches:
            # empty table with no columns
            return pa.Table.from_batches([], schema=self.arrow_schema())

        return pa.Table.from_batches(batches)

    def to_arrow_batches(
        self,
        parallel_pool: Optional[int] = 4
    ) -> Iterator[pa.RecordBatch]:
        """Stream the result as Arrow record batches.

        Args:
            parallel_pool: Maximum parallel fetch workers.

        Yields:
            Arrow RecordBatch objects.
        """
        if self.persisted:
            if self._arrow_table is not None:
                for batch in self._arrow_table.to_batches(max_chunksize=64 * 1024):
                    yield batch
            elif self._spark_df is not None:
                for batch in self._spark_df.toArrow().to_batches(max_chunksize=64 * 1024):
                    yield batch
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

            # ---- in your generator ----
            if self.persisted:
                if self._arrow_table is not None:
                    for batch in self._arrow_table.to_batches(max_chunksize=64 * 1024):
                        yield batch
                elif self._spark_df is not None:
                    for batch in self._spark_df.toArrow().to_batches(max_chunksize=64 * 1024):
                        yield batch
            else:
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
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)

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
    ) -> "pandas.DataFrame":
        """Return the result as a pandas DataFrame.

        Args:
            parallel_pool: Maximum parallel fetch workers.

        Returns:
            A pandas DataFrame with the result rows.
        """
        return self.to_arrow_table(parallel_pool=parallel_pool).to_pandas()

    def to_polars(
        self,
        parallel_pool: Optional[int] = 4
    ) -> "polars.DataFrame":
        """Return the result as a polars DataFrame.

        Args:
            parallel_pool: Maximum parallel fetch workers.

        Returns:
            A polars DataFrame with the result rows.
        """
        return polars.from_arrow(self.to_arrow_table(parallel_pool=parallel_pool))

    def to_spark(self):
        """Return the result as a Spark DataFrame, caching it locally.

        Returns:
            A Spark DataFrame with the result rows.
        """
        if self._spark_df:
            return self._spark_df

        self._spark_df = arrow_table_to_spark_dataframe(self.to_arrow_table())
        return self._spark_df
