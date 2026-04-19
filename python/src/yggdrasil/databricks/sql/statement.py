"""Single access point for managing a Databricks SQL statement execution.

``StatementResult`` binds a :class:`~yggdrasil.data.statement.PreparedStatement`
configuration to the Databricks backend: it carries the submission state
(``warehouse_id``, ``statement_id``, the cached response) and exposes the
Arrow-first result interface provided by
:class:`~yggdrasil.data.statement.StatementResult`.

A result is ``started`` once a ``statement_id`` is present; :meth:`start`
submits the query to a warehouse when it is not yet started.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, List, Optional, TYPE_CHECKING

import pyarrow as pa
import pyarrow.ipc as pipc
import urllib3
from databricks.sdk.service.sql import (
    Disposition,
    ExternalLink,
    StatementResponse,
    StatementState,
    StatementStatus,
)

from yggdrasil.concurrent.threading import Job, JobPoolExecutor
from yggdrasil.data import Field, Schema, schema
from yggdrasil.data.cast import CastOptions
from yggdrasil.data.statement import PreparedStatement
from yggdrasil.data.statement import StatementResult as BaseStatementResult

from .exceptions import SQLError
from .statements import Statements
from ..client import DatabricksResource

if TYPE_CHECKING:
    from .warehouse import SQLWarehouse

__all__ = ["PreparedStatement", "StatementResult"]


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


@dataclass
class StatementResult(BaseStatementResult, DatabricksResource):
    """Databricks-backed :class:`StatementResult` handler.

    Wraps a :class:`PreparedStatement` configuration together with the per-execution
    state (``warehouse_id``, ``statement_id``, cached ``StatementResponse``)
    that Databricks needs.  All config — text, parameters, external tables —
    lives on ``self.statement``.
    """

    statement: PreparedStatement = field(default_factory=PreparedStatement)
    service: Statements = field(
        default_factory=Statements.current,
        repr=False, compare=False, hash=False,
    )
    warehouse_id: str | None = None
    statement_id: str | None = None
    disposition: Optional[Disposition] = None

    _response: Optional[StatementResponse] = field(
        default=None, repr=False, compare=False, hash=False,
    )
    _history: Optional[dict] = field(
        default=None, repr=False, compare=False, hash=False,
    )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def prepare(
        cls,
        statement: "StatementResult | PreparedStatement | str",
        *,
        parameters=None,
        external_tables=None,
    ) -> "StatementResult":
        """Coerce ``statement`` into a :class:`StatementResult`.

        - Existing ``StatementResult`` instances are reused; extra ``parameters``
          / ``external_tables`` are merged into their inner
          :class:`PreparedStatement` config.
        - Plain :class:`PreparedStatement` configs are wrapped in an unstarted
          :class:`StatementResult`.
        - Strings are coerced via :meth:`PreparedStatement.prepare`.
        """
        if isinstance(statement, cls):
            if parameters or external_tables:
                merged = PreparedStatement.prepare(
                    statement.statement,
                    parameters=parameters,
                    external_tables=external_tables,
                )
                return replace(statement, statement=merged)
            return statement

        cfg = PreparedStatement.prepare(
            statement,
            parameters=parameters,
            external_tables=external_tables,
        )
        return cls(statement=cfg)

    def bind(self, **parameters) -> "StatementResult":
        """Return a copy with additional named parameters bound on the config."""
        if not parameters:
            return self
        return replace(self, statement=self.statement.bind(**parameters))

    def with_external_tables(self, **tables) -> "StatementResult":
        """Return a copy with additional external-table aliases on the config."""
        if not tables:
            return self
        return replace(
            self,
            statement=self.statement.with_external_tables(**tables),
        )

    def with_text(self, text: str) -> "StatementResult":
        """Return a copy with the SQL text replaced on the config."""
        if text == self.statement.text:
            return self
        return replace(self, statement=self.statement.with_text(text))

    def clear(self) -> "StatementResult":
        """Return a copy with an empty :class:`PreparedStatement` config."""
        return replace(self, statement=PreparedStatement())

    def to_parameter_list(self):
        """Delegate to :meth:`PreparedStatement.to_parameter_list`."""
        return self.statement.to_parameter_list()

    # Back-compat: ``looks_like_query`` used to live on the Databricks class.
    @staticmethod
    def looks_like_query(text):
        return PreparedStatement.looks_like_query(text)

    # ------------------------------------------------------------------
    # Execution lifecycle
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        """Whether the statement has been submitted (``statement_id`` present)."""
        return bool(self.statement_id)

    def start(
        self,
        *,
        warehouse: "Optional[SQLWarehouse]" = None,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        byte_limit: int | None = None,
        row_limit: int | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        disposition: Optional[Disposition] = None,
        wait=True,
        raise_error: bool = True,
    ) -> "StatementResult":
        """Submit the statement to a warehouse if not already started.

        When the result is already started (``statement_id`` is set) the
        existing instance is returned unchanged.  Otherwise the query is
        submitted to ``warehouse`` (or to the warehouse resolved by
        ``warehouse_id`` / ``warehouse_name``); ``statement_id``, response,
        and resolved ``warehouse_id``/``disposition`` are recorded on this
        instance before it is returned.
        """
        if self.started:
            return self

        from .warehouse import SQLWarehouse

        if warehouse is None:
            warehouse = SQLWarehouse(warehouse_id=warehouse_id, warehouse_name=warehouse_name)

        warehouse.execute(
            statement=self,
            warehouse_id=warehouse_id,
            warehouse_name=warehouse_name,
            byte_limit=byte_limit,
            disposition=disposition,
            row_limit=row_limit,
            catalog_name=catalog_name,
            schema_name=schema_name,
            wait=wait,
            raise_error=raise_error,
        )

        return self

    def cancel(self) -> "StatementResult":
        """Cancel the running statement on Databricks.

        No-op when the statement has not been started or has already
        reached a terminal state.  After a successful cancellation the
        cached response is refreshed so ``state``/``done``/``failed``
        reflect the cancelled status.
        """
        if not self.started or self.statement_id == "SparkSQL":
            return self
        if self._response is not None and self._response.status.state in DONE_STATES:
            return self

        self.client.workspace_client().statement_execution.cancel_execution(
            statement_id=self.statement_id,
        )
        object.__setattr__(self, "_response", None)
        self.refresh_status()
        return self

    # ------------------------------------------------------------------
    # Dunder / convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.started:
            return f"StatementResult(url='{self.monitoring_url}')"
        return f"StatementResult(text={self.statement.text!r})"

    def __str__(self) -> str:
        return self.monitoring_url if self.started else self.statement.text

    @property
    def monitoring_url(self) -> str:
        """Databricks UI monitoring URL for this statement execution."""
        return "%ssql/warehouses/%s/monitoring?queryId=%s" % (
            self.client.base_url.to_string(),
            self.warehouse_id,
            self.statement_id,
        )

    # ------------------------------------------------------------------
    # State / status
    # ------------------------------------------------------------------

    @property
    def response(self) -> StatementResponse:
        """Latest statement response, auto-refreshing until terminal."""
        self.refresh_status()
        return self._response

    def api_result_data_at_index(self, chunk_index: int):
        """Fetch a specific result chunk by chunk index via the Databricks SDK."""
        return self.client.workspace_client().statement_execution.get_statement_result_chunk_n(
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

        statement_execution = self.client.workspace_client().statement_execution

        if self._response is None:
            object.__setattr__(
                self, "_response",
                statement_execution.get_statement(self.statement_id)
            )
        elif self._response.status.state not in DONE_STATES:
            object.__setattr__(
                self, "_response",
                statement_execution.get_statement(self.statement_id)
            )

        return self

    @property
    def status(self) -> StatementStatus:
        """PreparedStatement status object."""
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
        """True when the statement failed or was canceled."""
        return self.state in FAILED_STATES

    def raise_for_status(self) -> "StatementResult":
        """Raise ``SQLError`` if the statement is in a failed state."""
        if self.failed:
            error = SQLError.from_statement(self)
            raise error
        return self

    # ------------------------------------------------------------------
    # Manifest / schema
    # ------------------------------------------------------------------

    @property
    def manifest(self):
        """SQL result manifest (Databricks API), if available."""
        self.wait()
        return self.response.manifest

    @property
    def result(self):
        """Raw statement result payload from the Databricks API."""
        self.wait()
        return self.response.result

    def collect_schema(self, full: bool = False) -> Schema:
        manifest = self.manifest
        metadata = {
            "engine": "databricks-sql",
            "statement_id": self.statement_id or ""
        }

        if manifest is None:
            return schema([], metadata=metadata)

        return Schema.from_any_fields(
            [Field.from_databricks(c) for c in (manifest.schema.columns or [])],
            metadata=metadata
        )

    # ------------------------------------------------------------------
    # External links
    # ------------------------------------------------------------------

    def external_links(self) -> Iterator[ExternalLink]:
        """Yield external result links for ``Disposition.EXTERNAL_LINKS``."""
        assert self.disposition == Disposition.EXTERNAL_LINKS, (
            "Cannot get external links from %s, disposition %s != %s"
            % (self, self.disposition, Disposition.EXTERNAL_LINKS)
        )

        result_data = self.result
        wsdk = self.client.workspace_client()

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

    # ------------------------------------------------------------------
    # Arrow conversions
    # ------------------------------------------------------------------

    def _to_arrow_batches(
        self,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        maintain_order: bool = False,
    ) -> Iterator[pa.RecordBatch]:
        if self.persisted:
            if self._arrow_table is not None:
                yield from self._arrow_table.to_batches(max_chunksize=batch_size)
                return
            if self._spark_df is not None:
                yield from self._spark_df.toArrow().to_batches(max_chunksize=batch_size)
                return
            raise NotImplementedError("Persisted without Arrow table or Spark DF")

        cast_options = CastOptions(
            safe=True,
            target_field=self.arrow_schema
        )

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
            resp = http.request("GET", url, preload_content=True)
            try:
                if resp.status >= 400:
                    raise RuntimeError(f"GET {url} failed: {resp.status}")
                buf = memoryview(resp.data)
            finally:
                resp.release_conn()

            with pa.input_stream(buf) as src:
                tb: pa.Table = (
                    pipc.open_stream(src)
                    .read_all()
                )
                casted = cast_options.cast_arrow_tabular(tb)
                return casted.to_batches()

        def jobs() -> Iterable[Job]:
            for link in self.external_links():
                if link.external_link:
                    yield Job.make(extract_batches, link.external_link)

        with JobPoolExecutor.parse(max_workers or 4) as ex:
            for result in ex.as_completed(
                jobs(),
                ordered=maintain_order,
                max_in_flight=max_in_flight,
                cancel_on_exit=True,
                shutdown_on_exit=True,
                shutdown_wait=False,
            ):
                for batch in result.result:
                    yield batch

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

    def to_pylist(
        self,
        *,
        max_workers: int = 4,
        max_in_flight: int | None = None,
        batch_size: int | None = None,
        maintain_order: bool = False,
    ):
        return self.to_arrow_reader(
            max_workers=max_workers,
            max_in_flight=max_in_flight,
            batch_size=batch_size,
            maintain_order=maintain_order,
        ).read_all().to_pylist()
