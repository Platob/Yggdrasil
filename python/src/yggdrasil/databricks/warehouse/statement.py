"""Databricks SQL warehouse-backed statements.

Three concrete types layered on the abstractions in
``yggdrasil.data.statement``:

- :class:`WarehousePreparedStatement` — adds typed routing
  (``warehouse_id`` / ``warehouse_name``), wire format, server-side wait,
  result caps, parameter bindings, and external-table aliases.
- :class:`WarehouseStatementResult` — tracks a single submission against
  the Databricks Statement Execution API: ``statement_id``, response
  caching, polling, and external-link fetch for Arrow streams.
- :class:`WarehouseStatementBatch` — re-uses the base batch contract; per-
  statement external-table aliases are resolved at coerce time, batch-wide
  scratch is cleaned up at teardown.

A few invariants the cleanup pass enforces:

- Each statement carries its own ``external_volume_paths``.  The batch
  doesn't maintain a parallel registry — the alias-substitution rewriter
  reads the per-statement field directly.  Single source of truth.
- ``_coerce`` returns the prepared statement (the previous version
  silently dropped it on the floor).
- Alias substitution doesn't mutate the input statement — it returns a
  rewritten copy, so re-submitting the same batch is safe.
"""

from __future__ import annotations

import copy as copy_mod
import logging
import re
from typing import (
    Any,
    ClassVar,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    TYPE_CHECKING, Literal,
)

import pyarrow as pa
import pyarrow.ipc as pipc
import urllib3
from databricks.sdk.service.sql import (
    Disposition,
    ExecuteStatementRequestOnWaitTimeout,
    ExternalLink,
    Format,
    StatementParameterListItem,
    StatementResponse,
    StatementState,
    StatementStatus,
)

from yggdrasil.concurrent.threading import Job, JobPoolExecutor
from yggdrasil.data import Schema, schema
from yggdrasil.data.options import CastOptions
from yggdrasil.data.statement import PreparedStatement, StatementResult, StatementBatch
from yggdrasil.databricks.sql.exceptions import SQLError
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg
from yggdrasil.io.enums import MimeType, MimeTypes, MediaTypes
from ..fs import VolumePath, DatabricksPath
from ..sql.types import parse_databricks_field

if TYPE_CHECKING:
    from yggdrasil.databricks.warehouse.warehouse import SQLWarehouse

__all__ = [
    "WarehousePreparedStatement",
    "WarehouseStatementResult",
    "WarehouseStatementBatch",
]

logger = logging.getLogger(__name__)


_DEFAULT_BYTE_SIZE = 32 * 1024 * 1024

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


# Aliases must survive ``{name}``-style substitution.
_VALID_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# WarehousePreparedStatement
# ---------------------------------------------------------------------------


class WarehousePreparedStatement(PreparedStatement):
    """Typed Databricks SQL prepared statement.

    Carries everything :meth:`SQLWarehouse._submit_statement` reads — no
    sentinel attributes, no ``getattr`` fallbacks.

    Routing & scope
    ---------------
    ``warehouse_id`` / ``warehouse_name`` are routing hints: when set,
    :meth:`SQLWarehouse.execute` redirects submission to the matching
    warehouse rather than ``self``.  ``catalog_name`` / ``schema_name``
    set the per-statement context.

    Wire format
    -----------
    ``disposition`` / ``format`` control how Databricks returns results.
    Defaults are applied by :class:`SQLWarehouse` when these are ``None``;
    CSV / ARROW_STREAM force ``EXTERNAL_LINKS`` because INLINE only
    supports JSON_ARRAY.

    Parameters & external tables
    ----------------------------
    ``parameters`` is the SDK-typed list (use :meth:`with_parameters` for
    a Mapping-friendly builder).  ``external_volume_paths`` maps query-
    text aliases to staged :class:`VolumePath` instances; :meth:`prepare`
    auto-stages tabular ``external_data`` into Parquet on a fresh
    :class:`VolumePath`.

    Retry
    -----
    Inherits ``retry`` (a :class:`WaitingConfig`) from
    :class:`PreparedStatement`.  Default is ``None`` (not retryable).
    Pass ``retry=WaitingConfig(...)``, ``retry=True`` for the standard
    default policy, or ``retry={"timeout": 60, "retries": 3}`` for a
    dict-shaped config; see :meth:`WaitingConfig.from_`.
    """

    # ---- Routing ----
    warehouse_id: Optional[str] = None
    warehouse_name: Optional[str] = None

    # ---- Scope ----
    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None

    # ---- Wire format ----
    disposition: Optional[Disposition] = None
    format: Optional[Format] = None

    # ---- Server-side wait ----
    on_wait_timeout: Optional[ExecuteStatementRequestOnWaitTimeout] = None
    wait_timeout: Optional[str] = None

    # ---- Result caps ----
    byte_limit: Optional[int] = None
    row_limit: Optional[int] = None

    # ---- Bindings ----
    parameters: Optional[List[StatementParameterListItem]] = None
    external_volume_paths: Optional[dict[str, VolumePath]] = None

    def __init__(
        self,
        text: str = "",
        *,
        key: Optional[str] = None,
        retry: Optional[WaitingConfigArg] = None,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        disposition: Optional[Disposition] = None,
        format: Optional[Format] = None,
        on_wait_timeout: Optional[ExecuteStatementRequestOnWaitTimeout] = None,
        wait_timeout: Optional[str] = None,
        byte_limit: Optional[int] = None,
        row_limit: Optional[int] = None,
        parameters: Optional[List[StatementParameterListItem]] = None,
        external_volume_paths: Optional[dict[str, VolumePath]] = None,
        **kwargs: Any,
    ):
        super().__init__(text, key=key, retry=retry)
        self.warehouse_id = warehouse_id
        self.warehouse_name = warehouse_name
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.disposition = disposition
        self.format = format
        self.on_wait_timeout = on_wait_timeout
        self.wait_timeout = wait_timeout
        self.byte_limit = byte_limit
        self.row_limit = row_limit
        self.parameters = parameters
        self.external_volume_paths = external_volume_paths

    # ------------------------------------------------------------------
    # External data validation / staging
    # ------------------------------------------------------------------

    @classmethod
    def check_external_data(
        cls,
        external_data: Optional[Mapping[str, Any]] = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        resource_name: Optional[str] = None,
        temporary: bool = True,
    ) -> dict[str, VolumePath]:
        """Validate ``external_data`` and stage tabular values to Parquet volumes.

        Each entry maps a query-text alias (used as ``{alias}`` in the
        statement text) to one of:

        - an existing :class:`VolumePath` — passed through.
        - tabular data (Arrow / polars / pandas / list / dict) — staged as
          Parquet onto a fresh :meth:`VolumePath.staging_path` under the
          supplied ``catalog_name`` / ``schema_name``.

        Returns a fresh ``dict[str, VolumePath]``; never mutates input.
        Empty / ``None`` input returns ``{}``.
        """
        if external_data is None:
            return {}
        if not isinstance(external_data, Mapping):
            raise TypeError(
                f"external_data must be a mapping; got {type(external_data).__name__}"
            )
        if not external_data:
            return {}

        out: dict[str, VolumePath] = {}
        for alias, value in external_data.items():
            cls._validate_alias(alias)

            if isinstance(value, VolumePath):
                out[alias] = value
                continue
            if isinstance(value, DatabricksPath):
                raise TypeError(
                    f"external_data[{alias!r}]: only VolumePath is supported, "
                    f"got {type(value).__name__}; stage to a Volume first"
                )
            if value is None:
                raise ValueError(
                    f"external_data[{alias!r}]: value is None; "
                    f"pass a VolumePath or tabular data"
                )

            out[alias] = cls._stage_external_value(
                alias=alias,
                value=value,
                catalog_name=catalog_name,
                schema_name=schema_name,
                resource_name=resource_name,
                temporary=temporary,
            )

        return out

    @staticmethod
    def _validate_alias(alias: Any) -> None:
        if not isinstance(alias, str) or not alias:
            raise ValueError(
                f"external_data alias must be a non-empty string; got {alias!r}"
            )
        if not _VALID_ALIAS_RE.match(alias):
            raise ValueError(
                f"external_data alias {alias!r} is not a valid identifier; "
                f"must match [A-Za-z_][A-Za-z0-9_]*"
            )

    @classmethod
    def _stage_external_value(
        cls,
        *,
        alias: str,
        value: Any,
        catalog_name: Optional[str],
        schema_name: Optional[str],
        resource_name: Optional[str],
        temporary: bool,
    ) -> VolumePath:
        """Stage tabular ``value`` to a fresh Parquet volume.  Override
        in subclasses for custom file formats / staging policies.
        """
        try:
            path = VolumePath.staging_path(
                catalog_name=catalog_name,
                schema_name=schema_name,
                resource_name=resource_name or alias,
                temporary=temporary,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to allocate staging volume for external_data[{alias!r}]: {e}"
            ) from e

        try:
            path.as_media(media_type=MediaTypes.PARQUET).write_table(value)
        except Exception as e:
            if temporary:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    logger.debug(
                        "Could not clean up staging path %r after stage failure",
                        path, exc_info=True,
                    )
            raise RuntimeError(
                f"Failed to stage external_data[{alias!r}] "
                f"({type(value).__name__}) as Parquet: {e}"
            ) from e

        return path

    # ------------------------------------------------------------------
    # Coercion / preparation
    # ------------------------------------------------------------------

    @classmethod
    def prepare(
        cls,
        statement: "WarehousePreparedStatement | PreparedStatement | str",
        *,
        parameters: Optional[Mapping[str, Any] | List[StatementParameterListItem]] = None,
        external_data: Optional[Mapping[str, Any]] = None,
        external_volume_paths: Optional[dict[str, VolumePath]] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        resource_name: Optional[str] = None,
        disposition: Optional[Disposition] = None,
        format: Optional[Format] = None,
        temporary: bool = True,
        retry: Optional[WaitingConfigArg] = None,
        **kwargs: Any,
    ) -> "WarehousePreparedStatement":
        """Coerce + bind parameters + stage external data.

        Merge precedence on alias collisions:
        existing volumes < caller-supplied < just-staged.

        ``retry`` is applied only when non-None — preserves whatever was
        already on the statement when ``prepare`` is given an existing
        :class:`WarehousePreparedStatement`.  Pass ``retry=False`` to
        explicitly clear an existing retry policy.
        """
        prepared = cls.from_(statement)

        # ---- External data: validate + stage ----
        staged = cls.check_external_data(
            external_data,
            catalog_name=catalog_name,
            schema_name=schema_name,
            resource_name=resource_name,
            temporary=temporary,
        )
        ext_paths: dict[str, VolumePath] = dict(prepared.external_volume_paths or {})
        if external_volume_paths:
            ext_paths.update(external_volume_paths)
        ext_paths.update(staged)

        # ---- Parameters: list or mapping ----
        new_params: list[StatementParameterListItem] = list(prepared.parameters or [])
        if parameters:
            if isinstance(parameters, Mapping):
                new_params.extend(_mapping_to_parameter_list(parameters))
            else:
                new_params.extend(parameters)

        prepared = prepared.with_parameters(new_params)

        # ---- Wire format defaults ----
        if format is not None:
            prepared.format = format
        elif prepared.format is None:
            prepared.format = Format.ARROW_STREAM

        if prepared.format is Format.JSON_ARRAY:
            prepared.disposition = disposition or prepared.disposition or Disposition.INLINE
        else:
            # CSV / ARROW_STREAM only support EXTERNAL_LINKS.
            prepared.disposition = Disposition.EXTERNAL_LINKS

        # ---- Retry config: only override when caller asked.
        # ``False`` explicitly disables; any other non-None value passes
        # through WaitingConfig.from_.
        if retry is not None:
            if retry is False:
                prepared.retry = None
            else:
                prepared.retry = WaitingConfig.from_(retry)

        # Apply any extra typed-field kwargs in one shot.
        if catalog_name is not None:
            prepared.catalog_name = catalog_name
        if schema_name is not None:
            prepared.schema_name = schema_name
        for k, v in kwargs.items():
            if hasattr(prepared, k):
                setattr(prepared, k, v)

        prepared.external_volume_paths = ext_paths or None
        return prepared

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def with_warehouse(
        self,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
    ) -> "WarehousePreparedStatement":
        copied = copy_mod.copy(self)
        copied.warehouse_id = warehouse_id if warehouse_id is not None else self.warehouse_id
        copied.warehouse_name = warehouse_name if warehouse_name is not None else self.warehouse_name
        return copied

    def with_parameters(
        self,
        parameters: "Mapping[str, Any] | List[StatementParameterListItem] | None",
        *,
        merge: bool = True,
        copy: bool = True,
    ) -> "WarehousePreparedStatement":
        """Return (or update in place) a copy with ``parameters`` set.

        ``parameters`` may be an SDK-typed list, a ``{name: value}``
        mapping, or ``None`` to clear.  ``merge=True`` (default) appends
        to existing parameters; ``copy=False`` mutates ``self`` in place.
        """
        if parameters is None:
            new_params: Optional[List[StatementParameterListItem]] = None
        elif isinstance(parameters, Mapping):
            new_params = _mapping_to_parameter_list(parameters)
        else:
            new_params = list(parameters)

        if new_params and merge and self.parameters:
            new_params = list(self.parameters) + new_params

        target = copy_mod.copy(self) if copy else self
        target.parameters = new_params or None
        return target

    def to_parameter_list(self) -> Optional[List[StatementParameterListItem]]:
        """Single override point for richer parameter representations."""
        return self.parameters

    # ------------------------------------------------------------------
    # Per-statement scratch cleanup
    # ------------------------------------------------------------------

    def clear_temporary_resources(self) -> None:
        """Unlink any temporary staged volumes and clear the registry."""
        if not self.external_volume_paths:
            return
        for alias, path in self.external_volume_paths.items():
            if getattr(path, "temporary", False):
                try:
                    Job.make(path.unlink, missing_ok=True).fire_and_forget()
                except Exception:
                    logger.exception(
                        "Failed to unlink temporary staged volume %r (alias=%r); continuing.",
                        path, alias,
                    )

        self.external_volume_paths = None


def _mapping_to_parameter_list(
    parameters: Mapping[str, Any],
) -> List[StatementParameterListItem]:
    """``{name: value}`` -> SDK ``StatementParameterListItem`` list.

    Already-typed values pass through; everything else goes as a stringified
    ``value`` and the SDK infers the type — matches Databricks' auto-coercion
    on untyped parameters.
    """
    out: List[StatementParameterListItem] = []
    for name, value in parameters.items():
        if isinstance(value, StatementParameterListItem):
            out.append(value)
            continue
        out.append(
            StatementParameterListItem(
                name=name,
                value=None if value is None else str(value),
            )
        )
    return out


# ---------------------------------------------------------------------------
# WarehouseStatementResult
# ---------------------------------------------------------------------------


class WarehouseStatementResult(StatementResult):
    """Databricks-backed :class:`StatementResult`.

    Wraps a :class:`WarehousePreparedStatement` plus per-execution state
    (``statement_id``, cached :class:`StatementResponse`).  Configuration
    (text, parameters, external tables, routing) lives on
    ``self.statement``.

    The ``warehouse_id`` field is the *resolved* warehouse the statement
    actually ran on (set after submission); ``self.statement.warehouse_id``
    is the *requested* routing hint (set by the caller before submission).

    Retry semantics are inherited from :class:`StatementResult`: the
    looping ``retry()`` method drives ``start(reset=True)`` per attempt,
    sleeping per ``self.statement.retry`` (a :class:`WaitingConfig`)
    between tries.  ``retryable`` is a derived property — non-retryable
    when ``self.statement.retry is None`` or the attempt budget is
    exhausted.
    """

    _PREPARED_STATEMENT_CLASS = WarehousePreparedStatement
    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls) -> "MimeType | None":
        return MimeTypes.DATABRICKS_STATEMENT_RESULT

    executor: "SQLWarehouse"
    statement: WarehousePreparedStatement
    statement_id: Optional[str] = None

    def __init__(
        self,
        executor: "SQLWarehouse",
        statement: Optional[WarehousePreparedStatement] = None,
        *,
        statement_id: Optional[str] = None,
        _response: Optional[StatementResponse] = None,
        **kwargs: Any,
    ):
        self.statement_id = statement_id
        self._response = _response
        super().__init__(statement=statement, executor=executor, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        """True once the statement has been submitted (``statement_id`` present)."""
        return bool(self.statement_id)

    @property
    def cached(self) -> bool:
        """True when the statement is in a terminal state (response is final)."""
        return self.done

    @property
    def client(self):
        return self.executor.client

    def persist(
        self,
        engine: Literal["arrow", "polars", "spark", "auto"] = "auto",
        *,
        data: Any | None = None,
    ) -> "WarehouseStatementResult":
        """No-op for warehouse results — backend already caches the response."""
        return self

    def unpersist(self) -> None:
        """No-op."""
        pass

    def set_api_response(self, response: StatementResponse) -> "WarehouseStatementResult":
        """Test hook: stuff a fully-formed API response into the result."""
        self._response = response
        self.statement_id = response.statement_id
        return self

    def start(
        self,
        reset: bool = False,
        *,
        warehouse: "Optional[SQLWarehouse]" = None,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
        byte_limit: Optional[int] = None,
        row_limit: Optional[int] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        disposition: Optional[Disposition] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        **kwargs: Any,
    ) -> "WarehouseStatementResult":
        """Submit the statement.  Idempotent on already-started results.

        ``reset=True`` cancels the existing submission (when not already
        terminal) and clears local state before resubmitting — this is
        the path :meth:`StatementResult.retry` drives.

        Caller kwargs override anything carried on ``self.statement`` for
        this submission only — the underlying statement's hints stay put.
        """
        if self.started:
            if not reset:
                return self

            # On retry-after-failure the prior submission is already in a
            # terminal state, so cancel() short-circuits.  But guard anyway
            # so a hung-PENDING resubmit also drops the prior server-side
            # state cleanly.
            try:
                self.cancel()
            except Exception:
                logger.exception(
                    "cancel() during start(reset=True) failed for %r; continuing.",
                    self.key,
                )
            self.statement_id = None
            self._response = None
            self._cached_schema = None

        from yggdrasil.databricks.warehouse.warehouse import SQLWarehouse

        eff_wh_id = warehouse_id or self.statement.warehouse_id
        eff_wh_name = warehouse_name or self.statement.warehouse_name

        if warehouse is None:
            warehouse = SQLWarehouse(
                warehouse_id=eff_wh_id,
                warehouse_name=eff_wh_name,
            )

        submitted = warehouse.execute(
            statement=self,
            warehouse_id=eff_wh_id,
            warehouse_name=eff_wh_name,
            byte_limit=byte_limit if byte_limit is not None else self.statement.byte_limit,
            disposition=disposition if disposition is not None else self.statement.disposition,
            row_limit=row_limit if row_limit is not None else self.statement.row_limit,
            catalog_name=catalog_name if catalog_name is not None else self.statement.catalog_name,
            schema_name=schema_name if schema_name is not None else self.statement.schema_name,
            wait=wait,
            raise_error=raise_error,
        )

        # Adopt server-resolved state.  Retry config rides on
        # ``self.statement.retry`` and is preserved by
        # ``self.statement = submitted.statement`` below.
        self.statement = submitted.statement
        self.statement_id = submitted.statement_id
        self.executor = submitted.executor

        return self

    def cancel(self) -> "WarehouseStatementResult":
        """Cancel the running statement.  No-op when not started or already terminal."""
        if not self.started or self.statement_id == "SparkSQL":
            return self
        if self._response is not None and self._response.status.state in DONE_STATES:
            return self

        self.client.workspace_client().statement_execution.cancel_execution(
            statement_id=self.statement_id,
        )
        self._response = None
        self.refresh_status()
        return self

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.started:
            return f"WarehouseStatementResult(url='{self.monitoring_url}')"
        return f"WarehouseStatementResult(text={self.statement.text!r})"

    def __str__(self) -> str:
        return self.monitoring_url if self.started else self.statement.text

    @property
    def monitoring_url(self) -> str:
        """Databricks UI monitoring URL for this statement execution."""
        return "%ssql/warehouses/%s/monitoring?queryId=%s" % (
            self.client.base_url.to_string(),
            self.executor.warehouse_id,
            self.statement_id,
        )

    # ------------------------------------------------------------------
    # State / status
    # ------------------------------------------------------------------

    @property
    def response(self) -> StatementResponse:
        """Latest statement response (auto-refreshes until terminal)."""
        self.refresh_status()
        return self._response

    def api_result_data_at_index(self, chunk_index: int):
        """Fetch a specific result chunk by index via the SDK."""
        return self.client.workspace_client().statement_execution.get_statement_result_chunk_n(
            statement_id=self.statement_id,
            chunk_index=chunk_index,
        )

    def refresh_status(self) -> "WarehouseStatementResult":
        if not self.statement_id:
            self._response = StatementResponse(
                statement_id="unknown",
                status=StatementStatus(state=StatementState.PENDING),
            )
            return self

        statement_execution = self.client.workspace_client().statement_execution
        if self._response is None or self._response.status.state not in DONE_STATES:
            self._response = statement_execution.get_statement(self.statement_id)
        return self

    @property
    def status(self) -> StatementStatus:
        return self.response.status

    @property
    def state(self) -> StatementState:
        return self.status.state

    @property
    def done(self) -> bool:
        return self.state in DONE_STATES

    @property
    def failed(self) -> bool:
        return self.state in FAILED_STATES

    # ------------------------------------------------------------------
    # Transient-failure detection (overrides base)
    # ------------------------------------------------------------------
    #
    # The auto-promote machinery lives on :class:`StatementResult`; this
    # subclass only declares Databricks-specific patterns and tells the
    # base how to read the failure message off a StatementResponse.
    #
    # Most entries are Delta concurrency error codes — they're write-
    # races, idempotent retries always make sense.  ``Please retry the
    # operation`` is the catch-all sentinel Databricks adds to several
    # transient conditions.

    _TRANSIENT_ERROR_PATTERNS: ClassVar[tuple[str, ...]] = (
        # Delta concurrent-append conflicts (partitioned tables w/o RLC,
        # MERGE on overlapping partitions, etc.)
        r"DELTA_CONCURRENT_APPEND",
        r"ConcurrentAppendException",
        # Generic Delta concurrency family — any of these are retry-safe
        # for idempotent writers.
        r"DELTA_CONCURRENT_DELETE_READ",
        r"DELTA_CONCURRENT_DELETE_DELETE",
        r"DELTA_CONCURRENT_WRITE",
        r"DELTA_METADATA_CHANGED",
        # Plain "please retry the operation" — Databricks emits this on
        # several transient conditions and it's a clear signal.
        r"Please retry the operation",
    )

    def _failure_message(self) -> str:
        """Read the backend failure off the cached StatementResponse.

        Pulls ``error_code`` and ``message`` off ``response.status.error``
        (SDK-typed) and falls back to the bare state name.  Returns ``""``
        when nothing is available — base ``_is_transient_failure`` then
        skips the regex search.
        """
        if self._response is None:
            return ""
        status = getattr(self._response, "status", None)
        if status is None:
            return ""
        error = getattr(status, "error", None)
        if error is None:
            return status.state.value if status.state else ""
        # SDK ServiceError carries error_code + message; either may match.
        parts = [
            getattr(error, "error_code", None),
            getattr(error, "message", None),
        ]
        return " ".join(str(p) for p in parts if p)

    def _raise_for_status(self) -> None:
        # Auto-promote happens in the base raise_for_status() before this
        # hook fires; we just raise the backend-specific exception.
        if self.failed:
            raise SQLError.from_statement(self)

    # ------------------------------------------------------------------
    # Manifest / schema
    # ------------------------------------------------------------------

    @property
    def manifest(self):
        self.wait()
        return self.response.manifest

    @property
    def result(self):
        self.wait()
        return self.response.result

    @property
    def disposition(self) -> Optional[Disposition]:
        return self.statement.disposition

    def _collect_schema(self, options) -> Schema:
        if self._cached_schema is None:
            self.wait()
            manifest = self.manifest
            metadata = {
                b"engine": b"databricks-sql",
                b"statement_id": (self.statement_id or "").encode(),
            }
            if manifest is None:
                return schema([], metadata=metadata)

            self._cached_schema = Schema.from_any_fields(
                [parse_databricks_field(c) for c in (manifest.schema.columns or [])],
                metadata=metadata,
            )
        return self._cached_schema

    # ------------------------------------------------------------------
    # External links
    # ------------------------------------------------------------------

    def external_links(self) -> Iterator[ExternalLink]:
        """Yield external result links for ``Disposition.EXTERNAL_LINKS``."""
        if self.disposition != Disposition.EXTERNAL_LINKS:
            raise RuntimeError(
                f"Cannot get external links from {self}; disposition is "
                f"{self.disposition!r}, not EXTERNAL_LINKS"
            )

        result_data = self.result
        wsdk = self.client.workspace_client()

        while True:
            links = result_data.external_links or []
            if not links:
                return
            yield from links

            next_internal = getattr(links[-1], "next_chunk_internal_link", None)
            if not next_internal:
                return

            try:
                chunk_index = int(next_internal.rstrip("/").split("/")[-1])
            except Exception as e:
                raise ValueError(f"Bad next_chunk_internal_link {next_internal!r}: {e}") from e

            try:
                result_data = wsdk.statement_execution.get_statement_result_chunk_n(
                    statement_id=self.statement_id,
                    chunk_index=chunk_index,
                )
            except Exception as e:
                raise ValueError(f"Cannot retrieve data batch from {next_internal!r}: {e}") from e

    # ------------------------------------------------------------------
    # Arrow conversions
    # ------------------------------------------------------------------

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        options = options.check_target(self.collect_schema)

        max_workers = 4
        max_in_flight = max_workers * 2

        retry = urllib3.Retry(
            total=4,
            backoff_factor=0.2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        http = urllib3.PoolManager(
            num_pools=max_workers * 2,
            maxsize=max_workers * 2,
            retries=retry,
            timeout=urllib3.Timeout(connect=2.0, read=5.0),
            headers={"Accept-Encoding": "gzip,deflate"},
            cert_reqs="CERT_REQUIRED",
        )

        row_size = options.row_size
        byte_size = options.byte_size or _DEFAULT_BYTE_SIZE
        max_chunksize = row_size if row_size else None

        def fetch_batches(url: str) -> Iterator[pa.RecordBatch]:
            resp = http.request("GET", url, preload_content=True)
            try:
                if resp.status >= 400:
                    raise RuntimeError(f"GET {url} failed: {resp.status}")
                buf = memoryview(resp.data)
            finally:
                resp.release_conn()

            with pa.input_stream(buf) as src:
                reader = pipc.open_stream(src)
                yield from reader

        def jobs() -> Iterable[Job]:
            for link in self.external_links():
                if link.external_link:
                    yield Job.make(fetch_batches, link.external_link)

        def raw_batches() -> Iterator[pa.RecordBatch]:
            with JobPoolExecutor.parse(max_workers) as ex:
                for result in ex.as_completed(
                    jobs(),
                    ordered=True,
                    max_in_flight=max_in_flight,
                    cancel_on_exit=True,
                    shutdown_on_exit=True,
                    shutdown_wait=False,
                ):
                    yield from result.result

        pending: List[pa.RecordBatch] = []
        pending_bytes = 0
        arrow_schema: Optional[pa.Schema] = None

        def flush() -> Iterator[pa.RecordBatch]:
            nonlocal pending, pending_bytes
            if not pending:
                return

            casted = options.cast_arrow_tabular(
                pa.concat_batches(pending, memory_pool=options.arrow_memory_pool)
            )
            pending = []
            pending_bytes = 0
            yield casted

        for batch in raw_batches():
            if arrow_schema is None:
                arrow_schema = batch.schema
            pending.append(batch)
            pending_bytes += batch.nbytes
            if pending_bytes >= byte_size:
                yield from flush()

        yield from flush()

    def _write_arrow_batches(self, batches: Iterable[pa.RecordBatch], options: CastOptions) -> None:
        raise NotImplementedError("Cannot write to Databricks SQL")


# ---------------------------------------------------------------------------
# WarehouseStatementBatch
# ---------------------------------------------------------------------------


class WarehouseStatementBatch(StatementBatch):
    """Warehouse-backed batch of statements.

    External-table aliases are resolved at coerce-time by reading the
    per-statement ``external_volume_paths`` and rewriting ``{alias}``
    occurrences in the text to ``parquet.\\`<full_path>\\```.  Rewriting
    happens on a *copy*, so re-submitting the same batch is safe.

    The optional ``external_paths`` constructor argument is a batch-wide
    set of aliases applied on top of any per-statement registry — useful
    for shared scratch volumes that every statement in the batch reads.
    Per-statement entries take precedence on alias collisions.

    :meth:`retry` is inherited from :class:`StatementBatch` — it walks
    the result map, picks every entry that is both ``failed`` and
    ``retryable``, and reissues each via :meth:`StatementResult.retry`
    on the configured ``parallel`` thread pool.
    """

    external_volume_paths: dict[str, VolumePath]

    def __init__(
        self,
        executor: "SQLWarehouse",
        statements: Optional[Iterable["WarehousePreparedStatement | str"]] = None,
        *,
        parallel: int = 1,
        external_paths: Optional[dict[str, VolumePath]] = None,
    ):
        super().__init__(executor=executor, statements=None, parallel=parallel)
        self.external_volume_paths = dict(external_paths) if external_paths else {}
        if statements:
            self.extend(statements)

    def _coerce(self, statement: "WarehousePreparedStatement | str") -> WarehousePreparedStatement:
        stmt = WarehousePreparedStatement.from_(statement)

        # Effective alias map for this statement: batch-wide + per-statement
        # (per-statement wins on collision).
        effective: dict[str, VolumePath] = dict(self.external_volume_paths)
        if stmt.external_volume_paths:
            effective.update(stmt.external_volume_paths)
        if not effective:
            return stmt

        # Substitute on a copy — never mutate the caller's statement.
        rewritten_text = stmt.text
        for alias, path in effective.items():
            rewritten_text = rewritten_text.replace(
                "{%s}" % alias,
                f"parquet.`{path.full_path()}`",
            )
        if rewritten_text == stmt.text:
            return stmt

        copied = copy_mod.copy(stmt)
        copied.text = rewritten_text
        return copied

    def clear_temporary_resources(self) -> "WarehouseStatementBatch":
        # Per-statement scratch first (each result owns its statement).
        super().clear_temporary_resources()

        # Batch-wide scratch second.
        for alias, path in list(self.external_volume_paths.items()):
            if getattr(path, "temporary", False):
                try:
                    Job.make(path.unlink, missing_ok=True).fire_and_forget()
                except Exception:
                    logger.exception(
                        "Failed to unlink temporary path %r (alias=%r); continuing.",
                        path, alias,
                    )

        self.external_volume_paths = None

        return self