"""
Databricks SQL Warehouse resource — individual warehouse lifecycle management.

This module exposes the :class:`SQLWarehouse` resource, a lightweight
wrapper around a single Databricks SQL Warehouse that provides:

- property-based status checks (running, pending, serverless)
- lifecycle helpers: start, stop, delete, wait_for_status
- configuration updates and permission management
- SQL statement execution with retry / back-off logic

Executor contract
-----------------
:class:`SQLWarehouse` is a concrete :class:`StatementExecutor` —
``_submit_statement`` reads typed warehouse fields off
:class:`WarehousePreparedStatement` and submits via the SDK.

Execution policy travels through :class:`DatabricksExecutionOptions`
(``wait``, ``raise_error``, ``parallel`` from the base, plus
``submit_wait``, ``external_data``, ``external_volume_paths``).  The base
:meth:`StatementExecutor._execute` hook handles submit + wait/raise; this
subclass overrides it to stage external data first and merge volume
paths onto the statement.

Collection-level operations (listing, finding, creating warehouses) live
in the companion :mod:`~yggdrasil.databricks.sql.service` module.
"""

from __future__ import annotations

import dataclasses as dc
import logging
import time
from dataclasses import dataclass, replace
from typing import Any, ClassVar, List, Mapping, Optional, Union

from databricks.sdk.errors import DeadlineExceeded
from databricks.sdk.service.sql import (
    Disposition,
    EndpointInfo,
    Format,
    State,
    WarehouseAccessControlRequest,
)

from yggdrasil.concurrent.threading import Job
from yggdrasil.data.executor import ExecutionOptions, StatementExecutor
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.pyutils.equality import dicts_equal
from .service import (
    DEFAULT_ALL_PURPOSE_CLASSIC_NAME,
    DEFAULT_ALL_PURPOSE_SERVERLESS_NAME,
    Warehouses,
    _EDIT_ARG_NAMES,
    _jitter_sleep_seconds,
    safeEndpointInfo,
)
from .statement import (
    WarehousePreparedStatement,
    WarehouseStatementResult,
    WarehouseStatementBatch,
)
from ..client import DatabricksResource
from ..fs import VolumePath

__all__ = [
    "SQLWarehouse",
    "DatabricksExecutionOptions",
    "DEFAULT_ALL_PURPOSE_SERVERLESS_NAME",
    "DEFAULT_ALL_PURPOSE_CLASSIC_NAME",
]

LOGGER = logging.getLogger(__name__)


_DEFAULT_SUBMIT_WAIT = WaitingConfig(
    timeout=300,
    interval=1.0,
    backoff=1.8,
    max_interval=8.0,
)

# This module assumes :class:`WarehousePreparedStatement` carries a
# ``submit_wait: Optional[WaitingConfig] = None`` field — set on the
# statement by :meth:`SQLWarehouse._apply_databricks_options` and read
# back by :meth:`_submit_statement`.  Without that field, the retry
# policy falls back to ``_DEFAULT_SUBMIT_WAIT`` everywhere.


# ---------------------------------------------------------------------------
# DatabricksExecutionOptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatabricksExecutionOptions(ExecutionOptions):
    """Databricks-specific :class:`ExecutionOptions`.

    Adds three knobs on top of the base contract:

    submit_wait
        Retry policy for the SDK ``execute_statement`` call itself
        (covers cold/busy warehouses returning ``DeadlineExceeded``).
        Distinct from ``wait``, which controls *result-level* polling.
    external_data
        ``{alias: tabular-or-VolumePath}``.  Tabular values are staged to
        Parquet on a fresh :class:`VolumePath`; existing volumes pass
        through.  Aliases used as ``{alias}`` in statement text.
    external_volume_paths
        ``{alias: VolumePath}``.  Already-staged volumes — same alias
        space as ``external_data``.  Use this when callers stage their
        own data and want to reuse the volume across calls.

    Stage / volume merge precedence (lowest → highest):
    statement's existing ``external_volume_paths`` <
    ``external_volume_paths`` from options <
    ``external_data`` (just-staged) from options.  This matches the
    "I just told you to stage this; use it" intent.
    """

    submit_wait: WaitingConfigArg = None
    external_data: Optional[Mapping[str, Any]] = None
    external_volume_paths: Optional[Mapping[str, VolumePath]] = None

    def with_external_data(
        self,
        external_data: Optional[Mapping[str, Any]],
    ) -> "DatabricksExecutionOptions":
        return replace(self, external_data=external_data)

    def with_external_volume_paths(
        self,
        external_volume_paths: Optional[Mapping[str, VolumePath]],
    ) -> "DatabricksExecutionOptions":
        return replace(self, external_volume_paths=external_volume_paths)

    def with_submit_wait(self, submit_wait: WaitingConfigArg) -> "DatabricksExecutionOptions":
        return replace(self, submit_wait=submit_wait)


# ---------------------------------------------------------------------------
# SQLWarehouse
# ---------------------------------------------------------------------------


@dc.dataclass(init=False, repr=False)
class SQLWarehouse(
    DatabricksResource,
    StatementExecutor[
        WarehousePreparedStatement,
        WarehouseStatementResult,
        WarehouseStatementBatch,
    ],
):
    """High-level Databricks SQL Warehouse resource and statement executor.

    Parameters
    ----------
    service:
        Parent :class:`~yggdrasil.databricks.sql.service.Warehouses` service.
    warehouse_id:
        Databricks warehouse id.
    warehouse_name:
        Warehouse display name.  When provided without ``warehouse_id``
        the warehouse is resolved by name during construction.

    Notes
    -----
    ``SQLWarehouse`` caches ``EndpointInfo`` details and refreshes them
    lazily.  Use :meth:`refresh` to force a reload.
    """

    # Pin concrete types so base coercion + result construction produce
    # the right subclasses.
    _PREPARED_STATEMENT_CLASS: ClassVar[type[WarehousePreparedStatement]] = WarehousePreparedStatement
    _STATEMENT_RESULT_CLASS: ClassVar[type[WarehouseStatementResult]] = WarehouseStatementResult
    _STATEMENT_BATCH_CLASS: ClassVar[type[WarehouseStatementBatch]] = WarehouseStatementBatch

    service: Warehouses = dc.field(
        default_factory=Warehouses.current,
        repr=False,
        compare=False,
    )
    warehouse_id: str | None = None
    warehouse_name: str | None = None

    _details: Optional[EndpointInfo] = dc.field(default=None, repr=False, hash=False, compare=False)

    def __init__(
        self,
        service: Warehouses | None = None,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        **kwargs,
    ):
        self.service = Warehouses.current() if service is None else service
        self.warehouse_id = warehouse_id
        self.warehouse_name = warehouse_name
        super().__init__()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.warehouse_name and not self.warehouse_id:
            found = self.service.find_warehouse(warehouse_name=self.warehouse_name)
            self.warehouse_id = found.warehouse_id
            self.warehouse_name = found.warehouse_name
            self._details = found._details

    def __call__(
        self,
        *,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
    ) -> "SQLWarehouse":
        if not warehouse_id and not warehouse_name:
            return self
        if warehouse_id == self.warehouse_id or warehouse_name == self.warehouse_name:
            return self
        return self.service.find_warehouse(
            warehouse_id=warehouse_id,
            warehouse_name=warehouse_name,
            raise_error=True,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.explore_url.to_string()!r})"

    def __str__(self) -> str:
        return self.warehouse_id or self.warehouse_name or f"{self.__class__.__name__}(<not initialized>)"

    # ------------------------------------------------------------------
    # Details caching and state
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.warehouse_name

    @property
    def id(self) -> str:
        return self.warehouse_id

    @property
    def details(self) -> EndpointInfo:
        """Return cached warehouse details, fetching them lazily if needed."""
        if self._details is None:
            self.refresh()
        return self._details

    def latest_details(self):
        """Return freshly-fetched warehouse details from the API."""
        return self.client.workspace_client().warehouses.get(id=self.warehouse_id)

    def refresh(self) -> "SQLWarehouse":
        """Refresh cached details from the API and return self."""
        self._details = safeEndpointInfo(self.latest_details())
        return self

    @property
    def state(self):
        """Return the current warehouse state (always hits the API)."""
        return self.latest_details().state

    @property
    def is_serverless(self) -> bool:
        return self.details.enable_serverless_compute

    @property
    def is_running(self) -> bool:
        return self.state == State.RUNNING

    @property
    def is_pending(self) -> bool:
        return self.state in {State.DELETING, State.STARTING, State.STOPPING}

    @property
    def explore_url(self):
        return self.client.base_url.joinpath(f"/sql/warehouses/{self.warehouse_id}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def wait_for_status(self, wait: WaitingConfigArg = None) -> "SQLWarehouse":
        """Poll until the warehouse leaves any pending state."""
        wait = WaitingConfig.default() if wait is None else WaitingConfig.from_(wait)

        start = time.time()
        iteration = 0

        if wait.timeout:
            LOGGER.debug(
                "Waiting for warehouse %s (%s) to leave pending state (timeout=%.0fs)",
                self.warehouse_name, self.warehouse_id, wait.timeout,
            )
            while self.is_pending:
                wait.sleep(iteration=iteration, start=start)
                iteration += 1
            LOGGER.debug(
                "Warehouse %s (%s) ready after %.1fs",
                self.warehouse_name, self.warehouse_id, time.time() - start,
            )
        return self

    def start(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "SQLWarehouse":
        """Start the warehouse if it is not already running."""
        client = self.client.workspace_client().warehouses

        if self.warehouse_id and not self.is_running:
            LOGGER.debug("Starting warehouse %s (%s)", self.warehouse_name, self.warehouse_id)
            try:
                response = client.start(id=self.warehouse_id)
            except Exception:
                if raise_error:
                    raise
                LOGGER.warning(
                    "Failed to start warehouse %s (%s)",
                    self.warehouse_name, self.warehouse_id,
                )
                return self

            if wait:
                wait_cfg = WaitingConfig.from_(wait)
                response.result(timeout=wait_cfg.timeout_timedelta)

            LOGGER.info("Started warehouse %s (%s)", self.warehouse_name, self.warehouse_id)
        return self

    def stop(self):
        """Stop the warehouse if it is running."""
        if not self.is_running:
            return self
        LOGGER.debug("Stopping warehouse %s (%s)", self.warehouse_name, self.warehouse_id)
        client = self.client.workspace_client().warehouses
        result = client.stop(id=self.warehouse_id)
        LOGGER.info("Stopped warehouse %s (%s)", self.warehouse_name, self.warehouse_id)
        return result

    def delete(self) -> None:
        """Delete the warehouse."""
        if not self.warehouse_id:
            return
        LOGGER.debug("Deleting warehouse %s (%s)", self.warehouse_name, self.warehouse_id)
        self.client.workspace_client().warehouses.delete(id=self.warehouse_id)
        LOGGER.info("Deleted warehouse %s (%s)", self.warehouse_name, self.warehouse_id)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        wait: WaitingConfigArg = None,
        permissions: Optional[List[Union[WarehouseAccessControlRequest, str]]] = None,
        **warehouse_specs,
    ) -> "SQLWarehouse":
        """Apply spec changes, skipping the API when already up-to-date."""
        if not warehouse_specs:
            LOGGER.debug("update: no specs provided for %s — skipping", self.warehouse_name)
            return self

        wait = WaitingConfig.from_(wait)

        existing_details = {
            k: v
            for k, v in self.details.as_shallow_dict().items()
            if k in _EDIT_ARG_NAMES
        }
        update_details = {
            k: v
            for k, v in (
                self.service._check_details(
                    details=self.details,
                    update=True,
                    keys=_EDIT_ARG_NAMES,
                    **warehouse_specs,
                )
                .as_shallow_dict()
                .items()
            )
            if k in _EDIT_ARG_NAMES
        }

        if not dicts_equal(existing_details, update_details, keys=_EDIT_ARG_NAMES):
            LOGGER.debug(
                "Updating warehouse %s (%s) with %s",
                self.warehouse_name, self.warehouse_id, update_details,
            )
            sdk_client = self.client.workspace_client().warehouses
            if wait.timeout:
                new_details = sdk_client.edit_and_wait(
                    timeout=wait.timeout_timedelta,
                    **update_details,
                )
            else:
                _ = sdk_client.edit(**update_details)
                new_details = EndpointInfo(**update_details)
            self._details = safeEndpointInfo(new_details)
        else:
            LOGGER.debug(
                "update: warehouse %s (%s) already up-to-date — skipping API call",
                self.warehouse_name, self.warehouse_id,
            )

        if permissions:
            self.update_permissions(permissions=permissions, wait=wait)

        LOGGER.info("Updated warehouse %s (%s)", self.warehouse_name, self.warehouse_id)
        return self

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def update_permissions(
        self,
        permissions: Optional[List[Union[WarehouseAccessControlRequest, str]]] = None,
        *,
        wait: WaitingConfigArg = True,
        warehouse_id: str | None = None,
    ) -> "SQLWarehouse":
        """Apply ACL entries to this warehouse."""
        sdk_client = self.client.workspace_client().warehouses
        warehouse_id = warehouse_id or self.warehouse_id

        checked = (
            [self.service.check_permission(p) for p in permissions]
            if permissions else []
        )

        if checked and warehouse_id:
            kwargs = dict(warehouse_id=warehouse_id, access_control_list=checked)
            if wait:
                sdk_client.update_permissions(**kwargs)
            else:
                Job.make(sdk_client.update_permissions, **kwargs).fire_and_forget()
        return self

    @staticmethod
    def check_permission(
        permission: Union[WarehouseAccessControlRequest, str],
    ) -> WarehouseAccessControlRequest:
        """Delegate to :meth:`Warehouses.check_permission` (kept for back-compat)."""
        return Warehouses.check_permission(permission)

    # ------------------------------------------------------------------
    # SQL execution — executor contract
    # ------------------------------------------------------------------

    def _execute(
        self,
        statement: WarehousePreparedStatement,
        options: ExecutionOptions,
    ) -> WarehouseStatementResult:
        """Stage external data and forward to the base ``_execute``.

        :class:`DatabricksExecutionOptions` carries Databricks-specific
        staging instructions; their effect is purely to mutate the
        statement before submission.  Plain :class:`ExecutionOptions` are
        accepted — they just skip the staging step.
        """
        if isinstance(options, DatabricksExecutionOptions):
            statement = self._apply_databricks_options(statement, options)
        return super()._execute(statement, options)

    def _apply_databricks_options(
        self,
        statement: WarehousePreparedStatement,
        options: DatabricksExecutionOptions,
    ) -> WarehousePreparedStatement:
        """Bake Databricks-specific options into the statement.

        - ``external_data`` is staged to fresh Parquet volumes.
        - ``external_volume_paths`` is merged onto the statement.
        - ``submit_wait`` is copied onto the statement so
          :meth:`_submit_statement` can read it without a side-channel.

        Merge precedence on alias collisions (lowest → highest): existing
        statement paths < ``options.external_volume_paths`` < just-staged.
        """
        # ---- Staging ----
        staged = WarehousePreparedStatement.check_external_data(
            options.external_data,
            catalog_name=statement.catalog_name,
            schema_name=statement.schema_name,
        )

        if staged or options.external_volume_paths:
            merged: dict[str, VolumePath] = dict(statement.external_volume_paths or {})
            if options.external_volume_paths:
                merged.update(options.external_volume_paths)
            merged.update(staged)
            statement.external_volume_paths = merged or None

        # ---- Submit-retry policy ----
        # Use setattr so this works even if WarehousePreparedStatement
        # hasn't been migrated to declare submit_wait as a typed field.
        if options.submit_wait is not None:
            statement.submit_wait = WaitingConfig.from_(options.submit_wait)

        return statement

    def _submit_statement(
        self,
        statement: WarehousePreparedStatement,
    ) -> WarehouseStatementResult:
        """Submit ``statement`` via the SDK, with busy-warehouse retry.

        Reads typed routing fields directly off the statement.  The
        ``submit_wait`` policy (if any) is picked up from
        :attr:`_submit_wait_for_call`, set by :meth:`_execute` from
        :class:`DatabricksExecutionOptions.submit_wait`.

        Defaults applied in-line:

        - ``format`` → :attr:`Format.ARROW_STREAM`
        - ``disposition`` → :attr:`Disposition.EXTERNAL_LINKS` (forced
          for CSV/ARROW_STREAM since INLINE only supports JSON_ARRAY)
        """
        target_wh_id = statement.warehouse_id or self.warehouse_id

        format_ = statement.format or Format.ARROW_STREAM
        if format_ in (Format.CSV, Format.ARROW_STREAM):
            disposition = Disposition.EXTERNAL_LINKS
        else:
            disposition = statement.disposition or Disposition.INLINE

        sdk_client = self.client.workspace_client().statement_execution

        LOGGER.debug(
            "Executing SQL on warehouse %s (%s):\n%s",
            self.warehouse_name, target_wh_id, statement.text,
        )

        # Submission-level retry: distinct from result-level polling.
        # Read from the statement (set by _apply_databricks_options).
        # `getattr` keeps this safe during the migration where older
        # WarehousePreparedStatement builds may not have the field yet.
        submit_wait = getattr(statement, "submit_wait", None) or _DEFAULT_SUBMIT_WAIT
        started_at = time.monotonic()
        deadline = (
            started_at + submit_wait.timeout
            if submit_wait and submit_wait.timeout
            else None
        )

        response = self._submit_with_retry(
            sdk_client=sdk_client,
            statement=statement,
            target_wh_id=target_wh_id,
            disposition=disposition,
            format_=format_,
            submit_wait=submit_wait,
            deadline=deadline,
        )

        result = WarehouseStatementResult(
            executor=self,
            statement=statement,
        ).set_api_response(response)

        LOGGER.info(
            "Executed %r on warehouse %s (%s)",
            result, self.warehouse_name, target_wh_id,
        )
        return result

    def _submit_with_retry(
        self,
        *,
        sdk_client,
        statement: WarehousePreparedStatement,
        target_wh_id: str | None,
        disposition: Disposition,
        format_: Format,
        submit_wait: WaitingConfig | None,
        deadline: float | None,
    ):
        """Inner retry loop for SDK ``execute_statement``.

        Retries on :class:`DeadlineExceeded` (warehouse busy/cold) until
        the deadline is exhausted.  Distinct from result-level polling,
        which the base ``wait()`` handles.
        """
        iteration = 0
        started_at = time.monotonic()
        while True:
            try:
                return sdk_client.execute_statement(
                    statement=statement.text,
                    warehouse_id=target_wh_id,
                    byte_limit=statement.byte_limit,
                    disposition=disposition,
                    format=format_,
                    on_wait_timeout=statement.on_wait_timeout,
                    parameters=statement.to_parameter_list(),
                    row_limit=statement.row_limit,
                    wait_timeout=statement.wait_timeout,
                    catalog=statement.catalog_name,
                    schema=statement.schema_name,
                )
            except DeadlineExceeded:
                remaining = (
                    None if deadline is None
                    else (deadline - time.monotonic())
                )
                if remaining is not None and remaining <= 0:
                    LOGGER.error(
                        "Submit deadline exceeded for warehouse %s (%s) "
                        "after %.2fs and %d retries",
                        self.warehouse_name, target_wh_id,
                        time.monotonic() - started_at, iteration,
                    )
                    raise

                sleep_for = _jitter_sleep_seconds(
                    submit_wait, iteration=iteration, remaining=remaining,
                )
                LOGGER.warning(
                    "Warehouse %s (%s) is busy; execute submit hit DeadlineExceeded. "
                    "Retrying in %.2fs (attempt=%d)",
                    self.warehouse_name, target_wh_id, sleep_for, iteration + 1,
                )
                if sleep_for > 0:
                    time.sleep(sleep_for)
                iteration += 1

    # ------------------------------------------------------------------
    # Public execute() — back-compat shim over the typed pipeline
    # ------------------------------------------------------------------

    def execute(
        self,
        statement: "str | WarehousePreparedStatement | WarehouseStatementResult | None" = None,
        *,
        options: Optional[DatabricksExecutionOptions] = None,
        # Routing
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        # Per-statement config (forwarded to PreparedStatement.prepare)
        byte_limit: int | None = None,
        disposition: Optional[Disposition] = None,
        format: Optional[Format] = None,
        parameters: Optional[Mapping[str, Any] | List[Any]] = None,
        row_limit: int | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        # External data — also exposable via DatabricksExecutionOptions
        external_data: Optional[Mapping[str, Any]] = None,
        external_volume_paths: Optional[Mapping[str, VolumePath]] = None,
        # Execution policy
        wait: WaitingConfigArg = True,
        submit_wait: WaitingConfigArg = None,
        raise_error: bool = True,
    ) -> WarehouseStatementResult:
        """Execute a SQL statement on this (or another) warehouse.

        Three ways to control execution policy:

        1. Per-call kwargs (``wait``, ``raise_error``, ``submit_wait``,
           ``external_data``, ``external_volume_paths``) — ergonomic, the
           default API.
        2. ``options=DatabricksExecutionOptions(...)`` — when the same
           policy is reused across many calls.
        3. Both — kwargs override fields they explicitly set.

        Already-started results (those carrying a ``statement_id``) are
        returned with a ``wait()`` rather than re-submitted.

        ``warehouse_id`` / ``warehouse_name`` redirect submission to a
        different warehouse — kept for back-compat with callers that use
        one warehouse handle as a dispatcher.
        """
        # Cross-warehouse redirect: resolve and delegate.
        if (warehouse_id and warehouse_id != self.warehouse_id) or (
            warehouse_name and warehouse_name != self.warehouse_name
        ):
            other = self.service.find_warehouse(
                warehouse_id=warehouse_id, warehouse_name=warehouse_name,
            )
            return other.execute(
                statement=statement,
                options=options,
                byte_limit=byte_limit,
                disposition=disposition,
                format=format,
                parameters=parameters,
                row_limit=row_limit,
                catalog_name=catalog_name,
                schema_name=schema_name,
                external_data=external_data,
                external_volume_paths=external_volume_paths,
                wait=wait,
                submit_wait=submit_wait,
                raise_error=raise_error,
            )

        # Already-started result: just wait.
        if isinstance(statement, WarehouseStatementResult) and statement.started:
            return statement.wait(wait=wait, raise_error=raise_error)

        # Coerce + bind onto a typed PreparedStatement.  prepare() handles
        # parameters, format/disposition defaults, and the staging+merge
        # of external_data / external_volume_paths in one shot.
        prepared = WarehousePreparedStatement.prepare(
            statement if statement is not None else "",
            parameters=parameters,
            external_data=external_data,
            external_volume_paths=external_volume_paths,
            catalog_name=catalog_name,
            schema_name=schema_name,
            disposition=disposition,
            format=format,
            byte_limit=byte_limit,
            row_limit=row_limit,
        )

        opts = self._build_options(
            options,
            wait=wait,
            raise_error=raise_error,
            submit_wait=submit_wait,
        )

        # Hand off to base lifecycle: _execute (overridden above) stages
        # any extra external data, then calls _submit_statement and
        # applies wait / raise_error.
        coerced = self._coerce_statement(prepared)
        return self._execute(coerced, opts)

    def _build_options(
        self,
        options: Optional[DatabricksExecutionOptions],
        *,
        wait: WaitingConfigArg,
        raise_error: bool,
        submit_wait: WaitingConfigArg,
    ) -> ExecutionOptions:
        """Merge per-call kwargs onto a base options object.

        Per-call ``submit_wait`` is honored when the kwarg is non-None;
        otherwise the existing ``options.submit_wait`` (if any) wins.
        """
        if options is None:
            if submit_wait is None and wait is True and raise_error is True:
                # Cheapest path — skip the dataclass construction.
                return ExecutionOptions()
            return DatabricksExecutionOptions(
                wait=wait,
                raise_error=raise_error,
                submit_wait=submit_wait,
            )

        # Diff against defaults so passing the kwarg defaults doesn't
        # clobber options' fields.
        defaults = DatabricksExecutionOptions()
        overrides: dict[str, Any] = {}
        if wait != defaults.wait:
            overrides["wait"] = wait
        if raise_error != defaults.raise_error:
            overrides["raise_error"] = raise_error
        if submit_wait is not None:
            overrides["submit_wait"] = submit_wait
        if not overrides:
            return options
        return replace(options, **overrides)