"""
Databricks SQL Warehouse resource – individual warehouse lifecycle management.

This module exposes the :class:`SQLWarehouse` resource, a lightweight wrapper
around a single Databricks SQL Warehouse that provides:

- property-based status checks (running, pending, serverless)
- lifecycle helpers: start, stop, delete, wait_for_status
- configuration updates and permission management
- SQL statement execution with retry / back-off logic

Collection-level operations (listing, finding, creating warehouses) live in the
companion :mod:`~yggdrasil.databricks.sql.service` module.
"""

from __future__ import annotations

import dataclasses as dc
import logging
import time
from typing import Optional, List, Union

from databricks.sdk.errors import DeadlineExceeded
from databricks.sdk.service.sql import (
    State, EndpointInfo,
    Disposition, Format,
    ExecuteStatementRequestOnWaitTimeout, StatementParameterListItem,
    WarehouseAccessControlRequest,
)

from yggdrasil.concurrent.threading import Job
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.pyutils.equality import dicts_equal
from .service import (
    Warehouses,
    DEFAULT_ALL_PURPOSE_CLASSIC_NAME,
    DEFAULT_ALL_PURPOSE_SERVERLESS_NAME,
    _EDIT_ARG_NAMES,
    _jitter_sleep_seconds,
    safeEndpointInfo,
)
from .statement_result import StatementResult
from ..client import DatabricksResource

__all__ = [
    "SQLWarehouse",
    "DEFAULT_ALL_PURPOSE_SERVERLESS_NAME",
    "DEFAULT_ALL_PURPOSE_CLASSIC_NAME",
]

LOGGER = logging.getLogger(__name__)


@dc.dataclass
class SQLWarehouse(DatabricksResource):
    """
    High-level Databricks SQL Warehouse resource.

    Parameters
    ----------
    service:
        Parent :class:`~yggdrasil.databricks.sql.service.Warehouses` service.
    warehouse_id:
        Databricks warehouse id.
    warehouse_name:
        Warehouse display name.  When provided without ``warehouse_id`` the
        warehouse is resolved by name during construction.

    Notes
    -----
    ``SQLWarehouse`` caches ``EndpointInfo`` details and refreshes them
    lazily.  Use :meth:`refresh` to force a reload.
    """

    service: Warehouses = dc.field(
        default_factory=Warehouses.current,
        repr=False,
        compare=False,
    )
    warehouse_id: Optional[str] = None
    warehouse_name: Optional[str] = None

    _details: Optional[EndpointInfo] = dc.field(
        default=None, repr=False, hash=False, compare=False,
    )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.warehouse_name and not self.warehouse_id:
            found = self.service.find_warehouse(warehouse_name=self.warehouse_name)
            object.__setattr__(self, "warehouse_id", found.warehouse_id)
            object.__setattr__(self, "warehouse_name", found.warehouse_name)
            object.__setattr__(self, "_details", found._details)

    def __call__(
        self,
        *,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
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
        return (
            f"{self.__class__.__name__}("
            f"warehouse_name={self.warehouse_name!r}, "
            f"warehouse_id={self.warehouse_id!r})"
        )

    # ------------------------------------------------------------------ #
    # Details caching and state
    # ------------------------------------------------------------------ #

    @property
    def details(self) -> EndpointInfo:
        """Return cached warehouse details, fetching them lazily if needed."""
        if self._details is None:
            self.refresh()
        return self._details

    def latest_details(self):
        """Return freshly fetched warehouse details from the API."""
        return self.client.workspace_client().warehouses.get(id=self.warehouse_id)

    def refresh(self) -> "SQLWarehouse":
        """Refresh cached details from the API and return self."""
        checked = safeEndpointInfo(self.latest_details())
        object.__setattr__(self, "_details", checked)
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

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def wait_for_status(self, wait: WaitingConfigArg = None) -> "SQLWarehouse":
        """Poll until the warehouse leaves any pending state."""
        wait = WaitingConfig.default() if wait is None else WaitingConfig.check_arg(wait)

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

        if self.warehouse_id:
            if not self.is_running:
                LOGGER.debug(
                    "Starting warehouse %s (%s)",
                    self.warehouse_name, self.warehouse_id,
                )
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
                    wait = WaitingConfig.check_arg(wait)
                    response.result(timeout=wait.timeout_timedelta)

                LOGGER.info(
                    "Started warehouse %s (%s)",
                    self.warehouse_name, self.warehouse_id,
                )

        return self

    def stop(self):
        """Stop the warehouse if it is running."""
        if self.is_running:
            LOGGER.debug(
                "Stopping warehouse %s (%s)",
                self.warehouse_name, self.warehouse_id,
            )
            client = self.client.workspace_client().warehouses
            result = client.stop(id=self.warehouse_id)
            LOGGER.info(
                "Stopped warehouse %s (%s)",
                self.warehouse_name, self.warehouse_id,
            )
            return result
        return self

    def delete(self) -> None:
        """Delete the warehouse."""
        if self.warehouse_id:
            LOGGER.debug(
                "Deleting warehouse %s (%s)",
                self.warehouse_name, self.warehouse_id,
            )
            self.client.workspace_client().warehouses.delete(id=self.warehouse_id)
            LOGGER.info(
                "Deleted warehouse %s (%s)",
                self.warehouse_name, self.warehouse_id,
            )

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(
        self,
        wait: WaitingConfigArg = None,
        permissions: Optional[List[Union[WarehouseAccessControlRequest, str]]] = None,
        **warehouse_specs,
    ) -> "SQLWarehouse":
        """Apply spec changes to this warehouse, skipping the API when already up-to-date."""
        if not warehouse_specs:
            LOGGER.debug(
                "update: no specs provided for %s — skipping", self.warehouse_name,
            )
            return self

        wait = WaitingConfig.check_arg(wait)

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

        same = dicts_equal(existing_details, update_details, keys=_EDIT_ARG_NAMES)

        if not same:
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

            object.__setattr__(self, "_details", safeEndpointInfo(new_details))
        else:
            LOGGER.debug(
                "update: warehouse %s (%s) already up-to-date — skipping API call",
                self.warehouse_name, self.warehouse_id,
            )

        if permissions:
            self.update_permissions(permissions=permissions, wait=wait)

        LOGGER.info("Updated warehouse %s (%s)", self.warehouse_name, self.warehouse_id)
        return self

    # ------------------------------------------------------------------ #
    # Permissions
    # ------------------------------------------------------------------ #

    def update_permissions(
        self,
        permissions: Optional[List[Union[WarehouseAccessControlRequest, str]]] = None,
        *,
        wait: WaitingConfigArg = True,
        warehouse_id: Optional[str] = None,
    ) -> "SQLWarehouse":
        """Apply ACL entries to this warehouse."""
        sdk_client = self.client.workspace_client().warehouses
        warehouse_id = warehouse_id or self.warehouse_id

        checked = [
            self.service.check_permission(p)
            for p in permissions
        ] if permissions else []

        if checked and warehouse_id:
            if wait:
                sdk_client.update_permissions(
                    warehouse_id=warehouse_id,
                    access_control_list=checked,
                )
            else:
                Job.make(
                    sdk_client.update_permissions,
                    warehouse_id=warehouse_id,
                    access_control_list=checked,
                ).fire_and_forget()

        return self

    @staticmethod
    def check_permission(
        permission: Union[WarehouseAccessControlRequest, str],
    ) -> WarehouseAccessControlRequest:
        """Delegate to :meth:`Warehouses.check_permission` (kept for back-compat)."""
        return Warehouses.check_permission(permission)

    # ------------------------------------------------------------------ #
    # SQL execution
    # ------------------------------------------------------------------ #

    def execute(
        self,
        statement: Optional[str] = None,
        *,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
        byte_limit: Optional[int] = None,
        disposition: Optional[Disposition] = None,
        format: Optional[Format] = None,
        on_wait_timeout: Optional[ExecuteStatementRequestOnWaitTimeout] = None,
        parameters: Optional[List[StatementParameterListItem]] = None,
        row_limit: Optional[int] = None,
        wait_timeout: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        wait: WaitingConfigArg = True,
        submit_wait: WaitingConfigArg = None,
        raise_error: bool = True,
    ) -> StatementResult:
        """Execute a SQL statement on this (or another) warehouse."""
        if format is None:
            format = Format.ARROW_STREAM

        if disposition is None:
            disposition = Disposition.EXTERNAL_LINKS
        elif format in (Format.CSV, Format.ARROW_STREAM):
            disposition = Disposition.EXTERNAL_LINKS

        wait = WaitingConfig.check_arg(wait)

        if submit_wait is None:
            submit_wait = WaitingConfig(
                timeout=300,
                interval=1.0,
                backoff=1.8,
                max_interval=8.0,
            )
        else:
            submit_wait = WaitingConfig.check_arg(submit_wait)

        # Resolve the target warehouse (self by default)
        if warehouse_id or warehouse_name:
            instance = self.service.find_warehouse(
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
            )
        else:
            instance = self

        resolved_wh_id = warehouse_id or instance.warehouse_id
        sdk_client = instance.client.workspace_client().statement_execution

        LOGGER.debug(
            "Executing SQL on warehouse %s (%s):\n%s",
            instance.warehouse_name, resolved_wh_id, statement,
        )

        started_at = time.monotonic()
        deadline = (
            started_at + submit_wait.timeout
            if submit_wait and submit_wait.timeout
            else None
        )

        response = None
        iteration = 0

        while True:
            try:
                response = sdk_client.execute_statement(
                    statement=statement,
                    warehouse_id=resolved_wh_id,
                    byte_limit=byte_limit,
                    disposition=disposition,
                    format=format,
                    on_wait_timeout=on_wait_timeout,
                    parameters=parameters,
                    row_limit=row_limit,
                    wait_timeout=wait_timeout,
                    catalog=catalog_name,
                    schema=schema_name,
                )
                break

            except DeadlineExceeded:
                remaining = (
                    None if deadline is None
                    else (deadline - time.monotonic())
                )

                if remaining is not None and remaining <= 0:
                    LOGGER.error(
                        "Submit deadline exceeded for warehouse %s (%s) "
                        "after %.2fs and %d retries",
                        instance.warehouse_name, resolved_wh_id,
                        time.monotonic() - started_at, iteration,
                    )
                    raise

                sleep_for = _jitter_sleep_seconds(
                    submit_wait, iteration=iteration, remaining=remaining,
                )

                LOGGER.warning(
                    "Warehouse %s (%s) is busy; execute submit hit DeadlineExceeded. "
                    "Retrying in %.2fs (attempt=%d)",
                    instance.warehouse_name, resolved_wh_id,
                    sleep_for, iteration + 1,
                )

                if sleep_for > 0:
                    time.sleep(sleep_for)

                iteration += 1

        execution = StatementResult(
            client=self.client,
            warehouse_id=resolved_wh_id,
            statement_id=response.statement_id,
            disposition=disposition,
            _response=response,
        )

        LOGGER.info(
            "Executed SQL statement_id=%s on warehouse %s (%s)",
            execution.statement_id, instance.warehouse_name, resolved_wh_id,
        )

        return execution.wait(wait=wait, raise_error=raise_error)

