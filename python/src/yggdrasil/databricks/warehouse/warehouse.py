"""
Databricks SQL Warehouse resource — individual warehouse lifecycle management.

This module exposes the :class:`SQLWarehouse` resource, a lightweight
wrapper around a single Databricks SQL Warehouse that provides:

- property-based status checks (running, pending, serverless)
- lifecycle helpers: start, stop, delete, wait_for_status
- configuration updates and permission management
- SQL statement execution

Executor contract
-----------------
:class:`SQLWarehouse` is a concrete :class:`StatementExecutor` —
``_submit_statement`` reads typed warehouse fields off
:class:`WarehousePreparedStatement` and submits via the SDK.  If the SDK
call fails (``DeadlineExceeded`` on a cold/busy warehouse, transport
errors, etc.) the exception propagates — there is no submission-level
retry layer.

Execution policy travels through :class:`DatabricksExecutionOptions`
(``wait``, ``raise_error``, ``parallel`` from the base, plus
``external_data``, ``external_volume_paths``).  The base
:meth:`StatementExecutor._execute` hook handles submit + wait/raise; this
subclass overrides it to stage external data first and merge volume
paths onto the statement.

Result-level retry
------------------
:meth:`StatementResult.retry` is opt-in by setting ``statement.retry``
to a :class:`WaitingConfig`.  After a *terminal-failure* result
(statement_id was issued, query ran, query failed), it drives
:meth:`StatementResult.start` ``reset=True`` per attempt with backoff
driven by the WaitingConfig.  Used for genuinely flaky queries, not for
warehouse availability — that's the caller's responsibility now that
submission-level retry is gone.

Collection-level operations (listing, finding, creating warehouses) live
in the companion :mod:`~yggdrasil.databricks.sql.service` module.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, ClassVar, List, Mapping, Optional, Union, TYPE_CHECKING

import urllib3
from databricks.sdk.errors import InternalError, DeadlineExceeded, TemporarilyUnavailable
from databricks.sdk.service.sql import (
    Disposition,
    EndpointInfo,
    Format,
    State,
    WarehouseAccessControlRequest,
)

from yggdrasil.concurrent.threading import Job
from yggdrasil.data.executor import ExecutionOptions, StatementExecutor
from yggdrasil.databricks.warehouse.wh_utils import (
    _EDIT_ARG_NAMES,
    indexed_name_parts,
    name_at_index,
    safeEndpointInfo,
    serverless_sibling_spec,
)
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.pyutils.equality import dicts_equal
from .statement import (
    WarehousePreparedStatement,
    WarehouseStatementResult,
    WarehouseStatementBatch,
)
from ..client import DatabricksResource
from ..fs import VolumePath

if TYPE_CHECKING:
    from .service import Warehouses

__all__ = [
    "SQLWarehouse",
    "DatabricksExecutionOptions",
]

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DatabricksExecutionOptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatabricksExecutionOptions(ExecutionOptions):
    """Databricks-specific :class:`ExecutionOptions`.

    Adds two knobs on top of the base contract:

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


# ---------------------------------------------------------------------------
# SQLWarehouse
# ---------------------------------------------------------------------------


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

    def __init__(
        self,
        service: "Warehouses | None" = None,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        *,
        details: Optional[EndpointInfo] = None,
    ):
        if service is None:
            from .service import Warehouses
            service = Warehouses.current()

        super().__init__(service=service)
        self.service = service
        self.warehouse_id = warehouse_id
        self.warehouse_name = warehouse_name
        self._details = details

        if self.warehouse_name and not self.warehouse_id:
            found = self.service.find_warehouse(warehouse_name=self.warehouse_name)
            self.warehouse_id = found.warehouse_id
            self.warehouse_name = found.warehouse_name
            self._details = found._details
        elif self.warehouse_id and not self.warehouse_name:
            found = self.service.find_warehouse(warehouse_id=self.warehouse_id)
            self.warehouse_id = found.warehouse_id
            self.warehouse_name = found.warehouse_name
            self._details = found._details

        # Existing init body unchanged.
        self._external_link_pool_lock = threading.Lock()
        self._external_link_pool_instance = None

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
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.explore_url.to_string()!r})"

    def __str__(self) -> str:
        return self.warehouse_id or self.warehouse_name or f"{self.__class__.__name__}(<not initialized>)"

    # ------------------------------------------------------------------
    # External-link fetch pool
    # ------------------------------------------------------------------

    def external_link_pool(self, max_workers: int = 8) -> "urllib3.PoolManager":
        """Return the cached :class:`urllib3.PoolManager` for chunk reads.

        Built lazily on the first ``EXTERNAL_LINKS`` chunk read and
        reused across every :class:`WarehouseStatementResult` attached
        to this warehouse.  Tying the pool to the warehouse instance
        means a long-running process can dispose of the pool (and its
        sockets) by dropping the warehouse handle — no module-level
        cache the runtime can never reach.

        ``max_workers`` is a sizing *hint* applied only on the first
        call; subsequent callers get the existing pool regardless of
        what they pass.  Per-warehouse pool sizing is a warehouse-level
        property, not a per-call one.  If you need a differently-sized
        pool, that's a separate warehouse handle.
        """
        pool = self._external_link_pool_instance
        if pool is not None:
            return pool
        with self._external_link_pool_lock:
            # Double-check after acquiring the lock — another thread
            # may have built it while we were waiting.
            if self._external_link_pool_instance is None:
                self._external_link_pool_instance = _build_external_link_pool(max_workers)
            return self._external_link_pool_instance

    def _release(self, committed: bool = False) -> None:
        """Release per-warehouse resources.

        Currently just clears the external-link pool's connections.
        Safe to call multiple times.  Existing lifecycle methods
        (:meth:`stop`, :meth:`delete`) don't call this — closing
        sockets is independent of remote warehouse state, since a
        stopped warehouse can be started again and the pool would
        still serve.
        """
        pool = self._external_link_pool_instance
        if pool is not None:
            try:
                pool.clear()
            except Exception:
                LOGGER.exception(
                    "Failed to clear external-link pool for %r; continuing.",
                    self,
                )
            self._external_link_pool_instance = None
        super()._release(committed=committed)

    def __del__(self) -> None:
        # Best-effort cleanup on GC.  `__del__` runs in an uncertain
        # interpreter-shutdown context, so we swallow everything — the
        # OS will reclaim sockets anyway, this is just to be tidy.
        try:
            self.close(force=True)
        except Exception:
            pass

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
        """Return the current warehouse state.

        Refreshes the cached :attr:`details` on every access — same
        semantics as :attr:`Cluster.state`. When you only need the
        most-recent-known state without paying for a round-trip, use
        :attr:`is_running` / :attr:`is_pending` instead; they read from
        the cache and the lifecycle methods refresh it for you.
        """
        self.refresh()
        return self._details.state if self._details is not None else None

    @property
    def is_serverless(self) -> bool:
        return self.details.enable_serverless_compute

    @property
    def is_running(self) -> bool:
        """``True`` if the *cached* details say RUNNING.

        Lifecycle helpers (:meth:`start`, :meth:`stop`,
        :meth:`wait_for_status`) call :meth:`refresh` before consulting
        this predicate, so they still see fresh state. External callers
        who polled ``is_running`` in a hot loop were paying one
        ``warehouses.get`` per check; that cost is gone — call
        :meth:`refresh` explicitly when you need to re-confirm.
        """
        details = self._details
        if details is None:
            details = self.details
        return details.state == State.RUNNING

    @property
    def is_pending(self) -> bool:
        """``True`` if the cached state is one of the transitional states.

        Cached. See :attr:`is_running` for the contract.
        """
        details = self._details
        if details is None:
            details = self.details
        return details.state in {State.DELETING, State.STARTING, State.STOPPING}

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
            # Refresh once per iteration explicitly — ``is_pending`` reads the
            # cached state, so without this the loop would spin on stale data.
            while True:
                self.refresh()
                if not self.is_pending:
                    break
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

        if not self.warehouse_id:
            return self

        # Refresh once so the cached ``is_running`` predicate reflects the
        # current remote state. Same round-trip count as the old preflight,
        # but the populated cache makes any follow-up ``is_running`` /
        # ``is_pending`` check free.
        self.refresh()
        if not self.is_running:
            LOGGER.debug("Starting warehouse %r", self)
            try:
                response = client.start(id=self.warehouse_id)
            except Exception:
                if raise_error:
                    raise
                LOGGER.warning(
                    "Failed to start warehouse %r", self
                )
                return self

            if wait:
                wait_cfg = WaitingConfig.from_(wait)
                response.result(timeout=wait_cfg.timeout_timedelta)

            LOGGER.info("Started warehouse%r", self)
        return self

    def stop(self):
        """Stop the warehouse if it is running."""
        if not self.warehouse_id:
            return self
        # Refresh once so ``is_running`` reflects the live state — see
        # ``start()`` for the rationale.
        self.refresh()
        if not self.is_running:
            return self
        LOGGER.debug("Stopping warehouse %r", self)
        client = self.client.workspace_client().warehouses
        result = client.stop(id=self.warehouse_id)
        LOGGER.info("Stopped warehouse %r", self)
        return result

    def delete(self) -> None:
        """Delete the warehouse."""
        if not self.warehouse_id:
            return
        LOGGER.debug("Deleting warehouse %r", self)
        self.client.workspace_client().warehouses.delete(id=self.warehouse_id)
        LOGGER.info("Deleted warehouse %r", self)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def get_or_create_sibling(self, index: int) -> "SQLWarehouse":
        """Return the sibling warehouse at absolute ``[index]``, creating it if missing.

        ``index`` is absolute against this warehouse's base name (suffix
        stripped).  Examples for current name ``"wh [3]"``:

        - ``get_or_create_sibling(1)`` → ``"wh"``      (unsuffixed original)
        - ``get_or_create_sibling(3)`` → ``self``      (same warehouse)
        - ``get_or_create_sibling(5)`` → ``"wh [5]"``  (find or create serverless)

        Created siblings are serverless (cloned spec via
        :func:`serverless_sibling_spec`).  ``self`` is **not** mutated.
        """
        if index < 1:
            raise ValueError(f"sibling index must be >= 1, got {index}")

        base, current_idx = indexed_name_parts(self.warehouse_name or "")
        if not base:
            raise ValueError(
                f"cannot derive sibling base from {self.warehouse_name!r}"
            )

        if index == current_idx:
            return self

        target_name = name_at_index(self.warehouse_name, index)

        # `find_warehouse(default=None)` returns None on miss; `create=False`
        # so we don't trigger the service's plain-create path (which would
        # ignore our serverless / cloned-spec intent).
        existing = self.service.find_warehouse(
            warehouse_name=target_name,
            create=False,
            default=None,
        )
        if existing is not None:
            return existing

        LOGGER.info(
            "Sibling %r not found; creating serverless sibling of %r",
            target_name, self.warehouse_name,
        )
        spec = serverless_sibling_spec(self.details, name=target_name)
        spec["permissions"] = ["users"]
        # spec contains `name` (or `warehouse_name`) plus the cloned + serverless-forced fields.
        return self.service.create(**spec)

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

        LOGGER.info("Updated warehouse %r", self)
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
        start: bool = True
    ) -> WarehouseStatementResult:
        """Stage external data and forward to the base ``_execute``.

        :class:`DatabricksExecutionOptions` carries Databricks-specific
        staging instructions; their effect is purely to mutate the
        statement before submission.  Plain :class:`ExecutionOptions` are
        accepted — they just skip the staging step.
        """
        if isinstance(options, DatabricksExecutionOptions):
            statement = self._apply_databricks_options(statement, options)
        return super()._execute(statement, options, start=start)

    def _apply_databricks_options(
        self,
        statement: WarehousePreparedStatement,
        options: DatabricksExecutionOptions,
    ) -> WarehousePreparedStatement:
        """Bake Databricks-specific options into the statement.

        - ``external_data`` is staged to fresh Parquet volumes.
        - ``external_volume_paths`` is merged onto the statement.

        Merge precedence on alias collisions (lowest → highest): existing
        statement paths < ``options.external_volume_paths`` < just-staged.
        """
        staged = WarehousePreparedStatement.check_external_data(
            options.external_data,
            client=self.client,
            catalog_name=statement.catalog_name,
            schema_name=statement.schema_name,
        )

        if staged or options.external_volume_paths:
            merged: dict[str, VolumePath] = dict(statement.external_volume_paths or {})
            if options.external_volume_paths:
                merged.update(options.external_volume_paths)
            merged.update(staged)
            statement.external_volume_paths = merged or None

        return statement

    def _submit_statement(
        self,
        statement: WarehousePreparedStatement,
        start: bool = True
    ) -> WarehouseStatementResult:
        """Submit ``statement`` via the SDK.

        Reads typed routing fields directly off the statement and hands
        them to ``execute_statement``. Any SDK exception (cold/busy
        warehouse ``DeadlineExceeded``, transport errors, validation
        errors, …) propagates unchanged — the caller decides whether to
        retry or fail.  Result-level retry for queries that ran and
        failed is still available via :meth:`StatementResult.retry`.

        Defaults applied in-line:

        - ``format`` → :attr:`Format.ARROW_STREAM`
        - ``disposition`` → :attr:`Disposition.EXTERNAL_LINKS` (forced
          for CSV/ARROW_STREAM since INLINE only supports JSON_ARRAY)
        """
        statement.warehouse_id = statement.warehouse_id or self.warehouse_id

        statement.format = statement.format or Format.ARROW_STREAM
        if statement.format in (Format.CSV, Format.ARROW_STREAM):
            statement.disposition = Disposition.EXTERNAL_LINKS
        else:
            statement.disposition = statement.disposition or Disposition.INLINE

        sdk_client = self.client.workspace_client().statement_execution

        result = WarehouseStatementResult(
            executor=self,
            statement=statement,
        )

        if not start:
            return result

        LOGGER.debug(
            "%r executing:\n%s",
            self, statement.text,
        )

        itry, start = 0, time.time()
        while True:
            try:
                response = sdk_client.execute_statement(
                    statement=statement.text,
                    warehouse_id=statement.warehouse_id,
                    byte_limit=statement.byte_limit,
                    disposition=statement.disposition,
                    format=statement.format,
                    on_wait_timeout=statement.on_wait_timeout,
                    parameters=statement.to_parameter_list(),
                    row_limit=statement.row_limit,
                    wait_timeout=statement.wait_timeout,
                    catalog=statement.catalog_name,
                    schema=statement.schema_name,
                )
                result.start_timestamp = time.time()
                break
            except (InternalError, DeadlineExceeded, TemporarilyUnavailable) as e:
                if itry > 4:
                    raise
                elif (time.time() - start) > 120:
                    raise
                LOGGER.warning(
                    "Failed to execute statement %r with exception %r",
                    statement.text, e,
                )
                itry += 1

        result.set_api_response(response)
        result.iteration = 1

        LOGGER.info(
            "%r executed %r",
            self, result,
        )
        return result

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
        raise_error: bool = True,
        retry: Optional[WaitingConfigArg] = None,
    ) -> WarehouseStatementResult:
        """Execute a SQL statement on this (or another) warehouse.

        Three ways to control execution policy:

        1. Per-call kwargs (``wait``, ``raise_error``, ``external_data``,
           ``external_volume_paths``, ``retry``) — ergonomic, the default
           API.
        2. ``options=DatabricksExecutionOptions(...)`` — when the same
           policy is reused across many calls.
        3. Both — kwargs override fields they explicitly set.

        Already-started results (those carrying a ``statement_id``) are
        returned with a ``wait()`` rather than re-submitted.

        ``warehouse_id`` / ``warehouse_name`` redirect submission to a
        different warehouse — kept for back-compat with callers that use
        one warehouse handle as a dispatcher.

        ``retry`` configures the *result-level* retry — what
        :meth:`StatementResult.retry` will do if the caller invokes it
        after a terminal failure.  Submission failures (cold/busy
        warehouse ``DeadlineExceeded``, transport errors, …) propagate
        directly — there's no submission-level retry layer.
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
                raise_error=raise_error,
                retry=retry,
            )

        # Already-started result: just wait.
        if isinstance(statement, WarehouseStatementResult) and statement.started:
            return statement.wait(wait=wait, raise_error=raise_error)

        # Coerce + bind onto a typed PreparedStatement.  prepare() handles
        # parameters, format/disposition defaults, and the staging+merge
        # of external_data / external_volume_paths in one shot.  ``retry``
        # is forwarded — None means "don't override" inside prepare().
        prepared = WarehousePreparedStatement.prepare(
            statement if statement is not None else "",
            client=self.client,
            parameters=parameters,
            external_data=external_data,
            external_volume_paths=external_volume_paths,
            catalog_name=catalog_name,
            schema_name=schema_name,
            disposition=disposition,
            format=format,
            byte_limit=byte_limit,
            row_limit=row_limit,
            retry=retry,
        )

        opts = self._build_options(
            options,
            wait=wait,
            raise_error=raise_error,
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
    ) -> ExecutionOptions:
        """Merge per-call kwargs onto a base options object."""
        if options is None:
            if wait is True and raise_error is True:
                # Cheapest path — skip the dataclass construction.
                return ExecutionOptions()
            return DatabricksExecutionOptions(
                wait=wait,
                raise_error=raise_error,
            )

        # Diff against defaults so passing the kwarg defaults doesn't
        # clobber options' fields.
        defaults = DatabricksExecutionOptions()
        overrides: dict[str, Any] = {}
        if wait != defaults.wait:
            overrides["wait"] = wait
        if raise_error != defaults.raise_error:
            overrides["raise_error"] = raise_error
        if not overrides:
            return options
        return replace(options, **overrides)



def _build_external_link_pool(max_workers: int) -> urllib3.PoolManager:
    """Build a :class:`urllib3.PoolManager` for external-link fetches.

    The pool is reused across every chunk read for the warehouse it's
    attached to (see :meth:`SQLWarehouse.external_link_pool`); pulling
    it onto the warehouse instance — rather than a module-level cache —
    ties the underlying connection lifecycle to the warehouse handle.
    When the warehouse is GC'd, the pool drops with it, so a
    long-running process holding many warehouses doesn't accumulate
    pooled sockets the runtime can't reach.

    Sizing is keyed off `max_workers` so a warehouse running with
    higher fetch parallelism gets a correspondingly larger pool;
    `maxsize` is a cap, not a floor, so idle slots cost nothing.
    Timeouts are generous because external-link payloads are Arrow
    IPC streams from cloud storage that can run hundreds of MB.
    """
    retry = urllib3.Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    return urllib3.PoolManager(
        num_pools=max(4, max_workers // 2),
        maxsize=max_workers * 2,
        retries=retry,
        timeout=urllib3.Timeout(connect=10.0, read=60.0),
        cert_reqs="CERT_REQUIRED",
    )
