"""
Databricks SQL Warehouse resource — individual warehouse lifecycle management.

This module exposes the :class:`SQLWarehouse` resource, a lightweight
wrapper around a single Databricks SQL Warehouse that provides:

- property-based status checks (running, pending, serverless)
- lifecycle helpers: start, stop, delete, wait_for_status
- configuration updates and permission management
- SQL statement execution via :meth:`execute` / :meth:`send`

``_submit_statement`` reads typed warehouse fields off
:class:`WarehousePreparedStatement` and submits via the SDK.  If the SDK
call fails the exception propagates — there is no submission-level
retry layer.  Result-level retry for queries that ran and failed is
still available via :meth:`DatabricksSQL.retry`.

Collection-level operations (listing, finding, creating warehouses) live
in the companion :mod:`~yggdrasil.databricks.sql.service` module.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, ClassVar, Iterable, List, Mapping, Optional, Union, TYPE_CHECKING

from yggdrasil.http_.exceptions import (
    exceptions as _http_exceptions,
    disable_warnings,
)
from yggdrasil.http_.retry import Retry
from yggdrasil.http_.timeout import Timeout

from databricks.sdk.errors import InternalError, DeadlineExceeded, TemporarilyUnavailable
from databricks.sdk.service.sql import (
    Disposition,
    EndpointInfo,
    Format,
    State,
    WarehouseAccessControlRequest,
)

from yggdrasil.concurrent.threading import Job
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
    DatabricksSQL,
    ExternalStatementData,
    WarehousePreparedStatement,
    WarehouseStatementBatch,
)
from ..client import DatabricksResource
from ..fs import VolumePath

if TYPE_CHECKING:
    from yggdrasil.http_ import HTTPSession

    from .service import Warehouses

__all__ = [
    "SQLWarehouse",
]

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQLWarehouse
# ---------------------------------------------------------------------------


class SQLWarehouse(DatabricksResource):
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

    Inherits :class:`Singleton` (``_SINGLETON_TTL = None``) so two
    callers asking for the same warehouse under the same service
    collapse to one instance — same cached ``EndpointInfo``, same
    external-link connection pool — across every ``client.warehouses[id]``
    or ``Warehouses.find_warehouse(...)`` lookup.
    """

    # Process-lifetime caching — warehouses are heavyweight (cached
    # ``EndpointInfo``, HTTP pool); we want the same id under the
    # same service to share state.
    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        service: "Warehouses | None" = None,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        # ``service`` is a hashable :class:`DatabricksService` (carries
        # its own client identity); pin on the (id ∨ name) the caller
        # provided. ``details=`` is opaque metadata, never part of the
        # identity.
        return (cls, service, warehouse_id, warehouse_name)

    def __init__(
        self,
        service: "Warehouses | None" = None,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        *,
        details: Optional[EndpointInfo] = None,
        singleton_ttl: Any = ...,
    ):
        # ``singleton_ttl`` is consumed by :meth:`Singleton.__new__`;
        # accept it here so the constructor signature stays open.
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return

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
        self._initialized = True

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------

    # Live thread/socket handles that must NOT cross a pickle boundary.
    # The lock is process-local; the HTTP pool owns sockets bound to the
    # source process. Both are rebuilt fresh from the un-pickled state.
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({
        "_external_link_pool_lock",
        "_external_link_pool_instance",
    })

    # ``Disposable`` carries its open/close bookkeeping in ``__slots__``,
    # so it isn't in ``__dict__`` and the parent ``DatabricksResource``
    # ``__getstate__`` would silently drop it. Carry it explicitly.
    _DISPOSABLE_SLOTS: ClassVar[tuple[str, ...]] = ("_acquired", "_dirty", "_depth")

    def __getstate__(self):
        state = super().__getstate__()
        for attr in self._TRANSIENT_STATE_ATTRS:
            state.pop(attr, None)
        for slot in self._DISPOSABLE_SLOTS:
            state[slot] = getattr(self, slot)
        return state

    def __setstate__(self, state):
        for slot in self._DISPOSABLE_SLOTS:
            object.__setattr__(self, slot, state.pop(slot, 0 if slot == "_depth" else False))
        super().__setstate__(state)
        # Rebuild the per-process external-link pool guard fresh; the new
        # process gets its own lock and a None pool that will lazily build
        # on the next chunk read.
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

    def __str__(self) -> str:
        return self.warehouse_id or self.warehouse_name or f"{self.__class__.__name__}(<not initialized>)"

    # ------------------------------------------------------------------
    # External-link fetch pool
    # ------------------------------------------------------------------

    def external_link_pool(self, max_workers: int = 8) -> "HTTPSession":
        """Return the cached :class:`HTTPSession` for chunk reads.

        :class:`HTTPSession` is the pool now — it owns the per-host
        socket cache, retries, redirect handling, and (with
        ``verify=False``) TLS-off support for Databricks-issued
        presigned URLs against cloud-storage hostnames whose
        certificates aren't always reachable from the workspace
        egress path.

        Built lazily on the first ``EXTERNAL_LINKS`` chunk read and
        reused across every :class:`DatabricksSQL` attached
        to this warehouse. ``max_workers`` is a sizing *hint* applied
        only on the first call; subsequent callers get the existing
        session regardless of what they pass.
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

        Currently just clears the external-link session's connections.
        Safe to call multiple times.  Existing lifecycle methods
        (:meth:`stop`, :meth:`delete`) don't call this — closing
        sockets is independent of remote warehouse state, since a
        stopped warehouse can be started again and the session would
        still serve.
        """
        pool = self._external_link_pool_instance
        if pool is not None:
            # The slot can hold a bare ``object()`` placeholder during
            # pickle / fixture round-trips (see
            # ``test_warehouse_pickle.py``), so don't blindly call
            # ``.clear_connections`` — it's an :class:`HTTPSession`
            # method, not a generic one.
            clear = getattr(pool, "clear_connections", None) or getattr(pool, "clear", None)
            if callable(clear):
                try:
                    clear()
                except Exception:
                    LOGGER.exception(
                        "Failed to clear external-link session for %r; continuing.",
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
                "Waiting for warehouse %r to leave pending state (timeout=%.0fs)",
                self, wait.timeout,
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
                "Warehouse %r ready after %.1fs", self, time.time() - start,
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

            LOGGER.info("Started warehouse %r", self)
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
            LOGGER.debug("No update specs provided for warehouse %r — skipping", self)
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
            LOGGER.debug("Updating warehouse %r with %s", self, update_details)
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
            LOGGER.debug("Warehouse %r already up-to-date — skipping update", self)

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
    # SQL execution
    # ------------------------------------------------------------------

    def send(
        self,
        statement: "WarehousePreparedStatement | str",
        start: bool = True,
    ) -> DatabricksSQL:
        """Coerce and submit a statement. Returns a :class:`DatabricksSQL`.

        ``start=False`` returns an unstarted result — useful when batching
        or when the caller wants to control the submission separately.
        """
        prepared = WarehousePreparedStatement.from_(statement)
        return self._submit_statement(prepared, start=start)

    def prepare(
        self,
        statement: "WarehousePreparedStatement | str",
    ) -> WarehousePreparedStatement:
        """Coerce *statement* to a :class:`WarehousePreparedStatement`."""
        return WarehousePreparedStatement.from_(statement)

    def execute_many(
        self,
        statements: "Iterable[str | WarehousePreparedStatement]",
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        parallel: Optional[int] = None,
    ) -> WarehouseStatementBatch:
        """Submit a collection of statements as a batch."""
        batch = WarehouseStatementBatch(
            executor=self,
            statements=statements,
            parallel=parallel or 1,
        )
        batch.wait(wait=wait, raise_error=raise_error)
        return batch

    def _submit_statement(
        self,
        statement: WarehousePreparedStatement,
        start: bool = True
    ) -> DatabricksSQL:
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

        result = DatabricksSQL(
            executor=self,
            statement=statement,
        )

        if not start:
            return result

        LOGGER.debug(
            "Executing %r:\n%s",
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
            "Executed %r",
            result
        )
        return result

    # ------------------------------------------------------------------
    # Public execute() — back-compat shim over the typed pipeline
    # ------------------------------------------------------------------

    def execute(
        self,
        statement: "str | WarehousePreparedStatement | DatabricksSQL | None" = None,
        *,
        # Routing
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        # Per-statement config
        byte_limit: int | None = None,
        disposition: Optional[Disposition] = None,
        format: Optional[Format] = None,
        parameters: Optional[Mapping[str, Any] | List[Any]] = None,
        row_limit: int | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        # External data
        external_data: Optional[Mapping[str, Any]] = None,
        external_volume_paths: Optional[Mapping[str, VolumePath]] = None,
        # Execution policy
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        retry: Optional[WaitingConfigArg] = None,
    ) -> DatabricksSQL:
        """Execute a SQL statement on this (or another) warehouse.

        Already-started results are returned with a ``wait()`` rather
        than re-submitted.

        ``warehouse_id`` / ``warehouse_name`` redirect submission to a
        different warehouse — kept for back-compat with callers that use
        one warehouse handle as a dispatcher.

        ``retry`` configures the *result-level* retry. Submission
        failures propagate directly — there's no submission-level retry
        layer.
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
        if isinstance(statement, DatabricksSQL) and statement.started:
            return statement.wait(wait=wait, raise_error=raise_error)

        # Coerce + bind. prepare() handles parameters, format/disposition
        # defaults, and the staging+merge of external_data /
        # external_volume_paths in one shot.
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

        result = self._submit_statement(prepared)
        result.wait(wait=wait, raise_error=raise_error)
        return result



def _build_external_link_pool(max_workers: int) -> "HTTPSession":
    """Build a TLS-off :class:`HTTPSession` for external-link fetches.

    The session is reused across every chunk read for the warehouse
    it's attached to (see :meth:`SQLWarehouse.external_link_pool`);
    pulling it onto the warehouse instance — rather than a module-level
    cache — ties the underlying connection lifecycle to the warehouse
    handle. When the warehouse is GC'd, the session drops with it, so
    a long-running process holding many warehouses doesn't accumulate
    pooled sockets the runtime can't reach.

    Sizing is keyed off ``max_workers`` so a warehouse running with
    higher fetch parallelism gets a correspondingly larger
    ``pool_maxsize``; idle slots cost nothing. TLS verification is
    disabled — the presigned URLs Databricks hands out point at
    cloud-storage hostnames whose certificates aren't always reachable
    from the workspace egress path (private-link / proxy-rewritten
    endpoints, customer-managed VPCs), and the URL itself carries a
    short-lived signed token so the transport doesn't need certificate
    based authentication.
    """
    from yggdrasil.dataclasses.waiting import WaitingConfig
    from yggdrasil.http_ import HTTPSession

    disable_warnings(_http_exceptions.InsecureRequestWarning)
    # Per-warehouse, TLS-off — pin singleton_ttl=False so each warehouse
    # gets its own session (the default HTTPSession singleton key would
    # collapse two TLS-off warehouses onto the same socket cache).
    return HTTPSession(
        base_url=None,
        verify=False,
        pool_maxsize=max_workers * 2,
        waiting=WaitingConfig(timeout=60),
    )
