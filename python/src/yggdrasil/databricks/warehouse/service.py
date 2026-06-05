"""
Databricks SQL Warehouse service – collection-level management.

This module provides the :class:`Warehouses` service, a thin layer around the
Databricks SDK ``WarehousesAPI`` that handles:

- listing / finding warehouses by name or id
- creating, updating, and deleting warehouses with sensible defaults
- managing warehouse permissions
- maintaining a short-lived in-process cache to avoid redundant API calls
"""

import logging
from typing import Optional, Sequence, Union, List, Iterator, Any

from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.sql import (
    State, EndpointInfo,
    EndpointTags, EndpointTagPair, EndpointInfoWarehouseType,
    WarehouseAccessControlRequest, WarehousePermissionLevel,
)
from yggdrasil.concurrent.threading import Job
from yggdrasil.databricks.client import DatabricksService, DatabricksClient
from yggdrasil.databricks.warehouse.wh_utils import DEFAULT_ALL_PURPOSE_CLASSIC_NAME, DEFAULT_ALL_PURPOSE_SERVERLESS_NAME, \
    safeEndpointInfo, _CREATE_ARG_NAMES
from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg

from .warehouse import SQLWarehouse

__all__ = [
    "Warehouses",
    "set_cached_warehouse",
    "get_cached_warehouse",
]

LOGGER = logging.getLogger(__name__)

# host -> ExpiringDict(warehouse_name -> SQLWarehouse)
CACHE_MAP: dict[str, ExpiringDict[str, "SQLWarehouse"]] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_cached_warehouse(
    client: DatabricksClient,
    warehouse: "SQLWarehouse",
) -> None:
    host = (client.host if client else None) or "default"
    existing = CACHE_MAP.get(host)
    if existing is None:
        existing = CACHE_MAP[host] = ExpiringDict(default_ttl=3600)
    existing[warehouse.warehouse_name] = warehouse


def get_cached_warehouse(
    client: DatabricksClient,
    warehouse_name: str,
) -> Optional["SQLWarehouse"]:
    host = (client.host if client else None) or "default"
    existing = CACHE_MAP.get(host)
    return existing.get(warehouse_name) if existing else None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class Warehouses(DatabricksService):
    """
    Collection-level SQL Warehouse management service.

    Provides list / find / create / update operations over SQL warehouses in
    a Databricks workspace.  Individual warehouse lifecycle operations (start,
    stop, execute …) live on the :class:`~yggdrasil.databricks.sql.warehouse.SQLWarehouse`
    resource returned by this service.
    """

    def __iter__(self):
        yield from self.list_warehouses()

    # ------------------------------------------------------------------ #
    # Listing / finding
    # ------------------------------------------------------------------ #

    def list_warehouses(self) -> Iterator["SQLWarehouse"]:
        """Yield all SQL warehouses visible to the current principal."""
        client = self.client.workspace_client().warehouses
        for info in client.list():
            warehouse = SQLWarehouse(
                service=self,
                warehouse_id=info.id,
                warehouse_name=info.name,
                details=safeEndpointInfo(info)
            )
            yield warehouse

    def find_warehouse(
        self,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        find_default: bool = False,
        create: bool = True,
        default: Any = ...,
    ) -> Optional["SQLWarehouse"]:
        """
        Resolve a warehouse by id or name.

        Parameters
        ----------
        warehouse_id:
            Exact warehouse id to resolve (the fastest path).
        warehouse_name:
            Warehouse name to resolve.  Uses an in-process cache first,
            then falls back to listing all warehouses.
        find_default:
            When True and no id/name is given, fall back to
            :meth:`find_default`.
        create:
            When True and id/name is given, create a new warehouse.
        default:
            When ..., raise :exc:`ResourceDoesNotExist` if not found.

        Returns
        -------
        A :class:`~yggdrasil.databricks.sql.warehouse.SQLWarehouse` resource,
        or ``None`` when *raise_error* is False and the warehouse is not found.
        """
        if warehouse_id:
            LOGGER.debug("Resolving warehouse by id %s", warehouse_id)
            return SQLWarehouse(
                service=self,
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
            )

        if warehouse_name:
            cached = get_cached_warehouse(client=self.client, warehouse_name=warehouse_name)
            if cached is not None:
                return cached

            LOGGER.debug("Listing warehouses to resolve name %r", warehouse_name)
            for warehouse in self.list_warehouses():
                if warehouse.warehouse_name == warehouse_name:
                    set_cached_warehouse(client=self.client, warehouse=warehouse)
                    return warehouse

            LOGGER.warning("Warehouse %r not found in workspace %r", warehouse_name, self.client)

            if create:
                return self.create(
                    name=warehouse_name,
                    permissions=[
                        WarehouseAccessControlRequest(
                            group_name="users",
                            permission_level=WarehousePermissionLevel.CAN_MANAGE,
                        )
                    ],
                )

            if default is ...:
                raise ResourceDoesNotExist(f"Cannot find SQL warehouse {warehouse_name!r}")
            return default

        if find_default:
            return self.find_default()

        if default is ...:
            raise ResourceDoesNotExist("Cannot find SQL warehouse, no parameters given")
        return default

    def find_default(self, raise_error: bool = True) -> Optional["SQLWarehouse"]:
        """
        Resolve (or create) the workspace's default SQL warehouse.

        Resolution order:
        1. Cached classic warehouse  (starts it if stopped)
        2. Cached serverless warehouse
        3. Listed classic warehouse  (starts it if stopped)
        4. Listed serverless warehouse
        5. Creates a new serverless warehouse
        6. Fires-and-forgets creating a classic warehouse
        7. Returns the first found warehouse
        8. Raises / creates a new serverless warehouse as a last resort
        """
        classic = get_cached_warehouse(
            client=self.client,
            warehouse_name=DEFAULT_ALL_PURPOSE_CLASSIC_NAME,
        )
        if classic is not None:
            if classic.state not in {State.RUNNING, State.STARTING, State.STOPPING}:
                classic.start(wait=False, raise_error=False)
            else:
                return classic

        serverless = get_cached_warehouse(
            client=self.client,
            warehouse_name=DEFAULT_ALL_PURPOSE_SERVERLESS_NAME,
        )
        if serverless is not None:
            return serverless

        first_found = None

        for warehouse in self.list_warehouses():
            if first_found is None:
                first_found = warehouse

            if warehouse.warehouse_name == DEFAULT_ALL_PURPOSE_CLASSIC_NAME:
                classic = warehouse
                set_cached_warehouse(client=self.client, warehouse=classic)

                state = classic.state
                if state == State.RUNNING:
                    return classic
                elif state != State.STARTING:
                    classic.start(wait=False, raise_error=False)

                if serverless is not None:
                    return serverless

            elif warehouse.warehouse_name == DEFAULT_ALL_PURPOSE_SERVERLESS_NAME:
                serverless = warehouse
                set_cached_warehouse(client=self.client, warehouse=serverless)

                if classic is not None:
                    return serverless

        if serverless is None:
            try:
                created = self.create(
                    name=DEFAULT_ALL_PURPOSE_SERVERLESS_NAME,
                    permissions=["users"],
                )
                set_cached_warehouse(client=self.client, warehouse=created)
                return created
            except Exception:
                pass

        if classic is None:
            Job.make(
                self.create,
                name=DEFAULT_ALL_PURPOSE_CLASSIC_NAME,
                permissions=["users"],
            ).fire_and_forget()

        if serverless is not None:
            return serverless

        if first_found is not None:
            return first_found

        if raise_error:
            raise ResourceDoesNotExist(
                "Cannot find default SQL warehouse, no parameters given"
            )

        return self.create_or_update(
            name=DEFAULT_ALL_PURPOSE_SERVERLESS_NAME,
            permissions=["users"],
        )

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def create_or_update(
        self,
        warehouse_id: str | None = None,
        name: str | None = None,
        permissions: Optional[List[Union[WarehouseAccessControlRequest, str]]] = None,
        wait: Optional[WaitingConfig] = None,
        **warehouse_specs,
    ) -> "SQLWarehouse":
        """Create a warehouse if it does not exist, otherwise update it."""
        found = self.find_warehouse(
            warehouse_id=warehouse_id,
            warehouse_name=name,
            default=None,
        )

        if found is not None:
            LOGGER.debug("Updating existing warehouse %r", found)
            return found.update(
                name=name,
                permissions=permissions,
                wait=wait,
                **warehouse_specs,
            )

        LOGGER.debug("Warehouse %r not found — creating", name)
        return self.create(
            name=name,
            permissions=permissions,
            wait=wait,
            **warehouse_specs,
        )

    def create(
        self,
        name: str | None = None,
        *,
        permissions: Optional[List[Union[WarehouseAccessControlRequest, str]]] = None,
        wait: WaitingConfigArg = None,
        **warehouse_specs,
    ) -> "SQLWarehouse":
        """Create a new SQL warehouse and return a resource handle."""
        client = self.client.workspace_client().warehouses
        wait = WaitingConfig.from_(wait)

        details = self._check_details(
            keys=_CREATE_ARG_NAMES,
            update=False,
            name=name,
            **warehouse_specs,
        )

        LOGGER.debug(
            "Creating warehouse %r (serverless=%s, cluster_size=%s)",
            details.name, details.enable_serverless_compute, details.cluster_size,
        )

        update_details = {
            k: v
            for k, v in details.as_shallow_dict().items()
            if k in _CREATE_ARG_NAMES
        }

        if wait:
            info = client.create_and_wait(
                timeout=wait.timeout_timedelta,
                **update_details,
            )
        else:
            raw = client.create(**update_details)
            update_details["id"] = raw.response.id
            info = EndpointInfo(**update_details)

        created = SQLWarehouse(
            service=self,
            warehouse_id=info.id,
            warehouse_name=info.name,
            details=info,
        )

        if permissions:
            created.update_permissions(permissions=permissions, wait=wait)

        LOGGER.info(
            "Created warehouse %r (serverless=%s)",
            created, details.enable_serverless_compute,
        )

        return created

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _check_details(
        self,
        keys: Sequence[str],
        update: bool,
        details: Optional[EndpointInfo] = None,
        **warehouse_specs,
    ) -> EndpointInfo:
        """Normalise and fill defaults for warehouse creation / update specs."""
        if details is None:
            details = EndpointInfo(**{
                k: v
                for k, v in warehouse_specs.items()
                if k in keys
            })
        else:
            kwargs = {**details.as_shallow_dict(), **warehouse_specs}
            details = EndpointInfo(**{k: v for k, v in kwargs.items() if k in keys})

        if details.cluster_size is None:
            details.cluster_size = "Small"

        if details.warehouse_type is None:
            details.warehouse_type = EndpointInfoWarehouseType.PRO

        if not details.name:
            if details.enable_serverless_compute:
                details.name = DEFAULT_ALL_PURPOSE_SERVERLESS_NAME
            else:
                details.name = DEFAULT_ALL_PURPOSE_CLASSIC_NAME

        if details.enable_serverless_compute is None:
            if details.name == DEFAULT_ALL_PURPOSE_CLASSIC_NAME:
                details.enable_serverless_compute = False
            elif details.name == DEFAULT_ALL_PURPOSE_SERVERLESS_NAME:
                details.enable_serverless_compute = True
            else:
                details.enable_serverless_compute = "verless" in details.name.lower()

        if details.enable_serverless_compute:
            details.warehouse_type = EndpointInfoWarehouseType.PRO

        if not details.auto_stop_mins:
            if not details.enable_serverless_compute:
                details.auto_stop_mins = 30

        default_tags = self.client.default_tags(update=update)

        if details.tags is None:
            details.tags = EndpointTags(custom_tags=[
                EndpointTagPair(key=k, value=v)
                for k, v in default_tags.items()
            ])
        else:
            tags = {pair.key: pair.value for pair in details.tags.custom_tags}
            tags.update(default_tags)
            details.tags = EndpointTags(custom_tags=[
                EndpointTagPair(key=k, value=v)
                for k, v in tags.items()
            ])

        if not details.max_num_clusters:
            details.max_num_clusters = 8

        return details

    @staticmethod
    def check_permission(
        permission: Union[WarehouseAccessControlRequest, str],
    ) -> WarehouseAccessControlRequest:
        """Normalise a permission string or ACL object."""
        if isinstance(permission, str):
            if permission == "users":
                return WarehouseAccessControlRequest(
                    group_name=permission,
                    permission_level=WarehousePermissionLevel.CAN_USE,
                )
            elif "@" in permission:
                return WarehouseAccessControlRequest(
                    user_name=permission,
                    permission_level=WarehousePermissionLevel.CAN_USE,
                )
            else:
                return WarehouseAccessControlRequest(
                    group_name=permission,
                    permission_level=WarehousePermissionLevel.CAN_MANAGE,
                )
        return permission

