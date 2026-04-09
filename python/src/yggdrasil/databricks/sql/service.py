"""
Databricks SQL Warehouse service – collection-level management.

This module provides the :class:`Warehouses` service, a thin layer around the
Databricks SDK ``WarehousesAPI`` that handles:

- listing / finding warehouses by name or id
- creating, updating and deleting warehouses with sensible defaults
- managing warehouse permissions
- maintaining a short-lived in-process cache to avoid redundant API calls
"""

import dataclasses as dc
import inspect
import logging
import random
from typing import Optional, Sequence, Any, Type, TypeVar, Union, List, Iterator, TYPE_CHECKING

from databricks.sdk import WarehousesAPI
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.sql import (
    State, EndpointInfo,
    EndpointTags, EndpointTagPair, EndpointInfoWarehouseType,
    GetWarehouseResponse, GetWarehouseResponseWarehouseType,
    WarehouseAccessControlRequest, WarehousePermissionLevel,
)

from yggdrasil.concurrent.threading import Job
from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from ..client import DatabricksService, DatabricksClient

if TYPE_CHECKING:
    from .warehouse import SQLWarehouse

__all__ = [
    "Warehouses",
    "DEFAULT_ALL_PURPOSE_SERVERLESS_NAME",
    "DEFAULT_ALL_PURPOSE_CLASSIC_NAME",
    "safeGetWarehouseResponse",
    "safeEndpointInfo",
    "set_cached_warehouse",
    "get_cached_warehouse",
]

LOGGER = logging.getLogger(__name__)

_CREATE_ARG_NAMES: set[str] = {_ for _ in inspect.signature(WarehousesAPI.create).parameters.keys()}
_EDIT_ARG_NAMES: set[str] = {_ for _ in inspect.signature(WarehousesAPI.edit).parameters.keys()}

# host -> ExpiringDict(warehouse_name -> SQLWarehouse)
CACHE_MAP: dict[str, ExpiringDict[str, "SQLWarehouse"]] = {}

T = TypeVar("T")

DEFAULT_ALL_PURPOSE_CLASSIC_NAME = "Yggdrasil All Purpose"
DEFAULT_ALL_PURPOSE_SERVERLESS_NAME = DEFAULT_ALL_PURPOSE_CLASSIC_NAME + " Serverless"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safeGetWarehouseResponse(src: Union[GetWarehouseResponse, EndpointInfo]) -> GetWarehouseResponse:
    if isinstance(src, GetWarehouseResponse):
        return src

    payload = _copy_common_fields(src, GetWarehouseResponse, skip={"warehouse_type"})
    payload["warehouse_type"] = _safe_map_enum(
        GetWarehouseResponseWarehouseType,
        getattr(src, "warehouse_type", None),
    )
    return GetWarehouseResponse(**payload)


def safeEndpointInfo(src: Union[GetWarehouseResponse, EndpointInfo]) -> EndpointInfo:
    if isinstance(src, EndpointInfo):
        return src

    payload = _copy_common_fields(src, EndpointInfo, skip={"warehouse_type"})
    payload["warehouse_type"] = _safe_map_enum(
        EndpointInfoWarehouseType,
        getattr(src, "warehouse_type", None),
    )
    return EndpointInfo(**payload)


def set_cached_warehouse(
    client: DatabricksClient,
    warehouse: "SQLWarehouse",
) -> None:
    host = client.base_url.to_string()
    existing = CACHE_MAP.get(host)
    if existing is None:
        existing = CACHE_MAP[host] = ExpiringDict(default_ttl=3600)
    existing[warehouse.warehouse_name] = warehouse


def get_cached_warehouse(
    client: DatabricksClient,
    warehouse_name: str,
) -> Optional["SQLWarehouse"]:
    host = client.base_url.to_string()
    existing = CACHE_MAP.get(host)
    return existing.get(warehouse_name) if existing else None


def _safe_map_enum(dst_enum: Type[T], src_val: Any) -> Optional[T]:
    """Best-effort enum mapping between two SDK enum types."""
    if src_val is None:
        return None
    if isinstance(src_val, dst_enum):
        return src_val
    try:
        return dst_enum(src_val)  # type: ignore[misc]
    except Exception:
        pass
    try:
        return dst_enum(src_val.value)  # type: ignore[misc]
    except Exception:
        pass
    try:
        return dst_enum[src_val.name]  # type: ignore[index]
    except Exception:
        pass
    try:
        return dst_enum(str(src_val))  # type: ignore[misc]
    except Exception:
        return None


def _copy_common_fields(src: Any, dst_cls: Type[T], *, skip: set[str] = frozenset()) -> dict:
    dst_field_names = {f.name for f in dc.fields(dst_cls)}
    return {
        name: getattr(src, name, None)
        for name in dst_field_names
        if name not in skip
    }


def _jitter_sleep_seconds(
    wait: WaitingConfig,
    *,
    iteration: int,
    remaining: Optional[float],
) -> float:
    """Exponential backoff with full jitter, capped by remaining deadline."""
    base = float(wait.interval or 1.0)
    backoff = max(float(wait.backoff or 1.0), 1.0)
    delay = base * (backoff ** iteration)
    if wait.max_interval:
        delay = min(delay, float(wait.max_interval))
    delay = random.uniform(0.0, delay)
    if remaining is not None:
        delay = min(delay, max(0.0, remaining))
    return max(0.0, delay)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

@dc.dataclass(frozen=True)
class Warehouses(DatabricksService):
    """
    Collection-level SQL Warehouse management service.

    Provides list / find / create / update operations over SQL warehouses in
    a Databricks workspace.  Individual warehouse lifecycle operations (start,
    stop, execute …) live on the :class:`~yggdrasil.databricks.sql.warehouse.SQLWarehouse`
    resource returned by this service.
    """

    @classmethod
    def service_name(cls) -> str:
        return "warehouses"

    # ------------------------------------------------------------------ #
    # Listing / finding
    # ------------------------------------------------------------------ #

    def list_warehouses(self) -> Iterator["SQLWarehouse"]:
        """Yield all SQL warehouses visible to the current principal."""
        from .warehouse import SQLWarehouse

        client = self.client.workspace_client().warehouses
        for info in client.list():
            warehouse = SQLWarehouse(
                service=self,
                warehouse_id=info.id,
                warehouse_name=info.name,
            )
            object.__setattr__(warehouse, "_details", safeEndpointInfo(info))
            yield warehouse

    def find_warehouse(
        self,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
        find_default: bool = False,
        raise_error: bool = True,
    ) -> Optional["SQLWarehouse"]:
        """
        Resolve a warehouse by id or name.

        Parameters
        ----------
        warehouse_id:
            Exact warehouse id to resolve (fastest path).
        warehouse_name:
            Warehouse name to resolve.  Uses an in-process cache first,
            then falls back to listing all warehouses.
        find_default:
            When True and no id/name is given, fall back to
            :meth:`find_default`.
        raise_error:
            When True, raise :exc:`ResourceDoesNotExist` if not found.

        Returns
        -------
        A :class:`~yggdrasil.databricks.sql.warehouse.SQLWarehouse` resource,
        or ``None`` when *raise_error* is False and the warehouse is not found.
        """
        from .warehouse import SQLWarehouse

        if warehouse_id:
            LOGGER.debug("find_warehouse: resolving by id=%s", warehouse_id)
            return SQLWarehouse(
                service=self,
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
            )

        if warehouse_name:
            cached = get_cached_warehouse(client=self.client, warehouse_name=warehouse_name)
            if cached is not None:
                LOGGER.debug(
                    "find_warehouse: cache hit name=%r id=%s",
                    warehouse_name, cached.warehouse_id,
                )
                return cached

            LOGGER.debug("find_warehouse: listing warehouses to resolve name=%r", warehouse_name)
            for warehouse in self.list_warehouses():
                if warehouse.warehouse_name == warehouse_name:
                    set_cached_warehouse(client=self.client, warehouse=warehouse)
                    return warehouse

            LOGGER.warning("find_warehouse: warehouse %r not found", warehouse_name)
            if raise_error:
                raise ResourceDoesNotExist(f"Cannot find SQL warehouse {warehouse_name!r}")
            return None

        if find_default:
            return self.find_default(raise_error=raise_error)

        if raise_error:
            raise ResourceDoesNotExist("Cannot find SQL warehouse, no parameters given")
        return None

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

                if classic._details.state == State.RUNNING:
                    return classic
                elif classic._details.state != State.STARTING:
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
        warehouse_id: Optional[str] = None,
        name: Optional[str] = None,
        permissions: Optional[List[Union[WarehouseAccessControlRequest, str]]] = None,
        wait: Optional[WaitingConfig] = None,
        **warehouse_specs,
    ) -> "SQLWarehouse":
        """Create a warehouse if it does not exist, otherwise update it."""
        found = self.find_warehouse(
            warehouse_id=warehouse_id,
            warehouse_name=name,
            raise_error=False,
        )

        if found is not None:
            LOGGER.debug(
                "create_or_update: updating existing warehouse %r (%s)",
                name, found.warehouse_id,
            )
            return found.update(
                name=name,
                permissions=permissions,
                wait=wait,
                **warehouse_specs,
            )

        LOGGER.debug("create_or_update: warehouse %r not found — creating", name)
        return self.create(
            name=name,
            permissions=permissions,
            wait=wait,
            **warehouse_specs,
        )

    def create(
        self,
        name: Optional[str] = None,
        *,
        permissions: Optional[List[Union[WarehouseAccessControlRequest, str]]] = None,
        wait: WaitingConfigArg = None,
        **warehouse_specs,
    ) -> "SQLWarehouse":
        """Create a new SQL warehouse and return a resource handle."""
        from .warehouse import SQLWarehouse

        client = self.client.workspace_client().warehouses
        wait = WaitingConfig.check_arg(wait)

        details = self._check_details(
            keys=_CREATE_ARG_NAMES,
            update=False,
            name=name,
            **warehouse_specs,
        )

        LOGGER.debug(
            "Creating warehouse name=%r serverless=%s cluster_size=%s",
            details.name, details.enable_serverless_compute, details.cluster_size,
        )

        update_details = {
            k: v
            for k, v in details.as_shallow_dict().items()
            if k in _CREATE_ARG_NAMES
        }

        if wait or permissions:
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
            _details=info,
        )

        if permissions:
            created.update_permissions(permissions=permissions, wait=wait)

        LOGGER.info(
            "Created warehouse %r (%s) serverless=%s",
            created.warehouse_name, created.warehouse_id,
            details.enable_serverless_compute,
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
            if details.name == DEFAULT_ALL_PURPOSE_CLASSIC_NAME:
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

