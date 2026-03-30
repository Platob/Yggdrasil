import dataclasses as dc
import inspect
import logging
import time
from typing import Optional, Sequence, Any, Type, TypeVar, Union, List

from databricks.sdk import WarehousesAPI
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.sql import (
    State, EndpointInfo,
    EndpointTags, EndpointTagPair, EndpointInfoWarehouseType,
    GetWarehouseResponse, GetWarehouseResponseWarehouseType,
    Disposition, Format,
    ExecuteStatementRequestOnWaitTimeout, StatementParameterListItem,
    WarehouseAccessControlRequest, WarehousePermissionLevel
)

from yggdrasil.concurrent.threading import Job
from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.pyutils.equality import dicts_equal
from .statement_result import StatementResult
from ..client import DatabricksService, DatabricksClient

_CREATE_ARG_NAMES = {_ for _ in inspect.signature(WarehousesAPI.create).parameters.keys()}
_EDIT_ARG_NAMES = {_ for _ in inspect.signature(WarehousesAPI.edit).parameters.keys()}

__all__ = [
    "SQLWarehouse",
    "DEFAULT_ALL_PURPOSE_SERVERLESS_NAME",
    "DEFAULT_ALL_PURPOSE_CLASSIC_NAME"
]


LOGGER = logging.getLogger(__name__)
CACHE_MAP: dict[str, ExpiringDict[str, "SQLWarehouse"]] = {}
T = TypeVar("T")


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
    warehouse: "SQLWarehouse"
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
    """
    Best-effort mapping:
      - if src_val is already a dst_enum member -> return it
      - try dst_enum(src_val) for value-based enums
      - try dst_enum(src_val.value) if src is enum-like
      - try dst_enum[src_val.name] if src is enum-like
      - try dst_enum[str(src_val)] as a last resort
    If nothing works -> None
    """
    if src_val is None:
        return None

    # already the right type
    if isinstance(src_val, dst_enum):
        return src_val

    # direct constructor (works if src_val is value-compatible)
    try:
        return dst_enum(src_val)  # type: ignore[misc]
    except Exception:
        pass

    # enum-like: .value
    try:
        return dst_enum(src_val.value)  # type: ignore[misc]
    except Exception:
        pass

    # enum-like: .name
    try:
        return dst_enum[src_val.name]  # type: ignore[index]
    except Exception:
        pass

    # string fallback
    try:
        return dst_enum(str(src_val))  # type: ignore[misc]
    except Exception:
        return None


def _copy_common_fields(src: Any, dst_cls: Type[T], *, skip: set[str] = frozenset()) -> dict:
    dst_field_names = {f.name for f in dc.fields(dst_cls)}
    payload = {}
    for name in dst_field_names:
        if name in skip:
            continue
        payload[name] = getattr(src, name, None)
    return payload


@dc.dataclass(frozen=True)
class SQLWarehouse(DatabricksService):
    warehouse_id: Optional[str] = None
    warehouse_name: Optional[str] = None

    _details: Optional[EndpointInfo] = dc.field(default=None, repr=False, hash=False, compare=False)

    def __call__(
        self,
        *,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None
    ):
        if not warehouse_id and not warehouse_name:
            return self

        if warehouse_id == self.warehouse_id or warehouse_name == self.warehouse_name:
            return self

        return self.find_warehouse(
            warehouse_id=warehouse_id,
            warehouse_name=warehouse_name,
            raise_error=True
        )

    def __post_init__(self):
        super().__post_init__()

        if self.warehouse_name and not self.warehouse_id:
            found = self.find_warehouse(warehouse_name=self.warehouse_name)

            object.__setattr__(self, "warehouse_id", found.warehouse_id)
            object.__setattr__(self, "warehouse_name", found.warehouse_name)
            object.__setattr__(self, "_details", found._details)

    @classmethod
    def service_name(cls):
        return "sqlwh"

    @property
    def details(self) -> EndpointInfo:
        if self._details is None:
            self.refresh()
        return self._details

    def latest_details(self):
        return self.client.workspace_client().warehouses.get(id=self.warehouse_id)

    def refresh(self):
        checked = safeEndpointInfo(self.latest_details())
        object.__setattr__(self, "_details", checked)
        return self

    @property
    def state(self):
        return self.latest_details().state

    @property
    def is_serverless(self):
        return self.details.enable_serverless_compute

    @property
    def is_running(self):
        return self.state == State.RUNNING

    @property
    def is_pending(self):
        return self.state in {
            State.DELETING, State.STARTING, State.STOPPING
        }

    def wait_for_status(
        self,
        wait: WaitingConfigArg = None
    ):
        """
        Polls until not pending, using wait.sleep(iteration, start).

        WaitingConfig:
          - timeout: total wall-clock seconds (0 => no timeout)
          - interval: base sleep seconds (0 => busy/no-sleep)
          - backoff: exponential factor (>= 1)
          - max_interval: cap for sleep seconds (0 => no cap)

        Returns self.
        """
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
        raise_error: bool = True
    ):
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

    def find_warehouse(
        self,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
        find_default: bool = False,
        raise_error: bool = True,
    ):
        if warehouse_id:
            if warehouse_id == self.warehouse_id:
                LOGGER.debug("find_warehouse: cache hit id=%s", warehouse_id)
                return self

            LOGGER.debug("find_warehouse: resolving by id=%s", warehouse_id)
            return SQLWarehouse(
                client=self.client,
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name
            )
        elif warehouse_name:
            warehouse_name = warehouse_name or self.warehouse_name

            cached = get_cached_warehouse(
                client=self.client,
                warehouse_name=warehouse_name
            )

            if cached is not None:
                LOGGER.debug("find_warehouse: cache hit name=%r id=%s", warehouse_name, cached.warehouse_id)
                return cached

            LOGGER.debug("find_warehouse: listing warehouses to resolve name=%r", warehouse_name)
            for warehouse in self.list_warehouses():
                if warehouse.warehouse_name == warehouse_name:
                    set_cached_warehouse(
                        client=self.client,
                        warehouse=warehouse
                    )

                    if warehouse_name == self.warehouse_name:
                        object.__setattr__(self, "warehouse_id", warehouse.warehouse_id)
                        object.__setattr__(self, "_details", warehouse._details)
                        return self

                    return warehouse

            LOGGER.warning("find_warehouse: warehouse %r not found", warehouse_name)
            if raise_error:
                raise ResourceDoesNotExist(
                    f"Cannot find SQL warehouse {warehouse_name!r}"
                )

            return None
        elif self.warehouse_id:
            return self
        elif self.warehouse_name:
            return self.find_warehouse(
                warehouse_name=self.warehouse_name,
                raise_error=raise_error,
                find_default=False
            )

        if find_default:
            return self.find_default(raise_error=raise_error)
        if raise_error:
            raise ResourceDoesNotExist(
                f"Cannot find SQL warehouse, no parameters given"
            )
        return None

    def find_default(self, raise_error: bool = True):
        classic = get_cached_warehouse(
            client=self.client,
            warehouse_name=DEFAULT_ALL_PURPOSE_CLASSIC_NAME
        )

        if classic is not None:
            if classic.state not in {State.RUNNING, State.STARTING, State.STOPPING}:
                classic.start(wait=False, raise_error=False)
            else:
                return classic

        serverless = get_cached_warehouse(
            client=self.client,
            warehouse_name=DEFAULT_ALL_PURPOSE_SERVERLESS_NAME
        )

        if serverless is not None:
            return serverless

        first_found = None

        for warehouse in self.list_warehouses():
            if first_found is None:
                first_found = warehouse

            if warehouse.warehouse_name == DEFAULT_ALL_PURPOSE_CLASSIC_NAME:
                classic = warehouse

                set_cached_warehouse(
                    client=self.client,
                    warehouse=classic
                )

                if classic._details.state == State.RUNNING:
                    return classic
                elif classic._details.state != State.STARTING:
                    classic.start(wait=False, raise_error=False)

                if serverless is not None:
                    return serverless
            elif warehouse.warehouse_name == DEFAULT_ALL_PURPOSE_SERVERLESS_NAME:
                serverless = warehouse

                set_cached_warehouse(
                    client=self.client,
                    warehouse=serverless
                )

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
            except:
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
                f"Cannot find default SQL warehouse, no parameters given"
            )

        return self.create_or_update(
            name=self.warehouse_name or DEFAULT_ALL_PURPOSE_SERVERLESS_NAME,
            permissions=["users"]
        )

    def list_warehouses(self):
        client = self.client.workspace_client().warehouses

        for info in client.list():
            warehouse = SQLWarehouse(
                client=self.client,
                warehouse_id=info.id,
                warehouse_name=info.name,
            )

            object.__setattr__(warehouse, "_details", safeEndpointInfo(info))

            yield warehouse

    def _check_details(
        self,
        keys: Sequence[str],
        update: bool,
        details: Optional[EndpointInfo] = None,
        **warehouse_specs
    ):
        if details is None:
            details = EndpointInfo(**{
                k: v
                for k, v in warehouse_specs.items()
                if k in keys
            })
        else:
            kwargs = {
                **details.as_shallow_dict(),
                **warehouse_specs
            }

            details = EndpointInfo(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in keys
                },
            )

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
            tags = {
                pair.key: pair.value
                for pair in details.tags.custom_tags
            }

            tags.update(default_tags)

        if details.tags is not None and not isinstance(details.tags, EndpointTags):
            details.tags = EndpointTags(custom_tags=[
                EndpointTagPair(key=k, value=v)
                for k, v in default_tags.items()
            ])

        if not details.max_num_clusters:
            details.max_num_clusters = 8

        return details

    def create_or_update(
        self,
        warehouse_id: Optional[str] = None,
        name: Optional[str] = None,
        permissions: Optional[List[WarehouseAccessControlRequest | str]] = None,
        wait: Optional[WaitingConfig] = None,
        **warehouse_specs
    ):
        name = name or self.warehouse_name
        found = self.find_warehouse(
            warehouse_id=warehouse_id,
            warehouse_name=name,
            raise_error=False,
        )

        if found is not None:
            LOGGER.debug("create_or_update: updating existing warehouse %r (%s)", name, found.warehouse_id)
            return found.update(
                name=name,
                permissions=permissions,
                wait=wait,
                **warehouse_specs
            )

        LOGGER.debug("create_or_update: warehouse %r not found — creating", name)
        return self.create(
            name=name,
            permissions=permissions,
            wait=wait,
            **warehouse_specs
        )

    def create(
        self,
        name: Optional[str] = None,
        *,
        permissions: Optional[List[WarehouseAccessControlRequest | str]] = None,
        wait: WaitingConfigArg = None,
        **warehouse_specs
    ):
        client = self.client.workspace_client().warehouses
        name = name or self.warehouse_name
        wait = WaitingConfig.check_arg(wait)

        details = self._check_details(
            keys=_CREATE_ARG_NAMES,
            update=False,
            name=name,
            **warehouse_specs
        )

        LOGGER.debug(
            "Creating warehouse name=%r serverless=%s cluster_size=%s",
            details.name,
            details.enable_serverless_compute,
            details.cluster_size,
        )

        update_details = {
            k: v
            for k, v in details.as_shallow_dict().items()
            if k in _CREATE_ARG_NAMES
        }

        if wait or permissions:
            info = client.create_and_wait(
                timeout=wait.timeout_timedelta,
                **update_details
            )
        else:
            info = client.create(**update_details)

            update_details["id"] = info.response.id

            info = EndpointInfo(**update_details)

        self.update_permissions(permissions=permissions, wait=wait)

        created = SQLWarehouse(
            client=self.client,
            warehouse_id=info.id,
            warehouse_name=info.name,
            _details=info
        )

        LOGGER.info(
            "Created warehouse %r (%s) serverless=%s",
            created.warehouse_name, created.warehouse_id,
            details.enable_serverless_compute,
        )

        return created

    def update(
        self,
        wait: WaitingConfigArg = None,
        permissions: Optional[List[WarehouseAccessControlRequest | str]] = None,
        **warehouse_specs
    ):
        if not warehouse_specs:
            LOGGER.debug("update: no specs provided for %s — skipping", self.warehouse_name)
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
                self._check_details(
                    details=self.details,
                    update=True,
                    keys=_EDIT_ARG_NAMES,
                    **warehouse_specs
                )
                .as_shallow_dict()
                .items()
            )
            if k in _EDIT_ARG_NAMES
        }

        same = dicts_equal(
            existing_details,
            update_details,
            keys=_EDIT_ARG_NAMES,
        )

        if not same:
            LOGGER.debug(
                "Updating warehouse %s (%s) with %s",
                self.warehouse_name, self.warehouse_id, update_details,
            )

            client = self.client.workspace_client().warehouses

            if wait.timeout:
                new_details = (
                    client
                    .edit_and_wait(
                        timeout=wait.timeout_timedelta,
                        **update_details
                    )
                )
            else:
                _ = client.edit(**update_details)
                new_details = EndpointInfo(**update_details)

            object.__setattr__(self, "_details", safeEndpointInfo(new_details))
        else:
            LOGGER.debug(
                "update: warehouse %s (%s) already up-to-date — skipping API call",
                self.warehouse_name, self.warehouse_id,
            )

        self.update_permissions(permissions=permissions, wait=wait)

        LOGGER.info("Updated warehouse %s (%s)", self.warehouse_name, self.warehouse_id)

        return self

    def update_permissions(
        self,
        permissions: Optional[List[WarehouseAccessControlRequest | str]] = None,
        *,
        wait: WaitingConfigArg = True,
        warehouse_id: Optional[str] = None
    ):
        client = self.client.workspace_client().warehouses
        warehouse_id = warehouse_id or self.warehouse_id

        permissions = [
            self.check_permission(_)
            for _ in permissions
        ] if permissions else []

        if permissions and warehouse_id:
            if wait:
                client.update_permissions(
                    warehouse_id=warehouse_id,
                    access_control_list=permissions
                )
            else:
                Job.make(
                    client.update_permissions,
                    warehouse_id=warehouse_id,
                    access_control_list=permissions
                ).fire_and_forget()

        return self

    @staticmethod
    def check_permission(
        permission: WarehouseAccessControlRequest | str
    ):
        if isinstance(permission, str):
            if permission == "users":
                return WarehouseAccessControlRequest(
                    group_name=permission,
                    permission_level=WarehousePermissionLevel.CAN_USE
                )
            elif "@" in permission:
                return WarehouseAccessControlRequest(
                    user_name=permission,
                    permission_level=WarehousePermissionLevel.CAN_USE
                )
            else:
                return WarehouseAccessControlRequest(
                    group_name=permission,
                    permission_level=WarehousePermissionLevel.CAN_MANAGE
                )

        return permission

    def delete(self):
        if self.warehouse_id:
            LOGGER.debug(
                "Deleting warehouse %s (%s)",
                self.warehouse_name, self.warehouse_id,
            )
            client = self.client.workspace_client().warehouses
            client.delete(id=self.warehouse_id)
            LOGGER.info(
                "Deleted warehouse %s (%s)",
                self.warehouse_name, self.warehouse_id,
            )

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
        raise_error: bool = True,
    ) -> StatementResult:
        """Execute a SQL statement via Spark or Databricks SQL Statement Execution API.

        Engine resolution:
        - If `engine` is not provided and a Spark session is active -> uses Spark.
        - Otherwise uses Databricks SQL API (warehouse).

        Waiting behavior (`wait_result`):
        - If True (default): returns a StatementResult in terminal state (SUCCEEDED/FAILED/CANCELED).
        - If False: returns immediately with the initial handle (caller can `.wait()` later).

        Args:
            statement: SQL statement to execute. If None, a `SELECT *` is generated from the table params.
            warehouse_id: Warehouse override (for API engine).
            warehouse_name: Warehouse name override (for API engine).
            byte_limit: Optional byte limit for results.
            disposition: Result disposition mode (API engine).
            format: Result format (API engine).
            on_wait_timeout: Timeout behavior for waiting (API engine).
            parameters: Optional statement parameters (API engine).
            row_limit: Optional row limit for results (API engine).
            wait_timeout: API wait timeout value.
            catalog_name: Optional catalog override for API engine.
            schema_name: Optional schema override for API engine.
            wait: Whether to block until completion (API engine).

        Returns:
            StatementResult
        """
        if format is None:
            format = Format.ARROW_STREAM

        if disposition is None:
            disposition = Disposition.EXTERNAL_LINKS
        elif format in (Format.CSV, Format.ARROW_STREAM):
            disposition = Disposition.EXTERNAL_LINKS

        instance = self.find_warehouse(warehouse_id=warehouse_id, warehouse_name=warehouse_name)
        warehouse_id = warehouse_id or instance.warehouse_id
        client = instance.client.workspace_client().statement_execution

        LOGGER.debug(
            "Executing SQL on warehouse %s (%s):\n%s",
            instance.warehouse_name, warehouse_id,
            statement,
        )

        response = client.execute_statement(
            statement=statement,
            warehouse_id=warehouse_id,
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

        execution = StatementResult(
            client=self.client,
            warehouse_id=warehouse_id,
            statement_id=response.statement_id,
            disposition=disposition,
            _response=response,
        )

        LOGGER.info(
            "Executed SQL statement_id=%s on warehouse %s (%s)",
            execution.statement_id, instance.warehouse_name, warehouse_id,
        )

        return execution.wait(wait=wait, raise_error=raise_error)


DEFAULT_ALL_PURPOSE_CLASSIC_NAME = "Yggdrasil All Purpose"
DEFAULT_ALL_PURPOSE_SERVERLESS_NAME = DEFAULT_ALL_PURPOSE_CLASSIC_NAME + " Serverless"
