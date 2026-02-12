import dataclasses as dc
import inspect
import logging
import time
from typing import Optional, Sequence, Any, Type, TypeVar, Union, List

from databricks.sdk import WarehousesAPI
from databricks.sdk.service.sql import (
    State, EndpointInfo,
    EndpointTags, EndpointTagPair, EndpointInfoWarehouseType,
    GetWarehouseResponse, GetWarehouseResponseWarehouseType,
    Disposition, Format,
    ExecuteStatementRequestOnWaitTimeout, StatementParameterListItem,
    WarehouseAccessControlRequest, WarehousePermissionLevel
)

from .statement_result import StatementResult
from ..workspaces import WorkspaceService
from ...pyutils.equality import dicts_equal
from ...pyutils.expiring_dict import ExpiringDict
from ...pyutils.waiting_config import WaitingConfig, WaitingConfigArg

_CREATE_ARG_NAMES = {_ for _ in inspect.signature(WarehousesAPI.create).parameters.keys()}
_EDIT_ARG_NAMES = {_ for _ in inspect.signature(WarehousesAPI.edit).parameters.keys()}

__all__ = [
    "SQLWarehouse"
]


LOGGER = logging.getLogger(__name__)
NAME_ID_CACHE: dict[str, ExpiringDict] = {}
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


def set_cached_warehouse_name(
    host: str,
    warehouse_name: str,
    warehouse_id: str
) -> None:
    existing = NAME_ID_CACHE.get(host)

    if not existing:
        existing = NAME_ID_CACHE[host] = ExpiringDict(default_ttl=60)

    existing[warehouse_name] = warehouse_id


def get_cached_warehouse_id(
    host: str,
    warehouse_name: str,
) -> str:
    existing = NAME_ID_CACHE.get(host)

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


@dc.dataclass
class SQLWarehouse(WorkspaceService):
    warehouse_id: Optional[str] = None
    warehouse_name: Optional[str] = None

    _details: Optional[EndpointInfo] = dc.field(default=None, repr=False, hash=False, compare=False)

    def warehouse_client(self):
        return self.workspace.sdk().warehouses

    @property
    def details(self) -> EndpointInfo:
        if self._details is None:
            self.refresh()
        return self._details

    def latest_details(self):
        return self.warehouse_client().get(id=self.warehouse_id)

    def refresh(self):
        self.details = self.latest_details()
        return self

    @details.setter
    def details(self, value: Union[GetWarehouseResponse, EndpointInfo]):
        self._details = safeEndpointInfo(value)

        if self._details is not None:
            self.warehouse_id = self._details.id
            self.warehouse_name = self._details.name

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
        wait: Optional[WaitingConfigArg] = None
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
            while self.is_pending:
                wait.sleep(iteration=iteration, start=start)
                iteration += 1

        return self

    def start(self):
        if not self.is_running:
            self.warehouse_client().start(id=self.warehouse_id)
        return self

    def stop(self):
        if self.is_running:
            return self.warehouse_client().stop(id=self.warehouse_id)
        return self

    def find_warehouse(
        self,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
        raise_error: bool = True,
        find_starter: bool = False
    ):
        if warehouse_id:
            if warehouse_id == self.warehouse_id:
                return self

            return SQLWarehouse(
                workspace=self.workspace,
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name
            )

        elif self.warehouse_id:
            return self

        starter_warehouse, starter_name = None, "Serverless Starter Warehouse"
        warehouse_name = warehouse_name or self.warehouse_name or self._make_default_name(enable_serverless_compute=True)

        if warehouse_name:
            if warehouse_name == self.warehouse_name:
                return self

            warehouse_id = get_cached_warehouse_id(
                host=self.workspace.safe_host,
                warehouse_name=warehouse_name
            )

            if warehouse_id:
                if warehouse_id == self.warehouse_id:
                    return self

                return SQLWarehouse(
                    workspace=self.workspace,
                    warehouse_id=warehouse_id,
                    warehouse_name=warehouse_name
                )

            for warehouse in self.list_warehouses():
                if warehouse.warehouse_name == warehouse_name:
                    set_cached_warehouse_name(
                        host=self.workspace.safe_host,
                        warehouse_name=warehouse_name,
                        warehouse_id=warehouse.warehouse_id
                    )

                    return warehouse

                elif warehouse.warehouse_name == starter_name:
                    starter_warehouse = warehouse

        if find_starter and starter_warehouse is not None:
            return starter_warehouse

        if raise_error:
            v = warehouse_name or warehouse_id

            raise ValueError(
                f"SQL Warehouse {v!r} not found"
            )

        return None

    def list_warehouses(self):
        for info in self.warehouse_client().list():
            warehouse = SQLWarehouse(
                workspace=self.workspace,
                warehouse_id=info.id,
                warehouse_name=info.name,
                _details=info
            )

            yield warehouse

    def _make_default_name(self, enable_serverless_compute: bool = True):
        return "%s%s" % (
            self.workspace.product or "yggdrasil",
            " serverless" if enable_serverless_compute else ""
        )

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

        if details.enable_serverless_compute is None:
            details.enable_serverless_compute = details.warehouse_type.value == EndpointInfoWarehouseType.PRO.value
        elif details.enable_serverless_compute:
            details.warehouse_type = EndpointInfoWarehouseType.PRO

        if not details.name:
            details.name = self._make_default_name(
                enable_serverless_compute=details.enable_serverless_compute
            )

        default_tags = self.workspace.default_tags(update=update)

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
            details.max_num_clusters = 4

        return details

    def create_or_update(
        self,
        warehouse_id: Optional[str] = None,
        name: Optional[str] = None,
        wait: Optional[WaitingConfig] = None,
        **warehouse_specs
    ):
        name = name or self.warehouse_name
        found = self.find_warehouse(
            warehouse_id=warehouse_id,
            warehouse_name=name,
            raise_error=False,
            find_starter=False
        )

        if found is not None:
            return found.update(name=name, wait=wait, **warehouse_specs)

        return self.create(name=name, wait=wait, **warehouse_specs)

    def create(
        self,
        name: Optional[str] = None,
        wait: Optional[WaitingConfigArg] = None,
        **warehouse_specs
    ):
        name = name or self.warehouse_name

        checked_wait = WaitingConfig.check_arg(wait)

        details = self._check_details(
            keys=_CREATE_ARG_NAMES,
            update=False,
            name=name,
            **warehouse_specs
        )

        update_details = {
            k: v
            for k, v in details.as_shallow_dict().items()
            if k in _CREATE_ARG_NAMES
        }

        if checked_wait.timeout:
            info = self.warehouse_client().create_and_wait(
                timeout=checked_wait.timeout_timedelta,
                **update_details
            )
        else:
            info = self.warehouse_client().create(**update_details)

            update_details["id"] = info.response.id

            info = EndpointInfo(**update_details)

        created = SQLWarehouse(
            workspace=self.workspace,
            warehouse_id=info.id,
            warehouse_name=info.name,
            _details=info
        )

        created.update_permissions()

        return created

    def update(
        self,
        wait: Optional[WaitingConfigArg] = None,
        **warehouse_specs
    ):
        if not warehouse_specs:
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
                "Updating %s with %s",
                self, update_details
            )

            if wait.timeout:
                self.details = self.warehouse_client().edit_and_wait(
                    timeout=wait.timeout_timedelta,
                    **update_details
                )
            else:
                _ = self.warehouse_client().edit(**update_details)

                self.details = EndpointInfo(**update_details)

            LOGGER.info(
                "Updated %s",
                self
            )

        return self

    def update_permissions(
        self,
        access_control_list: Optional[List[WarehouseAccessControlRequest]] = None,
        wait: Optional[WaitingConfigArg] = None
    ):
        if self.warehouse_id:
            client = self.warehouse_client()

            access_control_list = self._check_access_control_list(
                access_control_list=access_control_list
            )

            if access_control_list:
                client.update_permissions(
                    warehouse_id=self.warehouse_id,
                    access_control_list=access_control_list
                )

    def default_access_control_list(self, for_all: bool):
        if for_all:
            base = [
                WarehouseAccessControlRequest(
                    group_name="users",
                    permission_level=WarehousePermissionLevel.CAN_USE
                )
            ]
        else:
            base = []

        groups = self.workspace.current_user_groups(
            with_public=False
        )

        if groups:
            base.extend(
                WarehouseAccessControlRequest(
                    group_name=group.display,
                    permission_level=WarehousePermissionLevel.CAN_MANAGE
                )
                for group in groups
            )

        return base

    def _check_access_control_list(
        self,
        access_control_list: Optional[List[WarehouseAccessControlRequest]] = None
    ):
        if access_control_list is None:
            access_control_list = []

        access_control_list.extend(self.default_access_control_list(
            for_all=self.warehouse_name.startswith("yggdrasil") if self.warehouse_name else False
        ))

        return access_control_list

    def delete(self):
        if self.warehouse_id:
            LOGGER.debug(
                "Deleting %s",
                self
            )

            self.warehouse_client().delete(id=self.warehouse_id)

            LOGGER.info(
                "Deleted %s",
                self
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
        wait: Optional[WaitingConfigArg] = True
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
        workspace_client = instance.workspace.sdk()

        LOGGER.debug(
            "API SQL executing query:\n%s",
            statement
        )

        response = workspace_client.statement_execution.execute_statement(
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
            workspace_client=workspace_client,
            warehouse_id=warehouse_id,
            statement_id=response.statement_id,
            disposition=disposition,
            _response=response,
        )

        LOGGER.info(
            "API SQL executed %s",
            execution
        )

        return execution.wait(wait=wait)
