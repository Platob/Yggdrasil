import dataclasses as dc
import inspect
import logging
from typing import Optional, Sequence

from ..workspaces import WorkspaceService, Workspace
from ...pyutils.equality import dicts_equal, dict_diff
from ...pyutils.expiring_dict import ExpiringDict

try:
    from databricks.sdk import WarehousesAPI
    from databricks.sdk.service.sql import (
        State, EndpointInfo, EndpointTags, EndpointTagPair, EndpointInfoWarehouseType
)

    _CREATE_ARG_NAMES = {_ for _ in inspect.signature(WarehousesAPI.create).parameters.keys()}
    _EDIT_ARG_NAMES = {_ for _ in inspect.signature(WarehousesAPI.edit).parameters.keys()}
except ImportError:
    class WarehousesAPI:
        pass

    class State:
        pass

    class EndpointInfo:
        pass

    class EndpointTags:
        pass

    class EndpointTagPair:
        pass

    class EndpointInfoWarehouseType:
        pass


__all__ = [
    "SQLWarehouse"
]


LOGGER = logging.getLogger(__name__)
NAME_ID_CACHE: dict[str, ExpiringDict] = {}


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


@dc.dataclass
class SQLWarehouse(WorkspaceService):
    warehouse_id: Optional[str] = None
    warehouse_name: Optional[str] = None

    _details: Optional[EndpointInfo] = dc.field(default=None, repr=False)

    def warehouse_client(self):
        return self.workspace.sdk().warehouses

    def default(
        self,
        name: str = "YGG-DEFAULT",
        **kwargs
    ):
        return self.create_or_update(
            name=name,
            **kwargs
        )

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
    def details(self, value: EndpointInfo):
        self._details = value

        self.warehouse_id = value.id
        self.warehouse_name = value.name

    @property
    def state(self):
        return self.latest_details().state

    @property
    def running(self):
        return self.state in {State.RUNNING}

    @property
    def pending(self):
        return self.state in {State.DELETING, State.STARTING, State.STOPPING}

    def start(self):
        if not self.running:
            self.warehouse_client().start(id=self.warehouse_id)
        return self

    def stop(self):
        if self.running:
            return self.warehouse_client().stop(id=self.warehouse_id)
        return self

    def find_warehouse(
        self,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
        raise_error: bool = True
    ):
        if warehouse_id:
            return SQLWarehouse(
                workspace=self.workspace,
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name
            )

        if self.warehouse_id:
            return self

        warehouse_name = warehouse_name or self.warehouse_name

        warehouse_id = get_cached_warehouse_id(host=self.workspace.host, warehouse_name=warehouse_name)

        if warehouse_id:
            return SQLWarehouse(
                workspace=self.workspace,
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name
            )

        for warehouse in self.list_warehouses():
            if warehouse.warehouse_name == warehouse_name:
                set_cached_warehouse_name(host=self.workspace.host, warehouse_name=warehouse_name, warehouse_id=warehouse.warehouse_id)
                return warehouse

        if raise_error:
            raise ValueError(
                f"SQL Warehouse {warehouse_name!r} not found"
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

    def _check_details(
        self,
        keys: Sequence[str],
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

        if details.name is None:
            details.name = "YGG-%s" % details.cluster_size.upper()

        default_tags = self.workspace.default_tags()

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

        if details.warehouse_type is None:
            details.warehouse_type = EndpointInfoWarehouseType.CLASSIC

        return details

    def create_or_update(
        self,
        warehouse_id: Optional[str] = None,
        name: Optional[str] = None,
        **warehouse_specs
    ):
        name = name or self.warehouse_name
        found = self.find_warehouse(warehouse_id=warehouse_id, warehouse_name=name, raise_error=False)

        if found is not None:
            return found.update(name=name, **warehouse_specs)
        return self.create(name=name, **warehouse_specs)

    def create(
        self,
        name: Optional[str] = None,
        **warehouse_specs
    ):
        name = name or self.warehouse_name

        details = self._check_details(
            keys=_CREATE_ARG_NAMES,
            name=name,
            **warehouse_specs
        )

        info = self.warehouse_client().create_and_wait(**{
            k: v
            for k, v in details.as_shallow_dict().items()
            if k in _CREATE_ARG_NAMES
        })

        return SQLWarehouse(
            workspace=self.workspace,
            warehouse_id=info.id,
            warehouse_name=info.name,
            _details=info
        )

    def update(
        self,
        **warehouse_specs
    ):
        if not warehouse_specs:
            return self

        existing_details = {
            k: v
            for k, v in self.details.as_shallow_dict().items()
            if k in _EDIT_ARG_NAMES
        }

        update_details = {
            k: v
            for k, v in (
                self._check_details(details=self.details, keys=_EDIT_ARG_NAMES, **warehouse_specs)
                .as_shallow_dict()
                .items()
            )
            if k in _EDIT_ARG_NAMES
        }

        same = dicts_equal(
            existing_details,
            update_details,
            keys=_EDIT_ARG_NAMES,
            treat_missing_as_none=True,
            float_tol=0.0,  # set e.g. 1e-6 if you have float-y stuff
        )

        if not same:
            diff = {
                k: v[1]
                for k, v in dict_diff(existing_details, update_details, keys=_EDIT_ARG_NAMES).items()
            }

            LOGGER.debug(
                "Updating %s with %s",
                self, diff
            )

            self.warehouse_client().edit_and_wait(**update_details)

            LOGGER.info(
                "Updated %s",
                self
            )

        return self

    def sql(
        self,
        workspace: Optional[Workspace] = None,
        warehouse_id: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
    ):
        """Return a SQLEngine configured for this workspace.

        Args:
            workspace: Optional workspace override.
            warehouse_id: Optional SQL warehouse id.
            catalog_name: Optional catalog name.
            schema_name: Optional schema name.

        Returns:
            A SQLEngine instance.
        """

        return self.workspace.sql(
            workspace=workspace,
            warehouse_id=warehouse_id or self.warehouse_id,
            catalog_name=catalog_name,
            schema_name=schema_name
        )