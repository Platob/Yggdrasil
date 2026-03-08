"""Workspace configuration and Databricks SDK helpers."""

import logging
from dataclasses import dataclass, field

from databricks.sdk.service.provisioning import Workspace as SDKWorkspace

from yggdrasil.io.url import URL
from .service import Workspaces
from ..client import DatabricksClient, DatabricksResource

__all__ = [
    "Workspace",
    "WorkspaceResource"
]


LOGGER = logging.getLogger(__name__)


@dataclass
class Workspace(DatabricksClient):
    pass

@dataclass
class WorkspaceResource(DatabricksResource):
    service: Workspaces = field(
        default_factory=Workspaces.current,
        repr=False,
        compare=False,
        hash=False,
    )
    id: str = ""
    name: str = ""
    url: URL = URL.empty()

    @property
    def details(self):
        return self.client.account_client().workspaces.get(workspace_id=self.id)

    def set_details(self, details: SDKWorkspace) -> None:
        self.id = details.workspace_id
        self.name = details.workspace_name
        self.url = URL.parse_str(details.workspace_id)

        return self

    def refresh(self):
        details: SDKWorkspace = (
            self.client.account_client()
            .workspaces.get(workspace_id=self.id)
        )

        return self.set_details(details)
