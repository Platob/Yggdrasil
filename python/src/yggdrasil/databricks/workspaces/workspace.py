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
