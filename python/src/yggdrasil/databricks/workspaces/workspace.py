"""Workspace configuration and Databricks SDK helpers."""

import logging
from dataclasses import dataclass, field

from yggdrasil.io.url import URL

from .service import Workspaces
from ..client import DatabricksClient, DatabricksResource

__all__ = [
    "Workspace",
    "WorkspaceResource"
]


LOGGER = logging.getLogger(__name__)


class Workspace(DatabricksClient):
    pass


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

    def __init__(
        self,
        service: Workspaces | None = None,
        id: str | None = None,
        name: str | None = None,
        url: URL | None = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.service = Workspaces.current() if service is None else service
        self.id = id or ""
        self.name = name or ""
        self.url = url or URL.empty()

    @property
    def details(self):
        return self.client.account_client().workspaces.get(workspace_id=self.id)
