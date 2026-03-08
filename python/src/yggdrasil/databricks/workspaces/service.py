from typing import Iterator, TYPE_CHECKING

from ..client import DatabricksService

if TYPE_CHECKING:
    from .workspace import Workspace

__all__ = [
    "Workspaces"
]


class Workspaces(DatabricksService):

    def list(
        self,
    ) -> Iterator["Workspace"]:
        from .workspace import Workspace

        client = self.client.workspace_client().get_workspace_id()

        for details in client.workspaces.list():
            workspace = Workspace(

            )
