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
        ...
