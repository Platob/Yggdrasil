"""Genie service — code-oriented manipulation of Databricks AI/BI Genie.

``client.genie`` (also ``dbc.genie``) is the collection-level entry point
over the SDK ``GenieAPI``: list / find / get / create Genie spaces, and ask
a space a question. Individual space lifecycle + Q&A live on the
:class:`~yggdrasil.databricks.genie.space.GenieSpace` resource this service
returns; a completed turn comes back as a
:class:`~yggdrasil.databricks.genie.answer.GenieAnswer` (text + generated
SQL + a re-attachable tabular result).

    dbc.genie.spaces()                       # list spaces
    space = dbc.genie["01ef…"]               # a GenieSpace handle
    answer = space.ask("top 10 customers by revenue last quarter")
    answer.text                              # Genie's narrative
    answer.to_polars()                       # the rows it computed
"""
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Iterator, Optional

from ..client import DatabricksService
from .answer import GenieAnswer
from .space import _DEFAULT_TIMEOUT, GenieSpace

if TYPE_CHECKING:
    from ..client import DatabricksClient

__all__ = ["Genie"]


class Genie(DatabricksService):
    """Collection-level Databricks Genie service (``dbc.genie``)."""

    @property
    def api(self):
        """The underlying SDK ``GenieAPI`` bound to this client's workspace."""
        return self.client.workspace_client().genie

    # -- spaces ------------------------------------------------------------

    def list_spaces(self, *, page_size: Optional[int] = None) -> Iterator[GenieSpace]:
        """Iterate every Genie space, transparently paging the API."""
        token: Optional[str] = None
        while True:
            resp = self.api.list_spaces(page_size=page_size, page_token=token)
            for info in resp.spaces or []:
                yield GenieSpace(info.space_id, client=self.client, info=info)
            token = resp.next_page_token
            if not token:
                return

    def spaces(self) -> list[GenieSpace]:
        """All Genie spaces as a list."""
        return list(self.list_spaces())

    def space(self, space_id: str) -> GenieSpace:
        """A :class:`GenieSpace` handle for *space_id* (metadata is lazy)."""
        return GenieSpace(space_id, client=self.client)

    # dbc.genie["01ef…"] → GenieSpace
    __getitem__ = space

    def find(self, title: str) -> Optional[GenieSpace]:
        """Find a space by exact title, or ``None``."""
        for space in self.list_spaces():
            if space.title == title:
                return space
        return None

    def create_space(
        self,
        warehouse_id: str,
        serialized_space: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        parent_path: Optional[str] = None,
    ) -> GenieSpace:
        """Create a Genie space from a serialized definition."""
        info = self.api.create_space(
            warehouse_id,
            serialized_space,
            title=title,
            description=description,
            parent_path=parent_path,
        )
        return GenieSpace(info.space_id, client=self.client, info=info)

    # -- ask ---------------------------------------------------------------

    def ask(
        self,
        space_id: str,
        question: str,
        *,
        timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
    ) -> GenieAnswer:
        """Ask *question* of *space_id* in a fresh conversation and wait."""
        return self.space(space_id).ask(question, timeout=timeout)
