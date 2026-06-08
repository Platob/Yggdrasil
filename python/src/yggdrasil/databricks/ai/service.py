"""Databricks AI umbrella service.

Holds the AI-shaped sub-services (vector search today; model serving
and registered models are on the roadmap — see ``ygg-mlops``). Reach
each via ``client.ai.<service>``::

    client.ai.vector_search          # vector search endpoints + indexes
    client.ai.vector_search.endpoint("rag").ensure_created()
    client.ai.vector_search.index("main.rag.docs").query(query_text="…", columns=["id", "text"])
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from yggdrasil.databricks.service import DatabricksService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .vector_search import VectorSearch


__all__ = ["DatabricksAI"]


class DatabricksAI(DatabricksService):
    """Umbrella for Databricks AI sub-services on a single client."""

    def __init__(self, client=None):
        super().__init__(client=client)
        self._vector_search: "Optional[VectorSearch]" = None

    @property
    def vector_search(self) -> "VectorSearch":
        """Vector search service (lazy + cached on this :class:`DatabricksAI`)."""
        cached = self._vector_search
        if cached is None:
            from .vector_search import VectorSearch
            cached = VectorSearch(client=self.client)
            self._vector_search = cached
        return cached
