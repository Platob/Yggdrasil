"""Databricks AI umbrella service.

Holds the AI-shaped sub-services. Reach each via ``client.ai.<service>``::

    client.ai.vector_search          # vector search endpoints + indexes
    client.ai.vector_search.endpoint("rag").ensure_created()
    client.ai.vector_search.index("main.rag.docs").query(query_text="…", columns=["id", "text"])

    client.ai.serving                # model serving — LLMs, agents, external models
    client.ai.serving.endpoint("gpt-4o").serve_openai("gpt-4o", api_key_secret="llm/key")
    client.ai.serving.endpoint("databricks-meta-llama-3-3-70b-instruct").chat("Hi!").text
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from yggdrasil.databricks.service import DatabricksService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .serving import ModelServing
    from .vector_search import VectorSearch


__all__ = ["DatabricksAI"]


class DatabricksAI(DatabricksService):
    """Umbrella for Databricks AI sub-services on a single client."""

    def __init__(self, client=None):
        super().__init__(client=client)
        self._vector_search: "Optional[VectorSearch]" = None
        self._serving: "Optional[ModelServing]" = None

    @property
    def vector_search(self) -> "VectorSearch":
        """Vector search service (lazy + cached on this :class:`DatabricksAI`)."""
        cached = self._vector_search
        if cached is None:
            from .vector_search import VectorSearch
            cached = VectorSearch(client=self.client)
            self._vector_search = cached
        return cached

    @property
    def serving(self) -> "ModelServing":
        """Model serving service (lazy + cached on this :class:`DatabricksAI`)."""
        cached = self._serving
        if cached is None:
            from .serving import ModelServing
            cached = ModelServing(client=self.client)
            self._serving = cached
        return cached
