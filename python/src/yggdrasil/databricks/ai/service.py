"""Databricks AI umbrella service.

Holds the AI-shaped sub-services. Reach each via ``client.ai.<service>``::

    client.ai.vector_search          # vector search endpoints + indexes
    client.ai.vector_search.endpoint("rag").ensure_created()
    client.ai.vector_search.index("main.rag.docs").query(query_text="…", columns=["id", "text"])

    client.ai.serving                # model-serving (Foundation Models, …)
    client.ai.serving.complete("Refactor this loop", system="You are a code reviewer")

    client.ai.optimizer(repo_path="/Workspace/Shared/monteleq").run()  # propose-only agent
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from yggdrasil.databricks.service import DatabricksService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .optimizer import OptimizerConfig, RepoOptimizer
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
        """Model-serving service (lazy + cached on this :class:`DatabricksAI`)."""
        cached = self._serving
        if cached is None:
            from .serving import ModelServing
            cached = ModelServing(client=self.client)
            self._serving = cached
        return cached

    def optimizer(self, config: "Optional[OptimizerConfig]" = None, **overrides) -> "RepoOptimizer":
        """Build a :class:`RepoOptimizer` — the propose-only repo optimization agent.

        Pass an :class:`OptimizerConfig` or config fields as keywords
        (``repo_path=``, ``endpoint_name=``, ``max_files=``, …)::

            client.ai.optimizer(repo_path="/Workspace/Shared/monteleq").run()
        """
        from .optimizer import OptimizerConfig, RepoOptimizer

        if config is None:
            config = OptimizerConfig(**overrides)
        elif overrides:
            from dataclasses import replace

            config = replace(config, **overrides)
        return RepoOptimizer(client=self.client, config=config)
