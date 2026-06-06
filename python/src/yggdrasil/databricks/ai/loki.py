"""Loki skill for the **AI / model-serving** service (``dbc.ai``).

Databricks serving endpoints host Foundation Models and custom models. This
skill lists the serving endpoints, or queries one with a prompt (through the
same OpenAI-compatible client the :class:`DatabricksServingEngine` reasons on).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, names

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksServingSkill"]


@register
class DatabricksServingSkill(DatabricksServiceSkill):
    """List model-serving endpoints, or query one with a prompt."""

    name = "databricks-serving"
    description = "List Databricks serving endpoints, or query one with a prompt."
    preprompt = (
        "You drive Databricks model serving: list endpoints, or query one with "
        "a prompt via the OpenAI-compatible client. Prefer the smallest capable "
        "Foundation Model endpoint unless told otherwise."
    )

    def run(self, agent: "Loki", *, endpoint: Optional[str] = None,
            prompt: Optional[str] = None, **_: Any) -> dict[str, Any]:
        client = self._client(agent)
        if prompt:
            from yggdrasil.loki.engines import DatabricksServingEngine

            eng = DatabricksServingEngine(client=client, endpoint=endpoint)
            # The domain preprompt steers the served model toward the best answer.
            return {"endpoint": eng.endpoint, "reply": eng.generate(prompt, system=self.preprompt)}
        eps = client.workspace_client().serving_endpoints.list()
        return {"endpoints": names(eps, attrs=("name", "id"))}
