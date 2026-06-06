"""A :class:`TokenEngine` backed by another Loki agent (delegation)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional

from ..engine import DEFAULT_MAX_TOKENS, Completion, EngineType, TokenEngine

if TYPE_CHECKING:
    from ..agent import Loki

__all__ = ["LokiAgentEngine"]


class LokiAgentEngine(TokenEngine):
    """Reason by delegating to another :class:`~yggdrasil.loki.Loki` agent.

    Lets one agent use a (possibly specialized or spawned) peer agent as its
    brain — the ``loki_agent`` engine type.
    """

    name = "loki"
    type: ClassVar[EngineType] = EngineType.LOKI_AGENT

    def __init__(self, agent: Loki, *, model: Optional[str] = None) -> None:
        super().__init__(model=model or "loki-agent")
        self.agent = agent

    def available(self) -> bool:
        return self.agent is not None

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        complexity: Any = None,
        **options: Any,
    ) -> Completion:
        prompt = messages[-1]["content"] if messages else ""
        response = self.agent.reason(prompt, system=system, complexity=complexity)
        text = getattr(response, "text", None) or str(response)
        return Completion(text=text, model=self.model, raw=response)
