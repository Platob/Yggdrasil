"""Anthropic (Claude) -backed :class:`TokenEngine`."""
from __future__ import annotations

import os
from typing import Any, ClassVar, Optional

from ..engine import DEFAULT_MAX_TOKENS, Completion, EngineType, TokenEngine
from ..model import Provider, TokenModel

__all__ = ["ClaudeEngine"]


class ClaudeEngine(TokenEngine):
    """Reason via the Anthropic Messages API (``anthropic`` SDK)."""

    name = "claude"
    type: ClassVar[EngineType] = EngineType.CLAUDE
    provider: ClassVar[Provider] = Provider.ANTHROPIC
    #: The most capable current Claude model (see the claude-api reference).
    default_model: ClassVar[str] = TokenModel.CLAUDE_OPUS_4_8.id

    def __init__(self, *, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        super().__init__(model=model)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        complexity: Any = None,
        **options: Any,
    ) -> Completion:
        import anthropic

        # Anthropic keeps the system prompt out of the message list.
        kwargs: dict[str, Any] = {
            "model": self.resolve_model(complexity),
            "max_tokens": max_tokens,
            "messages": list(messages),
        }
        if system:
            kwargs["system"] = system
        kwargs.update(options)
        resp = anthropic.Anthropic(api_key=self.api_key).messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = getattr(resp, "usage", None)
        return Completion(
            text=text,
            model=getattr(resp, "model", self.model),
            usage={"input_tokens": getattr(usage, "input_tokens", None),
                   "output_tokens": getattr(usage, "output_tokens", None)} if usage else {},
            raw=resp,
        )
