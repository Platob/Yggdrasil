"""OpenAI-backed :class:`TokenEngine`."""
from __future__ import annotations

import os
from typing import Any, ClassVar, Optional

from ..engine import DEFAULT_MAX_TOKENS, Completion, TokenEngine

__all__ = ["OpenAIEngine"]


class OpenAIEngine(TokenEngine):
    """Reason via the OpenAI Chat Completions API (``openai`` SDK)."""

    name = "openai"
    default_model: ClassVar[str] = "gpt-4o-mini"

    def __init__(self, *, model: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None) -> None:
        super().__init__(model=model)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

    def available(self) -> bool:
        return bool(self.api_key)

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **options: Any,
    ) -> Completion:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        resp = self._client().chat.completions.create(
            model=self.model, messages=msgs, max_tokens=max_tokens, **options,
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return Completion(
            text=choice.message.content or "",
            model=getattr(resp, "model", self.model),
            usage=usage.model_dump() if hasattr(usage, "model_dump") else {},
            raw=resp,
        )
