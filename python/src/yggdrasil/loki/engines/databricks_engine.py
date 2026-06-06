"""Databricks model-serving -backed :class:`TokenEngine`.

Reasons against a Databricks serving endpoint (a Foundation Model API
endpoint, or any chat-shaped served model) using the authenticated
``DatabricksClient`` as the credential provider — so a Loki running on
Databricks needs no extra API key.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

from ..engine import DEFAULT_MAX_TOKENS, Completion, TokenEngine

__all__ = ["DatabricksServingEngine"]

# The **lowest** (smallest / cheapest) broadly-available Databricks Foundation
# Model API chat endpoint — the default so reasoning is cheap unless a caller
# opts up. Override per workspace via ``endpoint=`` / the agent's configured
# endpoint.
DEFAULT_ENDPOINT = "databricks-meta-llama-3-1-8b-instruct"


class DatabricksServingEngine(TokenEngine):
    """Reason via a Databricks serving endpoint."""

    name = "databricks"

    def __init__(
        self,
        *,
        client: Any = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        # The endpoint name *is* the model selector here.
        super().__init__(model=model or endpoint or DEFAULT_ENDPOINT)
        self.endpoint = endpoint or model or DEFAULT_ENDPOINT
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from yggdrasil.databricks import DatabricksClient

            self._client = DatabricksClient.current()
        return self._client

    def available(self) -> bool:
        try:
            return bool(self.client and self.client.base_url)
        except Exception:
            return False

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **options: Any,
    ) -> Completion:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        w = self.client.workspace_client()

        # Preferred path: the SDK's OpenAI-compatible client points straight at
        # the workspace's serving endpoints — same chat.completions surface.
        get_oai = getattr(w.serving_endpoints, "get_open_ai_client", None)
        if callable(get_oai):
            resp = get_oai().chat.completions.create(
                model=self.endpoint, messages=msgs, max_tokens=max_tokens, **options,
            )
            choice = resp.choices[0]
            return Completion(
                text=choice.message.content or "",
                model=getattr(resp, "model", self.endpoint),
                raw=resp,
            )

        # Fallback: the native query API.
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        chat = [
            ChatMessage(role=ChatMessageRole(m["role"].upper()), content=m["content"])
            for m in msgs
        ]
        resp = w.serving_endpoints.query(
            name=self.endpoint, messages=chat, max_tokens=max_tokens,
        )
        choices = getattr(resp, "choices", None) or []
        text = choices[0].message.content if choices else ""
        return Completion(text=text or "", model=self.endpoint, raw=resp)
