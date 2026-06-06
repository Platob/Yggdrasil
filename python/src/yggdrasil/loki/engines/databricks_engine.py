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
        # The serving endpoint *is* the model selector and is workspace-specific
        # (no assumed fast/deep pair), so this engine pins to one endpoint rather
        # than adapting; ``tier`` is accepted for contract parity and ignored.
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

    #: base_url → OpenAI-compatible client, cached per workspace. Building it
    #: resolves the workspace client + auth and stands up a fresh connection
    #: pool; rebuilding on every completion is the bulk of serving latency, so
    #: reuse one client (and its keep-alive pool) across turns.
    _OAI: "dict[str, Any]" = {}

    def _oai_client(self):
        """The OpenAI-compatible client for the workspace's serving endpoints.

        `serving_endpoints.get_open_ai_client()` needs the ``openai`` package
        (shipped as the ``databricks-sdk[openai]`` extra) — auto-install it on
        first use so reasoning just works. Cached per workspace so the client
        and its connection pool are reused across completions.
        """
        key = getattr(self.client, "base_url", "") or ""
        cached = type(self)._OAI.get(key)
        if cached is not None:
            return cached
        from ..runtime import load

        load("openai", "databricks-sdk[openai]")
        client = self.client.workspace_client().serving_endpoints.get_open_ai_client()
        type(self)._OAI[key] = client
        return client

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tier: Optional[str] = None,
        **options: Any,
    ) -> Completion:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        oai = self._oai_client()
        endpoint, resp = self._create(oai, msgs, max_tokens, options, stream=False)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        self._record(
            endpoint,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            messages=messages, system=system, text=text,
        )
        return Completion(text=text, model=getattr(resp, "model", endpoint), raw=resp)

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tier: Optional[str] = None,
        **options: Any,
    ):
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        oai = self._oai_client()
        endpoint, resp = self._create(oai, msgs, max_tokens, options, stream=True)
        parts: list[str] = []
        for chunk in resp:
            if chunk.choices:
                delta = getattr(chunk.choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if piece:
                    parts.append(piece)
                    yield piece
        self._record(endpoint, messages=messages, system=system, text="".join(parts))

    # -- endpoint resolution ----------------------------------------------
    #
    # The configured endpoint may not be deployed in a given workspace. On a
    # not-found, list the serving endpoints once, pick a chat-capable one, and
    # cache it per workspace so the session self-heals to a working model.

    _RESOLVED: "dict[str, str]" = {}

    def _endpoint(self) -> str:
        key = getattr(self.client, "base_url", "") or ""
        return type(self)._RESOLVED.get(key, self.endpoint)

    def _resolve_after_failure(self) -> str:
        names = [e.name for e in self.client.workspace_client().serving_endpoints.list()]
        chat = next(
            (n for n in names if any(
                k in n.lower() for k in
                ("gpt", "llama", "claude", "qwen", "mixtral", "dbrx", "gemma", "instruct"))),
            names[0] if names else self.endpoint,
        )
        type(self)._RESOLVED[getattr(self.client, "base_url", "") or ""] = chat
        return chat

    def _create(self, oai, msgs, max_tokens, options, *, stream):
        endpoint = self._endpoint()
        try:
            resp = oai.chat.completions.create(
                model=endpoint, messages=msgs, max_tokens=max_tokens, stream=stream, **options,
            )
        except Exception as exc:  # endpoint missing → resolve a real one, retry once
            if "ENDPOINT_NOT_FOUND" not in str(exc) and "404" not in str(exc):
                raise
            endpoint = self._resolve_after_failure()
            resp = oai.chat.completions.create(
                model=endpoint, messages=msgs, max_tokens=max_tokens, stream=stream, **options,
            )
        return endpoint, resp
