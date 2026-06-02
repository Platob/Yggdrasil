"""Resources for the Databricks model-serving service.

A :class:`ServingEndpoint` is a thin, ergonomic handle over a Databricks
serving endpoint (Foundation Model APIs, external models, or custom
served models). The two entry points are :meth:`ServingEndpoint.chat`
(role/content messages) and :meth:`ServingEndpoint.complete` (a single
prompt, optionally with a system instruction). Both return a
:class:`ChatResult` carrying the assistant text plus token usage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

from yggdrasil.dataclasses import WaitingConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk.service.serving import (
        QueryEndpointResponse,
        ServingEndpointDetailed,
    )

    from .service import ModelServing


__all__ = [
    "DEFAULT_SERVING_ENDPOINT",
    "ChatResult",
    "ServingDefaults",
    "ServingEndpoint",
    "MessageLike",
]


#: A pay-per-token Foundation Model endpoint that ships in every workspace —
#: a sensible default for code reasoning. Override per-call or via
#: :attr:`ServingDefaults.endpoint_name`.
DEFAULT_SERVING_ENDPOINT = "databricks-claude-3-7-sonnet"


#: A chat message is either an SDK ``ChatMessage`` or a ``{"role", "content"}``
#: mapping / ``(role, content)`` pair — normalised by :meth:`ServingEndpoint.chat`.
MessageLike = Union[Mapping[str, str], Sequence[str], Any]


@dataclass
class ServingDefaults:
    """Service-wide model-serving defaults.

    Set once on the service (``client.ai.serving.defaults = replace(...)``)
    so callers stop repeating ``endpoint_name`` / generation params on every
    query — mirrors :class:`~yggdrasil.databricks.ai.VectorSearchDefaults`.
    """

    endpoint_name: str = DEFAULT_SERVING_ENDPOINT
    max_tokens: int = 1024
    temperature: float = 0.0
    wait: Optional[WaitingConfig] = None


@dataclass
class ChatResult:
    """The assistant turn returned by a serving-endpoint query."""

    content: str
    model: str = ""
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    raw: Any = None

    def __str__(self) -> str:
        return self.content

    @classmethod
    def from_response(cls, resp: "QueryEndpointResponse") -> "ChatResult":
        """Pull the first choice's assistant text out of a chat response.

        Handles both the chat shape (``choices[].message.content``) and the
        legacy completion shape (``choices[].text`` / ``predictions``)."""
        content = ""
        finish = ""
        choices = getattr(resp, "choices", None) or []
        if choices:
            first = choices[0]
            finish = getattr(first, "finish_reason", "") or ""
            message = getattr(first, "message", None)
            if message is not None and getattr(message, "content", None):
                content = message.content or ""
            elif getattr(first, "text", None):
                content = first.text or ""
        if not content:
            # Custom / pyfunc endpoints answer via ``predictions``.
            preds = getattr(resp, "predictions", None)
            if preds:
                content = preds[0] if isinstance(preds, list) else str(preds)
        usage_obj = getattr(resp, "usage", None)
        usage = usage_obj.as_dict() if usage_obj is not None and hasattr(usage_obj, "as_dict") else {}
        return cls(
            content=content if isinstance(content, str) else str(content),
            model=getattr(resp, "model", "") or "",
            finish_reason=finish,
            usage=usage,
            raw=resp,
        )


@dataclass
class ServingEndpoint:
    """Handle to a single Databricks serving endpoint."""

    service: "ModelServing"
    endpoint_name: str
    details: "Optional[ServingEndpointDetailed]" = None

    @property
    def client(self):
        return self.service.client

    def refresh(self) -> "ServingEndpoint":
        """Fetch (and cache) the endpoint's :class:`ServingEndpointDetailed`."""
        self.details = self.service.endpoints_api.get(name=self.endpoint_name)
        return self

    def exists(self) -> bool:
        from databricks.sdk.errors import NotFound

        try:
            self.refresh()
            return True
        except NotFound:
            return False

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: Sequence[MessageLike],
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
        **extra: Any,
    ) -> ChatResult:
        """Run a chat completion against this endpoint.

        ``messages`` accepts SDK ``ChatMessage`` objects, ``{"role", "content"}``
        mappings, or ``(role, content)`` pairs. ``system`` prepends a system
        message. Generation params fall back to :class:`ServingDefaults`.
        """
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        defaults = self.service.defaults
        normalised: list[ChatMessage] = []
        if system:
            normalised.append(ChatMessage(role=ChatMessageRole.SYSTEM, content=system))
        for msg in messages:
            if isinstance(msg, ChatMessage):
                normalised.append(msg)
                continue
            if isinstance(msg, Mapping):
                role, content = msg.get("role", "user"), msg.get("content", "")
            else:  # (role, content) pair
                role, content = msg[0], msg[1]
            normalised.append(
                ChatMessage(role=ChatMessageRole(str(role)), content=str(content))
            )
        resp = self.service.endpoints_api.query(
            name=self.endpoint_name,
            messages=normalised,
            max_tokens=defaults.max_tokens if max_tokens is None else max_tokens,
            temperature=defaults.temperature if temperature is None else temperature,
            stop=list(stop) if stop else None,
            **extra,
        )
        return ChatResult.from_response(resp)

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **extra: Any,
    ) -> ChatResult:
        """Single-prompt convenience over :meth:`chat`."""
        return self.chat(
            [{"role": "user", "content": prompt}],
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            **extra,
        )
