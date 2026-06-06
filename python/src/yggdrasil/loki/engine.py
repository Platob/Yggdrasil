"""TokenEngine — the LLM reasoning contract Loki agents run on.

A :class:`TokenEngine` turns a chat message list into a completion. It's the
single seam between Loki and whatever model backs it: the OpenAI API, the
Anthropic (Claude) API, or a Databricks serving endpoint. Engines declare
whether they're :meth:`available` (credentials/config present) so an agent
can pick the best reachable brain, and implement :meth:`complete`.

Messages use the portable ``[{"role": ..., "content": ...}]`` shape shared
by every provider; ``system`` is passed separately (Anthropic keeps it out
of the message list, and the others accept a leading system message).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

__all__ = ["Completion", "TokenEngine"]

# Sensible non-streaming output ceiling (see the claude-api guidance: ~16k
# keeps a non-streamed request under SDK HTTP timeouts).
DEFAULT_MAX_TOKENS = 16000


@dataclass
class Completion:
    """The result of a single engine turn."""

    text: str
    model: Optional[str] = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def __str__(self) -> str:
        return self.text


class TokenEngine(ABC):
    """A pluggable LLM backend Loki reasons with."""

    #: Engine name (``"openai"`` / ``"claude"`` / ``"databricks"``).
    name: ClassVar[str]
    #: Default model id when the caller doesn't pass one.
    default_model: ClassVar[Optional[str]] = None

    def __init__(self, *, model: Optional[str] = None) -> None:
        self.model = model or self.default_model

    @abstractmethod
    def available(self) -> bool:
        """True when this engine has the credentials/config to run."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **options: Any,
    ) -> Completion:
        """Run one chat completion and return a :class:`Completion`."""

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        **options: Any,
    ) -> str:
        """Convenience: complete a single user *prompt* → reply text."""
        return self.complete(
            [{"role": "user", "content": prompt}], system=system, **options
        ).text

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, available={self.available()})"
