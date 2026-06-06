"""TokenEngine — the LLM reasoning contract Loki agents run on.

A :class:`TokenEngine` turns a chat message list into a completion. It's the
single seam between Loki and whatever model backs it: the OpenAI API, the
Anthropic (Claude) API, or a Databricks serving endpoint. Engines declare
whether they're :meth:`available` (credentials/config present) so an agent
can pick the best reachable brain, and implement :meth:`complete`.

Messages use the portable ``[{"role": ..., "content": ...}]`` shape shared
by every provider; ``system`` is passed separately (Anthropic keeps it out
of the message list, and the others accept a leading system message).

**Adaptive model selection.** Each engine declares a small :attr:`MODELS`
tier map — a ``"fast"`` model and a ``"deep"`` (more capable) one. When the
caller pins neither a model nor a tier, the engine **adapts**: light, short
requests resolve to the fast model; long or reasoning-heavy ones resolve to
the deep model (:meth:`choose_tier`). Pinning a ``model=`` always wins, and
passing ``tier="deep"`` / ``"fast"`` forces the choice — adaptivity is only
the default, never an override of an explicit decision.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

__all__ = ["Completion", "TokenEngine"]

# Sensible non-streaming output ceiling (see the claude-api guidance: ~16k
# keeps a non-streamed request under SDK HTTP timeouts).
DEFAULT_MAX_TOKENS = 16000

#: At or above this many characters of message content, the adaptive default
#: reaches for the capable ("deep") tier instead of the fast one.
ADAPTIVE_DEEP_CHARS = 2000
#: Substrings that mark a request as reasoning-heavy — pick the deep tier
#: regardless of length when any appears in the prompt/system.
ADAPTIVE_DEEP_SIGNALS = (
    "refactor", "debug", "prove", "analy", "architect", "design", "plan",
    "optimi", "root cause", "trace", "step by step", "reason", "why ",
    "implement", "derive", "diagnos",
)


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
    #: The model shown / used when nothing adapts — the engine's capable tier.
    default_model: ClassVar[Optional[str]] = None
    #: Tier → model id, fast → capable. Adaptive selection picks among these
    #: when the caller pins no model/tier. Engines override.
    MODELS: ClassVar[dict[str, str]] = {}

    def __init__(self, *, model: Optional[str] = None, tier: Optional[str] = None) -> None:
        #: Explicit model pin (``None`` → resolve adaptively per request).
        self.model = model
        #: Forced tier (``None`` → choose adaptively per request).
        self.tier = tier

    # -- model resolution --------------------------------------------------

    def choose_tier(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        system: Optional[str] = None,
    ) -> str:
        """Adaptive tier for this request: ``"deep"`` or ``"fast"``.

        Sizes on the *message* content (the actual work, not the fixed system
        boilerplate) and scans both message and system text for reasoning
        signals. Long or signalled requests get the deep tier; the rest stay
        fast. Override for a smarter policy.
        """
        parts = [
            m["content"] for m in (messages or [])
            if isinstance(m.get("content"), str)
        ]
        size = sum(len(p) for p in parts)
        blob = (" ".join(parts) + " " + (system or "")).lower()
        if size >= ADAPTIVE_DEEP_CHARS or any(s in blob for s in ADAPTIVE_DEEP_SIGNALS):
            return "deep"
        return "fast"

    def resolve_model(
        self,
        *,
        messages: Optional[list[dict[str, Any]]] = None,
        system: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> Optional[str]:
        """The model id to use for this request.

        An explicit ``self.model`` pin wins. Otherwise a forced tier (arg or
        ``self.tier``) selects from :attr:`MODELS`; with neither, the tier is
        chosen adaptively. Falls back to :attr:`default_model` when the tier
        isn't in the map.
        """
        if self.model:
            return self.model
        tier = tier or self.tier or self.choose_tier(messages, system)
        return self.MODELS.get(tier) or self.default_model

    @property
    def model_label(self) -> str:
        """Human label for status output — the pin, or the adaptive ceiling."""
        if self.model:
            return self.model
        if self.MODELS:
            return f"{self.MODELS.get('deep', self.default_model)} (adaptive)"
        return str(self.default_model)

    # -- contract ----------------------------------------------------------

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
        tier: Optional[str] = None,
        **options: Any,
    ) -> Completion:
        """Run one chat completion and return a :class:`Completion`.

        ``tier`` forces ``"fast"`` / ``"deep"`` model selection for this call;
        ``None`` (the default) lets the engine adapt.
        """

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        tier: Optional[str] = None,
        **options: Any,
    ) -> str:
        """Convenience: complete a single user *prompt* → reply text."""
        return self.complete(
            [{"role": "user", "content": prompt}], system=system, tier=tier, **options
        ).text

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model_label!r}, available={self.available()})"
