"""TokenEngine — the LLM reasoning contract Loki agents run on.

A :class:`TokenEngine` turns a chat message list into a completion. It's the
single seam between Loki and whatever model backs it: the OpenAI API, the
Anthropic (Claude) API, a Databricks serving endpoint, or another Loki
agent. Engines declare their :class:`EngineType`, whether they're
:meth:`available`, and adapt the model they use to the task
:class:`~yggdrasil.loki.model.Complexity`.

Messages use the portable ``[{"role": ..., "content": ...}]`` shape shared
by every provider; ``system`` is passed separately (Anthropic keeps it out
of the message list, and the others accept a leading system message).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Optional, Union

from .model import Complexity, Provider, select_model

__all__ = ["EngineType", "Completion", "AgentResponse", "TokenEngine"]

# Sensible non-streaming output ceiling (see the claude-api guidance: ~16k
# keeps a non-streamed request under SDK HTTP timeouts).
DEFAULT_MAX_TOKENS = 16000


class EngineType(str, Enum):
    """The kind of reasoning engine — for grouping and selection."""

    OPENAI = "openai"
    CLAUDE = "claude"
    DATABRICKS = "databricks"
    LOKI_AGENT = "loki_agent"   # an engine backed by another Loki agent


@dataclass
class Completion:
    """The raw result of a single engine turn."""

    text: str
    model: Optional[str] = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def __str__(self) -> str:
        return self.text


def _as_tabular(obj: Any) -> Any:
    """Return *obj* if it's a tabular frame (Arrow / Polars / pandas), else None."""
    if obj is None:
        return None
    root = type(obj).__module__.split(".", 1)[0]
    name = type(obj).__name__
    if root == "pyarrow" and name in ("Table", "RecordBatch"):
        return obj
    if root in ("polars", "pandas") and name == "DataFrame":
        return obj
    return None


@dataclass
class AgentResponse:
    """An agent's answer — narrative ``text`` plus, when the result is
    tabular-like, an optional ``tabular`` frame (Arrow / Polars / pandas)."""

    text: str = ""
    data: Any = None
    tabular: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_tabular(self) -> bool:
        return self.tabular is not None

    @classmethod
    def from_(
        cls,
        result: Any,
        *,
        text: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> AgentResponse:
        """Coerce any behavior/engine result into an :class:`AgentResponse`.

        Detects a tabular frame and attaches it; carries the raw value as
        ``data``; uses *result* as ``text`` when it's a string.
        """
        if isinstance(result, AgentResponse):
            return result
        tabular = _as_tabular(result)
        if tabular is not None:
            return cls(text=text or "", data=result, tabular=tabular, meta=meta or {})
        if isinstance(result, str):
            return cls(text=result, data=result, meta=meta or {})
        return cls(text=text or "", data=result, meta=meta or {})

    def to_polars(self):
        """The tabular result as a Polars DataFrame (raises when not tabular)."""
        if self.tabular is None:
            raise ValueError("this AgentResponse has no tabular result")
        import polars as pl

        if type(self.tabular).__module__.split(".", 1)[0] == "polars":
            return self.tabular
        return pl.from_arrow(self.tabular) if hasattr(self.tabular, "to_batches") else pl.from_pandas(self.tabular)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "is_tabular": self.is_tabular,
            "data": None if self.is_tabular else self.data,
            "meta": self.meta,
        }

    def __str__(self) -> str:
        return self.text


class TokenEngine(ABC):
    """A pluggable LLM backend Loki reasons with."""

    #: Engine name (``"openai"`` / ``"claude"`` / ``"databricks"``).
    name: ClassVar[str]
    #: Engine kind, for grouping/selection.
    type: ClassVar[EngineType]
    #: Model provider — drives complexity-adaptive model selection.
    provider: ClassVar[Optional[Provider]] = None
    #: Default model id when neither a per-call model nor complexity is given.
    default_model: ClassVar[Optional[str]] = None

    def __init__(self, *, model: Optional[str] = None) -> None:
        self.model = model or self.default_model

    @abstractmethod
    def available(self) -> bool:
        """True when this engine has the credentials/config to run."""

    def resolve_model(self, complexity: Union[Complexity, int, str, None] = None) -> Optional[str]:
        """Pick the model for this turn — adapts to *complexity* when given."""
        if complexity is None or self.provider is None:
            return self.model
        spec = select_model(self.provider, complexity, default=self.model)
        return spec.id if spec is not None else self.model

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        complexity: Union[Complexity, int, str, None] = None,
        **options: Any,
    ) -> Completion:
        """Run one chat completion and return a :class:`Completion`."""

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        complexity: Union[Complexity, int, str, None] = None,
        **options: Any,
    ) -> str:
        """Convenience: complete a single user *prompt* → reply text."""
        return self.complete(
            [{"role": "user", "content": prompt}],
            system=system, complexity=complexity, **options,
        ).text

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, available={self.available()})"
