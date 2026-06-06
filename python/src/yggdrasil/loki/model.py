"""TokenModel — the catalog of LLM models Loki can reason with.

Models are grouped by :class:`Provider` and by :class:`Complexity` (the tier
of task a model is the right tool for). :class:`TokenModel` enumerates the
built-ins; the catalog is **extensible** at runtime — :func:`register_model`
/ :func:`unregister_model` add or drop models, and :func:`select_model` picks
the right one for a provider + complexity so an engine can adapt to the task.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional, Union

__all__ = [
    "Provider",
    "Complexity",
    "ModelSpec",
    "TokenModel",
    "register_model",
    "unregister_model",
    "models",
    "select_model",
]


class Provider(str, Enum):
    """Where a model is served from."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DATABRICKS = "databricks"


class Complexity(IntEnum):
    """How hard a task is — drives which model tier to reach for."""

    LOW = 1      # classification, extraction, short replies → cheap/fast
    MEDIUM = 2   # summarization, routine reasoning → balanced
    HIGH = 3     # multi-step reasoning, agentic work → most capable

    @classmethod
    def from_(cls, value: Union[Complexity, int, str, None], default: Complexity = None) -> Complexity:
        if value is None:
            return default if default is not None else cls.MEDIUM
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(max(1, min(3, value)))
        return cls[str(value).upper()]


@dataclass(frozen=True)
class ModelSpec:
    """One model: its serving id, provider, and the complexity tier it fits."""

    id: str
    provider: Provider
    complexity: Complexity

    def __str__(self) -> str:
        return self.id


class TokenModel(Enum):
    """Built-in models, grouped by provider + complexity tier.

    Model ids are exact (see the claude-api reference for Claude strings).
    Extend the live catalog with :func:`register_model`.
    """

    GPT_4O_MINI = ModelSpec("gpt-4o-mini", Provider.OPENAI, Complexity.LOW)
    GPT_4O = ModelSpec("gpt-4o", Provider.OPENAI, Complexity.MEDIUM)
    GPT_5 = ModelSpec("gpt-5", Provider.OPENAI, Complexity.HIGH)

    CLAUDE_HAIKU_4_5 = ModelSpec("claude-haiku-4-5", Provider.ANTHROPIC, Complexity.LOW)
    CLAUDE_SONNET_4_6 = ModelSpec("claude-sonnet-4-6", Provider.ANTHROPIC, Complexity.MEDIUM)
    CLAUDE_OPUS_4_8 = ModelSpec("claude-opus-4-8", Provider.ANTHROPIC, Complexity.HIGH)

    DBX_LLAMA_3_3_70B = ModelSpec(
        "databricks-meta-llama-3-3-70b-instruct", Provider.DATABRICKS, Complexity.MEDIUM
    )
    DBX_LLAMA_3_1_405B = ModelSpec(
        "databricks-meta-llama-3-1-405b-instruct", Provider.DATABRICKS, Complexity.HIGH
    )
    DBX_GPT_OSS_20B = ModelSpec(
        "databricks-gpt-oss-20b", Provider.DATABRICKS, Complexity.LOW
    )

    @property
    def id(self) -> str:
        return self.value.id

    @property
    def provider(self) -> Provider:
        return self.value.provider

    @property
    def complexity(self) -> Complexity:
        return self.value.complexity


# Live, mutable catalog — seeded from the enum, then add/remove at runtime.
_REGISTRY: dict[str, ModelSpec] = {m.id: m.value for m in TokenModel}


def register_model(
    model: Union[ModelSpec, str],
    *,
    provider: Union[Provider, str, None] = None,
    complexity: Union[Complexity, int, str, None] = None,
) -> ModelSpec:
    """Add (or replace) a model in the catalog. Returns the stored spec."""
    if isinstance(model, ModelSpec):
        spec = model
    else:
        if provider is None or complexity is None:
            raise ValueError("register_model needs provider and complexity for a bare id")
        spec = ModelSpec(model, Provider(provider), Complexity.from_(complexity))
    _REGISTRY[spec.id] = spec
    return spec


def unregister_model(model: Union[ModelSpec, str]) -> None:
    """Drop a model from the catalog by id (no error if absent)."""
    _REGISTRY.pop(model.id if isinstance(model, ModelSpec) else model, None)


def models(
    provider: Union[Provider, str, None] = None,
    complexity: Union[Complexity, int, str, None] = None,
) -> list[ModelSpec]:
    """Catalog models, optionally filtered by provider and/or complexity."""
    prov = Provider(provider) if provider is not None else None
    comp = Complexity.from_(complexity) if complexity is not None else None
    out = [
        s for s in _REGISTRY.values()
        if (prov is None or s.provider == prov) and (comp is None or s.complexity == comp)
    ]
    return sorted(out, key=lambda s: (s.provider.value, s.complexity, s.id))


def select_model(
    provider: Union[Provider, str],
    complexity: Union[Complexity, int, str, None] = None,
    *,
    default: Optional[str] = None,
) -> Optional[ModelSpec]:
    """Pick the best model for *provider* at *complexity*.

    Prefers an exact-tier match, then the nearest tier at-or-above the
    request, then the strongest available below it. Returns ``None`` when the
    provider has no registered models (unless *default* is given and known).
    """
    prov = Provider(provider)
    want = Complexity.from_(complexity)
    candidates = [s for s in _REGISTRY.values() if s.provider == prov]
    if not candidates:
        return _REGISTRY.get(default) if default else None
    exact = [s for s in candidates if s.complexity == want]
    if exact:
        return exact[0]
    at_or_above = sorted((s for s in candidates if s.complexity >= want), key=lambda s: s.complexity)
    if at_or_above:
        return at_or_above[0]
    return max(candidates, key=lambda s: s.complexity)
