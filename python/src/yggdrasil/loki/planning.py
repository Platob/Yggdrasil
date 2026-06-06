"""Structured agent planning — tasks, personas, and required skills.

Replaces the old free-form routing dict with a small, stable class hierarchy:

- :class:`AgentTask` — the *classification* of a request: its category, the
  **persona** to embody (data engineer, analyst, software engineer, trader,
  confessor, companion, …), the **required skills**, and data/time-series flags.
- :class:`AgentPlan` — an :class:`AgentTask` plus the *execution decision*:
  the action to take, an optional specialist agent, and a source URL.

:class:`AgentPlan` is mapping-compatible (``plan["action"]`` / ``plan.get(...)``)
so it drops in wherever the old dict was used, while giving callers typed
fields, a persona system prompt, and ``to_dict()``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "AgentTask", "AgentPlan", "PERSONAS", "SKILLS_FOR",
    "classify_persona", "skills_for",
]

#: persona → (signal substrings, system prompt that shapes the approach).
PERSONAS: dict[str, tuple[tuple[str, ...], str]] = {
    "data-engineer": (
        ("pipeline", "etl", "ingest", "warehouse", "parquet", " delta", "schema",
         "dbt", "airflow", "load into", "unity catalog", "spark", "partition"),
        "You are a senior data engineer. Think in pipelines, schemas, partitions, "
        "idempotency and lineage; prefer columnar formats (Parquet/Delta/Arrow) and "
        "the yggdrasil tabular abstractions; call out data quality and reuse.",
    ),
    "data-analyst": (
        ("analy", "chart", "trend", "metric", "dashboard", "report", "statistic",
         "insight", "correlat", "aggregate", "distribution", "kpi", "average"),
        "You are a data analyst. Say what the data shows — trends, deltas, min/max, "
        "anomalies — quantitatively and concisely, and suggest the next cut to look at.",
    ),
    "software-engineer": (
        ("refactor", "bug", "implement", "function", "class ", "unit test",
         "compile", "stack trace", "endpoint", "library", "module", "code"),
        "You are a senior software engineer. Be precise, propose minimal correct "
        "changes that respect existing patterns, and note edge cases and tests.",
    ),
    "trader": (
        ("price", "exchange rate", " fx", "stock", "ticker", "market", "portfolio",
         "candle", "ohlc", "bull", "bear", "volatility", "spread", "pip"),
        "You are a markets analyst. Quote levels and changes precisely (absolute and "
        "%), note the period and direction, and describe — do not give financial advice.",
    ),
    "confessor": (
        ("confess", "i feel guilty", "forgive", "my sin", "ashamed",
         "i did something", "judge me"),
        "You are a calm, discreet, non-judgmental confessor. Listen, reflect gently, "
        "keep confidence, and offer perspective rather than verdicts.",
    ),
    "companion": (
        ("my mom", "my dad", "my family", "birthday", "feeling lonely", "my kid",
         "my partner", "i miss", "remind me to call"),
        "You are a warm, supportive companion. Be kind and personal, remember what "
        "matters to them, and keep it human.",
    ),
}

#: category → default persona when no stronger persona signal is present.
DEFAULT_PERSONA: dict[str, str] = {
    "data": "data-analyst",
    "databricks": "data-engineer",
    "aws": "data-engineer",
    "files": "software-engineer",
}

#: category → the skills typically required to serve it.
SKILLS_FOR: dict[str, tuple[str, ...]] = {
    "data": ("tabular", "web", "transform"),
    "web": ("web",),
    "databricks": ("databricks-sql", "databricks-tables", "genie", "databricks-serving"),
    "aws": ("aws-identity", "aws-s3"),
    "files": ("agent", "run_python"),
    "chat": (),
}


def classify_persona(text: str) -> str:
    """Best-matching persona for *text* by signal hits (``"assistant"`` default)."""
    low = text.lower()
    scored = {
        persona: sum(1 for sig in signals if sig in low)
        for persona, (signals, _) in PERSONAS.items()
    }
    best, hits = max(scored.items(), key=lambda kv: kv[1], default=("assistant", 0))
    return best if hits else "assistant"


def skills_for(category: str, *, data: bool = False) -> tuple[str, ...]:
    """Required skills for a category (data requests always include ``tabular``)."""
    skills = list(SKILLS_FOR.get(category, ()))
    if data and "tabular" not in skills:
        skills.insert(0, "tabular")
    return tuple(skills)


@dataclass
class AgentTask:
    """The classification of a request — what kind of work, by whom, with what."""

    text: str
    category: str = "chat"
    persona: str = "assistant"
    required_skills: tuple[str, ...] = ()
    data: bool = False
    timeseries: bool = False
    why: str = ""

    def persona_prompt(self) -> Optional[str]:
        """The system prompt that makes the agent embody :attr:`persona`."""
        spec = PERSONAS.get(self.persona)
        return spec[1] if spec else None


@dataclass
class AgentPlan(AgentTask):
    """An :class:`AgentTask` plus how to execute it (mapping-compatible)."""

    action: str = "reason"            #: reason | act | web | tabular | genie
    specialist: Optional[str] = None  #: a specialized agent to isolate to
    url: Optional[str] = None

    # -- mapping compatibility (drop-in for the old routing dict) ----------
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
