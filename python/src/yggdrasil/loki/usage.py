"""Token accounting — per-model usage, USD pricing, and a spend budget.

Every :class:`~yggdrasil.loki.engine.TokenEngine` records what it spends into
the process-global :data:`METER` after each completion: input/output tokens
keyed by ``(engine, model)``. The meter rolls those up per model and globally,
prices them against :data:`PRICING` (USD per million tokens — defaults set
here, override per workspace), and enforces an optional **budget** so an
interactive session stops and asks the user to raise the cap rather than
spending without bound.

    from yggdrasil.loki.usage import METER

    METER.set_limit(50_000)          # cap total tokens
    ...                              # engines record automatically
    METER.total().total_tokens       # global tokens so far
    METER.total_cost                 # global USD so far
    METER.check_budget()             # raises TokenBudgetExceeded when over
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from yggdrasil.exceptions import TokenBudgetExceeded

__all__ = [
    "ModelPricing",
    "ModelUsage",
    "TokenMeter",
    "PRICING",
    "METER",
    "estimate_tokens",
    "price_for",
]


@dataclass(frozen=True)
class ModelPricing:
    """USD price per **million** tokens, split input/output."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_usd_per_mtok
            + output_tokens * self.output_usd_per_mtok
        ) / 1_000_000


#: Default USD/1M-token pricing, keyed by ``(engine, model)``. A ``(engine,
#: "*")`` row is the per-engine fallback; :data:`DEFAULT_PRICING` is the last
#: resort. These are sensible defaults for now — retune per workspace/contract.
PRICING: dict[tuple[str, str], ModelPricing] = {
    # Anthropic (see the claude-api pricing reference).
    ("claude", "claude-opus-4-8"): ModelPricing(5.0, 25.0),
    ("claude", "claude-opus-4-7"): ModelPricing(5.0, 25.0),
    ("claude", "claude-opus-4-6"): ModelPricing(5.0, 25.0),
    ("claude", "claude-sonnet-4-6"): ModelPricing(3.0, 15.0),
    ("claude", "claude-haiku-4-5"): ModelPricing(1.0, 5.0),
    ("claude", "*"): ModelPricing(5.0, 25.0),
    # OpenAI (public list prices).
    ("openai", "gpt-4o"): ModelPricing(2.5, 10.0),
    ("openai", "gpt-4o-mini"): ModelPricing(0.15, 0.60),
    ("openai", "*"): ModelPricing(2.5, 10.0),
    # Databricks serving — nominal placeholder until wired to workspace DBU.
    ("databricks", "*"): ModelPricing(0.20, 0.60),
}

#: Last-resort price when neither the exact model nor the engine is known.
DEFAULT_PRICING = ModelPricing(1.0, 3.0)


def price_for(engine: str, model: str) -> ModelPricing:
    """Resolve pricing: exact ``(engine, model)`` → ``(engine, "*")`` → default."""
    return (
        PRICING.get((engine, model))
        or PRICING.get((engine, "*"))
        or DEFAULT_PRICING
    )


def estimate_tokens(text: str) -> int:
    """Cheap offline token estimate (~4 chars/token) for when a provider
    response carries no usage block."""
    return max(1, len(text) // 4) if text else 0


@dataclass
class ModelUsage:
    """Running consumption for one ``(engine, model)`` pair (or the global roll-up)."""

    engine: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return price_for(self.engine, self.model).cost(
            self.input_tokens, self.output_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class TokenMeter:
    """Accumulates per-model token usage and enforces a token budget.

    Recording never raises — it only counts. Enforcement is explicit:
    :meth:`check_budget` raises :class:`TokenBudgetExceeded` when a limit is
    set and crossed, so the caller (the REPL) can stop and offer to raise it.
    """

    #: Default budget the interactive CLI starts with (tokens). Not enforced
    #: until a limit is actually set via :meth:`set_limit`.
    DEFAULT_LIMIT = 50_000
    #: How much :meth:`raise_limit` adds each step.
    DEFAULT_STEP = 25_000

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], ModelUsage] = {}
        self.limit: Optional[int] = None
        self.step: int = self.DEFAULT_STEP

    # -- recording ---------------------------------------------------------

    def record(
        self, engine: str, model: str, input_tokens: int, output_tokens: int
    ) -> ModelUsage:
        """Add one completion's tokens to the ``(engine, model)`` row."""
        key = (engine, model or "?")
        row = self._rows.get(key)
        if row is None:
            row = self._rows[key] = ModelUsage(engine=engine, model=key[1])
        row.calls += 1
        row.input_tokens += max(0, int(input_tokens))
        row.output_tokens += max(0, int(output_tokens))
        return row

    # -- reporting ---------------------------------------------------------

    def rows(self) -> list[ModelUsage]:
        """Per-model usage rows, busiest first."""
        return sorted(self._rows.values(), key=lambda r: -r.total_tokens)

    def rows_for(self, engine: str) -> list[ModelUsage]:
        return [r for r in self.rows() if r.engine == engine]

    def total(self) -> ModelUsage:
        """The global roll-up across every engine and model."""
        agg = ModelUsage(engine="*", model="global")
        for r in self._rows.values():
            agg.calls += r.calls
            agg.input_tokens += r.input_tokens
            agg.output_tokens += r.output_tokens
        return agg

    @property
    def total_tokens(self) -> int:
        return self.total().total_tokens

    @property
    def total_cost(self) -> float:
        """Global USD — summed per row so each is priced at its own rate."""
        return sum(r.cost_usd for r in self._rows.values())

    # -- budget ------------------------------------------------------------

    def set_limit(self, limit: Optional[int]) -> None:
        """Set (or clear, with ``None``) the total-token budget."""
        self.limit = None if limit is None else max(0, int(limit))

    def raise_limit(self, by: Optional[int] = None) -> int:
        """Bump the budget by *by* (or one :attr:`step`); returns the new limit."""
        base = self.limit if self.limit is not None else self.total_tokens
        self.limit = base + (by if by is not None else self.step)
        return self.limit

    def remaining(self) -> Optional[int]:
        return None if self.limit is None else self.limit - self.total_tokens

    def over_budget(self) -> bool:
        return self.limit is not None and self.total_tokens >= self.limit

    def check_budget(self) -> None:
        """Raise :class:`TokenBudgetExceeded` if the budget is set and crossed."""
        if self.over_budget():
            raise TokenBudgetExceeded(self.total_tokens, self.limit)

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self._rows.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "rows": [r.to_dict() for r in self.rows()],
            "total": self.total().to_dict(),
            "total_cost_usd": round(self.total_cost, 6),
            "limit": self.limit,
            "remaining": self.remaining(),
        }


#: Process-global meter every engine records into.
METER = TokenMeter()
