"""Loki working memory — a short, self-compressing session context.

A continuous development session can't keep every turn verbatim — the context
grows, tokens balloon, and the model drifts. :class:`LokiMemory` keeps the
**recent** turns verbatim and folds everything older into a compact
**synthesis**: when the raw context crosses a size threshold, the agent
summarizes the old turns (decisions, facts, paths, identifiers, open threads)
into a token-efficient memory another LLM can pick up cleanly, then drops the
raw turns. The result is a bounded, scalable context that stays precise.

It persists to the session's ``memory/`` dir, so a session's context survives
across process restarts and is reusable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .agent import Loki

__all__ = ["LokiMemory"]

#: Synthesis prompt — compress for *reuse by an LLM*, not for a human reader.
_COMPRESS_SYSTEM = (
    "You compress a development session's context for another LLM to resume "
    "from. Preserve decisions, concrete facts, file paths, identifiers, code "
    "signatures, and unresolved threads; drop greetings, filler, and restated "
    "context. Output ONLY the compact memory."
)


class LokiMemory:
    """Recent turns + a rolling synthesis of older context, auto-compressed."""

    def __init__(
        self,
        path: "str | Path | None" = None,
        *,
        keep_recent: int = 8,
        compress_chars: int = 6000,
    ) -> None:
        self.path = Path(path) if path else None
        self.keep_recent = keep_recent
        self.compress_chars = compress_chars
        self.synthesis: str = ""
        self.turns: list[dict[str, str]] = []
        if self.path is not None and self.path.exists():
            self.load()

    # -- accumulation ------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """Append a turn (``role`` is ``"user"`` / ``"assistant"``)."""
        if content:
            self.turns.append({"role": role, "content": content})
            self.save()

    def system_context(self) -> Optional[str]:
        """The memory rendered as a system note for the next reasoning call.

        Synthesis first (the long tail, compressed), then the recent turns
        verbatim — bounded and token-efficient. ``None`` when empty.
        """
        if not self.synthesis and not self.turns:
            return None
        parts: list[str] = []
        if self.synthesis:
            parts.append("Session memory (synthesized):\n" + self.synthesis)
        recent = self.turns[-self.keep_recent:]
        if recent:
            parts.append(
                "Recent turns:\n"
                + "\n".join(f"{t['role']}: {t['content']}" for t in recent)
            )
        return "\n\n".join(parts)

    def chars(self) -> int:
        return len(self.synthesis) + sum(len(t["content"]) for t in self.turns)

    # -- compression -------------------------------------------------------

    def maybe_compress(self, agent: "Loki", *, engine: Optional[str] = None) -> bool:
        """Fold older turns into the synthesis when context grows too large.

        Keeps the last :attr:`keep_recent` turns verbatim and summarizes the
        rest via the agent's fast engine. Returns whether it compressed.
        Reasoning failures (no engine) leave the raw turns intact.
        """
        if self.chars() < self.compress_chars or len(self.turns) <= self.keep_recent:
            return False
        old = self.turns[: -self.keep_recent]
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in old)
        prompt = (
            "Fold these older turns into the running session memory. Merge with "
            "the existing synthesis; keep it compact.\n\n"
            f"Existing synthesis:\n{self.synthesis or '(none)'}\n\n"
            f"Older turns:\n{transcript}"
        )
        try:
            self.synthesis = agent.reason(
                prompt, engine=engine, tier="fast", system=_COMPRESS_SYSTEM
            ).strip()
        except Exception:
            return False
        self.turns = self.turns[-self.keep_recent:]
        self.save()
        return True

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"synthesis": self.synthesis, "turns": self.turns}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2), "utf-8")

    def load(self) -> None:
        data = json.loads(self.path.read_text("utf-8"))
        self.synthesis = data.get("synthesis", "")
        self.turns = list(data.get("turns", []))
