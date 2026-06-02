"""Self-driving Genie agent.

:class:`GenieAgent` turns a single goal into an autonomous, multi-turn
Genie investigation. It opens a conversation, reads each answer, and —
when Genie responds with clarifying *suggested questions* instead of a
concrete data answer — picks one and keeps going on its own until it
lands a query-backed answer or exhausts its turn budget. Every step
reuses the project's own simplifications: :class:`GenieAnswer` (which
materialises results through the shared
:class:`~yggdrasil.databricks.warehouse.statement.WarehouseStatementResult`
→ Arrow / polars / pandas), the :class:`~yggdrasil.dataclasses.WaitingConfig`
budget, and the live :class:`GenieConversation` thread.

::

    run = client.genie.agent(space_id="01ef…").run("why did Q3 revenue dip?")
    print(run.summary())          # transcript of every turn
    print(run.sql)                # SQL behind the final data answer
    df = run.to_polars()          # the final result as a DataFrame
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.dataclasses import WaitingConfigArg

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import polars as pl
    import pyarrow as pa

    from .resources import GenieAnswer, GenieConversation, GenieSpace


__all__ = ["GenieAgent", "AgentRun", "AgentTurn"]


LOGGER = logging.getLogger(__name__)

#: How many conversational turns the agent will drive before stopping,
#: counting the opening question. Four is enough to resolve a couple of
#: rounds of Genie clarifications without looping forever on a vague goal.
DEFAULT_MAX_TURNS = 4


@dataclass
class AgentTurn:
    """One question/answer step in an :class:`AgentRun`."""

    question: str
    answer: "GenieAnswer"
    #: ``True`` when the agent chose this follow-up itself (vs the caller's
    #: opening goal).
    autonomous: bool = False


@dataclass
class AgentRun:
    """The full transcript of a :meth:`GenieAgent.run` investigation."""

    space: "GenieSpace"
    goal: str
    conversation: "GenieConversation"
    turns: list[AgentTurn] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(goal={self.goal!r}, turns={len(self.turns)}, "
            f"has_data={self.answer is not None and self.answer.has_query})"
        )

    # ------------------------------------------------------------------ #
    # Final answer resolution
    # ------------------------------------------------------------------ #
    @property
    def answer(self) -> "Optional[GenieAnswer]":
        """The last answer in the run (the agent's final word)."""
        return self.turns[-1].answer if self.turns else None

    @property
    def data_answer(self) -> "Optional[GenieAnswer]":
        """The most recent answer that actually ran a query, if any."""
        for turn in reversed(self.turns):
            if turn.answer.has_query:
                return turn.answer
        return None

    @property
    def text(self) -> Optional[str]:
        ans = self.answer
        return ans.text if ans is not None else None

    @property
    def sql(self) -> Optional[str]:
        ans = self.data_answer
        return ans.sql if ans is not None else None

    def to_arrow(self) -> "Optional[pa.Table]":
        ans = self.data_answer
        return ans.to_arrow() if ans is not None else None

    def to_polars(self) -> "Optional[pl.DataFrame]":
        ans = self.data_answer
        return ans.to_polars() if ans is not None else None

    def to_pandas(self) -> "Optional[pd.DataFrame]":
        ans = self.data_answer
        return ans.to_pandas() if ans is not None else None

    def rows(self) -> list[dict[str, Any]]:
        ans = self.data_answer
        return ans.rows() if ans is not None else []

    # ------------------------------------------------------------------ #
    # Transcript
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        """Render the full investigation as a readable transcript."""
        lines: list[str] = [f"Goal: {self.goal}"]
        for i, turn in enumerate(self.turns, 1):
            tag = "agent" if turn.autonomous else "you"
            lines.append(f"\n[{i}] ({tag}) {turn.question}")
            ans = turn.answer
            if ans.text:
                lines.append(f"    {ans.text.strip()}")
            if ans.sql:
                lines.append(f"    SQL: {ans.sql.strip()}")
            if ans.failed and ans.error:
                lines.append(f"    ERROR: {ans.error}")
            elif not ans.text and ans.questions:
                joined = "; ".join(ans.questions)
                lines.append(f"    (suggested: {joined})")
        return "\n".join(lines)


class GenieAgent:
    """An autonomous driver over a single :class:`GenieSpace`."""

    def __init__(
        self,
        space: "GenieSpace",
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        follow_suggestions: bool = True,
        wait: WaitingConfigArg = None,
    ):
        self.space = space
        self.max_turns = max(1, int(max_turns))
        self.follow_suggestions = follow_suggestions
        self.wait = wait

    def __repr__(self) -> str:
        return f"{type(self).__name__}(space_id={self.space.space_id!r}, max_turns={self.max_turns})"

    @property
    def service(self):
        return self.space.service

    def run(self, goal: str, *, max_turns: Optional[int] = None) -> AgentRun:
        """Drive an autonomous investigation toward ``goal``.

        Opens a conversation with the goal, then keeps going on its own:
        whenever Genie answers with clarifying suggested questions rather
        than a concrete data answer, the agent picks the first suggestion
        and asks it — until it lands a query-backed answer, hits a failure,
        runs out of suggestions, or exhausts ``max_turns``.
        """
        budget = max(1, int(max_turns)) if max_turns is not None else self.max_turns
        LOGGER.debug("GenieAgent.run goal=%r (max_turns=%d) on %r", goal, budget, self.space)

        conv, answer = self.space.start_conversation(goal, wait=self.wait)
        run = AgentRun(space=self.space, goal=goal, conversation=conv)
        run.turns.append(AgentTurn(question=goal, answer=answer, autonomous=False))

        while len(run.turns) < budget:
            answer = run.turns[-1].answer
            # Stop on a concrete, query-backed answer or a terminal failure.
            if answer.has_query or answer.failed:
                break
            # No data answer — follow Genie's own suggested next question.
            questions = answer.questions
            if not (self.follow_suggestions and questions):
                break
            follow_up = questions[0]
            LOGGER.debug("GenieAgent following suggestion: %s", follow_up)
            next_answer = conv.ask(follow_up, wait=self.wait)
            run.turns.append(
                AgentTurn(question=follow_up, answer=next_answer, autonomous=True)
            )

        return run

    def ask(self, question: str) -> "GenieAnswer":
        """One-shot passthrough — ask without driving follow-ups."""
        return self.space.ask(question, wait=self.wait)
