"""Self-driving Genie agent.

:class:`GenieAgent` turns a single goal into an autonomous, multi-turn
Genie investigation. It opens a conversation, reads each answer, and
decides the next move on its own until the goal is met or the turn budget
runs out. Two brains drive it:

- **Planner brain (fully autonomous)** — when a ``planner`` model is set,
  the agent asks a Databricks **Model Serving** LLM (e.g.
  ``databricks-claude-sonnet-4``) what to ask Genie next, given the goal
  and the transcript so far, or whether the goal is answered. This closes
  the loop end-to-end across two Databricks AI services: the serving LLM
  reasons, Genie executes the SQL, and the agent observes the result.
- **Heuristic brain (no LLM)** — with no planner, the agent follows
  Genie's own *suggested questions* until it lands a query-backed answer.

Every step reuses the project's simplifications: :class:`GenieAnswer`
(results → Arrow / polars / pandas), the
:class:`~yggdrasil.dataclasses.WaitingConfig` budget, and the live
:class:`GenieConversation` thread.

::

    # Fully autonomous: an LLM plans, Genie executes
    run = client.genie.agent(
        space_id="01ef…", planner="databricks-claude-sonnet-4",
    ).run("why did Q3 revenue dip?")
    print(run.summary())          # transcript of every turn
    print(run.sql)                # SQL behind the final data answer
    df = run.to_polars()          # the final result as a DataFrame
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

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

#: Default Model Serving endpoint used as the planner brain when the agent
#: is created with ``planner=True``. Override with ``$YGG_GENIE_PLANNER``.
DEFAULT_PLANNER_ENDPOINT = os.environ.get(
    "YGG_GENIE_PLANNER", "databricks-claude-sonnet-4",
)

#: Instruction handed to the planner LLM each turn.
_PLANNER_SYSTEM = (
    "You are the planner for a Databricks Genie analytics agent. Given a "
    "GOAL and the conversation so far (each turn is a question to Genie and "
    "its answer + SQL), decide the SINGLE most useful next question to ask "
    "Genie to make progress toward the goal. If the goal is already fully "
    "answered, reply with exactly DONE. Reply with ONLY the next question, "
    "or DONE — no preamble, no numbering, no quotes."
)

#: A planner is either a Model Serving endpoint name, ``True`` for the
#: default endpoint, or a callable ``(AgentRun) -> Optional[str]`` returning
#: the next question (or ``None`` to stop).
PlannerArg = Union[str, bool, Callable[["AgentRun"], Optional[str]], None]


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
        planner: PlannerArg = None,
        wait: WaitingConfigArg = None,
    ):
        self.space = space
        self.max_turns = max(1, int(max_turns))
        self.follow_suggestions = follow_suggestions
        # Normalise the planner: ``True`` → the default serving endpoint.
        self.planner: Union[str, Callable[["AgentRun"], Optional[str]], None]
        self.planner = DEFAULT_PLANNER_ENDPOINT if planner is True else (planner or None)
        self.wait = wait

    def __repr__(self) -> str:
        brain = f"planner={self.planner!r}" if self.planner else "heuristic"
        return (
            f"{type(self).__name__}(space_id={self.space.space_id!r}, "
            f"max_turns={self.max_turns}, {brain})"
        )

    @property
    def service(self):
        return self.space.service

    def run(self, goal: str, *, max_turns: Optional[int] = None) -> AgentRun:
        """Drive an autonomous investigation toward ``goal``.

        Opens a conversation with the goal, then keeps going on its own.
        With a ``planner`` an LLM decides each next question (and when the
        goal is met); without one the agent follows Genie's own suggested
        questions. Stops on the planner saying DONE, a query-backed answer
        (heuristic mode), a terminal failure, or ``max_turns``.
        """
        budget = max(1, int(max_turns)) if max_turns is not None else self.max_turns
        LOGGER.debug(
            "GenieAgent.run goal=%r (max_turns=%d, planner=%s) on %r",
            goal, budget, self.planner, self.space,
        )

        conv, answer = self.space.start_conversation(goal, wait=self.wait)
        run = AgentRun(space=self.space, goal=goal, conversation=conv)
        run.turns.append(AgentTurn(question=goal, answer=answer, autonomous=False))

        while len(run.turns) < budget:
            answer = run.turns[-1].answer
            if answer.failed:
                break
            follow_up = self._next_question(run)
            if not follow_up:
                break
            LOGGER.debug("GenieAgent next question: %s", follow_up)
            next_answer = conv.ask(follow_up, wait=self.wait)
            run.turns.append(
                AgentTurn(question=follow_up, answer=next_answer, autonomous=True)
            )

        return run

    def _next_question(self, run: AgentRun) -> Optional[str]:
        """Decide the next question to ask Genie, or ``None`` to stop.

        Planner brain when one is configured (LLM or callable); otherwise
        the heuristic — follow Genie's suggested questions, and stop once a
        query-backed answer is in hand.
        """
        if self.planner is not None:
            if callable(self.planner):
                return self.planner(run)
            return self._plan_with_llm(run)

        answer = run.turns[-1].answer
        if answer.has_query:
            return None
        questions = answer.questions
        if self.follow_suggestions and questions:
            return questions[0]
        return None

    def _plan_with_llm(self, run: AgentRun) -> Optional[str]:
        """Ask the planner Model Serving LLM for the next question (or DONE)."""
        endpoint = self.service.client.ai.serving.endpoint(str(self.planner))
        reply = endpoint.chat(
            [
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user", "content": f"GOAL: {run.goal}\n\n{run.summary()}"},
            ],
            max_tokens=120,
            temperature=0.0,
        ).text
        reply = (reply or "").strip()
        if not reply or reply.upper().startswith("DONE"):
            return None
        # Guard against a chatty model returning a one-line "Next: …".
        return reply.splitlines()[0].lstrip("-•* ").strip() or None

    def ask(self, question: str) -> "GenieAnswer":
        """One-shot passthrough — ask without driving follow-ups."""
        return self.space.ask(question, wait=self.wait)
