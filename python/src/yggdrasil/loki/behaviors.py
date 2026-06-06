"""Built-in Loki behaviors.

The seed of the behavior catalog — currently the Databricks **Genie**
behavior, which shows the pattern: guard on a detected backend, then drive
a Databricks service endpoint through Loki's token provider
(``agent.databricks``). Replication, inter-agent messaging, HTTP ingestion
and serving land here next.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .behavior import LokiBehavior, register

if TYPE_CHECKING:
    from .agent import Loki

__all__ = ["GenieBehavior"]


@register
class GenieBehavior(LokiBehavior):
    """Ask a Databricks Genie space a question and return its answer."""

    name = "genie"
    description = "Ask a Databricks AI/BI Genie space a question (text + SQL + rows)."
    requires = "databricks"

    def run(
        self,
        agent: "Loki",
        *,
        question: str,
        space: Optional[str] = None,
        rows: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        client = agent.databricks
        if client is None:  # available() already guards, belt-and-suspenders
            raise RuntimeError("no Databricks session")

        # Autonomy: when no space is named, reason against the first space the
        # current user can reach.
        if space is None:
            spaces = client.genie.spaces()
            if not spaces:
                raise RuntimeError("no Genie spaces are accessible to this user")
            target = spaces[0]
        else:
            target = client.genie.space(space)

        answer = target.ask(question)
        out: dict[str, Any] = {
            "space_id": target.space_id,
            "conversation_id": answer.conversation_id,
            "text": answer.text,
            "query": answer.query,
            "statement_id": answer.statement_id,
        }
        if rows and answer.query:
            out["rows"] = answer.to_polars()
        return out
