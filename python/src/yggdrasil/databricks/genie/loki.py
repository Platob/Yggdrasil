"""Loki skill for the **Genie** service (``dbc.genie``).

AI/BI Genie answers natural-language questions over a curated **space** of
tables — returning prose, the SQL it ran, and the result rows (Tabular). When
no space is named the skill reasons against the first space the user can reach.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, tabular

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["GenieSkill"]


@register
class GenieSkill(DatabricksServiceSkill):
    """Ask a Databricks AI/BI Genie space a question (text + SQL + rows)."""

    name = "genie"
    description = "Ask a Databricks AI/BI Genie space a question (text + SQL + rows)."
    preprompt = (
        "You ask AI/BI Genie via dbc.genie.space(id).ask(question). Genie is "
        "scoped to a curated space of tables; ask about that data only. The "
        "answer carries prose, the SQL it ran, and rows (Tabular)."
    )

    def run(self, agent: "Loki", *, question: str, space: Optional[str] = None,
            rows: bool = False, **_: Any) -> dict[str, Any]:
        client = self._client(agent)
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
            out["rows"] = tabular(answer)
        return out
