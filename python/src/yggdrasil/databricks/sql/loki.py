"""Loki skill for the Databricks **SQL** service (``dbc.sql``).

``dbc.sql`` runs SQL on a serverless SQL warehouse and returns a statement
result that is **Tabular** — Arrow under the hood, convertible with
``to_polars`` / ``to_pylist`` / ``to_arrow``. This is the workhorse for reading
and transforming Unity Catalog data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, tabular

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksSQLSkill"]


@register
class DatabricksSQLSkill(DatabricksServiceSkill):
    """Execute a SQL query on a Databricks SQL warehouse → rows (a Tabular result)."""

    name = "databricks-sql"
    description = "Execute a SQL query on a serverless Databricks SQL warehouse → rows."
    preprompt = (
        "You write Databricks SQL run via dbc.sql.execute(query). Results are a "
        "Tabular (Arrow) — aggregate, filter, and LIMIT on the warehouse rather "
        "than pulling whole tables to the client; reference objects as "
        "catalog.schema.table; convert rows with to_polars()/to_pylist()."
    )

    def run(self, agent: "Loki", *, query: str, rows: bool = True, **_: Any) -> dict[str, Any]:
        result = self._client(agent).sql.execute(query)
        out: dict[str, Any] = {"query": query, "statement_id": getattr(result, "statement_id", None)}
        if rows:
            frame = tabular(result)
            out["rows"] = frame
            out["row_count"] = getattr(frame, "height", None)
        return out
