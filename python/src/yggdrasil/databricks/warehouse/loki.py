"""Loki skill for the **SQL warehouses** service (``dbc.warehouses``).

A SQL warehouse is the compute that runs ``dbc.sql`` queries. Prefer a
**serverless** warehouse for inner I/O. This skill lists the warehouses the
agent can reach (their names / ids / state).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, names

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksWarehousesSkill"]


@register
class DatabricksWarehousesSkill(DatabricksServiceSkill):
    """List the SQL warehouses reachable to the agent."""

    name = "databricks-warehouses"
    description = "List the Databricks SQL warehouses (the compute behind dbc.sql)."
    preprompt = (
        "You list SQL warehouses via dbc.warehouses — the compute behind "
        "dbc.sql. Prefer a serverless warehouse for inner I/O; surface name, "
        "id, and state so the user can pick one."
    )

    def run(self, agent: "Loki", **_: Any) -> dict[str, Any]:
        return {"warehouses": names(self._client(agent).warehouses.list_warehouses())}
