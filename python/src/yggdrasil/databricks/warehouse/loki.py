"""Loki skill for the **SQL warehouses** service (``dbc.warehouses``).

A SQL warehouse is the compute that runs ``dbc.sql`` queries. Prefer a
**serverless** warehouse for inner I/O. This skill lists the warehouses the
agent can reach (their names / ids / state).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, names

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksWarehousesSkill"]


@register
class DatabricksWarehousesSkill(DatabricksServiceSkill):
    """List SQL warehouses, or start / stop one by name or id."""

    name = "databricks-warehouses"
    description = "List Databricks SQL warehouses, or start/stop one (the compute behind dbc.sql)."
    preprompt = (
        "You manage SQL warehouses via dbc.warehouses — the compute behind "
        "dbc.sql: list them, or start/stop one by name or id. Prefer a "
        "serverless warehouse for inner I/O; starting one is billable."
    )

    def run(self, agent: "Loki", *, op: str = "list",
            warehouse: Optional[str] = None, **_: Any) -> dict[str, Any]:
        whs = self._client(agent).warehouses
        if op == "list":
            return {"warehouses": names(whs.list_warehouses())}
        if not warehouse:
            raise ValueError(f"{op} needs warehouse= (a name or id)")
        target = whs.find_warehouse(warehouse)
        if target is None:
            return {"warehouse": warehouse, "found": False}
        if op == "start":
            target.start()
            return {"started": warehouse}
        if op in ("stop", "terminate"):
            target.stop()
            return {"stopped": warehouse}
        raise ValueError(f"unknown op {op!r}; use list/start/stop")
