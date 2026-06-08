"""Loki skill for the **compute / clusters** service (``dbc.compute``).

All-purpose and job clusters — the general (non-SQL) compute. For yggdrasil
work prefer **serverless** for inner Databricks I/O and a **single-user
cluster** only for external-resource access. This skill lists the clusters.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, names

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksClustersSkill"]


@register
class DatabricksClustersSkill(DatabricksServiceSkill):
    """List compute clusters, or start / stop / restart one by name or id."""

    name = "databricks-clusters"
    description = "List Databricks compute clusters, or start/stop/restart one."
    preprompt = (
        "You manage compute clusters via dbc.compute.clusters: list them, or "
        "start/stop/restart one by name or id. Prefer serverless for inner I/O "
        "and a single-user cluster only for external access; starting a cluster "
        "is billable, terminating it is stateful — be explicit."
    )

    def run(self, agent: "Loki", *, op: str = "list",
            cluster: Optional[str] = None, **_: Any) -> dict[str, Any]:
        clusters = self._client(agent).compute.clusters
        if op == "list":
            return {"clusters": names(clusters.list())}
        if not cluster:
            raise ValueError(f"{op} needs cluster= (a cluster name or id)")
        target = clusters.find_cluster(cluster)
        if target is None:
            return {"cluster": cluster, "found": False}
        if op == "start":
            target.start()
            return {"started": cluster}
        if op in ("stop", "terminate"):
            target.delete()                       # terminate (cluster stays defined)
            return {"stopped": cluster}
        if op == "restart":
            target.restart()
            return {"restarted": cluster}
        raise ValueError(f"unknown op {op!r}; use list/start/stop/restart")
