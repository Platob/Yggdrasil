"""Loki skill for the **compute / clusters** service (``dbc.compute``).

All-purpose and job clusters — the general (non-SQL) compute. For yggdrasil
work prefer **serverless** for inner Databricks I/O and a **single-user
cluster** only for external-resource access. This skill lists the clusters.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, names

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksClustersSkill"]


@register
class DatabricksClustersSkill(DatabricksServiceSkill):
    """List the compute clusters (all-purpose / job)."""

    name = "databricks-clusters"
    description = "List the Databricks compute clusters (dbc.compute.clusters)."
    preprompt = (
        "You list compute clusters via dbc.compute.clusters. Prefer serverless "
        "for inner I/O and a single-user cluster only for external-resource "
        "access; never a multi-node cluster for work serverless handles."
    )

    def run(self, agent: "Loki", **_: Any) -> dict[str, Any]:
        return {"clusters": names(self._client(agent).compute.clusters.list())}
