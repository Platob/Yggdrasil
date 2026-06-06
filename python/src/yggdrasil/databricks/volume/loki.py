"""Loki skill for the Unity Catalog **volumes** service (``dbc.volumes``).

A UC volume is governed file storage under ``catalog.schema`` (mounted at
``/Volumes/<catalog>/<schema>/<volume>/…``) — the right home for files
(Parquet/CSV/images) that yggdrasil reads/writes through its io handlers and
the ``dbfs:``/Volumes path abstraction.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, names

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksVolumesSkill"]


@register
class DatabricksVolumesSkill(DatabricksServiceSkill):
    """List Unity Catalog volumes (optionally within a catalog.schema)."""

    name = "databricks-volumes"
    description = "List Unity Catalog volumes (governed file storage under catalog.schema)."
    preprompt = (
        "You list UC volumes via dbc.volumes — governed file storage at "
        "/Volumes/<catalog>/<schema>/<volume>. Read/write files there through "
        "yggdrasil's io handlers (IO.from_) and the Volumes path, not raw SDK."
    )

    def run(self, agent: "Loki", *, catalog: Optional[str] = None,
            schema: Optional[str] = None, **_: Any) -> dict[str, Any]:
        # catalog_name / schema_name are keyword-only and default to the scope.
        vols = self._client(agent).volumes.list(catalog_name=catalog, schema_name=schema)
        return {"catalog": catalog, "schema": schema, "volumes": names(vols)}
