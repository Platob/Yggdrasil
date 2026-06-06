"""Loki skill for the Unity Catalog **catalogs** service (``dbc.catalogs``).

The top of the Unity Catalog namespace: catalogs contain schemas contain
tables/volumes/functions. This is the "what data is here?" entry point — it
fans out over visible catalogs and, given one, lists its schemas. Pure UC REST
API: no SQL warehouse required.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, names

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksCatalogsSkill"]


@register
class DatabricksCatalogsSkill(DatabricksServiceSkill):
    """Navigate Unity Catalog — list catalogs, or the schemas within one."""

    name = "databricks-catalogs"
    description = "List Unity Catalog catalogs, or the schemas in a catalog (no warehouse needed)."
    preprompt = (
        "You navigate Unity Catalog through dbc.catalogs (REST, no warehouse). "
        "Catalogs → schemas → tables/volumes/functions. Start here to discover "
        "what data exists before querying; use three-level names downstream."
    )

    def run(self, agent: "Loki", *, catalog: Optional[str] = None, **_: Any) -> dict[str, Any]:
        client = self._client(agent)
        if catalog:
            schemas = [
                getattr(s, "schema_name", None) or getattr(s, "name", None) or str(s)
                for s in client.catalogs.catalog(catalog).schemas()
            ]
            return {"catalog": catalog, "schemas": schemas}
        return {"catalogs": names(client.catalogs.list_catalogs())}
