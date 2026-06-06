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
    description = "List / create / drop Unity Catalog catalogs, or list a catalog's schemas."
    preprompt = (
        "You navigate and manage Unity Catalog catalogs through dbc.catalogs "
        "(REST) and dbc.sql (DDL). List catalogs, list a catalog's schemas, or "
        "create/drop a catalog. Catalogs → schemas → tables; create/drop are "
        "real, stateful actions."
    )

    def run(self, agent: "Loki", *, catalog: Optional[str] = None,
            op: str = "list", **_: Any) -> dict[str, Any]:
        client = self._client(agent)
        if op == "create":
            if not catalog:
                raise ValueError("create needs catalog=")
            client.sql.execute(f"CREATE CATALOG IF NOT EXISTS {catalog}")
            return {"created": catalog}
        if op == "drop":
            if not catalog:
                raise ValueError("drop needs catalog=")
            client.sql.execute(f"DROP CATALOG IF EXISTS {catalog}")
            return {"dropped": catalog}
        if op == "list" and catalog:              # the catalog's schemas
            schemas = [
                getattr(s, "schema_name", None) or getattr(s, "name", None) or str(s)
                for s in client.catalogs.catalog(catalog).schemas()
            ]
            return {"catalog": catalog, "schemas": schemas}
        return {"catalogs": names(client.catalogs.list_catalogs())}
