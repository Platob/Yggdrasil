"""Loki skill for the Unity Catalog **tables** service (``dbc.tables``).

A table (managed or external) or view is the leaf of the Unity Catalog
namespace — ``catalog.schema.table`` — with typed columns. ``dbc.tables`` lists
and describes them over the UC REST API (no SQL warehouse), returning yggdrasil
``DataType``-typed column metadata.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, names

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksTablesSkill"]


@register
class DatabricksTablesSkill(DatabricksServiceSkill):
    """List tables in a catalog.schema, or describe one (typed columns, no warehouse)."""

    name = "databricks-tables"
    description = "List tables in a catalog.schema, or describe a table (Unity Catalog API)."
    preprompt = (
        "You inspect Unity Catalog tables via dbc.tables (REST, no warehouse): "
        "list a catalog.schema, or describe a table for its typed columns "
        "(yggdrasil DataType). Use this for schema discovery before writing SQL."
    )

    def run(
        self,
        agent: "Loki",
        *,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any]:
        client = self._client(agent)
        if table:
            full = ".".join(p for p in (catalog, schema, table) if p)
            t = client.tables.get(full)
            if t is None:
                return {"table": full, "found": False}
            return {
                "table": t.full_name(),
                "type": str(t.table_type),
                "columns": [
                    {"name": c.name, "type": str(getattr(getattr(c, "field", None), "dtype", ""))}
                    for c in t.columns
                ],
            }
        tables = client.tables.list_tables(catalog_name=catalog, schema_name=schema)
        return {"catalog": catalog, "schema": schema, "tables": names(tables)}
