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
    """Inspect and manipulate Unity Catalog tables — list / describe / preview /
    create / drop. Metadata reads go over the UC REST API (``dbc.tables``, no
    warehouse); data preview and DDL go through ``dbc.sql``."""

    name = "databricks-tables"
    description = "List / describe / preview / create / drop Unity Catalog tables."
    preprompt = (
        "You manage Unity Catalog tables via dbc.tables (REST metadata) and "
        "dbc.sql (data + DDL): list a catalog.schema, describe a table's typed "
        "columns, preview its rows (SELECT * LIMIT), create one (CTAS), or drop "
        "one. Use three-level names; treat create/drop as real, stateful actions."
    )

    def run(
        self,
        agent: "Loki",
        *,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        op: str = "auto",
        as_select: Optional[str] = None,
        limit: int = 20,
        **_: Any,
    ) -> dict[str, Any]:
        client = self._client(agent)
        full = ".".join(p for p in (catalog, schema, table) if p)
        if op == "auto":
            op = "describe" if table else "list"

        if op == "list":
            tables = client.tables.list_tables(catalog_name=catalog, schema_name=schema)
            return {"catalog": catalog, "schema": schema, "tables": names(tables)}
        if op == "describe":
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
        if op == "preview":                       # a data sample (rows, a Tabular)
            return {"table": full, "rows": client.sql.execute(f"SELECT * FROM {full} LIMIT {int(limit)}")}
        if op == "create":                        # CREATE TABLE … AS <select>
            if not as_select:
                raise ValueError("create needs as_select= (the SELECT that defines the table)")
            client.sql.execute(f"CREATE TABLE {full} AS {as_select}")
            return {"created": full}
        if op == "drop":
            client.sql.execute(f"DROP TABLE IF EXISTS {full}")
            return {"dropped": full}
        raise ValueError(f"unknown op {op!r}; use list/describe/preview/create/drop")
