"""Loki skill for the Unity Catalog **schemas** service (``dbc.schemas``).

A schema (a.k.a. database) is the middle level of the Unity Catalog namespace
— ``catalog.schema`` — and holds tables, views, volumes, and functions.
``dbc.schemas`` lists them across (or within) a catalog over the UC REST API.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki.skill import register

from ..loki.base import DatabricksServiceSkill, names

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksSchemasSkill"]


@register
class DatabricksSchemasSkill(DatabricksServiceSkill):
    """List Unity Catalog schemas (databases) — across, or within, a catalog."""

    name = "databricks-schemas"
    description = "List / create / drop Unity Catalog schemas (databases), within a catalog."
    preprompt = (
        "You manage Unity Catalog schemas via dbc.schemas (REST) and dbc.sql "
        "(DDL). A schema is catalog.schema and holds tables/views/volumes. List "
        "them (scoped by catalog), or create/drop one (CASCADE drops contents) "
        "— create/drop are real, stateful actions."
    )

    def run(self, agent: "Loki", *, catalog: Optional[str] = None, schema: Optional[str] = None,
            op: str = "list", cascade: bool = False, **_: Any) -> dict[str, Any]:
        client = self._client(agent)
        if op in ("create", "drop"):
            if not (catalog and schema):
                raise ValueError(f"{op} needs catalog= and schema=")
            full = f"{catalog}.{schema}"
            if op == "create":
                client.sql.execute(f"CREATE SCHEMA IF NOT EXISTS {full}")
                return {"created": full}
            client.sql.execute(f"DROP SCHEMA IF EXISTS {full}{' CASCADE' if cascade else ''}")
            return {"dropped": full}
        schemas = client.schemas.list(catalog_name=catalog)
        return {"catalog": catalog,
                "schemas": names(schemas, attrs=("schema_name", "name", "full_name"))}
