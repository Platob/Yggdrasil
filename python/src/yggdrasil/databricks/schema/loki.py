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
    description = "List Unity Catalog schemas, optionally scoped to a catalog (no warehouse needed)."
    preprompt = (
        "You list Unity Catalog schemas via dbc.schemas (REST). A schema is "
        "catalog.schema and holds tables/views/volumes/functions. Scope by "
        "catalog to narrow; this is the step between catalogs and tables."
    )

    def run(self, agent: "Loki", *, catalog: Optional[str] = None, **_: Any) -> dict[str, Any]:
        schemas = self._client(agent).schemas.list(catalog_name=catalog)
        return {"catalog": catalog, "schemas": names(schemas, attrs=("schema_name", "name", "full_name"))}
