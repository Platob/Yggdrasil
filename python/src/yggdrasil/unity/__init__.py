"""Backend-agnostic Unity-Catalog-style resource model.

Public surface
--------------
* :class:`UnityEngine`   — top-level facade owning a set of catalogs.
* :class:`UnityCatalog`  — owns schemas.
* :class:`UnitySchema`   — owns tables and views.
* :class:`UnityTable`    — managed table; mixes :class:`Tabular` so reads
                           and writes flow through the standard Arrow /
                           Polars / Pandas / Spark surface.
* :class:`UnityView`     — read-only :class:`Tabular` projection over a
                           registered table.

The info dataclasses :class:`CatalogInfo` / :class:`SchemaInfo` /
:class:`TableInfo` / :class:`ViewInfo` are the on-the-wire payloads
every backend round-trips.

Concrete backends live in subpackages — start with :mod:`yggdrasil.unity.fs`
for a filesystem-backed catalog over the project's :class:`Path` surface.
"""

from yggdrasil.unity.base import UnityResource
from yggdrasil.unity.catalog import UnityCatalog
from yggdrasil.unity.engine import UnityEngine
from yggdrasil.unity.info import CatalogInfo, SchemaInfo, TableInfo, ViewInfo
from yggdrasil.unity.schema import UnitySchema
from yggdrasil.unity.table import UnityTable
from yggdrasil.unity.view import UnityView

__all__ = [
    "UnityResource",
    "UnityEngine",
    "UnityCatalog",
    "UnitySchema",
    "UnityTable",
    "UnityView",
    "CatalogInfo",
    "SchemaInfo",
    "TableInfo",
    "ViewInfo",
]
