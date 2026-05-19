"""Backend-agnostic Unity-Catalog-style resource model.

Public surface
--------------
Resource layer:

* :class:`UnityEngine`   — top-level facade owning a set of catalogs.
                           ALSO a :class:`yggdrasil.data.executor.StatementExecutor`
                           (and therefore a :class:`yggdrasil.io.session.Session`),
                           so it can run :class:`UnityStatement` instances
                           via ``execute`` / ``execute_many`` / ``send``.
* :class:`UnityCatalog`  — owns schemas.
* :class:`UnitySchema`   — owns tables and views.
* :class:`UnityTable`    — managed table; mixes :class:`Tabular` so reads
                           and writes flow through the standard Arrow /
                           Polars / Pandas / Spark surface.
* :class:`UnityView`     — read-only :class:`Tabular` projection over a
                           registered table.

Statement layer:

* :class:`UnityStatement` — base :class:`PreparedStatement` for unity
                            operations. Concrete subclasses: :class:`CreateCatalog`
                            / :class:`CreateSchema` / :class:`CreateTable`
                            / :class:`CreateView` / :class:`DropCatalog`
                            / :class:`DropSchema` / :class:`DropTable`
                            / :class:`DropView` / :class:`Insert` / :class:`Select`
                            / :class:`ShowCatalogs` / :class:`ShowSchemas`
                            / :class:`ShowTables` / :class:`ShowViews`.
* :class:`UnityStatementResult` — synchronous :class:`StatementResult`
                                  whose :attr:`output` carries the operation's
                                  return value (resource handle, row count,
                                  :class:`Tabular`).
* :class:`UnityExecutionPlan` — immutable ordered sequence of statements
                                with a builder API (``then_create_catalog``,
                                ``then_insert``, …) and ``execute(engine)``.

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
from yggdrasil.unity.plan import UnityExecutionPlan
from yggdrasil.unity.result import UnityStatementResult
from yggdrasil.unity.schema import UnitySchema
from yggdrasil.unity.statement import (
    CreateCatalog,
    CreateSchema,
    CreateTable,
    CreateView,
    DropCatalog,
    DropSchema,
    DropTable,
    DropView,
    Insert,
    Select,
    ShowCatalogs,
    ShowSchemas,
    ShowTables,
    ShowViews,
    UnityStatement,
)
from yggdrasil.unity.table import UnityTable
from yggdrasil.unity.view import UnityView

__all__ = [
    # Resource layer
    "UnityResource",
    "UnityEngine",
    "UnityCatalog",
    "UnitySchema",
    "UnityTable",
    "UnityView",
    # Info dataclasses
    "CatalogInfo",
    "SchemaInfo",
    "TableInfo",
    "ViewInfo",
    # Statement layer
    "UnityStatement",
    "UnityStatementResult",
    "UnityExecutionPlan",
    "CreateCatalog",
    "CreateSchema",
    "CreateTable",
    "CreateView",
    "DropCatalog",
    "DropSchema",
    "DropTable",
    "DropView",
    "Insert",
    "Select",
    "ShowCatalogs",
    "ShowSchemas",
    "ShowTables",
    "ShowViews",
]
