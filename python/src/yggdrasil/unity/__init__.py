"""Backend-agnostic catalog resource model with an execution-plan layer.

Public surface
--------------
Resource layer:

* :class:`ExecutionEngine`   — top-level facade owning a set of catalogs.
                               ALSO a :class:`yggdrasil.data.executor.StatementExecutor`
                               (and therefore a :class:`yggdrasil.io.session.Session`),
                               so it can run :class:`ExecutionStatement` instances
                               via ``execute`` / ``execute_many`` / ``send``.
* :class:`ExecutionCatalog`  — owns schemas.
* :class:`ExecutionSchema`   — owns tables and views.
* :class:`ExecutionTable`    — managed table; mixes :class:`Tabular` so reads
                               and writes flow through the standard Arrow /
                               Polars / Pandas / Spark surface.
* :class:`ExecutionView`     — read-only :class:`Tabular` projection over a
                               registered table.

Statement layer:

* :class:`ExecutionStatement` — base :class:`PreparedStatement` for catalog
                                operations. Concrete subclasses: :class:`CreateCatalog`
                                / :class:`CreateSchema` / :class:`CreateTable`
                                / :class:`CreateView` / :class:`DropCatalog`
                                / :class:`DropSchema` / :class:`DropTable`
                                / :class:`DropView` / :class:`Insert` / :class:`Select`
                                / :class:`ShowCatalogs` / :class:`ShowSchemas`
                                / :class:`ShowTables` / :class:`ShowViews`.
* :class:`ExecutionStatementResult` — synchronous :class:`StatementResult`
                                      whose :attr:`output` carries the operation's
                                      return value (resource handle, row count,
                                      :class:`Tabular`).
* :class:`ExecutionPlan` — immutable ordered sequence of statements
                           with a builder API and ``execute(engine)``.
* :class:`InsertExecutionPlan` / :class:`SelectExecutionPlan` /
  :class:`MutationExecutionPlan` / :class:`ShowExecutionPlan` —
  specialised sub-plans with their own narrower builders and read /
  summary helpers.

Classification:

* :class:`PlanTypeId` — IntEnum identifying every concrete statement shape.
* :class:`PlanCategory` — coarse grouping (DDL / DML / DQL / META).
  Every :class:`ExecutionStatement` carries a ``plan_type_id`` ClassVar,
  and :class:`ExecutionPlan` exposes :attr:`plan_type_ids` /
  :attr:`categories` / :attr:`is_homogeneous` / :attr:`is_mutation` /
  :attr:`is_query` for plan-level filtering.

The info dataclasses :class:`CatalogInfo` / :class:`SchemaInfo` /
:class:`TableInfo` / :class:`ViewInfo` are the on-the-wire payloads
every backend round-trips.

Concrete backends live in subpackages — start with :mod:`yggdrasil.unity.fs`
for a filesystem-backed catalog over the project's :class:`Path` surface.
"""

from yggdrasil.unity.base import ExecutionResource
from yggdrasil.unity.catalog import ExecutionCatalog
from yggdrasil.unity.engine import ExecutionEngine
from yggdrasil.unity.info import CatalogInfo, SchemaInfo, TableInfo, ViewInfo
from yggdrasil.unity.plan import ExecutionPlan
from yggdrasil.unity.plan_type import PlanCategory, PlanTypeId
from yggdrasil.unity.plans import (
    InsertExecutionPlan,
    MutationExecutionPlan,
    SelectExecutionPlan,
    ShowExecutionPlan,
)
from yggdrasil.unity.result import ExecutionStatementResult
from yggdrasil.unity.schema import ExecutionSchema
from yggdrasil.unity.statement import (
    CreateCatalog,
    CreateSchema,
    CreateTable,
    CreateView,
    DropCatalog,
    DropSchema,
    DropTable,
    DropView,
    ExecutionStatement,
    Insert,
    Select,
    ShowCatalogs,
    ShowSchemas,
    ShowTables,
    ShowViews,
)
from yggdrasil.unity.table import ExecutionTable
from yggdrasil.unity.view import ExecutionView

__all__ = [
    # Resource layer
    "ExecutionResource",
    "ExecutionEngine",
    "ExecutionCatalog",
    "ExecutionSchema",
    "ExecutionTable",
    "ExecutionView",
    # Info dataclasses
    "CatalogInfo",
    "SchemaInfo",
    "TableInfo",
    "ViewInfo",
    # Classification
    "PlanCategory",
    "PlanTypeId",
    # Statement layer
    "ExecutionStatement",
    "ExecutionStatementResult",
    "ExecutionPlan",
    "InsertExecutionPlan",
    "SelectExecutionPlan",
    "MutationExecutionPlan",
    "ShowExecutionPlan",
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
