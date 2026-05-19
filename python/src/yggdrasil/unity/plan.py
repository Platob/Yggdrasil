"""Ordered sequence of :class:`UnityStatement` for batched execution.

A :class:`UnityExecutionPlan` is an immutable, append-only builder
around a tuple of statements. It's intentionally simpler than
:class:`yggdrasil.data.statement.StatementBatch`: no in-flight result
map, no parallel scheduler, no result-level aggregate state — just an
ordered list with one-shot sequential execution against a
:class:`UnityEngine`.

::

    plan = (
        UnityExecutionPlan()
        .then_create_catalog("main")
        .then_create_schema("main", "default")
        .then_create_table("main", "default", "sales", schema=my_schema)
        .then_insert("main", "default", "sales", arrow_table)
    )
    results = plan.execute(engine)

Each ``then_*`` returns a new plan so plans can be branched / reused
without mutation surprises.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping

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

if TYPE_CHECKING:
    from yggdrasil.unity.engine import UnityEngine
    from yggdrasil.unity.result import UnityStatementResult


__all__ = ["UnityExecutionPlan"]


logger = logging.getLogger(__name__)


class UnityExecutionPlan:
    """Immutable, ordered sequence of :class:`UnityStatement`.

    Each ``then_*`` builder method returns a new plan with the
    statement appended. The original plan is unchanged, so a
    half-built plan can be branched into multiple downstream variants
    without copying machinery.
    """

    __slots__ = ("statements",)

    def __init__(
        self,
        statements: "Iterable[UnityStatement] | None" = None,
    ) -> None:
        if statements is None:
            self.statements: tuple[UnityStatement, ...] = ()
            return
        coerced = tuple(statements)
        for index, stmt in enumerate(coerced):
            if not isinstance(stmt, UnityStatement):
                raise TypeError(
                    f"UnityExecutionPlan entries must be UnityStatement "
                    f"instances; got {type(stmt).__name__} at index {index}."
                )
        self.statements = coerced

    # ── identity ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if not self.statements:
            return f"{type(self).__name__}([])"
        preview = " | ".join(s.text for s in self.statements[:3])
        more = "" if len(self.statements) <= 3 else f" | … (+{len(self.statements) - 3})"
        return f"{type(self).__name__}([{preview}{more}])"

    def __len__(self) -> int:
        return len(self.statements)

    def __iter__(self) -> Iterator[UnityStatement]:
        return iter(self.statements)

    def __getitem__(self, index: int) -> UnityStatement:
        return self.statements[index]

    def __add__(self, other: "UnityExecutionPlan") -> "UnityExecutionPlan":
        if not isinstance(other, UnityExecutionPlan):
            return NotImplemented
        return UnityExecutionPlan(self.statements + other.statements)

    # ── append ─────────────────────────────────────────────────────────

    def then(self, statement: UnityStatement) -> "UnityExecutionPlan":
        """Append *statement* and return a new plan."""
        if not isinstance(statement, UnityStatement):
            raise TypeError(
                f"UnityExecutionPlan.then expects a UnityStatement; "
                f"got {type(statement).__name__}: {statement!r}."
            )
        return UnityExecutionPlan(self.statements + (statement,))

    # ── builder sugar ──────────────────────────────────────────────────

    def then_create_catalog(
        self,
        name: str,
        *,
        comment: "str | None" = None,
        owner: "str | None" = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> "UnityExecutionPlan":
        return self.then(CreateCatalog(
            name=name, comment=comment, owner=owner,
            properties=properties, if_not_exists=if_not_exists,
        ))

    def then_create_schema(
        self,
        catalog_name: str,
        name: str,
        *,
        comment: "str | None" = None,
        owner: "str | None" = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> "UnityExecutionPlan":
        return self.then(CreateSchema(
            catalog_name=catalog_name, name=name,
            comment=comment, owner=owner,
            properties=properties, if_not_exists=if_not_exists,
        ))

    def then_create_table(
        self,
        catalog_name: str,
        schema_name: str,
        name: str,
        schema: Any,
        *,
        format: Any = ...,
        partition_by: "tuple[str, ...] | list[str] | None" = None,
        comment: "str | None" = None,
        owner: "str | None" = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> "UnityExecutionPlan":
        return self.then(CreateTable(
            catalog_name=catalog_name, schema_name=schema_name, name=name,
            schema=schema, format=format, partition_by=partition_by,
            comment=comment, owner=owner, properties=properties,
            if_not_exists=if_not_exists,
        ))

    def then_create_view(
        self,
        catalog_name: str,
        schema_name: str,
        name: str,
        source: Any,
        *,
        definition: "str | None" = None,
        comment: "str | None" = None,
        owner: "str | None" = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> "UnityExecutionPlan":
        return self.then(CreateView(
            catalog_name=catalog_name, schema_name=schema_name, name=name,
            source=source, definition=definition,
            comment=comment, owner=owner, properties=properties,
            if_not_exists=if_not_exists,
        ))

    def then_drop_catalog(
        self,
        name: str,
        *,
        recursive: bool = False,
        missing_ok: bool = True,
    ) -> "UnityExecutionPlan":
        return self.then(DropCatalog(
            name=name, recursive=recursive, missing_ok=missing_ok,
        ))

    def then_drop_schema(
        self,
        catalog_name: str,
        name: str,
        *,
        recursive: bool = False,
        missing_ok: bool = True,
    ) -> "UnityExecutionPlan":
        return self.then(DropSchema(
            catalog_name=catalog_name, name=name,
            recursive=recursive, missing_ok=missing_ok,
        ))

    def then_drop_table(
        self,
        catalog_name: str,
        schema_name: str,
        name: str,
        *,
        purge_data: bool = True,
        missing_ok: bool = True,
    ) -> "UnityExecutionPlan":
        return self.then(DropTable(
            catalog_name=catalog_name, schema_name=schema_name, name=name,
            purge_data=purge_data, missing_ok=missing_ok,
        ))

    def then_drop_view(
        self,
        catalog_name: str,
        schema_name: str,
        name: str,
        *,
        missing_ok: bool = True,
    ) -> "UnityExecutionPlan":
        return self.then(DropView(
            catalog_name=catalog_name, schema_name=schema_name, name=name,
            missing_ok=missing_ok,
        ))

    def then_insert(
        self,
        catalog_name: str,
        schema_name: str,
        table_name: str,
        data: Any,
        *,
        mode: Any = ...,
        match_by: "list[str] | None" = None,
    ) -> "UnityExecutionPlan":
        return self.then(Insert(
            catalog_name=catalog_name, schema_name=schema_name,
            table_name=table_name, data=data,
            mode=mode, match_by=match_by,
        ))

    def then_select(
        self,
        catalog_name: str,
        schema_name: str,
        name: str,
        *,
        options: Any = None,
    ) -> "UnityExecutionPlan":
        return self.then(Select(
            catalog_name=catalog_name, schema_name=schema_name, name=name,
            options=options,
        ))

    def then_show_catalogs(self) -> "UnityExecutionPlan":
        return self.then(ShowCatalogs())

    def then_show_schemas(self, catalog_name: str) -> "UnityExecutionPlan":
        return self.then(ShowSchemas(catalog_name=catalog_name))

    def then_show_tables(
        self,
        catalog_name: str,
        schema_name: str,
    ) -> "UnityExecutionPlan":
        return self.then(ShowTables(
            catalog_name=catalog_name, schema_name=schema_name,
        ))

    def then_show_views(
        self,
        catalog_name: str,
        schema_name: str,
    ) -> "UnityExecutionPlan":
        return self.then(ShowViews(
            catalog_name=catalog_name, schema_name=schema_name,
        ))

    # ── execution ──────────────────────────────────────────────────────

    def execute(
        self,
        engine: "UnityEngine",
        *,
        raise_error: bool = True,
    ) -> "list[UnityStatementResult]":
        """Run every statement against *engine*, in order.

        Stops at the first failure when ``raise_error=True`` (default):
        the exception propagates, and the partial result list is dropped.
        With ``raise_error=False`` each failure is captured on its
        :class:`UnityStatementResult` (``failed=True``,
        ``raise_for_status()`` re-raises) and the iteration continues
        through the remaining statements.
        """
        logger.debug(
            "Executing %d statement(s) on %r", len(self.statements), engine,
        )
        results: list = []
        for stmt in self.statements:
            result = engine.execute(stmt, raise_error=raise_error)
            results.append(result)
        return results
