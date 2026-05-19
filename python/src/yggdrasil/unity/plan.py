"""Ordered sequence of :class:`ExecutionStatement` for batched execution.

A :class:`ExecutionPlan` is an immutable, append-only builder
around a tuple of statements. It's intentionally simpler than
:class:`yggdrasil.data.statement.StatementBatch`: no in-flight result
map, no parallel scheduler, no result-level aggregate state — just an
ordered list with one-shot sequential execution against a
:class:`ExecutionEngine`.

::

    plan = (
        ExecutionPlan()
        .then_create_catalog("main")
        .then_create_schema("main", "default")
        .then_create_table("main", "default", "sales", schema=my_schema)
        .then_insert("main", "default", "sales", arrow_table)
    )
    results = plan.execute(engine)

Each ``then_*`` returns a new plan (of the same concrete type as
``self``) so plans can be branched / reused without mutation surprises.

Sub-plans
---------
Specialised plans live in :mod:`yggdrasil.unity.plans`:

* :class:`InsertExecutionPlan` — homogeneous DML_INSERT plan targeting
  a single table; carries an :attr:`target_full_name`, supports
  :meth:`InsertExecutionPlan.then_data` / :meth:`with_mode` /
  :meth:`total_rows`.
* :class:`SelectExecutionPlan` — wraps a single DQL_SELECT statement
  with chained refinements (:meth:`where`, :meth:`limit`,
  :meth:`columns`) and Tabular-shaped read shortcuts.
* :class:`MutationExecutionPlan` — only DDL / DML statements; exposes
  :meth:`MutationExecutionPlan.summary` to project results into
  per-category counts.
* :class:`ShowExecutionPlan` — only metadata listings; collapses
  results into a ``dict[PlanTypeId, list[str]]``.

Coerce a generic plan into a specialised one via
:meth:`as_insert_plan` / :meth:`as_select_plan` /
:meth:`as_mutation_plan` / :meth:`as_show_plan` — each validates and
returns the narrower type.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Mapping

from yggdrasil.unity.plan_type import PlanCategory, PlanTypeId
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
    ExecutionStatement,
)

if TYPE_CHECKING:
    from yggdrasil.unity.engine import ExecutionEngine
    from yggdrasil.unity.plans import (
        InsertExecutionPlan,
        MutationExecutionPlan,
        SelectExecutionPlan,
        ShowExecutionPlan,
    )
    from yggdrasil.unity.result import ExecutionStatementResult


__all__ = ["ExecutionPlan"]


logger = logging.getLogger(__name__)


class ExecutionPlan:
    """Immutable, ordered sequence of :class:`ExecutionStatement`.

    Each ``then_*`` builder method returns a new plan with the
    statement appended. The original plan is unchanged, so a
    half-built plan can be branched into multiple downstream variants
    without copying machinery.

    Sub-plans (:class:`InsertExecutionPlan`, :class:`SelectExecutionPlan`,
    :class:`MutationExecutionPlan`, :class:`ShowExecutionPlan`) inherit
    from this class and constrain the statements they accept via
    :attr:`ALLOWED_PLAN_TYPE_IDS` / :attr:`ALLOWED_CATEGORIES`. The
    generic parent leaves both unset — anything goes.
    """

    __slots__ = ("statements",)

    #: Whitelist of accepted :class:`PlanTypeId`. ``None`` = any.
    #: Sub-plans tighten this to enforce their homogeneity contract.
    ALLOWED_PLAN_TYPE_IDS: ClassVar["frozenset[PlanTypeId] | None"] = None

    #: Whitelist of accepted :class:`PlanCategory`. ``None`` = any.
    #: Looser than :attr:`ALLOWED_PLAN_TYPE_IDS` — useful for
    #: :class:`MutationExecutionPlan` which accepts every DDL + DML
    #: statement without enumerating each one.
    ALLOWED_CATEGORIES: ClassVar["frozenset[PlanCategory] | None"] = None

    def __init__(
        self,
        statements: "Iterable[ExecutionStatement] | None" = None,
    ) -> None:
        if statements is None:
            self.statements: tuple[ExecutionStatement, ...] = ()
            return
        coerced = tuple(statements)
        for index, stmt in enumerate(coerced):
            self._validate_statement(stmt, index=index)
        self.statements = coerced

    # ── validation hooks ───────────────────────────────────────────────

    @classmethod
    def _validate_statement(
        cls,
        stmt: ExecutionStatement,
        *,
        index: "int | None" = None,
    ) -> None:
        """Reject *stmt* when it doesn't match this plan's contract.

        Generic plans accept anything that's a :class:`ExecutionStatement`;
        sub-plans tighten via the whitelists. The ``index`` argument is
        threaded into the error message so a bulk-construction call
        points the caller at the offending entry.
        """
        if not isinstance(stmt, ExecutionStatement):
            where = f" at index {index}" if index is not None else ""
            raise TypeError(
                f"{cls.__name__} entries must be ExecutionStatement instances; "
                f"got {type(stmt).__name__}{where}: {stmt!r}."
            )

        allowed_ids = cls.ALLOWED_PLAN_TYPE_IDS
        if allowed_ids is not None and stmt.plan_type_id not in allowed_ids:
            where = f" at index {index}" if index is not None else ""
            raise TypeError(
                f"{cls.__name__} only accepts statements with "
                f"plan_type_id in {sorted(t.name for t in allowed_ids)!r}; "
                f"got {stmt.plan_type_id.name!r}{where}: {stmt!r}."
            )

        allowed_cats = cls.ALLOWED_CATEGORIES
        if allowed_cats is not None and stmt.category not in allowed_cats:
            where = f" at index {index}" if index is not None else ""
            raise TypeError(
                f"{cls.__name__} only accepts statements in categories "
                f"{sorted(c.name for c in allowed_cats)!r}; "
                f"got {stmt.category.name!r}{where}: {stmt!r}."
            )

    # ── identity ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if not self.statements:
            return f"{type(self).__name__}([])"
        preview = " | ".join(s.text for s in self.statements[:3])
        more = "" if len(self.statements) <= 3 else f" | … (+{len(self.statements) - 3})"
        return f"{type(self).__name__}([{preview}{more}])"

    def __len__(self) -> int:
        return len(self.statements)

    def __iter__(self) -> Iterator[ExecutionStatement]:
        return iter(self.statements)

    def __getitem__(self, index: int) -> ExecutionStatement:
        return self.statements[index]

    def __add__(self, other: "ExecutionPlan") -> "ExecutionPlan":
        if not isinstance(other, ExecutionPlan):
            return NotImplemented
        # The wider plan type wins — concatenating an
        # :class:`InsertExecutionPlan` with a :class:`MutationExecutionPlan`
        # demotes to a generic :class:`ExecutionPlan` rather than
        # silently breaking either side's contract.
        if type(self) is type(other):
            return self._clone(self.statements + other.statements)
        return ExecutionPlan(self.statements + other.statements)

    # ── categorization queries ────────────────────────────────────────

    @property
    def plan_type_ids(self) -> "tuple[PlanTypeId, ...]":
        """Ordered :class:`PlanTypeId` of every statement."""
        return tuple(s.plan_type_id for s in self.statements)

    @property
    def categories(self) -> "frozenset[PlanCategory]":
        """Distinct :class:`PlanCategory` values present in the plan."""
        return frozenset(s.category for s in self.statements)

    @property
    def is_homogeneous(self) -> bool:
        """``True`` when every statement shares the same :class:`PlanTypeId`."""
        ids = self.plan_type_ids
        if not ids:
            return True
        first = ids[0]
        return all(other is first for other in ids[1:])

    @property
    def is_mutation(self) -> bool:
        """``True`` when every statement mutates state (DDL / DML)."""
        return bool(self.statements) and all(
            s.is_mutation for s in self.statements
        )

    @property
    def is_query(self) -> bool:
        """``True`` when every statement is read-only (DQL / META)."""
        return bool(self.statements) and all(
            s.is_query for s in self.statements
        )

    # ── specialiser coercion ──────────────────────────────────────────

    def as_insert_plan(self) -> "InsertExecutionPlan":
        """Validate and return as an :class:`InsertExecutionPlan`."""
        from yggdrasil.unity.plans import InsertExecutionPlan

        return InsertExecutionPlan(self.statements)

    def as_select_plan(self) -> "SelectExecutionPlan":
        """Validate and return as a :class:`SelectExecutionPlan`."""
        from yggdrasil.unity.plans import SelectExecutionPlan

        return SelectExecutionPlan(self.statements)

    def as_mutation_plan(self) -> "MutationExecutionPlan":
        """Validate and return as a :class:`MutationExecutionPlan`."""
        from yggdrasil.unity.plans import MutationExecutionPlan

        return MutationExecutionPlan(self.statements)

    def as_show_plan(self) -> "ShowExecutionPlan":
        """Validate and return as a :class:`ShowExecutionPlan`."""
        from yggdrasil.unity.plans import ShowExecutionPlan

        return ShowExecutionPlan(self.statements)

    # ── clone (sub-class override hook) ───────────────────────────────

    def _clone(
        self,
        statements: "Iterable[ExecutionStatement]",
    ) -> "ExecutionPlan":
        """Construct another plan of the same type with new statements.

        Sub-plans that carry extra state outside :attr:`statements`
        (e.g. :attr:`InsertExecutionPlan.target_full_name`) override
        this to thread that state into the clone.
        """
        return type(self)(statements)

    # ── append ─────────────────────────────────────────────────────────

    def then(self, statement: ExecutionStatement) -> "ExecutionPlan":
        """Append *statement* and return a new plan of the same type."""
        self._validate_statement(statement)
        return self._clone(self.statements + (statement,))

    # ── builder sugar ──────────────────────────────────────────────────

    def then_create_catalog(
        self,
        name: str,
        *,
        comment: "str | None" = None,
        owner: "str | None" = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> "ExecutionPlan":
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
    ) -> "ExecutionPlan":
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
    ) -> "ExecutionPlan":
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
    ) -> "ExecutionPlan":
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
    ) -> "ExecutionPlan":
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
    ) -> "ExecutionPlan":
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
    ) -> "ExecutionPlan":
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
    ) -> "ExecutionPlan":
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
    ) -> "ExecutionPlan":
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
    ) -> "ExecutionPlan":
        return self.then(Select(
            catalog_name=catalog_name, schema_name=schema_name, name=name,
            options=options,
        ))

    def then_show_catalogs(self) -> "ExecutionPlan":
        return self.then(ShowCatalogs())

    def then_show_schemas(self, catalog_name: str) -> "ExecutionPlan":
        return self.then(ShowSchemas(catalog_name=catalog_name))

    def then_show_tables(
        self,
        catalog_name: str,
        schema_name: str,
    ) -> "ExecutionPlan":
        return self.then(ShowTables(
            catalog_name=catalog_name, schema_name=schema_name,
        ))

    def then_show_views(
        self,
        catalog_name: str,
        schema_name: str,
    ) -> "ExecutionPlan":
        return self.then(ShowViews(
            catalog_name=catalog_name, schema_name=schema_name,
        ))

    # ── execution ──────────────────────────────────────────────────────

    def execute(
        self,
        engine: "ExecutionEngine",
        *,
        raise_error: bool = True,
    ) -> "list[ExecutionStatementResult]":
        """Run every statement against *engine*, in order.

        Stops at the first failure when ``raise_error=True`` (default):
        the exception propagates, and the partial result list is dropped.
        With ``raise_error=False`` each failure is captured on its
        :class:`ExecutionStatementResult` (``failed=True``,
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
