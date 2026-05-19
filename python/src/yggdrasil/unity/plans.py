"""Specialised :class:`ExecutionPlan` subclasses.

Each sub-plan tightens :attr:`ExecutionPlan.ALLOWED_PLAN_TYPE_IDS`
(or :attr:`ALLOWED_CATEGORIES`) so its constructor rejects out-of-scope
statements, then adds an ecosystem of methods specific to that
operation shape. Plans are still immutable — every builder returns a
new instance.

* :class:`InsertExecutionPlan` — homogeneous DML_INSERT plan targeting
  a single table. Carries :attr:`target_full_name`; adds
  :meth:`then_data`, :meth:`with_mode`, :meth:`with_match_by`,
  :meth:`total_rows`.
* :class:`SelectExecutionPlan` — wraps a single DQL_SELECT statement
  with chained refinements (:meth:`where`, :meth:`limit`,
  :meth:`columns`) and read shortcuts (:meth:`read_arrow_table`,
  :meth:`read_polars_frame`, :meth:`read_pandas_frame`).
* :class:`MutationExecutionPlan` — accepts every DDL + DML statement
  via :attr:`ALLOWED_CATEGORIES`. :meth:`summary` projects executed
  results into per-plan-type counts.
* :class:`ShowExecutionPlan` — accepts every metadata listing. Adds
  :meth:`results` to fold execution outputs into
  ``dict[PlanTypeId, list[str]]``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Iterable

from yggdrasil.data.options import CastOptions
from yggdrasil.unity.plan import ExecutionPlan
from yggdrasil.unity.plan_type import PlanCategory, PlanTypeId
from yggdrasil.unity.statement import (
    Insert,
    Select,
    ShowCatalogs,
    ShowSchemas,
    ShowTables,
    ShowViews,
    ExecutionStatement,
)

if TYPE_CHECKING:
    import pyarrow as pa
    import polars as pl
    import pandas as pd

    from yggdrasil.unity.engine import ExecutionEngine
    from yggdrasil.unity.result import ExecutionStatementResult


__all__ = [
    "InsertExecutionPlan",
    "SelectExecutionPlan",
    "MutationExecutionPlan",
    "ShowExecutionPlan",
]


logger = logging.getLogger(__name__)


# ── InsertExecutionPlan ─────────────────────────────────────────────────


class InsertExecutionPlan(ExecutionPlan):
    """Plan of :class:`Insert` statements all targeting the same table.

    Two construction routes:

    * :meth:`for_table` — start empty against ``catalog.schema.name``
      and chain :meth:`then_data` calls.
    * Pass a non-empty iterable of :class:`Insert` to the constructor;
      ``target_full_name`` is inferred from the first statement and
      enforced across the rest.
    """

    __slots__ = ("target_full_name",)

    ALLOWED_PLAN_TYPE_IDS: ClassVar = frozenset({PlanTypeId.INSERT})

    def __init__(
        self,
        statements: "Iterable[ExecutionStatement] | None" = None,
        *,
        target_full_name: "str | None" = None,
    ) -> None:
        super().__init__(statements)
        if not self.statements and target_full_name is None:
            raise ValueError(
                "InsertExecutionPlan requires either a non-empty statements "
                "iterable or target_full_name=. Use InsertExecutionPlan."
                "for_table(catalog, schema, name) to start empty."
            )
        if target_full_name is None:
            target_full_name = self.statements[0].target_full_name  # type: ignore[attr-defined]
        for stmt in self.statements:
            if stmt.target_full_name != target_full_name:  # type: ignore[attr-defined]
                raise ValueError(
                    f"InsertExecutionPlan statements must all target "
                    f"{target_full_name!r}; got {stmt.target_full_name!r}."  # type: ignore[attr-defined]
                )
        # ``__slots__`` blocks ``self.x = y`` shortcuts via the parent's
        # missing slot list, so go through object.__setattr__ to bypass.
        object.__setattr__(self, "target_full_name", target_full_name)

    # ── ctor sugar ─────────────────────────────────────────────────────

    @classmethod
    def for_table(
        cls,
        catalog_name: str,
        schema_name: str,
        table_name: str,
    ) -> "InsertExecutionPlan":
        """Start an empty insert plan targeting *catalog.schema.table*."""
        return cls(
            statements=(),
            target_full_name=f"{catalog_name}.{schema_name}.{table_name}",
        )

    # ── clone / append ─────────────────────────────────────────────────

    def _clone(
        self,
        statements: "Iterable[ExecutionStatement]",
    ) -> "InsertExecutionPlan":
        return InsertExecutionPlan(
            statements=statements,
            target_full_name=self.target_full_name,
        )

    # ── insert-specific builders ──────────────────────────────────────

    def then_data(
        self,
        data: Any,
        *,
        mode: Any = ...,
        match_by: "list[str] | None" = None,
    ) -> "InsertExecutionPlan":
        """Append another :class:`Insert` against :attr:`target_full_name`."""
        cat, sch, tbl = self.target_full_name.split(".")
        return self.then(Insert(
            catalog_name=cat, schema_name=sch, table_name=tbl,
            data=data, mode=mode, match_by=match_by,
        ))

    def with_mode(self, mode: Any) -> "InsertExecutionPlan":
        """Return a plan where every :class:`Insert` carries *mode*."""
        cat, sch, tbl = self.target_full_name.split(".")
        rewritten = tuple(
            Insert(
                catalog_name=cat, schema_name=sch, table_name=tbl,
                data=stmt.data, mode=mode, match_by=stmt.match_by,  # type: ignore[attr-defined]
            )
            for stmt in self.statements
        )
        return InsertExecutionPlan(
            statements=rewritten,
            target_full_name=self.target_full_name,
        )

    def with_match_by(
        self,
        match_by: "list[str] | None",
    ) -> "InsertExecutionPlan":
        """Return a plan where every :class:`Insert` carries *match_by*."""
        cat, sch, tbl = self.target_full_name.split(".")
        rewritten = tuple(
            Insert(
                catalog_name=cat, schema_name=sch, table_name=tbl,
                data=stmt.data, mode=stmt.mode, match_by=match_by,  # type: ignore[attr-defined]
            )
            for stmt in self.statements
        )
        return InsertExecutionPlan(
            statements=rewritten,
            target_full_name=self.target_full_name,
        )

    # ── execution helpers ─────────────────────────────────────────────

    @staticmethod
    def total_rows(results: "Iterable[ExecutionStatementResult]") -> int:
        """Sum the row counts from every :class:`Insert` result.

        Skips entries whose :attr:`output` isn't an ``int`` so a
        partial-failure result (``output=None`` after a failed insert)
        doesn't blow up the tally.
        """
        return sum(
            r.output for r in results
            if isinstance(getattr(r, "output", None), int)
        )

    def then_insert(self, *args: Any, **kwargs: Any) -> "InsertExecutionPlan":
        """Override the parent's ``then_insert`` to forbid cross-table appends.

        :class:`InsertExecutionPlan` carries a single target by design;
        callers reaching for ``then_insert(catalog, schema, table, ...)``
        on the wrong table are surfaced loudly here rather than at
        :meth:`ExecutionPlan._validate_statement`'s deeper error.
        """
        raise TypeError(
            "InsertExecutionPlan is single-target; use plan.then_data(data, "
            "mode=...) to append rows. To insert into a different table "
            "build a new InsertExecutionPlan.for_table(...) or use "
            "ExecutionPlan."
        )


# ── SelectExecutionPlan ─────────────────────────────────────────────────


class SelectExecutionPlan(ExecutionPlan):
    """Plan wrapping exactly one :class:`Select` with chained refinements.

    The class supports the common case where the caller wants to
    refine a read with predicates / limits / column projection before
    executing. Each refinement returns a new plan whose underlying
    :class:`Select.options` payload (a :class:`CastOptions`) carries
    the merged state.

    Reads bypass the result's Tabular forwarding and consult the
    plan's stored options directly — keeps the result class simple
    and the option-threading discoverable on the plan.
    """

    __slots__ = ("_column_projection",)

    ALLOWED_PLAN_TYPE_IDS: ClassVar = frozenset({PlanTypeId.SELECT})

    def __init__(
        self,
        statements: "Iterable[ExecutionStatement] | None" = None,
        *,
        column_projection: "tuple[str, ...] | None" = None,
    ) -> None:
        super().__init__(statements)
        if len(self.statements) != 1:
            raise ValueError(
                f"SelectExecutionPlan wraps exactly one Select; got "
                f"{len(self.statements)} statement(s). Use "
                "SelectExecutionPlan.for_target(catalog, schema, name)."
            )
        # ``__slots__`` blocks direct assignment via the parent's
        # missing slot list, so route through object.__setattr__.
        object.__setattr__(self, "_column_projection", column_projection)

    # ── ctor sugar ─────────────────────────────────────────────────────

    @classmethod
    def for_target(
        cls,
        catalog_name: str,
        schema_name: str,
        name: str,
        *,
        options: "CastOptions | None" = None,
    ) -> "SelectExecutionPlan":
        """Build a plan reading from *catalog.schema.name*."""
        return cls((Select(
            catalog_name=catalog_name, schema_name=schema_name, name=name,
            options=options,
        ),))

    # ── accessors ──────────────────────────────────────────────────────

    @property
    def select(self) -> Select:
        """The wrapped :class:`Select` statement."""
        return self.statements[0]  # type: ignore[return-value]

    @property
    def target_full_name(self) -> str:
        return self.select.target_full_name

    @property
    def options(self) -> "CastOptions | None":
        """Current :class:`CastOptions` attached to the inner Select."""
        opts = self.select.options
        return opts if isinstance(opts, CastOptions) else None

    # ── clone (preserves projection) ──────────────────────────────────

    def _clone(
        self,
        statements: "Iterable[ExecutionStatement]",
    ) -> "SelectExecutionPlan":
        return SelectExecutionPlan(
            statements=statements,
            column_projection=self._column_projection,
        )

    # ── refinements (return new plan) ─────────────────────────────────

    def with_options(self, **overrides: Any) -> "SelectExecutionPlan":
        """Return a new plan whose inner Select carries the merged options."""
        base = self.options if self.options is not None else CastOptions()
        merged = base.copy(**overrides) if overrides else base
        return SelectExecutionPlan(
            statements=(Select(
                catalog_name=self.select.catalog_name,
                schema_name=self.select.schema_name,
                name=self.select.name,
                options=merged,
            ),),
            column_projection=self._column_projection,
        )

    def where(self, predicate: Any) -> "SelectExecutionPlan":
        """Attach a row-level predicate to the inner Select."""
        return self.with_options(predicate=predicate)

    def limit(self, n: int) -> "SelectExecutionPlan":
        """Cap the number of rows read by the inner Select."""
        if n < 0:
            raise ValueError(f"limit must be >= 0; got {n!r}.")
        return self.with_options(row_limit=n)

    def columns(self, *names: str) -> "SelectExecutionPlan":
        """Project a subset of columns from the read.

        Stored as a separate :attr:`_column_projection` slot — it can't
        live on :class:`CastOptions` because its ``target`` field coerces
        through :meth:`Field.from_` which doesn't accept a bare name
        list. Applied after :class:`pa.Table` materialisation by the
        :meth:`read_*` shortcuts.
        """
        if not names:
            raise ValueError("columns(...) requires at least one column name.")
        return SelectExecutionPlan(
            statements=self.statements,
            column_projection=tuple(names),
        )

    @property
    def column_projection(self) -> "tuple[str, ...] | None":
        """Column names requested via :meth:`columns`, or ``None``."""
        return self._column_projection

    # ── execution / read shortcuts ────────────────────────────────────

    def execute(
        self,
        engine: "ExecutionEngine",
        *,
        raise_error: bool = True,
    ) -> "list[ExecutionStatementResult]":
        return super().execute(engine, raise_error=raise_error)

    def _resolve(self, engine: "ExecutionEngine") -> Any:
        """Run the inner Select and return the resolved Tabular."""
        result = engine.execute(self.select)
        return result.output

    def _apply_projection(self, table: "pa.Table") -> "pa.Table":
        names = self._column_projection
        if not names:
            return table
        missing = [n for n in names if n not in table.column_names]
        if missing:
            raise KeyError(
                f"columns({names!r}) requested columns {missing!r} not "
                f"present in {self.target_full_name!r} "
                f"(have {table.column_names!r})."
            )
        return table.select(list(names))

    def read_arrow_table(
        self,
        engine: "ExecutionEngine",
        *,
        options: "CastOptions | None" = None,
    ) -> "pa.Table":
        target = self._resolve(engine)
        effective = options if options is not None else self.options
        table = target.read_arrow_table(options=effective)
        return self._apply_projection(table)

    def read_polars_frame(
        self,
        engine: "ExecutionEngine",
        *,
        options: "CastOptions | None" = None,
    ) -> "pl.DataFrame":
        # Defer to read_arrow_table for column projection consistency.
        from yggdrasil.polars.lib import polars

        arrow = self.read_arrow_table(engine, options=options)
        return polars.from_arrow(arrow)  # type: ignore[return-value]

    def read_pandas_frame(
        self,
        engine: "ExecutionEngine",
        *,
        options: "CastOptions | None" = None,
    ) -> "pd.DataFrame":
        return self.read_arrow_table(engine, options=options).to_pandas()


# ── MutationExecutionPlan ───────────────────────────────────────────────


class MutationExecutionPlan(ExecutionPlan):
    """Plan composed only of DDL + DML statements — never reads.

    Accepts any :class:`ExecutionStatement` whose category is
    :attr:`PlanCategory.DDL` or :attr:`PlanCategory.DML`; rejects
    :class:`Select` / :class:`ShowCatalogs` / etc. at construction.
    """

    __slots__ = ()

    ALLOWED_CATEGORIES: ClassVar = frozenset({
        PlanCategory.DDL,
        PlanCategory.DML,
    })

    @staticmethod
    def summary(
        results: "Iterable[ExecutionStatementResult]",
    ) -> "dict[PlanTypeId, int]":
        """Project results into per-:class:`PlanTypeId` counts.

        DDL entries (create / drop) increment by 1 per successful
        statement; :class:`Insert` increments by the row-count carried
        on the result's :attr:`output`. Failed results are skipped so
        the summary reflects what actually landed.
        """
        out: dict[PlanTypeId, int] = {}
        for r in results:
            if r.failed:
                continue
            stmt = r.statement
            if stmt.plan_type_id is PlanTypeId.INSERT:
                rows = r.output if isinstance(r.output, int) else 0
                out[PlanTypeId.INSERT] = out.get(PlanTypeId.INSERT, 0) + rows
            else:
                out[stmt.plan_type_id] = out.get(stmt.plan_type_id, 0) + 1
        return out


# ── ShowExecutionPlan ───────────────────────────────────────────────────


class ShowExecutionPlan(ExecutionPlan):
    """Plan composed only of metadata listings.

    Accepts the four :class:`PlanCategory.META` statements (catalogs /
    schemas / tables / views); :meth:`results` folds the executed
    outputs into a ``dict[PlanTypeId, list[str]]`` for ergonomic
    consumption.
    """

    __slots__ = ()

    ALLOWED_PLAN_TYPE_IDS: ClassVar = frozenset({
        PlanTypeId.SHOW_CATALOGS,
        PlanTypeId.SHOW_SCHEMAS,
        PlanTypeId.SHOW_TABLES,
        PlanTypeId.SHOW_VIEWS,
    })

    # ── builders that route to homogeneous statements ────────────────

    def then_show_catalogs(self) -> "ShowExecutionPlan":
        return self.then(ShowCatalogs())  # type: ignore[return-value]

    def then_show_schemas(self, catalog_name: str) -> "ShowExecutionPlan":
        return self.then(ShowSchemas(catalog_name=catalog_name))  # type: ignore[return-value]

    def then_show_tables(
        self,
        catalog_name: str,
        schema_name: str,
    ) -> "ShowExecutionPlan":
        return self.then(ShowTables(  # type: ignore[return-value]
            catalog_name=catalog_name, schema_name=schema_name,
        ))

    def then_show_views(
        self,
        catalog_name: str,
        schema_name: str,
    ) -> "ShowExecutionPlan":
        return self.then(ShowViews(  # type: ignore[return-value]
            catalog_name=catalog_name, schema_name=schema_name,
        ))

    @staticmethod
    def results(
        results: "Iterable[ExecutionStatementResult]",
    ) -> "dict[PlanTypeId, list[str]]":
        """Project results into ``{plan_type_id: list[name]}``.

        Multiple ``SHOW_*`` statements of the same type are merged
        (sorted, dedup'd) so a plan that lists schemas in two
        catalogs collapses to one entry. Failed results are skipped.
        """
        out: dict[PlanTypeId, set[str]] = {}
        for r in results:
            if r.failed:
                continue
            names = r.output
            if not isinstance(names, list):
                continue
            bucket = out.setdefault(r.statement.plan_type_id, set())
            bucket.update(names)
        return {k: sorted(v) for k, v in out.items()}
