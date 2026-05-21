"""In-process SQL engine on top of :class:`Tabular` + :class:`Predicate`.

What :class:`Engine` is
-----------------------

The end-to-end driver for the in-process SQL stack:

::

    engine = Engine(sources={"trades": trades_io})
    engine.execute(
        "SELECT symbol, SUM(qty) AS total FROM trades GROUP BY symbol"
    ).read_arrow_table()

Three things sit behind that one-liner:

1. :class:`yggdrasil.sql.catalog.SqlContext` resolves ``trades`` to
   a :class:`Tabular`. Anything :func:`coerce_source` accepts
   (pyarrow Table, polars / pandas frame, list of dicts, path
   string, existing Tabular) registers transparently.
2. :class:`yggdrasil.sql.planner.Planner` parses the SQL via sqlglot
   and emits an :class:`yggdrasil.sql.plan.PlanNode` tree —
   ``Scan`` / ``Filter`` / ``Project`` / ``Aggregate`` / ``Sort`` /
   ``Limit`` / ``Join``. Predicates and column lists fold into the
   ``Scan`` so the source Tabular's native pushdown (Parquet, Delta)
   gets a chance.
3. :class:`PlanNode.execute` walks the tree in Arrow, returning a
   :class:`Tabular` (an :class:`ArrowTabular` for in-memory results,
   a :class:`ParquetFile` folder when ``persist="path"``).

The handle returned to the caller is a
:class:`yggdrasil.sql.statement.SqlStatementResult` — already a
:class:`Tabular` itself, with the full read surface
(``read_arrow_table`` / ``read_polars_frame`` /
``read_pandas_frame`` / ``read_spark_frame`` / ``read_pylist`` /
``read_records``). The materialized payload is cached on the result
so repeat reads short-circuit.

Why a separate Engine class
---------------------------

The legacy :func:`yggdrasil.sql.sql` entry point (and its polars /
Arrow executors) stays as-is. :class:`Engine` is the
plan-driven sibling: same :class:`SqlStatementResult` output type,
same :class:`SqlContext`-style registration shape, but a typed
:class:`PlanNode` you can inspect via :meth:`Engine.plan`. Tests and
optimizers can pull the plan out before execution; engine-aware
operators (a hypothetical Polars or Spark scan) plug into the plan
tree directly.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Mapping, Optional

import pyarrow as pa

from yggdrasil.io.tabular.execution.expr import Expression, Predicate
from yggdrasil.io.tabular import Tabular

from yggdrasil.io.tabular.execution.sql.catalog import SqlContext
from yggdrasil.io.tabular.execution.sql.dialect import Dialect, resolve_dialect
from yggdrasil.io.tabular.execution.sql.plan import PlanNode
from yggdrasil.io.tabular.execution.sql.planner import Planner
from yggdrasil.io.tabular.execution.sql.statement import (
    PersistTarget,
    SqlPreparedStatement,
    SqlStatementResult,
)


if TYPE_CHECKING:
    pass


__all__ = ["Engine", "EnginePlan"]


# ---------------------------------------------------------------------------
# EnginePlan — prepared statement + plan tree, the unit of caching
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class EnginePlan:
    """Frozen pair of ``(prepared statement, plan tree)``.

    Returned by :meth:`Engine.prepare`; consumed by
    :meth:`Engine.run_plan`. Keeping the SQL text and the typed plan
    in one value lets callers cache "I know this statement compiles
    to that plan" — useful for dashboards / scheduled jobs that
    re-execute the same query against changing data.
    """

    statement: SqlPreparedStatement
    plan: PlanNode


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Engine:
    """Cross-flavor SQL engine on Arrow + :class:`Tabular`.

    The engine is small on purpose: the parser, planner, catalog, and
    operators are all separate modules, and this class only wires
    them together. The methods you'll call:

    - :meth:`register` / :meth:`deregister` — manage the catalog.
    - :meth:`prepare` — parse + plan only, no execution.
    - :meth:`execute` — prepare + run, returns a
      :class:`SqlStatementResult`. Use :meth:`SqlStatementResult.read_arrow_table`
      (or any other ``read_*``) to materialize.
    - :meth:`run_plan` — run a :class:`PlanNode` directly. Useful for
      programmatic plan construction or tests.
    """

    def __init__(
        self,
        sources: "Mapping[str, Any] | None" = None,
        *,
        dialect: "Dialect | str | None" = None,
        catalog: "SqlContext | None" = None,
    ) -> None:
        self.dialect: Dialect = resolve_dialect(dialect)
        self.catalog: SqlContext = catalog or SqlContext(sources)
        if catalog is not None and sources:
            self.catalog.register_many(sources)

    # ==================================================================
    # Catalog convenience — pass-through to the underlying catalog
    # ==================================================================

    def register(self, name: str, source: Any) -> "Engine":
        self.catalog.register(name, source)
        return self

    def register_many(self, sources: Mapping[str, Any]) -> "Engine":
        self.catalog.register_many(sources)
        return self

    def deregister(self, name: str) -> "Tabular | None":
        return self.catalog.deregister(name)

    def names(self) -> "list[str]":
        return self.catalog.names()

    # ==================================================================
    # Prepare — parse + plan, no execution
    # ==================================================================

    def prepare(
        self,
        query: str,
        *,
        dialect: "Dialect | str | None" = None,
        sources: "Mapping[str, Any] | None" = None,
        where: "Expression | str | None" = None,
        persist: PersistTarget = "memory",
        path: "str | None" = None,
    ) -> EnginePlan:
        """Parse *query* and return the plan tree without running it.

        ``where`` AND-merges into the parsed ``WHERE`` (lifted to a
        :class:`Predicate`); identical semantics to
        :func:`yggdrasil.sql.sql`'s ``where=`` kwarg.

        ``sources`` are scoped to this prepare call only — the
        engine's catalog gets a child with these overrides. The
        returned :class:`EnginePlan.statement` carries them.
        """
        d = resolve_dialect(dialect or self.dialect)

        # Compose the WHERE override into a Predicate and pre-thread
        # it onto the statement so :meth:`run_plan` can re-apply at
        # execute time (the planner doesn't see the override; we
        # apply it as an outer Filter on the plan).
        where_pred: Optional[Predicate] = None
        if where is not None:
            lifted = (
                where if isinstance(where, Expression)
                else Expression.from_sql(where, dialect=d.value)
            )
            if not isinstance(lifted, Predicate):
                raise TypeError(
                    f"where= must be a boolean predicate; got {type(lifted).__name__}: {lifted!r}."
                )
            where_pred = lifted

        scoped_sources: dict[str, Tabular] = {}
        if sources:
            child = self.catalog.child(sources)
            for name in child.names():
                hit = child.get(name)
                if hit is not None:
                    scoped_sources[name] = hit

        statement = SqlPreparedStatement(
            text=query,
            dialect=d,
            sources=scoped_sources or None,
            predicate=where_pred,
            persist=persist,
            path=path,
        )

        plan_tree = Planner(dialect=d).plan(query)
        if where_pred is not None:
            from yggdrasil.io.tabular.execution.sql.plan import Filter as _Filter
            plan_tree = _Filter(child=plan_tree, predicate=where_pred)
        return EnginePlan(statement=statement, plan=plan_tree)

    # ==================================================================
    # Execute — public entry point
    # ==================================================================

    def execute(
        self,
        query: "str | EnginePlan",
        *,
        dialect: "Dialect | str | None" = None,
        sources: "Mapping[str, Any] | None" = None,
        where: "Expression | str | None" = None,
        persist: PersistTarget = "memory",
        path: "str | None" = None,
        pushdown: bool = True,
    ) -> "SqlStatementResult | Any":
        """Compile + run *query*, return the live result handle.

        The returned object is always a :class:`Tabular` so every read
        method (``read_arrow_table`` / ``read_polars_frame`` /
        ``read_pandas_frame`` / ``read_spark_frame`` / ``read_pylist``
        / ``read_records``) works against the materialized payload.

        Pushdown
        --------

        When ``pushdown=True`` (the default), and *query* is a SQL
        string, the engine first tries to push the *whole* query down
        to a remote SQL engine when every referenced source supports
        it. Currently the only adapter shipped in-tree is
        :mod:`yggdrasil.sql.databricks_pushdown` — when every name in
        the SQL resolves to a
        :class:`yggdrasil.databricks.table.table.Table` on one shared
        client, the engine rewrites the FROM names to the fully-
        qualified ``catalog.schema.table`` form and submits to that
        client's :class:`SQLEngine`. The returned
        :class:`WarehouseStatementResult` *is* a :class:`Tabular` —
        the call site doesn't see a difference except for the join /
        aggregate / filter actually running on the warehouse.

        When the pushdown adapter says "not applicable" (mixed
        sources, unknown name, etc.), execution falls through to the
        in-process planner. Pass ``pushdown=False`` to force the
        Arrow path even when pushdown would have applied (useful for
        debugging / benchmarking).

        ``persist="path"`` with ``path="..."`` spills the materialized
        result to a parquet folder instead of keeping it in memory.
        """
        if isinstance(query, EnginePlan):
            return self.run_plan(query)

        # Whole-query pushdown — only attempted on raw SQL input.
        if pushdown:
            ctx = self.catalog if not sources else self.catalog.child(sources)
            from yggdrasil.io.tabular.execution.sql.databricks_pushdown import try_databricks_pushdown

            pushed = try_databricks_pushdown(
                query,
                dialect=dialect or self.dialect,
                catalog=ctx,
                where=where,
            )
            if pushed is not None:
                return pushed

        engine_plan = self.prepare(
            query,
            dialect=dialect,
            sources=sources,
            where=where,
            persist=persist,
            path=path,
        )
        return self.run_plan(engine_plan)

    # ==================================================================
    # run_plan — execute a prepared / hand-built plan
    # ==================================================================

    def run_plan(
        self,
        engine_plan: "EnginePlan | PlanNode",
        *,
        statement: "SqlPreparedStatement | None" = None,
    ) -> SqlStatementResult:
        """Evaluate *plan* and wrap the result.

        Accepts either a packaged :class:`EnginePlan` or a bare
        :class:`PlanNode` (the second form is convenient for
        programmatic plans). When given a bare :class:`PlanNode`, an
        empty :class:`SqlPreparedStatement` is synthesized so the
        :class:`SqlStatementResult` lifecycle has something to
        reference.
        """
        if isinstance(engine_plan, EnginePlan):
            stmt = engine_plan.statement
            plan_tree = engine_plan.plan
        else:
            plan_tree = engine_plan
            stmt = statement or SqlPreparedStatement(text="<programmatic plan>")

        result = SqlStatementResult(
            statement=stmt, executor=None, context=None,
        )
        # Skip the executor lookup the result's own ``start`` would
        # do — we run the plan directly and stash the holder
        # ourselves. The result then behaves as an already-started
        # cached Tabular.
        try:
            holder = self._materialize(plan_tree, stmt)
            row_count = self._row_count(holder)
        except BaseException as exc:
            result._failure = exc
            result._started = True
            raise

        result._persisted_data = holder
        result._row_count = row_count
        result._started = True
        return result

    # ==================================================================
    # Internals
    # ==================================================================

    def _materialize(
        self, plan_tree: PlanNode, statement: SqlPreparedStatement,
    ) -> Tabular:
        """Walk the plan, then route through the persist target."""
        # Per-execute child catalog so per-statement ``sources``
        # don't leak into the global registry.
        ctx = self.catalog
        if statement.sources:
            ctx = self.catalog.child(statement.sources)

        result = plan_tree.execute(ctx)

        target = statement.persist or "memory"
        if target == "memory":
            return result
        if target == "path":
            if not statement.path:
                raise ValueError(
                    "persist='path' requires a non-empty `path` "
                    "(or pass path='...' to engine.execute(...))."
                )
            return self._spill(result, statement.path)
        raise ValueError(
            f"Unknown persist target {target!r}. Valid: 'memory', "
            "'path', or None (= memory)."
        )

    @staticmethod
    def _row_count(holder: Tabular) -> int:
        """Best-effort row count for the materialized holder.

        :class:`ArrowTabular` carries one — every other shape goes
        through one extra :meth:`read_arrow_table` to count. That's
        fine because the table is already in memory at this point.
        """
        # ArrowTabular publishes its row count cheaply.
        try:
            from yggdrasil.io.tabular import ArrowTabular as _AT

            if isinstance(holder, _AT):
                table = holder.read_arrow_table()
                return int(table.num_rows)
        except Exception:
            pass
        try:
            return int(holder.read_arrow_table().num_rows)
        except Exception:
            return -1

    @staticmethod
    def _spill(holder: Tabular, path: str) -> Tabular:
        """Write the holder out to a parquet file or folder."""
        import os
        import pyarrow.parquet as pq

        table = holder.read_arrow_table()
        if path.endswith(".parquet"):
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            pq.write_table(table, path)
        else:
            os.makedirs(path, exist_ok=True)
            for index, batch in enumerate(table.to_batches()):
                pq.write_table(
                    pa.Table.from_batches([batch]),
                    os.path.join(path, f"part-{index:08d}.parquet"),
                )
        # Return the same in-memory holder; the on-disk copy is the
        # caller's to read back when they want.
        return holder
