"""Saga — the unified, autonomous, lazy data engine.

A :class:`Saga` ties together the three layers that live in this package
and turns them into a single entry point:

- the expression / predicate AST (:mod:`yggdrasil.saga.expr`),
- the lazy execution plans (:mod:`yggdrasil.saga.plan`), and
- SQL parse / emit across dialects.

It holds a *catalog* of named :class:`~yggdrasil.io.tabular.base.Tabular`
sources plus a default SQL dialect, then lets a caller:

- register / scan tables into the catalog,
- parse SQL from many dialects into a plan-node tree (:meth:`parse`) or a
  mutable :class:`~yggdrasil.saga.plan.ExecutionPlan` (:meth:`plan`),
- run SQL end-to-end against the catalog (:meth:`sql`),
- build deferred lazy pipelines (``saga.scan("t").filter(...)``), and
- execute any plan node / :class:`ExecutionPlan` against the catalog.

The engine itself owns no data — every concrete Tabular it executes
against comes from the catalog or is passed in. That keeps the same
contract every yggdrasil backend uses: leverage ``Tabular`` / ``Field`` /
``DataType``, defer compute, and let each source dispatch to its native
engine (Arrow C++ kernels, Polars, Spark, Databricks SQL).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.io.tabular.base import Tabular, is_tabular_source

from .plan import BUILTIN_REGISTRY, ExecutionPlan, LazyTabular
from .plan.nodes import PlanNode

if TYPE_CHECKING:
    from yggdrasil.enums import Dialect

    from .plan import FunctionRegistry


class Saga:
    """Unified lazy data engine over yggdrasil Tabulars.

    Construct empty or with a starting catalog and default dialect::

        saga = Saga(dialect="databricks")
        saga.register("users", users_tabular)

        # SQL end-to-end against the catalog
        top = saga.sql("SELECT name, score FROM users WHERE score > 80")

        # Lazy, deferred pipeline
        out = (saga.scan("users")
                   .filter("region = 'US'")
                   .select("name", "score")
                   .limit(5)
                   .read_arrow_table())

        # Parse without executing
        node = saga.parse("SELECT * FROM users")
        plan = saga.plan("INSERT INTO archive SELECT * FROM users")
    """

    __slots__ = ("_tables", "_dialect", "_registry")

    def __init__(
        self,
        tables: dict[str, Tabular] | None = None,
        *,
        dialect: Dialect | str | None = None,
        registry: FunctionRegistry | None = None,
    ) -> None:
        self._tables: dict[str, Tabular] = dict(tables) if tables else {}
        self._dialect: Dialect | str | None = dialect
        self._registry: FunctionRegistry = registry if registry is not None else BUILTIN_REGISTRY

    # -- Catalog ---------------------------------------------------------

    def register(self, name: str, tabular: Tabular) -> Saga:
        """Upsert a named Tabular into the catalog and return self (chainable)."""
        self._tables[name] = tabular
        return self

    def unregister(self, name: str, *, missing_ok: bool = True) -> Saga:
        if name in self._tables:
            del self._tables[name]
        elif not missing_ok:
            raise KeyError(name)
        return self

    def table(self, name: str) -> Tabular:
        """Return the catalog Tabular *name*, resolving paths/URLs on miss."""
        if name in self._tables:
            return self._tables[name]
        if is_tabular_source(name):
            resolved = Tabular.from_(name, default=None)
            if resolved is not None:
                self._tables[name] = resolved
                return resolved
        raise KeyError(f"Table {name!r} not registered. Available: {sorted(self._tables)}")

    @property
    def tables(self) -> dict[str, Tabular]:
        return self._tables

    @property
    def registry(self) -> FunctionRegistry:
        return self._registry

    @property
    def dialect(self) -> Dialect | str | None:
        return self._dialect

    def with_dialect(self, dialect: Dialect | str | None) -> Saga:
        """Return a sibling engine sharing this catalog but a different dialect."""
        sibling = Saga(dialect=dialect, registry=self._registry)
        sibling._tables = self._tables  # share the same catalog by reference
        return sibling

    def __contains__(self, name: str) -> bool:
        return name in self._tables

    def __getitem__(self, name: str) -> Tabular:
        return self.table(name)

    def __setitem__(self, name: str, tabular: Tabular) -> None:
        self._tables[name] = tabular

    # -- Parsing ---------------------------------------------------------

    def parse(
        self,
        sql: str,
        *,
        dialect: Dialect | str | None = None,
        default: Any = ...,
    ) -> PlanNode | Any:
        """Parse *sql* into an immutable :class:`PlanNode` tree.

        Falls back to the engine's default dialect; returns *default* on
        parse failure when *default* is supplied.
        """
        from .plan.sql_parser import parse_sql

        return parse_sql(sql, dialect=dialect if dialect is not None else self._dialect, default=default)

    def plan(
        self,
        sql: str,
        *,
        dialect: Dialect | str | None = None,
        default: Any = ...,
    ) -> ExecutionPlan | Any:
        """Parse *sql* into a mutable :class:`ExecutionPlan` (Select/Insert/Merge)."""
        return ExecutionPlan.from_sql(
            sql,
            dialect=dialect if dialect is not None else self._dialect,
            default=default,
        )

    def to_sql(self, plan: PlanNode | ExecutionPlan, *, dialect: Dialect | str | None = None) -> str:
        """Emit SQL for a plan node or :class:`ExecutionPlan` in the given dialect."""
        return plan.to_sql(dialect=dialect if dialect is not None else self._dialect)

    # -- Execution -------------------------------------------------------

    def sql(
        self,
        query: str,
        *,
        dialect: Dialect | str | None = None,
        tables: dict[str, Tabular] | None = None,
    ) -> Tabular:
        """Parse and execute *query* against the catalog, returning a Tabular.

        Extra one-off *tables* are layered over the catalog without
        mutating it (handy for ad-hoc sources in a single query).
        """
        node = self.parse(query, dialect=dialect)
        return self.execute(node, tables=tables)

    def execute(
        self,
        plan: PlanNode | ExecutionPlan,
        *,
        tables: dict[str, Tabular] | None = None,
    ) -> Tabular:
        """Execute a plan node / :class:`ExecutionPlan` against the catalog."""
        catalog = self._tables if not tables else {**self._tables, **tables}
        if isinstance(plan, ExecutionPlan):
            # Standalone plans already carry their bound source(s); the
            # catalog only matters for the node path below.
            return plan.execute()
        if isinstance(plan, PlanNode):
            return plan.execute(tables=catalog)
        raise TypeError(f"Cannot execute {type(plan).__name__}; expected PlanNode or ExecutionPlan")

    # -- Lazy building ---------------------------------------------------

    def scan(self, source: str | Tabular) -> LazyTabular:
        """Open a deferred :class:`LazyTabular` over a catalog name or Tabular.

        Transformations (``select``/``filter``/``join``/…) accumulate in
        an :class:`ExecutionPlan` and only run on the first ``read_*``.
        """
        tab = source if isinstance(source, Tabular) else self.table(source)
        return tab.lazy()

    # ``lazy`` reads more naturally for an already-held Tabular.
    lazy = scan

    def __repr__(self) -> str:
        return f"Saga(tables={sorted(self._tables)}, dialect={self._dialect!r})"
