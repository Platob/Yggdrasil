"""Top-level facade for a Unity-Catalog-style backend.

A :class:`ExecutionEngine` owns the set of catalogs the caller can navigate
through (``engine["main"]["default"]["sales"]``) and doubles as a
:class:`StatementExecutor` for the statement-shaped surface
(:class:`ExecutionStatement` → :class:`ExecutionStatementResult`,
:class:`ExecutionPlan` for ordered batches).

Inheritance shape
-----------------
``ExecutionEngine`` is a :class:`StatementExecutor` — which is itself a
:class:`Session`. That gives every concrete backend three things for
free:

1. **Singleton-by-config** via the inherited
   :class:`yggdrasil.dataclasses.singleton.Singleton` plumbing — two
   callers constructing an ``FSEngine(base=same_path)`` share one
   live instance, so the in-memory metadata caches and any per-host
   pools survive across call sites.
2. **Pickle-friendliness** — the singleton key restores in-process
   identity on unpickle; transient handles named in
   :attr:`_TRANSIENT_STATE_ATTRS` rebuild on first use.
3. **prepare / send / execute** statement-shaped surface so the same
   verbs that drive :class:`yggdrasil.spark.executor.SparkStatementExecutor`,
   ``WarehouseStatementExecutor``, etc. drive a Unity catalog too —
   ``engine.execute(CreateCatalog("main"))``, ``engine.execute_many([...])``,
   ``ExecutionPlan(...).execute(engine)``.

Backends supply :meth:`catalog` / :meth:`catalogs` / :meth:`create_catalog`;
the statement-layer dispatch is polymorphic
(:meth:`ExecutionStatement.apply` calls back into those methods), so a
backend that implements the resource trio gets the statement surface
without writing any per-statement dispatch.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Iterator

from yggdrasil.data.executor import StatementExecutor
from yggdrasil.data.statement import StatementBatch
from yggdrasil.unity.result import ExecutionStatementResult
from yggdrasil.unity.statement import ExecutionStatement

if TYPE_CHECKING:
    from yggdrasil.unity.catalog import ExecutionCatalog
    from yggdrasil.unity.plan import ExecutionPlan


__all__ = ["ExecutionEngine"]


logger = logging.getLogger(__name__)


class ExecutionEngine(StatementExecutor[ExecutionStatement, ExecutionStatementResult, StatementBatch]):
    """Root of a Unity-Catalog-style backend; doubles as a :class:`StatementExecutor`."""

    # Pin the statement-layer types so the inherited ``prepare`` / ``send``
    # pipeline produces concrete :class:`ExecutionStatement` /
    # :class:`ExecutionStatementResult` instances without per-call coercion.
    _PREPARED_CLASS: ClassVar[type[ExecutionStatement]] = ExecutionStatement
    _RESPONSE_CLASS: ClassVar[type[ExecutionStatementResult]] = ExecutionStatementResult

    # Process-lifetime singleton caching — two callers building the
    # same engine config collapse onto one instance.
    _SINGLETON_TTL: ClassVar[Any] = None

    # ── catalog navigation (concrete backend hooks) ─────────────────────

    @abstractmethod
    def catalog(self, name: str) -> "ExecutionCatalog":
        """Return a :class:`ExecutionCatalog` bound to *name*.

        Does NOT verify existence — the returned handle's
        :attr:`ExecutionResource.exists` property is the canonical probe.
        """

    @abstractmethod
    def catalogs(self) -> Iterator["ExecutionCatalog"]:
        """Iterate over every catalog visible to this engine."""

    @abstractmethod
    def create_catalog(
        self,
        name: str,
        *,
        if_not_exists: bool = True,
        **kwargs: Any,
    ) -> "ExecutionCatalog":
        """Create *name* in the backend and return its handle."""

    # ── statement-layer hooks ───────────────────────────────────────────

    def _submit_statement(
        self,
        statement: ExecutionStatement,
        start: bool = True,
    ) -> ExecutionStatementResult:
        """Build a :class:`ExecutionStatementResult` and optionally run it.

        Unity execution is synchronous: when *start* is ``True`` the
        result comes back terminal (errors captured on the result
        instead of raised, so the inherited ``execute`` /
        ``raise_for_status`` flow stays uniform). When *start* is
        ``False`` the result is returned idle so the caller can drive
        :meth:`ExecutionStatementResult.start` themselves.
        """
        result = self._RESPONSE_CLASS(statement=statement, executor=self)
        if start:
            result.start(raise_error=False)
        return result

    # ── plan execution ──────────────────────────────────────────────────

    def execute_plan(
        self,
        plan: "ExecutionPlan",
        *,
        raise_error: bool = True,
    ) -> "list[ExecutionStatementResult]":
        """Run every statement in *plan*, in order.

        Thin sugar over :meth:`ExecutionPlan.execute`; kept on the
        engine so the discoverability of ``engine.execute_plan(plan)``
        mirrors ``engine.execute(stmt)`` / ``engine.execute_many([...])``.
        """
        return plan.execute(self, raise_error=raise_error)

    # ── default navigation surface ──────────────────────────────────────

    def __getitem__(self, name: str) -> "ExecutionCatalog":
        catalog = self.catalog(name)
        if not catalog.exists:
            available = sorted(c.name for c in self.catalogs())
            raise KeyError(
                f"Catalog {name!r} does not exist on {self!r}. "
                f"Available: {available!r}. Create via "
                "engine.create_catalog(name)."
            )
        return catalog

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.catalog(name).exists

    def __iter__(self) -> Iterator["ExecutionCatalog"]:
        return self.catalogs()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
