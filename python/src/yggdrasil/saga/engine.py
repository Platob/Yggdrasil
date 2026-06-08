"""Saga — the unified, autonomous, lazy data engine.

A :class:`Saga` ties together the three layers in this package and turns
them into a single entry point:

- the expression / predicate AST (:mod:`yggdrasil.saga.expr`),
- the lazy execution plans (:mod:`yggdrasil.saga.plan`), and
- SQL parse / emit across dialects.

Saga is **stateless with respect to data**. It does not hold a catalog of
tables (named-table registration will come later). Instead it parses the
``FROM`` sources out of the SQL / plan and **live-builds** them: a
path/URL-shaped source is resolved through :meth:`Tabular.from_` (the IO
layer, specialized on serialized data), and an in-memory engine frame
(arrow / polars / pandas / spark) through :meth:`Tabular.new`. Execution
runs entirely through Saga's own plan executor — no statement-executor /
warehouse machinery is involved.

Each engine owns a :class:`SagaSession` whose **local-disk staging area**
(``~/.saga/<session-id>/staging``) is used to spill results, so large
outputs don't have to live in RAM. The staging tree is auto-cleaned when
the session is closed (explicitly, as a context manager, or at process
exit).
"""

from __future__ import annotations

import atexit
import logging
import os
import pathlib
import shutil
import time
from typing import TYPE_CHECKING, Any

from yggdrasil.io.tabular.base import Tabular

from .plan import BUILTIN_REGISTRY, ExecutionPlan
from .plan.nodes import PlanNode

if TYPE_CHECKING:
    from yggdrasil.enums import Dialect

    from .plan import ExecutionResult, FunctionRegistry, LazyTabular

logger = logging.getLogger(__name__)

# Result spill threshold — a collected result smaller than this stays fully
# in memory; past it the holder spills parts to the session staging dir.
_DEFAULT_RESULT_SPILL_BYTES = 64 * 1024 * 1024


class SagaSession:
    """Owns a local-disk staging area for a Saga, auto-cleaned on close.

    Layout: ``<root>/<session-id>/staging`` where *root* defaults to
    ``~/.saga``. Spilled results are append-only IPC parts under
    ``staging``; the whole session tree is removed on :meth:`close` (also
    registered with :mod:`atexit` so a forgotten session still cleans up).
    """

    __slots__ = ("session_id", "root", "staging", "_closed", "_atexit")

    def __init__(self, session_id: str | None = None, *, root: str | os.PathLike | None = None) -> None:
        # Sortable, process-unique, human-readable — not a domain ID, just
        # a directory name (avoid crypto hashes per project convention).
        self.session_id: str = session_id or f"{int(time.time() * 1000):013d}-{os.getpid()}"
        base = pathlib.Path(root) if root is not None else pathlib.Path.home() / ".saga"
        self.root: pathlib.Path = base / self.session_id
        self.staging: pathlib.Path = self.root / "staging"
        self._closed: bool = False
        # Best-effort reap on interpreter shutdown for sessions the caller
        # never closed; close() unregisters so we don't hold the instance.
        atexit.register(self.close)
        self._atexit = True

    def stage_dir(self) -> pathlib.Path:
        """Ensure and return the staging directory."""
        self.staging.mkdir(parents=True, exist_ok=True)
        return self.staging

    def spill(self, tabular: Tabular, *, spill_bytes: int = _DEFAULT_RESULT_SPILL_BYTES) -> Tabular:
        """Materialize *tabular* into a staging-backed holder.

        Streams the source's Arrow batches into an
        :class:`~yggdrasil.arrow.tabular.ArrowTabular` whose spill folder
        is this session's staging dir — small results stay in memory,
        larger ones spill to local IPC parts. The holder does not own the
        staging dir (the session reaps it), so dropping the result never
        deletes another result's parts.
        """
        from yggdrasil.arrow.tabular import ArrowTabular

        holder = ArrowTabular(spill_path=str(self.stage_dir()), spill_bytes=spill_bytes)
        holder.write_table(tabular)
        return holder

    def close(self) -> None:
        """Remove the session tree and stop the atexit reaper."""
        if self._closed:
            return
        self._closed = True
        if self._atexit:
            try:
                atexit.unregister(self.close)
            except Exception:
                pass
            self._atexit = False
        shutil.rmtree(self.root, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"SagaSession(id={self.session_id!r}, staging={self.staging!s})"


class Saga:
    """Unified lazy data engine over yggdrasil Tabulars.

    Parse SQL from any dialect, live-resolve the ``FROM`` sources, and
    build / execute autonomous lazy plans::

        saga = Saga(dialect="databricks")

        # FROM resolves a path/URL source live — no registration needed
        top = saga.sql("SELECT name, score FROM 'users.parquet' WHERE score > 80")

        # Lazy pipeline over any source (path, URL, or in-memory frame)
        out = (saga.scan(users_frame)
                   .filter("region = 'US'")
                   .select("name", "score")
                   .read_arrow_table())

        # Parse without executing
        node = saga.parse("SELECT * FROM t")
        plan = saga.plan("INSERT INTO 'archive.parquet' SELECT * FROM 't.parquet'")

        # Collect, spilling a large result to local disk
        big = saga.collect("SELECT * FROM 'huge.parquet'", spill=True)
    """

    __slots__ = ("_dialect", "_registry", "_session")

    def __init__(
        self,
        *,
        dialect: Dialect | str | None = None,
        registry: FunctionRegistry | None = None,
        session: SagaSession | None = None,
    ) -> None:
        self._dialect: Dialect | str | None = dialect
        self._registry: FunctionRegistry = registry if registry is not None else BUILTIN_REGISTRY
        self._session: SagaSession | None = session

    # -- Engine state ----------------------------------------------------

    @property
    def session(self) -> SagaSession:
        """The disk-staging session, created lazily on first use."""
        if self._session is None:
            self._session = SagaSession()
        return self._session

    @property
    def registry(self) -> FunctionRegistry:
        return self._registry

    @property
    def dialect(self) -> Dialect | str | None:
        return self._dialect

    def with_dialect(self, dialect: Dialect | str | None) -> Saga:
        """Return a sibling engine sharing this session but a new dialect."""
        return Saga(dialect=dialect, registry=self._registry, session=self._session)

    # -- Parsing ---------------------------------------------------------

    def parse(
        self,
        sql: str,
        *,
        dialect: Dialect | str | None = None,
        default: Any = ...,
    ) -> PlanNode | Any:
        """Parse *sql* into an immutable :class:`PlanNode` tree."""
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
        """Parse and execute *query*, live-resolving its ``FROM`` sources.

        Path/URL-shaped sources in the ``FROM`` clause are resolved through
        the IO layer automatically. Optional ad-hoc *tables* supply named
        in-memory sources for this one query (nothing is stored on the
        engine — registration will come later).
        """
        node = self.parse(query, dialect=dialect)
        return self.execute(node, tables=tables)

    def execute(
        self,
        plan: PlanNode | ExecutionPlan,
        *,
        tables: dict[str, Tabular] | None = None,
    ) -> Tabular:
        """Execute a plan node / :class:`ExecutionPlan`.

        Named *tables* are coerced live via :meth:`Tabular.new` so callers
        can hand in raw engine frames as well as Tabulars.
        """
        if isinstance(plan, ExecutionPlan):
            return plan.execute()
        if isinstance(plan, PlanNode):
            resolved = {k: Tabular.new(v) for k, v in tables.items()} if tables else None
            return plan.execute(tables=resolved)
        raise TypeError(f"Cannot execute {type(plan).__name__}; expected PlanNode or ExecutionPlan")

    def submit(
        self,
        query: str | PlanNode | ExecutionPlan,
        *,
        dialect: Dialect | str | None = None,
        tables: dict[str, Tabular] | None = None,
    ) -> ExecutionResult:
        """Build a lazy, awaitable :class:`ExecutionResult` for *query*.

        Parses (if a SQL string) and wraps the plan without running it —
        the handle executes on a background thread the first time it is
        read, awaited, or :meth:`~ExecutionResult.start`-ed. Ad-hoc
        *tables* are coerced live via :meth:`Tabular.new`.
        """
        from .plan import ExecutionResult

        if isinstance(query, str):
            node = self.parse(query, dialect=dialect)
        else:
            node = query
        if isinstance(node, ExecutionPlan):
            return ExecutionResult(node)
        resolved = {k: Tabular.new(v) for k, v in tables.items()} if tables else None
        return ExecutionResult(node, tables=resolved)

    def collect(
        self,
        query_or_plan: str | PlanNode | ExecutionPlan,
        *,
        spill: bool = False,
        dialect: Dialect | str | None = None,
        tables: dict[str, Tabular] | None = None,
    ) -> Tabular:
        """Execute and return the result, optionally spilling it to disk.

        With ``spill=True`` the result is materialized into the session's
        local-disk staging area (small results stay in memory, large ones
        spill to IPC parts), keeping peak memory bounded for big outputs.
        """
        if isinstance(query_or_plan, str):
            result = self.sql(query_or_plan, dialect=dialect, tables=tables)
        else:
            result = self.execute(query_or_plan, tables=tables)
        return self.session.spill(result) if spill else result

    # -- Lazy building ---------------------------------------------------

    def scan(self, source: Any) -> LazyTabular:
        """Open a deferred :class:`LazyTabular` over *source*.

        *source* may be a :class:`Tabular`, a path / URL (resolved through
        the IO layer), or an in-memory engine frame (arrow / polars /
        pandas / spark, wrapped via :meth:`Tabular.new`). Transformations
        accumulate in an :class:`ExecutionPlan` and only run on the first
        ``read_*``.
        """
        return self._coerce(source).lazy()

    # ``lazy`` reads more naturally for an already-held Tabular / frame.
    lazy = scan

    @staticmethod
    def _coerce(source: Any) -> Tabular:
        if isinstance(source, Tabular):
            return source
        if isinstance(source, (str, os.PathLike)):
            return Tabular.from_(source)
        return Tabular.new(source)

    # -- Lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Clean up the disk-staging session, if one was created."""
        if self._session is not None:
            self._session.close()

    def __enter__(self) -> Saga:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        sess = self._session.session_id if self._session is not None else None
        return f"Saga(dialect={self._dialect!r}, session={sess!r})"
