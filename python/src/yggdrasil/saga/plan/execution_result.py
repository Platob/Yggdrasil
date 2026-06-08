""":class:`ExecutionResult` — a lazy, awaitable, graph-executing plan handle.

An ``ExecutionResult`` *is* an execution plan that hasn't started yet. It is
the read-side sibling of :class:`~yggdrasil.data.statement.StatementResult`:
where ``StatementResult`` wraps a :class:`PreparedStatement`, this wraps a
Saga :class:`ExecutionPlan` (or an immutable :class:`PlanNode`) and exposes
its execution as both a :class:`Tabular` and an :class:`Awaitable`.

Because a plan is fully self-describing, the handle runs it as a **graph of
inner ``ExecutionResult``s**: each independent input of the plan (the bound
source, every join's right side, every union's other side) becomes a child
``ExecutionResult``. Independent children run in **parallel**; a chain of
dependent plans nests and runs in **sequence**. Each node has an :attr:`id`
and a live :attr:`state`, and :meth:`tree` / :meth:`display` render the
graph so you can watch the executions resolve.

:class:`LazyTabular` is an alias of this class — a lazy tabular is just an
``ExecutionResult`` over a ``SelectPlan`` bound to a source, not yet started.
The transform builders (:meth:`select` / :meth:`filter` / :meth:`join` / …)
mutate the held plan while it is still idle and return ``self`` for chaining.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Generic, Iterator, TypeVar

import pyarrow as pa
import xxhash

from yggdrasil.data.options import CastOptions
from yggdrasil.dataclasses.awaitable import Awaitable
from yggdrasil.enums.state import State
from yggdrasil.io.tabular.base import Tabular

from .execution_plan import ExecutionPlan, SelectPlan
from .nodes import PlanNode
from .operation_result import OperationResult

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema

O = TypeVar("O", bound=CastOptions)

_STATE_GLYPH = {
    State.IDLE: "○",
    State.PENDING: "◔",
    State.RUNNING: "◑",
    State.SUCCEEDED: "●",
    State.FAILED: "✗",
    State.CANCELED: "⊘",
}


class ExecutionResult(Tabular[O], Awaitable, Generic[O]):
    """Lazy, awaitable, graph-executing handle to an :class:`ExecutionPlan`."""

    @classmethod
    def default_media_type(cls) -> Any:
        return None  # in-memory handle — not a wire format

    @classmethod
    def options_class(cls) -> "type[O]":
        return CastOptions  # type: ignore[return-value]

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        # Each handle is a distinct run — never coalesce in the singleton
        # cache (the held plan / tables are mutable and unhashable).
        return (cls, id(object()))

    def __init__(
        self,
        plan: ExecutionPlan | PlanNode | Tabular,
        *,
        tables: dict[str, Tabular] | None = None,
        name: str | None = None,
        max_concurrency: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(plan, (ExecutionPlan, PlanNode)):
            self._plan: ExecutionPlan | PlanNode = plan
        elif isinstance(plan, Tabular):
            # A lazy view over a concrete source — the LazyTabular shape.
            self._plan = SelectPlan(source=plan)
        else:
            raise TypeError(
                f"ExecutionResult expects an ExecutionPlan, PlanNode, or source "
                f"Tabular; got {type(plan).__name__}."
            )
        self._tables: dict[str, Tabular] | None = tables
        self._name: str | None = name
        self._max_concurrency: int | None = max_concurrency
        self._result: Tabular | OperationResult | None = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._children: list[ExecutionResult] | None = None
        self._child_for: dict[int, ExecutionResult] = {}
        # int64 id: xxh32(plan key) << 32 | timestamp_ms (project convention).
        key = f"{type(self._plan).__name__}:{id(self._plan):x}"
        self.id: int = (xxhash.xxh32(key.encode()).intdigest() << 32) | (int(time.time() * 1000) & 0xFFFFFFFF)

    # -- Accessors -------------------------------------------------------

    @property
    def plan(self) -> ExecutionPlan | PlanNode:
        return self._plan

    @property
    def source(self) -> Tabular | None:
        """The bound source, for a ``SelectPlan``-backed (lazy) handle."""
        return self._plan.source if isinstance(self._plan, SelectPlan) else None

    @property
    def label(self) -> str:
        return self._name or type(self._plan).__name__

    @property
    def short_id(self) -> str:
        # Low 24 bits of the xxh32 half (the per-plan key) — distinct per
        # node even when several are minted in the same millisecond.
        return f"{(self.id >> 32) & 0xFFFFFF:06x}"

    @property
    def result(self) -> Tabular | OperationResult | None:
        """The raw plan result once succeeded, else ``None``."""
        return self._result

    @property
    def operation_result(self) -> OperationResult | None:
        """The :class:`OperationResult` for a write-side run, else ``None``."""
        return self._result if isinstance(self._result, OperationResult) else None

    @property
    def done(self) -> bool:
        return self._state.is_done

    @property
    def failed(self) -> bool:
        return self._state.is_failed

    def __repr__(self) -> str:
        return f"ExecutionResult(#{self.short_id} {self.label} state={self._state})"

    # -- Awaitable bridge ------------------------------------------------

    def _start(self) -> None:
        self._state = State.RUNNING
        # Pure-local leaf plans (concrete in-memory inputs, no parallel
        # children) run inline in the caller's thread — no thread-spawn or
        # poll latency, so a hot `.read_arrow_table()` stays cheap. Work with
        # overlap potential (parallel child branches, or an input that is
        # itself an Awaitable such as a warehouse statement) goes to a
        # background thread so `start(wait=False)` is non-blocking.
        if not self._needs_thread():
            self._run()
            return
        thread = threading.Thread(
            target=self._run, name=f"saga-exec-{self.short_id}", daemon=True,
        )
        self._thread = thread
        thread.start()

    def _needs_thread(self) -> bool:
        # Parallel child branches are fanned out to a pool inside _run, so they
        # overlap whether or not *this* node is threaded — no need to thread the
        # parent for them (threading it only adds poll latency to a blocking
        # read). Use a background thread only when an input is a *foreign* async
        # handle (a warehouse StatementResult, a job run, …) so start(wait=False)
        # is genuinely non-blocking for I/O-bound work.
        return any(
            isinstance(inp, Awaitable) and not isinstance(inp, (ExecutionResult, ExecutionPlan))
            for inp in self._independent_inputs()
        )

    def _run(self) -> None:
        try:
            children = self._ensure_children()
            if children:
                self._schedule(children)
                for child in children:
                    if child.failed:
                        raise child.error or RuntimeError(f"child #{child.short_id} failed")
            result = self._combine()
        except BaseException as exc:  # noqa: BLE001 — surfaced via _error_for_status
            self._error = exc
            self._state = State.FAILED
            self._sleeper.set()
            return
        if self._state is State.CANCELED:
            self._sleeper.set()
            return
        self._result = result
        self._state = State.SUCCEEDED
        self._sleeper.set()

    def _schedule(self, children: list[ExecutionResult]) -> None:
        """Run the child graph — sequentially or in parallel."""
        concurrency = self._concurrency(len(children))
        if concurrency <= 1:
            for child in children:          # sequential: one at a time
                child.start(wait=True, raise_error=False)
            return
        # Parallel: fan the children out to a pool so they truly overlap —
        # each child's own execution (sync or threaded) runs on a pool worker.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(child.start, wait=True, raise_error=False) for child in children]
            for future in futures:
                future.result()

    def _combine(self) -> Tabular | OperationResult:
        """Run the plan with each child input replaced by its result."""
        plan = self._plan
        if isinstance(plan, SelectPlan):
            return plan.map_inputs(self._resolve_input).execute()
        if isinstance(plan, ExecutionPlan):
            return plan.execute()
        return plan.execute(tables=self._tables)

    def _resolve_input(self, source: Tabular) -> Tabular:
        child = self._child_for.get(id(source))
        return child.collect() if child is not None else source

    def _poll(self) -> None:
        # The worker thread owns the state transitions; nothing to refresh.
        return None

    def _error_for_status(self) -> BaseException | None:
        return self._error

    def __await__(self):
        # ``await result`` runs a not-yet-started handle rather than hanging.
        if not self.started:
            self.start(wait=False)
        return super().__await__()

    # -- Execution graph -------------------------------------------------

    def _independent_inputs(self) -> list[Tabular]:
        if isinstance(self._plan, SelectPlan):
            return self._plan.input_tabulars()
        return []

    def _ensure_children(self) -> list[ExecutionResult]:
        """Build (once) the child ``ExecutionResult`` per lazy plan input.

        Concrete in-memory inputs are already materialised — they stay as
        leaves and aren't scheduled. Inputs that are themselves plans
        (``ExecutionResult`` / ``ExecutionPlan``) become child nodes.
        """
        if self._children is not None:
            return self._children
        children: list[ExecutionResult] = []
        for inp in self._independent_inputs():
            if isinstance(inp, ExecutionResult):
                child = inp
            elif isinstance(inp, ExecutionPlan):
                child = ExecutionResult(inp)
            else:
                continue
            children.append(child)
            self._child_for[id(inp)] = child
        self._children = children
        return children

    def _concurrency(self, n: int) -> int:
        if self._max_concurrency is not None:
            return self._max_concurrency
        return n  # independent inputs — fully parallel by default

    @property
    def children(self) -> list[ExecutionResult]:
        return self._ensure_children()

    @property
    def child_mode(self) -> str:
        kids = self._ensure_children()
        if not kids:
            return "leaf"
        return "parallel" if self._concurrency(len(kids)) > 1 and len(kids) > 1 else "sequential"

    # -- Display ---------------------------------------------------------

    def graph(self) -> dict[str, Any]:
        """Structured snapshot of this node and its child executions."""
        return {
            "id": self.id,
            "short_id": self.short_id,
            "label": self.label,
            "state": str(self._state),
            "mode": self.child_mode,
            "children": [c.graph() for c in self._ensure_children()],
        }

    def tree(self) -> str:
        """Render the execution graph as an indented tree.

        Each node shows its state glyph, short id, plan label and state; a
        node with children is annotated ``∥ parallel`` or ``→ sequential``.
        """
        lines: list[str] = []
        self._render(lines, "", is_root=True, is_last=True)
        return "\n".join(lines)

    def _node_line(self) -> str:
        glyph = _STATE_GLYPH.get(self._state, "○")
        return f"{glyph} #{self.short_id} {self.label} [{self._state}]"

    def _render(self, lines: list[str], prefix: str, *, is_root: bool, is_last: bool) -> None:
        connector = "" if is_root else ("└─ " if is_last else "├─ ")
        lines.append(f"{prefix}{connector}{self._node_line()}")
        kids = self._ensure_children()
        if not kids:
            return
        child_prefix = prefix + ("" if is_root else ("   " if is_last else "│  "))
        marker = "∥ parallel" if self.child_mode == "parallel" else "→ sequential"
        lines.append(f"{child_prefix}{marker}:")
        for i, child in enumerate(kids):
            child._render(lines, child_prefix + "  ", is_root=False, is_last=i == len(kids) - 1)

    def display(self, *, live: bool = False, interval: float = 0.15) -> ExecutionResult:
        """Print the execution graph; with ``live=True`` refresh it in place.

        ``live`` starts the run (if idle) and re-renders the tree until every
        node is done — a small dashboard of the sequential / parallel node
        executions and their ids.
        """
        if not live:
            print(self.tree())
            return self
        from yggdrasil.cli.style import LiveDisplay

        if not self.started:
            self.start(wait=False)
        screen = LiveDisplay()
        try:
            while True:
                screen.update(self.tree().splitlines())
                if self.is_done:
                    break
                time.sleep(interval)
        finally:
            screen.stop()
        print(self.tree())
        return self

    # -- Tabular bridge (read triggers execution) ------------------------

    def collect(self, *, wait: Any = True, raise_error: bool = True) -> Tabular:
        """Drive the run to completion and return the result as a Tabular.

        A write-side :class:`OperationResult` is surfaced as its one-row
        metadata table (matching the plan's own Tabular contract).
        """
        if not self.started:
            self.start(wait=wait, raise_error=raise_error)
        else:
            self.wait(wait=wait, raise_error=raise_error)
        if raise_error:
            self.raise_for_status()
        result = self._result
        if result is None:
            return Tabular.new(None)
        if isinstance(result, OperationResult):
            return result.to_arrow_tabular()
        return result

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        yield from self.collect().read_arrow_batches(options)

    def _collect_schema(self, options: O) -> "Schema":
        # Fast path: a schema-preserving, idle SelectPlan can't reshape the
        # source's columns — read the schema off the source without running.
        plan = self._plan
        if (
            not self.started
            and isinstance(plan, SelectPlan)
            and plan.is_schema_preserving
            and plan.source is not None
        ):
            return plan.source.collect_schema(options)
        return self.collect().collect_schema(options)

    def _write_arrow_batches(self, batches: Any, options: O) -> None:
        raise TypeError("ExecutionResult is read-only; write into the plan's target instead.")

    def _delete(
        self,
        predicate: Any = None,
        *,
        wait: Any = True,
        missing_ok: bool = False,
        delete_staging: bool = True,
        **kwargs: Any,
    ) -> int:
        raise NotImplementedError("ExecutionResult is read-only; delete on the source instead.")

    # -- Lazy transform builders (mutate the idle SelectPlan) ------------

    def _builder_plan(self) -> SelectPlan:
        if self.started:
            raise RuntimeError(
                "Cannot transform an ExecutionResult after it has started; "
                "build the pipeline before the first read / await."
            )
        if not isinstance(self._plan, SelectPlan):
            raise TypeError(
                f"transform builders require a SelectPlan-backed handle; "
                f"this one holds a {type(self._plan).__name__}."
            )
        return self._plan

    def select(self, *columns: Any) -> ExecutionResult[O]:
        self._builder_plan().select(*columns); return self

    def drop(self, *columns: Any) -> ExecutionResult[O]:
        self._builder_plan().drop(*columns); return self

    def filter(self, predicate: Any) -> ExecutionResult[O]:
        self._builder_plan().filter(predicate); return self

    def unique(self, by: Any) -> ExecutionResult[O]:
        self._builder_plan().unique(by); return self

    def limit(self, n: int) -> ExecutionResult[O]:
        self._builder_plan().limit(n); return self

    def offset(self, n: int) -> ExecutionResult[O]:
        self._builder_plan().offset(n); return self

    def cast(self, options: Any = None, **kw: Any) -> ExecutionResult[O]:
        self._builder_plan().cast(options, **kw); return self

    def group_by(self, *keys: Any, aggregations: Any = None) -> ExecutionResult[O]:
        self._builder_plan().group_by(*keys, aggregations=aggregations); return self

    def having(self, predicate: Any) -> ExecutionResult[O]:
        self._builder_plan().having(predicate); return self

    def order_by(self, *keys: Any) -> ExecutionResult[O]:
        self._builder_plan().order_by(*keys); return self

    def with_cte(self, name: str, plan: Any) -> ExecutionResult[O]:
        self._builder_plan().with_cte(name, plan); return self

    def join(self, right: Any, on: Any, how: Any = "inner", *, suffix: str = "_right") -> ExecutionResult[O]:
        self._builder_plan().join(right, on=on, how=how, suffix=suffix); return self

    def union(self, other: Any, *, mode: Any = None) -> ExecutionResult[O]:
        if isinstance(other, Tabular):
            self._builder_plan().union(other, mode=mode); return self
        return super().union(other, mode=mode)

    def resample(self, on: Any, sampling: Any, *, partition_by: Any = None,
                 fill_strategy: str = "ffill") -> ExecutionResult[O]:
        self._builder_plan().resample(on, sampling, partition_by=partition_by, fill_strategy=fill_strategy)
        return self

    def lazy(self) -> ExecutionResult[O]:
        return self

    def copy(self) -> ExecutionResult[O]:
        plan = self._plan.copy() if isinstance(self._plan, (ExecutionPlan,)) else self._plan
        return ExecutionResult(plan, tables=self._tables, name=self._name,
                               max_concurrency=self._max_concurrency)

    # -- Autonomous write-plan helpers (carry the lazy SELECT as source) -

    def into(self, target: Tabular, *, mode: Any = None) -> Any:
        """Build an :class:`InsertPlan` feeding this lazy SELECT into *target*."""
        from .execution_plan import InsertPlan

        plan = self._builder_plan().copy()
        if self.source is not None:
            plan.bind(self.source)
        return InsertPlan(target=target, source=plan, mode=mode)

    def merge_into(self, target: Tabular, *, on: Any) -> Any:
        """Open a :class:`MergePlan` with this lazy SELECT as the source."""
        from .execution_plan import MergePlan

        plan = self._builder_plan().copy()
        if self.source is not None:
            plan.bind(self.source)
        return MergePlan(target=target, source=plan).on(on)
