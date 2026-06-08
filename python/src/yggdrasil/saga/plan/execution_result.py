""":class:`ExecutionResult` — a lazy, awaitable handle to a plan run.

The read-side sibling of :class:`~yggdrasil.data.statement.StatementResult`.
Where ``StatementResult`` wraps a :class:`PreparedStatement` and exposes its
backend execution as a :class:`Tabular` + :class:`Awaitable`,
``ExecutionResult`` wraps a Saga :class:`ExecutionPlan` (or an immutable
:class:`PlanNode`) and exposes *its* execution the same way:

- as a :class:`Tabular` — reading it runs the plan once and streams the
  result rows (lazy: nothing executes until the first read / await);
- as an :class:`Awaitable` — ``start`` / ``wait`` / ``cancel`` lifecycle,
  ``state`` / ``done`` / ``progress`` introspection, and ``await result``.

Execution runs on a background thread so ``start(wait=False)`` returns
immediately and ``await`` / ``wait`` are honest — the plan (which may itself
front an async source such as a warehouse statement) resolves off the
calling thread. The result is cached, so reads after completion are cheap
and repeatable.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Generic, Iterator, TypeVar

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.dataclasses.awaitable import Awaitable
from yggdrasil.enums.state import State
from yggdrasil.io.tabular.base import Tabular

from .execution_plan import ExecutionPlan
from .nodes import PlanNode
from .operation_result import OperationResult

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema

O = TypeVar("O", bound=CastOptions)


class ExecutionResult(Tabular[O], Awaitable, Generic[O]):
    """Lazy, awaitable handle to the execution of an :class:`ExecutionPlan`."""

    @classmethod
    def default_media_type(cls) -> Any:
        return None  # in-memory handle — not a wire format

    @classmethod
    def options_class(cls) -> "type[O]":
        return CastOptions  # type: ignore[return-value]

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        # Each handle is a distinct run — never coalesce in the singleton
        # cache (the held plan / tables are mutable and unhashable). Mirror
        # ExecutionPlan's per-instance key.
        return (cls, id(object()))

    def __init__(
        self,
        plan: ExecutionPlan | PlanNode,
        *,
        tables: dict[str, Tabular] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._plan: ExecutionPlan | PlanNode = plan
        self._tables: dict[str, Tabular] | None = tables
        self._result: Tabular | OperationResult | None = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    # -- Accessors -------------------------------------------------------

    @property
    def plan(self) -> ExecutionPlan | PlanNode:
        return self._plan

    @property
    def result(self) -> Tabular | OperationResult | None:
        """The raw plan result once succeeded, else ``None``.

        A SELECT-side plan yields a :class:`Tabular`; a write-side plan
        (INSERT / MERGE / …) yields an :class:`OperationResult`.
        """
        return self._result

    @property
    def operation_result(self) -> OperationResult | None:
        """The :class:`OperationResult` for a write-side run, else ``None``."""
        return self._result if isinstance(self._result, OperationResult) else None

    # Friendly terminal-state aliases (parity with StatementResult).
    @property
    def done(self) -> bool:
        return self._state.is_done

    @property
    def failed(self) -> bool:
        return self._state.is_failed

    def __repr__(self) -> str:
        return f"ExecutionResult(state={self._state}, plan={type(self._plan).__name__})"

    # -- Awaitable bridge ------------------------------------------------

    def _start(self) -> None:
        self._state = State.RUNNING
        thread = threading.Thread(
            target=self._run, name=f"saga-exec-{id(self):x}", daemon=True,
        )
        self._thread = thread
        thread.start()

    def _run(self) -> None:
        try:
            result = self._execute_plan()
        except BaseException as exc:  # noqa: BLE001 — surfaced via _error_for_status
            self._error = exc
            self._state = State.FAILED
            self._sleeper.set()
            return
        # A late cancel wins — drop the result rather than flip back to done.
        if self._state is State.CANCELED:
            self._sleeper.set()
            return
        self._result = result
        self._state = State.SUCCEEDED
        self._sleeper.set()

    def _execute_plan(self) -> Tabular | OperationResult:
        if isinstance(self._plan, ExecutionPlan):
            return self._plan.execute()
        return self._plan.execute(tables=self._tables)

    def _poll(self) -> None:
        # The worker thread owns the state transitions; nothing to refresh.
        return None

    def _error_for_status(self) -> BaseException | None:
        return self._error

    def __await__(self):
        # ``await result`` should *run* a not-yet-started handle, not hang
        # waiting on an idle one. Kick off the background execution, then
        # defer to the base async wait loop.
        if not self.started:
            self.start(wait=False)
        return super().__await__()

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
