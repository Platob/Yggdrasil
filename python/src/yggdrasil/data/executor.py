"""Backend-agnostic statement executor.

Public surface
--------------
- :class:`ExecutionOptions` — cross-backend execution policy (waiting,
  raise-on-failure, parallelism).  *Not* statement configuration —
  parameters, external tables, byte/row limits, routing hints all live as
  typed fields on the :class:`PreparedStatement` subclass.
- :class:`StatementExecutor` — abstract base with a single subclass hook
  (:meth:`_submit_statement`).  Coercion, batching, lifecycle, dispose
  semantics, and the ``execute`` / ``execute_many`` driver methods are
  provided here.

Subclassing
-----------
Subclasses pin their concrete types via the three :class:`ClassVar`
attributes ``_PREPARED_STATEMENT_CLASS``, ``_STATEMENT_RESULT_CLASS``,
``_STATEMENT_BATCH_CLASS``, and implement :meth:`_submit_statement`.
Cross-cutting behavior (logging, retries, metrics) is best added by
overriding :meth:`_execute` — it sees an already-coerced statement and a
resolved :class:`ExecutionOptions`, so it doesn't have to re-implement
the kwargs dance.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Generic, Iterable, Mapping, Optional, TypeVar

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.disposable import Disposable
from .statement import (
    PreparedStatement,
    StatementBatch,
    StatementResult,
)

__all__ = [
    "ExecutionOptions",
    "StatementExecutor",
]


# Forward-declared module-level singleton — instantiated after
# :class:`ExecutionOptions` is defined.
_DEFAULT_EXECUTION_OPTIONS: "ExecutionOptions"

logger = logging.getLogger(__name__)


PS = TypeVar("PS", bound="PreparedStatement")
SR = TypeVar("SR", bound="StatementResult")
SB = TypeVar("SB", bound="StatementBatch")


# ---------------------------------------------------------------------------
# ExecutionOptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    """Cross-backend execution policy.

    Decoupled from statement-level config (parameters, external tables,
    byte/row limits, etc.) which lives as typed fields on the concrete
    :class:`PreparedStatement` subclass.  An ``ExecutionOptions`` instance
    only describes *how* to run, never *what*.

    Fields
    ------
    wait
        Waiting policy.  ``True`` (default polling), ``False`` (return
        once submitted), or a :class:`WaitingConfig` for custom timing.
    raise_error
        Whether to raise on a backend-reported failure.  When ``False``,
        the caller inspects ``result.failed`` / ``result.raise_for_status``
        themselves.
    parallel
        Used only by :meth:`StatementExecutor.execute_many`.  ``None`` =
        executor default.  ``<= 1`` = sequential wait.

    Construction
    ------------
    Build with ``ExecutionOptions(wait=False, parallel=4)`` or coerce
    from a kwargs dict via :meth:`from_kwargs`.  :meth:`replace` returns a
    derived options object with overrides applied — useful for layered
    overrides (e.g. a batch-level default + per-statement tweaks).
    """

    wait: WaitingConfigArg = True
    raise_error: bool = True
    parallel: Optional[int] = None

    @classmethod
    def from_(
        cls,
        value: "ExecutionOptions | Mapping[str, Any] | None" = None,
        **overrides: Any,
    ) -> "ExecutionOptions":
        """Coerce to an :class:`ExecutionOptions`, applying any overrides.

        - ``None`` -> defaults
        - an existing instance -> returned with ``replace()`` applied
        - a Mapping -> constructed from it, then overrides applied
        """
        if value is None:
            # Reuse the singleton — every executor ``execute`` /
            # ``execute_many`` call hits this path when the caller
            # didn't supply an options object.
            base = _DEFAULT_EXECUTION_OPTIONS
        elif isinstance(value, cls):
            base = value
        elif isinstance(value, Mapping):
            base = cls(**value)
        else:
            raise TypeError(f"Cannot coerce {type(value).__name__} to ExecutionOptions")
        return replace(base, **overrides) if overrides else base

    def with_wait(self, wait: WaitingConfigArg) -> "ExecutionOptions":
        return replace(self, wait=wait)

    def with_raise_error(self, raise_error: bool) -> "ExecutionOptions":
        return replace(self, raise_error=raise_error)

    def with_parallel(self, parallel: Optional[int]) -> "ExecutionOptions":
        return replace(self, parallel=parallel)

    @property
    def waits(self) -> bool:
        """True if this policy will block on completion at all."""
        # Fast-path the common scalar values that don't need a
        # ``WaitingConfig.from_`` roundtrip — ``True``/``False`` are
        # what ``execute`` / ``execute_many`` pass by default; numeric
        # forms are what callers supply for "wait at most X seconds".
        w = self.wait
        if w is True:
            return True
        if w is False or w is None:
            return False
        if isinstance(w, (int, float)) and not isinstance(w, bool):
            return w > 0
        return bool(WaitingConfig.from_(w))


_DEFAULT_EXECUTION_OPTIONS = ExecutionOptions()


# ---------------------------------------------------------------------------
# StatementExecutor
# ---------------------------------------------------------------------------


class StatementExecutor(Disposable, ABC, Generic[PS, SR, SB]):
    """Abstract base for backend-specific statement executors.

    Subclasses implement exactly one hook — :meth:`_submit_statement` —
    which turns a coerced :class:`PreparedStatement` into a backend-
    specific :class:`StatementResult`.  Everything else (coercion,
    batching, lifecycle, dispose, options resolution) is provided here.

    Class-level configuration
    -------------------------
    ``_PREPARED_STATEMENT_CLASS`` / ``_STATEMENT_RESULT_CLASS`` /
    ``_STATEMENT_BATCH_CLASS`` let subclasses pin concrete types.  They
    are :class:`ClassVar` so they don't leak into ``__init__`` or ``__eq__``.
    """

    max_workers: Optional[int] = None

    _PREPARED_STATEMENT_CLASS: ClassVar[type[PreparedStatement]] = PreparedStatement
    _STATEMENT_RESULT_CLASS: ClassVar[type[StatementResult]] = StatementResult
    _STATEMENT_BATCH_CLASS: ClassVar[type[StatementBatch]] = StatementBatch

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    # -------------------------------------------------------------------------
    # Subclass contract
    # -------------------------------------------------------------------------

    @abstractmethod
    def _submit_statement(self, statement: PS) -> SR:
        """Hand ``statement`` to the backend and return a tracking result.

        The result need not have completed — callers use
        :meth:`StatementResult.wait` to block when needed.
        """

    # -------------------------------------------------------------------------
    # Coercion
    # -------------------------------------------------------------------------

    def _coerce_statement(self, statement: "PS | PreparedStatement | str") -> PS:
        """Coerce ``statement`` into ``_PREPARED_STATEMENT_CLASS``.

        Foreign-typed :class:`PreparedStatement` instances are rebuilt so
        subclasses always see their own concrete type — important when the
        subclass adds fields (parameters, external tables, …).
        """
        cls = self._PREPARED_STATEMENT_CLASS
        if isinstance(statement, cls):
            return statement  # type: ignore[return-value]
        return cls.from_(statement)  # type: ignore[return-value]

    # -------------------------------------------------------------------------
    # Single-statement execution
    # -------------------------------------------------------------------------

    def execute(
        self,
        statement: "PS | PreparedStatement | str",
        *,
        options: Optional[ExecutionOptions] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> SR:
        """Submit a single statement and optionally wait for completion.

        Two ways to pass execution policy:

        - Per-call kwargs ``wait`` / ``raise_error`` (ergonomic, matches
          the previous public API).
        - An :class:`ExecutionOptions` via ``options=`` (when you want to
          reuse the same policy across many calls or compose from layered
          defaults).

        The two are merged: ``options`` provides the base, kwargs override
        any field they explicitly set.  Unknown kwargs go nowhere — they
        are not forwarded to the backend.  Use a typed
        :class:`PreparedStatement` subclass for backend-specific
        configuration (parameters, byte limits, routing, etc.).
        """
        opts = self._resolve_options(options, wait=wait, raise_error=raise_error)
        coerced = self._coerce_statement(statement)
        return self._execute(coerced, opts)

    def _execute(self, statement: PS, options: ExecutionOptions) -> SR:
        """Hot-path execution: submit + wait/raise per ``options``.

        Subclasses can override to add cross-cutting behavior (logging,
        retries, metrics) without re-implementing coercion or kwargs
        handling.  The default implementation:

        1. Calls :meth:`_submit_statement`.
        2. Tracks the result in ``self._live_results`` for dispose.
        3. Honors ``options.wait`` / ``options.raise_error``.
        """
        result = self._submit_statement(statement)

        if options.waits:
            result.wait(wait=options.wait, raise_error=options.raise_error)
        elif options.raise_error:
            # Surface eager backend rejections even when not waiting.
            result.raise_for_status()

        return result

    # -------------------------------------------------------------------------
    # Batch execution
    # -------------------------------------------------------------------------

    def execute_many(
        self,
        statements: Iterable["PS | PreparedStatement | str"],
        *,
        options: Optional[ExecutionOptions] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        parallel: Optional[int] = None,
        **batch_kwargs: Any,
    ) -> SB:
        """Run several statements as a batch and return the populated batch.

        Convenience wrapper around :meth:`batch`: enqueues every statement,
        submits, and (by default) waits.  ``parallel`` controls the wait
        phase only — submission itself is sequential, since most backends
        either accept fast or reject fast.

        ``**batch_kwargs`` are forwarded to the batch constructor (e.g.
        ``external_paths=`` for :class:`WarehouseStatementBatch`).
        """
        opts = self._resolve_options(
            options, wait=wait, raise_error=raise_error, parallel=parallel,
        )
        batch = self.batch(statements=statements, parallel=opts.parallel, **batch_kwargs)
        # The batch's submit/wait honors the same options.
        batch.submit(wait=opts.wait, raise_error=opts.raise_error)
        return batch

    def batch(
        self,
        statements: Optional[Iterable["PS | PreparedStatement | str"]] = None,
        *,
        executor: "StatementExecutor | None" = None,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> SB:
        """Construct a batch bound to this executor."""
        return self._STATEMENT_BATCH_CLASS(
            statements=statements,
            executor=self if executor is None else executor,
            parallel=parallel,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Options resolution
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_options(
        options: Optional[ExecutionOptions],
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        parallel: Optional[int] = None,
    ) -> ExecutionOptions:
        """Merge an optional base ``options`` with per-call kwargs.

        Per-call kwargs override base fields only when they differ from
        the dataclass defaults — so passing ``options=opts`` and no kwargs
        leaves ``opts`` intact, while ``options=opts, wait=False`` turns
        the wait off without disturbing the rest.
        """
        if options is None:
            # Steady state: every ``execute()`` / ``execute_many()`` hit
            # with the public defaults lands here. Reuse the singleton
            # when nothing diverges so the kwargless path is allocation-
            # free.
            if wait is True and raise_error is True and parallel is None:
                return _DEFAULT_EXECUTION_OPTIONS
            return ExecutionOptions(wait=wait, raise_error=raise_error, parallel=parallel)

        # Only apply overrides that diverge from the dataclass defaults.
        # This lets callers do `executor.execute(stmt, options=opts)`
        # without accidentally clobbering opts.wait with the kwarg default.
        # Inline the comparisons against the singleton defaults — saves
        # one ``ExecutionOptions()`` allocation per call (was the most
        # expensive line of the resolver).
        overrides: dict[str, Any] = {}
        if wait is not True:
            overrides["wait"] = wait
        if raise_error is not True:
            overrides["raise_error"] = raise_error
        if parallel is not None:
            overrides["parallel"] = parallel
        if not overrides:
            return options
        return replace(options, **overrides)

    # -------------------------------------------------------------------------
    # Bulk lifecycle
    # -------------------------------------------------------------------------

    def cancel_all(self) -> None:
        """Best-effort cancel every live result this executor has produced."""
        pass
    # -------------------------------------------------------------------------
    # Disposable
    # -------------------------------------------------------------------------

    def _release(self, committed: bool = False) -> None:
        """Cancel outstanding work on dispose.

        Subclasses holding backend connections override and call
        ``super()._release()`` so the cancel loop runs before connection
        teardown.
        """
        self.cancel_all()