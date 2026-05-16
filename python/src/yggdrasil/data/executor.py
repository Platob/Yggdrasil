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
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Generic, Iterable, Mapping, Optional, TypeVar

from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.disposable import Disposable
from yggdrasil.io.session import Session
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

    def __post_init__(self):
        if self.parallel is not None:
            if isinstance(self.parallel, bool):
                object.__setattr__(self, "parallel", os.cpu_count() if self.parallel else 1)
            elif self.parallel <= 0:
                raise ValueError("Parallelism must be positive or None")

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


class StatementExecutor(Session, Disposable, Generic[PS, SR, SB]):
    """Abstract base for backend-specific statement executors.

    A :class:`StatementExecutor` IS a :class:`Session` over a transport
    that speaks SQL instead of HTTP: the same prepare → send pipeline
    drives both, and the same singleton-by-config + pickle pattern
    keeps connection pools and in-flight result maps shared across
    callers in-process.

    Subclasses implement exactly one hook — :meth:`_submit_statement` —
    which turns a coerced :class:`PreparedStatement` into a backend-
    specific :class:`StatementResult`. The base provides the
    Session-shaped surface:

    - :meth:`prepare` — coerce raw input into the typed
      :class:`PreparedStatement` subclass (analogue of
      :meth:`Session.prepare_request_before_send`),
    - :meth:`send` — dispatch a prepared statement and return its
      :class:`StatementResult`. ``lazy=True`` returns an idled
      result whose backend submission is deferred until
      :meth:`StatementResult.start` fires — same shape as
      :meth:`Session.send` with ``lazy=True``,
    - :meth:`execute` / :meth:`execute_many` — kwargs-friendly sugar
      that resolves :class:`ExecutionOptions` and waits.

    ``submit_statement`` is kept as a backward-compatible alias that
    routes through :meth:`send`.

    Singleton + pickle
    ------------------
    Concrete subclasses opt into the inherited :class:`Session` /
    :class:`Singleton` cache by:

    1. setting ``_SINGLETON_TTL = None`` (process-lifetime caching),
    2. overriding :meth:`_singleton_key` to project the identity-
       bearing constructor arguments into a hashable tuple,
    3. guarding ``__init__`` with ``if getattr(self, "_initialized",
       False): return`` so Python's re-entry after a cache hit doesn't
       clobber live state,
    4. extending ``_TRANSIENT_STATE_ATTRS`` with any non-picklable
       handles they hold (locks, urllib3 pools, live SDK sessions).

    The base default — ``_SINGLETON_TTL = ...`` (from :class:`Singleton`) —
    keeps caching opt-in so executor subclasses that genuinely don't
    have a stable identity (a hand-rolled test double, an anonymous
    executor) still work without surprise sharing.

    Class-level configuration
    -------------------------
    The prepared / response / batch types are pinned on the inherited
    :class:`Session` ClassVars — :attr:`Session._PREPARED_CLASS` /
    :attr:`Session._RESPONSE_CLASS` / :attr:`Session._BATCH_CLASS` —
    overridden here to the SQL-shaped defaults. Older subclasses that
    pin the historical ``_PREPARED_STATEMENT_CLASS`` /
    ``_STATEMENT_RESULT_CLASS`` / ``_STATEMENT_BATCH_CLASS`` names still
    work; :meth:`__init_subclass__` mirrors either set of names onto
    the other so the prepare → send pipeline reaches the right
    concrete type regardless of which alias the subclass pinned.
    """

    max_workers: Optional[int] = None

    # SQL-shaped overrides for the inherited Session ClassVars.
    _PREPARED_CLASS: ClassVar[type[PreparedStatement]] = PreparedStatement
    _RESPONSE_CLASS: ClassVar[type[StatementResult]] = StatementResult
    _BATCH_CLASS: ClassVar[type[StatementBatch]] = StatementBatch

    # Backward-compat aliases (long names that several subclasses pin).
    # ``__init_subclass__`` keeps the short and long names in lockstep.
    _PREPARED_STATEMENT_CLASS: ClassVar[type[PreparedStatement]] = PreparedStatement
    _STATEMENT_RESULT_CLASS: ClassVar[type[StatementResult]] = StatementResult
    _STATEMENT_BATCH_CLASS: ClassVar[type[StatementBatch]] = StatementBatch

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Source-of-truth precedence: the long ``_STATEMENT_*`` name
        # when the subclass set it explicitly, otherwise the short
        # alias. Either way the prepare/send pipeline ends up seeing
        # the right concrete type without the subclass restating both.
        for short, long in (
            ("_PREPARED_CLASS", "_PREPARED_STATEMENT_CLASS"),
            ("_RESPONSE_CLASS", "_STATEMENT_RESULT_CLASS"),
            ("_BATCH_CLASS", "_STATEMENT_BATCH_CLASS"),
        ):
            own = cls.__dict__
            if long in own and short not in own:
                setattr(cls, short, own[long])
            elif short in own and long not in own:
                setattr(cls, long, own[short])

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        """Fall back to the generic :class:`Singleton` key.

        :class:`Session._singleton_key` projects ``base_url`` through
        :meth:`URL.from_`, which raises on ``None`` — SQL executors
        don't have an HTTP-style ``base_url``, so the base shape goes
        back to ``(cls, args, kwargs-items)``. Concrete subclasses
        (``SQLEngine`` / ``SparkStatementExecutor`` / …) override this
        to project their own identity-bearing arguments.
        """
        return (cls, args, tuple(sorted(kwargs.items())))

    def __new__(
        cls,
        *args: Any,
        singleton_ttl: Any = ...,
        **kwargs: Any,
    ) -> "StatementExecutor":
        # Bypass :class:`Session.__new__` (which gates on a truthy
        # ``base_url`` for the HTTP path) and route directly through
        # :meth:`Singleton.__new__`. Executors key off their own
        # backend identity, not a URL.
        return Singleton.__new__(cls, *args, singleton_ttl=singleton_ttl, **kwargs)

    def __getnewargs_ex__(self) -> "tuple[tuple, dict]":
        # Session's default reads ``self.base_url`` / ``self.key`` —
        # SQL executors don't have those. Hand pickle empty args;
        # ``__setstate__`` (from :class:`Singleton`) restores the
        # identity-bearing fields from the pickled ``__dict__``
        # afterwards.
        return (), {}

    def __init__(self, *args: Any, **kwargs: Any):
        # Idempotent init — Singleton's ``__new__`` may hand back a
        # cache hit Python is about to re-invoke ``__init__`` on, and
        # we don't want to flush whatever state the live instance has.
        # Subclasses that override should set ``_initialized = True``
        # at the end of their own ``__init__`` and start with the
        # same guard.
        if getattr(self, "_initialized", False):
            return
        # Skip :class:`Session.__init__` (HTTP fields — base_url /
        # verify / pool_maxsize / headers / waiting / auth) and route
        # to :class:`Disposable` directly. SQL executors don't carry
        # HTTP transport state; the per-backend ``__init__`` handles
        # its own setup.
        Disposable.__init__(self)
        self._initialized = True

    # -------------------------------------------------------------------------
    # Subclass contract
    # -------------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Session-shaped surface: prepare / send
    # ------------------------------------------------------------------

    def prepare(self, statement: "PS | PreparedStatement | str") -> PS:
        """Coerce *statement* into this executor's prepared-statement type.

        Mirrors :meth:`Session.prepare_request_before_send`: takes
        whatever the caller passed (raw string, cross-backend
        :class:`PreparedStatement`, already-typed instance) and
        returns the concrete :attr:`_PREPARED_CLASS` every downstream
        hook expects. Subclasses that need to inject per-statement
        defaults (warehouse routing, catalog binding) override this —
        same shape as Session's hook.
        """
        return self._coerce_statement(statement)

    def send(
        self,
        statement: "PS | PreparedStatement | str",
        *,
        lazy: bool = False,
    ) -> SR:
        """Dispatch *statement* and return its tracking :class:`StatementResult`.

        Mirrors :meth:`Session.send`. ``lazy=False`` (default) routes
        the prepared statement through :meth:`_submit_statement`
        eagerly — the result comes back already in flight (or
        already terminal for synchronous backends). ``lazy=True``
        returns the idled :class:`StatementResult` whose backend
        submission is deferred until :meth:`StatementResult.start`
        fires, matching the Session ``lazy=True`` shape for HTTP.

        The returned result is always bound to this executor — every
        subclass ``_submit_statement`` is supposed to thread
        ``executor=self`` through the constructor, but that's easy
        to forget and downstream code (``StatementResult.wait``,
        ``retry``, ``raise_for_status``) needs the back-reference.
        Setting it here when it's missing makes the contract
        enforceable from one place instead of audited per backend.
        """
        prepared = self.prepare(statement)
        result = self._submit_statement(prepared, start=not lazy)
        if getattr(result, "executor", None) is None:
            result.executor = self
        return result

    def submit_statement(self, statement: PS, start: bool = True) -> SR:
        """Backward-compatible alias for :meth:`send`.

        Older call sites (and most of the codebase) reach for
        ``executor.submit_statement(...)``; route through
        :meth:`send` so the lazy / non-lazy plumbing stays in one
        place.
        """
        return self.send(statement, lazy=not start)

    @abstractmethod
    def _submit_statement(self, statement: PS, start: bool = True) -> SR:
        """Hand ``statement`` to the backend and return a tracking result.

        The result need not have completed — callers use
        :meth:`StatementResult.wait` to block when needed.
        """

    # ------------------------------------------------------------------
    # Session abstract hooks — SQL executors don't speak HTTP
    # ------------------------------------------------------------------

    def _local_send(self, request: Any, config: Any) -> Any:
        """Stubs out the HTTP local-send hook from :class:`Session`.

        :class:`StatementExecutor` IS a :class:`Session` (singleton +
        pickle + prepare/send pattern) but the transport is SQL, not
        HTTP — :meth:`_submit_statement` is the analogous concrete
        hook. Raising here keeps :meth:`Session.send` from accidentally
        routing a :class:`PreparedRequest` through a SQL executor;
        :meth:`StatementExecutor.send` drives the prepare → submit
        pipeline directly without touching this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is a SQL executor; the HTTP "
            "_local_send pathway is not applicable. Submit a "
            "PreparedStatement via .send() / .submit_statement() / "
            ".execute() instead."
        )

    def _build_idle_response(self, request: Any, config: Any) -> Any:
        """Idle-result shim — SQL executors route through
        :meth:`send(lazy=True)` directly, which builds the idled
        :class:`StatementResult` via :meth:`_submit_statement(start=False)`
        without re-entering this method."""
        raise NotImplementedError(
            f"{type(self).__name__}: use .send(statement, lazy=True) "
            "to build an idled StatementResult."
        )

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
        start: bool = True,
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

    def _execute(self, statement: PS, options: ExecutionOptions, start: bool = True) -> SR:
        """Hot-path execution: submit + wait/raise per ``options``.

        Subclasses can override to add cross-cutting behavior (logging,
        retries, metrics) without re-implementing coercion or kwargs
        handling.  The default implementation:

        1. Calls :meth:`_submit_statement`.
        2. Tracks the result in ``self._live_results`` for dispose.
        3. Honors ``options.wait`` / ``options.raise_error``.
        """
        result = self._submit_statement(statement, start=start)

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
        batch = self.batch(
            statements=statements, parallel=opts.parallel,
            **batch_kwargs
        )
        batch.wait(wait=opts.wait, raise_error=opts.raise_error)
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
        if raise_error:
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