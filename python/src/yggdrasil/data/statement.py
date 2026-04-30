"""Backend-agnostic statement abstractions.

Three concrete types, each with a clear single responsibility:

- :class:`PreparedStatement` — a value object: the SQL text plus any
  binding metadata.  No execution state.
- :class:`StatementResult` — a handle to a statement *being or having
  been* executed by some backend.  Carries the per-execution state
  (statement_id, response, materialized data) and exposes Arrow I/O via
  :class:`TabularIO`.  Lifecycle hooks: ``start``, ``cancel``,
  ``refresh_status``, ``done``, ``failed``.
- :class:`StatementBatch` — a collection of pending statements + their
  in-flight / completed results.  *Not* a TabularIO: the batch as a whole
  has no rows, only its individual results do.  Convenience for
  add-then-wait flows; parallelism is opt-in.

A few design rules the cleanup pass enforces:

- ``StatementBatch`` is a plain collection — it doesn't pretend to be a
  tabular result.  Iterating it yields keys; ``__getitem__`` looks up a
  result by key; ``materialized()`` walks finished results.
- ``add()`` returns the key (per its docstring); ``submit()`` is *not* a
  generator and returns ``self``.
- ``clear()`` cancels everything and drops it; ``clear_temporary_resources()``
  only releases per-statement scratch, never drops results.
- Subclass overrides have one place to land: ``_coerce`` for input
  normalization, ``_submit_one`` for backend dispatch, ``_after_submit``
  for any post-submit bookkeeping.
"""

from __future__ import annotations

import logging
import os
import re
import time
from abc import abstractmethod
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Iterable, Iterator, Literal, Optional, TypeVar

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.disposable import Disposable
from yggdrasil.io.tabular import TabularIO

if TYPE_CHECKING:
    from yggdrasil.data.executor import StatementExecutor

logger = logging.getLogger(__name__)

__all__ = [
    "BatchConcatMode",
    "PreparedStatement",
    "StatementBatch",
    "StatementResult",
    "PS",
    "SR",
]

BatchConcatMode = Literal[
    "vertical",
    "vertical_relaxed",
    "diagonal",
    "diagonal_relaxed",
]


_SQL_COMMENT_OR_WS_RE = re.compile(
    r"\A(?:\s+|--[^\n]*\n|--[^\n]*\Z|/\*.*?\*/)+",
    re.DOTALL,
)
_SQL_QUERY_LEAD_RE = re.compile(
    r"(?:SELECT|WITH|VALUES|TABLE|FROM)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# PreparedStatement
# ---------------------------------------------------------------------------


class PreparedStatement(Disposable):
    """Configuration for a single statement execution.

    Plain value object — SQL text, parameter bindings, external-table
    aliases.  Runtime/execution state lives on :class:`StatementResult`.

    Mutators (``with_text``, ``clear``) return a new instance unless
    ``inplace=True`` is passed.
    """

    text: str = ""

    def __init__(
        self,
        text: str = "",
        key: Optional[str] = None,
    ):
        Disposable.__init__(self)
        self.text = str(text) if text else ""
        self.key = key or _new_key()

    def with_text(self, value: str, inplace: bool = False) -> "PreparedStatement":
        """Return a copy with ``text`` replaced (or mutate in place)."""
        new_text = str(value) if value else ""
        if inplace:
            self.text = new_text
            return self
        copied = self.__class__.__new__(self.__class__)
        copied.__dict__.update(self.__dict__)
        copied.text = new_text
        return copied

    @staticmethod
    def looks_like_query(text: Any) -> bool:
        """Return ``True`` when ``text`` parses as a SQL ``SELECT``-like query.

        Skips leading whitespace and SQL comments; a string is treated as a
        query when its first keyword is ``SELECT``, ``WITH``, ``VALUES``,
        ``TABLE``, or ``FROM``.  Non-string inputs return ``False``.
        """
        if not isinstance(text, str) or not text:
            return False
        stripped = text.lstrip()
        if not stripped:
            return False
        while True:
            match = _SQL_COMMENT_OR_WS_RE.match(stripped)
            if not match:
                break
            stripped = stripped[match.end():]
        return bool(_SQL_QUERY_LEAD_RE.match(stripped))

    @classmethod
    def from_(cls, statement: "PreparedStatement | StatementResult | str") -> "PreparedStatement":
        """Coerce ``statement`` into an instance of ``cls``.

        Already-an-instance pass-through, str → ``cls(str)``, ``StatementResult``
        → recurse on its underlying statement.  Subclasses can extend this
        but the common cases all fall through here.
        """
        if isinstance(statement, cls):
            return statement
        if isinstance(statement, str):
            return cls(statement)
        if isinstance(statement, StatementResult):
            return cls.from_(statement.statement)
        # Cross-class coercion — copy text + key from another PreparedStatement.
        if isinstance(statement, PreparedStatement):
            return cls(statement.text, key=statement.key)
        raise TypeError(f"Cannot prepare {statement!r} as {cls.__name__}.")

    @classmethod
    def prepare(cls, statement: "PreparedStatement | str", **kwargs: Any) -> "PreparedStatement":
        """Coerce + bind metadata.  Base impl handles only the text;
        subclasses override to thread parameters / external tables onto
        their typed fields.
        """
        return cls.from_(statement)

    def clear_temporary_resources(self) -> None:
        """Release per-statement scratch (staged volumes, temp views, ...).

        Default is a no-op.  Subclasses that allocate scratch on
        :meth:`prepare` override this.  Idempotent — callers may invoke
        it more than once.
        """
        pass

    def clear(self) -> None:
        """Clear all state associated with this statement."""
        self.clear_temporary_resources()
        self.text = ""


PS = TypeVar("PS", bound="PreparedStatement")


def _new_key() -> str:
    return f"{int(time.time() * 1e6)}-{os.urandom(4).hex()}"


# ---------------------------------------------------------------------------
# StatementResult
# ---------------------------------------------------------------------------


class StatementResult(TabularIO, Generic[PS]):
    """Backend-agnostic handle to a running or completed statement.

    Subclasses fill in lifecycle hooks (``done``, ``failed``,
    ``refresh_status``, ``start``, ``cancel``).  The base provides:

    - polling-based :meth:`wait` that drives those hooks
    - cached schema collection
    - generic ``persist`` / ``cached`` so synchronous backends (Spark,
      in-memory) can stash a materialized result without a backend-specific
      override
    """

    _PREPARED_STATEMENT_CLASS: ClassVar[type[PreparedStatement]] = PreparedStatement

    def __init__(
        self,
        statement: PS,
        *,
        key: Optional[str] = None,
        executor: Optional["StatementExecutor"] = None,
        **kwargs: Any,
    ):
        self.executor = executor
        self.statement = self._PREPARED_STATEMENT_CLASS.from_(statement)
        self.key = key or self.statement.key
        self._cached_schema: Optional[Schema] = None
        self._persisted_data: Any = None
        super().__init__(**kwargs)

    # -------------------------------------------------------------------------
    # Execution lifecycle contract
    # -------------------------------------------------------------------------

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the statement is in a terminal state."""

    @property
    @abstractmethod
    def failed(self) -> bool:
        """Whether the statement failed or was canceled."""

    @abstractmethod
    def refresh_status(self) -> None:
        """Refresh execution state from the backend."""

    @abstractmethod
    def start(
        self,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        **kwargs: Any,
    ) -> "StatementResult":
        """Submit the statement for execution.  Idempotent on already-started results."""

    @abstractmethod
    def cancel(self) -> "StatementResult":
        """Request cancellation.  Idempotent / no-op when not started or already terminal."""

    def raise_for_status(self) -> None:
        """Raise an exception if the statement failed or was canceled.

        Releases any per-statement scratch on failure (so the caller doesn't
        leak temp volumes when handling the exception).
        """
        if not self.failed:
            return
        try:
            self.statement.clear_temporary_resources()
        except Exception:
            logger.exception("clear_temporary_resources failed during raise_for_status; continuing.")
        return self._raise_for_status()

    @abstractmethod
    def _raise_for_status(self) -> None:
        """Subclass hook: raise the backend-specific failure."""

    def clear_temporary_resources(self) -> None:
        """Sweep per-statement scratch — does NOT touch result-level state.

        Subclasses with their own scratch (cached HTTP pools, intermediate
        files, ...) override and call ``super()``.
        """
        self.statement.clear_temporary_resources()

    # -------------------------------------------------------------------------
    # Convenience
    # -------------------------------------------------------------------------

    @property
    def text(self) -> str:
        return self.statement.text

    # -------------------------------------------------------------------------
    # Wait
    # -------------------------------------------------------------------------

    def wait(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "StatementResult":
        """Poll until the statement reaches a terminal state.

        ``wait=False`` returns immediately (still respects ``raise_error``
        if the result is already failed).  Otherwise drives
        :meth:`refresh_status` on a :class:`WaitingConfig` schedule until
        ``done`` is true.
        """
        wait_cfg = WaitingConfig.from_(wait)

        if not wait_cfg:
            if raise_error:
                self.raise_for_status()
            return self

        iteration = 0
        start = time.time()

        self.refresh_status()
        while not self.done:
            wait_cfg.sleep(iteration=iteration, start=start)
            iteration += 1
            self.refresh_status()

        if raise_error:
            self.raise_for_status()

        # Successful terminal — drop any per-statement scratch (staged
        # volumes, temp views) now that we don't need them.
        try:
            self.statement.clear_temporary_resources()
        except Exception:
            logger.exception("clear_temporary_resources failed after wait; continuing.")

        return self

    # -------------------------------------------------------------------------
    # Schema (cached) / Arrow contract
    # -------------------------------------------------------------------------

    def _collect_schema(self, options: CastOptions) -> Schema:
        if self._cached_schema is None:
            self.wait()
            self._cached_schema = super()._collect_schema(options)
        return self._cached_schema


SR = TypeVar("SR", bound="StatementResult")


# ---------------------------------------------------------------------------
# StatementBatch
# ---------------------------------------------------------------------------


class StatementBatch(Generic[PS, SR]):
    """A pending queue of statements plus a map of in-flight / completed results.

    *Not* a :class:`TabularIO` — a batch as a whole has no rows, only its
    individual results do.  Iterate via :meth:`materialized` (or just walk
    ``self.results``) to drain results into Arrow, polars, etc.

    Lifecycle::

        batch.add(stmt_or_str)         # enqueue (key auto-generated)
        batch.add(stmt, key="custom")  # enqueue with explicit key
        batch.submit()                 # drain queue, hand each to the executor
        batch.wait()                   # block until all results terminal
        for key, result in batch.results.items():
            ...

    ``parallel > 1`` runs the wait phase concurrently — appropriate
    because each :meth:`StatementResult.wait` is I/O-bound polling.
    """

    executor: "StatementExecutor"
    statements: deque[PS]
    results: "OrderedDict[str, SR]"
    parallel: int

    def __init__(
        self,
        executor: "StatementExecutor",
        statements: Optional[Iterable["PS | str"]] = None,
        *,
        parallel: int = 1,
    ):
        self.executor = executor
        self.parallel = max(1, parallel or 1)
        self.statements = deque()
        self.results = OrderedDict()
        if statements:
            self.extend(statements)

    # -------------------------------------------------------------------------
    # Subclass hooks
    # -------------------------------------------------------------------------

    def _coerce(self, statement: "PS | str") -> PS:
        """Normalize ``statement`` into a backend-specific PreparedStatement.

        Default uses the base :meth:`PreparedStatement.from_`.  Subclasses
        override to coerce into their own subclass and to apply any
        batch-wide rewrites (e.g. external-table substitution).
        """
        return PreparedStatement.from_(statement)  # type: ignore[return-value]

    def _submit_one(self, stmt: PS, **kwargs: Any) -> SR:
        """Hand a single statement to the executor.

        Default delegates to ``self.executor.execute(stmt, wait=False,
        raise_error=False, **kwargs)``.  Subclasses override only when
        they need custom dispatch (e.g. routing per warehouse).
        """
        return self.executor.execute(stmt, wait=False, raise_error=False, **kwargs)

    # -------------------------------------------------------------------------
    # Mutation: add / extend / remove / clear
    # -------------------------------------------------------------------------

    def add(self, statement: "PS | str", key: Optional[str] = None) -> str:
        """Enqueue a statement; return its key.

        ``key`` collisions (against pending statements *or* completed
        results) raise :class:`ValueError`.
        """
        stmt = self._coerce(statement)
        if key is not None:
            if key in self:
                raise ValueError(f"Duplicate batch key {key!r}.")
            stmt.key = key
        self.statements.append(stmt)
        return stmt.key

    def extend(self, statements: Iterable["PS | str"]) -> list[str]:
        """Enqueue multiple; return the list of assigned keys."""
        return [self.add(s) for s in statements]

    def remove(self, key: str) -> Optional[SR]:
        """Remove an entry by key.

        Pending statement → dropped, ``None`` returned.  In-flight result
        → cancelled, scratch released, instance returned.  Unknown key →
        :class:`KeyError`.
        """
        # Pending side first — cheaper and avoids cancelling something we never started.
        for i, stmt in enumerate(self.statements):
            if stmt.key == key:
                del self.statements[i]
                return None

        result = self.results.pop(key, None)
        if result is None:
            raise KeyError(f"StatementBatch has no entry {key!r}.")
        _safe(result.cancel, "cancel", key)
        _safe(result.clear_temporary_resources, "clear_temporary_resources", key)
        return result

    def clear(self) -> "StatementBatch":
        """Cancel every in-flight result, drop every pending statement.

        Removes results from ``self.results`` after cancelling.  For a
        cancel-but-keep version (so callers can still inspect failures),
        call :meth:`cancel` instead.
        """
        self.statements.clear()
        for key, result in list(self.results.items()):
            _safe(result.cancel, "cancel", key)
            _safe(result.clear_temporary_resources, "clear_temporary_resources", key)
        self.results.clear()
        return self

    def clear_temporary_resources(self) -> "StatementBatch":
        """Release per-result scratch.  Does not cancel or drop anything."""
        for key, result in self.results.items():
            _safe(result.clear_temporary_resources, "clear_temporary_resources", key)
        return self

    # -------------------------------------------------------------------------
    # Container protocol
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.statements) + len(self.results)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self.results:
            return True
        return any(stmt.key == key for stmt in self.statements)

    def __getitem__(self, key: str) -> SR:
        try:
            return self.results[key]
        except KeyError:
            if any(stmt.key == key for stmt in self.statements):
                raise KeyError(
                    f"Batch item {key!r} has not been submitted yet; call submit() first."
                )
            raise KeyError(f"StatementBatch has no entry {key!r}.")

    def __iter__(self) -> Iterator[str]:
        """Iterate over submitted result keys (in submission order)."""
        return iter(self.results)

    def materialized(self) -> Iterator[tuple[str, SR]]:
        """Yield ``(key, result)`` pairs for every submitted result."""
        return iter(self.results.items())

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def submit(
        self,
        statements: Optional[Iterable["PS | str"]] = None,
        *,
        wait: WaitingConfigArg = False,
        raise_error: bool = False,
        **kwargs: Any,
    ) -> "StatementBatch":
        """Drain the pending queue, asking the executor to start each statement.

        Returns as soon as every statement has been handed to the backend
        (does not block until completion by default).  Pass ``wait=True``
        to fold a :meth:`wait` call in afterwards.

        On any submission exception, every result already collected is
        cancelled and the original exception propagates.
        """
        if statements:
            for stmt in statements:
                self.statements.append(self._coerce(stmt))

        try:
            while self.statements:
                stmt = self.statements.popleft()
                result = self._submit_one(stmt, **kwargs)
                self.results[result.key] = result
        except BaseException:
            # Best-effort cleanup of whatever did make it into results.
            self.cancel()
            raise

        if wait:
            self.wait(wait=wait, raise_error=raise_error)
        elif raise_error:
            self.raise_for_status()
        return self

    def wait(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "StatementBatch":
        """Wait for every submitted statement to reach a terminal state.

        Auto-submits any pending statements first so callers can ``add()``
        then ``wait()`` without an intermediate ``submit()``.  When
        ``parallel > 1`` the per-result waits run on a thread pool — each
        :meth:`StatementResult.wait` is I/O-bound polling.
        """
        if self.statements:
            self.submit(wait=False, raise_error=False)

        if not self.results:
            return self

        if self.parallel <= 1 or len(self.results) <= 1:
            # Don't raise mid-loop — let every sibling settle first so
            # cancel() / raise_for_status() see a coherent picture.
            for result in self.results.values():
                result.wait(wait=wait, raise_error=False)
        else:
            workers = min(self.parallel, len(self.results))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stmt-batch") as pool:
                futures = {
                    pool.submit(result.wait, wait, False): key
                    for key, result in self.results.items()
                }
                for fut in as_completed(futures):
                    key = futures[fut]
                    exc = fut.exception()
                    if exc is not None:
                        # wait() itself blew up — log; failed-status will be picked
                        # up by raise_for_status if requested.
                        logger.exception("wait() raised for batch item %r: %s", key, exc)

        if raise_error:
            self.raise_for_status()

        self.clear_temporary_resources()
        return self

    def cancel(self) -> "StatementBatch":
        """Cancel every in-flight statement; drop everything still pending.

        Idempotent.  Does *not* drop completed results from ``self.results``
        — callers may still want to inspect failure status.
        """
        self.statements.clear()
        for key, result in self.results.items():
            _safe(result.cancel, "cancel", key)
        return self

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    @property
    def done(self) -> bool:
        if self.statements:
            return False
        return all(result.done for result in self.results.values())

    @property
    def failed(self) -> bool:
        return any(result.failed for result in self.results.values())

    def refresh_status(self) -> "StatementBatch":
        for result in self.results.values():
            result.refresh_status()
        return self

    def raise_for_status(self) -> "StatementBatch":
        for key, result in self.results.items():
            try:
                result.raise_for_status()
            except Exception as exc:
                raise RuntimeError(f"Batch item {key!r} failed.") from exc
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(fn, label: str, key: str) -> None:
    """Run ``fn()`` swallowing exceptions with a logged warning.

    Used in cancel/cleanup hot paths where one failing entry must not
    block the rest of the batch from being torn down.
    """
    try:
        fn()
    except Exception:
        logger.exception("%s failed for batch item %r; continuing.", label, key)