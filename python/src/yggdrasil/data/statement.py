"""Backend-agnostic statement abstractions.

Three concrete types, each with a clear single responsibility:

- :class:`PreparedStatement` — a value object: the SQL text plus any
  binding metadata.  No execution state.
- :class:`StatementResult` — a handle to a statement *being or having
  been* executed by some backend.  Carries the per-execution state
  (statement_id, response, materialized data) and exposes Arrow I/O via
  :class:`Tabular`.  Lifecycle hooks: ``start``, ``cancel``,
  ``refresh_status``, ``done``, ``failed``.
- :class:`StatementBatch` — a collection of pending statements + their
  in-flight / completed results.  *Not* a Tabular: the batch as a whole
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

import copy as copy_mod
import logging
import os
import re
import time
from abc import abstractmethod
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Iterable, Iterator, Literal, Mapping, Optional, TypeVar

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.disposable import Disposable
from yggdrasil.data.enums import MimeType, MimeTypes, State
from yggdrasil.io.tabular import Tabular

if TYPE_CHECKING:
    from yggdrasil.data.executor import StatementExecutor

logger = logging.getLogger(__name__)

__all__ = [
    "BatchConcatMode",
    "ExternalStatementData",
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
# ExternalStatementData
# ---------------------------------------------------------------------------


# ``{name}``-style placeholder must be a plain identifier — keeps
# ``str.replace`` substitution unambiguous and rules out anything that
# would need quoting in SQL.
_EXTERNAL_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ExternalStatementData:
    """A tabular binding that a statement carries alongside its SQL text.

    Lets a :class:`PreparedStatement` reference data by a placeholder
    name (``{text_key}``) which an engine resolves to a concrete
    expression at submit time.  Three fields:

    - ``text_key`` — the placeholder identifier; ``{text_key}`` in the
      statement text is replaced with ``text_value`` on submit.
    - ``text_value`` — the SQL fragment substituted into the text.  May
      be ``None`` initially: engines that materialize the binding
      (Spark registers a temporary view, the warehouse stages a
      Parquet volume) fill it in before substitution.  Pre-set this
      yourself when you've already arranged the storage (e.g. an
      existing :class:`VolumePath` you wrote to up front).
    - ``tabular`` — the data the placeholder represents.  Engines pull
      a frame off it via :meth:`Tabular.read_spark_frame` /
      :meth:`Tabular.read_arrow_batches` to register / stage it.

    Plain value object — engines treat it as mutable scratch and may
    rewrite ``text_value`` during materialization.
    """

    __slots__ = ("text_key", "text_value", "tabular")

    text_key: str
    text_value: Optional[str]
    tabular: Optional[Tabular]

    def __init__(
        self,
        text_key: str,
        tabular: Optional[Tabular] = None,
        *,
        text_value: Optional[str] = None,
    ) -> None:
        if not isinstance(text_key, str) or not text_key:
            raise ValueError(
                f"ExternalStatementData.text_key must be a non-empty string; "
                f"got {text_key!r}"
            )
        if not _EXTERNAL_KEY_RE.match(text_key):
            raise ValueError(
                f"ExternalStatementData.text_key {text_key!r} is not a valid "
                f"identifier; must match [A-Za-z_][A-Za-z0-9_]*"
            )
        if tabular is None and not text_value:
            raise ValueError(
                f"ExternalStatementData[{text_key!r}]: at least one of "
                f"tabular or text_value must be supplied"
            )
        self.text_key = text_key
        self.tabular = tabular
        self.text_value = text_value if text_value else None

    def __repr__(self) -> str:
        return (
            f"ExternalStatementData(text_key={self.text_key!r}, "
            f"text_value={self.text_value!r}, tabular={self.tabular!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExternalStatementData):
            return NotImplemented
        return (
            self.text_key == other.text_key
            and self.text_value == other.text_value
            and self.tabular is other.tabular
        )

    @classmethod
    def from_(
        cls,
        value: "ExternalStatementData | Tabular | str | tuple",
        *,
        text_key: Optional[str] = None,
    ) -> "ExternalStatementData":
        """Coerce ``value`` into an :class:`ExternalStatementData`.

        - Already an instance → pass-through (``text_key`` ignored).
        - :class:`Tabular` → bind under the supplied ``text_key``.
        - ``str`` → ``text_value`` only (no tabular, caller staged it).
        - ``(tabular, text_value)`` tuple → both fields set.

        Raises ``ValueError`` when ``text_key`` is required but missing.
        """
        if isinstance(value, ExternalStatementData):
            return value
        if text_key is None:
            raise ValueError(
                "ExternalStatementData.from_: text_key is required when "
                "coercing a non-ExternalStatementData value"
            )
        if isinstance(value, Tabular):
            return cls(text_key, tabular=value)
        if isinstance(value, str):
            return cls(text_key, tabular=None, text_value=value)
        if isinstance(value, tuple) and len(value) == 2:
            tabular, text_value = value
            return cls(text_key, tabular=tabular, text_value=text_value)
        raise TypeError(
            f"Cannot coerce {type(value).__name__} to ExternalStatementData"
        )


def _coerce_external_data(
    external_data: Optional[Mapping[str, "ExternalStatementData | Tabular | str | tuple"]],
) -> Optional[dict[str, ExternalStatementData]]:
    """Normalize an ``external_data`` mapping into ``{key: entry}``."""
    if not external_data:
        return None
    if not isinstance(external_data, Mapping):
        raise TypeError(
            f"external_data must be a mapping; got {type(external_data).__name__}"
        )
    out: dict[str, ExternalStatementData] = {}
    for key, value in external_data.items():
        entry = ExternalStatementData.from_(value, text_key=key)
        # If the caller used a different text_key inside an explicit
        # ExternalStatementData, the dict key wins — substitution is
        # driven by the dict key everywhere downstream.
        if entry.text_key != key:
            entry = ExternalStatementData(
                key, tabular=entry.tabular, text_value=entry.text_value,
            )
        out[key] = entry
    return out or None


# ---------------------------------------------------------------------------
# PreparedStatement
# ---------------------------------------------------------------------------


class PreparedStatement(Disposable):
    """Configuration for a single statement execution.

    Plain value object — SQL text, parameter bindings, external-table
    aliases.  Runtime/execution state lives on :class:`StatementResult`.

    Mutators (``with_text``, ``clear``) return a new instance unless
    ``inplace=True`` is passed.

    Retry config (read by :meth:`StatementResult.retry`):

    - ``retry`` — :class:`WaitingConfig` controlling the result-level
      retry loop.  ``None`` (the default) means *not retryable*.  When
      set, ``retry.retries + 1`` total attempts are made, with sleep
      between attempts driven by :meth:`WaitingConfig.sleep` (exponential
      backoff capped at ``max_interval``, terminated by ``timeout``).

    Subclasses set their own retry default by passing a
    :class:`WaitingConfig` to ``super().__init__(retry=...)`` from their
    own ``__init__`` defaults.  The base default is ``None`` — caller
    must opt in.
    """

    text: str = ""
    retry: Optional[WaitingConfig] = None
    external_data: Optional[dict[str, ExternalStatementData]] = None

    def __init__(
        self,
        text: str = "",
        key: Optional[str] = None,
        retry: Optional[WaitingConfigArg] = None,
        *,
        external_data: Optional[
            Mapping[str, "ExternalStatementData | Tabular | str | tuple"]
        ] = None,
    ):
        Disposable.__init__(self)
        self.text = str(text) if text else ""
        self.key = key or _new_key()
        # WaitingConfig.from_ accepts WaitingConfig | dict | int | float |
        # timedelta | datetime | bool, but we want None to stay None
        # (= not retryable) and only run from_ when the caller actually
        # passed something.
        self.retry = WaitingConfig.from_(retry) if retry is not None else None
        self.external_data = _coerce_external_data(external_data)

    @property
    def retryable(self) -> bool:
        """Whether a non-None retry policy has been configured.

        Convenience for the lifecycle code; ``self.retry is not None``
        works equivalently.
        """
        return self.retry is not None

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

    def with_retry(
        self,
        retry: Optional[WaitingConfigArg],
        *,
        inplace: bool = False,
    ) -> "PreparedStatement":
        """Return (or update in place) a copy with ``retry`` set.

        ``retry=None`` clears the policy (statement becomes non-retryable);
        anything else is normalized through :meth:`WaitingConfig.from_`.
        """
        target = self if inplace else copy_mod.copy(self)
        target.retry = WaitingConfig.from_(retry) if retry is not None else None
        return target

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
            coerced = cls(statement.text, key=statement.key)
            if statement.external_data:
                coerced.external_data = dict(statement.external_data)
            return coerced
        raise TypeError(f"Cannot prepare {statement!r} as {cls.__name__}.")

    @classmethod
    def prepare(cls, statement: "PreparedStatement | str", **kwargs: Any) -> "PreparedStatement":
        """Coerce + bind metadata.  Base impl handles only the text;
        subclasses override to thread parameters / external tables onto
        their typed fields.
        """
        return cls.from_(statement)

    @staticmethod
    def apply_external_substitution(
        text: str,
        external_data: Optional[Mapping[str, ExternalStatementData]],
    ) -> str:
        """Substitute every ``{text_key}`` in ``text`` with its ``text_value``.

        Engine-agnostic: the caller (Spark / warehouse / ...) is
        responsible for filling in each entry's ``text_value`` (registering
        a temp view, staging a Parquet volume, ...) before invoking this.
        Entries whose ``text_value`` is still ``None`` raise — better to
        fail loudly than silently leave an unsubstituted placeholder in
        the SQL.
        """
        if not external_data:
            return text
        rewritten = text
        for key, entry in external_data.items():
            if entry.text_value is None:
                raise ValueError(
                    f"ExternalStatementData[{key!r}].text_value is unset; "
                    f"the engine must materialize the binding before "
                    f"applying substitution"
                )
            rewritten = rewritten.replace("{%s}" % key, entry.text_value)
        return rewritten

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


class StatementResult(Tabular, Generic[PS]):
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

    @classmethod
    def default_media_type(cls) -> MimeType:
        return MimeTypes.STATEMENT_RESULT

    _state_snapshot: Optional[State] = None

    def __init__(
        self,
        statement: PS,
        *,
        key: Optional[str] = None,
        executor: Optional["StatementExecutor"] = None,
        num_try: int = 0,
        **kwargs: Any,
    ):
        self.executor = executor
        self.statement = self._PREPARED_STATEMENT_CLASS.from_(statement)
        self.key = key or self.statement.key
        self._cached_schema: Optional[Schema] = None
        self.num_try = num_try or 0
        self._state_snapshot = None
        super().__init__(**kwargs)

    # -------------------------------------------------------------------------
    # Execution lifecycle contract
    # -------------------------------------------------------------------------

    @abstractmethod
    def _compute_state(self) -> State:
        """Refresh from the backend and return the current unified :class:`State`.

        Subclass hook — every backend translates its own state
        vocabulary (Databricks ``StatementState``, Spark's local
        ``_started`` / ``_failure`` flags, pymongo command response,
        ...) into the unified enum here. Called once per ``state``
        access except inside a :meth:`state_snapshot` block, where the
        first call's result is cached and re-used.
        """

    @property
    def state(self) -> State:
        """Current unified :class:`State`.

        Reads the per-instance snapshot when a :meth:`state_snapshot`
        block is active so multiple state-derived accesses
        (``done``, ``failed``, ``started``) in the same block share a
        single ``refresh_status`` call. Outside a snapshot, every
        access goes through :meth:`_compute_state`.
        """
        cached = self._state_snapshot
        if cached is not None:
            return cached
        return self._compute_state()

    @contextmanager
    def state_snapshot(self) -> Iterator[State]:
        """Pin :attr:`state` for the duration of the block.

        Single refresh on enter — ``state`` / ``done`` / ``failed`` /
        ``started`` accesses inside the block re-use the snapshotted
        value. Re-entrant: nested ``state_snapshot()`` blocks reuse
        the outer pin.
        """
        if self._state_snapshot is not None:
            yield self._state_snapshot
            return
        snap = self._compute_state()
        self._state_snapshot = snap
        try:
            yield snap
        finally:
            self._state_snapshot = None

    @property
    def started(self) -> bool:
        """Whether the statement has been started.

        Derived from :attr:`state` — :attr:`State.is_started` is
        ``True`` for anything past :attr:`State.PENDING`. Override in a
        subclass only when ``started`` needs to be sticky across
        terminal states that the unified enum doesn't distinguish
        (e.g. Spark's ``_started`` flag stays ``True`` after a failure
        before the result is reset).
        """
        return self.state.is_started

    @property
    def done(self) -> bool:
        """Whether the statement is in a terminal state."""
        return self.state.is_done

    @property
    def failed(self) -> bool:
        """Whether the statement failed or was canceled."""
        return self.state.is_failed

    @abstractmethod
    def refresh_status(self) -> None:
        """Refresh execution state from the backend."""

    @abstractmethod
    def start(
        self,
        reset: bool = False,
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
        leak temp volumes when handling the exception).  Auto-promotes
        the underlying statement to ``retryable=True`` once if the
        failure matches a known-transient pattern declared by the
        subclass — the caller's :meth:`retry` will then pick it up.

        The state checks live inside a :meth:`state_snapshot` block so
        the ``failed`` / ``done`` reads here and inside
        ``_auto_promote_transient_retry`` share a single backend
        ``refresh_status`` call.
        """
        with self.state_snapshot():
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
    # Retry
    # -------------------------------------------------------------------------

    @property
    def retryable(self) -> bool:
        """Whether another retry attempt is allowed.

        Two gates: the statement must opt in by setting ``statement.retry``
        to a :class:`WaitingConfig`, and we must not have exhausted
        ``retry.total_try_count``.  The ``num_try`` counter records
        *completed* attempts, so the original ``start()`` counts as
        attempt 1.
        """
        cfg = self.statement.retry
        if cfg is None:
            return False
        return self.num_try < cfg.total_try_count

    def retry(
        self,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        **kwargs: Any,
    ) -> "StatementResult":
        """Retry the statement until success or attempts are exhausted.

        Loops internally driven by ``self.statement.retry``
        (a :class:`WaitingConfig`).  Each iteration calls :meth:`start`
        with ``reset=True`` and waits for terminal state; on success it
        returns immediately, on failure it sleeps via
        :meth:`WaitingConfig.sleep` and tries again.  When attempts are
        exhausted (or the WaitingConfig timeout elapses), behaves like
        :meth:`raise_for_status` if ``raise_error`` is True.

        Preconditions:

        - ``self.statement.retry`` must be a :class:`WaitingConfig`, else
          ``RuntimeError``.
        - ``self.failed`` must be True (otherwise nothing to retry —
          returns ``self``).

        ``num_try`` is incremented *before* each attempt so a crash inside
        ``start()`` still counts toward the budget.  Extra ``**kwargs``
        are forwarded to ``start()``.
        """
        cfg = self.statement.retry or WaitingConfig.default()
        total_tries = max(1, cfg.total_try_count)

        while self.num_try < total_tries:
            self.num_try += 1
            self.start(reset=True, wait=wait, raise_error=False, **kwargs)
            if self.state.is_succeeded:
                return self

        # Exhausted budget.  Surface the latest failure if asked.
        if raise_error:
            self.raise_for_status()
        return self

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
        state = self.state

        while not state.is_done:
            wait_cfg.sleep(iteration=iteration, start=start)
            iteration += 1
            state = self.state

        if raise_error:
            self.raise_for_status()

        if not self.failed:
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

    def _read_spark_frame(self, options: CastOptions):
        """Skip the Arrow round-trip when a Spark-native cache exists.

        The default :class:`Tabular` implementation collects the
        result via Arrow and rebuilds a Spark DataFrame on the
        driver. When :attr:`_persisted_data` is itself a
        Spark-backed :class:`Tabular` (e.g. :class:`SparkTabular`,
        produced by a SparkSQL fallback or by an explicit
        ``persist(data=df)``), its ``_read_spark_frame`` returns
        the inner Spark DataFrame as-is — no driver collect.
        """
        persisted = getattr(self, "_persisted_data", None)
        if persisted is not None:
            return persisted._read_spark_frame(options)
        return super()._read_spark_frame(options)


SR = TypeVar("SR", bound="StatementResult")


# ---------------------------------------------------------------------------
# StatementBatch
# ---------------------------------------------------------------------------


class StatementBatch(Generic[PS, SR]):
    """A pending queue of statements plus a map of in-flight / completed results.

    *Not* a :class:`Tabular` — a batch as a whole has no rows, only its
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
        **kwargs: Any,
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
            self._run_parallel(
                {key: (result.wait, (wait, False), {}) for key, result in self.results.items()},
                op_label="wait",
            )

        if raise_error:
            self.raise_for_status()

        # Per-result :meth:`wait` already cleared scratch for every
        # success. Skip failed ones — :meth:`retry` may re-run them
        # against the same staged source, and unlinking it here would
        # turn a transient failure into a non-recoverable PATH_NOT_FOUND
        # on the retry attempt.
        for key, result in self.results.items():
            if not result.failed:
                _safe(result.clear_temporary_resources, "clear_temporary_resources", key)
        return self

    def retry(
        self,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        **kwargs: Any,
    ) -> "StatementBatch":
        """Retry every failed result whose statement is retryable.

        Walks ``self.results`` once, picks the entries that are both
        ``failed`` and ``retryable``, and calls :meth:`StatementResult.retry`
        on each.  Honors ``self.parallel`` exactly like :meth:`wait`.

        Non-retryable failures are left alone — they'll surface through
        ``raise_for_status`` at the end if ``raise_error=True``.  Pending
        statements (never submitted) are submitted first, same as
        :meth:`wait`.
        """
        if self.statements:
            self.submit(wait=False, raise_error=False)

        # Refresh once so .failed reflects backend reality before we pick targets.
        for key, result in self.results.items():
            try:
                result.refresh_status()
            except Exception:
                logger.exception("refresh_status failed for %r before retry; continuing.", key)

        targets = {
            key: result
            for key, result in self.results.items()
            if result.failed and result.retryable
        }

        if not targets:
            if raise_error:
                self.raise_for_status()
            return self

        if self.parallel <= 1 or len(targets) <= 1:
            for key, result in targets.items():
                try:
                    result.retry(wait=wait, raise_error=False, **kwargs)
                except Exception:
                    logger.exception("retry() raised for batch item %r; continuing.", key)
        else:
            self._run_parallel(
                {
                    key: (result.retry, (), {"wait": wait, "raise_error": False, **kwargs})
                    for key, result in targets.items()
                },
                op_label="retry",
            )

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
        """Surface the latest backend failure directly — no generic wrapper.

        Walks ``self.results`` in submission order; with one or more
        failed items, propagates the *last* failed result's typed
        backend exception (e.g. :class:`SQLError` with the full
        ``DELTA_CONCURRENT_APPEND`` payload) so the caller sees the
        actual error instead of a wrapped ``RuntimeError("Batch item
        ... failed.")``.  Earlier failures are logged via
        :meth:`StatementResult.raise_for_status` so their diagnostics
        aren't swallowed by the one we re-raise.
        """
        failed_keys = [key for key, result in self.results.items() if result.failed]
        if not failed_keys:
            return self

        last_key = failed_keys[-1]
        # Surface preceding failures through the logger so the user
        # still sees them when several items in the batch failed.
        for key in failed_keys[:-1]:
            try:
                self.results[key].raise_for_status()
            except Exception:
                logger.exception(
                    "Batch item %r failed; superseded by a later batch failure.",
                    key,
                )
        # Re-raise the latest failure directly so the typed backend
        # exception (and its message) is what bubbles up.
        logger.debug("Re-raising backend failure from batch item %r.", last_key)
        self.results[last_key].raise_for_status()
        return self

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _run_parallel(
        self,
        jobs: dict[str, tuple[Any, tuple, dict]],
        *,
        op_label: str,
    ) -> None:
        """Run ``jobs`` (key → (callable, args, kwargs)) on a bounded pool.

        Exceptions are caught and logged per-key; the caller decides what
        to do with the resulting state (typically :meth:`raise_for_status`).
        """
        workers = min(self.parallel, len(jobs))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stmt-batch") as pool:
            futures = {
                pool.submit(fn, *args, **kwargs): key
                for key, (fn, args, kwargs) in jobs.items()
            }
            for fut in as_completed(futures):
                key = futures[fut]
                exc = fut.exception()
                if exc is not None:
                    logger.exception(
                        "%s() raised for batch item %r: %s", op_label, key, exc,
                    )


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