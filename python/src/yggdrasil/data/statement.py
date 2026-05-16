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

import logging
import os
import re
import time
from abc import abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Iterable, Iterator, Literal, Mapping, Optional, \
    TypeVar

import pyarrow as pa

from yggdrasil.data import Mode
from yggdrasil.data.enums import MimeType, MimeTypes, State
from yggdrasil.data.schema import Schema
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.disposable import Disposable
from yggdrasil.io.tabular import Tabular, O

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
# Single-pass matcher for ``looks_like_query`` — skips leading
# whitespace + SQL comments and asserts a query-leading keyword in
# one regex hop instead of a slice-and-rematch loop.
_SQL_LOOKS_LIKE_QUERY_RE = re.compile(
    r"\A(?:\s+|--[^\n]*(?:\n|\Z)|/\*.*?\*/)*"
    r"(?:SELECT|WITH|VALUES|TABLE|FROM)\b",
    re.IGNORECASE | re.DOTALL,
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
        # Most callers pass a non-empty ``str`` literal; the
        # ``isinstance`` check is cheaper than the unconditional
        # ``str()`` round-trip when the contract is already met.
        self.text = text if isinstance(text, str) else (str(text) if text else "")
        self.key = key if key is not None else _new_key()
        # WaitingConfig.from_ accepts WaitingConfig | dict | int | float |
        # timedelta | datetime | bool, but we want None to stay None
        # (= not retryable) and only run from_ when the caller actually
        # passed something.
        self.retry = WaitingConfig.from_(retry) if retry is not None else None
        # Avoid the function-call frame when nothing was passed —
        # 99% of statement constructions don't bind external data.
        self.external_data = _coerce_external_data(external_data) if external_data else None

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
        if inplace:
            target = self
        else:
            # Skip ``copy.copy``'s ``__reduce_ex__`` dance — same shape
            # as ``with_text`` so a copy is one ``__new__`` plus a
            # ``__dict__`` update.
            target = self.__class__.__new__(self.__class__)
            target.__dict__.update(self.__dict__)
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
        # Single-pass regex over the original string — the comment/ws
        # prefix and the leading keyword check fuse so we avoid the
        # per-iteration slice the older two-regex loop paid.
        return _SQL_LOOKS_LIKE_QUERY_RE.match(text) is not None

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
            text_value = entry.text_value
            if text_value is None:
                raise ValueError(
                    f"ExternalStatementData[{key!r}].text_value is unset; "
                    f"the engine must materialize the binding before "
                    f"applying substitution"
                )
            # ``"{" + key + "}"`` beats ``"{%s}" % key`` on small keys —
            # avoids the format-spec parse the latter pays per call.
            rewritten = rewritten.replace("{" + key + "}", text_value)
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


def _new_key(
    _ns: Callable[[], int] = time.time_ns,
    _rand: Callable[[int], bytes] = os.urandom,
) -> str:
    # ``time_ns`` avoids the float-multiply-then-int dance the older
    # ``int(time.time() * 1e6)`` did; ``os.urandom(4).hex()`` is kept
    # for the 32-bit collision-resistant tail. Default args bind the
    # globals once so the call site doesn't pay the LOAD_GLOBAL twice.
    return f"{_ns() // 1000}-{_rand(4).hex()}"


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

    def __init__(
        self,
        statement: PS,
        *,
        key: Optional[str] = None,
        executor: Optional["StatementExecutor"] = None,
        iteration: int = 0,
        start_timestamp: Optional[int] = None,
        **kwargs: Any,
    ):
        self.executor = executor
        self.statement = self._PREPARED_STATEMENT_CLASS.from_(statement)
        self.key = key or self.statement.key
        self.start_timestamp: int | None = start_timestamp
        self.iteration = iteration or 0

        self._cached_schema: Optional[Schema] = None
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
        return self._compute_state()

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

    def retry(self, wait: WaitingConfigArg = None, raise_error: bool = False, **kwargs) -> "StatementResult":
        state = self.state

        if state.is_succeeded:
            return self
        if state.is_failed:
            if self.retryable:
                return self.start(
                    reset=True,
                    wait=wait,
                    raise_error=raise_error,
                    **kwargs,
                )
            self.raise_for_status()
            # raise_for_status raised; this line is unreachable, but keep
            # the explicit return for type-checker happiness.
            return self

        # Non-terminal state — caller explicitly asked to retry something
        # still in flight.  This isn't a hot path (wait() guards against
        # it), so the error is the right answer.
        raise RuntimeError(f"{self!r} in current state {state!r} cannot be retried.")

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
    def cancel(self, wait: WaitingConfigArg = None, raise_error: bool = False, **kwargs) -> "StatementResult":
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
        # Fast-path the steady-state success branch — every executor
        # ``wait`` / ``execute`` hop pays this. One state read; no
        # contextmanager allocation when the result is fine.
        state = self._compute_state()
        if not state.is_failed:
            return None

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
        try:
            self.statement.clear_temporary_resources()
        except Exception:
            logger.exception("clear_temporary_resources failed during clear; continuing.")

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
        return False

    @property
    def elapsed_timestamp(self) -> float:
        """Time elapsed since the statement was started, in seconds."""
        if not self.start_timestamp:
            return 0.0
        return time.time() - self.start_timestamp

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

        Auto-retry: when the result reaches a terminal *failed* state and
        :attr:`retryable` is true, the wait sleeps with jittered backoff
        and invokes :meth:`retry`, then resumes polling for the
        resubmitted attempt.  Jitter decorrelates retry storms when many
        results fail on a shared upstream conflict (e.g. concurrent
        Delta appends).  The loop terminates when the statement
        succeeds, fails non-retryably, or the subclass's
        :attr:`retryable` flips to ``False`` (typically because the
        iteration / elapsed budget is exhausted).
        """
        self.start(reset=False, wait=False, raise_error=False)
        wait_cfg = WaitingConfig.from_(wait)

        if not wait_cfg:
            if raise_error:
                self.raise_for_status()
            return self

        while True:
            # Poll to terminal for the current submission.
            start = time.time()
            state = self.state
            while not state.is_done:
                wait_cfg.sleep(iteration=0, start=start, max_interval=5)
                state = self._compute_state()

            if state.is_failed and self.retryable:
                logger.info(
                    "%r failed but is retryable; resubmitting (iteration=%d).",
                    self, self.iteration,
                )
                wait_cfg.jittered_sleep(iteration=self.iteration)
                self.retry(wait=False, raise_error=False)
                continue

            break

        if raise_error:
            self.raise_for_status()

        # Only clear scratch on success.  ``raise_for_status`` already
        # cleared on failure; double-clearing is idempotent but noisy.
        if not state.is_failed:
            self.clear_temporary_resources()
        return self


SR = TypeVar("SR", bound="StatementResult")


# ---------------------------------------------------------------------------
# StatementBatch
# ---------------------------------------------------------------------------


class StatementBatch(Tabular, Generic[PS, SR]):
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
    results: "OrderedDict[str, SR]"

    def __init__(
        self,
        executor: "StatementExecutor",
        statements: Optional[Iterable["PS | str"]] = None,
        parallel: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.executor = executor
        self.results = OrderedDict()

        if parallel is None:
            parallel = 1
        elif isinstance(parallel, bool):
            parallel = os.cpu_count() if parallel else 1
        else:
            parallel = max(1, parallel)

        self.parallel = parallel

        if statements:
            self.extend(statements)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(results={self.results!r})"

    def __hash__(self):
        return hash(tuple(s for s in self.results.keys()))

    def _collect_schema(self, options: O) -> Schema:
        if options.target:
            return options.target

        if not self.results:
            return Schema.empty()  # or whatever your empty sentinel is

        it = iter(self.results.values())
        schema = next(it).collect_schema(options)
        for result in it:
            schema = schema.merge_with(result.collect_schema(options), mode=Mode.APPEND)

        self._persist_schema(schema)
        return schema

    # -------------------------------------------------------------------------
    # Mutation: add / extend / remove / clear
    # -------------------------------------------------------------------------

    def add(
        self,
        statement: "PS | str",
        key: Optional[str] = None,
    ) -> str:
        """Enqueue a statement; return its key.

        ``key`` collisions (against pending statements *or* completed
        results) raise :class:`ValueError`.
        """
        stmt = self.executor._PREPARED_STATEMENT_CLASS.from_(statement)
        if key is not None:
            if key in self:
                raise ValueError(f"Duplicate batch key {key!r}.")
            stmt.key = key
        self.results[stmt.key] = self.executor.submit_statement(stmt, start=True)
        return stmt.key

    def extend(self, statements: Iterable["PS | str"]) -> list[str]:
        """Enqueue multiple; return the list of assigned keys.

        Inlined over :meth:`add` to hoist the ``self.executor`` /
        ``_PREPARED_STATEMENT_CLASS`` / ``submit_statement`` lookups
        out of the per-item loop. The auto-key path skips ``add``'s
        collision check — :meth:`PreparedStatement.__init__` already
        mints a fresh key per statement, so no duplicates are possible
        from the auto-keyed path. Callers needing explicit keys still
        route through :meth:`add`.
        """
        executor = self.executor
        prepared_cls = executor._PREPARED_STATEMENT_CLASS
        submit = executor.submit_statement
        results = self.results
        keys: list[str] = []
        append = keys.append
        for statement in statements:
            prepared = prepared_cls.from_(statement)
            results[prepared.key] = submit(prepared, start=True)
            append(prepared.key)
        return keys

    def remove(self, key: str) -> Optional[SR]:
        """Remove an entry by key.

        Pending statement → dropped, ``None`` returned.  In-flight result
        → cancelled, scratch released, instance returned.  Unknown key →
        :class:`KeyError`.
        """
        # Pending side first — cheaper and avoids cancelling something we never started.
        result = self.results.pop(key, None)
        if result is None:
            logger.warning("remove(%r): no such result", key)
            return None
        return result

    def clear(self) -> "StatementBatch":
        """Cancel every in-flight result, drop every pending statement.

        Removes results from ``self.results`` after cancelling.  For a
        cancel-but-keep version (so callers can still inspect failures),
        call :meth:`cancel` instead.
        """
        self.results.clear()
        return self

    def clear_temporary_resources(self) -> "StatementBatch":
        """Release per-result scratch.  Does not cancel or drop anything."""
        return self

    # -------------------------------------------------------------------------
    # Container protocol
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.results)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self.results

    def __getitem__(self, key: str) -> SR:
        return self.results[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over submitted result keys (in submission order)."""
        return iter(self.results)

    def materialized(self) -> Iterator[tuple[str, SR]]:
        """Yield ``(key, result)`` pairs for every submitted result."""
        return iter(self.results.items())

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(
        self,
        reset: bool = False,
        **kwargs: Any,
    ):
        """Submit all statements in the batch."""
        parallel = self.parallel
        wait = parallel <= 1

        for result in self.results.values():
            result.start(reset=reset, wait=wait, raise_error=False, **kwargs)

    def wait(
        self,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        **kwargs: Any,
    ) -> "StatementBatch":
        """Wait for every submitted statement to reach a terminal state.

        Auto-submits any pending statements first so callers can ``add()``
        then ``wait()`` without an intermediate ``submit()``.  When
        ``parallel > 1`` the per-result waits run on a thread pool — each
        :meth:`StatementResult.wait` is I/O-bound polling.

        Per-result scratch (:meth:`StatementResult.clear_temporary_resources`)
        fires from inside :meth:`StatementResult.wait` on success — we
        don't re-sweep here because the cleanup is idempotent and the
        re-walk is pure overhead. Batch-wide scratch (e.g. warehouse-
        level :attr:`external_volume_paths`) stays under the typed
        :meth:`clear_temporary_resources` override on the subclass and
        runs when the caller closes / drops the batch.
        """
        if not self.results:
            return self

        wait = WaitingConfig.from_(wait)
        if not wait:
            return self

        parallel = self.parallel

        if parallel <= 1:
            for result in self.results.values():
                result.wait(wait=wait, raise_error=raise_error)
        else:
            # First, ensure started
            for result in self.results.values():
                result.start(wait=False, raise_error=False)

            for result in self.results.values():
                result.wait(wait=wait, raise_error=raise_error)

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

        return self

    def cancel(self, wait: WaitingConfigArg = False, raise_error: bool = False, **kwargs) -> "StatementBatch":
        """Cancel every in-flight statement; drop everything still pending.

        Idempotent.  Does *not* drop completed results from ``self.results``
        — callers may still want to inspect failure status.
        """
        for key, result in self.results.items():
            result.cancel(wait=wait, raise_error=raise_error, **kwargs)
        return self

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    @property
    def done(self) -> bool:
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

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        options = options.with_target(self.collect_schema(options))
        for key, result in self.materialized():
            yield from result._read_arrow_batches(options)

    def _write_arrow_batches(self, batches: Iterable[pa.RecordBatch], options: O) -> None:
        raise NotImplementedError("Cannot write in %r its a reading statement" % self)

    def _read_spark_frame(self, options: O) -> "SparkDataFrame":
        options = options.with_target(self.collect_schema(options))
        first = None
        for key, result in self.materialized():
            if first is None:
                first = result._read_spark_frame(options)
            else:
                first.unionByName(result._read_spark_frame(options))
        return first

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