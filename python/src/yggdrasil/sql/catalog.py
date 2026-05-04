"""Source registry — what ``FROM <name>`` resolves to.

A :class:`SqlContext` is a dict-like mapping from table name to
something that can answer Arrow batches: a :class:`TabularIO`, a
:class:`pyarrow.Table` / :class:`pyarrow.RecordBatch`, a polars /
pandas / Spark DataFrame, a list of row dicts, or a path string that
yggdrasil can open as a :class:`TabularIO`.

Anything we accept goes through :func:`coerce_source` on registration
so the rest of the executor only ever sees :class:`TabularIO`. That
keeps the Polars-SQL fast path symmetric with the Arrow fallback —
both drain Arrow batches from a registered :class:`TabularIO`.

The module also keeps a process-wide :data:`default_context` so
``yggdrasil.sql.register(...)`` and ``yggdrasil.sql.sql(...)`` work
without an explicit context object — handy for notebooks and the
common one-shot ``sql("SELECT ... FROM trades")`` pattern.
"""

from __future__ import annotations

import threading
from typing import Any, Iterator, Mapping, MutableMapping

from yggdrasil.io.buffer.base import TabularIO

__all__ = [
    "SqlContext",
    "coerce_source",
    "default_context",
    "register",
    "deregister",
    "registered",
]


def coerce_source(obj: Any) -> TabularIO:
    """Lift *obj* into a :class:`TabularIO`.

    Thin shim around :meth:`TabularIO.from_` plus a path-string
    fast path. The bulk of the type-handling (pyarrow Table /
    RecordBatch, polars DF / LazyFrame, pandas DF, pyspark DF,
    ``list[dict]``, ``dict[str, list]``) lives on :meth:`TabularIO.from_`
    so anywhere else in yggdrasil that wants to lift "anything"
    into a :class:`TabularIO` gets the same shape support without
    re-implementing the ladder.

    The path-string branch stays here because :func:`TabularIO.from_`
    treats unknown strings as bytes-buffer fodder; for SQL
    registration we want path strings to open via
    :meth:`TabularIO.from_path` so a parquet file or folder lands
    as the right subclass.
    """
    if isinstance(obj, TabularIO):
        return obj

    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        # Path-like: open as the right registered media leaf so a
        # parquet file / folder / CSV registers as itself rather
        # than as a generic byte buffer.
        return TabularIO.from_path(obj)  # type: ignore[arg-type]

    return TabularIO.from_(obj)


class SqlContext:
    """Mapping of identifier → :class:`TabularIO` source.

    Lookup is case-sensitive because that's what SQL parsers preserve
    when an identifier is quoted. For the unquoted-identifier case
    sqlglot lowercases / uppercases by dialect *before* it reaches
    us, so callers writing ``SELECT * FROM Trades`` and registering
    ``"trades"`` get a hit on Postgres (lowercased) and a miss on
    Databricks (preserved). When you don't care, lower-case both
    ends and move on.

    Thread-safe for register / deregister / lookup — writes take an
    :class:`RLock`. Iteration takes a snapshot so a concurrent
    register doesn't blow up the iterator.
    """

    def __init__(
        self,
        sources: "Mapping[str, Any] | None" = None,
        *,
        parent: "SqlContext | None" = None,
    ) -> None:
        self._sources: MutableMapping[str, TabularIO] = {}
        self._lock = threading.RLock()
        self._parent = parent
        if sources:
            self.register_many(sources)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, name: str, source: Any) -> "SqlContext":
        """Register *source* under *name*. Replaces any prior binding."""
        if not name:
            raise ValueError("SQL source name must be a non-empty string.")
        io = coerce_source(source)
        with self._lock:
            self._sources[name] = io
        return self

    def register_many(self, sources: Mapping[str, Any]) -> "SqlContext":
        """Register every ``{name: source}`` pair. Idempotent on duplicates."""
        for name, source in sources.items():
            self.register(name, source)
        return self

    def deregister(self, name: str) -> "TabularIO | None":
        """Drop *name* and return the prior binding, or ``None`` if absent."""
        with self._lock:
            return self._sources.pop(name, None)

    def clear(self) -> "SqlContext":
        """Drop every binding (does not touch the parent context)."""
        with self._lock:
            self._sources.clear()
        return self

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> "TabularIO | None":
        """Look up *name*; falls through to the parent context if missing."""
        with self._lock:
            hit = self._sources.get(name)
        if hit is not None:
            return hit
        if self._parent is not None:
            return self._parent.get(name)
        return None

    def __getitem__(self, name: str) -> TabularIO:
        hit = self.get(name)
        if hit is None:
            available = list(self.names())
            suggestions = _suggest(name, available)
            raise KeyError(
                f"No SQL source registered as {name!r}. "
                f"Registered: {available!r}."
                + (f" Did you mean {suggestions!r}?" if suggestions else "")
                + " Register via ctx.register(name, source) or "
                "yggdrasil.sql.register(name, source)."
            )
        return hit

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.get(name) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())

    def names(self) -> "list[str]":
        """All registered names visible from this context (parents merged in)."""
        with self._lock:
            local = list(self._sources.keys())
        if self._parent is None:
            return local
        seen = set(local)
        merged = list(local)
        for n in self._parent.names():
            if n not in seen:
                seen.add(n)
                merged.append(n)
        return merged

    def snapshot(self) -> "dict[str, TabularIO]":
        """Materialized view of every visible binding. Safe to iterate."""
        out: "dict[str, TabularIO]" = {}
        if self._parent is not None:
            out.update(self._parent.snapshot())
        with self._lock:
            out.update(self._sources)
        return out

    def child(self, sources: "Mapping[str, Any] | None" = None) -> "SqlContext":
        """Return a child context inheriting from self.

        Useful for scoped overrides: a one-off query that registers
        an extra alias without polluting the global registry.
        """
        return SqlContext(sources, parent=self)


# ---------------------------------------------------------------------------
# Process-wide default
# ---------------------------------------------------------------------------


default_context: SqlContext = SqlContext()


def register(name: str, source: Any) -> SqlContext:
    """Register *source* on the process-wide :data:`default_context`."""
    return default_context.register(name, source)


def deregister(name: str) -> "TabularIO | None":
    """Drop *name* from the process-wide :data:`default_context`."""
    return default_context.deregister(name)


def registered() -> "list[str]":
    """Names visible on the process-wide :data:`default_context`."""
    return default_context.names()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _suggest(name: str, choices: "list[str]") -> "list[str]":
    """Cheap typo helper — top-3 close matches via difflib."""
    if not choices:
        return []
    try:
        import difflib

        return difflib.get_close_matches(name, choices, n=3, cutoff=0.6)
    except Exception:
        return []
