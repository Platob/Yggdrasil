"""Three-level tabular catalog — ``catalog.schema.name`` → :class:`Tabular`.

:class:`TabularEngine` is the centralized metadata holder for a set of
:class:`Tabular` sources, keyed by the standard SQL three-part
identifier (``catalog`` / ``schema`` / ``name``). Entries store the
:class:`Tabular` itself (the implementation is the compressed canonical
form, not a copy of its bytes) plus a cached :class:`Schema` so the
planner-style "what columns does X have" lookup collapses to one
``collect_schema`` call per registration.

Why a three-level engine in addition to :class:`SqlContext`
-----------------------------------------------------------

:class:`yggdrasil.io.tabular.execution.sql.SqlContext` is the flat
``name → Tabular`` mapping the SQL executor uses. It deliberately
ignores catalog / schema namespacing because the executor's call sites
only need a single identifier. :class:`TabularEngine` covers the
opposite need: code (catalog browsers, integration glue, multi-tenant
session state) that *does* care about the full ``catalog.schema.name``
shape and wants to ask "list every table in this schema" without
reparsing strings. The two registries are deliberately independent;
register into whichever fits the call site.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional, Tuple

from yggdrasil.io.tabular.base import Tabular

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema


__all__ = ["TabularEntry", "TabularEngine"]


_Key = Tuple[str, str, str]


@dataclass
class TabularEntry:
    """One registered ``catalog.schema.name`` row.

    Holds a :class:`Tabular` reference plus a lazily-computed
    :class:`Schema` cache. The :class:`Tabular` is stored as-is — the
    engine assumes the implementation is already the canonical /
    compressed shape the caller wants tracked, and never copies it.
    """

    catalog: str
    schema: str
    name: str
    tabular: Tabular
    _schema_cache: "Schema | None" = field(default=None, repr=False, compare=False)

    @property
    def qualified_name(self) -> str:
        """Dotted identifier ``catalog.schema.name``."""
        return f"{self.catalog}.{self.schema}.{self.name}"

    @property
    def key(self) -> _Key:
        """Tuple key used by the engine's internal index."""
        return (self.catalog, self.schema, self.name)

    def get_schema(self) -> "Schema":
        """Return the entry's :class:`Schema`, caching the first result.

        Subsequent calls reuse the cache; call :meth:`invalidate_schema`
        after mutating the underlying :class:`Tabular` to force a
        re-collect on the next read.
        """
        if self._schema_cache is None:
            self._schema_cache = self.tabular.collect_schema()
        return self._schema_cache

    def invalidate_schema(self) -> None:
        """Drop the cached schema so the next :meth:`get_schema` re-collects."""
        self._schema_cache = None


class TabularEngine:
    """Registry of :class:`Tabular` sources keyed by ``catalog.schema.name``.

    Thread-safe for register / deregister / lookup — writes take an
    :class:`RLock`, iteration takes a snapshot of the current keys so a
    concurrent register doesn't blow up the iterator.

    Identifier matching is case-sensitive — same rationale as
    :class:`SqlContext`. Lower-case both ends when you don't care.
    """

    def __init__(
        self,
        sources: "Mapping[tuple[str, str, str] | str, Any] | None" = None,
    ) -> None:
        self._entries: "dict[_Key, TabularEntry]" = {}
        self._lock = threading.RLock()
        if sources:
            self.register_many(sources)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(
        self,
        catalog: str,
        schema: str,
        name: str,
        tabular: Any,
    ) -> TabularEntry:
        """Register *tabular* under ``catalog.schema.name``.

        Replaces any prior binding at the same key. *tabular* is
        coerced via the SQL catalog's :func:`coerce_to_tabular`, so any
        pyarrow / polars / pandas frame, ``list[dict]``, ``dict[str,
        list]``, or path-like accepted by the SQL registry also works
        here.
        """
        key = self._check_key(catalog, schema, name)
        from yggdrasil.io.tabular.execution.sql.dynamic_catalog import coerce_to_tabular
        io = coerce_to_tabular(tabular)
        entry = TabularEntry(catalog=key[0], schema=key[1], name=key[2], tabular=io)
        with self._lock:
            self._entries[key] = entry
        return entry

    def register_many(
        self,
        sources: "Mapping[tuple[str, str, str] | str, Any]",
    ) -> "TabularEngine":
        """Bulk register. Keys are either ``(catalog, schema, name)``
        tuples or dotted ``"catalog.schema.name"`` strings."""
        for key, src in sources.items():
            cat, sch, nm = self._split_key(key)
            self.register(cat, sch, nm, src)
        return self

    def deregister(
        self,
        catalog: str,
        schema: str,
        name: str,
    ) -> "TabularEntry | None":
        """Drop the entry at ``catalog.schema.name`` and return it,
        or ``None`` if absent."""
        key = self._check_key(catalog, schema, name)
        with self._lock:
            return self._entries.pop(key, None)

    def clear(self) -> "TabularEngine":
        """Drop every entry."""
        with self._lock:
            self._entries.clear()
        return self

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(
        self,
        catalog: str,
        schema: str,
        name: str,
    ) -> "TabularEntry | None":
        """Look up the entry at ``catalog.schema.name``, or ``None``."""
        with self._lock:
            return self._entries.get((catalog, schema, name))

    def get_tabular(
        self,
        catalog: str,
        schema: str,
        name: str,
    ) -> Tabular:
        """Return the registered :class:`Tabular`. Raises on miss."""
        return self[catalog, schema, name].tabular

    def get_schema(
        self,
        catalog: str,
        schema: str,
        name: str,
    ) -> "Schema":
        """Return the cached :class:`Schema` for ``catalog.schema.name``."""
        return self[catalog, schema, name].get_schema()

    def __getitem__(self, key: "_Key | str") -> TabularEntry:
        cat, sch, nm = self._split_key(key)
        hit = self.get(cat, sch, nm)
        if hit is None:
            available = self.qualified_names()
            raise KeyError(
                f"No tabular registered as {cat!r}.{sch!r}.{nm!r}. "
                f"Registered: {available!r}. Register via "
                "engine.register(catalog, schema, name, tabular)."
            )
        return hit

    def __contains__(self, key: object) -> bool:
        try:
            cat, sch, nm = self._split_key(key)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return self.get(cat, sch, nm) is not None

    def __iter__(self) -> Iterator[TabularEntry]:
        with self._lock:
            return iter(list(self._entries.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Listing — catalogs / schemas / tables
    # ------------------------------------------------------------------

    def catalogs(self) -> "list[str]":
        """Sorted list of distinct catalog names."""
        with self._lock:
            return sorted({k[0] for k in self._entries})

    def schemas(self, catalog: "str | None" = None) -> "list[str]":
        """Sorted list of distinct schema names, optionally scoped to *catalog*."""
        with self._lock:
            return sorted({
                k[1] for k in self._entries
                if catalog is None or k[0] == catalog
            })

    def tables(
        self,
        catalog: "str | None" = None,
        schema: "str | None" = None,
    ) -> "list[str]":
        """Sorted list of table names, optionally scoped to *catalog* / *schema*."""
        with self._lock:
            return sorted(
                k[2] for k in self._entries
                if (catalog is None or k[0] == catalog)
                and (schema is None or k[1] == schema)
            )

    def entries(
        self,
        catalog: "str | None" = None,
        schema: "str | None" = None,
    ) -> "list[TabularEntry]":
        """Snapshot of every entry, optionally scoped to *catalog* / *schema*."""
        with self._lock:
            return [
                e for k, e in self._entries.items()
                if (catalog is None or k[0] == catalog)
                and (schema is None or k[1] == schema)
            ]

    def qualified_names(self) -> "list[str]":
        """Sorted dotted identifiers — useful for error messages."""
        with self._lock:
            return sorted(e.qualified_name for e in self._entries.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_key(catalog: str, schema: str, name: str) -> _Key:
        if not catalog or not schema or not name:
            raise ValueError(
                "TabularEngine keys require non-empty (catalog, schema, "
                f"name); got ({catalog!r}, {schema!r}, {name!r})."
            )
        return (catalog, schema, name)

    @staticmethod
    def _split_key(key: Any) -> _Key:
        if isinstance(key, str):
            parts = key.split(".")
            if len(parts) != 3:
                raise ValueError(
                    f"Dotted key {key!r} must have exactly three parts "
                    "(catalog.schema.name)."
                )
            return TabularEngine._check_key(*parts)
        if isinstance(key, tuple) and len(key) == 3:
            return TabularEngine._check_key(*key)
        raise TypeError(
            f"TabularEngine key must be a 3-tuple or dotted string; "
            f"got {type(key).__name__}: {key!r}"
        )
