"""Three-level tabular catalog — ``catalog.schema.name`` → :class:`Tabular`.

:class:`TabularEngine` is the centralized metadata holder for a set of
:class:`Tabular` sources, keyed by the standard SQL three-part
identifier (``catalog`` / ``schema`` / ``name``). Entries store the
:class:`Tabular` itself (the implementation is the compressed canonical
form, not a copy of its bytes) plus a cached :class:`Schema` so the
planner-style "what columns does X have" lookup collapses to one
``collect_schema`` call per registration.

"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Optional, Tuple

from yggdrasil.io.tabular.base import Tabular

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema


__all__ = [
    "TabularEntry",
    "TabularEngine",
    "SYSTEM_ENGINE",
    "register",
    "deregister",
    "get",
    "resolve",
]


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
        coerced via the SQL catalog's :func:`coerce_source`, so any
        pyarrow / polars / pandas frame, ``list[dict]``, ``dict[str,
        list]``, or path-like accepted by the SQL registry also works
        here.
        """
        key = self._check_key(catalog, schema, name)
        if not isinstance(tabular, Tabular):
            from yggdrasil.io import Holder
            tabular = Holder.from_(tabular)
        entry = TabularEntry(catalog=key[0], schema=key[1], name=key[2], tabular=tabular)
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

    # ------------------------------------------------------------------
    # Flat-name resolution — SqlContext-shaped surface so the engine
    # can feed the SQL executor without a wrapper class.
    # ------------------------------------------------------------------

    def resolve(self, name: str) -> Tabular:
        """Resolve a flat *name* to a :class:`Tabular`. Raises on miss.

        Accepts the dotted ``catalog.schema.name`` form, the partial
        ``schema.name`` form, or the bare leaf ``name``. Partial
        matches return the only entry whose tail matches; an
        ambiguous leaf (registered under two different schemas)
        raises :class:`KeyError`.
        """
        hit = self.lookup(name)
        if hit is None:
            raise KeyError(
                f"Tabular {name!r} is not registered on this engine. "
                f"Available: {self.qualified_names()!r}. Register via "
                "engine.register(catalog, schema, name, tabular)."
            )
        return hit

    def lookup(self, name: str) -> "Tabular | None":
        """Soft form of :meth:`resolve` — returns ``None`` on miss.

        Resolution order: dotted three-part match first, then
        ``schema.name``, then the bare leaf. The leaf form returns
        ``None`` (rather than picking arbitrarily) when more than one
        entry shares the leaf name.
        """
        if not name:
            return None
        parts = name.split(".")
        with self._lock:
            if len(parts) == 3:
                hit = self._entries.get((parts[0], parts[1], parts[2]))
                return hit.tabular if hit is not None else None
            if len(parts) == 2:
                matches = [
                    e for k, e in self._entries.items()
                    if k[1] == parts[0] and k[2] == parts[1]
                ]
                if len(matches) == 1:
                    return matches[0].tabular
                return None
            matches = [e for k, e in self._entries.items() if k[2] == name]
        if len(matches) == 1:
            return matches[0].tabular
        return None

    # SqlContext-shaped lookup so the SQL planner can drive lookups
    # through the same surface. ``names`` returns dotted identifiers
    # so the planner sees stable, fully-qualified table names.
    def get_by_name(self, name: str) -> "Tabular | None":
        return self.lookup(name)

    def names(self) -> "list[str]":
        return self.qualified_names()

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


# ---------------------------------------------------------------------------
# Process-wide singleton + module-level shortcuts
# ---------------------------------------------------------------------------
# :class:`TabularEngine` is the canonical home for ``catalog.schema.name``
# registrations. The SQL stack (``Join.right`` strings, the per-execute
# ``sources`` merge in :meth:`execute_sql`) routes back through
# :data:`SYSTEM_ENGINE` so a single registration becomes visible to
# both the lazy-plan path and the SQL path without the caller wiring two
# registries.

SYSTEM_ENGINE: TabularEngine = TabularEngine()


def register(catalog: str, schema: str, name: str, tabular: Any) -> TabularEntry:
    """Register on the process-wide :data:`SYSTEM_ENGINE`."""
    return SYSTEM_ENGINE.register(catalog, schema, name, tabular)


def deregister(catalog: str, schema: str, name: str) -> "TabularEntry | None":
    """Deregister from the process-wide :data:`SYSTEM_ENGINE`."""
    return SYSTEM_ENGINE.deregister(catalog, schema, name)


def get(catalog: str, schema: str, name: str) -> "TabularEntry | None":
    """Look up an entry on the process-wide :data:`SYSTEM_ENGINE`."""
    return SYSTEM_ENGINE.get(catalog, schema, name)


def resolve(name: str) -> Tabular:
    """Resolve a flat / dotted *name* on :data:`SYSTEM_ENGINE`."""
    return SYSTEM_ENGINE.resolve(name)
