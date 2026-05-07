"""Dynamic catalog — name → :class:`Tabular` resolution for the engine.

Wraps :class:`SqlContext` (the existing process-wide registry) and
adds the bits the engine needs that don't make sense to bolt onto
the older catalog:

- **Resolver chain.** Inline tables registered on a per-execute call
  (``engine.execute(query, sources={"t": tabular})``) are looked up
  before the parent catalog. Multiple parents are walked in order
  so an engine instance can declare its own permanent tables on top
  of a process-wide default.
- **Tabular auto-coercion.** Anything :func:`coerce_source` accepts
  (a pyarrow Table, a polars DataFrame, an iterable of dicts, a
  path string) lifts to a :class:`Tabular` on register so the rest
  of the engine sees one shape.
- **Schema cache.** Repeat name → schema lookups (the planner does
  one per scan + one per join column resolution) collapse to one
  ``collect_schema`` call per Tabular per planner pass.

Why a separate class
--------------------

The legacy :class:`SqlContext` is what :func:`yggdrasil.sql.sql` and
the polars / Arrow executors use. We don't want to bend its semantics
for the engine path; instead the engine takes a :class:`DynamicCatalog`
that *delegates* to one (or many) :class:`SqlContext` instances and
adds the engine-specific helpers on top. Existing call sites are
unaffected.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Optional

from yggdrasil.io.tabular import ArrowTabular, Tabular

from yggdrasil.io.tabular.execution.sql.catalog import SqlContext, default_context

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema


__all__ = ["DynamicCatalog", "coerce_to_tabular"]


def coerce_to_tabular(obj: Any) -> Tabular:
    """Lift *obj* into a :class:`Tabular`.

    Supported inputs (in priority order):

    - already a :class:`Tabular` → passthrough.
    - :class:`pyarrow.Table` / :class:`pyarrow.RecordBatch` →
      :class:`ArrowTabular`.
    - :class:`polars.DataFrame` / :class:`polars.LazyFrame` → bridge
      to Arrow then :class:`ArrowTabular`.
    - :class:`pandas.DataFrame` → ``pa.Table.from_pandas`` →
      :class:`ArrowTabular`.
    - ``list[dict]`` / ``dict[str, list]`` → :class:`pa.Table` →
      :class:`ArrowTabular`.
    - path-like (``str`` / ``bytes`` / ``__fspath__``) → open via
      :class:`Path` and wrap with :class:`Tabular.for_holder`.

    The matching is duck-typed where practical so optional packages
    don't have to be imported just for the dispatch.
    """
    import pyarrow as pa  # local — keep the import out of cold paths

    if isinstance(obj, Tabular):
        return obj
    if isinstance(obj, pa.Table):
        return ArrowTabular(obj)
    if isinstance(obj, pa.RecordBatch):
        return ArrowTabular(pa.Table.from_batches([obj]))
    if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
        return ArrowTabular(pa.Table.from_pylist(obj))
    if isinstance(obj, dict):
        return ArrowTabular(pa.table(obj))
    # Polars frame ducktype.
    if hasattr(obj, "to_arrow") and type(obj).__module__.startswith("polars"):
        try:
            arrow = obj.to_arrow() if not _is_lazy_polars(obj) else obj.collect().to_arrow()
        except Exception:
            arrow = None
        if isinstance(arrow, pa.Table):
            return ArrowTabular(arrow)
    # Pandas frame ducktype.
    if type(obj).__module__.startswith("pandas") and hasattr(obj, "columns"):
        return ArrowTabular(pa.Table.from_pandas(obj))
    # Path-like — open and wrap.
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        from yggdrasil.io.path import Path

        path = Path.from_(obj)
        return Tabular.for_holder(path)
    raise TypeError(
        f"Cannot register {type(obj).__name__} as a SQL source. "
        "Pass a Tabular, pyarrow Table/RecordBatch, polars / pandas frame, "
        "list[dict], dict[str, list], or a path string."
    )


def _is_lazy_polars(obj: Any) -> bool:
    return type(obj).__name__ == "LazyFrame"


class DynamicCatalog:
    """Engine-side resolver of names to :class:`Tabular` sources."""

    __slots__ = ("_locals", "_parents", "_schema_cache", "_lock")

    def __init__(
        self,
        sources: "Mapping[str, Any] | None" = None,
        *,
        parents: "Iterable[SqlContext | DynamicCatalog] | None" = None,
    ) -> None:
        self._locals: dict[str, Tabular] = {}
        # Default parent chain: the process-wide system catalog (which
        # itself parents the legacy SqlContext-style default_context).
        # Tests and one-off scopes can opt out by passing ``parents=[]``
        # (or a custom chain). Lazy import — the system catalog module
        # imports this class, so the top-level cycle would deadlock.
        if parents is None:
            parents = (_default_parent_chain(),)
            parents = tuple(p for p in parents if p is not None)
        self._parents: list[Any] = list(parents)
        self._schema_cache: "dict[str, Schema]" = {}
        self._lock = threading.RLock()
        if sources:
            self.register_many(sources)

    # ==================================================================
    # Mutation
    # ==================================================================

    def register(self, name: str, source: Any) -> "DynamicCatalog":
        """Register *source* under *name*. Replaces any prior local binding.

        Locally-registered names always shadow names visible through
        the parent chain.
        """
        if not name:
            raise ValueError("Catalog source name must be a non-empty string.")
        io = source if isinstance(source, Tabular) else coerce_to_tabular(source)
        with self._lock:
            self._locals[name] = io
            self._schema_cache.pop(name, None)
        return self

    def register_many(self, sources: Mapping[str, Any]) -> "DynamicCatalog":
        for name, source in sources.items():
            self.register(name, source)
        return self

    def deregister(self, name: str) -> "Tabular | None":
        with self._lock:
            self._schema_cache.pop(name, None)
            return self._locals.pop(name, None)

    # ==================================================================
    # Lookup
    # ==================================================================

    def get(self, name: str) -> "Tabular | None":
        with self._lock:
            hit = self._locals.get(name)
        if hit is not None:
            return hit
        # Search parents in order.
        for parent in self._parents:
            if isinstance(parent, DynamicCatalog):
                hit = parent.get(name)
            else:
                hit = parent.get(name)
            if hit is not None:
                return hit
        # Try the leaf-only name when the qualified form missed —
        # analysts often register ``trades`` and query
        # ``main.warehouse.trades``.
        if "." in name:
            leaf = name.rsplit(".", 1)[-1]
            if leaf != name:
                return self.get(leaf)
        return None

    def resolve(self, name: str) -> Tabular:
        """Strict lookup — raises with a helpful message on miss."""
        hit = self.get(name)
        if hit is not None:
            return hit
        available = self.names()
        suggestions = _suggest(name, available)
        raise KeyError(
            f"SQL source {name!r} is not registered. Available: "
            f"{available!r}."
            + (f" Did you mean {suggestions!r}?" if suggestions else "")
            + " Register via engine.register(name, source) or pass "
            "sources={name: source} to engine.execute(...)."
        )

    def names(self) -> "list[str]":
        """All names visible from this catalog (locals + parents merged)."""
        seen: set[str] = set()
        out: list[str] = []
        with self._lock:
            for n in self._locals:
                if n not in seen:
                    seen.add(n)
                    out.append(n)
        for parent in self._parents:
            for n in parent.names():
                if n not in seen:
                    seen.add(n)
                    out.append(n)
        return out

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.get(name) is not None

    # ==================================================================
    # Schema introspection
    # ==================================================================

    def schema_of(self, name: str) -> "Schema":
        """Cached :meth:`Tabular.collect_schema` for *name*.

        The planner needs the schema to resolve unqualified column
        references, validate joins, and infer aggregate output types.
        Tabular's :meth:`collect_schema` can be expensive (parquet
        footer parse, remote call); caching keeps a planner pass to
        one fetch per source.
        """
        with self._lock:
            cached = self._schema_cache.get(name)
        if cached is not None:
            return cached
        source = self.resolve(name)
        schema = source.collect_schema()
        with self._lock:
            self._schema_cache[name] = schema
        return schema

    def invalidate_schema(self, name: "Optional[str]" = None) -> None:
        with self._lock:
            if name is None:
                self._schema_cache.clear()
            else:
                self._schema_cache.pop(name, None)

    # ==================================================================
    # Scoped child for per-execute overrides
    # ==================================================================

    def child(
        self, sources: "Mapping[str, Any] | None" = None,
    ) -> "DynamicCatalog":
        """Return a child catalog inheriting from self.

        Per-execute ``sources`` kwargs build a child so the locals of
        the parent stay untouched after the call returns.
        """
        return DynamicCatalog(sources, parents=[self])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_parent_chain() -> Any:
    """Return the canonical default parent for a freshly-built catalog.

    Lazy lookup so :mod:`yggdrasil.sql.system_catalog` (which imports
    this module) doesn't form a cycle at module load. Returns ``None``
    on the very first call from inside the system catalog's own
    construction — :class:`DynamicCatalog` strips Nones from the
    parent chain so a missing system catalog isn't fatal.
    """
    try:
        from yggdrasil.io.tabular.execution.sql.system_catalog import SYSTEM_CATALOG
    except ImportError:
        return None
    return SYSTEM_CATALOG


def _suggest(name: str, choices: "list[str]") -> "list[str]":
    if not choices:
        return []
    try:
        import difflib

        return difflib.get_close_matches(name, choices, n=3, cutoff=0.6)
    except Exception:
        return []
