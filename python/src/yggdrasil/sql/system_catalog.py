"""Process-wide system catalog — the canonical place for global SQL sources.

What it is
----------

:data:`SYSTEM_CATALOG` is a :class:`DynamicCatalog` instantiated once
per process. Every :class:`Engine` and :class:`DynamicCatalog`
constructed without an explicit ``parents=`` chain inherits from it,
so a source registered here resolves from any engine without the
caller threading a context object around.

Two-layer view
--------------

The system catalog has the legacy :data:`yggdrasil.sql.default_context`
(:class:`SqlContext`) wired in as its parent — so anything written
via the older :func:`yggdrasil.sql.register` surface is also visible
to the new engine, and vice versa is not (writes through
:data:`SYSTEM_CATALOG` stay on the system catalog's locals). That
keeps the new path clean (no one writes to a legacy SqlContext by
accident) while letting old code's registrations still show up.

Typical usage
-------------

::

    import yggdrasil.sql as ysql

    # Register once, anywhere.
    ysql.system_catalog.register("trades", trades_tabular)
    ysql.system_catalog.register("orders", orders_io)

    # Every Engine instance picks them up by default.
    eng = ysql.Engine()
    eng.execute("SELECT * FROM trades JOIN orders USING (id)")

    # A scoped engine with extra sources doesn't pollute the global:
    one_off = ysql.Engine(sources={"shadow_trades": ...})
    one_off.execute("SELECT * FROM shadow_trades")
    # ``shadow_trades`` is gone now; ``trades`` and ``orders`` remain.

The module-level :func:`register` / :func:`deregister` / :func:`names`
/ :func:`get` / :func:`clear` operate on :data:`SYSTEM_CATALOG`. They
mirror the surface :class:`DynamicCatalog` exposes so you don't have
to choose between an instance method and a global function — pick
either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

from yggdrasil.sql.catalog import default_context as _legacy_default_context
from yggdrasil.sql.dynamic_catalog import DynamicCatalog


if TYPE_CHECKING:
    from yggdrasil.io.tabular import Tabular


__all__ = [
    "SYSTEM_CATALOG",
    "register",
    "register_many",
    "deregister",
    "names",
    "get",
    "resolve",
    "clear",
    "snapshot",
]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


#: The process-wide system catalog. Use :func:`register` /
#: :func:`deregister` (or the methods on this object directly) to
#: manage its bindings.
#:
#: The legacy :data:`yggdrasil.sql.default_context` is wired in as its
#: parent so older code's registrations remain visible to the new
#: :class:`Engine` without callers having to migrate.
SYSTEM_CATALOG: DynamicCatalog = DynamicCatalog(parents=[_legacy_default_context])


# ---------------------------------------------------------------------------
# Module-level convenience surface
# ---------------------------------------------------------------------------


def register(name: str, source: Any) -> DynamicCatalog:
    """Register *source* under *name* on :data:`SYSTEM_CATALOG`.

    Returns the system catalog for chaining. Replaces any prior
    binding on the system catalog (legacy bindings on
    :data:`default_context` are not touched — they still shadow
    through the parent chain when the system catalog has no local
    binding for the name).
    """
    return SYSTEM_CATALOG.register(name, source)


def register_many(sources: Mapping[str, Any]) -> DynamicCatalog:
    """Bulk register every ``{name: source}`` pair on :data:`SYSTEM_CATALOG`."""
    return SYSTEM_CATALOG.register_many(sources)


def deregister(name: str) -> "Tabular | None":
    """Drop *name* from :data:`SYSTEM_CATALOG`'s locals. Returns the prior value."""
    return SYSTEM_CATALOG.deregister(name)


def names() -> "list[str]":
    """All names visible from :data:`SYSTEM_CATALOG` (locals + parents)."""
    return SYSTEM_CATALOG.names()


def get(name: str) -> "Tabular | None":
    """Look up *name* on :data:`SYSTEM_CATALOG`. Returns ``None`` on miss."""
    return SYSTEM_CATALOG.get(name)


def resolve(name: str) -> "Tabular":
    """Strict :func:`get` — raises :class:`KeyError` on miss."""
    return SYSTEM_CATALOG.resolve(name)


def clear() -> None:
    """Drop every local binding on :data:`SYSTEM_CATALOG`.

    Parent registrations (legacy :data:`default_context` entries) are
    not touched. Useful for resetting between test runs.
    """
    # No public ``clear()`` on DynamicCatalog yet — iterate locals.
    for name in list(SYSTEM_CATALOG._locals):  # type: ignore[attr-defined]
        SYSTEM_CATALOG.deregister(name)


def snapshot() -> "dict[str, Tabular]":
    """Materialized ``{name: Tabular}`` view of every visible binding."""
    out: "dict[str, Tabular]" = {}
    for n in SYSTEM_CATALOG.names():
        hit = SYSTEM_CATALOG.get(n)
        if hit is not None:
            out[n] = hit
    return out
