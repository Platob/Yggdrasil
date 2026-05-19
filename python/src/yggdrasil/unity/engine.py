"""Top-level facade for a Unity-Catalog-style backend.

A :class:`UnityEngine` owns the set of catalogs the caller can navigate
through. Catalog → Schema → Table/View navigation hangs off the engine
(``engine["main"]["default"]["sales"]``); ``__iter__`` walks every
catalog and ``__contains__`` is a cheap existence probe.

The class is abstract — backends supply :meth:`catalog` /
:meth:`catalogs` / :meth:`_default_catalog_name`. Everything else
(``__getitem__`` etc.) is wired here.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from yggdrasil.unity.catalog import UnityCatalog


__all__ = ["UnityEngine"]


logger = logging.getLogger(__name__)


class UnityEngine(ABC):
    """Root of a Unity-Catalog-style backend.

    Subclasses bind to whatever lives below (a filesystem :class:`Path`,
    a SQL connection, a remote service) and return :class:`UnityCatalog`
    instances through :meth:`catalog`.
    """

    # ── catalog navigation ──────────────────────────────────────────────

    @abstractmethod
    def catalog(self, name: str) -> "UnityCatalog":
        """Return a :class:`UnityCatalog` bound to *name*.

        Does NOT verify existence — the returned handle's :attr:`exists`
        property is the canonical probe.
        """

    @abstractmethod
    def catalogs(self) -> Iterator["UnityCatalog"]:
        """Iterate over every catalog visible to this engine."""

    @abstractmethod
    def create_catalog(
        self,
        name: str,
        *,
        if_not_exists: bool = True,
        **kwargs,
    ) -> "UnityCatalog":
        """Create *name* in the backend and return its handle."""

    # ── default surface (wired off the abstract hooks) ──────────────────

    def __getitem__(self, name: str) -> "UnityCatalog":
        catalog = self.catalog(name)
        if not catalog.exists:
            available = sorted(c.name for c in self.catalogs())
            raise KeyError(
                f"Catalog {name!r} does not exist on {self!r}. "
                f"Available: {available!r}. Create via "
                "engine.create_catalog(name)."
            )
        return catalog

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.catalog(name).exists

    def __iter__(self) -> Iterator["UnityCatalog"]:
        return self.catalogs()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
