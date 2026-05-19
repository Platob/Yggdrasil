"""Abstract catalog — middle level of the ``catalog.schema.table`` tree.

A :class:`UnityCatalog` owns a set of :class:`UnitySchema`. Navigation
mirrors :class:`UnityEngine`: ``catalog["default"]`` returns a schema,
``in catalog`` is a cheap existence probe, ``for s in catalog`` walks
every schema. The base wires those off two backend hooks.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Iterator

from yggdrasil.unity.base import UnityResource

if TYPE_CHECKING:
    from yggdrasil.unity.schema import UnitySchema


__all__ = ["UnityCatalog"]


logger = logging.getLogger(__name__)


class UnityCatalog(UnityResource):
    """Abstract catalog — child schemas plus the inherited lifecycle."""

    @property
    def full_name(self) -> str:
        return self.name

    # ── schema navigation ───────────────────────────────────────────────

    @abstractmethod
    def schema(self, name: str) -> "UnitySchema":
        """Return a :class:`UnitySchema` bound to *name*.

        Existence is not verified — probe via :attr:`exists` on the
        returned handle.
        """

    @abstractmethod
    def schemas(self) -> Iterator["UnitySchema"]:
        """Iterate over every schema in this catalog."""

    @abstractmethod
    def create_schema(
        self,
        name: str,
        *,
        if_not_exists: bool = True,
        **kwargs,
    ) -> "UnitySchema":
        """Create *name* under this catalog and return its handle."""

    # ── default surface ─────────────────────────────────────────────────

    def __getitem__(self, name: str) -> "UnitySchema":
        schema = self.schema(name)
        if not schema.exists:
            available = sorted(s.name for s in self.schemas())
            raise KeyError(
                f"Schema {name!r} does not exist in catalog {self.name!r}. "
                f"Available: {available!r}. Create via "
                "catalog.create_schema(name)."
            )
        return schema

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.schema(name).exists

    def __iter__(self) -> Iterator["UnitySchema"]:
        return self.schemas()
