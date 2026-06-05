"""Dict-like base for Unity Catalog securable collection services.

:class:`SecurableMapping` makes a :class:`DatabricksService` over a named
securable collection (credentials, external locations, …) a
:class:`~collections.abc.MutableMapping`::

    svc[name]                 # fetch the resource (KeyError if absent)
    svc[name] = spec          # create it (or update if it exists)
    del svc[name]             # delete it
    name in svc               # exists?
    list(svc) / len(svc)      # the securable names
    svc.get(name) / svc.pop(name) / svc.keys() / .values() / .items()

So the full mapping mixin (``get`` / ``pop`` / ``keys`` / ``values`` /
``items`` / ``setdefault`` / …) comes for free. ``clear()`` is refused — a
bulk wipe of a metastore collection is never what you meant.

A subclass implements six hooks: :meth:`resolve` (flexible coercion → a lazy
handle), :meth:`_infos` (SDK info iterator), :meth:`get_info`, :meth:`_resource`
(handle factory), :meth:`delete`, and :meth:`_apply` (create-or-update).
"""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from typing import Any, Iterator

from databricks.sdk.errors import NotFound

from yggdrasil.databricks.service import DatabricksService

__all__ = ["SecurableMapping"]


class SecurableMapping(DatabricksService, MutableMapping):
    # A service is an identity object (the singleton-cached handle for a
    # client's collection), but ``MutableMapping`` sets ``__hash__ = None`` and
    # gives a value-based ``__eq__``. Restore identity hash/eq so the service
    # stays usable as a dict / singleton key (the resources key on it).
    __hash__ = object.__hash__

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __ne__(self, other: Any) -> bool:
        return self is not other

    # -- subclass hooks -------------------------------------------------
    @abstractmethod
    def resolve(self, obj: Any = None, **kwargs: Any) -> Any:
        """Coerce ``(resource | name | id | kwargs)`` to a **lazy** resource
        handle (no fetch for the plain-name case)."""

    @abstractmethod
    def _infos(self) -> Iterator[Any]:
        """Iterate the SDK info objects for the collection (each has ``.name``)."""

    @abstractmethod
    def get_info(self, name: str) -> Any:
        """Fetch one securable's SDK info (raises :class:`NotFound`)."""

    @abstractmethod
    def _resource(self, name: str, info: Any = None) -> Any:
        """Build a resource handle, optionally seeded with fetched *info*."""

    @abstractmethod
    def delete(self, name: str, *, force: bool = False) -> None:
        ...

    @abstractmethod
    def _apply(self, name: str, spec: Any, *, exists: bool) -> Any:
        """Create (``exists=False``) or update (``exists=True``) from *spec*."""

    # -- MutableMapping -------------------------------------------------
    def __getitem__(self, key: Any) -> Any:
        name = self.resolve(key).name
        try:
            info = self.get_info(name)
        except NotFound:
            raise KeyError(key)
        return self._resource(name, info=info)

    def __setitem__(self, key: Any, value: Any) -> None:
        name = self.resolve(key).name
        self._apply(name, value, exists=self.exists(name))

    def __delitem__(self, key: Any) -> None:
        name = self.resolve(key).name
        if not self.exists(name):
            raise KeyError(key)
        self.delete(name, force=True)

    def __iter__(self) -> Iterator[str]:
        return (info.name for info in self._infos())

    def __len__(self) -> int:
        return sum(1 for _ in self._infos())

    def __contains__(self, key: Any) -> bool:
        try:
            name = self.resolve(key).name
        except Exception:
            return False
        return self.exists(name)

    def clear(self) -> None:  # safety — never bulk-wipe a UC collection
        raise NotImplementedError(
            f"refusing clear() on {type(self).__name__}; delete securables explicitly."
        )

    # -- convenience (public) ------------------------------------------
    def exists(self, name: str) -> bool:
        try:
            self.get_info(name)
            return True
        except NotFound:
            return False

    def names(self) -> "list[str]":
        return [info.name for info in self._infos()]

    def list(self) -> Iterator[Any]:
        """Iterate every securable as a resource handle (info pre-seeded)."""
        for info in self._infos():
            yield self._resource(info.name, info=info)

    @staticmethod
    def _as_spec(value: Any) -> dict:
        return dict(value) if isinstance(value, Mapping) else value
