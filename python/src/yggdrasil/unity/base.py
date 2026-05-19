"""Abstract base class for every :mod:`yggdrasil.unity` resource.

A :class:`UnityResource` is the common contract every catalog / schema
/ table / view satisfies: identity (``name`` / ``full_name``), bound
:class:`UnityEngine`, cached :attr:`info` payload, ``create`` /
``delete`` / ``exists`` / ``ensure_created`` lifecycle. Backends
implement the abstract hooks; navigation (Catalog → Schema →
Table/View) is wired by the leaf subclasses.

The pattern deliberately mirrors :class:`yggdrasil.databricks.catalog.Catalog`
without forcing every backend through a remote Path. A filesystem
backend (``unity.fs``), a SQLite registry, or a remote service all
slot in by overriding the hooks below.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from yggdrasil.unity.engine import UnityEngine


__all__ = ["UnityResource"]


logger = logging.getLogger(__name__)


class UnityResource(ABC):
    """Common surface for Unity-Catalog-style resources.

    Subclasses provide:

    * :attr:`engine`     — the bound :class:`UnityEngine`.
    * :attr:`name`       — short identifier (e.g. ``"sales"``).
    * :attr:`full_name`  — dotted qualified name.
    * :meth:`_read_info` — backend hook returning the info dataclass.
    * :meth:`create`     — backend hook minting the resource.
    * :meth:`delete`     — backend hook tearing it down.

    Everything else (``exists`` / ``info`` caching / ``ensure_created`` /
    ``__repr__`` / equality) lives on the base.
    """

    #: Default TTL (seconds) for the cached :attr:`info` payload.
    #: ``None`` keeps the cache live for the process lifetime — the
    #: same shape as :class:`yggdrasil.databricks.catalog.Catalog`'s
    #: ``_infos_ttl``. Subclasses override per backend.
    _INFO_TTL: ClassVar[float | None] = None

    _info_cached: Any = None
    _info_fetched_at: float = 0.0

    # ── identity ──────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def engine(self) -> "UnityEngine":
        """The :class:`UnityEngine` this resource is bound to."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unqualified resource name."""

    @property
    @abstractmethod
    def full_name(self) -> str:
        """Dotted qualified name (``catalog`` / ``catalog.schema`` /
        ``catalog.schema.table``)."""

    # ── info / existence ─────────────────────────────────────────────────

    @abstractmethod
    def _read_info(self) -> Any:
        """Backend hook: fetch this resource's info dataclass.

        Raises :class:`FileNotFoundError` (or any backend-native
        ``NotFound`` shape) when the resource does not exist. The
        :attr:`exists` property catches that to flip to ``False``.
        """

    @property
    def info(self) -> Any:
        """Cached info dataclass. Refetches when the TTL window expires."""
        cached = self._info_cached
        if cached is not None:
            ttl = self._INFO_TTL
            if ttl is None or (time.time() - self._info_fetched_at) < ttl:
                return cached
            logger.debug(
                "Cache expired for %r (age=%.0fs, ttl=%.0fs) — refreshing",
                self, time.time() - self._info_fetched_at, ttl,
            )
        logger.debug("Fetching info for %r from backend", self)
        info = self._read_info()
        self._store_info(info)
        return info

    def _store_info(self, info: Any) -> None:
        """Stamp *info* into the per-instance cache. Backends call this
        from ``create`` / ``delete`` paths so the next read avoids a
        round trip."""
        self._info_cached = info
        self._info_fetched_at = time.time()

    def _invalidate_info(self) -> None:
        """Drop the cached info payload."""
        self._info_cached = None
        self._info_fetched_at = 0.0

    @property
    def exists(self) -> bool:
        """``True`` if :meth:`_read_info` resolves without raising."""
        try:
            _ = self.info
            return True
        except FileNotFoundError:
            return False

    # ── lifecycle ────────────────────────────────────────────────────────

    @abstractmethod
    def create(self, *, if_not_exists: bool = True, **kwargs: Any) -> "UnityResource":
        """Create this resource in the backend.

        ``if_not_exists=True`` (default) is a silent no-op when the
        resource already exists; ``False`` lets the underlying
        ``AlreadyExists`` propagate.
        """

    @abstractmethod
    def delete(self, *, missing_ok: bool = True, **kwargs: Any) -> "UnityResource":
        """Delete this resource from the backend.

        ``missing_ok=True`` (default) swallows ``NotFound`` so callers
        can call ``delete`` defensively; ``False`` re-raises.
        """

    def ensure_created(self, **kwargs: Any) -> "UnityResource":
        """Create the resource if it does not already exist."""
        if not self.exists:
            self.create(if_not_exists=True, **kwargs)
        return self

    # ── identity dunders ─────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.full_name!r})"

    def __str__(self) -> str:
        return self.full_name

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, UnityResource):
            return NotImplemented
        return (
            type(self) is type(other)
            and self.engine is other.engine
            and self.full_name == other.full_name
        )

    def __hash__(self) -> int:
        return hash((type(self), id(self.engine), self.full_name))
