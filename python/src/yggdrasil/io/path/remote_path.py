"""Abstract base for network-backed :class:`Holder` implementations.

The single seam every remote backend (S3, Databricks, future Azure
/ GCS / SFTP / WebDAV) shares: **predicate pins.**
``is_remote_path = True``, the other two ``False``. Concrete
subclasses no longer reimplement these.

Subclasses implement :meth:`_stat_uncached`; the base wraps it via
:meth:`_stat` and stores the result on ``self._stat_cached``.
Mutating ops (writes, deletes) must call :meth:`invalidate_singleton`
so follow-up reads see fresh metadata. Sister of
:class:`yggdrasil.io.fs.local_path.LocalPath`: same :class:`Holder`
substrate, different backing.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from threading import RLock
from typing import Any, ClassVar

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.io.path.path import Path


__all__ = ["RemotePath"]


#: Default freshness window for a seeded :class:`IOStats` entry.
#: Beyond this, :meth:`RemotePath._stat` discards the cached entry
#: and re-issues the backend probe. Five minutes matches the
#: lifetime of a typical Databricks / S3 credential refresh cycle —
#: long enough to collapse the dozen probes a Delta replay makes
#: against the same key, short enough that a stale entry doesn't
#: outlive a meaningful change to the underlying object.
_STAT_CACHE_TTL: float = 300.0
logger = logging.getLogger(__name__)


class RemotePath(Path):
    """Abstract :class:`Holder` for network-backed backends.

    Subclasses pick a ``scheme`` (``s3``, ``dbfs``, …), implement the
    five :class:`Holder` primitives against their network client, and
    override :meth:`_stat_uncached` for the metadata probe. Everything
    else (predicate pins, stat caching, singleton identity caching)
    is inherited from this base.

    ``RemotePath`` activates the :class:`Singleton` machinery that
    :class:`Holder` ships deactivated by default: two callers asking
    for the same URL (and client, where the subclass keys on it)
    inside the 5-minute window share the live instance — same stat
    cache, same lazily-bound transport. ``iterdir``-style hot loops
    pass ``singleton_ttl=False`` to keep the bounded cache from
    filling with short-lived children; long-lived consumers that
    want stronger sharing pass ``singleton_ttl=None``.
    """

    # Bound the freshness window for both probe-populated and
    # listing-seeded entries. ``Path`` ships the slot at ``None``
    # (live forever) since LocalPath / Memory don't need a TTL;
    # remote backends pay 5-minute round trips and want a window
    # that beats credential / consistency drift.
    STAT_CACHE_TTL: ClassVar["float | None"] = _STAT_CACHE_TTL

    # Activate the :class:`Singleton` cache for every concrete remote
    # backend: 5-minute default TTL, bounded at 10 000 entries as
    # defence-in-depth against accidental cardinality explosions.
    # The default ``_singleton_key`` includes ``cls`` so S3Path /
    # DatabricksPath / future Azure paths can share one ``_INSTANCES``
    # dict without colliding.
    _SINGLETON_TTL: ClassVar[Any] = _STAT_CACHE_TTL
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=_STAT_CACHE_TTL, max_size=10_000,
    )
    _INSTANCES_LOCK: ClassVar[RLock] = RLock()

    # ------------------------------------------------------------------
    # Backing-shape predicates
    # ------------------------------------------------------------------

    @property
    def is_memory(self) -> bool:
        return False

    @property
    def is_local_path(self) -> bool:
        return False

    @property
    def is_remote_path(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Stat — cached probe, subclass implements the network call
    # ------------------------------------------------------------------

    @property
    def size_known(self) -> bool:
        """``True`` only when the stat cache carries a fresh entry.

        Lets ``ParquetIO`` / ``CsvIO`` / ``ArrowIPCIO`` skip a probe
        round trip just to short-circuit on ``size == 0``: when the
        cache is cold the format reader will trip its own EOF /
        empty-file error which the caller catches and translates to
        an empty schema. When the cache is warm the cheap ``size``
        read fires unchanged.
        """
        return self._stat_cached_fresh() is not None

    def _stat(self) -> IOStats:
        """Cached :class:`IOStats` probe.

        Entries expire after :attr:`stat_cache_ttl` seconds — past
        the budget we re-issue :meth:`_stat_uncached` instead of
        handing back a stale snapshot. On miss (or expiry),
        delegates to :meth:`_stat_uncached` and stores the fresh
        result. Subclasses override :meth:`_stat_uncached`, never
        this.
        """
        cached = self._stat_cached_fresh()
        if cached is not None:
            return cached
        result = self._stat_uncached()
        self._persist_stat_cache(result)
        return result

    @abstractmethod
    def _stat_uncached(self) -> IOStats:
        """Backend-specific :class:`IOStats` probe. One network call."""

    def invalidate_singleton(self, remove_global: bool = True) -> None:
        """Drop this path's cached :class:`IOStats`, schema, and
        ``_INSTANCES`` entry — see :meth:`Path.invalidate_singleton`."""
        super().invalidate_singleton(remove_global=remove_global)
        self._unpersist_schema()

    # ------------------------------------------------------------------
    # Resize is a no-op on remote backends — the upload IS the resize
    # ------------------------------------------------------------------

    def resize(self, n: int) -> int:
        """No-op for remote-backend paths.

        :class:`Holder.resize` would call :meth:`truncate` to pre-grow
        a holder before a positional write. On remote backends every
        ``truncate`` is a full-object upload, so the pre-grow would
        double the network traffic for every write. The upload that
        :meth:`write_mv` runs next will materialize the right size on
        its own.
        """
        if n < 0:
            raise ValueError(f"resize size must be >= 0, got {n!r}")
        return n
