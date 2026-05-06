"""Abstract base for network-backed :class:`Path` implementations.

Why this exists
---------------
The abstract :class:`Path` is already remote-friendly by default —
its transaction-buffer machinery downloads on first read, splices
positional writes into a local :class:`BytesIO`, and commits via a
single ``_pwrite`` on flush. ``LocalPath`` is the special case that
overrides that with a long-lived file descriptor.

What was missing was a taxonomic seam: every remote backend (S3,
Databricks, future Azure / GCS / SFTP / WebDAV) had to repeat the
same handful of class-level facts, *and* re-implement the short
stat cache that hot loops (DeltaIO replay, repeated ``exists()``
checks) need to keep the round-trip count manageable.

:class:`RemotePath` is that seam. It pins ``is_local = False`` and
funnels every backend's ``_stat`` through a shared
:class:`ExpiringDict` keyed by :meth:`Path.full_path`. Concrete
subclasses implement the network probe via :meth:`_stat_uncached`
and call :meth:`_invalidate_stat_cache` after writes / deletes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.io.fs.path import Path
from yggdrasil.io.io_stats import IOStats

__all__ = ["RemotePath"]


#: Default TTL (seconds) for cached :class:`IOStats` entries. Short
#: enough to avoid stale reads in typical workflows, long enough to
#: collapse the 10+ probes a single DeltaIO replay or repeated
#: ``exists()`` loop makes against the same key.
_STAT_CACHE_TTL: float = 30.0

#: Maximum cached stat entries across every :class:`RemotePath`
#: instance. Bounded so a million-key walk can't pin unbounded RAM.
_STAT_CACHE_MAX: int = 4096


# Process-wide :class:`IOStats` cache, shared across every
# :class:`RemotePath` subclass. Keyed by :meth:`Path.full_path`, which
# is unique enough across schemes (``s3://...`` vs ``dbfs:/...``) that
# different backends can share the same dict without colliding.
_STAT_CACHE: "ExpiringDict[str, IOStats]" = ExpiringDict(
    default_ttl=_STAT_CACHE_TTL,
    max_size=_STAT_CACHE_MAX,
)


class RemotePath(Path, ABC):
    """Abstract :class:`Path` for network-backed backends.

    Concrete subclasses inherit the base :class:`Path`'s
    transaction-buffer machinery by default; they only override
    :meth:`pread` / :meth:`pwrite` / :meth:`write_stream` when the
    backend has a cheaper primitive than download-and-slice or
    upload-the-whole-buffer (e.g. S3 Range GETs, DBFS FUSE).

    Stat caching is mutualized here. Subclasses implement the raw
    network probe in :meth:`_stat_uncached`; :meth:`_stat` wraps it
    in the shared :data:`_STAT_CACHE`. Mutating ops
    (:meth:`_remove_file`, :meth:`_remove_dir`, ``_pwrite``,
    ``write_stream``) must call :meth:`_invalidate_stat_cache` so
    follow-up reads see fresh metadata.
    """

    __slots__ = ()

    #: Backing :class:`ExpiringDict`. Class-level so a subclass can
    #: bind a private bucket if it wants to (e.g. tests that need
    #: isolation), but the default points everyone at the shared
    #: process-wide cache.
    _stat_cache: ClassVar["ExpiringDict[str, IOStats]"] = _STAT_CACHE

    @property
    def is_local_path(self) -> bool:
        return False

    @property
    def is_remote_path(self) -> bool:
        return True

    # ==================================================================
    # Stat caching — the mutual layer
    # ==================================================================

    def _stat_cache_key(self) -> str:
        """Cache key for :meth:`_stat`.

        Defaults to :meth:`full_path`; backends with a cheaper
        canonical key (S3's ``bucket/key``) override.
        """
        return self.full_path()

    def _stat(self) -> IOStats:
        """Cached stat. Subclasses implement :meth:`_stat_uncached`."""
        cache = type(self)._stat_cache
        key = self._stat_cache_key()
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = self._stat_uncached()
        cache.set(key, result)
        return result

    @abstractmethod
    def _stat_uncached(self) -> IOStats:
        """Backend-specific stat probe — wired into the cache by :meth:`_stat`."""

    def _invalidate_stat_cache(self) -> None:
        """Drop this path's cached stat entry. Call after writes / deletes."""
        type(self)._stat_cache.pop(self._stat_cache_key(), None)

    @classmethod
    def clear_stat_cache(cls) -> None:
        """Drop every cached stat entry on this class's cache.

        Useful in tests; production code should prefer per-path
        :meth:`_invalidate_stat_cache` so other paths keep their
        warm entries.
        """
        cls._stat_cache.clear()

    def _seed_stat_cache(self, stats: IOStats) -> None:
        """Pre-populate the cache for this path with a known result.

        Backends that learn ``IOStats`` as a side-effect of a
        listing (S3 ``ListObjectsV2`` returns size + mtime per
        object) can warm the cache so the next ``_stat()`` is a
        local hit.
        """
        type(self)._stat_cache.set(self._stat_cache_key(), stats)
