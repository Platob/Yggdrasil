"""Abstract base for network-backed :class:`Holder` implementations.

Two seams that every remote backend (S3, Databricks, future Azure /
GCS / SFTP / WebDAV) shares:

1. **Predicate pins.** ``is_remote_path = True``, the other two
   ``False``. Concrete subclasses no longer reimplement these.

2. **Stat cache.** Remote stat probes are slow and hot loops hammer
   them — a Delta replay, a tight ``exists()`` poll, a folder listing
   that re-stats each child. The cache is process-wide, keyed by URL,
   and shared across every :class:`RemotePath` subclass so an
   ``s3://`` and a ``dbfs:/`` of the same logical resource never
   collide (different URLs).

Subclasses implement :meth:`_stat_uncached`; the base wraps it in the
cache via :meth:`_stat`. Mutating ops (writes, deletes) must call
:meth:`_invalidate_stat_cache` so follow-up reads see fresh metadata.
Sister of :class:`yggdrasil.io.fs.local_path.LocalPath`: same
:class:`Holder` substrate, different backing.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.io.io_stats import IOStats
from yggdrasil.io.path.path import Path


__all__ = ["RemotePath"]


class RemotePath(Path):
    """Abstract :class:`Holder` for network-backed backends.

    Subclasses pick a ``scheme`` (``s3``, ``dbfs``, …), implement the
    five :class:`Holder` primitives against their network client, and
    override :meth:`_stat_uncached` for the metadata probe. Everything
    else (predicate pins, stat caching) is inherited from this base.
    """

    __slots__ = (
        "_stat_cached"
    )

    #: Process-wide cache shared across every :class:`RemotePath`
    #: subclass. URL is canonical and unique across schemes, so
    #: there's no collision risk between (e.g.) S3 and DBFS sharing
    #: the same dict. 30s TTL, 4096-entry cap — short enough to dodge
    #: stale reads in normal workflows, long enough to collapse the
    #: dozen probes a single Delta replay makes against the same key.
    _STAT_CACHE: ClassVar["ExpiringDict[str, IOStats]"] = ExpiringDict(
        default_ttl=30.0,
        max_size=4096,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stat_cached: IOStats | None = None

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

    def _stat(self) -> IOStats:
        """Cached :class:`IOStats` probe.

        Looks up by ``str(self.url)`` in :data:`_STAT_CACHE`; on miss,
        delegates to :meth:`_stat_uncached` and caches the result.
        Subclasses override :meth:`_stat_uncached`, never this.
        """
        if self._stat_cached is not None:
            return self._stat_cached

        key = str(self.url)
        hit = self._STAT_CACHE.get(key)
        if hit is not None:
            return hit
        result = self._stat_uncached()
        self._STAT_CACHE.set(key, result)
        return result

    @abstractmethod
    def _stat_uncached(self) -> IOStats:
        """Backend-specific :class:`IOStats` probe. One network call."""

    def _invalidate_stat_cache(self) -> None:
        """Drop this path's cached entry. Call after writes / deletes."""
        self._stat_cached = None
        self._STAT_CACHE.pop(str(self.url), None)

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

    def _seed_stat_cache(self, stats: IOStats) -> None:
        """Pre-populate the cache with a known :class:`IOStats`.

        Useful for backends that learn metadata as a side-effect of a
        listing (S3 ``ListObjectsV2`` returns size + mtime per object,
        Databricks ``dbutils.fs.ls`` returns size). The next
        :meth:`_stat` call on the warmed path is a local hit.
        """
        self._stat_cached = stats
        self._STAT_CACHE.set(str(self.url), stats)