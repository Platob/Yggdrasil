"""Abstract base for network-backed :class:`Holder` implementations.

The single seam every remote backend (S3, Databricks, future Azure
/ GCS / SFTP / WebDAV) shares: **predicate pins.**
``is_remote_path = True``, the other two ``False``. Concrete
subclasses no longer reimplement these.

Subclasses implement :meth:`_stat_uncached`; the base wraps it via
:meth:`_stat` and stores the result on ``self._stat_cached``.
Mutating ops (writes, deletes) must call :meth:`_invalidate_stat_cache`
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

    stat_cache_ttl: ClassVar["float | None"] = _STAT_CACHE_TTL

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

    # Stat caches are per-process, not part of the pickled identity.
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({
        "_stat_cached", "_stat_cached_at",
    })

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stat_cached: IOStats | None = None
        self._stat_cached_at: float = 0.0

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
        cached = self._stat_cached
        if cached is None:
            return False
        ttl = self.stat_cache_ttl
        if ttl is None:
            return True
        return (time.monotonic() - self._stat_cached_at) <= ttl

    def _stat(self) -> IOStats:
        """Cached :class:`IOStats` probe.

        Entries expire after :attr:`stat_cache_ttl` seconds — past
        the budget we re-issue :meth:`_stat_uncached` instead of
        handing back a stale snapshot. On miss (or expiry),
        delegates to :meth:`_stat_uncached` and stores the fresh
        result. Subclasses override :meth:`_stat_uncached`, never
        this.
        """
        cached = self._stat_cached
        if cached is not None:
            ttl = self.stat_cache_ttl
            if ttl is None or (time.monotonic() - self._stat_cached_at) <= ttl:
                return cached
        result = self._stat_uncached()
        self._stat_cached = result
        self._stat_cached_at = time.monotonic()
        return result

    @abstractmethod
    def _stat_uncached(self) -> IOStats:
        """Backend-specific :class:`IOStats` probe. One network call."""

    def _invalidate_stat_cache(self, remove_global: bool = True) -> None:
        """Drop this path's cached entry. Call after writes / deletes.

        ``remove_global`` is accepted for backward compatibility with
        the legacy singleton-cache invalidation path; it is now a
        no-op since :class:`RemotePath` no longer maintains a
        process-wide instance cache.
        """
        del remove_global
        self._stat_cached = None
        self._stat_cached_at = 0.0

        self._unpersist_schema()

        logger.debug(f"Invalidated stat cache for {self!r}")

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

    # ------------------------------------------------------------------
    # Ancestor existence — propagate from a known-existing child
    # ------------------------------------------------------------------

    @property
    def parent(self) -> "Path":
        p = super().parent
        self._propagate_existence_to_ancestor(p)
        return p

    @property
    def parents(self) -> "Tuple[Path, ...]":
        ps = super().parents
        for ancestor in ps:
            self._propagate_existence_to_ancestor(ancestor)
        return ps

    def _propagate_existence_to_ancestor(self, ancestor: "Path") -> None:
        """Seed *ancestor*'s stat cache as a DIRECTORY when ``self`` exists.

        If this path's stat cache reports a present object (file or
        directory), every ancestor on the URL must be a directory:
        the backend can't host the child otherwise. Seeding the
        ancestor lets a follow-up ``parent.exists()`` /
        ``parent.is_dir()`` collapse into a local hit — no
        ``head_object`` / ``get_status`` / ``get_metadata`` round
        trip just to confirm the obvious.

        Only fires when *ancestor* is itself a :class:`RemotePath` and
        has no fresh cached entry of its own (we never overwrite a
        backend-confirmed stat with the inferred one).
        """
        if not isinstance(ancestor, RemotePath):
            return
        cached = self._stat_cached
        if cached is None or cached.kind == IOKind.MISSING:
            return
        existing = ancestor._stat_cached
        if existing is not None and existing.kind != IOKind.MISSING:
            return
        ancestor._seed_stat_cache(IOStats(kind=IOKind.DIRECTORY))

    def _seed_stat_cache(self, stats: IOStats) -> None:
        """Pre-populate the cache with a known :class:`IOStats`.

        Useful for backends that learn metadata as a side-effect of a
        listing (S3 ``ListObjectsV2`` returns size + mtime per object,
        Databricks ``dbutils.fs.ls`` returns size) or a read/write
        (the response body's length IS the file size). Stamps the
        cache time so the entry observes the same TTL budget as one
        produced by :meth:`_stat_uncached`. Passing the existing
        ``self._stat_cached`` (after an in-place mutation) is the
        canonical way to refresh the TTL — the assignment is a no-op,
        the timestamp moves. The next :meth:`_stat` call on the
        warmed path is a local hit.
        """
        self._stat_cached = stats
        self._stat_cached_at = time.monotonic()
