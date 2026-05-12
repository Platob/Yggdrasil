"""Abstract base for network-backed :class:`Holder` implementations.

Two seams that every remote backend (S3, Databricks, future Azure /
GCS / SFTP / WebDAV) shares:

1. **Predicate pins.** ``is_remote_path = True``, the other two
   ``False``. Concrete subclasses no longer reimplement these.

2. **Singleton-by-URL.** Two callers building a :class:`RemotePath`
   with the same ``(cls, url)`` receive the **same** instance — and
   therefore share one ``_stat_cached`` slot. Remote stat probes are
   slow and hot loops hammer them (Delta replay, tight ``exists()``
   poll, folder listing that re-stats each child); singleton paths
   collapse those into one network call per URL, no separate cache
   dict required. The :class:`ExpiringDict` cache is process-wide and
   shared across every :class:`RemotePath` subclass — URLs are
   canonical and unique across schemes, so an ``s3://`` and a
   ``dbfs:/`` of the same logical resource never collide.

Subclasses implement :meth:`_stat_uncached`; the base wraps it via
:meth:`_stat` and stores the result on ``self._stat_cached``.
Mutating ops (writes, deletes) must call :meth:`_invalidate_stat_cache`
so follow-up reads see fresh metadata. Sister of
:class:`yggdrasil.io.fs.local_path.LocalPath`: same :class:`Holder`
substrate, different backing.
"""

from __future__ import annotations

import threading
import time
from abc import abstractmethod
from typing import Any, ClassVar, Tuple

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.io.path.path import Path
from yggdrasil.io.url import URL


__all__ = ["RemotePath"]


#: Default freshness window for a seeded :class:`IOStats` entry.
#: Beyond this, :meth:`RemotePath._stat` discards the cached entry
#: and re-issues the backend probe. Five minutes matches the
#: lifetime of a typical Databricks / S3 credential refresh cycle —
#: long enough to collapse the dozen probes a Delta replay makes
#: against the same key, short enough that a stale entry doesn't
#: outlive a meaningful change to the underlying object.
_STAT_CACHE_TTL: float = 300.0


def _resolve_url_from_args(args: tuple, kwargs: dict) -> "URL | None":
    """Best-effort :class:`URL` from constructor args/kwargs.

    ``__new__`` runs before ``__init__`` normalizes its inputs, so we
    peek at whatever is already URL-shaped — and try a best-effort
    :meth:`URL.from_` parse on string / :class:`pathlib.PurePath`
    seeds so the common ``S3Path("s3://bucket/key")`` shape hits the
    singleton cache instead of allocating a fresh instance per call.
    Returns ``None`` only when nothing parses cleanly; that path
    falls back to a fresh allocation rather than crashing
    construction.

    The parsed URL is stashed onto the new instance as
    :attr:`_resolved_url` so the eventual subclass ``__init__`` can
    reuse it without re-parsing — round-trip costs roughly half on
    string-shaped constructor calls.
    """
    url = kwargs.get("url")
    if isinstance(url, URL):
        return url
    if url is not None:
        try:
            return URL.from_(url)
        except Exception:
            return None
    if args:
        seed = args[0]
        if isinstance(seed, URL):
            return seed
        if isinstance(seed, (str, bytes)):
            try:
                return URL.from_(seed)
            except Exception:
                return None
        # pathlib / os.PathLike fall through here. ``URL.from_``
        # accepts ``__fspath__``-able objects.
        try:
            return URL.from_(seed)
        except Exception:
            return None
    return None


class RemotePath(Path):
    """Abstract :class:`Holder` for network-backed backends.

    Subclasses pick a ``scheme`` (``s3``, ``dbfs``, …), implement the
    five :class:`Holder` primitives against their network client, and
    override :meth:`_stat_uncached` for the metadata probe. Everything
    else (predicate pins, stat caching, singleton-by-URL) is inherited
    from this base.
    """

    __slots__ = (
        "_stat_cached",
        "_stat_cached_at",
        "_initialized",
        # Parsed-URL stash populated by :meth:`__new__` so subclass
        # ``__init__`` (S3Path, DBFSPath, VolumePath, …) doesn't
        # re-parse the same string. Reset on the first ``__init__``
        # pass so it doesn't keep a reference to the URL after the
        # holder's own ``_url`` slot is populated.
        "_resolved_url",
    )

    #: Per-class freshness window for ``_stat_cached``. Subclasses can
    #: tighten or loosen the budget — e.g. a path on an append-only
    #: object store could push this higher; an interactive notebook
    #: surface could drop it. ``None`` disables the TTL check (the
    #: entry lives until ``_invalidate_stat_cache`` drops it).
    stat_cache_ttl: ClassVar["float | None"] = _STAT_CACHE_TTL

    #: Process-wide singleton cache shared across every
    #: :class:`RemotePath` subclass. Keyed by ``(cls, str(url))`` so
    #: subclasses with the same URL stay distinct (an ``S3Path`` and a
    #: hypothetical ``S3SignedPath`` for the same key are still
    #: different instances). ``default_ttl=None`` keeps entries for the
    #: process lifetime; 4096-entry cap evicts least-recently-set when
    #: full.
    _INSTANCES: ClassVar["ExpiringDict[Tuple[type, str], RemotePath]"] = ExpiringDict(
        default_ttl=None,
        max_size=4096,
    )

    #: Serializes first-time ``__init__`` on a freshly allocated
    #: singleton. ``__new__`` can hand the same instance to two
    #: threads racing on the same URL; without a lock both would run
    #: ``super().__init__`` and reset ``_stat_cached`` / ``_stat_cached_at``
    #: to their defaults, potentially overwriting a seed produced
    #: by another caller in between. The lock is held only for the
    #: brief setup window; the fast path (``_initialized`` already
    #: True) short-circuits before touching it.
    _INIT_LOCK: ClassVar[threading.Lock] = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton-by-URL construction
    # ------------------------------------------------------------------

    def __new__(cls, *args: Any, **kwargs: Any):
        # Singleton fast path: look the cache up *before* falling
        # through to ``super().__new__`` (Holder → URLBased → Disposable
        # — each layer does some work, and on a warm hit we shouldn't
        # pay for it). The cache lookup needs a parsed URL key, but
        # ``URL.from_`` on a string is much cheaper than the full
        # construction chain, so the trade is a clear win.
        url_obj = _resolve_url_from_args(args, kwargs)
        if url_obj is not None:
            key = (cls, str(url_obj))
            existing = cls._INSTANCES.get(key)
            if existing is not None and type(existing) is cls:
                return existing

        # Let the abstract-class dispatch chain (Holder.__new__ /
        # DatabricksPath.__new__) settle on a concrete subclass.
        instance = super().__new__(cls, *args, **kwargs)
        # Dispatch landed on a different class (e.g. ``DatabricksPath``
        # forwarding to ``VolumePath``); that branch ran its own
        # ``__new__`` and either cached or chose not to. Don't override.
        if type(instance) is not cls:
            return instance
        if url_obj is None:
            return instance
        # Stash the parsed URL onto the new instance so the subclass
        # ``__init__`` reuses it instead of calling ``URL.from_(data)``
        # a second time.
        try:
            object.__setattr__(instance, "_resolved_url", url_obj)
        except AttributeError:
            pass
        # ``get_or_set`` is atomic under :class:`ExpiringDict`'s lock —
        # racing constructors collapse to the same singleton without an
        # external mutex.
        return cls._INSTANCES.get_or_set((cls, str(url_obj)), lambda: instance)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Singleton-cached instances are re-entered on every constructor
        # call (Python always invokes ``__init__`` after ``__new__``);
        # skip the second pass so the live ``_stat_cached`` and any
        # subclass-side state stay untouched. The lock + double-check
        # serializes first-time init: two threads racing on the same
        # URL won't both reset the defaults.
        if getattr(self, "_initialized", False):
            return
        with self._INIT_LOCK:
            if getattr(self, "_initialized", False):
                return
            super().__init__(*args, **kwargs)
            self._stat_cached: IOStats | None = None
            self._stat_cached_at: float = 0.0
            # Drop the parsed-URL stash now that ``Holder.__init__`` has
            # bound ``self._url`` — keeping the slot around just adds a
            # dangling reference.
            try:
                object.__setattr__(self, "_resolved_url", None)
            except AttributeError:
                pass
            self._initialized = True

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

        Singleton-by-URL means ``self._stat_cached`` is effectively a
        process-wide entry for this URL: a peer constructed elsewhere
        with the same URL shares this slot. Entries expire after
        :attr:`stat_cache_ttl` seconds — past the budget we re-issue
        :meth:`_stat_uncached` instead of handing back a stale
        snapshot. On miss (or expiry), delegates to
        :meth:`_stat_uncached` and stores the fresh result.
        Subclasses override :meth:`_stat_uncached`, never this.
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
        """Drop this path's cached entry. Call after writes / deletes."""
        self._stat_cached = None
        self._stat_cached_at = 0.0

        if remove_global:
            self._INSTANCES.pop((type(self), str(self.url)), None)

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
