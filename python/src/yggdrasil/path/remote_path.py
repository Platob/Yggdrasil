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
from typing import Any, ClassVar, Optional

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.path.memory import Memory
from yggdrasil.path.path import Path

__all__ = ["RemotePath"]


#: Default freshness window for a seeded :class:`IOStats` entry.
#: Beyond this, :meth:`RemotePath._stat` discards the cached entry
#: and re-issues the backend probe. Five minutes matches the
#: lifetime of a typical Databricks / S3 credential refresh cycle —
#: long enough to collapse the dozen probes a Delta replay makes
#: against the same key, short enough that a stale entry doesn't
#: outlive a meaningful change to the underlying object.
_STAT_CACHE_TTL: float = 300.0

#: Default page size for the inner read/write buffer. A 4 MiB grain
#: matches Parquet row-group / Arrow IPC chunk sizes — one page covers
#: a typical footer or batch, so the first read populates exactly one
#: page and follow-up reads against the same region collapse to
#: in-process slicing.
_DEFAULT_BUFFER_SIZE: int = 4 * 1024 * 1024
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

    # Default page size for the inner buffered-page cache. ``None``
    # disables paging entirely; an int / ByteUnit / size string sets
    # the per-page grain. Subclasses with a hard backend-imposed
    # block size (e.g. an SDK that only supports whole-object PUTs
    # below a certain threshold) can pin their own default here.
    DEFAULT_BUFFER_SIZE: ClassVar["int | None"] = _DEFAULT_BUFFER_SIZE

    # Activate the :class:`Singleton` cache for every concrete remote
    # backend: 5-minute default TTL, bounded at 10 000 entries as
    # defence-in-depth against accidental cardinality explosions.
    # The default ``_singleton_key`` includes ``cls`` so S3Path /
    # DatabricksPath / future Azure paths can share one ``_INSTANCES``
    # dict without colliding.
    _SINGLETON_TTL: ClassVar[Any] = _STAT_CACHE_TTL
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=_STAT_CACHE_TTL,
        max_size=10_000,
    )
    _INSTANCES_LOCK: ClassVar[RLock] = RLock()

    # ------------------------------------------------------------------
    # Inner buffered-page cache — reduces remote round trips for
    # repeated reads against the same byte ranges, batches partial
    # writes into a single backend PUT on :meth:`flush` / release.
    # ------------------------------------------------------------------

    def __init__(
        self,
        *args: Any,
        page_size: Any = ...,
        buffersize: Any = ...,
        **kwargs: Any,
    ) -> None:
        """Wire the page-buffer state alongside the standard :class:`Path` init.

        ``page_size`` accepts an int (bytes), a
        :class:`~yggdrasil.enums.byteunit.ByteUnit` member, or a
        size string (``"4 MB"``). ``None`` disables paging and routes
        every read/write straight through to the subclass primitives.
        Omitting the kwarg uses :attr:`DEFAULT_BUFFER_SIZE`.

        ``buffersize`` is a deprecated alias for ``page_size``. When
        both are supplied, ``page_size`` wins.

        Singleton-aware: a re-init on a cached instance preserves the
        pages already in flight so a second constructor call doesn't
        silently lose dirty buffered writes from the first.
        """
        super().__init__(*args, **kwargs)
        if hasattr(self, "_page_size"):
            # Singleton re-init: keep pages + dirty markers from the
            # original construction so a second ``RemotePath(...)`` call
            # against the same key doesn't strand pending writes.
            return
        # Resolve deprecated ``buffersize`` alias: ``page_size`` wins
        # when both are supplied; ``buffersize`` is accepted silently
        # so existing callers keep working.
        effective = page_size if page_size is not ... else buffersize
        self._page_size: Optional[int] = self._normalize_page_size(effective)
        # Lazy: only allocate the dict when paging actually fires.
        self._pages: Optional[ExpiringDict[int, Memory]] = None
        self._dirty_pages: set[int] = set()
        # Tracks the logical size while buffered writes outrun what
        # the backend has committed. ``None`` means "ask the backend"
        # via the normal stat-cache path.
        self._buffered_size: Optional[int] = None

    @classmethod
    def _normalize_page_size(cls, value: Any) -> Optional[int]:
        if value is ...:
            value = cls.DEFAULT_BUFFER_SIZE
        if value is None:
            return None
        # Plain non-negative ``int`` covers the default buffer size and
        # every realistic caller (``page_size=4 * 1024 * 1024``,
        # ``page_size=None``, ``page_size=...``). Bypass the
        # ``ByteUnit.parse_size`` round trip — five isinstance probes
        # plus a function-call frame — when we already have the
        # canonical type. ``bool`` is an ``int`` subclass; reject it
        # before the fast path so ``page_size=True`` still raises.
        if type(value) is int:
            if value < 0:
                raise ValueError(
                    f"page_size must be a non-negative byte count "
                    f"(int / ByteUnit / size string / None), got {value!r}"
                )
            return value if value > 0 else None
        from yggdrasil.enums.byteunit import ByteUnit

        try:
            n = ByteUnit.parse_size(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"page_size must be a non-negative byte count "
                f"(int / ByteUnit / size string / None), got {value!r}"
            ) from exc
        return n if n > 0 else None

    @property
    def page_size(self) -> Optional[int]:
        """Page size for buffered reads/writes, or ``None`` when disabled."""
        return self._page_size

    def _ensure_pages(self) -> "ExpiringDict[int, Memory]":
        if self._pages is None:
            # Pages share the stat cache's TTL: a backend object that's
            # been quiet for 5 minutes is the same horizon at which a
            # cached size / mtime is no longer trusted. No ``max_size``
            # — dirty pages must not be evicted under the caller's feet;
            # callers manage memory via explicit :meth:`flush` /
            # :meth:`invalidate_singleton`.
            self._pages = ExpiringDict(default_ttl=self.STAT_CACHE_TTL)
        return self._pages

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

        Lets ``ParquetFile`` / ``CSVFile`` / ``ArrowIPCFile`` skip a probe
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
        ``_INSTANCES`` entry — see :meth:`Path.invalidate_singleton`.

        Also clears the inner page cache: a mutation just ran, so the
        bytes the pages held are no longer authoritative. Dirty pages
        are dropped on the floor — callers must :meth:`flush` before
        invalidating if they want pending writes to survive.
        """
        if self._pages is not None:
            self._pages.clear()
        self._dirty_pages.clear()
        self._buffered_size = None
        super().invalidate_singleton(remove_global=remove_global)
        self._unpersist_schema()

    # ------------------------------------------------------------------
    # Resize is a no-op on remote backends — the upload IS the resize
    # ------------------------------------------------------------------

    def _bread(self, n: int, pos: int, mode) -> "IO":
        from yggdrasil.io.holder import IO
        del mode
        if n == 0:
            return IO()
        try:
            data = bytes(self._read_mv(n, pos))
        except FileNotFoundError:
            data = b""
        return IO(data)

    def _bwrite(self, data, pos: int, mode) -> int:
        del mode
        if hasattr(data, "to_bytes"):
            payload = data.to_bytes()
        elif hasattr(data, "read"):
            payload = data.read()
        else:
            payload = bytes(data)
        return self._write_mv(memoryview(payload), pos)

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

    # ==================================================================
    # Buffered read / write — page cache over backend ``_read_mv`` /
    # ``_write_mv``. Public ``read_mv`` / ``write_mv`` route through
    # here when :attr:`page_size` is set; subclass primitives still do
    # the actual network calls, just one per page miss instead of one
    # per logical access.
    # ==================================================================

    def _effective_total(self) -> int:
        """Logical byte count, including buffered writes.

        Prefer the buffered-write tip when set; fall back to the
        subclass's ``size`` accessor (which may itself be a stat-cache
        probe) otherwise.
        """
        if self._buffered_size is not None:
            return self._buffered_size
        return int(self.size)

    def _stamp_buffered_size(self, new_total: int) -> None:
        """Reflect a buffered write into the stat cache.

        Buffered writes don't go to the backend until :meth:`flush`,
        but follow-up :meth:`size` / :meth:`exists` / :meth:`is_file`
        calls must observe the post-write state — subclasses (S3Path,
        VolumePath, ...) shadow :attr:`size` with their own stat-cache
        reads, so we keep the cache itself in sync rather than fighting
        the property override.
        """
        self._buffered_size = new_total
        cached = self._stat_cached
        now = time.time()
        if cached is None or cached.kind == IOKind.MISSING:
            self._persist_stat_cache(
                IOStats(
                    size=new_total,
                    kind=IOKind.FILE,
                    mtime=now,
                    media_type=self.media_type,
                )
            )
            return
        cached.size = new_total
        cached.mtime = now
        self._persist_stat_cache(cached)

    def _cache_after_upload(self, content: bytes, size: int) -> None:
        """Populate the page cache with committed content after upload.

        Called after a successful ``_upload`` so that a subsequent
        read on the same instance hits the local cache instead of
        re-downloading. Also sets ``_buffered_size`` to the committed
        size and clears dirty markers — the backend is now
        authoritative for these bytes.
        """
        self._buffered_size = size
        self._dirty_pages.clear()
        if self._page_size is None or size == 0:
            return
        page_size = self._page_size
        pages = self._ensure_pages()
        pages.clear()
        n_pages = (size + page_size - 1) // page_size
        for idx in range(n_pages):
            start = idx * page_size
            end = min(start + page_size, size)
            chunk = content[start:end]
            buf = bytearray(page_size)
            buf[: len(chunk)] = chunk
            pages.set(idx, Memory.from_bytearray(buf, size=len(chunk)))

    def read_mv(
        self,
        size: int = -1,
        offset: int = 0,
        *,
        cursor: bool = False,
    ) -> memoryview:
        if self._page_size is None or self._parent is not None:
            return super().read_mv(size, offset, cursor=cursor)
        from yggdrasil.io.holder import _resolve_pos

        if cursor:
            offset = self._pos
        total = self._effective_total()
        offset = _resolve_pos(offset, total)
        if offset < 0 or offset > total:
            raise ValueError(
                f"Offset {offset} is out of bounds for "
                f"{type(self).__name__} of size {total}"
            )
        if size < 0:
            size = total - offset
        if size < 0 or offset + size > total:
            raise ValueError(
                f"Range [{offset}, {offset + size}) is out of bounds for "
                f"{type(self).__name__} of size {total}"
            )
        out = self._paged_read(size, offset) if size > 0 else memoryview(b"")
        if cursor:
            self._pos = offset + size
        return out

    def write_mv(
        self,
        data: memoryview,
        offset: int = 0,
        *,
        size: int = -1,
        overwrite: bool = False,
        update_stat: bool = True,
        cursor: bool = False,
    ) -> int:
        if self._page_size is None or self._parent is not None:
            return super().write_mv(
                data,
                offset,
                size=size,
                overwrite=overwrite,
                update_stat=update_stat,
                cursor=cursor,
            )
        from yggdrasil.io.holder import _resolve_pos

        if cursor:
            offset = self._pos
        if size >= 0 and len(data) > size:
            data = data[:size]
        if overwrite and offset == 0:
            total = 0
            self._stamp_buffered_size(0)
        else:
            total = self._effective_total()
        offset = _resolve_pos(offset, total)
        if offset < 0:
            raise ValueError(
                f"Offset {offset} is out of bounds for "
                f"{type(self).__name__} of size {total}"
            )
        n = len(data)
        end = offset + n
        if n == 0:
            if overwrite and end < total:
                self._discard_pages_past(end)
                self._stamp_buffered_size(end)
                if update_stat:
                    self.mark_dirty()
            if cursor:
                self._pos = offset
            return 0
        new_total = end if overwrite else max(total, end)
        self._paged_write(memoryview(data), offset)
        if overwrite and new_total < total:
            self._discard_pages_past(new_total)
        elif not overwrite:
            # The stat probe can lie on some backends (e.g. a Volumes
            # directory-heuristic miss reports a file as size 0); the
            # page we just loaded carries the real tail. Stretch
            # ``new_total`` so the trailing bytes survive the flush.
            tail = self._actual_tail()
            if tail > new_total:
                new_total = tail
        self._stamp_buffered_size(new_total)
        if update_stat:
            self.mark_dirty()
        if cursor:
            self._pos = end
        # Outside an explicit acquire (``with path:`` /
        # ``path.open(owns_holder=True)``) there's no release hook to
        # flush on, so the closed-state direct-write contract still
        # demands the bytes land on the backend before this call
        # returns. The page cache still serves follow-up reads via
        # :meth:`_paged_read`.
        if not self._acquired:
            self.flush()
        return n

    def write_arrow_io(self, payload: "Any") -> int:
        """Commit an Arrow-encoded payload directly to the backend.

        Accepts a ``pa.Buffer``, ``bytes``, ``bytearray``, or
        ``memoryview`` and uploads it in one backend call — no
        page buffer, no truncate, no stat probe. Tabular IO files
        (ParquetFile, ArrowIPCFile, etc.) route through this after
        the format encoder finishes so the encoded bytes go straight
        to the remote object without intermediate copies.
        """
        if isinstance(payload, (bytes, bytearray)):
            data = payload
        elif isinstance(payload, memoryview):
            data = bytes(payload)
        else:
            data = bytes(memoryview(payload))
        self._upload(data)
        self._touch_stat(size=len(data))
        return len(data)

    def _upload(self, content: bytes) -> int:
        """Backend-specific atomic upload. Subclasses must override.

        Accepts a materialised ``bytes`` payload and writes it as the
        entire object in one round trip. Returns the byte count.
        Called by :meth:`flush` and :meth:`_write_mv` (``pos == 0``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _upload(content)."
        )

    def _write_mv(self, data: memoryview, pos: int) -> int:
        """Splice *data* at *pos* via the backend upload primitive.

        ``pos == 0`` is the fast path — a direct :meth:`_upload` with
        no preceding download. ``pos > 0`` falls back to
        read-modify-write: download the existing object, splice the
        new bytes in, re-upload.
        """
        n = len(data)
        if n == 0:
            return 0
        if pos == 0:
            self._upload(bytes(data))
            return n
        try:
            existing = bytes(self._read_mv(-1, 0))
        except FileNotFoundError:
            existing = b""
        if pos > len(existing):
            existing = existing + b"\x00" * (pos - len(existing))
        payload = existing[:pos] + bytes(data) + existing[pos + n:]
        self._upload(payload)
        return n

    def flush(self) -> None:
        """Commit dirty buffered pages to the backend in one upload."""
        if self._dirty_pages:
            size = self._effective_total()
            payload = bytes(self._paged_read(size, 0)) if size > 0 else b""
            self._upload(payload)
            self._cache_after_upload(payload, len(payload))
        super().flush()

    def _release(self) -> None:
        """Flush dirty pages before the standard release.

        Called from :meth:`close` (explicit) and :meth:`__del__` (GC).
        A failed flush at GC time is logged and swallowed — never
        raise from ``__del__`` per :class:`Disposable`.
        """
        try:
            self.flush()
        except Exception as exc:
            logger.warning(
                "Buffered flush failed during release of %r: %s",
                self,
                exc,
            )
        super()._release()

    def truncate(self, n: int) -> int:
        """Page-buffered truncate for the ``truncate(0)`` overwrite prelude.

        ``truncate(0)`` clears the page cache and stamps the logical
        size without a network call — the subsequent write + flush
        replaces the object atomically. ``truncate(n > 0)`` falls
        through to the base read-modify-write path.
        """
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        if n == 0 and self._page_size is not None:
            if self._pages is not None:
                self._discard_pages_past(0)
            self._stamp_buffered_size(0)
            self._touch_stat(size=0)
            return 0
        return super().truncate(n)



    # ------------------------------------------------------------------
    # Page-level helpers
    # ------------------------------------------------------------------

    def _paged_read(self, n: int, pos: int) -> memoryview:
        page_size = self._page_size
        pages = self._ensure_pages()
        first_page = pos // page_size
        last_page = (pos + n - 1) // page_size
        if first_page == last_page:
            page = pages.get(first_page)
            if page is None:
                page = self._fetch_page(first_page)
                pages.set(first_page, page)
            page_start = first_page * page_size
            local = pos - page_start
            return memoryview(bytes(page._buf[local : local + n]))
        out = bytearray(n)
        out_pos = 0
        end = pos + n
        for page_idx in range(first_page, last_page + 1):
            page = pages.get(page_idx)
            if page is None:
                page = self._fetch_page(page_idx)
                pages.set(page_idx, page)
            page_start = page_idx * page_size
            slice_start = max(pos, page_start) - page_start
            slice_end = min(end, page_start + page_size) - page_start
            take = slice_end - slice_start
            out[out_pos : out_pos + take] = page._buf[slice_start:slice_end]
            out_pos += take
        return memoryview(bytes(out))

    def _fetch_page(self, page_idx: int) -> Memory:
        """Load one page from the backend via the subclass primitive.

        Asks for ``page_size`` bytes and trusts the response length —
        backends that downloads the whole object (Databricks Volumes)
        return what they have; ranged backends (S3) cap at EOF on
        their own. A missing object surfaces as an empty page.
        """
        page_size = self._page_size
        page_offset = page_idx * page_size
        try:
            chunk = bytes(self._read_mv(page_size, page_offset))
        except FileNotFoundError:
            return Memory.from_bytearray(bytearray(page_size), size=0)
        if not chunk:
            return Memory.from_bytearray(bytearray(page_size), size=0)
        buf = bytearray(page_size)
        n = min(len(chunk), page_size)
        buf[:n] = chunk[:n]
        return Memory.from_bytearray(buf, size=n)

    def _paged_write(self, data: memoryview, offset: int) -> None:
        page_size = self._page_size
        pages = self._ensure_pages()
        n = len(data)
        end = offset + n
        first_page = offset // page_size
        last_page = (end - 1) // page_size
        src_pos = 0
        backend_total = Path.size.fget(self)  # type: ignore[attr-defined]
        for page_idx in range(first_page, last_page + 1):
            page_start = page_idx * page_size
            page = pages.get(page_idx)
            slice_start = max(offset, page_start) - page_start
            slice_end = min(end, page_start + page_size) - page_start
            take = slice_end - slice_start
            page_backend_len = min(
                page_size,
                max(0, backend_total - page_start),
            )
            if page is None:
                # Avoid the backend hit when the write fully covers
                # both this page's slot and any backend tail past the
                # write — there's nothing to preserve.
                if slice_start == 0 and slice_end >= page_backend_len:
                    page = Memory.from_bytearray(bytearray(page_size), size=slice_end)
                else:
                    page = self._fetch_page(page_idx)
                pages.set(page_idx, page)
            if page._size < slice_end:
                page._size = slice_end
            page._buf[slice_start:slice_end] = data[src_pos : src_pos + take]
            src_pos += take
            self._dirty_pages.add(page_idx)
            # Re-stamp with no expiry so a dirty page can't TTL out
            # from under a pending flush.
            pages.set(page_idx, page, ttl=None)

    def _actual_tail(self) -> int:
        """Largest logical byte index any loaded page reports.

        The page cache learns the file's real tail as a side effect of
        :meth:`_fetch_page` — backends that download whole objects
        regardless of the requested range surface their true size on
        the loaded :class:`Memory`, even when :meth:`_stat` lies about
        it. Walk the live pages and report the max ``page_start +
        page._size`` so partial writes don't truncate the unread tail.
        """
        if self._pages is None:
            return 0
        page_size = self._page_size
        tail = 0
        for key in list(self._pages.keys()):
            page = self._pages.get(key)
            if page is None:
                continue
            candidate = key * page_size + page._size
            if candidate > tail:
                tail = candidate
        return tail

    def _discard_pages_past(self, end: int) -> None:
        if self._pages is None:
            return
        page_size = self._page_size
        # Index of the first page that lies entirely past ``end``.
        cutoff = (end + page_size - 1) // page_size
        for key in list(self._pages.keys()):
            if key >= cutoff:
                self._pages.pop(key, None)
                self._dirty_pages.discard(key)
