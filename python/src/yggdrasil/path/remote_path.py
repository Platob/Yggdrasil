"""Abstract base for network-backed :class:`Holder` implementations.

The single seam every remote backend (S3, Databricks, future Azure
/ GCS / SFTP / WebDAV) shares: **predicate pins.**
``is_remote_path = True``, the other two ``False``. Concrete
subclasses no longer reimplement these.

Subclasses implement :meth:`_stat_uncached` (the metadata probe),
:meth:`_read_mv` (a positional / ranged GET) and :meth:`_upload` (an
atomic whole-object PUT). The base wraps the stat probe via
:meth:`_stat` and stores the result on ``self._stat_cached``.

I/O model — **whole-blob, no page cache.** A read pulls the requested
window straight through the subclass :meth:`_read_mv` (ranged where the
backend supports it, whole-object otherwise); a write replaces the whole
object via :meth:`_upload`. There is no inner buffering layer: callers
that want to coalesce small writes wrap the path in
:class:`yggdrasil.io.holder.IO`. Large Arrow/Parquet writes still stream
— when a backend advertises :attr:`SUPPORTS_STREAMING_UPLOAD` the encode
spills to a temp file and :meth:`_upload_stream` pushes it in bounded
chunks; large reads can range-read via :meth:`arrow_random_access_file`
when :attr:`SUPPORTS_RANGED_RANDOM_ACCESS` is set.

Mutating ops (writes, deletes) call :meth:`invalidate_singleton` so
follow-up reads see fresh metadata. Sister of
:class:`yggdrasil.io.fs.local_path.LocalPath`: same :class:`Holder`
substrate, different backing.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Any, ClassVar

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.path.path import Path

__all__ = ["RemotePath"]


#: Default freshness window for a seeded :class:`IOStats` entry.
#: Beyond this, :meth:`RemotePath._stat` discards the cached entry
#: and re-issues the backend probe. One minute is short enough that a
#: stale ``exists`` / ``is_file`` / ``size`` read can't outlive a
#: mutation made through another path instance, node, or external tool
#: (Databricks UI, ``aws s3 rm`` …) by more than a tick — while still
#: collapsing the burst of probes a single Delta replay or directory
#: walk makes against the same key. Mutating ops (write, remove) drop
#: the entry immediately via :meth:`invalidate_singleton`; this TTL only
#: governs how long a *read*-populated snapshot is trusted.
_STAT_CACHE_TTL: float = 60.0
logger = logging.getLogger(__name__)


class _RangedArrowReader:
    """Seekable binary reader over a holder's ranged ``_read_mv``.

    Wrapped in :class:`pyarrow.PythonFile` so pyarrow issues range reads
    for the bytes it actually touches (a Parquet footer + projected
    column chunks) instead of downloading the whole object. Read-only;
    bounded by the path's known size so a seek-to-end resolves without a
    fetch. Used by :meth:`RemotePath.arrow_random_access_file`.
    """

    __slots__ = ("_holder", "_size", "_pos")

    def __init__(self, holder: "RemotePath", size: int) -> None:
        self._holder = holder
        self._size = size
        self._pos = 0

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        else:
            self._pos = self._size + offset
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            data = bytes(self._holder._read_mv(-1, self._pos))
        else:
            remaining = self._size - self._pos
            if n == 0 or remaining <= 0:
                return b""
            data = bytes(self._holder._read_mv(min(n, remaining), self._pos))
        self._pos += len(data)
        return data

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class RemotePath(Path):
    """Abstract :class:`Holder` for network-backed backends.

    Subclasses pick a ``scheme`` (``s3``, ``dbfs``, …), implement the
    filesystem hooks plus :meth:`_read_mv` (ranged / whole-object GET)
    and :meth:`_upload` (atomic whole-object PUT) against their network
    client, and override :meth:`_stat_uncached` for the metadata probe.
    Everything else (predicate pins, stat caching, whole-blob read /
    write, optional streaming) is inherited from this base.

    ``RemotePath`` leaves the :class:`Singleton` machinery
    **deactivated** (``_SINGLETON_TTL = ...``), the same as
    :class:`Holder`: a leaf object/file path is a cheap redirector
    onto a long-lived backend resource (the :class:`S3Bucket`, the
    :class:`DatabricksService` / client), so there's nothing
    expensive to share — and *not* interning instances means two
    callers asking for the same URL get independent stat caches, so a
    delete or external mutation observed through one path is never
    masked by a sibling's cached snapshot. Only the heavyweight
    *container* resources that genuinely benefit from a shared,
    rarely-changing identity — :class:`S3Bucket`, UC
    :class:`~yggdrasil.databricks.catalog.UCCatalog` /
    :class:`~yggdrasil.databricks.schema.UCSchema` /
    :class:`~yggdrasil.databricks.volume.Volume` — opt back in by
    setting their own ``_SINGLETON_TTL``. ``iterdir``-style hot loops
    still pass ``singleton_ttl=False`` explicitly; a caller that wants
    a specific leaf path interned can pass ``singleton_ttl=None`` or a
    seconds count.
    """

    # Bound the freshness window for both probe-populated and
    # listing-seeded entries. ``Path`` ships the slot at ``None``
    # (live forever) since LocalPath / Memory don't need a TTL;
    # remote backends pay round-trip stat probes and want a window
    # that beats credential / consistency drift.
    STAT_CACHE_TTL: ClassVar["float | None"] = _STAT_CACHE_TTL

    # Backends whose :meth:`_read_mv` genuinely range-reads (only the
    # requested window crosses the wire — S3 GetObject Range, the
    # Volumes Files API Range) set this so :class:`ParquetFile` can hand
    # pyarrow a ranged random-access file for column / row-group
    # projection (footer + projected chunks only) instead of
    # snapshotting the whole object. Backends that download the whole
    # object per read (Workspace) or chunk awkwardly (DBFS) leave it
    # False so projection keeps the single-GET snapshot.
    SUPPORTS_RANGED_RANDOM_ACCESS: ClassVar[bool] = False

    # When True, an Arrow/Parquet write to this backend spills the encode to a
    # temp file and the upload streams it from disk in bounded chunks (see
    # :meth:`_upload_stream`) — so a multi-GB write never materialises whole in
    # memory. Backends whose upload rides a third-party transport that already
    # streams (S3 via boto3, Workspace via the Databricks SDK) leave it False
    # and keep the in-memory commit. Defaults False; :class:`VolumePath` opts in.
    SUPPORTS_STREAMING_UPLOAD: ClassVar[bool] = False

    # Singleton instance-caching stays OFF for leaf object/file paths
    # (``...`` = the :class:`Singleton` base default, "don't cache").
    # A leaf path holds no expensive state of its own — the client,
    # connection pool, and listing cache all live on the container
    # resource it redirects to (:class:`S3Bucket`, the Databricks
    # client) — so interning leaf instances bought nothing but a way
    # for one caller's stale stat snapshot to leak into another's view
    # of a freshly mutated object. Container resources that DO benefit
    # from shared identity (:class:`S3Bucket`, ``UCCatalog`` /
    # ``UCSchema`` / ``Volume``) override this with their own TTL.
    #
    # ``_INSTANCES`` is still a per-class dict so that an explicit
    # ``singleton_ttl=`` (or :meth:`to_singleton`) on a subclass
    # partitions away from a hot ``__new__`` on a sibling class; the
    # ``_singleton_key`` already includes ``cls`` so collisions are
    # impossible, partitioning by class only buys parallelism. No
    # companion lock anywhere — :class:`ExpiringDict.get_or_set` (used
    # by :class:`Singleton.__new__`) is GIL-atomic and cannot deadlock.
    _SINGLETON_TTL: ClassVar[Any] = ...
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=_STAT_CACHE_TTL,
        max_size=10_000,
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Disk scratch for an *acquired* write window (``with path:`` /
        # ``path.open("wb")``). ``None`` means no buffered writes; otherwise a
        # temp :class:`LocalPath` that successive writes (and the
        # ``truncate(0)`` an ``open("wb")`` does on acquire) splice into, so
        # the window coalesces into a *single* streamed upload on
        # :meth:`flush` / release instead of one PUT per ``write()`` — and the
        # bytes page through the OS file cache rather than piling up whole in
        # process memory. Un-acquired writes skip the scratch and upload
        # straight through. Guard the singleton re-init so a second constructor
        # call on a cached container path can't strand a scratch handle.
        if not hasattr(self, "_scratch"):
            self._scratch: "Any | None" = None

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

        Entries expire after :attr:`STAT_CACHE_TTL` seconds — past
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

        A mutation just ran, so the cached metadata is no longer
        authoritative; the next read re-probes the backend. Discards any
        un-flushed write scratch (callers must :meth:`flush` first to keep
        pending writes).
        """
        self._discard_scratch()
        super().invalidate_singleton(remove_global=remove_global)
        self._unpersist_schema()

    # ------------------------------------------------------------------
    # Acquired-window write scratch — disk paging, single streamed commit
    # ------------------------------------------------------------------
    def _ensure_scratch(self, *, seed: bool):
        """Lazily mint the acquired-window scratch — a temp :class:`LocalPath`
        writes splice into. When *seed*, download the existing object into it
        (one GET, streamed to disk) so positional / append writes preserve the
        prior content; otherwise start empty. Subclasses with a cheaper local
        staging surface (e.g. a cluster-mounted volume) may override."""
        if self._scratch is not None:
            return self._scratch
        from yggdrasil.io.base import _mint_spill_path
        from yggdrasil.path.local_path import LocalPath

        scratch = LocalPath(str(_mint_spill_path("yggwb", 3600)))
        if seed:
            try:
                existing = self._read_mv(-1, 0)
            except FileNotFoundError:
                existing = b""
            if len(existing):
                scratch.write_bytes(bytes(existing), overwrite=True)
        self._scratch = scratch
        return scratch

    def _discard_scratch(self) -> None:
        """Drop the scratch and reclaim its temp file (no upload)."""
        scratch = self._scratch
        self._scratch = None
        if scratch is not None:
            try:
                scratch.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass

    def _touch_stat(
        self,
        *,
        size: "int | None" = None,
        mtime: "float | None" = None,
        media_type: Any = None,
    ) -> None:
        """Post-write metadata update, kept consistent for remote paths.

        :class:`Holder._touch_stat` only updates the holder-owned ``_size``, but
        a remote path reads :attr:`size` / :attr:`exists` from the **stat
        cache** — so after a write the cache would stay stale (e.g. ``MISSING``
        from a pre-write probe). Mirror the new size / media type into the stat
        cache (creating it, or flipping it to ``FILE``) so a follow-up
        ``size`` / ``exists`` reflects the write without a re-stat round trip.
        """
        super()._touch_stat(size=size, mtime=mtime, media_type=media_type)
        cached = self._stat_cached
        if cached is None or cached.kind == IOKind.MISSING:
            self._persist_stat_cache(
                IOStats(
                    size=int(size) if size is not None else 0,
                    kind=IOKind.FILE,
                    mtime=mtime if mtime is not None else time.time(),
                    media_type=media_type if media_type is not None else self.media_type,
                )
            )
            return
        if size is not None:
            cached.size = int(size)
        cached.mtime = mtime if mtime is not None else time.time()
        if media_type is not None:
            cached.media_type = media_type
        self._persist_stat_cache(cached)

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

    # ==================================================================
    # Whole-blob read / write — straight through the subclass primitives
    # ==================================================================

    def arrow_random_access_file(self):
        """Yield a pyarrow random-access file backed by ranged ``_read_mv``.

        Lets pyarrow readers seek and pull only the bytes they touch — a
        Parquet column / row-group projection fetches the footer plus the
        projected chunks, instead of snapshotting the whole object the
        way :meth:`arrow_input_stream` does. :class:`ParquetFile` reaches
        for this when a projection is bound and the backend advertises
        :attr:`SUPPORTS_RANGED_RANDOM_ACCESS` (S3, Volumes); a full read
        still snapshots. Generic over any holder via ``_read_mv`` +
        ``size``.
        """
        import contextlib

        import pyarrow as pa

        @contextlib.contextmanager
        def _ctx():
            handle = pa.PythonFile(
                _RangedArrowReader(self, int(self.size)), mode="r",
            )
            try:
                yield handle
            finally:
                handle.close()

        return _ctx()

    def read_byte_range(self, offset: int, length: int = -1) -> memoryview:
        """Read exactly *length* bytes from *offset* — a ranged backend fetch.

        The explicit byte-range surface for tabular / format readers that
        want a specific window (a Parquet footer, an Arrow IPC block) without
        snapshotting the whole object. Works whether the holder is opened or
        not: an in-flight write scratch is served from disk, otherwise the
        subclass :meth:`_read_mv` issues a ranged GET on backends that
        support it. ``length < 0`` reads to EOF.

        An explicit non-negative window goes straight to :meth:`_read_mv` —
        no ``self.size`` (HEAD) bounds probe, so a footer fetch is a single
        ranged GET. A short read near EOF is the caller's to interpret.
        """
        if offset >= 0 and length >= 0 and self._scratch is None:
            return self._read_mv(length, offset)
        return self.read_mv(length, offset)

    def read_mv(
        self,
        size: int = -1,
        offset: int = 0,
        *,
        cursor: bool = False,
    ) -> memoryview:
        """Read — served from the acquired write scratch when one is in
        flight, otherwise straight through to the backend.

        Inside an acquired window with un-flushed writes the scratch file is
        the authoritative view (read-after-write within the handle); without
        a scratch this is the base :class:`Holder` read over the subclass
        :meth:`_read_mv` (ranged where the backend supports it).
        """
        if self._parent is not None or self._scratch is None:
            return super().read_mv(size, offset, cursor=cursor)
        if cursor:
            offset = self._pos
        out = self._scratch.read_mv(size, offset)
        if cursor:
            self._pos = offset + len(out) if offset >= 0 else self._scratch.size
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
        """Whole-blob write — direct upload when closed, disk-paged when open.

        - **Closed (un-acquired).** A whole-object overwrite from the start
          (``offset == 0``, ``overwrite``, no cursor; what
          ``write_bytes(...)`` resolves to) is a *single* :meth:`_upload`,
          no stat probe, no read-modify-write — the atomic PUT replaces the
          object. Positional / partial writes defer to the base
          :class:`Holder` splice (download, splice, re-upload via
          :meth:`_write_mv`).
        - **Open (acquired** — ``with path:`` / ``path.open("wb")``**).**
          The write splices into a temp-file scratch (paging through the OS
          cache, not piling up in memory); :meth:`flush` / release streams
          the scratch to the backend in one upload.
        """
        if self._parent is not None:
            return super().write_mv(
                data, offset, size=size, overwrite=overwrite,
                update_stat=update_stat, cursor=cursor,
            )
        if size >= 0 and len(data) > size:
            data = data[:size]
        if cursor:
            offset = self._pos
        n = len(data)

        if self._acquired:
            # Disk-paged window: splice into the scratch, commit on flush.
            # A whole-object overwrite at 0 needn't preserve prior bytes, so
            # it never seeds; positional / append seeds from the backend once.
            whole = offset == 0 and overwrite
            scratch = self._ensure_scratch(seed=not whole)
            if whole:
                scratch.truncate(0)
            scratch.write_mv(memoryview(bytes(data)), offset, overwrite=overwrite)
            if update_stat:
                self._touch_stat(size=int(scratch.size))
            if cursor:
                self._pos = offset + n
            return n

        if offset == 0 and overwrite:
            payload = bytes(data)
            self._upload(payload)
            if update_stat:
                self._touch_stat(size=len(payload))
            if cursor:
                self._pos = len(payload)
            return len(payload)
        return super().write_mv(
            data, offset, size=size, overwrite=overwrite,
            update_stat=update_stat, cursor=False,
        )

    def write_arrow_io(self, payload: "Any") -> int:
        """Commit an Arrow-encoded payload directly to the backend.

        Accepts a ``pa.Buffer``, ``bytes``, ``bytearray``, or
        ``memoryview`` and uploads it in one backend call — no
        truncate, no stat probe. Tabular IO files (ParquetFile,
        ArrowIPCFile, etc.) route through this after the format encoder
        finishes so the encoded bytes go straight to the remote object
        without intermediate copies. Whole-object replace: any in-flight
        write scratch is superseded.
        """
        if isinstance(payload, (bytes, bytearray)):
            data = payload
        elif isinstance(payload, memoryview):
            data = bytes(payload)
        else:
            data = bytes(memoryview(payload))
        self._discard_scratch()
        self._upload(data)
        self._touch_stat(size=len(data))
        return len(data)

    def _upload(self, content: bytes) -> int:
        """Backend-specific atomic upload. Subclasses must override.

        Accepts a materialised ``bytes`` payload and writes it as the
        entire object in one round trip. Returns the byte count.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _upload(content)."
        )

    def _upload_stream(self, source: "Any") -> int:
        """Upload *source* (a seekable, sized Holder) as the whole object.

        Default: materialise *source* and defer to :meth:`_upload` — keeps
        backends that don't stream (S3, Workspace, DBFS) on their existing
        path. Only reached when ``SUPPORTS_STREAMING_UPLOAD`` is set, so the
        default is a safety net; :class:`VolumePath` overrides it to push
        *source* to the Files API in bounded chunks. Returns the byte count.
        """
        return self._upload(source.read_bytes())

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

    def truncate(self, n: int) -> int:
        """Resize the object to exactly *n* bytes.

        - **Acquired** (the ``open("wb")`` truncate-on-acquire, and
          explicit truncates inside a ``with``): resize the disk scratch;
          the commit happens once on :meth:`flush`. An empty ``truncate(0)``
          therefore costs no PUT until release — and ``open("wb")``
          immediately followed by a write coalesces to a single upload.
        - **Closed**: a whole-object upload — ``truncate(0)`` PUTs an
          empty object; ``truncate(n > 0)`` downloads, slices /
          zero-extends, re-uploads.
        """
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        if self._acquired:
            scratch = self._ensure_scratch(seed=n > 0)
            scratch.truncate(n)
            self._touch_stat(size=n)
            return n
        if n == 0:
            self._upload(b"")
            self._touch_stat(size=0)
            return 0
        try:
            existing = bytes(self._read_mv(-1, 0))
        except FileNotFoundError:
            existing = b""
        if len(existing) >= n:
            payload = existing[:n]
        else:
            payload = existing + b"\x00" * (n - len(existing))
        self._upload(payload)
        self._touch_stat(size=n)
        return n

    def flush(self) -> None:
        """Commit the acquired write scratch to the backend in one upload.

        The single (streamed) PUT that an ``open("wb")`` window produces —
        every ``write()`` since acquire spliced into the disk scratch, and
        this drains it. The scratch streams off disk (bounded memory) on
        backends that support it; others read it back for the SDK's
        whole-object upload. A no-op when nothing was buffered.
        ``with path.open("wb"): pass`` still materialises an empty object
        (the acquire-time ``truncate(0)`` seeded an empty scratch).
        """
        scratch = self._scratch
        if scratch is not None:
            self._scratch = None
            try:
                size = int(scratch.size)
                if self.SUPPORTS_STREAMING_UPLOAD:
                    self._upload_stream(scratch)
                else:
                    self._upload(scratch.read_bytes())
                self._touch_stat(size=size)
            finally:
                try:
                    scratch.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass
        super().flush()

    def _release(self) -> None:
        """Flush the acquired write scratch before the standard release.

        Called from :meth:`close` (explicit) and :meth:`__del__` (GC). A
        failed flush at GC time is logged and swallowed — never raise from
        ``__del__`` per :class:`Disposable`.
        """
        try:
            self.flush()
        except Exception as exc:
            logger.warning(
                "Buffered flush failed during release of %r: %s", self, exc,
            )
        super()._release()
