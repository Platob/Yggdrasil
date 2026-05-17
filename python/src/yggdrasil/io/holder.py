"""Abstract byte holder — the substrate :class:`BytesIO` plays on top of.

A :class:`Holder` is "a thing that holds N bytes addressable by
position." Two concrete shapes:

- :class:`yggdrasil.io.memory.Memory` — a :class:`bytearray` we
  manage directly. Every read/write hits memory; ``reserve`` grows
  the bytearray; ``truncate`` resizes the visible slice.
- :class:`yggdrasil.io.fs.LocalPath` /
  :class:`yggdrasil.io.fs.RemotePath` — path-bound holders. Local
  paths back the storage with a long-lived :func:`os.open` fd;
  remote paths with a transaction buffer flushed on commit.

The five abstract primitives are :meth:`_read_mv`, :meth:`_write_mv`,
:meth:`reserve`, :meth:`truncate`, :meth:`clear` and the :attr:`size`
property; :meth:`resize` is concrete and built on :meth:`truncate`.
Everything else (:meth:`pread` / :meth:`pwrite` / :meth:`read_bytes`
/ :meth:`write_bytes` / :meth:`read_text` / :meth:`write_text` /
:meth:`write_local_path`) builds on those, so a new backend gets the
full convenience surface for free.

The default way to interact with a holder's bytes is via
:meth:`open`, which returns a :class:`yggdrasil.io.buffer.bytes_io.BytesIO`
— a cursor + ``IO[bytes]`` view that is also a
:class:`yggdrasil.tabular.Tabular`, so reading the holder as Arrow
record batches is the same call::

    with LocalPath("data.parquet").open() as bio:
        table = bio.read_arrow_table()

For lifecycle without the BytesIO wrapper, use :meth:`acquire` /
:meth:`close`. Multiple :class:`BytesIO` instances can borrow one
holder, each with its own cursor; see :meth:`open` for patterns.
"""

from __future__ import annotations

import os
import pathlib
import struct
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Union, Any, ClassVar, IO, Iterable, Iterator

import pyarrow as pa

from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.disposable import Disposable
from yggdrasil.io.tabular.base import O, Tabular

from .io_stats import IOStats, IOKind
from .url import URL, URLBased

if TYPE_CHECKING:
    from yggdrasil.io.bytes_io import BytesIO

__all__ = ["Holder"]


PathLike = Union[str, "os.PathLike[str]", pathlib.PurePath]


_COPY_CHUNK = 1024 * 1024


def _resolve_pos(pos: int, size: int) -> int:
    """Normalize a position argument with append-at-end semantics.

    - ``pos == -1`` is the explicit "at end of stream" sentinel and
      resolves to ``size`` (POSIX ``SEEK_END`` with offset 0). Reads
      from this position yield zero bytes; writes append.
    - Other negative values count from the end: ``-2`` → ``size - 2``,
      ``-3`` → ``size - 3``, etc. Note the one-step discontinuity at
      ``-1``: this is intentional, so callers have a stable append
      sentinel without giving up from-end indexing.
    - Non-negative values pass through unchanged.

    The result is **not** range-checked; callers do their own bounds
    checks against the operation they're about to perform.
    """
    if pos == -1:
        return size
    if pos < 0:
        return size + pos
    return pos


def _resolve_subclass(
    *,
    scheme: str | None = None,
    url: URL | None = None,
    binary: bytes | bytearray | memoryview | None = None,
    path: PathLike | None = None,
    data: Any = None,
) -> type[Holder]:
    """Pick the concrete :class:`Holder` subclass for the given inputs.

    Pure routing — no instance allocation. Lives outside :meth:`__new__`
    so the dispatch is testable in isolation and so :meth:`__new__` can
    short-circuit ``cls is Holder`` without nesting.
    """
    if url is not None:
        url_obj = URL.from_(url)
        scheme = url_obj.scheme or scheme

    if scheme:
        try:
            return URLBased.for_scheme(scheme)
        except (ValueError, ImportError) as exc:
            raise ValueError(f"Unknown scheme '{scheme}'") from exc

    if path is not None:
        # Resolve the path's URL scheme via the URLBased registry
        # (file:// → LocalPath, s3:// → S3Path, …). The abstract
        # :class:`Path` itself isn't instantiable, so a missing scheme
        # falls back to LocalPath — that's the only path-shaped backend
        # that's always available.
        from .path.local_path import LocalPath
        url_obj = URL.from_(path)
        scheme_from_path = url_obj.scheme
        if scheme_from_path:
            try:
                return URLBased.for_scheme(scheme_from_path)
            except (ValueError, ImportError):
                pass
        return LocalPath

    if isinstance(data, Holder):
        return type(data)

    # binary, str, pathlib.Path, None, bytes-like — all default to memory
    from .memory import Memory
    return Memory


class Holder(Singleton, URLBased, Tabular[O], Disposable):
    """Position-addressable byte holder + :class:`Disposable` lifecycle
    + :class:`Tabular` view of its bytes.

    A holder IS a Disposable: it can be opened, closed, used in a
    ``with`` block, marked dirty / clean. It is also a :class:`Tabular`
    — the default :meth:`_read_arrow_batches` / :meth:`_write_arrow_batches`
    contextually open the holder (``with self.open() as bio:``) and
    delegate to whichever format-specific :class:`BytesIO` leaf the
    holder's :class:`MediaType` resolves to. That means
    ``LocalPath("data.xlsx").read_pandas_frame()`` works the same way
    ``LocalPath("data.xlsx").open()`` does — the open / dispatch /
    close cycle is hidden behind the Tabular surface.

    Concrete subclasses (:class:`yggdrasil.io.memory.Memory`,
    :class:`yggdrasil.io.fs.Path`) plug acquire/release into the
    Disposable hooks so :class:`BytesIO` can compose with either
    one through the same API and seamlessly swap (e.g. on spill)
    without branching at every call site.

    Subclasses implement five primitives:

    - :meth:`_read_mv(n, pos)` — slice ``n`` bytes from ``pos`` as a
      :class:`memoryview`. Receives normalized ``(n, pos)``.
    - :meth:`_write_mv(data, pos)` — splice ``data`` at ``pos``,
      growing the holder if needed. Returns bytes written.
    - :meth:`reserve(n)` — pre-grow the underlying capacity to *at
      least* ``n`` bytes without changing the visible :attr:`size`.
    - :meth:`truncate(n)` — set the visible :attr:`size` to ``n``.
      Shrinks drop the tail; extends zero-pad.
    - :meth:`clear` — drop the payload entirely.

    Plus the :attr:`size` property and :meth:`resize` (concrete,
    built on :meth:`truncate`).
    """

    #: Inherited from :class:`URLBased`. ``None`` on the abstract base
    #: — concrete subclasses override with ``Scheme.X`` and let
    #: :meth:`URLBased.__init_subclass__` register them in the
    #: cross-cutting :data:`_URL_BASED_REGISTRY`.

    __slots__ = (
        "_url",
        "_size",
        "_mtime",
        "_media_type",
        "temporary",
        # Cached payload digest. ``_xxh3_64_size`` / ``_xxh3_64_mtime``
        # form the invalidation key — bumped together by
        # :meth:`_touch_stat`, which every write path eventually flows
        # through. ``-1`` means "never computed".
        "_xxh3_64_cached",
        "_xxh3_64_size",
        "_xxh3_64_mtime",
    )

    # ------------------------------------------------------------------
    # URLBased — round-trip through a :class:`URL`
    # ------------------------------------------------------------------

    def to_url(self) -> "URL":
        """The canonical :class:`URL` that addresses this holder."""
        return self.url

    def __repr__(self) -> str:
        opened = "open" if self.opened else "closed"
        return f"<{type(self).__name__} {self.url!r} [{opened}] {self.stat()!r}>"

    def __hash__(self) -> int:
        # Content-based hash. Mutates with the payload — caller's
        # problem if you stick a Memory in a dict and then write to it.
        return self.url.__hash__()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Holder):
            return (
                self.stat().size == other.stat().size
                and self.memoryview() == other.memoryview()
            )
        if isinstance(other, (bytes, bytearray, memoryview)):
            return self.memoryview() == memoryview(other)
        return False

    def __new__(
        cls,
        data: Any = None,
        *,
        stat: IOStats | None = None,
        scheme: str | None = None,
        url: URL | None = None,
        binary: bytes | bytearray | memoryview | None = None,
        path: PathLike | None = None,
        **kwargs: Any,
    ):
        """Create a new holder.

        When called on the abstract :class:`Holder` itself, dispatches
        to the concrete subclass implied by the inputs (scheme/url
        registry → ``binary`` → ``path`` → ``data`` type → memory
        default). When called on a concrete subclass directly, allocates
        an instance of that subclass.

        Non-routing kwargs (``stat``, ``temporary``, ``media_type``,
        ``auto_open``, …) ride through ``**kwargs`` so subclass
        ``__new__`` and the eventual ``__init__`` see them.
        """
        if cls.__subclasses__() and not cls.__subclasses__().__contains__(cls):
            target = _resolve_subclass(
                scheme=scheme, url=url, binary=binary, path=path, data=data,
            )
            return target.__new__(
                target,
                data=data,
                stat=stat,
                scheme=scheme,
                url=url,
                binary=binary,
                path=path,
                **kwargs,
            )

        # Forward construction args to :class:`Singleton.__new__` so the
        # default ``_singleton_key`` (or a subclass override) can read
        # ``url`` / ``data`` / ``client`` off them. Concrete leaves that
        # opt out of caching (``_SINGLETON_TTL = ...`` on
        # :class:`Holder` itself) short-circuit before this matters.
        return super().__new__(
            cls,
            data=data,
            stat=stat,
            scheme=scheme,
            url=url,
            binary=binary,
            path=path,
            **kwargs,
        )

    def __init__(
        self,
        data: Any = None,
        *,
        stat: IOStats | None = None,
        url: URL | None = None,
        binary: bytes | bytearray | memoryview | None = None,
        path: PathLike | None = None,
        temporary: bool = False,
        **kwargs,
    ):
        """Initialize the holder.

        Exactly one of ``url`` / ``binary`` / ``path`` / ``data``
        determines the seed; the rest are mutually exclusive.

        ``temporary=True`` marks the holder for self-cleanup on release:
        :meth:`_release` calls :meth:`clear` so the payload is dropped
        when the holder closes. Default ``False`` — clears only happen
        when the caller asks.

        ``stat`` lets callers seed the metadata cache (size / mtime /
        media_type) when they already know it — saves a backend probe
        on the first :meth:`stat` call.
        """
        super().__init__(**kwargs)

        self._url: URL | None = None
        if url is not None:
            self.url = url
        # Holder owns its own size + mtime + media_type. Subclasses
        # update these via :meth:`_touch_stat` (or direct mutation
        # on hot paths); :meth:`_stat` snapshots them into a fresh
        # :class:`IOStats` on demand.
        self._size: int = int(stat.size) if stat is not None else 0
        self._mtime: float = float(stat.mtime) if stat is not None else 0.0
        # Lazy xxh3_64 digest cache — paid on first call, valid until
        # ``_size`` or ``_mtime`` shifts (every write goes through
        # :meth:`_touch_stat`, which updates one or the other).
        self._xxh3_64_cached: int = 0
        self._xxh3_64_size: int = -1
        self._xxh3_64_mtime: float = -1.0
        if stat is not None and stat.media_type is not None:
            self._media_type = stat.media_type
        else:
            # Defer the ``url.infer_media_type`` resolve to the first
            # :attr:`media_type` read. Sibling-construction shapes
            # (Path.parent / Path.joinpath / Path.parents) build a
            # fresh holder per step and don't observe ``media_type``
            # in between — paying for the mime walk on every step was
            # the dominant cost of path traversal. ``...`` is the
            # project-wide "not yet computed" sentinel.
            self._media_type = ...
        self.temporary: bool = bool(temporary)

        # ``url=`` only fixes identity; payload-bearing seeds
        # (binary / path / data) are still routed below. Skip ``data``
        # if it duplicates an explicit binary/path seed — caller already
        # picked their lane.
        for prio in (binary, path, data):
            if prio is not None:
                self._init_from(prio)
                break

    def _init_from(self, data: Any) -> None:
        if isinstance(data, Holder):
            self._init_from_holder(data)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            self._init_from_bytes(data)
        elif isinstance(data, str):
            self._init_from_str(data)
        elif isinstance(data, pathlib.PurePath):
            self._init_from_pathlib(data)
        elif isinstance(data, URL):
            self._init_from_url(data)
        else:
            raise TypeError(
                f"Cannot initialize {type(self).__name__} from "
                f"{type(data).__name__}: {data!r}"
            )

    def _init_from_holder(self, holder: Holder) -> None:
        if not self._url:
            self.url = holder.url

        if not self._url_matches(holder.url):
            self.write_bytes(holder.read_bytes())

    def _init_from_bytes(self, data: bytes | bytearray | memoryview) -> None:
        self.write_bytes(data)

    def _init_from_local_path(self, path: PathLike) -> None:
        url = URL.from_(path)
        if not self._url:
            self.url = url

        # Path-shaped seed on a path-shaped holder is identity only —
        # there's nothing to copy, and the file may not exist yet.
        # The cross-backend case (Memory seeded from a local file)
        # still routes through write_local_path.
        if self._url_matches(url):
            return
        self.write_local_path(path)

    def _init_from_pathlib(self, path: pathlib.PurePath) -> None:
        self._init_from_local_path(os.fspath(path))

    def _init_from_str(self, value: str) -> None:
        if URL.is_urlish(value):
            self._init_from_url(URL.from_(value))
            return

        raise ValueError(
            f"Cannot initialize {type(self).__name__} from string {value!r}: "
            "not a recognized URL"
        )

    def _init_from_url(self, url: URL) -> None:
        if not self._url:
            self.url = url

        if self._url_matches(url):
            return

        self._init_from_local_path(url.__fspath__())

    def _url_matches(self, candidate: URL) -> bool:
        """True when *candidate* points at the same place as :attr:`_url`.

        Compares both the canonical URL and the local fspath so the
        check survives a setter that adds/strips a scheme. Returns
        ``False`` when no URL is bound yet.
        """
        if self._url is None:
            return False
        if self._url == candidate:
            return True
        try:
            return self._url.__fspath__() == candidate.__fspath__()
        except Exception:
            return False

    def _init_from_file_like(self, data: IO[bytes]) -> None:
        offset = 0
        while True:
            chunk = data.read(_COPY_CHUNK)
            if not chunk:
                break
            self.write_bytes(chunk, offset=offset)
            offset += len(chunk)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_(
        cls,
        obj: Any,
        *,
        url: URL | None = None,
        **kwargs,
    ) -> Holder:
        if isinstance(obj, cls):
            return obj

        return cls(data=obj, url=url, **kwargs)

    @classmethod
    def from_url(cls, url: URL, **kwargs) -> Holder:
        """Create a new holder from a URL."""
        return cls(url=URL.from_(url), **kwargs)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> Holder:
        """Create a new holder from bytes."""
        return cls(binary=data, **kwargs)

    # ------------------------------------------------------------------
    # Abstract primitives
    # ------------------------------------------------------------------

    def read_mv(self, size: int = -1, offset: int = 0) -> memoryview:
        total = self.size
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

        return self._read_mv(size, offset)

    @abstractmethod
    def _read_mv(self, n: int, pos: int) -> memoryview:
        """Return a memoryview over ``n`` bytes starting at ``pos``.

        Bounds and negative-index normalization happen in :meth:`read_mv`;
        this hook receives non-negative, in-range ``(n, pos)`` with
        ``0 <= pos <= size`` and ``0 <= n <= size - pos``. The append
        point ``pos == size`` is reachable via ``pos = -1`` and always
        pairs with ``n == 0`` — return an empty view in that case.

        The view's lifetime tracks the underlying storage; subclasses
        MAY return a view that backs onto a transient buffer (e.g. a
        remote download) — in that case the caller must consume / copy
        the view before any other I/O against the holder.
        """

    def write_mv(
        self, data: memoryview, offset: int = 0, *, update_stat: bool = True,
    ) -> int:
        """Splice ``data`` at ``pos``, pre-growing the holder as needed.

        Pipeline:

        1. Normalize ``pos`` (``-1`` → append, ``-N`` → ``size - N``).
        2. Pre-grow visible :attr:`size` to cover the splice via
           :meth:`resize` — one call to the size-management primitive
           instead of nudging size up inside ``_write_mv``.
        3. Hand the normalized ``(data, pos)`` to :meth:`_write_mv`,
           which now only has to put bytes down at a valid range.
        4. Mark dirty + bump cached mtime if anything was written.

        Doing the resize up front means ``_write_mv`` implementations
        across backends don't each reimplement the grow logic. It also
        gives subclasses with a cheap grow path (S3 multipart capacity
        hint, ``ftruncate`` on local fd) a chance to skip the work
        ``_write_mv`` would have done byte-by-byte.

        ``update_stat=False`` skips the post-write
        :meth:`_touch_stat` and :meth:`mark_dirty` calls. Use it for
        bulk loops that want a single stat refresh at the end (one
        :func:`time.time` call instead of one per write); the caller
        is then responsible for calling :meth:`_touch_stat` (or
        re-statting via the path-side ``_stat`` for filesystem
        backends) once the loop finishes.
        """
        total = self.size
        offset = _resolve_pos(offset, total)
        if offset < 0:
            raise ValueError(
                f"Offset {offset} is out of bounds for "
                f"{type(self).__name__} of size {total}"
            )

        n = len(data)
        if n == 0:
            return 0

        # Pre-grow the visible size so _write_mv just lays bytes down
        # at a known-valid range. resize() is a no-op when offset+n
        # <= size (in-place overwrite case), so the fast path stays fast.
        end = offset + n
        if end > total:
            self.resize(end)

        written = self._write_mv(data, offset)

        if written > 0 and update_stat:
            self._touch_stat(size=max(end, self.size))
            self.mark_dirty()
        return written

    @abstractmethod
    def _write_mv(self, data: memoryview, pos: int) -> int:
        """Splice ``data`` at ``pos``. Returns bytes actually written.

        Receives a normalized non-negative ``pos`` and a holder that's
        already been grown (via :meth:`resize`) to cover ``pos +
        len(data)``. Subclasses just put bytes down — no size
        management, no negative-index normalization. Dirty marking and
        stat-cache updates happen in :meth:`write_mv`.
        """

    @abstractmethod
    def reserve(self, n: int) -> None:
        """Pre-grow capacity to *at least* ``n`` bytes.

        Capacity-only — does NOT change :attr:`size`. Idempotent
        when capacity ≥ ``n`` already. Subclasses with no growable
        capacity layer may treat this as a no-op.
        """

    def resize(self, n: int) -> int:
        """Grow visible :attr:`size` to at least ``n`` bytes (one-way).

        Sister of :meth:`truncate`, but never shrinks. Used by
        :meth:`write_mv` to pre-allocate a known target before the
        splice so :meth:`_write_mv` doesn't have to manage size.

        - ``n <= size`` → no-op, returns current :attr:`size`.
        - ``n  > size`` → extends with zero-padding via
          :meth:`truncate`, returns ``n``.

        Subclasses with a native grow-only primitive (capacity hint to
        a remote upload session, ``posix_fallocate`` on local fd)
        override for the cheaper path; the default works on every
        backend.
        """
        if n < 0:
            raise ValueError(f"resize size must be >= 0, got {n!r}")
        current = self.size
        if n <= current:
            return current
        return self.truncate(n)

    @abstractmethod
    def truncate(self, n: int) -> int:
        """Set the visible :attr:`size` to exactly ``n`` bytes.

        Shrinks drop the tail; extends zero-pad. Returns ``n``.
        """

    def clear(self) -> None:
        """Drop the holder's payload entirely.

        :class:`Memory` resets the underlying ``bytearray`` to zero
        bytes (capacity drops too). :class:`yggdrasil.io.fs.Path`
        unlinks the backing file with ``missing_ok=True`` so the
        operation is idempotent. After :meth:`clear`, :attr:`size`
        reads ``0`` and the holder is still usable — subsequent
        writes grow it from scratch.
        """
        self._clear()

    @abstractmethod
    def _clear(self) -> None:
        """Drop the holder's payload entirely.

        :class:`Memory` resets the underlying ``bytearray`` to zero
        bytes (capacity drops too). :class:`yggdrasil.io.fs.Path`
        unlinks the backing file with ``missing_ok=True`` so the
        operation is idempotent. After :meth:`clear`, :attr:`size`
        reads ``0`` and the holder is still usable — subsequent
        writes grow it from scratch.
        """

    @property
    @abstractmethod
    def size(self) -> int:
        """Current visible size in bytes."""

    @property
    def size_known(self) -> bool:
        """``True`` when reading :attr:`size` won't trigger a backend probe.

        Always true for in-memory holders (size is a slot). Path
        holders override to ``True`` only when their stat cache is
        warm — callers that want to short-circuit on an empty buffer
        (parquet / arrow IPC / CSV readers checking ``size == 0``)
        can guard the check on this predicate so a cold remote path
        doesn't pay a ``HeadObject`` / ``get_status`` / ``get_metadata``
        round trip just to discover the file is non-empty.
        """
        return True

    def is_empty(self):
        return self.size == 0

    # ------------------------------------------------------------------
    # IOStats — built fresh from holder-owned slots
    # ------------------------------------------------------------------
    #
    # The holder itself owns the canonical ``_size`` / ``_mtime`` /
    # ``_media_type`` fields. :meth:`_stat` is the abstract hook
    # subclasses implement to snapshot those (plus any backend-derived
    # fields like ``kind``) into a fresh :class:`IOStats`. Callers that
    # need to mutate metadata go through the typed surfaces
    # (``holder.media_type = ...``, :meth:`_touch_stat`) — mutating the
    # returned ``IOStats`` no longer round-trips, since each call
    # produces a fresh instance.

    def stat(self) -> IOStats:
        """Snapshot the holder's metadata into a fresh :class:`IOStats`.

        Delegates to :meth:`_stat` for the backend-specific fields
        (``kind`` and the live size for path-bound holders); mutating
        the returned instance does NOT round-trip onto the holder.
        Use the holder's own setters / :meth:`_touch_stat` when you
        need to update metadata.
        """
        return self._stat()

    @abstractmethod
    def _stat(self) -> IOStats:
        """Snapshot the holder's metadata into a fresh :class:`IOStats`.

        Subclasses build the :class:`IOStats` from their authoritative
        state — ``self._size`` / ``self._mtime`` for in-memory
        holders, a backend round-trip for path holders. The base
        :meth:`stat` always routes through this hook so callers don't
        need to know which backend they're against.
        """

    def _touch_stat(
        self,
        *,
        size: int | None = None,
        mtime: float | None = None,
        media_type: Any = None,
    ) -> None:
        """Update the holder-owned metadata fields after a successful write.

        Centralized so :meth:`write_mv` (and any subclass with a
        cheaper write path that bypasses :meth:`write_mv`) can keep
        ``size`` / ``media_type`` fresh without duplicating the
        bookkeeping.

        ``mtime`` is **only** updated when the caller passes it
        explicitly. The previous behavior — bumping ``mtime`` to
        ``time.time()`` on every write — added a syscall-equivalent
        clock read to every byte-level call and dominated tight
        write loops; callers that actually want the freshness should
        either pass ``mtime=`` or call :meth:`touch_mtime` once at
        the end of the operation.
        """
        if size is not None:
            self._size = int(size)
        if mtime is not None:
            self._mtime = float(mtime)
        if media_type is not None:
            self._media_type = media_type

    def touch_mtime(self, when: float | None = None) -> None:
        """Stamp the holder's mtime with the current time.

        Bulk-write helper — call once after a write loop instead of
        letting every :meth:`write_mv` call sample the clock. ``when``
        accepts an explicit timestamp (e.g. an upstream "Last-Modified"
        header); ``None`` defaults to :func:`time.time`.
        """
        self._mtime = float(when) if when is not None else time.time()

    @property
    def mtime(self) -> float:
        """Last-modified time stamp."""
        return self._mtime

    @property
    def media_type(self):
        """The holder's :class:`MediaType`, or ``None`` if unset.

        Resolves lazily on first read: a fresh holder bound only by URL
        carries the sentinel ``...`` in :attr:`_media_type` and runs
        :meth:`URL.infer_media_type` here once, caching the result back
        onto the slot. Subsequent reads (and pickling, IOStats
        snapshots, codec dispatch, …) hit the cached value.
        """
        mt = self._media_type
        if mt is ...:
            url = self._url
            try:
                mt = url.infer_media_type(default=None) if url is not None else None
            except Exception:
                mt = None
            self._media_type = mt
        return mt

    @media_type.setter
    def media_type(self, value: Any) -> None:
        """Stamp a :class:`MediaType` onto the holder.

        Accepts anything :meth:`MediaType.from_` can coerce (a
        :class:`MediaType`, a :class:`MimeType`, a string mime form,
        or ``None`` to clear).
        """
        if value is None:
            self._media_type = None
            return
        try:
            from yggdrasil.data.enums.media_type import MediaType
            mt = MediaType.from_(value, default=None)
        except Exception:
            mt = value
        self._media_type = mt

    # ------------------------------------------------------------------
    # Per-open lifecycle — Path overrides; Memory and other always-live
    # holders inherit no-ops so :class:`BytesIO` can call them blind.
    # ------------------------------------------------------------------

    def acquire(self) -> "Holder":
        """Bring the holder's backing into the acquired state.

        Lifecycle primitive — idempotent. Returns ``self``.
        :meth:`__enter__` calls this; so does :meth:`open` before
        constructing its :class:`BytesIO`. Use this anywhere the
        previous ``open()``-as-lifecycle pattern was wanted, since
        :meth:`open` now returns a :class:`BytesIO`.
        """
        if not self._acquired:
            Disposable.open(self)
        return self

    def open(
        self,
        mode: ModeLike = "rb+",
        *,
        media_type: "MediaType | None" = None,
        owns_holder: bool = False,
        auto_open: bool = True,
        **kwargs: Any,
    ) -> "YIO":
        """Acquire the holder and return a generic :class:`IO` cursor.

        Dispatches to the format-specific :class:`IO` leaf via the
        holder's stamped media type (or *media_type* override), so
        ``LocalPath("data.parquet").open()`` lands on
        :class:`ParquetIO`, ``LocalPath("data.csv").open()`` on
        :class:`CsvIO`, and an unknown / no-media holder falls back
        to a plain :class:`IO`.

        Pattern::

            with LocalPath("/tmp/x.bin").open("wb") as bio:
                bio.write(b"hello")
            # path released here.

            with LocalPath("data.parquet").open() as bio:
                table = bio.read_arrow_table()  # Tabular surface
            # path released here.

        The default ``owns_holder=False`` returns a non-owning
        cursor — closing the cursor leaves the holder open, so the
        caller can mint multiple cursors against the same holder.
        Pass ``owns_holder=True`` to transfer close-ownership of the
        holder to the cursor (the cursor's close then also closes
        the holder).
        """
        from .base import IO as _IO

        self.acquire()
        return _IO.from_holder(
            holder=self,
            owns_holder=owns_holder,
            mode=mode,
            auto_open=auto_open,
            media_type=self.media_type if media_type is None else media_type,
            **kwargs,
        )

    def __enter__(self) -> "Holder":
        """``with holder:`` yields the holder, not a cursor.

        Override of :class:`Disposable.__enter__` (which would
        otherwise call :meth:`open` and hand back a
        :class:`BytesIO`). Use ``with holder.open() as bio:`` to get
        a cursor bound to the with-block lifetime.
        """
        self.acquire()
        return self

    # ==================================================================
    # Tabular surface — open contextually, delegate to the dispatched leaf
    # ==================================================================

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Stream batches from a borrowed cursor on the dispatched leaf.

        Routes through :meth:`open` so the same format-leaf dispatch
        (ParquetIO / XlsxIO / CsvIO / …) and ``acquire`` / ``release``
        accounting that drives explicit ``with holder.open() as bio:``
        usage handles the contextual read too. Options are re-homed
        onto the leaf's options class so format-specific knobs (sheet
        name, delimiter, …) survive the hop.
        """
        with self.open(mode="rb") as bio:
            leaf_options = type(bio).check_options(options=options)
            yield from bio._read_arrow_batches(leaf_options)

    def _write_arrow_batches(
        self,
        batches: "Iterable[pa.RecordBatch]",
        options: O,
    ) -> None:
        """Write batches via :meth:`open` on the dispatched leaf.

        Mirrors :meth:`_read_arrow_batches` — one open / dispatch /
        close cycle handles every format leaf, no separate
        ``BytesIO(...)`` allocation path to keep in sync.
        """
        with self.open(mode="wb") as bio:
            leaf_options = type(bio).check_options(options=options)
            bio._write_arrow_batches(batches, leaf_options)

    def flush(self) -> None:
        """Push buffered writes to the durable backing. Default no-op."""
        return self.commit()

    def close(self, force: bool = False) -> None:
        """Release the holder; on :attr:`temporary`, discard pending
        writes instead of committing them.
        """
        super().close(force=force)

    def _release(self) -> None:
        """:class:`Disposable` release hook — drops the payload when
        :attr:`temporary` is set.
        """
        if self.temporary:
            self.clear()
        super()._release()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def url(self) -> "URL":
        """Canonical URL identifying this holder."""
        if self._url is None:
            return URL.from_memory_address(self)
        return self._url

    @url.setter
    def url(self, value: "URL") -> None:
        # Fast path: when ``value`` is already a :class:`URL` with the
        # expected scheme, skip the ``with_scheme`` rebuild — Holder
        # construction off ``Path.parent`` / ``Path.joinpath`` always
        # produces same-schemed URLs and was paying for an unconditional
        # ``_replace`` copy here.
        if isinstance(value, URL) and value.scheme == self.scheme:
            self._url = value
            return
        self._url = URL.from_(value).with_scheme(self.scheme)

    # ------------------------------------------------------------------
    # Backing-shape predicates
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def is_memory(self) -> bool:
        """True when the holder lives entirely in process memory."""

    @property
    @abstractmethod
    def is_local_path(self) -> bool:
        """True when the holder is a path on the local filesystem."""

    @property
    @abstractmethod
    def is_remote_path(self) -> bool:
        """True when the holder is a path on a non-local backend."""

    @property
    def is_local(self) -> bool:
        return self.is_memory or self.is_local_path

    @property
    def is_remote(self) -> bool:
        return self.is_remote_path

    # ------------------------------------------------------------------
    # Cursorless I/O — the canonical surface :class:`BytesIO` consumes
    # ------------------------------------------------------------------

    def pread(self, n: int, pos: int) -> bytes:
        """Positional read. Returns at most ``n`` bytes at *pos*."""
        return bytes(self.read_mv(n, pos))

    def pwrite(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
        *,
        update_stat: bool = True,
    ) -> int:
        """Positionally write. Returns bytes actually written.

        ``update_stat=False`` defers the post-write stat refresh to
        the caller — see :meth:`write_mv` for the bulk-write rationale.
        """
        return self.write_mv(_as_byte_mv(data), pos, update_stat=update_stat)

    def memoryview(self) -> memoryview:
        """View over the holder's visible bytes."""
        return self.read_mv(-1, 0)

    # ------------------------------------------------------------------
    # Bytes / text convenience surface
    # ------------------------------------------------------------------

    def read_bytes(self, size: int = -1, offset: int = 0) -> bytes:
        """Read ``size`` bytes starting at ``offset`` as :class:`bytes`.

        ``size=-1`` reads to EOF; ``offset`` accepts negative
        indices via :func:`_resolve_pos` (``-1`` → ``size``,
        ``-N`` → ``self.size - N``).
        """
        return bytes(self.read_mv(size, offset))

    def write_bytes(
        self,
        data: Any,
        offset: int = 0,
    ) -> int:
        """Splice ``data`` at ``offset``. Returns bytes written.

        Type-directed dispatch — bytes-like payloads
        (:class:`bytes`, :class:`bytearray`, :class:`memoryview`,
        and ``str`` after UTF-8 encoding) splice through
        :meth:`write_mv`; file-like sources (anything exposing
        ``.read``) drain through :meth:`write_stream` so backends
        with an atomic whole-object uploader can push a single
        request instead of buffering the payload. Subclasses
        override :meth:`_write_mv` and/or :meth:`write_stream`
        rather than this dispatch.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        if hasattr(data, "read"):
            return self.write_stream(data, offset=offset)
        return self.write_mv(_as_byte_mv(data), offset)

    def read_text(
        self,
        encoding: str = "utf-8",
        errors: str = "strict",
        *,
        size: int = -1,
        offset: int = 0,
    ) -> str:
        """Decode ``size`` bytes at ``offset`` as text."""
        return self.read_bytes(size, offset).decode(encoding, errors=errors)

    def write_text(
        self,
        text: str,
        encoding: str = "utf-8",
        errors: str = "strict",
        *,
        offset: int = 0,
    ) -> int:
        """Encode ``text`` and splice at ``offset``. Returns bytes written."""
        return self.write_bytes(
            text.encode(encoding, errors=errors), offset,
        )

    # ------------------------------------------------------------------
    # Cursorless read/write primitives — IO subclasses add cursor
    # ------------------------------------------------------------------

    def readinto(self, buffer: Any, *, offset: int = 0) -> int:
        """Fill *buffer* with bytes starting at ``offset``.

        Returns the number of bytes written into *buffer* —
        ``min(len(buffer), self.size - offset)``. Matches the
        stdlib :meth:`io.RawIOBase.readinto` shape but stays
        cursorless; subclasses with a cursor (``IO``) wrap this
        and advance after the call.
        """
        mv = memoryview(buffer)
        capacity = len(mv)
        if capacity == 0:
            return 0
        chunk = self.read_bytes(capacity, offset)
        n = len(chunk)
        if n:
            mv[:n] = chunk
        return n

    def readline(self, limit: int = -1, *, offset: int = 0) -> bytes:
        """Read up to the next newline starting at ``offset``.

        Returns the line including the trailing ``\\n`` (or short
        when EOF lands first). ``limit >= 0`` caps the byte
        count. Cursorless — :class:`IO` subclasses wrap this and
        advance the cursor by the returned length.
        """
        total = self.size
        if offset >= total:
            return b""
        chunk_len = total - offset
        if limit is not None and limit >= 0:
            chunk_len = min(limit, chunk_len)
        if chunk_len <= 0:
            return b""
        chunk = self.read_bytes(chunk_len, offset)
        nl = chunk.find(b"\n")
        return chunk if nl == -1 else chunk[: nl + 1]

    def readlines(self, hint: int = -1, *, offset: int = 0) -> list[bytes]:
        """Read every line from ``offset`` to EOF (or until ``hint`` bytes)."""
        lines: list[bytes] = []
        cursor = offset
        total = 0
        while True:
            line = self.readline(offset=cursor)
            if not line:
                break
            lines.append(line)
            total += len(line)
            cursor += len(line)
            if hint is not None and hint > 0 and total >= hint:
                break
        return lines

    # ---- structured binary helpers — fixed-width little-endian ---------

    def _read_struct(self, fmt: str, n: int, offset: int) -> Any:
        return struct.unpack(fmt, self.read_bytes(n, offset))[0]

    def _write_struct(self, fmt: str, value: Any, offset: int) -> int:
        return self.write_bytes(struct.pack(fmt, value), offset)

    def read_int8(self, *, offset: int = 0) -> int: return self._read_struct("<b", 1, offset)
    def write_int8(self, v: int, *, offset: int = 0) -> int: return self._write_struct("<b", int(v), offset)
    def read_uint8(self, *, offset: int = 0) -> int: return self._read_struct("<B", 1, offset)
    def write_uint8(self, v: int, *, offset: int = 0) -> int: return self._write_struct("<B", int(v), offset)
    def read_int16(self, *, offset: int = 0) -> int: return self._read_struct("<h", 2, offset)
    def write_int16(self, v: int, *, offset: int = 0) -> int: return self._write_struct("<h", int(v), offset)
    def read_uint16(self, *, offset: int = 0) -> int: return self._read_struct("<H", 2, offset)
    def write_uint16(self, v: int, *, offset: int = 0) -> int: return self._write_struct("<H", int(v), offset)
    def read_int32(self, *, offset: int = 0) -> int: return self._read_struct("<i", 4, offset)
    def write_int32(self, v: int, *, offset: int = 0) -> int: return self._write_struct("<i", int(v), offset)
    def read_uint32(self, *, offset: int = 0) -> int: return self._read_struct("<I", 4, offset)
    def write_uint32(self, v: int, *, offset: int = 0) -> int: return self._write_struct("<I", int(v), offset)
    def read_int64(self, *, offset: int = 0) -> int: return self._read_struct("<q", 8, offset)
    def write_int64(self, v: int, *, offset: int = 0) -> int: return self._write_struct("<q", int(v), offset)
    def read_uint64(self, *, offset: int = 0) -> int: return self._read_struct("<Q", 8, offset)
    def write_uint64(self, v: int, *, offset: int = 0) -> int: return self._write_struct("<Q", int(v), offset)
    def read_f32(self, *, offset: int = 0) -> float: return self._read_struct("<f", 4, offset)
    def write_f32(self, v: float, *, offset: int = 0) -> int: return self._write_struct("<f", float(v), offset)
    def read_f64(self, *, offset: int = 0) -> float: return self._read_struct("<d", 8, offset)
    def write_f64(self, v: float, *, offset: int = 0) -> int: return self._write_struct("<d", float(v), offset)
    def read_bool(self, *, offset: int = 0) -> bool: return bool(self.read_uint8(offset=offset))
    def write_bool(self, v: bool, *, offset: int = 0) -> int: return self.write_uint8(1 if v else 0, offset=offset)

    # ------------------------------------------------------------------
    # Local-path bridge
    # ------------------------------------------------------------------

    def write_local_path(
        self,
        path: PathLike,
        *,
        pos: int = 0,
        n: int = -1,
        chunk_size: int = _COPY_CHUNK,
    ) -> int:
        """Load ``path``'s bytes into this holder at ``pos``.

        ``n < 0`` reads the whole file; ``n >= 0`` caps the source
        bytes pulled at *n*. Streams in ``chunk_size`` slices so a
        large file doesn't materialize into memory.

        Pre-allocates the holder via :meth:`resize` when the source
        size is known up front (``n >= 0`` or local stat available),
        so the inner loop only writes — no per-chunk grow.
        """
        if pos < 0:
            raise ValueError("write_local_path pos must be >= 0")
        os_path = os.fspath(path)

        # Pre-grow the holder when we know the target end position.
        # n < 0 → fall back to source stat; failure is non-fatal (the
        # write loop still grows incrementally via write_mv → resize).
        target_end: int | None = None
        if n >= 0:
            target_end = pos + n
        else:
            try:
                target_end = pos + os.path.getsize(os_path)
            except OSError:
                pass
        if target_end is not None and target_end > self.size:
            self.resize(target_end)

        total = 0
        cursor = pos
        remaining = n if n >= 0 else None
        with open(os_path, "rb") as fh:
            while True:
                want = chunk_size
                if remaining is not None:
                    if remaining <= 0:
                        break
                    want = min(want, remaining)
                chunk = fh.read(want)
                if not chunk:
                    break
                written = self.write_mv(memoryview(chunk), cursor)
                if written == 0:
                    break
                cursor += written
                total += written
                if remaining is not None:
                    remaining -= written
        return total

    def write_stream(self, src: Any, *, offset: int = 0) -> int:
        """Drain a binary source into this holder at ``offset``.

        Public entry point: accepts a yggdrasil :class:`IO[bytes]`,
        a stdlib :class:`typing.BinaryIO` (``io.BytesIO``,
        ``open(..., "rb")``, urllib3 responses, …), or any file-like
        carrying a ``.read``. Non-:class:`IO` sources are coerced
        via :meth:`IO.from_` so subclass-side :meth:`_write_stream`
        always receives a real :class:`IO[bytes]`.
        """
        if offset < 0:
            raise ValueError("write_stream offset must be >= 0")
        from yggdrasil.io.base import IO as _IO
        from yggdrasil.io.bytes_io import BytesIO as _YggBytesIO

        io_src = src if isinstance(src, _IO) else _YggBytesIO.from_(src)
        return self._write_stream(io_src, offset=offset)

    def _write_stream(self, src: "IO[bytes]", *, offset: int) -> int:
        """Splice ``src``'s bytes into this holder starting at ``offset``.

        Default implementation: real chunked streaming — read
        :data:`_COPY_CHUNK` bytes at a time and splice each chunk
        through :meth:`write_bytes`, so multi-GB sources never
        materialise as a single :class:`bytes` object in Python.

        Subclass override hook: backends with an atomic
        whole-object upload (Volumes ``files.upload``, Workspace
        ``workspace.upload``, S3 ``PutObject``) replace this with
        a single request that consumes *src* directly — the
        chunked default is a strict loss for those.

        *src* is always a real :class:`IO[bytes]`; the public
        :meth:`write_stream` does the coercion so subclass code
        gets a stable type.
        """
        cursor = offset
        total = 0
        while True:
            chunk = src.read(_COPY_CHUNK)
            if not chunk:
                break
            n = self.write_bytes(chunk, offset=cursor)
            cursor += n
            total += n
        return total

    # ------------------------------------------------------------------
    # Byte transfer — upload / download to any byte sink
    # ------------------------------------------------------------------

    def upload(
        self, src: Any, *, size: int = -1, offset: int = 0,
    ) -> "Holder":
        """Upload *src*'s bytes into this holder.

        Symmetric to :meth:`download` but indexed from the
        destination side — ``dst.upload(src)`` makes the
        destination's content equal to the source's.

        *src* accepts any of:

        - :class:`Holder` (incl. any :class:`Path` subclass) —
          its bytes are pulled starting at *offset*.
        - :class:`IO` cursor — *offset* (if non-zero) seeks
          before ``read()``; otherwise the cursor's current
          position is honoured.
        - ``str`` / :class:`os.PathLike` — coerced via
          ``Path.from_(src)`` and treated as a holder.

        *size* and *offset* slice the source: ``size=-1`` (default)
        reads to EOF, ``size>=0`` caps the byte count, ``offset``
        is the starting offset. Slicing forces the whole-payload
        fast path in :meth:`_transfer_to` to defer to a bytes
        copy (the backend-specific shortcuts —
        ``shutil.copyfile``, ``write_local_path`` — don't expose
        a window).

        When *self* is a :class:`Path` whose URL ends in a
        trailing ``/`` (directory shape), the source's filename
        (``src.url.name`` or ``"download"`` for nameless holders)
        is joined onto it. No remote ``stat`` is issued — the
        trailing slash is a purely local, ``cp``-style hint.

        Returns the resolved destination so chains like
        ``dst.upload(src).read_bytes()`` work.

        Subclasses with a faster move (e.g. local→local via
        ``sendfile``, local→remote chunked stream) override
        :meth:`_transfer_to`, not this method.
        """
        from yggdrasil.io.base import IO
        from yggdrasil.io.path.path import Path

        source = _coerce_transfer_endpoint(src)
        target = _join_dir_hint(self, source)
        if isinstance(source, Path) and source.is_dir():
            # Directory tree: only a :class:`Path` target can hold
            # it. ``size`` / ``offset`` slicing is a file-only knob.
            if size != -1 or offset != 0:
                raise IsADirectoryError(
                    f"Holder.upload: source {source.full_path()!r} is "
                    f"a directory; size / offset slicing applies to "
                    f"file uploads only."
                )
            if not isinstance(target, Path):
                raise IsADirectoryError(
                    f"Holder.upload: source {source.full_path()!r} is "
                    f"a directory; target must be a Path to hold the "
                    f"tree, got {type(target).__name__}."
                )
            target.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                (target / child.name).upload(child)
            return target
        if not isinstance(source, IO) and size < 0 and offset == 0:
            # Whole-payload :class:`Holder` source → defer to the
            # backend-aware fast path (``shutil.copyfile``,
            # ``write_local_path``, …). Sliced or IO sources fall
            # through to the byte-pump below.
            source._transfer_to(target)
        else:
            target.write_bytes(
                _read_slice_from_source(source, size=size, offset=offset),
            )
        return target

    def download(
        self, to: Any = None, *, size: int = -1, offset: int = 0,
    ) -> "Holder | IO":
        """Copy this holder's bytes to a local target.

        When *to* is :data:`None`, bytes land in the user's
        ``~/Downloads`` folder under :attr:`url.name` (or
        ``"download"`` for nameless holders), with browser-style
        ``(1)`` / ``(2)`` / … suffixes appended on name conflict.
        Otherwise *to* accepts the same shapes as :meth:`upload`
        (:class:`Holder`, :class:`IO`, ``str`` / :class:`os.PathLike`).
        *size* and *offset* slice this holder: ``size=-1`` (default)
        reads to EOF, ``size>=0`` caps the byte count, ``offset``
        is the starting offset. Returns the resolved target.
        """
        from yggdrasil.io.path.path import Path

        if to is None:
            to = _default_download_target(self._transfer_filename())
        target = _join_dir_hint(_coerce_transfer_endpoint(to), self)
        if isinstance(self, Path) and self.is_dir():
            # Symmetric with :meth:`upload` — delegate to the
            # destination-side recurse: ``target.upload(self)``
            # knows how to ``mkdir`` and walk *self*'s children.
            if size != -1 or offset != 0:
                raise IsADirectoryError(
                    f"Holder.download: source {self.full_path()!r} is "
                    f"a directory; size / offset slicing applies to "
                    f"file downloads only."
                )
            if not isinstance(target, Path):
                raise IsADirectoryError(
                    f"Holder.download: source {self.full_path()!r} is "
                    f"a directory; target must be a Path to hold the "
                    f"tree, got {type(target).__name__}."
                )
            return target.upload(self)
        if size < 0 and offset == 0:
            self._transfer_to(target)
        else:
            target.write_bytes(
                _read_slice_from_source(self, size=size, offset=offset),
            )
        return target


    def _transfer_filename(self) -> str:
        """Filename used when joining onto a directory-shaped target.

        :class:`Memory` holders address themselves with auto-minted
        ``mem://<host>/<hex_addr>`` URLs whose ``name`` is the
        object address — useless as a download filename. Fall back
        to ``"download"`` for memory-backed holders and any holder
        whose URL has no nameable segment.
        """
        if self.is_memory:
            return "download"
        return self.url.name or "download"

    def _transfer_to(self, target: "Holder | IO") -> None:
        """Default transfer: pull self's bytes, push into *target*.

        Subclasses override to take advantage of backend-side fast
        paths (e.g. :class:`Path` uses :func:`shutil.copyfile` for
        local-to-local and :meth:`write_local_path` for
        local-to-remote so neither path materialises the full
        payload).
        """
        target.write_bytes(self.read_bytes())

    # ------------------------------------------------------------------
    # Hashing — full-payload digests over the durable bytes.
    # ------------------------------------------------------------------
    #
    # Lives on the holder rather than only on :class:`BytesIO` because
    # callers that only have a holder shouldn't have to open a cursor
    # just to compute a digest — the holder owns the bytes.

    def to_bytes(self) -> bytes:
        """Full payload as :class:`bytes` — alias for ``read_bytes()``."""
        return self.read_bytes()

    def getvalue(self) -> bytes:
        """Stdlib :class:`io.BytesIO` parity — alias for :meth:`to_bytes`."""
        return self.to_bytes()

    def decode(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        """Decode the whole payload as text. Cursorless — does not seek."""
        return self.to_bytes().decode(encoding, errors=errors)

    def to_base64(self, urlsafe: bool = True) -> str:
        """Return the payload base64-encoded as an ASCII ``str``.

        ``urlsafe=True`` (default) uses :func:`base64.urlsafe_b64encode`
        — ``-`` / ``_`` in place of ``+`` / ``/`` so the result drops
        cleanly into a URL or filename. ``urlsafe=False`` falls back
        to the standard alphabet.
        """
        import base64

        b = self.to_bytes()
        if urlsafe:
            return base64.urlsafe_b64encode(b).decode("ascii")
        return base64.b64encode(b).decode("ascii")

    def xxh3_64(self):
        """Return an :class:`xxhash.xxh3_64` instance over the payload.

        Always rebuilds an updatable :class:`xxhash.xxh3_64` so callers
        can keep mixing more bytes in if they want. The expensive
        part — walking the payload — is short-circuited via the
        cached digest; we just seed a fresh hasher with the cached
        value's bytes when available.
        """
        import xxhash
        return xxhash.xxh3_64(self.read_bytes())

    def xxh3_int64(self) -> int:
        """64-bit xxh3 hash of the payload as a signed int64.

        ``xxh3_64`` produces an unsigned 64-bit value; downstream Arrow
        schemas pin the field as ``int64``, so the digest is wrapped
        into signed range ``[-2**63, 2**63)``. Memoized against
        ``(_size, _mtime)`` — which every write path bumps via
        :meth:`_touch_stat` — so repeated reads pay the walk once.
        """
        if (
            self._xxh3_64_size != -1
            and self._xxh3_64_size == self._size
            and self._xxh3_64_mtime == self._mtime
        ):
            return self._xxh3_64_cached
        import xxhash
        v = xxhash.xxh3_64(self.read_bytes()).intdigest()
        if v >= 2 ** 63:
            v -= 2 ** 64
        self._xxh3_64_cached = v
        self._xxh3_64_size = self._size
        self._xxh3_64_mtime = self._mtime
        return v

    @property
    def xxh3_64_digest(self) -> bytes:
        """8-byte big-endian payload digest — equivalent to
        ``xxh3_64().digest()`` but served from the cached
        :meth:`xxh3_int64` so callers mixing the digest into a parent
        hash don't re-walk the payload."""
        v = self.xxh3_int64()
        if v < 0:
            v += 2 ** 64
        return v.to_bytes(8, "big")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __bytes__(self) -> bytes:
        return self.read_bytes()


def _coerce_transfer_endpoint(value: Any) -> "Holder | IO":
    """Coerce a transfer endpoint into a :class:`Holder` or :class:`IO`.

    Used by :meth:`Holder.upload` / :meth:`Holder.download` to
    accept the same four input shapes — :class:`Holder`,
    :class:`IO`, ``str``, :class:`os.PathLike` — regardless of
    whether the value names the source or the target side.
    """
    from yggdrasil.io.base import IO
    from yggdrasil.io.path.path import Path

    if isinstance(value, (Holder, IO)):
        return value
    if isinstance(value, (str, os.PathLike)):
        return Path.from_(value)
    raise TypeError(
        f"Holder.upload/download: expected a Holder, IO, str, or "
        f"os.PathLike endpoint; got {type(value).__name__}: {value!r}"
    )


def _read_slice_from_source(
    source: "Holder | IO", *, size: int, offset: int,
) -> bytes:
    """Pull ``[offset, offset+size)`` bytes from a coerced transfer source.

    Hides the :class:`Holder` vs :class:`IO` split for the
    upload/download read side:

    - :class:`Holder` sources use ``read_bytes(size, offset)``
      — cursorless positional read.
    - :class:`IO` sources :meth:`~IO.seek` to ``offset`` (when
      non-zero) and consume ``size`` bytes from the cursor, so
      the caller-supplied stream position is the natural origin.

    ``size < 0`` means "to EOF" in both cases.
    """
    from yggdrasil.io.base import IO

    if isinstance(source, IO):
        if offset:
            source.seek(offset)
        return source.read() if size < 0 else source.read(size)
    return source.read_bytes(size=size, offset=offset)


def _join_dir_hint(
    dst: "Holder | IO", src: "Holder | IO",
) -> "Holder | IO":
    """Apply ``cp``-style directory hint when *dst* is a slash-terminated Path.

    ``dst_dir_slash.upload(src)`` lands at ``dst_dir/<src.name>``;
    a non-Path *dst* (Memory, IO cursor) or a non-directory path
    is returned untouched. The source's filename is taken from
    :meth:`Holder._transfer_filename` so :class:`Memory` /
    nameless holders fall back to ``"download"``.
    """
    from yggdrasil.io.path.path import Path

    if isinstance(dst, Path) and _looks_like_directory(dst.url):
        return dst / src._transfer_filename()  # type: ignore[union-attr]
    return dst


def _looks_like_directory(url: URL) -> bool:
    """Trailing-slash check: ``True`` iff *url*'s path ends in ``/``.

    Used by the upload/download directory-hint helpers to apply
    ``cp``-style "into this directory" semantics without a remote
    stat round trip. The canonical signal is an empty trailing
    element in :attr:`URL.parts`.
    """
    parts = url.parts
    return bool(parts) and parts[-1] == ""


def _default_download_target(name: str) -> "Holder":
    """Resolve a fresh :class:`LocalPath` under ``~/Downloads`` for *name*.

    Browser-style default: drop the file under the user's
    Downloads folder, and on a name clash append ``(1)``, ``(2)``,
    … before the suffix until a free slot is found. The directory
    is created on demand; the file itself is not.
    """
    from yggdrasil.io.path.local_path import LocalPath

    downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    candidate = os.path.join(downloads_dir, name)
    if not os.path.exists(candidate):
        return LocalPath(candidate)

    stem, suffix = os.path.splitext(name)
    i = 1
    while True:
        candidate = os.path.join(downloads_dir, f"{stem} ({i}){suffix}")
        if not os.path.exists(candidate):
            return LocalPath(candidate)
        i += 1


def _as_byte_mv(data: Union[bytes, bytearray, memoryview]) -> memoryview:
    """Normalize a bytes-like to a 1-D, contiguous, unsigned-byte memoryview.

    Centralizes the pwrite/write_bytes prelude so callers don't repeat
    the cast/contiguity dance and the rules stay in one place.
    """
    mv = memoryview(data)
    if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
        mv = mv.cast("B")
    if not mv.c_contiguous:
        mv = memoryview(bytes(mv))
    return mv