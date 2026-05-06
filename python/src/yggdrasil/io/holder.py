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
import time
from abc import abstractmethod
from typing import Union, Any, ClassVar, IO

from yggdrasil.disposable import Disposable

from .io_stats import IOStats, IOKind
from .url import URL

__all__ = ["Holder"]


PathLike = Union[str, "os.PathLike[str]", pathlib.PurePath]


_COPY_CHUNK = 1024 * 1024
_HOLDER_SCHEMES: dict[str, type[Holder]] = {}


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
        existing = _HOLDER_SCHEMES.get(scheme)
        if existing is None:
            raise ValueError(f"Unknown scheme '{scheme}'")
        return existing

    if path is not None:
        # Resolve the path's URL scheme via the registry (file:// →
        # LocalPath, s3:// → S3Path, …). The abstract :class:`Path`
        # itself isn't instantiable, so a missing scheme falls back to
        # LocalPath — that's the only path-shaped backend that's
        # always available.
        from .path.local_path import LocalPath
        url_obj = URL.from_(path)
        scheme_from_path = url_obj.scheme
        if scheme_from_path:
            existing = _HOLDER_SCHEMES.get(scheme_from_path)
            if existing is not None:
                return existing
        return LocalPath

    if isinstance(data, Holder):
        return type(data)

    # binary, str, pathlib.Path, None, bytes-like — all default to memory
    from .memory import Memory
    return Memory


class Holder(Disposable):
    """Position-addressable byte holder + :class:`Disposable` lifecycle.

    A holder IS a Disposable: it can be opened, closed, used in a
    ``with`` block, marked dirty / clean. Concrete subclasses
    (:class:`yggdrasil.io.memory.Memory`,
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

    scheme: ClassVar[str] = ""

    __slots__ = (
        "_url",
        "_cached_stat",
        "temporary",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        scheme = cls.scheme

        if scheme:
            existing = _HOLDER_SCHEMES.get(scheme)
            if existing is not None and existing is not cls:
                raise RuntimeError(
                    f"Duplicate scheme '{scheme}' for {cls.__name__} "
                    f"(already registered to {existing.__name__})"
                )
            _HOLDER_SCHEMES[scheme] = cls

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
        if cls is Holder:
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

        return super().__new__(cls)

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
        self._cached_stat: IOStats = IOStats() if stat is None else stat
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
        pos = 0
        while True:
            chunk = data.read(_COPY_CHUNK)
            if not chunk:
                break
            self.write_bytes(chunk, pos=pos)
            pos += len(chunk)

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
        return cls(url=url, **kwargs)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> Holder:
        """Create a new holder from bytes."""
        return cls(binary=data, **kwargs)

    # ------------------------------------------------------------------
    # Abstract primitives
    # ------------------------------------------------------------------

    def read_mv(self, n: int, pos: int) -> memoryview:
        size = self.size
        pos = _resolve_pos(pos, size)
        if pos < 0 or pos > size:
            raise ValueError(
                f"Position {pos} is out of bounds for "
                f"{type(self).__name__} of size {size}"
            )
        if n < 0:
            n = size - pos
        if n < 0 or pos + n > size:
            raise ValueError(
                f"Range [{pos}, {pos + n}) is out of bounds for "
                f"{type(self).__name__} of size {size}"
            )

        return self._read_mv(n, pos)

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

    def write_mv(self, data: memoryview, pos: int) -> int:
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
        """
        size = self.size
        pos = _resolve_pos(pos, size)
        if pos < 0:
            raise ValueError(
                f"Position {pos} is out of bounds for "
                f"{type(self).__name__} of size {size}"
            )

        n = len(data)
        if n == 0:
            return 0

        # Pre-grow the visible size so _write_mv just lays bytes down
        # at a known-valid range. resize() is a no-op when pos+n <= size
        # (in-place overwrite case), so the fast path stays fast.
        end = pos + n
        if end > size:
            self.resize(end)

        written = self._write_mv(data, pos)

        if written > 0:
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

    # ------------------------------------------------------------------
    # IOStats — the canonical metadata holder
    # ------------------------------------------------------------------
    #
    # Every concrete :class:`Holder` keeps a single mutable
    # :class:`IOStats` instance. Writes mutate it in place
    # (``stats.size = new_size``, ``stats.mtime = time.time()``);
    # readers either pin :meth:`stat` and observe the live values, or
    # use the convenience properties below which read straight off the
    # same object.

    def stat(self) -> IOStats:
        """The mutable :class:`IOStats` carrying this holder's metadata.

        Lazy: first call materializes an :class:`IOStats` seeded from
        the URL's media-type. Subsequent calls return the same
        instance, so callers can pin it to observe live size / mtime
        / media_type updates as writes land.
        """
        if self._cached_stat is None:
            kind = IOKind.MEMORY if self.is_memory else IOKind.MISSING
            self._cached_stat = IOStats(
                kind=kind,
                media_type=self.url.infer_media_type(default=None),
            )
        return self._cached_stat

    def _touch_stat(
        self,
        *,
        size: int | None = None,
        mtime: float | None = None,
        media_type: Any = None,
    ) -> None:
        """Update the cached :class:`IOStats` after a successful write.

        Mutates the existing instance in place if one is materialized;
        otherwise lazily creates one via :meth:`stat`. Centralized so
        :meth:`write_mv` (and any subclass with a cheaper write path
        that bypasses :meth:`write_mv`) can keep size/mtime fresh
        without duplicating the bookkeeping.
        """
        s = self.stat()
        if size is not None:
            s.size = size
        s.mtime = mtime if mtime is not None else time.time()
        if media_type is not None:
            s.media_type = media_type

    @property
    def mtime(self) -> float:
        """Convenience accessor — same as ``self.stat().mtime``."""
        return self.stat().mtime

    @property
    def media_type(self):
        """Convenience accessor — same as ``self.stat().media_type``."""
        return self.stat().media_type

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

    def open(self, mode: str = "rb+") -> "BytesIO":
        """Acquire the holder and return a :class:`BytesIO` cursor.

        The default way to interact with a holder's bytes — and,
        because :class:`BytesIO` IS-A :class:`Tabular`, the default
        way to read it as Arrow record batches too. Pattern::

            with LocalPath("/tmp/x.bin").open("wb") as bio:
                bio.write(b"hello")
            # path released here.

            with LocalPath("data.parquet").open() as bio:
                table = bio.read_arrow_table()  # Tabular surface
            # path released here.

        The returned cursor owns the close — when it closes, the
        holder closes too. For multi-cursor / non-owning use,
        construct :class:`BytesIO` directly with ``holder=`` and
        leave ``owns_holder=False`` (the default)::

            mem = Memory(b"shared bytes").acquire()
            try:
                c1 = BytesIO(holder=mem)  # borrow, own cursor
                c2 = BytesIO(holder=mem)  # borrow, own cursor
                ...
            finally:
                mem.close()
        """
        from yggdrasil.io.bytes_io import BytesIO
        self.acquire()
        return BytesIO(
            holder=self, owns_holder=True, mode=mode, auto_open=True,
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
    ) -> int:
        """Positionally write. Returns bytes actually written."""
        return self.write_mv(_as_byte_mv(data), pos)

    def memoryview(self) -> memoryview:
        """View over the holder's visible bytes."""
        return self.read_mv(-1, 0)

    # ------------------------------------------------------------------
    # Bytes / text convenience surface
    # ------------------------------------------------------------------

    def read_bytes(self, n: int = -1, pos: int = 0) -> bytes:
        """Read ``n`` bytes at ``pos`` as :class:`bytes`."""
        return bytes(self.read_mv(n, pos))

    def write_bytes(
        self,
        data: Union[bytes, bytearray, memoryview, str],
        pos: int = 0,
    ) -> int:
        """Splice bytes-like ``data`` at ``pos``. Returns bytes written."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return self.write_mv(_as_byte_mv(data), pos)

    def read_text(
        self,
        encoding: str = "utf-8",
        errors: str = "strict",
        *,
        n: int = -1,
        pos: int = 0,
    ) -> str:
        """Decode ``n`` bytes at ``pos`` as text."""
        return self.read_bytes(n, pos).decode(encoding, errors=errors)

    def write_text(
        self,
        text: str,
        encoding: str = "utf-8",
        errors: str = "strict",
        *,
        pos: int = 0,
    ) -> int:
        """Encode ``text`` and splice at ``pos``. Returns bytes written."""
        return self.write_bytes(
            text.encode(encoding, errors=errors), pos,
        )

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

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __bytes__(self) -> bytes:
        return self.read_bytes()


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