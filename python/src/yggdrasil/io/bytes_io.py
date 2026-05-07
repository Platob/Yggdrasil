"""Cursor + ``IO[bytes]`` + tabular view over a :class:`Holder`.

:class:`BytesIO` IS-A :class:`Tabular`. It carries the cursor that a
holder lacks (every public op is "do this at the cursor"), exposes
the full ``IO[bytes]`` surface, and inherits the row-oriented surface
from :class:`Tabular` so byte-level and Arrow-level operations live
on the same handle.

The default :class:`BytesIO` doesn't know what format its bytes
encode, so the two abstract :class:`Tabular` hooks
(:meth:`_read_arrow_batches` / :meth:`_write_arrow_batches`) raise
unless overridden. Format-specific leaves (Parquet, CSV, Arrow IPC,
…) subclass :class:`BytesIO` and override the hooks against the
same byte holder.

Construction shapes
-------------------

- **Borrow** — ``BytesIO(holder=h)``. Closing the BytesIO never
  touches *h*. Two cursors over a long-lived buffer:
  ``c1, c2 = BytesIO(holder=mem), BytesIO(holder=mem)``.
- **Own** — ``BytesIO(holder=h, owns_holder=True)``. Closing the
  BytesIO closes *h*. This is what :meth:`Holder.open` returns.
- **Auto** — ``BytesIO(data)`` or ``BytesIO.from_(obj)`` routes any
  bytes-like / path-like / file-like / holder input through the
  right :class:`Holder` constructor and returns an owning BytesIO.

Lifecycle — closed = direct, open = buffered transaction
--------------------------------------------------------

A :class:`BytesIO` has two operating modes, switched by its
:class:`Disposable` state:

- **Closed** — every read / write goes **straight to the durable
  holder**. No buffering. Useful when you want each op to commit
  immediately (logging, append-only sinks, throwaway one-shots).
  The durable holder must already be acquired by someone — typically
  the caller.

- **Open** — a :class:`Memory` scratch buffer sits between the cursor
  and the holder. :meth:`_acquire` seeds scratch from the durable
  bytes (or empty for ``"wb"``-class modes). All ops route through
  scratch; the durable holder is **untouched** until :meth:`flush`
  or :meth:`_release` commits. This is the standard pattern: edit
  in-memory, save on close::

      with LocalPath("config.json").open("rb+") as bio:
          bio.write(b"...changes...")
      # Now the file gets the new bytes — atomic at commit time.

The dirty bit drives commit decisions: writes through the open
BytesIO call :meth:`mark_dirty`; :meth:`flush` and :meth:`_release`
check :attr:`_dirty` before pushing scratch onto the durable holder.
Setting ``temporary=True`` on the holder discards the scratch on
close instead of committing.

Modes
-----

Mode strings follow stdlib :func:`open`: ``"rb"`` / ``"wb"`` /
``"ab"`` / ``"xb"`` plus ``+`` variants. Effects on top of a holder:

- Initial cursor — ``"ab"`` lands at EOF; everything else at 0.
- ``readable`` / ``writable`` reporting.
- ``"wb"`` / ``"w+b"`` truncates the holder on :meth:`_acquire`.
- ``"xb"`` raises :class:`FileExistsError` on :meth:`_acquire` when
  the holder is non-empty.

The mode never touches the holder itself — a :class:`Memory` doesn't
know or care what mode the BytesIO over it is in.
"""

from __future__ import annotations

import base64
import io
import struct
from collections.abc import Iterable
from typing import IO, Any, Iterator, Optional, TypeVar, Union

import pyarrow as pa
from yggdrasil.data.options import CastOptions
from yggdrasil.disposable import Disposable
from yggdrasil.io.holder import Holder
from yggdrasil.io.memory import Memory
from yggdrasil.io.tabular import Tabular

__all__ = ["BytesIO"]


BytesLike = Union[bytes, bytearray, memoryview]


O = TypeVar("O", bound=CastOptions)


def _as_byte_mv(data: BytesLike) -> memoryview:
    """Normalize bytes-like input to a 1-D unsigned-byte memoryview.

    pyarrow ``Buffer`` arrives as ``format='b'`` with itemsize 1,
    which trips bytearray slice assignment unless we cast. Centralize
    the cast/contiguity dance so every write path stays consistent.
    """
    mv = memoryview(data)
    if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
        mv = mv.cast("B")
    if not mv.c_contiguous:
        mv = memoryview(bytes(mv))
    return mv


# ===========================================================================
# BytesIO
# ===========================================================================


class BytesIO(Tabular[O], Disposable, IO[bytes]):
    """Cursor + ``IO[bytes]`` + tabular view over a :class:`Holder`.

    Two operating modes:

    - **Closed** — ops route directly to the durable holder. Each
      write commits synchronously; useful for append-only sinks or
      one-shot reads where buffering would be overhead.
    - **Open** — :meth:`_acquire` builds a :class:`Memory` scratch
      seeded from the durable holder; ops route through scratch,
      the durable holder is committed-to on :meth:`flush` /
      :meth:`_release`. The standard "edit in memory, save on
      close" transaction.

    Mode-aware but format-agnostic at this layer. Mode controls
    cursor position (``"ab"`` → EOF), ``readable`` / ``writable``
    reporting, whether :meth:`_acquire` starts scratch empty
    (``"wb"``) or seeds from durable bytes, and the
    :class:`FileExistsError` for ``"xb"``.

    The two :class:`Tabular` batch hooks default to
    :class:`NotImplementedError` — a plain :class:`BytesIO` doesn't
    know what its bytes encode. Subclasses (ParquetIO, CsvIO,
    ArrowIPCIO, …) override the hooks to do format-specific decoding
    against the same holder.
    """

    __slots__ = (
        "_holder",
        "_owns_holder",
        "_pos",
        "_mode",
        "_scratch",
    )

    def __new__(
        cls,
        data: Any = None,
        *,
        holder: "Holder | None" = None,
        owns_holder: bool = False,
        mode: str = "rb+",
        media_type: Any = None,
        **kwargs: Any,
    ):
        """Dispatch to a registered Tabular leaf when *media_type*
        identifies one.

        Lets ``BytesIO(data, media_type=MediaTypes.PARQUET)`` land on
        :class:`ParquetIO` directly, which is what the pickle ser
        layer relies on for round-tripping a typed buffer through
        ``BytesIO(payload, media_type=...)`` — same shape as the
        scheme dispatch on :class:`Holder`. Subclass calls
        (``ParquetIO(...)``) skip the dispatch and stay on the
        concrete class.
        """
        if cls is BytesIO and media_type is not None:
            from yggdrasil.io.tabular.base import Tabular
            from yggdrasil.data.enums.media_type import MediaType
            mt = MediaType.from_(media_type, default=None)
            if mt is not None:
                target = Tabular.class_for_media_type(mt, default=None)
                if target is not None and issubclass(target, BytesIO) and target is not cls:
                    return target.__new__(
                        target,
                        data=data,
                        holder=holder,
                        owns_holder=owns_holder,
                        mode=mode,
                        media_type=media_type,
                        **kwargs,
                    )
        return super().__new__(cls)

    def __init__(
        self,
        data: Any = None,
        *,
        holder: "Holder | None" = None,
        owns_holder: bool = False,
        mode: str = "rb+",
        media_type: Any = None,
        **kwargs: Any,
    ) -> None:
        """Construct a cursor over a :class:`Holder`. Does NOT open.

        Pass exactly one of *holder* or *data*:

        - *holder*: borrow an existing holder. Set ``owns_holder=True``
          to transfer close-ownership to this BytesIO (typically only
          :meth:`Holder.open` does this).
        - *data*: routed through :meth:`from_` to build a fresh
          holder; the new BytesIO owns it.

        ``mode`` follows stdlib :func:`open` semantics. The BytesIO
        is constructed in the **closed** state — ops in that state
        commit synchronously to the durable holder. To enter the
        buffered-transaction mode (scratch buffer + commit-on-close),
        call :meth:`acquire`, use a ``with`` block, or go through
        :meth:`Holder.open`.
        """
        super().__init__(**kwargs)

        if holder is None:
            if data is None:
                # Empty memory holder — equivalent to stdlib io.BytesIO().
                holder = Memory()
                owns_holder = True
            else:
                tmp = self.from_(data, mode=mode)
                holder = tmp._holder
                owns_holder = True
        elif data is not None:
            raise TypeError(
                "BytesIO accepts holder= OR data, not both. "
                "Use BytesIO(data) for the auto-construct shape, or "
                "BytesIO(holder=h) to borrow an existing holder."
            )

        self._holder: "Holder" = holder
        self._owns_holder: bool = bool(owns_holder)
        self._pos: int = 0
        self._mode: str = mode
        # Memory scratch buffer — populated by :meth:`_acquire`,
        # commits to ``self._holder`` on :meth:`flush` /
        # :meth:`_release`. ``None`` when the BytesIO is closed; in
        # that state, ops route directly to ``self._holder``.
        self._scratch: "Holder | None" = None

        # Stamp media type onto the holder's IOStats — gives the
        # codec auto-handling path something to inspect, and makes
        # the buffer self-describing for downstream serializers.
        if media_type is not None:
            try:
                from yggdrasil.data.enums.media_type import MediaType
                mt = MediaType.from_(media_type, default=None)
                if mt is not None:
                    self._holder.stat().media_type = mt
            except Exception:
                pass

    # ==================================================================
    # Construction routing
    # ==================================================================

    @classmethod
    def from_(cls, obj: Any, *, mode: str = "rb+", **kwargs: Any) -> "BytesIO":
        """Auto-route *obj* to the right holder, return an owning BytesIO.

        - :class:`BytesIO` — pass through (idempotent).
        - :class:`Holder` — borrow into a :class:`BytesIO`.
        - bytes-like (``bytes`` / ``bytearray`` / ``memoryview``) —
          wrap in a fresh :class:`Memory`.
        - path-like (``str`` / ``pathlib.Path`` / ``URL``) — wrap in
          a fresh holder via :class:`Holder` registry dispatch (file,
          s3, dbfs, …).
        - file-like (has ``read``) — drain into a fresh
          :class:`Memory`.

        The returned BytesIO always owns its holder unless *obj* was
        already a BytesIO.
        """
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, Holder):
            return cls(holder=obj, owns_holder=False, mode=mode, **kwargs)

        if isinstance(obj, (bytes, bytearray, memoryview)):
            return cls(
                holder=Memory(binary=obj), owns_holder=True, mode=mode, **kwargs,
            )

        if hasattr(obj, "read") and not isinstance(obj, (str, bytes)):
            mem = Memory()
            mem.acquire()
            try:
                pos = 0
                while True:
                    chunk = obj.read(64 * 1024)
                    if not chunk:
                        break
                    mem.pwrite(chunk, pos)
                    pos += len(chunk)
            except BaseException:
                mem.close()
                raise
            return cls(holder=mem, owns_holder=True, mode=mode, **kwargs)

        # Path-like — let Holder dispatch decide the scheme via its
        # __new__ registry routing.
        try:
            holder = Holder(data=obj)
            return cls(holder=holder, owns_holder=True, mode=mode, **kwargs)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Cannot wrap {type(obj).__name__} as a BytesIO. "
                f"Accepted: BytesIO, Holder, bytes-like, file-like, "
                f"str/PurePath/URL. Got {obj!r}."
            ) from exc

    # ==================================================================
    # Disposable lifecycle — open is "buffered transaction"
    # ==================================================================

    def _acquire(self) -> None:
        """Open a buffered transaction over the holder.

        Builds a :class:`Memory` scratch holder seeded from the
        durable holder's bytes (or empty for ``"wb"``-class modes
        that truncate at open). All read / write ops while open
        route through scratch — the durable holder is untouched
        until :meth:`flush` or :meth:`_release` commits.

        Mode side-effects:

        - ``"wb"`` / ``"w+b"`` — scratch starts empty (the durable
          truncate happens at commit time).
        - ``"ab"`` / ``"a+b"`` — scratch seeded from durable bytes,
          cursor parked at EOF.
        - ``"xb"`` — fail-fast :class:`FileExistsError` if the
          durable holder is non-empty.
        - ``"rb"`` / ``"rb+"`` / default — scratch seeded from
          durable bytes, cursor at 0.

        Note: must NOT call ``self._holder.open()`` — that's the
        BytesIO-returning convenience and would recurse.
        """
        if self._owns_holder:
            self._holder.acquire()

        if "x" in self._mode and self._holder.size > 0:
            raise FileExistsError(
                f"{type(self).__name__} opened with mode={self._mode!r} "
                f"but holder is non-empty ({self._holder.size} bytes)."
            )

        from yggdrasil.io.memory import Memory
        scratch = Memory()
        scratch.acquire()

        # Seed scratch from the durable holder for read-friendly
        # modes; "wb" / "w+b" start clean (truncate-on-commit).
        if "w" not in self._mode and self._holder.size > 0:
            scratch.pwrite(self._holder.read_bytes(), 0)

        self._scratch = scratch
        self._pos = scratch.size if "a" in self._mode else 0

    def _commit(self) -> None:
        """Push the scratch buffer's bytes onto the durable holder.

        :class:`Disposable._close` calls :meth:`commit` (which routes
        here when :attr:`_dirty`) before :meth:`_release`, so by the
        time :meth:`_release` runs the durable holder already has the
        new bytes. ``temporary`` holders skip this — :meth:`_release`
        on the holder side drops the payload.
        """
        scratch = self._scratch
        if scratch is None:
            return
        if getattr(self._holder, "temporary", False):
            return
        self._commit_scratch(scratch)

    def _release(self) -> None:
        """Tear down scratch and release the durable holder if owned.

        Commit is :meth:`_commit`'s job — this method is pure cleanup.
        """
        scratch = self._scratch
        if scratch is not None:
            try:
                scratch.close()
            except Exception:
                pass
            self._scratch = None

        if self._owns_holder:
            try:
                self._holder.close()
            except Exception:
                pass

    def _commit_scratch(self, scratch: "Holder") -> None:
        """Push *scratch*'s bytes onto the durable holder.

        Truncate-then-rewrite: the simplest correct shape that
        preserves "what's in scratch" exactly. Backends with cheaper
        partial-update primitives (a remote multipart upload, a
        Delta append) override at the holder level — this is the
        universal fallback.
        """
        size = scratch.size
        self._holder.truncate(size)
        if size > 0:
            self._holder.pwrite(scratch.read_bytes(), 0)
        self._holder.flush()

    def _active(self) -> "Holder":
        """The holder ops should hit right now — scratch when open,
        durable otherwise.

        Centralizing the dispatch means every ``read`` / ``write`` /
        ``truncate`` / ``size`` call site stays unchanged whether
        the BytesIO is in a transaction or not.
        """
        return self._scratch if self._scratch is not None else self._holder

    # ==================================================================
    # Identity / state
    # ==================================================================

    @property
    def holder(self) -> "Holder":
        """The underlying :class:`Holder`."""
        return self._holder

    def view(
        self,
        *,
        pos: int = 0,
        size: Optional[int] = None,
        mode: str = "rb",
    ) -> "BytesIO":
        """Return a fresh, non-owning :class:`BytesIO` over the buffer.

        With *size* unset the view shares the same holder as ``self``
        — zero copy, cursor seeded at *pos*. Useful for Parquet
        footer probes, zip directory walks, magic-byte sniffs.

        With *size* set, the view holds an in-memory copy of bytes
        ``[pos, pos+size)``. That's the right shape for a *bounded*
        sub-view that should not race with later mutations of the
        parent buffer (the pickle ser layer slices a header's
        payload section out of a packed wire format like this).
        """
        if size is None:
            v = BytesIO(holder=self._holder, owns_holder=False, mode=mode)
            v._pos = int(pos)
            return v
        if size < 0:
            raise ValueError(f"view size must be >= 0, got {size!r}")
        # Bounded view: snapshot the requested range.
        payload = self.pread(int(size), int(pos))
        return BytesIO(payload)

    # ==================================================================
    # Codec auto-handling — peeks at the holder's MediaType
    # ==================================================================

    def _codec(self):
        """The codec on this buffer's :class:`MediaType`, or ``None``.

        Path-bound holders learn their media type from the URL
        suffix at construction (``data.csv.gz`` → CSV + GZIP);
        callers that build a :class:`Memory` holder by hand can
        seed ``stat().media_type`` to opt the buffer into codec
        round-tripping.
        """
        holder = self._holder
        if holder is None:
            return None
        try:
            mt = holder.stat().media_type
        except Exception:
            return None
        return getattr(mt, "codec", None) if mt is not None else None

    def _format_view(self) -> "BytesIO":
        """A read-only :class:`BytesIO` over the *format* bytes.

        When the holder is uncompressed (no codec on the media type),
        returns a non-owning :meth:`view` of ``self``. When a codec
        is present, returns a fresh in-memory :class:`BytesIO` whose
        bytes are the decompressed payload — leaf readers parse the
        format directly from it without knowing the wire was
        compressed.

        The returned buffer is the caller's to close.
        """
        codec = self._codec()
        if codec is None:
            return self.view(pos=0)
        # ``codec.decompress`` accepts the source BytesIO and returns
        # a freshly-allocated decompressed BytesIO; caller closes.
        return codec.decompress(self)

    def _format_buffer(self) -> "_FormatBufferContext":
        """Context manager yielding a buffer to write raw format bytes into.

        For an uncompressed holder, the yielded buffer is ``self``,
        already truncated to zero so the writer starts clean.
        For a codec-tagged holder, the yielded buffer is a fresh
        in-memory :class:`BytesIO`; on exit the bytes are compressed
        and committed to ``self``.
        """
        return _FormatBufferContext(self)

    @property
    def owns_holder(self) -> bool:
        """Whether closing self also closes the holder."""
        return self._owns_holder

    @property
    def size(self) -> int:
        """Live size from the active holder — scratch when open,
        durable otherwise."""
        return self._active().size

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        return True

    def __bytes__(self) -> bytes:
        """Snapshot the active payload as :class:`bytes`."""
        return self.to_bytes()

    def __repr__(self) -> str:
        state = "open" if self._acquired else "closed"
        own = "owns" if self._owns_holder else "borrows"
        return (
            f"<{type(self).__name__} {state} {own} holder={self._holder!r} "
            f"pos={self._pos} mode={self._mode!r}>"
        )

    # ==================================================================
    # Tabular hooks — default raises; format-specific leaves override
    # ==================================================================

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Default — opaque buffer can't honestly yield Arrow batches.

        Format-specific subclasses (ParquetIO, CsvIO, ArrowIPCIO, …)
        override against the same holder. For dispatch by media type,
        construct via the format leaf directly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no tabular decoder. "
            "Construct via the format leaf (ParquetIO, CsvIO, …) "
            "to read Arrow record batches from this byte buffer."
        )

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Default — opaque buffer can't honestly accept Arrow batches.

        Format-specific subclasses override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no tabular encoder. "
            "Construct via the format leaf (ParquetIO, CsvIO, …) "
            "to write Arrow record batches into this byte buffer."
        )

    # ==================================================================
    # IO[bytes] protocol — modes & predicates
    # ==================================================================

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def name(self) -> str:
        return str(self._holder.url)

    @property
    def media_type(self):
        """The buffer's :class:`MediaType`, or ``None``.

        Convenience over ``self._holder.stat().media_type`` — same
        thing the codec auto-handling reads.
        """
        try:
            return self._holder.stat().media_type
        except Exception:
            return None

    def with_media_type(self, media_type: Any, *, copy: bool = False) -> "BytesIO":
        """Stamp *media_type* onto the holder's :class:`IOStats`.

        With ``copy=False`` (the default), mutates ``self`` and
        returns it. ``copy=True`` allocates a fresh holder over the
        same bytes and returns a new BytesIO over it.
        """
        from yggdrasil.data.enums.media_type import MediaType
        mt = MediaType.from_(media_type, default=None) if media_type is not None else None
        if copy:
            payload = self.to_bytes()
            new_io = BytesIO(payload, media_type=mt)
            return new_io
        if mt is not None:
            self._holder.stat().media_type = mt
        return self

    @property
    def closed(self) -> bool:
        """Stdlib ``IO[bytes]`` parity — ``False`` while the durable
        holder is reachable.

        Stdlib semantics: ``closed`` means "file unusable for I/O."
        Our holders are reachable in both the closed-Disposable
        (direct synchronous commit) and open-Disposable (scratch
        transaction) states, so the property only flips when a
        ``close(force=True)`` has actually run teardown. This matters
        for pyarrow / pandas / polars / zipfile, which guard every
        op with an ``assert not closed`` and would otherwise refuse
        to write into a fresh, never-explicitly-opened ``BytesIO``.
        """
        # Holder cleared its scratch and we own it but have nothing
        # to fall back onto — that's the only honestly-closed state.
        return self._holder is None

    def readable(self) -> bool:
        return "r" in self._mode or "+" in self._mode

    def writable(self) -> bool:
        return any(c in self._mode for c in "wax+")

    def seekable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        """Underlying fd if the holder exposes one. Raises otherwise."""
        fileno = getattr(self._holder, "fileno", None)
        if fileno is None:
            raise OSError(
                f"{type(self).__name__} over {type(self._holder).__name__} "
                "has no underlying file descriptor."
            )
        return fileno()

    # ==================================================================
    # Cursor
    # ==================================================================

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Seek to *offset* relative to *whence*.

        Mirrors :meth:`io.IOBase.seek` with two ergonomic deviations
        that match the rest of the codebase:

        * ``seek(-1, SEEK_SET)`` is a "go to end" sentinel — pairs
          with ``read(-1)`` / "read all". Any other negative
          ``SEEK_SET`` offset raises :class:`ValueError`.
        * ``SEEK_CUR`` / ``SEEK_END`` with a negative offset that
          would land before byte 0 clamps to 0 instead of raising.
        """
        offset = int(offset)
        size = self.size
        if whence == io.SEEK_SET:
            if offset == -1:
                self._pos = size
            elif offset < 0:
                raise ValueError(
                    f"Negative SEEK_SET offset {offset!r} is invalid; "
                    f"use SEEK_END to count from the end."
                )
            else:
                self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos = max(0, self._pos + offset)
        elif whence == io.SEEK_END:
            self._pos = max(0, size + offset)
        else:
            raise ValueError(f"Invalid whence: {whence!r}")
        return self._pos

    # ==================================================================
    # Cursorless I/O — pass through to the holder
    # ==================================================================

    def pread(self, n: int, pos: int) -> bytes:
        """Positional read against the active holder. Cursor untouched."""
        return self._active().pread(n, pos)

    def pwrite(self, data: BytesLike, pos: int) -> int:
        """Positional write against the active holder. Cursor untouched.

        Marks the BytesIO dirty when scratch is active so the commit
        on :meth:`flush` / :meth:`_release` actually fires.
        """
        n = self._active().pwrite(data, pos)
        if n > 0 and self._scratch is not None:
            self.mark_dirty()
        return n

    def memoryview(self) -> memoryview:
        """Memoryview over the active holder's full payload."""
        return self._active().memoryview()

    # ==================================================================
    # IO[bytes] core
    # ==================================================================

    def read(self, size: int = -1) -> bytes:
        remaining = max(0, self.size - self._pos)
        if size is None or size < 0:
            size = remaining
        else:
            # Cap to remaining bytes — stdlib ``IOBase.read`` returns
            # fewer than *size* when EOF is reached, so we do the same
            # rather than asking the holder for an out-of-range slice.
            size = min(size, remaining)
        if size == 0:
            return b""
        out = self._active().pread(size, self._pos)
        self._pos += len(out)
        return out

    def readall(self) -> bytes:
        """Read from cursor to EOF, advancing the cursor."""
        return self.read(-1)

    def readinto(self, b: Any) -> int:
        mv = memoryview(b)
        n = len(mv)
        if n == 0:
            return 0
        chunk = self._active().pread(n, self._pos)
        got = len(chunk)
        if got:
            mv[:got] = chunk
            self._pos += got
        return got

    def readinto1(self, b: Any) -> int:
        return self.readinto(b)

    def readline(self, limit: int = -1) -> bytes:
        size = self.size
        if self._pos >= size:
            return b""
        if limit is None or limit < 0:
            chunk_len = size - self._pos
        else:
            chunk_len = min(limit, size - self._pos)
        if chunk_len <= 0:
            return b""

        chunk = self._active().pread(chunk_len, self._pos)
        nl = chunk.find(b"\n")
        if nl == -1:
            self._pos += len(chunk)
            return chunk

        line = chunk[: nl + 1]
        self._pos += len(line)
        return line

    def readlines(self, hint: int = -1) -> list[bytes]:
        lines: list[bytes] = []
        total = 0
        while True:
            line = self.readline()
            if not line:
                break
            lines.append(line)
            total += len(line)
            if hint is not None and hint > 0 and total >= hint:
                break
        return lines

    def write(self, b: Any) -> int:
        """Write *b* at the cursor, advancing it.

        Accepts bytes-like, ``str`` (UTF-8), ``io.BytesIO``, or any
        file-like with ``.read``. The buffer-protocol fallback catches
        things like :class:`pyarrow.Buffer` that aren't
        bytes/bytearray/memoryview but ARE memoryview-able.
        """
        if b is None:
            return 0
        if isinstance(b, str):
            return self.write_bytes(b.encode("utf-8"))
        if isinstance(b, (bytes, bytearray, memoryview)):
            return self.write_bytes(b)
        if hasattr(b, "read"):
            total = 0
            while True:
                chunk = b.read(1024 * 1024)
                if not chunk:
                    break
                total += self.write_bytes(chunk)
            return total
        return self.write_bytes(memoryview(b))

    def write_bytes(self, b: BytesLike) -> int:
        mv = _as_byte_mv(b)
        if len(mv) == 0:
            return 0
        n = self._active().pwrite(mv, self._pos)
        self._pos += n
        if n > 0 and self._scratch is not None:
            self.mark_dirty()
        return n

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def truncate(self, size: Optional[int] = None) -> int:
        if size is None:
            size = self._pos
        size = int(size)
        active = self._active()
        prev = active.size
        n = active.truncate(size)
        if self._pos > n:
            self._pos = n
        if n != prev and self._scratch is not None:
            self.mark_dirty()
        return n

    def flush(self) -> None:
        """Push scratch buffer to the durable holder.

        When open: commit the scratch buffer's bytes onto the durable
        holder, then clear the dirty bit. The scratch stays — further
        writes continue to buffer.
        When closed: no-op (every closed-mode op already committed
        synchronously through the durable holder).
        """
        scratch = self._scratch
        if scratch is None:
            return
        if self._dirty:
            self._commit_scratch(scratch)
            self.clear_dirty()

    def close(self, force: bool = False) -> None:
        """Close the BytesIO; closes the holder iff :attr:`owns_holder`."""
        super().close(force=force)

    # ==================================================================
    # Convenience drains
    # ==================================================================

    def to_bytes(self) -> bytes:
        """Whole-payload snapshot from the active holder. Cursor untouched."""
        return self._active().read_bytes()

    def getvalue(self) -> bytes:
        """Stdlib-compatible alias for :meth:`to_bytes`."""
        return self.to_bytes()

    def decode(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        """Decode the whole payload as text. Cursor untouched."""
        return self.to_bytes().decode(encoding, errors=errors)

    def to_base64(self, urlsafe: bool = True) -> str:
        b = self.to_bytes()
        if urlsafe:
            return base64.urlsafe_b64encode(b).decode("ascii")
        return base64.b64encode(b).decode("ascii")

    # ==================================================================
    # Iteration
    # ==================================================================

    def __iter__(self) -> "BytesIO":
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    # ==================================================================
    # Structured binary I/O — fixed-width little-endian primitives
    # ==================================================================

    def _read_exact(self, n: int) -> bytes:
        data = self.read(n)
        if len(data) != n:
            raise EOFError(f"expected {n} bytes, got {len(data)}")
        return data

    def read_int8(self) -> int: return struct.unpack("<b", self._read_exact(1))[0]
    def write_int8(self, v: int) -> int: return self.write_bytes(struct.pack("<b", int(v)))
    def read_uint8(self) -> int: return struct.unpack("<B", self._read_exact(1))[0]
    def write_uint8(self, v: int) -> int: return self.write_bytes(struct.pack("<B", int(v)))
    def read_int16(self) -> int: return struct.unpack("<h", self._read_exact(2))[0]
    def write_int16(self, v: int) -> int: return self.write_bytes(struct.pack("<h", int(v)))
    def read_uint16(self) -> int: return struct.unpack("<H", self._read_exact(2))[0]
    def write_uint16(self, v: int) -> int: return self.write_bytes(struct.pack("<H", int(v)))
    def read_int32(self) -> int: return struct.unpack("<i", self._read_exact(4))[0]
    def write_int32(self, v: int) -> int: return self.write_bytes(struct.pack("<i", int(v)))
    def read_uint32(self) -> int: return struct.unpack("<I", self._read_exact(4))[0]
    def write_uint32(self, v: int) -> int: return self.write_bytes(struct.pack("<I", int(v)))
    def read_int64(self) -> int: return struct.unpack("<q", self._read_exact(8))[0]
    def write_int64(self, v: int) -> int: return self.write_bytes(struct.pack("<q", int(v)))
    def read_uint64(self) -> int: return struct.unpack("<Q", self._read_exact(8))[0]
    def write_uint64(self, v: int) -> int: return self.write_bytes(struct.pack("<Q", int(v)))
    def read_f32(self) -> float: return struct.unpack("<f", self._read_exact(4))[0]
    def write_f32(self, v: float) -> int: return self.write_bytes(struct.pack("<f", float(v)))
    def read_f64(self) -> float: return struct.unpack("<d", self._read_exact(8))[0]
    def write_f64(self, v: float) -> int: return self.write_bytes(struct.pack("<d", float(v)))
    def read_bool(self) -> bool: return bool(self.read_uint8())
    def write_bool(self, v: bool) -> int: return self.write_uint8(1 if v else 0)

    def read_bytes_u32(self) -> bytes:
        """Length-prefixed (uint32 LE) bytes blob."""
        return self._read_exact(self.read_uint32())

    def write_bytes_u32(self, data: BytesLike) -> int:
        mv = memoryview(data)
        return self.write_uint32(len(mv)) + self.write_bytes(mv)

    def read_str_u32(self, encoding: str = "utf-8") -> str:
        """Length-prefixed UTF-8 string."""
        return self.read_bytes_u32().decode(encoding)

    def write_str_u32(self, s: str, encoding: str = "utf-8") -> int:
        return self.write_bytes_u32(s.encode(encoding))

    # ==================================================================
    # Hashing convenience — duck-typed for callers that do
    # ``buffer.xxh3_int64()`` for fingerprinting / dedup
    # ==================================================================

    # ==================================================================
    # Convenience: parse / decompress
    # ==================================================================

    def json_load(self, *, media_type: Any = None, orient: Any = None) -> Any:
        """Parse the buffer as JSON and return the Python object.

        ``media_type`` and ``orient`` are accepted for compatibility
        with the response layer — when ``orient`` is set the buffer
        is treated as a pandas-shaped JSON document. The default
        path is the stdlib ``json.loads`` over the decoded bytes.
        """
        import json as _json
        text = self.to_bytes().decode("utf-8", errors="replace")
        if not text.strip():
            return None
        if orient is not None:
            try:
                import pandas as pd
                return pd.read_json(text, orient=orient)
            except Exception:
                # Fall through to plain json on any pandas snag.
                pass
        return _json.loads(text)

    def decompress(self, *, codec: Any = None, copy: bool = True) -> "BytesIO":
        """Return a new :class:`BytesIO` over the decompressed payload.

        ``codec`` may be a :class:`Codec`, a codec name (``"gzip"``,
        ``"zstd"``, …), or a :class:`MediaType`-shaped object whose
        ``codec`` attribute is read. Returns the original buffer
        when no codec is set / supplied.
        """
        if codec is None:
            codec_obj = self._codec()
        else:
            inner = getattr(codec, "codec", None)
            if inner is not None:
                codec_obj = inner
            else:
                from yggdrasil.data.enums.codec import Codec
                codec_obj = Codec.from_(codec, default=None)
        if codec_obj is None:
            if copy:
                return BytesIO(self.to_bytes())
            return self
        out = codec_obj.decompress(self)
        return out

    def xxh3_64(self):
        """Return an :class:`xxhash.xxh3_64` instance over the payload."""
        import xxhash
        return xxhash.xxh3_64(self.to_bytes())

    def xxh3_int64(self) -> int:
        """64-bit xxh3 hash of the buffer's payload as a signed int64.

        ``xxh3_64`` itself produces an unsigned 64-bit value;
        downstream Arrow schemas pin the field as ``int64``, so we
        wrap into signed range ``[-2**63, 2**63)`` here.
        """
        import xxhash
        v = xxhash.xxh3_64(self.to_bytes()).intdigest()
        if v >= 2 ** 63:
            v -= 2 ** 64
        return v


# ===========================================================================
# Codec writer context manager
# ===========================================================================


class _FormatBufferContext:
    """Writer-side of :meth:`BytesIO._format_buffer`.

    Caller does ``with bio._format_buffer() as buf: writer(buf)``;
    the yielded ``buf`` accepts raw format bytes. On exit:

    * No codec → ``buf is bio``; we just leave the bytes in place.
      (We pre-truncate so the writer sees an empty target.)
    * Codec set → ``buf`` is a fresh in-memory BytesIO; on exit the
      bytes are compressed and committed to ``bio``.
    """

    def __init__(self, parent: "BytesIO") -> None:
        self._parent = parent
        self._buf: "BytesIO | None" = None
        self._codec = parent._codec()

    def __enter__(self) -> "BytesIO":
        if self._codec is None:
            # Direct write path: pre-truncate so the leaf writer
            # opens onto an empty target.
            self._parent.seek(0)
            self._parent.truncate(0)
            self._buf = self._parent
            return self._parent
        # Codec path: scratch buffer; we compress on exit.
        self._buf = BytesIO()
        return self._buf

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._buf is None or exc_type is not None:
            return
        if self._codec is None:
            return
        # Compress scratch into the durable buffer.
        compressed = self._codec.compress(self._buf)
        try:
            payload = compressed.to_bytes()
        finally:
            try:
                compressed.close()
            except Exception:
                pass
        try:
            self._buf.close()
        except Exception:
            pass
        self._parent.seek(0)
        self._parent.truncate(0)
        self._parent.write(payload)