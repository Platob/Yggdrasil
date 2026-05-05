"""Per-open I/O contexts bound to a :class:`Path`.

A :class:`OpenContext` is the live state behind one :meth:`Path.open_context`
call: a long-lived OS file descriptor for local paths, or a scratch
:class:`BytesIO` holding the file's working bytes for remote paths.

Why this lives on Path
----------------------

Buffers used to manage two kinds of backing themselves: a spill ``fd``
for local paths and a "transaction buffer" (an inner :class:`BytesIO`
loaded from ``path.pread`` and committed via ``path.pwrite``) for
non-local paths. The buffer's three primitives — ``_slice``,
``_write_at``, ``_set_size`` — branched on backing kind on every call.

Moving the live state to :class:`OpenContext` collapses both shapes
into a single ``ctx.pread`` / ``ctx.pwrite`` / ``ctx.truncate`` surface
backed by the *path*. Buffers stop branching; they ask the path for a
context and forward to it.

Subclasses
----------

- :class:`_FdOpenContext` — local paths. Wraps a single ``os.open`` fd,
  serves ``pread`` / ``pwrite`` via :func:`os.pread` / :func:`os.pwrite`,
  exposes :meth:`fileno` for callers that need raw fd access (mmap,
  pyarrow ``OSFile``).
- :class:`_BufferOpenContext` — remote / abstract paths. Builds a scratch
  :class:`BytesIO` from :meth:`Path._pread` on acquire, services reads
  and writes against it, commits via :meth:`Path._pwrite` on flush.
"""

from __future__ import annotations

import io
import mmap
import os
from typing import TYPE_CHECKING, Any

from yggdrasil.io.buffer.bytes_io import BytesIO

if TYPE_CHECKING:
    from yggdrasil.io.fs.path import Path


__all__ = [
    "OpenContext",
    "_FdOpenContext",
    "_BufferOpenContext",
]


_HAS_PREAD = hasattr(os, "pread")
_HAS_PWRITE = hasattr(os, "pwrite")


def _pread_bounded(fd: int, n: int, pos: int) -> bytes:
    """:func:`os.pread` with a Windows fallback (lseek + read)."""
    if _HAS_PREAD:
        return os.pread(fd, n, pos)
    saved = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, pos, os.SEEK_SET)
        return os.read(fd, n)
    finally:
        try:
            os.lseek(fd, saved, os.SEEK_SET)
        except OSError:
            pass


def _pwrite_bounded(fd: int, data, pos: int) -> int:
    """:func:`os.pwrite` with a Windows fallback (lseek + write)."""
    mv = memoryview(data)
    if not mv.c_contiguous:
        mv = memoryview(bytes(mv))
    if _HAS_PWRITE:
        return os.pwrite(fd, mv, pos)
    saved = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, pos, os.SEEK_SET)
        return os.write(fd, mv)
    finally:
        try:
            os.lseek(fd, saved, os.SEEK_SET)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class OpenContext:
    """Per-open I/O context bound to a :class:`Path`.

    The minimal surface every backing has to provide. Concrete subclasses
    implement the primitives in whatever shape fits the backing (fd,
    in-memory bytes, scratch buffer, …).
    """

    __slots__ = ("path", "mode", "_size", "_mtime", "_dirty", "_closed")

    def __init__(self, path: "Path", mode: str) -> None:
        self.path = path
        self.mode = mode
        self._size: int = 0
        self._mtime: float = 0.0
        self._dirty: bool = False
        self._closed: bool = False

    # -- introspection -------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    @property
    def mtime(self) -> float:
        return self._mtime

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def is_writing(self) -> bool:
        return any(c in self.mode for c in "wax+")

    # -- primitives (override) -----------------------------------------

    def pread(self, n: int, pos: int) -> bytes:
        raise NotImplementedError

    def pwrite(self, data, pos: int) -> int:
        raise NotImplementedError

    def truncate(self, n: int) -> int:
        raise NotImplementedError

    def fileno(self) -> int:
        raise OSError(
            f"{type(self).__name__} has no underlying file descriptor"
        )

    def memoryview(self) -> memoryview:
        if self._size == 0:
            return memoryview(b"")
        return memoryview(self.pread(self._size, 0))

    def flush(self) -> None:
        return None

    # -- lifecycle -----------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        finally:
            self._closed = True
            try:
                self._do_close()
            except Exception:
                pass

    def _do_close(self) -> None:
        return None

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} path={self.path.full_path()!r} "
            f"mode={self.mode!r} size={self._size} "
            f"{'dirty' if self._dirty else 'clean'} "
            f"{'closed' if self._closed else 'open'}>"
        )


# ---------------------------------------------------------------------------
# Local fd context
# ---------------------------------------------------------------------------


class _FdOpenContext(OpenContext):
    """Local context: a single long-lived ``os.open`` fd.

    All ops route through :func:`os.pread` / :func:`os.pwrite` /
    :func:`os.ftruncate` against this fd. ``memoryview()`` returns an
    mmap-backed view (read-only). ``fileno()`` exposes the fd for
    consumers like :class:`pyarrow.OSFile` that want it directly.
    """

    __slots__ = ("_fd",)

    def __init__(self, path: "Path", mode: str, *, fd: int) -> None:
        super().__init__(path, mode)
        self._fd = fd
        try:
            stat = os.fstat(fd)
            self._size = int(stat.st_size)
            self._mtime = float(stat.st_mtime)
        except OSError:
            self._size = 0
            self._mtime = 0.0

    def fileno(self) -> int:
        return self._fd

    def pread(self, n: int, pos: int) -> bytes:
        if n <= 0:
            return b""
        return _pread_bounded(self._fd, n, pos)

    def pwrite(self, data, pos: int) -> int:
        mv = memoryview(data)
        if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
            mv = mv.cast("B")
        if not mv.c_contiguous:
            mv = memoryview(bytes(mv))
        if len(mv) == 0:
            return 0
        written = _pwrite_bounded(self._fd, mv, pos)
        if pos + written > self._size:
            self._size = pos + written
        return written

    def truncate(self, n: int) -> int:
        os.ftruncate(self._fd, n)
        self._size = n
        return n

    def memoryview(self) -> memoryview:
        # Re-stat — fd writes may have grown the file beyond the
        # cached _size if some other consumer of the same fd wrote
        # too. (mmap on a stale length truncates the view.)
        try:
            size = os.fstat(self._fd).st_size
            self._size = int(size)
        except OSError:
            size = self._size
        if size == 0:
            return memoryview(b"")
        return memoryview(mmap.mmap(self._fd, size, access=mmap.ACCESS_READ))

    def flush(self) -> None:
        # Writes already in the kernel via pwrite — nothing else.
        return None

    def _do_close(self) -> None:
        fd, self._fd = self._fd, -1
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

    @property
    def size(self) -> int:
        # Live fstat: another writer through the same path could have
        # grown the file beneath us. Cheaper than path.stat() because
        # we already hold the fd.
        if not self._closed and self._fd >= 0:
            try:
                self._size = int(os.fstat(self._fd).st_size)
            except OSError:
                pass
        return self._size

    @property
    def mtime(self) -> float:
        if not self._closed and self._fd >= 0:
            try:
                self._mtime = float(os.fstat(self._fd).st_mtime)
            except OSError:
                pass
        return self._mtime


# ---------------------------------------------------------------------------
# Remote / generic buffer context
# ---------------------------------------------------------------------------


class _BufferOpenContext(OpenContext):
    """Remote context: a scratch :class:`BytesIO` holding the working bytes.

    On acquire the path's existing bytes (if any) are pulled in via
    :meth:`Path._pread`. Reads and writes hit the inner buffer. The
    first successful write flips ``_dirty``; on :meth:`flush` the
    buffer is committed back to the path via :meth:`Path._pwrite`.
    Read-only opens skip the commit entirely.

    Mode mapping mirrors ``open()``:

    - ``rb`` / ``rb+`` — pread the existing bytes; missing path raises
      ``FileNotFoundError`` (``rb+`` is tolerant: an empty buffer is
      fine).
    - ``wb`` / ``wb+`` — start empty (truncate semantics). The flush
      writes the new content.
    - ``ab`` / ``ab+`` — pread (or empty if missing); cursor is the
      caller's concern. Flush rewrites the whole file with the
      buffer's contents.
    - ``xb`` / ``xb+`` — raise ``FileExistsError`` if the path exists;
      otherwise start empty.
    """

    __slots__ = ("_buffer",)

    def __init__(
        self,
        path: "Path",
        mode: str,
        *,
        spill_bytes: int = 128 * 1024 * 1024,
        spill_ttl: int = 86400,
    ) -> None:
        super().__init__(path, mode)
        self._buffer: BytesIO = BytesIO(
            spill_bytes=spill_bytes,
            spill_ttl=spill_ttl,
            mode="rb+",
            auto_open=True,
        )
        self._load_initial(mode)

    # -- introspection ----

    @property
    def buffer(self) -> BytesIO:
        return self._buffer

    # -- init helpers ----

    def _load_initial(self, mode: str) -> None:
        wants_truncate = "w" in mode
        wants_excl = "x" in mode
        if wants_truncate:
            self._size = 0
            return

        # Fail-first existence: try the read; the backend tells us
        # via FileNotFoundError whether the file was there.
        existing: BytesIO | None = None
        already_present = False
        try:
            existing = self.path._pread()
            already_present = True
        except FileNotFoundError:
            already_present = False

        if wants_excl and already_present:
            try:
                if existing is not None:
                    existing.close()
            except Exception:
                pass
            raise FileExistsError(
                f"Cannot exclusively create {self.path.full_path()!r}: "
                "file exists."
            )

        if wants_excl:
            self._size = 0
            return

        if existing is None:
            self._size = 0
            return

        try:
            size = existing.size
            if size:
                # Drain in chunks to honour the inner buffer's spill
                # threshold without ever materialising the whole
                # payload as a single bytes object.
                chunk_size = 4 * 1024 * 1024
                pos = 0
                while pos < size:
                    want = min(chunk_size, size - pos)
                    chunk = existing.pread(want, pos)
                    if not chunk:
                        break
                    self._buffer.pwrite(chunk, pos)
                    pos += len(chunk)
                self._size = self._buffer.size
            else:
                self._size = 0
            self._mtime = existing.mtime
        finally:
            try:
                existing.close()
            except Exception:
                pass

    # -- primitives ----

    def pread(self, n: int, pos: int) -> bytes:
        if n <= 0:
            return b""
        return self._buffer.pread(n, pos)

    def pwrite(self, data, pos: int) -> int:
        written = self._buffer.pwrite(data, pos)
        if written and self.is_writing:
            self._dirty = True
        if pos + written > self._size:
            self._size = pos + written
        return written

    def truncate(self, n: int) -> int:
        self._buffer.truncate(n)
        self._size = n
        if self.is_writing:
            self._dirty = True
        return n

    def memoryview(self) -> memoryview:
        return self._buffer.memoryview()

    def flush(self) -> None:
        if not self._dirty:
            return
        if not self.is_writing:
            self._dirty = False
            return
        # Commit the whole working buffer back to the path. One round
        # trip with truncate-and-replace semantics. Backends that want
        # streaming uploads pick that up inside their _pwrite.
        pos = self._buffer.tell()
        try:
            self._buffer.seek(0)
            self.path._pwrite(self._buffer)
        finally:
            try:
                self._buffer.seek(pos)
            except Exception:
                pass
        self._dirty = False

    def _do_close(self) -> None:
        try:
            self._buffer.close()
        except Exception:
            pass
