"""Per-open I/O contexts bound to a :class:`Path`.

A :class:`OpenContext` is the live state behind one
:meth:`Path.open_context` call. It exposes the small primitive
surface (``pread`` / ``pwrite`` / ``truncate`` / ``memoryview`` /
``fileno``) the rest of the buffer machinery uses to do I/O
without caring which backing the path has.

Why this lives on Path
----------------------

Buffers used to manage two kinds of backing themselves: a spill
``fd`` for local paths and a "transaction buffer" (an inner
:class:`BytesIO` loaded from ``path.pread`` and committed via
``path.pwrite``) for non-local paths. The buffer's three primitives
— ``_slice``, ``_write_at``, ``_set_size`` — branched on backing
kind on every call.

Moving the live state to :class:`OpenContext` collapsed both shapes
into a single ``ctx.pread`` / ``ctx.pwrite`` / ``ctx.truncate``
surface backed by the *path*. Buffers stop branching; they ask the
path for a context and forward to it.

Subclasses
----------

- :class:`_FdOpenContext` — local paths. Wraps a single
  ``os.open`` fd, serves ``pread`` / ``pwrite`` via
  :func:`os.pread` / :func:`os.pwrite`, exposes :meth:`fileno` for
  callers that need raw fd access (mmap, pyarrow ``OSFile``).
- :class:`_PathOpenContext` — abstract / remote paths. Pure
  passthrough: every primitive routes to the path's own
  :meth:`pread` / :meth:`pwrite` / :meth:`truncate`. Backends that
  benefit from batching, multipart uploads, or persistent
  connections override :meth:`Path.open_context` to return a
  custom context — but the default carries no buffering and no
  hidden allocation.
"""

from __future__ import annotations

import mmap
import os
from typing import TYPE_CHECKING, Any

from yggdrasil.io.path_stat import PathKind

if TYPE_CHECKING:
    from yggdrasil.io.fs.path import Path


__all__ = [
    "OpenContext",
    "_FdOpenContext",
    "_PathOpenContext",
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

    The minimal surface every backing has to provide. Concrete
    subclasses implement the primitives in whatever shape fits the
    backing (fd, direct path passthrough, …). Closing the context
    releases per-open resources; the path itself stays alive.
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
    :func:`os.ftruncate` against this fd. ``memoryview()`` returns
    an mmap-backed view (read-only). ``fileno()`` exposes the fd for
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
# Generic path passthrough
# ---------------------------------------------------------------------------


class _PathOpenContext(OpenContext):
    """Default context: every primitive routes straight to the path.

    No buffering, no scratch BytesIO, no flush commit. ``ctx.pread``
    calls :meth:`Path.pread`; ``ctx.pwrite`` calls
    :meth:`Path.pwrite`; ``ctx.truncate`` calls
    :meth:`Path.truncate`. Memory cost is exactly what the backend
    chooses to use — no hidden duplicate buffers.

    Mode handling
    -------------

    The constructor enforces the mode contract once, against the
    path's current state, and lets the caller drive every op
    afterward:

    - ``rb`` / ``rb+`` — fail-fast if the path is missing (matches
      :func:`open`).
    - ``wb`` / ``wb+`` — :meth:`Path.truncate(0)` so the file is
      empty going in.
    - ``ab`` / ``ab+`` — leave bytes alone; cursor handling is the
      caller's job.
    - ``xb`` / ``xb+`` — fail if the path already exists.

    Backends that benefit from batching (S3 multipart, Databricks
    Volumes, paginated APIs) override :meth:`Path.open_context` to
    return a custom context. The base case is a clean passthrough.
    """

    __slots__ = ()

    def __init__(self, path: "Path", mode: str) -> None:
        super().__init__(path, mode)

        # Probe the path once so subsequent ops don't pay for repeat
        # stat round-trips. ``path.stat()`` is the single source of
        # truth on existence; we don't second-guess it.
        try:
            stat = path.stat()
        except Exception:
            stat = None

        existing = (
            stat is not None and stat.kind != PathKind.MISSING
        )
        if existing and stat is not None:
            self._size = int(stat.size)
            self._mtime = float(stat.mtime or 0.0)

        wants_truncate = "w" in mode
        wants_excl = "x" in mode
        wants_read_only = "r" in mode and "+" not in mode

        if wants_excl and existing:
            raise FileExistsError(
                f"Cannot exclusively create {path.full_path()!r}: "
                "file exists."
            )

        if wants_read_only and not existing:
            raise FileNotFoundError(path.full_path())

        if wants_truncate and existing:
            try:
                path.truncate(0)
            except FileNotFoundError:
                pass
            self._size = 0

    def pread(self, n: int, pos: int) -> bytes:
        if n <= 0:
            return b""
        try:
            return self.path.pread(n, pos)
        except FileNotFoundError:
            return b""

    def pwrite(self, data, pos: int) -> int:
        mv = memoryview(data)
        if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
            mv = mv.cast("B")
        if not mv.c_contiguous:
            mv = memoryview(bytes(mv))
        if len(mv) == 0:
            return 0
        written = self.path.pwrite(mv, pos)
        if pos + written > self._size:
            self._size = pos + written
        if written:
            self._dirty = True
        return written

    def truncate(self, n: int) -> int:
        try:
            self.path.truncate(n)
        except FileNotFoundError:
            # Truncating a missing path to non-zero is meaningless;
            # truncate-to-zero against missing is a no-op.
            if n > 0:
                raise
        self._size = n
        if self.is_writing:
            self._dirty = True
        return n

    def memoryview(self) -> memoryview:
        size = self.size
        if size == 0:
            return memoryview(b"")
        return memoryview(self.pread(size, 0))

    def flush(self) -> None:
        # Every primitive already hit the path on the way in —
        # there's nothing buffered to commit.
        self._dirty = False

    @property
    def size(self) -> int:
        # Re-stat the path on demand. Kept cheap by backends — the
        # local fast path on :meth:`LocalPath._stat` is a single
        # ``os.stat``; remote backends can cache here if they want.
        try:
            stat = self.path.stat()
        except Exception:
            return self._size
        if stat.kind == PathKind.MISSING:
            return self._size
        self._size = int(stat.size)
        return self._size

    @property
    def mtime(self) -> float:
        try:
            stat = self.path.stat()
        except Exception:
            return self._mtime
        if stat.kind == PathKind.MISSING:
            return self._mtime
        self._mtime = float(stat.mtime or 0.0)
        return self._mtime
