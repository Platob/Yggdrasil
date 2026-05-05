"""In-memory :class:`Path` whose URL points at the bytes' address.

Each :class:`MemoryPath` instance owns one :class:`BytesIO`. The
URL is :meth:`URL.from_memory_address` of that buffer — a
``mem:///0x<hex>`` handle whose hex part is ``id(buffer)`` — so the
URL identity *is* the in-process pointer to the bytes.

What the URL gives you
----------------------

- **Round-tripping.** ``Path("mem:///0x7fa1b2c3d4e0")`` resolves the
  hex address back to the original :class:`BytesIO` (via
  :func:`resolve_memory_address`) and yields a :class:`MemoryPath`
  view over the same bytes. This works only inside the originating
  process — same caveats as
  :meth:`URL.from_memory_address`.
- **Cache keys / dispatch.** Anything that takes a URL key (media
  dispatch, request cache, log lines) accepts a :class:`MemoryPath`
  URL without special-casing.

What's intentionally gone
-------------------------

The previous implementation tried to be a real virtual filesystem:
a process-wide ``ExpiringDict`` keyed by URL, claim-aware eviction,
implicit "directories" derived from key prefixes, a separate
:class:`MemoryIO` wrapper. None of that earned its keep — callers
that want a hierarchy use a real backend; callers that want a
named in-memory buffer get one cleanly here. The path *is* the
buffer.

I/O surface
-----------

:meth:`open_context` returns a :class:`_MemoryOpenContext` that
forwards directly to the buffer — no copy on acquire, no commit on
flush. Two opens against the same path see each other's writes
immediately (POSIX-shared-fd semantics).

:meth:`_pread` / :meth:`_pwrite` produce an autonomous copy of the
bytes / replace them wholesale; :meth:`pread` / :meth:`pwrite`
forward straight to the buffer for the cheap-read / cheap-splice
case. ``write_bytes`` / ``read_bytes`` ride on top of these via the
base :class:`Path` machinery.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterator, Optional, Union

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.fs._open_context import OpenContext
from yggdrasil.io.path_stat import PathKind, PathStats
from yggdrasil.io.url import URL, resolve_memory_address

from .path import Path, register_path_class

__all__ = ["MemoryPath"]


# ---------------------------------------------------------------------------
# Open context — forwards straight to the buffer
# ---------------------------------------------------------------------------


class _MemoryOpenContext(OpenContext):
    """In-memory open context: forwards every primitive to the path's buffer.

    Same passthrough shape as :class:`_PathOpenContext` but specialised
    for the in-memory case: ``ctx.pread`` / ``ctx.pwrite`` /
    ``ctx.truncate`` go straight against the path's :class:`BytesIO`,
    no path-level method dispatch in the middle. There is no scratch
    buffer and no commit on flush — writes are already final the
    moment they land on the underlying buffer.

    Mode handling matches the rest of the OpenContext family:

    - ``rb`` / ``rb+`` — open against existing bytes; the path's
      :meth:`_open` already enforces "exists" for ``rb``.
    - ``wb`` / ``wb+`` — truncate the buffer to zero on acquire.
    - ``ab`` / ``ab+`` — leave the buffer alone; the caller drives
      the cursor.
    - ``xb`` / ``xb+`` — fail on entry if the buffer is non-empty.
    """

    __slots__ = ("_buf",)

    def __init__(self, path: "MemoryPath", mode: str) -> None:
        super().__init__(path, mode)
        # Reopen the underlying buffer if the caller closed it —
        # closure of the path is just lifecycle bookkeeping, the
        # bytes themselves stay alive.
        buf = path._ensure_buffer()
        self._buf = buf

        if "x" in mode and buf.size > 0:
            raise FileExistsError(
                f"Cannot exclusively create {path.full_path()!r}: "
                "buffer is non-empty."
            )
        if "w" in mode:
            buf.truncate(0)

        self._size = buf.size
        self._mtime = buf.mtime

    def pread(self, n: int, pos: int) -> bytes:
        if n <= 0:
            return b""
        return self._buf.pread(n, pos)

    def pwrite(self, data, pos: int) -> int:
        written = self._buf.pwrite(data, pos)
        if pos + written > self._size:
            self._size = pos + written
        return written

    def truncate(self, n: int) -> int:
        self._buf.truncate(n)
        self._size = n
        return n

    def memoryview(self) -> memoryview:
        return self._buf.memoryview()

    def fileno(self) -> int:
        # Memory buffers may have spilled to a local temp file under
        # the hood — defer to the buffer's own fileno (raises if
        # there's no fd to expose).
        return self._buf.fileno()

    def flush(self) -> None:
        return None

    def _do_close(self) -> None:
        # The buffer is shared with the path itself — closing it here
        # would tear bytes out from under any other handle. The
        # path's :meth:`_release` is what owns the buffer's lifetime.
        return None

    @property
    def size(self) -> int:
        # Live read — another handle (or a direct ``path.pwrite``)
        # could have grown the buffer beneath us.
        try:
            self._size = self._buf.size
        except Exception:
            pass
        return self._size

    @property
    def mtime(self) -> float:
        try:
            return self._buf.mtime
        except Exception:
            return self._mtime


# ===========================================================================
# MemoryPath
# ===========================================================================


class MemoryPath(Path):
    """:class:`Path` whose URL is the memory address of an in-memory buffer.

    Construction
    ------------

    Three shapes:

    - ``MemoryPath()`` — mints a fresh empty :class:`BytesIO`. URL
      becomes :meth:`URL.from_memory_address` of that buffer.
    - ``MemoryPath(b"...")`` / ``MemoryPath(memoryview(...))`` —
      mints a buffer seeded with the given bytes.
    - ``MemoryPath(url=...)`` — given a ``mem:///0x<hex>`` URL,
      :func:`resolve_memory_address` looks up the live
      :class:`BytesIO` and the path views it. Falls back to a fresh
      buffer if the address can't be resolved (object collected,
      slot reused, …) so the URL never produces a corrupt state.

    The buffer is exposed via :attr:`buffer` for callers that need
    direct access (zero-copy hand-off to pyarrow, in-process
    pipelines that want to splice bytes without going through the
    full path I/O surface).

    Lifetime
    --------

    The path holds a strong reference to its buffer, so the URL
    stays valid for round-trip resolution as long as the path is
    alive. Closing the path (:meth:`close`) closes the buffer.
    Re-resolving the URL after that returns whatever Python now
    has at that address — ``mem://`` is a *handle*, not a
    persistent reference.
    """

    scheme: ClassVar[str] = "mem"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        obj: Any = None,
        *,
        url: URL | None = None,
        temporary: bool = False,
        auto_open: bool = True,
    ) -> None:
        # Resolve the buffer first, then hand the URL to ``Path.__init__``.
        buffer = self._resolve_buffer(obj, url)
        if not buffer.opened:
            buffer.open()
        self._buffer: BytesIO = buffer

        # The URL identity IS the buffer's memory address. Re-derive
        # it even when the caller supplied a URL — if they passed a
        # stale handle that didn't resolve back to a BytesIO, we want
        # the URL to track the freshly-minted buffer instead.
        rooted_url = URL.from_memory_address(buffer)

        super().__init__(
            obj=None,
            url=rooted_url,
            temporary=temporary,
            auto_open=auto_open,
        )

    @staticmethod
    def _resolve_buffer(obj: Any, url: URL | None) -> BytesIO:
        """Find or mint the :class:`BytesIO` this path will view."""
        # Caller passed a buffer directly — view it.
        if isinstance(obj, BytesIO):
            return obj

        # Caller passed a URL — try to resolve the address.
        candidate_url: URL | None = None
        if url is not None:
            candidate_url = URL.from_(url)
        elif isinstance(obj, URL):
            candidate_url = obj
        elif isinstance(obj, str) and obj.startswith("mem:"):
            candidate_url = URL.from_str(obj)

        if candidate_url is not None and candidate_url.is_memory_address:
            try:
                resolved = resolve_memory_address(candidate_url.memory_address)
            except Exception:
                resolved = None
            if isinstance(resolved, BytesIO):
                return resolved

        # No usable URL — mint a fresh buffer, optionally seeded.
        bio = BytesIO()
        bio.open()
        if obj is None or isinstance(obj, (URL, str)):
            return bio
        if isinstance(obj, (bytes, bytearray, memoryview)):
            mv = memoryview(obj)
            if len(mv):
                bio.pwrite(mv, 0)
            return bio
        # Anything else: fall back to BytesIO's own coercion via
        # ``BytesIO.from_`` so we don't reinvent the wheel.
        try:
            return BytesIO.from_(obj)
        except Exception:
            return bio

    # ------------------------------------------------------------------
    # Disposable hooks
    # ------------------------------------------------------------------

    def _release(self) -> None:
        try:
            super()._release()
        except Exception:
            pass
        # The path owns its buffer — close it so any spill file
        # backing the buffer is dropped.
        try:
            if self._buffer is not None and not self._buffer.closed:
                self._buffer.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public buffer accessor
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> BytesIO:
        """The underlying :class:`BytesIO` — the bytes themselves."""
        return self._ensure_buffer()

    def _ensure_buffer(self) -> BytesIO:
        """Return the live buffer, reopening it if the caller closed it.

        Closing a path closes its buffer, but the bytearray inside
        survives — closure is just "no longer disposable-claimed",
        not "freed." Subsequent path I/O reopens transparently so
        the path stays usable across close/reopen cycles. The URL
        identity (the buffer's ``id()``) stays stable.
        """
        buf = self._buffer
        if buf is None:
            buf = BytesIO()
            buf.open()
            self._buffer = buf
        elif buf.closed:
            try:
                buf.open()
            except Exception:
                # Buffer can't be reopened — mint a fresh one. Loses
                # bytes, but keeps the path semantically alive.
                buf = BytesIO()
                buf.open()
                self._buffer = buf
        return buf

    @property
    def is_local(self) -> bool:
        # Same-process bytes — no network. ``is_local`` is consulted
        # by I/O routing code to decide whether mmap / sendfile are
        # in scope. For a memory holder the answer is "moot but yes."
        return True

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    def full_path(self) -> str:
        return self.url.to_string()

    def _stat(self) -> PathStats:
        # A closed buffer still has bytes — treat it as alive so
        # callers don't see a transient ``MISSING`` between
        # ``path.close()`` and the next operation. The path is
        # ``MISSING`` only when there's no buffer at all, which the
        # constructor prevents.
        buf = self._buffer
        if buf is None:
            return PathStats(kind=PathKind.MISSING)
        return PathStats(
            kind=PathKind.FILE,
            size=int(buf.size),
            mtime=buf.mtime,
        )

    def _ls(
        self,
        recursive: bool = False,
        allow_not_found: bool = True,
    ) -> Iterator["Path"]:
        # A memory holder is a single buffer, not a directory.
        # ``allow_not_found=False`` is the only way to ask for a
        # missing-directory error; otherwise return an empty iterator.
        del recursive
        if not allow_not_found:
            raise NotADirectoryError(self.full_path())
        return iter(())

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        # No directories in this filesystem — a mkdir against a
        # memory holder is meaningless. Honour ``exist_ok=True`` as
        # the existing-no-op path; raise on the explicit case so
        # callers don't silently get the wrong shape.
        del parents
        if exist_ok:
            return
        raise NotADirectoryError(
            f"{self.full_path()!r} is a memory holder, not a directory."
        )

    def _remove_file(self, allow_not_found: bool = True) -> None:
        # "Remove" the bytes — keep the buffer alive (and the URL
        # valid) but drop its content. Closing the buffer would
        # invalidate the URL and break any other reference the
        # caller still holds.
        del allow_not_found  # the buffer always exists; "missing" never fires.
        try:
            self._ensure_buffer().truncate(0)
        except Exception:
            pass

    def _remove_dir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> None:
        del recursive, with_root
        if not allow_not_found:
            raise NotADirectoryError(self.full_path())

    # ------------------------------------------------------------------
    # I/O surface
    # ------------------------------------------------------------------

    def open_context(self, mode: str = "rb", **kwargs: Any) -> OpenContext:
        """Return an :class:`OpenContext` over this path's buffer.

        The context forwards directly to :attr:`buffer` — no copy on
        acquire, no commit on flush. Spill is the inner buffer's
        own concern; the path doesn't touch it.
        """
        del kwargs
        return _MemoryOpenContext(self, mode)

    def _pread(self) -> BytesIO:
        """Return a fresh :class:`BytesIO` carrying a copy of the bytes."""
        src = self._ensure_buffer()
        bio = BytesIO()
        bio.open()
        size = src.size
        if size:
            bio.pwrite(src.pread(size, 0), 0)
            bio.seek(0)
        return bio

    def _pwrite(self, data: BytesIO) -> int:
        if not data.opened:
            data.open()
        buf = self._ensure_buffer()
        buf.truncate(0)
        size = data.size
        if size:
            buf.pwrite(data.pread(size, 0), 0)
        return size

    def pread(self, n: int, pos: int, *, default: Any = ...) -> bytes:
        del default  # the buffer always exists; "missing" never fires.
        if pos < 0:
            raise ValueError("pread position must be >= 0")
        buf = self._ensure_buffer()
        if n < 0:
            n = max(0, buf.size - pos)
        return buf.pread(n, pos)

    def pwrite(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
        *,
        parents: bool = True,
    ) -> int:
        del parents  # No directories to materialize in memory.
        if pos < 0:
            raise ValueError("pwrite position must be >= 0")
        return self._ensure_buffer().pwrite(data, pos)

    def truncate(self, n: int, *, parents: bool = True) -> int:
        del parents
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        self._ensure_buffer().truncate(n)
        return n

    def memoryview(
        self,
        *,
        offset: int = 0,
        size: Optional[int] = None,
        raise_error: bool = True,
    ) -> memoryview:
        """Return a memoryview over the buffer's bytes."""
        del raise_error  # the buffer always exists; "missing" never fires.
        if offset < 0:
            raise ValueError("memoryview offset must be >= 0")
        mv = self._ensure_buffer().memoryview()
        total = len(mv)
        end = total if size is None else min(total, offset + max(0, int(size)))
        return mv[offset:end]


# Defensive registration — ``__init_subclass__`` already does this,
# but explicit registration covers module-reload edge cases.
register_path_class(MemoryPath)
