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
  process — same caveats as :meth:`URL.from_memory_address`.
- **Cache keys / dispatch.** Anything that takes a URL key (media
  dispatch, request cache, log lines) accepts a :class:`MemoryPath`
  URL without special-casing.

I/O surface
-----------

The path's transaction buffer IS the underlying :class:`BytesIO` —
no copy on acquire, no commit on flush. Two opens against the
same memory path see each other's writes immediately.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterator, Optional, Union

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.enums import Mode
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL, resolve_memory_address

from .path import Path, register_path_class

__all__ = ["MemoryPath"]


# ===========================================================================
# MemoryPath
# ===========================================================================


class MemoryPath(Path):
    """:class:`Path` whose URL is the memory address of an in-memory buffer."""

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
        mode: str = "rb+",
        auto_open: bool = True,
    ) -> None:
        buffer = self._resolve_buffer(obj, url)
        if not buffer.opened:
            buffer.open()
        self._buffer: BytesIO = buffer

        rooted_url = URL.from_memory_address(buffer)

        super().__init__(
            obj=None,
            url=rooted_url,
            temporary=temporary,
            mode=mode,
            auto_open=auto_open,
        )

    @staticmethod
    def _resolve_buffer(obj: Any, url: URL | None) -> BytesIO:
        if isinstance(obj, BytesIO):
            return obj

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

        bio = BytesIO()
        bio.open()
        if obj is None or isinstance(obj, (URL, str)):
            return bio
        if isinstance(obj, (bytes, bytearray, memoryview)):
            mv = memoryview(obj)
            if len(mv):
                bio.pwrite(mv, 0)
            return bio
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
        return self._ensure_buffer()

    def _ensure_buffer(self) -> BytesIO:
        buf = self._buffer
        if buf is None:
            buf = BytesIO()
            buf.open()
            self._buffer = buf
        elif buf.closed:
            try:
                buf.open()
            except Exception:
                buf = BytesIO()
                buf.open()
                self._buffer = buf
        return buf

    @property
    def is_local_path(self) -> bool:
        # In-process bytes are reachable without network IO; we count
        # MemoryPath as local even though it has no real filesystem.
        return True

    @property
    def is_remote_path(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Acquire — alias the path's transaction buffer to the inner BytesIO
    # ------------------------------------------------------------------

    def _ensure_io(self) -> None:
        # Skip the local-fd / remote-buffer split — a memory path is
        # always backed by its own BytesIO, no fd, no commit cycle.
        if self._transaction_buffer is not None:
            return
        buf = self._ensure_buffer()
        mode = self._mode
        if "x" in mode and buf.size > 0:
            raise FileExistsError(
                f"Cannot exclusively create {self.full_path()!r}: "
                "buffer is non-empty."
            )
        if "w" in mode:
            buf.truncate(0)
        self._transaction_buffer = buf

    def close_io(self) -> None:
        # The buffer is shared with the path itself — closing it here
        # would tear bytes out from under any other handle. Just drop
        # the transaction-buffer reference and clear the dirty flag.
        self._transaction_buffer = None
        self._dirty = False

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    def full_path(self) -> str:
        return self.url.to_string()

    def _stat(self) -> IOStats:
        buf = self._buffer
        if buf is None:
            return IOStats(kind=IOKind.MISSING)
        return IOStats(
            kind=IOKind.FILE,
            size=int(buf.size),
            mtime=float(buf.mtime or 0.0),
        )

    def _ls(
        self,
        recursive: bool = False,
        allow_not_found: bool = True,
    ) -> Iterator["Path"]:
        del recursive
        if not allow_not_found:
            raise NotADirectoryError(self.full_path())
        return iter(())

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        del parents
        if exist_ok:
            return
        raise NotADirectoryError(
            f"{self.full_path()!r} is a memory holder, not a directory."
        )

    def _remove_file(self, allow_not_found: bool = True) -> None:
        del allow_not_found
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
    # I/O surface — direct against the buffer
    # ------------------------------------------------------------------

    def _bread(self, n: int, pos: int, mode: "Mode") -> BytesIO:
        """Native positional read — slice the inner BytesIO directly."""
        del mode
        src = self._ensure_buffer()
        bio = BytesIO()
        bio.open()
        if n == 0:
            return bio
        size = src.size
        if pos >= size:
            return bio
        want = (size - pos) if n < 0 else min(n, size - pos)
        if want <= 0:
            return bio
        bio.pwrite(src.pread(want, pos), 0)
        bio.seek(0)
        return bio

    def _bwrite(self, data: BytesIO, pos: int, mode: "Mode") -> int:
        """Native positional write — splice into the inner BytesIO.

        ``pos == 0`` with :class:`Mode.OVERWRITE` truncates the
        backing first to mirror legacy whole-file write semantics;
        any other shape splices in place via :meth:`BytesIO.pwrite`.
        """
        if not data.opened:
            data.open()
        buf = self._ensure_buffer()
        if pos == 0 and mode is Mode.OVERWRITE:
            buf.truncate(0)
        size = data.size
        if size:
            buf.pwrite(data.pread(size, 0), pos)
        return size

    def pread(self, n: int, pos: int, *, default: Any = ...) -> bytes:
        del default
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
        del parents
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
        del raise_error
        if offset < 0:
            raise ValueError("memoryview offset must be >= 0")
        mv = self._ensure_buffer().memoryview()
        total = len(mv)
        end = total if size is None else min(total, offset + max(0, int(size)))
        return mv[offset:end]


register_path_class(MemoryPath)
