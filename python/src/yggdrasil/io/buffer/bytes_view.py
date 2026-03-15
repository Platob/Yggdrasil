from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bytes_io import BytesIO

__all__ = ["BytesIOView"]


class BytesIOView(io.RawIOBase):
    """
    Binary file-like window over a parent ``BytesIO``.

    The view exposes a bounded, view-relative interface over a parent buffer
    that supports cursorless random-access operations.

    Semantics
    ---------
    - ``start`` is the absolute byte offset into the parent buffer.
    - ``size`` is the current visible length of the view.
    - ``max_size`` caps growth of the view. ``None`` means unbounded growth.
    - ``pos`` is the current cursor, relative to the view, not the parent.

    Read semantics
    --------------
    - Reads are bounded by ``size``.
    - ``pread()`` performs cursorless reads using a view-relative offset.
    - ``read()`` advances ``pos``.

    Write semantics
    ---------------
    - Writes occur at a view-relative offset.
    - ``pwrite()`` does not modify ``pos``.
    - ``write()`` advances ``pos``.
    - Writes may grow ``size`` up to ``max_size`` if set.

    Parent contract
    ---------------
    The parent is expected to provide at least:
    - ``pread(n: int, pos: int) -> bytes``
    - ``pwrite(data, pos: int) -> int``
    - ``flush()``

    The implementation also opportunistically uses:
    - ``buffer()``
    - ``_buf``
    - ``_size``
    - ``_invalidate_mmap()``
    """

    __slots__ = (
        "parent",
        "start",
        "size",
        "pos",
        "max_size",
        "_closed",
    )

    def __init__(
        self,
        parent: "BytesIO",
        *,
        start: int = 0,
        size: int = 0,
        pos: int = 0,
        max_size: int | None = None,
    ) -> None:
        super().__init__()

        start = int(start)
        size = int(size)
        pos = int(pos)
        max_size = None if max_size is None else int(max_size)

        if start < 0:
            raise ValueError("start must be >= 0")
        if size < 0:
            raise ValueError("size must be >= 0")
        if pos < 0:
            raise ValueError("pos must be >= 0")
        if max_size is not None and max_size < 0:
            raise ValueError("max_size must be >= 0")
        if max_size is not None and size > max_size:
            raise ValueError("size cannot exceed max_size")

        self.parent = parent
        self.start = start
        self.size = size
        self.pos = pos
        self.max_size = max_size
        self._closed = False

    def __repr__(self) -> str:
        if self._closed:
            return "<BytesIOView [closed]>"
        return (
            f"<BytesIOView start={self.start} size={self.size} "
            f"pos={self.pos} max_size={self.max_size}>"
        )

    def __enter__(self) -> "BytesIOView":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def end(self) -> int:
        """Absolute end offset in the parent."""
        return self.start + self.size

    @property
    def remaining(self) -> int:
        """Bytes remaining from the current cursor to the view end."""
        return max(0, self.size - self.pos)

    def _coerce_bytes_like(self, b) -> memoryview:
        if b is None:
            return memoryview(b"")
        if isinstance(b, memoryview):
            return b
        if isinstance(b, (bytes, bytearray)):
            return memoryview(b)
        return memoryview(bytes(b))

    def _invalidate_parent_mmap_if_needed(self) -> None:
        """
        Invalidate parent mmap/cache after writes or truncation when relevant.

        Convention assumed from parent implementation:
        - ``_buf is None`` means spilled/file-backed mode
        - file-backed mode may use mmap that should be invalidated
        """
        if getattr(self.parent, "_buf", None) is None:
            invalidate = getattr(self.parent, "_invalidate_mmap", None)
            if callable(invalidate):
                invalidate()

    def _resolve_seek_pos(self, offset: int, whence: int) -> int:
        offset = int(offset)

        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self.pos + offset
        elif whence == io.SEEK_END:
            new_pos = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence!r}")

        if new_pos < 0:
            raise ValueError("negative seek position")

        return new_pos

    # ------------------------------------------------------------------
    # io.RawIOBase
    # ------------------------------------------------------------------

    def close(self) -> None:
        if not self._closed:
            self._closed = True
        super().close()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def flush(self) -> None:
        try:
            self.parent.flush()
        except Exception:
            pass

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = self._resolve_seek_pos(offset, whence)
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""

        if size is None or size < 0:
            n = self.remaining
        else:
            n = min(int(size), self.remaining)

        out = self.pread(n, pos=self.pos)
        self.pos += len(out)
        return out

    def readinto(self, b) -> int:
        mv = memoryview(b)
        if len(mv) == 0:
            return 0

        chunk = self.pread(len(mv), pos=self.pos)
        n = len(chunk)
        if n:
            mv[:n] = chunk
            self.pos += n
        return n

    def readinto1(self, b) -> int:
        return self.readinto(b)

    def readall(self) -> bytes:
        """Read the full view contents without modifying ``pos``."""
        return self.pread(self.size, pos=0)

    def write(self, b) -> int:
        n = self.pwrite(b, pos=self.pos)
        self.pos += n
        return n

    def truncate(self, size: int | None = None) -> int:
        """
        Truncate the view and underlying parent to ``start + size``.

        Notes
        -----
        This truncates the underlying parent storage, not just the logical view.

        - ``size=None`` truncates to the current cursor.
        - If ``max_size`` is set, truncation is capped at that limit.
        - Returns the new absolute size in the parent.
        """
        new_size = self.pos if size is None else int(size)
        if new_size < 0:
            raise ValueError("negative size not allowed")

        if self.max_size is not None:
            new_size = min(new_size, self.max_size)

        abs_end = self.start + new_size

        if getattr(self.parent, "_buf", None) is not None:
            # in-memory mode
            if hasattr(self.parent, "_size"):
                self.parent._size = min(self.parent._size, abs_end)
        else:
            # spilled/file-backed mode
            fh = self.parent.buffer()
            try:
                fh.truncate(abs_end)
                fh.flush()
            except Exception:
                try:
                    os.ftruncate(fh.fileno(), abs_end)
                except Exception:
                    raise
            finally:
                try:
                    fh.close()
                except Exception:
                    pass

            self._invalidate_parent_mmap_if_needed()

        self.size = min(self.size, new_size)
        self.pos = min(self.pos, self.size)
        return abs_end

    # ------------------------------------------------------------------
    # random access
    # ------------------------------------------------------------------

    def pread(self, n: int, pos: int = 0) -> bytes:
        """
        Read up to ``n`` bytes from a view-relative offset without changing ``pos``.

        Parameters
        ----------
        n:
            Maximum number of bytes to read. Must be >= 0.
        pos:
            View-relative offset. Must be >= 0.

        Returns
        -------
        bytes
            The bytes read, possibly fewer than requested at EOF.
        """
        n = int(n)
        pos = int(pos)

        if n < 0:
            raise ValueError("n must be >= 0")
        if pos < 0:
            raise ValueError("pos must be >= 0")
        if n == 0 or pos >= self.size:
            return b""

        n = min(n, self.size - pos)
        return self.parent.pread(n, self.start + pos)

    def pwrite(self, b, pos: int = 0) -> int:
        """
        Write bytes at a view-relative offset without changing ``pos``.

        Parameters
        ----------
        b:
            Bytes-like object to write.
        pos:
            View-relative offset. Must be >= 0.

        Returns
        -------
        int
            Number of bytes written.

        Notes
        -----
        Writes may extend the visible size of the view. If ``max_size`` is set,
        writes are clipped to fit the configured cap.
        """
        pos = int(pos)
        if pos < 0:
            raise ValueError("pos must be >= 0")

        mv = self._coerce_bytes_like(b)
        if len(mv) == 0:
            return 0

        if self.max_size is not None:
            allowed = self.max_size - pos
            if allowed <= 0:
                return 0
            mv = mv[:allowed]

        n = int(self.parent.pwrite(mv, self.start + pos))

        end_pos = pos + n
        if end_pos > self.size:
            self.size = end_pos

        self._invalidate_parent_mmap_if_needed()
        return n

    # ------------------------------------------------------------------
    # convenience helpers
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Return the full contents of the view without modifying ``pos``."""
        return self.readall()