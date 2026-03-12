# yggdrasil/io/buffer/bytes_view.py
from __future__ import annotations

import io
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .bytes_io import BytesIO


__all__ = ["BytesIOView", "open_bytes_view"]


class BytesIOView(io.RawIOBase):
    """
    Binary file-like view over a BytesIO.
    Owns its own cursor; reads/writes via parent._pread/_pwrite (cursorless).

    Semantics:
      - start is an absolute offset into the parent buffer
      - length limits the visible bytes from start (None => to EOF and can grow)
      - this view has its own cursor (relative to the view, not the parent)
    """

    __slots__ = ("_parent", "_start", "_end", "_pos", "_max_len")

    def __init__(self, parent: "BytesIO", *, start: int = 0, size: Optional[int] = None) -> None:
        super().__init__()
        if parent._closed:
            raise ValueError("I/O operation on closed BytesIO")

        total = int(parent.size)
        start = int(start)
        if start < 0:
            raise ValueError("Negative start position")

        self._parent = parent
        self._start = min(start, total)

        # If size is provided, we enforce a fixed window [start, start+size)
        # If size is None, we view [start, current EOF) and allow growth on write.
        self._max_len = None if size is None else int(size)
        if self._max_len is not None:
            if self._max_len < 0:
                raise ValueError("Negative length")
            self._end = min(self._start + self._max_len, total)
        else:
            self._end = total

        self._pos = 0  # view-local position

    def __enter__(self) -> "BytesIOView":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- io.RawIOBase -------------------------------------------------

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return int(self._pos)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        offset = int(offset)
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = int(self._pos) + offset
        elif whence == io.SEEK_END:
            pos = (self._end - self._start) + offset
        else:
            raise ValueError(f"Invalid whence: {whence!r}")

        if pos < 0:
            raise ValueError("Negative seek position")

        self._pos = pos
        return int(self._pos)

    def _avail(self) -> int:
        # current visible length (can be >=0)
        return max(0, (self._end - self._start) - int(self._pos))

    def read(self, size: int = -1) -> bytes:
        avail = self._avail()
        if avail <= 0:
            return b""

        if size is None or size < 0:
            n = avail
        else:
            n = min(int(size), avail)

        abs_pos = self._start + int(self._pos)
        out = self._parent.pread(n, abs_pos)
        self._pos += len(out)
        return out

    def readinto(self, b) -> int:
        mv = memoryview(b)
        avail = self._avail()
        if avail <= 0:
            return 0

        n = min(len(mv), avail)
        abs_pos = self._start + int(self._pos)
        chunk = self._parent.pread(n, abs_pos)
        mv[: len(chunk)] = chunk
        self._pos += len(chunk)
        return len(chunk)

    def readinto1(self, b) -> int:
        return self.readinto(b)

    # --- writing ------------------------------------------------------

    def write(self, b) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if b is None:
            return 0

        mv = memoryview(b if isinstance(b, (bytes, bytearray, memoryview)) else bytes(b))
        if len(mv) == 0:
            return 0

        # Enforce fixed window if max_len set
        if self._max_len is not None:
            remaining = self._max_len - int(self._pos)
            if remaining <= 0:
                return 0
            mv = mv[:remaining]

        abs_pos = self._start + int(self._pos)
        n = self._parent.pwrite(mv, abs_pos)
        self._pos += int(n)

        # Update end if we grew (only possible when _max_len is None or writing within window end)
        new_end = max(self._end, abs_pos + int(n))
        if self._max_len is not None:
            # still cannot exceed the capped window end
            cap_end = self._start + self._max_len
            self._end = min(new_end, cap_end)
        else:
            self._end = new_end

        # spilled writes can stale parent's mmap; parent handles that on write_bytes(),
        # but we're calling _pwrite directly so we should invalidate mmap too.
        if self._parent._buf is None:
            self._parent._invalidate_mmap()

        return int(n)

    def truncate(self, size: Optional[int] = None) -> int:
        """
        Truncate the *view window* (and underlying parent) to start+size.
        - If size is None: truncates to current position (view-local).
        - If max_len is set, cannot truncate beyond that.
        """
        if self.closed:
            raise ValueError("I/O operation on closed file.")

        if size is None:
            size = int(self._pos)
        else:
            size = int(size)
        if size < 0:
            raise ValueError("Negative size not allowed")

        if self._max_len is not None:
            size = min(size, self._max_len)

        abs_end = self._start + size

        # Memory mode: shrink logical size if needed
        if self._parent._buf is not None:
            self._parent._size = min(self._parent._size, abs_end)
        else:
            fh = self._parent.buffer()
            try:
                fh.truncate(abs_end)
                fh.flush()
            except Exception:
                # last resort: use os.ftruncate if possible
                try:
                    import os as _os
                    _os.ftruncate(fh.fileno(), abs_end)
                except Exception:
                    raise

            self._parent._invalidate_mmap()

        self._end = min(self._end, abs_end)
        self._pos = min(self._pos, size)
        return abs_end

    def flush(self) -> None:
        try:
            self._parent.flush()
        except Exception:
            pass


def open_bytes_view(
    parent: "BytesIO",
    *,
    text: bool = False,
    encoding: str = "utf-8",
    errors: str = "strict",
    newline: str = "",
    start: int = 0,
    length: Optional[int] = None,
) -> io.IOBase:
    raw = BytesIOView(parent, start=start, size=length)

    if not text:
        return raw

    # IMPORTANT: for writing, you want BufferedWriter not BufferedReader.
    # But TextIOWrapper can wrap a BufferedRWPair or BufferedWriter.
    #
    # Simplest: use BufferedRWPair so both json.load and json.dump work
    # on the same handle when text=True.
    #
    # Since our raw supports read+write, we can just use BufferedReader for reads
    # or BufferedWriter for writes. To support both seamlessly, use BufferedRandom.
    br = io.BufferedRandom(raw)
    return io.TextIOWrapper(br, encoding=encoding, errors=errors, newline=newline)