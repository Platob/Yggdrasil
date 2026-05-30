"""SendIO — a seekable, zero-copy, file-like request-body adapter over a
yggdrasil :class:`~yggdrasil.io.holder.Holder` (``Memory`` / ``MemoryStream``).

``http.client`` sends a file-like body by looping ``body.read(blocksize)`` and
``sock.sendall(chunk)``. Handing it a :class:`SendIO` — instead of a single
``read_mv(-1, 0)`` memoryview over the whole payload — means a large or
spill-backed body goes to the wire in bounded chunks, never materialised whole:

* **zero-copy** — each :meth:`read` returns a :class:`memoryview` straight into
  the holder's live in-memory window, or a bounded disk read for the spilled
  region; ``sock.sendall`` writes the view without an intermediate copy.
* **seekable / safe to replay** — a connection retry rewinds with ``seek(0)``
  and re-sends from the start, valid as long as the holder still retains byte 0
  (true for any un-evicted request body — reads never evict).
* **bounded memory** — peak stays at ~one chunk plus the holder's own in-memory
  window, not the full body, so a multi-GB PUT off a spilled ``MemoryStream``
  uploads in roughly constant memory.

The adapter reads by *absolute* offset (``base + pos``) and never touches the
holder's own cursor, so independent :class:`SendIO` instances over one holder
(e.g. one per retry attempt) don't interfere.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from yggdrasil.io.holder import Holder

__all__ = ["SendIO", "SEND_CHUNK"]

# 256 KiB per ``read`` → amortises send syscalls without holding much resident.
SEND_CHUNK = 256 * 1024


class SendIO:
    """File-like, seekable, bounded-memory view of a holder for use as a
    request body. See the module docstring."""

    __slots__ = ("_holder", "_base", "_length", "chunk_size", "_pos")

    def __init__(
        self,
        holder: "Holder",
        *,
        base: int = 0,
        length: Optional[int] = None,
        chunk_size: int = SEND_CHUNK,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size!r}")
        size = holder.size
        if base < 0 or base > size:
            raise ValueError(f"base {base} out of bounds for holder of size {size}")
        avail = size - base
        self._holder = holder
        self._base = base
        self._length = avail if length is None else max(0, min(length, avail))
        self.chunk_size = chunk_size
        self._pos = 0  # bytes already produced, relative to base

    def __len__(self) -> int:
        return self._length

    def read(self, n: int = -1) -> memoryview:
        remaining = self._length - self._pos
        if remaining <= 0:
            return memoryview(b"")
        want = remaining if (n is None or n < 0) else min(n, remaining)
        # Cap each pull at one chunk even when asked for everything, so a
        # spilled body is read from disk in bounded windows rather than
        # materialised whole.
        want = min(want, self.chunk_size)
        mv = self._holder.read_mv(want, self._base + self._pos)
        self._pos += len(mv)
        return mv

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            new = offset
        elif whence == 1:
            new = self._pos + offset
        elif whence == 2:
            new = self._length + offset
        else:
            raise ValueError(f"invalid whence {whence!r}")
        self._pos = 0 if new < 0 else (self._length if new > self._length else new)
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False
