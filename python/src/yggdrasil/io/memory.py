"""Fully in-memory :class:`Holder` backed by a :class:`bytearray`.

:class:`Memory` is the simplest concrete :class:`Holder`: every
read/write hits an internally-managed :class:`bytearray`. No fd, no
spill file, no transaction layer. Capacity grows with the standard
1.5× amortization pattern so back-to-back appends stay cheap.

Composes with :class:`yggdrasil.io.buffer.BytesIO`: a memory-mode
``BytesIO`` is conceptually a Memory holder plus a cursor and the
TabularIO read/write surface.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Union

from .holder import Holder


__all__ = ["Memory"]


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class Memory(Holder):
    """Fully memory-resident byte holder.

    Construction shapes:

    - ``Memory()`` — empty, zero capacity.
    - ``Memory(int n)`` — empty, capacity ``n`` reserved.
    - ``Memory(bytes_like)`` — seeded with the given bytes; visible
      size is ``len(bytes_like)``.
    - ``Memory(other_memory)`` — deep copy.

    The visible :attr:`size` and the underlying ``bytearray`` capacity
    are tracked separately. ``reserve(n)`` grows capacity without
    moving :attr:`size`; :meth:`truncate` moves :attr:`size`,
    zero-padding on extend up to capacity.
    """

    __slots__ = ("_buf", "_size", "_mtime", "_media_type")

    def __init__(
        self,
        data: Optional[Union[
            int,
            bytes,
            bytearray,
            memoryview,
            "Memory",
        ]] = None,
        *,
        media_type: Any = None,
    ) -> None:
        self._mtime: float = time.time()
        self._media_type: Any = media_type

        if data is None:
            self._buf: bytearray = bytearray()
            self._size: int = 0
            return

        if isinstance(data, Memory):
            self._buf = bytearray(memoryview(data._buf)[: data._size])
            self._size = data._size
            if media_type is None:
                self._media_type = data._media_type
            return

        if isinstance(data, int):
            if data < 0:
                raise ValueError(
                    f"Memory(int) capacity must be >= 0, got {data!r}"
                )
            self._buf = bytearray(data)
            self._size = 0
            return

        if isinstance(data, (bytes, bytearray, memoryview)):
            mv = memoryview(data)
            if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
                mv = mv.cast("B")
            self._buf = bytearray(mv if mv.c_contiguous else bytes(mv))
            self._size = len(self._buf)
            return

        raise TypeError(
            f"Memory does not accept data of type {type(data).__name__!r}. "
            "Pass bytes / bytearray / memoryview / int (capacity) / Memory."
        )

    # ------------------------------------------------------------------
    # Holder primitives
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    @property
    def mtime(self) -> float:
        return self._mtime

    @property
    def media_type(self):
        return self._media_type

    @property
    def capacity(self) -> int:
        """Current allocated capacity (``len(bytearray)``)."""
        return len(self._buf)

    def read_mv(self, n: int, pos: int) -> memoryview:
        if pos < 0:
            raise ValueError(f"read_mv pos must be >= 0, got {pos!r}")
        size = self._size
        if pos >= size:
            return memoryview(b"")
        if n < 0:
            n = size - pos
        end = min(pos + max(0, n), size)
        if end <= pos:
            return memoryview(b"")
        return memoryview(self._buf)[pos:end]

    def write_mv(self, data: memoryview, pos: int) -> int:
        if pos < 0:
            raise ValueError(f"write_mv pos must be >= 0, got {pos!r}")
        if data.format != "B" or data.ndim != 1 or data.itemsize != 1:
            data = data.cast("B")
        if not data.c_contiguous:
            data = memoryview(bytes(data))
        n = len(data)
        if n == 0:
            return 0
        need = pos + n
        if need > len(self._buf):
            self.reserve(need)
        memoryview(self._buf)[pos:need] = data
        if need > self._size:
            self._size = need
        self._mtime = time.time()
        return n

    def reserve(self, n: int) -> None:
        if n < 0:
            raise ValueError(f"reserve size must be >= 0, got {n!r}")
        cur = len(self._buf)
        if n <= cur:
            return
        # 1.5× amortization so tight-loop appends don't reallocate
        # on every chunk.
        new_cap = max(n, int(cur * 1.5) + 1)
        self._buf.extend(b"\x00" * (new_cap - cur))

    def truncate(self, n: int) -> int:
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        if n != self._size:
            self._mtime = time.time()
        if n < self._size:
            self._size = n
            return n
        if n > self._size:
            self.reserve(n)
            # The reserved tail is already zero — bytearray.extend(b"\x00")
            # gives us zero-padding for free.
            self._size = n
        return n

    # ------------------------------------------------------------------
    # Direct bytearray accessors — for callers that want zero-copy
    # ------------------------------------------------------------------

    def memoryview(self) -> memoryview:
        """Memoryview over the visible payload (size-bounded)."""
        return memoryview(self._buf)[: self._size]

    def to_bytes(self) -> bytes:
        return bytes(self.memoryview())

    def clear(self) -> None:
        """Drop all bytes; reset capacity AND size to zero."""
        self._buf = bytearray()
        self._size = 0
        self._mtime = time.time()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Memory):
            return (
                self._size == other._size
                and self.memoryview() == other.memoryview()
            )
        if isinstance(other, (bytes, bytearray, memoryview)):
            return self.memoryview() == memoryview(other)
        return NotImplemented

    def __hash__(self) -> int:
        # Equality is value-based; hash by content. Mutable, so use
        # the bytes form (immutable snapshot) for the hash.
        return hash(self.to_bytes())

    def __repr__(self) -> str:
        return f"Memory(size={self._size}, capacity={len(self._buf)})"
