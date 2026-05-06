"""Fully in-memory :class:`Holder` backed by a :class:`bytearray`.

:class:`Memory` is the simplest concrete :class:`Holder`: every
read/write hits an internally-managed :class:`bytearray`. No fd, no
spill file, no transaction layer. Capacity grows with the standard
1.5× amortization pattern so back-to-back appends stay cheap.

Composes with :class:`yggdrasil.io.buffer.BytesIO`: a memory-mode
``BytesIO`` is conceptually a Memory holder plus a cursor and the
TabularIO read/write surface.

Metadata model
--------------
All IO metadata (visible size, mtime, media-type) lives on a single
mutable :class:`IOStats` instance — :attr:`stats`. Writes mutate it
in place; readers are free to call :meth:`stats` and pin the same
object (it's never replaced for the holder's lifetime).
"""

from __future__ import annotations

import time
from typing import Any, Optional, Union

from yggdrasil.disposable import Disposable
from yggdrasil.io.io_stats import IOKind, IOStats

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

    __slots__ = ("_buf", "_stats")

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
        auto_open: bool = True,
    ) -> None:
        Disposable.__init__(self)
        self._stats: IOStats = IOStats(
            mtime=time.time(),
            kind=IOKind.SOCKET,
            media_type=media_type,
        )

        if data is None:
            self._buf: bytearray = bytearray()
        elif isinstance(data, Memory):
            self._buf = bytearray(memoryview(data._buf)[: data._stats.size])
            self._stats.size = data._stats.size
            if media_type is None:
                self._stats.media_type = data._stats.media_type
        elif isinstance(data, int):
            if data < 0:
                raise ValueError(
                    f"Memory(int) capacity must be >= 0, got {data!r}"
                )
            self._buf = bytearray(data)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            mv = memoryview(data)
            if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
                mv = mv.cast("B")
            self._buf = bytearray(mv if mv.c_contiguous else bytes(mv))
            self._stats.size = len(self._buf)
        else:
            raise TypeError(
                f"Memory does not accept data of type {type(data).__name__!r}. "
                "Pass bytes / bytearray / memoryview / int (capacity) / Memory."
            )

        if auto_open:
            self.open()

    # ``_acquire`` / ``_release`` are inherited as no-ops from
    # :class:`Disposable`. Memory's bytes survive a ``close()`` —
    # the bytearray is owned by Python and freed on GC. Use
    # :meth:`clear` to drop the payload explicitly.

    @classmethod
    def view(
        cls,
        buf: bytearray,
        size: Optional[int] = None,
        *,
        media_type: Any = None,
    ) -> "Memory":
        """Construct a :class:`Memory` aliasing an existing bytearray.

        Zero-copy: ``buf`` is shared with the returned Memory.
        Mutations through :meth:`write_mv` / :meth:`truncate` /
        :meth:`reserve` propagate to ``buf`` directly. Closing the
        returned Memory does not free ``buf`` — the bytearray's
        lifetime is the caller's.

        ``size`` defaults to ``len(buf)``; pass an explicit size when
        the visible payload is shorter than the underlying capacity
        (e.g. when the caller pre-allocated extra bytes).

        Used by :class:`BytesIO` as the in-memory ``_owner``: the
        same bytearray BytesIO mutates directly is exposed through
        the :class:`Holder` interface for backend-agnostic callers.
        """
        m = cls.__new__(cls)
        Disposable.__init__(m)
        m._buf = buf
        m._stats = IOStats(
            size=len(buf) if size is None else int(size),
            mtime=time.time(),
            kind=IOKind.SOCKET,
            media_type=media_type,
        )
        m._acquired = True
        return m

    # ------------------------------------------------------------------
    # Holder primitives
    # ------------------------------------------------------------------

    @property
    def is_memory(self) -> bool:
        return True

    @property
    def is_local_path(self) -> bool:
        return False

    @property
    def is_remote_path(self) -> bool:
        return False

    @property
    def size(self) -> int:
        return self._stats.size

    @property
    def mtime(self) -> float:
        return self._stats.mtime

    @property
    def media_type(self):
        return self._stats.media_type

    def stat(self) -> IOStats:
        """The mutable :class:`IOStats` carrying this holder's metadata.

        Always returns the same instance for the holder's lifetime —
        callers can pin it to observe live size/mtime updates.
        """
        return self._stats

    @property
    def capacity(self) -> int:
        """Current allocated capacity (``len(bytearray)``)."""
        return len(self._buf)

    def read_mv(self, n: int, pos: int) -> memoryview:
        if pos < 0:
            raise ValueError(f"read_mv pos must be >= 0, got {pos!r}")
        size = self._stats.size
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
        stats = self._stats
        if need > stats.size:
            stats.size = need
        stats.mtime = time.time()
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
        stats = self._stats
        if n != stats.size:
            stats.mtime = time.time()
        if n < stats.size:
            stats.size = n
            return n
        if n > stats.size:
            self.reserve(n)
            # The reserved tail is already zero — bytearray.extend(b"\x00")
            # gives us zero-padding for free.
            stats.size = n
        return n

    # ------------------------------------------------------------------
    # Direct bytearray accessors — for callers that want zero-copy
    # ------------------------------------------------------------------

    def memoryview(self) -> memoryview:
        """Memoryview over the visible payload (size-bounded)."""
        return memoryview(self._buf)[: self._stats.size]

    def to_bytes(self) -> bytes:
        return bytes(self.memoryview())

    def clear(self) -> None:
        """Drop all bytes; reset capacity AND size to zero."""
        self._buf = bytearray()
        self._stats.size = 0
        self._stats.mtime = time.time()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Memory):
            return (
                self._stats.size == other._stats.size
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
        return f"Memory(size={self._stats.size}, capacity={len(self._buf)})"
