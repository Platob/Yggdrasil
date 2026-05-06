"""Fully in-memory :class:`Holder` backed by a :class:`bytearray`.

:class:`Memory` is the simplest concrete :class:`Holder`: every
read/write hits an internally-managed :class:`bytearray`. No fd, no
spill file, no transaction layer. Capacity grows with the standard
1.5× amortization pattern so back-to-back appends stay cheap.

Composes with :class:`yggdrasil.io.buffer.BytesIO`: a memory-mode
``BytesIO`` is conceptually a Memory holder plus a cursor and the
Tabular read/write surface.

Metadata model
--------------
All IO metadata (visible size, mtime, media-type) lives on a single
mutable :class:`IOStats` instance — :meth:`stat`. Writes mutate it
in place via :meth:`Holder._touch_stat`.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from yggdrasil.disposable import Disposable
from yggdrasil.io.io_stats import IOKind, IOStats

from .holder import Holder


__all__ = ["Memory"]


class Memory(Holder):
    """Fully memory-resident byte holder.

    Construction shapes (in addition to those inherited from
    :class:`Holder`):

    - ``Memory()``          — empty, zero capacity.
    - ``Memory(int n)``     — empty, capacity ``n`` reserved.
    - ``Memory(other_mem)`` — deep copy of another Memory.

    Visible :attr:`size` and underlying ``bytearray`` capacity are
    tracked separately. ``reserve(n)`` grows capacity without moving
    :attr:`size`; :meth:`truncate` moves :attr:`size`, zero-padding
    on extend.

    Implements the five :class:`Holder` primitives — :meth:`_read_mv`,
    :meth:`_write_mv`, :meth:`reserve`, :meth:`truncate`, :meth:`clear`
    — plus :attr:`size` and the lazy :meth:`stat`. Everything else
    (positional normalization, bounds checking, pre-grow via
    :meth:`resize`, ``mark_dirty``, bytes/text convenience,
    append-at-end ``pos = -1`` semantics) comes from the base class.
    """

    __slots__ = ("_buf",)

    def __init__(
        self,
        data: Any = None,
        **kwargs,
    ) -> None:
        self._buf: bytearray = bytearray()

        # ``int`` is Memory-specific (reserve capacity, no payload);
        # everything else routes through Holder's _init_from_* dispatch.
        if isinstance(data, int) and not isinstance(data, bool):
            if data < 0:
                raise ValueError(
                    f"Memory(int) capacity must be >= 0, got {data!r}"
                )
            self._buf = bytearray(data)
            data = None

        super().__init__(data, **kwargs)



    # ``_acquire`` / ``_release`` inherited as no-ops from Disposable.
    # Memory's bytes survive a close() — the bytearray is owned by
    # Python and freed on GC. Use clear() to drop the payload.

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
        returned Memory does not free ``buf``.

        Used by :class:`BytesIO` as the in-memory ``_owner``: the
        same bytearray BytesIO mutates directly is exposed through
        the :class:`Holder` interface for backend-agnostic callers.
        """
        m = cls.__new__(cls)
        Disposable.__init__(m)
        m._url = None
        m.temporary = False
        m._buf = buf
        m._stat = IOStats(
            size=len(buf) if size is None else int(size),
            mtime=time.time(),
            kind=IOKind.MEMORY,
            media_type=media_type,
        )
        m._acquired = True
        return m

    # ------------------------------------------------------------------
    # Backing-shape predicates
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

    # ------------------------------------------------------------------
    # Holder primitives
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self.stat().size

    @property
    def capacity(self) -> int:
        """Current allocated capacity (``len(bytearray)``)."""
        return len(self._buf)

    def _read_mv(self, n: int, pos: int) -> memoryview:
        # Holder.read_mv has already normalized pos and bounded n;
        # 0 <= pos <= size and 0 <= n <= size - pos.
        return memoryview(self._buf)[pos : pos + n]

    def _write_mv(self, data: memoryview, pos: int) -> int:
        """Splice bytes at ``pos`` — size already grown by :meth:`write_mv`.

        :class:`Holder.write_mv` has pre-grown the visible size via
        :meth:`resize` (which delegates to :meth:`truncate` here),
        which in turn called :meth:`reserve` to grow the underlying
        bytearray. So at this point ``len(self._buf)`` is already
        guaranteed ≥ ``pos + len(data)`` and we just lay bytes down.

        Stat-cache mutation (size / mtime) lives in :meth:`write_mv`
        via :meth:`_touch_stat`, so this method does pure splice.
        """
        n = len(data)
        if n == 0:
            return 0
        memoryview(self._buf)[pos : pos + n] = data
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
        stats = self.stat()
        if n == stats.size:
            return n
        if n > len(self._buf):
            # bytearray.extend(b"\x00"*…) gives zero-padding for free,
            # so reserve() does both the capacity grow and the zero-fill.
            self.reserve(n)
        stats.size = n
        stats.mtime = time.time()
        return n

    def _clear(self) -> None:
        """Drop all bytes; reset capacity AND size to zero."""
        self._buf = bytearray()
        stats = self.stat()
        stats.size = 0
        stats.mtime = time.time()

    # ------------------------------------------------------------------
    # Memory-specific zero-copy accessors
    # ------------------------------------------------------------------

    def memoryview(self) -> memoryview:
        """Memoryview over the visible payload (size-bounded).

        Override of :meth:`Holder.memoryview` that aliases the
        bytearray directly — no copy, no per-byte dispatch.
        """
        return memoryview(self._buf)[: self.stat().size]

    def to_bytes(self) -> bytes:
        return bytes(self.memoryview())
