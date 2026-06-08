"""Minimal byte-buffer for pickle-module internal use.

The yggdrasil :class:`~yggdrasil.io.BytesIO` is a full IO object with
singleton dispatch, URL routing, format detection, stat tracking, and
spill-to-disk support — all entirely unnecessary for the scratch buffers
the serializer creates and immediately consumes.  :class:`_ScratchBuf`
wraps :class:`io.BytesIO` (stdlib) and exposes exactly the surface that
``header.py`` / ``serialized.py`` / ``collections.py`` / ``serde.py`` use.
This cuts per-object allocation overhead by ~10× compared with the full
:class:`~yggdrasil.io.IO`.

The interface is intentionally kept identical to the yggdrasil IO
subset used in the pickle module — no new abstractions, no leaking outside
this package.
"""
from __future__ import annotations

import io as _stdlib_io

__all__ = ["_ScratchBuf"]


class _ScratchBuf:
    """Lightweight byte accumulator for pickle-module internals."""

    __slots__ = ("_buf",)

    def __init__(self, data: bytes | bytearray | memoryview = b"") -> None:
        self._buf: _stdlib_io.BytesIO = _stdlib_io.BytesIO(
            bytes(data) if not isinstance(data, bytes) else data
        )

    # ------------------------------------------------------------------
    # write surface
    # ------------------------------------------------------------------

    def write(self, b: bytes | bytearray | memoryview) -> int:
        if isinstance(b, memoryview):
            # stdlib BytesIO.write accepts memoryview directly.
            return self._buf.write(b)
        return self._buf.write(b)

    # ------------------------------------------------------------------
    # read surface (cursor-based)
    # ------------------------------------------------------------------

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def tell(self) -> int:
        return self._buf.tell()

    def seek(self, pos: int, whence: int = 0) -> int:
        return self._buf.seek(pos, whence)

    # ------------------------------------------------------------------
    # positional read (no cursor movement)
    # ------------------------------------------------------------------

    def pread(self, size: int, *, pos: int) -> bytes:
        return self._buf.getvalue()[pos : pos + size]

    # ------------------------------------------------------------------
    # bulk-access (cursorless)
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        return self._buf.getvalue()

    def getvalue(self) -> bytes:
        return self._buf.getvalue()

    def memoryview(self) -> memoryview:
        # getbuffer() on stdlib BytesIO returns a zero-copy memoryview of
        # the *writable* underlying buffer — no allocation, no copy.
        return self._buf.getbuffer()

    # ------------------------------------------------------------------
    # slice view (positional, like yggdrasil IO.view)
    # ------------------------------------------------------------------

    def view(self, *, pos: int = 0, size: int = -1) -> "_ScratchBuf":
        data = self._buf.getvalue()
        if size < 0:
            return _ScratchBuf(data[pos:])
        return _ScratchBuf(data[pos : pos + size])
