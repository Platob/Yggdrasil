"""Abstract byte holder — the substrate :class:`BytesIO` plays on top of.

A :class:`Holder` is "a thing that holds N bytes addressable by
position." Two concrete shapes:

- :class:`Memory` — a :class:`bytearray` we manage directly. Every
  read/write hits memory; ``reserve`` grows the bytearray; ``truncate``
  resizes the visible slice.
- :class:`yggdrasil.io.fs.Path` — a path-bound holder. Local paths
  back the storage with a long-lived :func:`os.open` fd; remote paths
  with a transaction :class:`BytesIO` flushed via the path's
  whole-file ``_pwrite`` on commit.

The four abstract primitives are :meth:`read_mv`, :meth:`write_mv`,
:meth:`reserve`, :meth:`truncate` and the :attr:`size` property.
Everything else (:meth:`read_bytes` / :meth:`write_bytes` /
:meth:`read_text` / :meth:`write_text` / :meth:`read_local_path` /
:meth:`write_local_path`) builds on those, so a new backend gets
the full convenience surface for free.
"""

from __future__ import annotations

import os
import pathlib
from abc import abstractmethod
from typing import Union

from yggdrasil.disposable import Disposable

from .io_stats import IOStats


__all__ = ["Holder"]


PathLike = Union[str, "os.PathLike[str]", pathlib.PurePath]


_COPY_CHUNK = 4 * 1024 * 1024


class Holder(Disposable):
    """Position-addressable byte holder + :class:`Disposable` lifecycle.

    A holder IS a Disposable: it can be opened, closed, used in a
    ``with`` block, marked dirty / clean. Concrete subclasses
    (:class:`yggdrasil.io.memory.Memory`,
    :class:`yggdrasil.io.fs.Path`) plug acquire/release into the
    Disposable hooks so :class:`BytesIO` can compose with either
    one through the same API and seamlessly swap (e.g. on spill)
    without branching at every call site.

    Subclasses implement five primitives:

    - :meth:`read_mv(n, pos)` — slice ``n`` bytes from ``pos`` as a
      :class:`memoryview`. ``n < 0`` means "to end of holder."
    - :meth:`write_mv(data, pos)` — splice ``data`` at ``pos``,
      growing the holder if needed. Returns bytes written.
    - :meth:`reserve(n)` — pre-grow the underlying capacity to *at
      least* ``n`` bytes without changing the visible :attr:`size`.
    - :meth:`truncate(n)` — set the visible :attr:`size` to ``n``.
      Shrinks drop the tail; extends zero-pad.
    - :attr:`size` — current visible size in bytes.
    """

    # ------------------------------------------------------------------
    # Abstract primitives
    # ------------------------------------------------------------------

    @abstractmethod
    def read_mv(self, n: int, pos: int) -> memoryview:
        """Return a memoryview over ``n`` bytes starting at ``pos``.

        ``n < 0`` is interpreted as "from ``pos`` to end of holder."
        ``pos`` past the end yields a zero-length view. The view's
        lifetime tracks the underlying storage; subclasses MAY return
        a view that backs onto a transient buffer (e.g. a remote
        download) — in that case the caller must consume / copy the
        view before any other I/O against the holder.
        """

    @abstractmethod
    def write_mv(self, data: memoryview, pos: int) -> int:
        """Splice ``data`` at ``pos``. Returns bytes actually written.

        Grows the holder when ``pos + len(data) > size``. The caller
        is responsible for handing in a 1-byte memoryview shape;
        subclasses may cast/normalize internally for portability.
        """

    @abstractmethod
    def reserve(self, n: int) -> None:
        """Pre-grow capacity to *at least* ``n`` bytes.

        Capacity-only — does NOT change :attr:`size`. Idempotent
        when capacity ≥ ``n`` already. Subclasses with no growable
        capacity layer may treat this as a no-op.
        """

    @abstractmethod
    def truncate(self, n: int) -> int:
        """Set the visible :attr:`size` to exactly ``n`` bytes.

        Shrinks drop the tail; extends zero-pad. Returns ``n``.
        """

    @property
    @abstractmethod
    def size(self) -> int:
        """Current visible size in bytes."""

    # ------------------------------------------------------------------
    # IOStats — size / mtime / media_type triple
    # ------------------------------------------------------------------

    @property
    def mtime(self) -> float:
        """Last modification time. ``0.0`` when no meaningful mtime is available."""
        return 0.0

    @property
    def media_type(self):
        """:class:`MediaType` for this holder, or ``None`` if unknown."""
        return None

    def stats(self) -> IOStats:
        """Snapshot ``size`` / ``mtime`` / ``media_type`` as an :class:`IOStats`.

        Default impl reads :attr:`size` / :attr:`mtime` / :attr:`media_type`.
        Backends with cheaper batched probes (e.g. one ``stat`` call
        for both size and mtime) can override.
        """
        return IOStats(
            size=int(self.size),
            mtime=float(self.mtime),
            media_type=self.media_type,
        )

    # ------------------------------------------------------------------
    # Bytes / text convenience surface — built on the abstract primitives
    # ------------------------------------------------------------------

    def read_bytes(self, n: int = -1, pos: int = 0) -> bytes:
        """Read ``n`` bytes at ``pos`` as :class:`bytes`.

        ``n < 0`` reads to end of holder. Always returns a fresh
        :class:`bytes` so the caller can hold onto it past further
        I/O against the holder.
        """
        return bytes(self.read_mv(n, pos))

    def write_bytes(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int = 0,
    ) -> int:
        """Splice bytes-like ``data`` at ``pos``. Returns bytes written."""
        mv = memoryview(data)
        if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
            mv = mv.cast("B")
        if not mv.c_contiguous:
            mv = memoryview(bytes(mv))
        return self.write_mv(mv, pos)

    def read_text(
        self,
        encoding: str = "utf-8",
        errors: str = "strict",
        *,
        n: int = -1,
        pos: int = 0,
    ) -> str:
        """Decode ``n`` bytes at ``pos`` as text."""
        return self.read_bytes(n, pos).decode(encoding, errors=errors)

    def write_text(
        self,
        text: str,
        encoding: str = "utf-8",
        errors: str = "strict",
        *,
        pos: int = 0,
    ) -> int:
        """Encode ``text`` and splice at ``pos``. Returns bytes written."""
        return self.write_bytes(
            text.encode(encoding, errors=errors), pos,
        )

    # ------------------------------------------------------------------
    # Local-path bridge — bytes ↔ files on the local filesystem
    # ------------------------------------------------------------------

    def read_local_path(
        self,
        path: PathLike,
        *,
        pos: int = 0,
        n: int = -1,
        chunk_size: int = _COPY_CHUNK,
    ) -> int:
        """Load ``path``'s bytes into this holder at ``pos``.

        ``n < 0`` reads the whole file; ``n >= 0`` caps the number of
        bytes pulled from the source at *n*. Streams the source file
        in ``chunk_size`` slices so a large file doesn't materialize
        into memory. Returns the number of bytes spliced in.
        """
        if pos < 0:
            raise ValueError("read_local_path pos must be >= 0")
        os_path = os.fspath(path)
        total = 0
        cursor = pos
        remaining = n if n >= 0 else None
        with open(os_path, "rb") as fh:
            while True:
                want = chunk_size
                if remaining is not None:
                    if remaining <= 0:
                        break
                    want = min(want, remaining)
                chunk = fh.read(want)
                if not chunk:
                    break
                written = self.write_mv(memoryview(chunk), cursor)
                if written == 0:
                    break
                cursor += written
                total += written
                if remaining is not None:
                    remaining -= written
        return total

    def write_local_path(
        self,
        path: PathLike,
        *,
        pos: int = 0,
        n: int = -1,
        chunk_size: int = _COPY_CHUNK,
    ) -> int:
        """Drain bytes from ``pos`` into the local file ``path``.

        ``n < 0`` drains to end of holder. Returns bytes written to
        the file. Auto-creates the parent directory so the caller
        doesn't have to pre-mkdir.
        """
        if pos < 0:
            raise ValueError("write_local_path pos must be >= 0")
        os_path = os.fspath(path)
        parent = os.path.dirname(os_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)

        size = self.size
        if n < 0:
            remaining = max(0, size - pos)
        else:
            remaining = max(0, min(n, size - pos))

        cursor = pos
        total = 0
        with open(os_path, "wb") as fh:
            while remaining > 0:
                want = min(chunk_size, remaining)
                mv = self.read_mv(want, cursor)
                written = fh.write(mv)
                if written is None:
                    written = len(mv)
                if written == 0:
                    break
                cursor += written
                remaining -= written
                total += written
        return total

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __bytes__(self) -> bytes:
        return self.read_bytes()
