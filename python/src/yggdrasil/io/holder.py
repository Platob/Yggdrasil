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
Everything else (:meth:`pread` / :meth:`pwrite` / :meth:`read_bytes` /
:meth:`write_bytes` / :meth:`read_text` / :meth:`write_text` /
:meth:`read_local_path` / :meth:`write_local_path`) builds on those,
so a new backend gets the full convenience surface for free.

:class:`yggdrasil.io.buffer.BytesIO` keeps a single ``_holder: Holder``
slot and routes every cursorless I/O op straight through
:meth:`pread` / :meth:`pwrite` / :meth:`truncate` / :attr:`size` —
the holder mutates from :class:`Memory` to :class:`Path` on spill
without any change to the call sites.
"""

from __future__ import annotations

import os
import pathlib
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Union

from yggdrasil.disposable import Disposable

from .io_stats import IOStats


if TYPE_CHECKING:
    from yggdrasil.io.enums import Mode


__all__ = ["Holder"]


PathLike = Union[str, "os.PathLike[str]", pathlib.PurePath]


_COPY_CHUNK = 4 * 1024 * 1024


# Stdlib ``open()`` mode-string lookup for the standard
# :class:`Mode` values that have an OS-mode counterpart. Used by
# :meth:`Holder.open` to translate a :class:`Mode` argument into a
# string ``acquire_io`` understands.
_MODE_TO_OS_MODE = {
    "auto": "rb+",
    "read_only": "rb",
    "overwrite": "wb+",
    "append": "ab+",
    "truncate": "wb+",
    "error_if_exists": "xb+",
}


def _resolve_mode_string(mode: "Mode | str | None") -> str:
    """Translate *mode* (Mode enum, alias, or os-style string) → os-mode str."""
    if mode is None:
        return "rb+"
    if isinstance(mode, str) and mode and not _MODE_TO_OS_MODE.get(mode.lower()):
        # Already looks like an OS-mode string (e.g. "rb+", "wb").
        # Don't re-normalize — preserve flag combinations.
        if all(c in "rwaxbt+" for c in mode):
            return mode
    from yggdrasil.io.enums import Mode  # avoid import cycle
    parsed = Mode.from_(mode)
    return _MODE_TO_OS_MODE.get(parsed.value, "rb+")


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
    # IOStats — the canonical metadata holder
    # ------------------------------------------------------------------
    #
    # Every concrete :class:`Holder` keeps a single mutable
    # :class:`IOStats` instance. Writes mutate it in place
    # (``stats.size = new_size``, ``stats.mtime = time.time()``);
    # readers either pin :meth:`stats` and observe the live values, or
    # use the convenience properties below which read straight off the
    # same object. ``size`` / ``mtime`` / ``media_type`` are no longer
    # stored on separate slots — :class:`IOStats` is the holder.

    @abstractmethod
    def stat(self) -> IOStats:
        """The mutable :class:`IOStats` carrying this holder's metadata.

        Concrete holders return the *same* instance for the holder's
        lifetime; callers can pin it to observe live size / mtime /
        media_type updates as writes land.
        """

    @property
    def mtime(self) -> float:
        """Convenience accessor — same as ``self.stat().mtime``."""
        return self.stat().mtime

    @property
    def media_type(self):
        """Convenience accessor — same as ``self.stat().media_type``."""
        return self.stat().media_type

    # ------------------------------------------------------------------
    # Per-open lifecycle — Path overrides; Memory and other always-live
    # holders inherit no-ops so :class:`BytesIO` can call them blind.
    # ------------------------------------------------------------------

    def open(self, mode: "Mode | str | None" = None) -> "Holder":
        """Open the holder for IO at *mode*; returns ``self``.

        Wrapper that drives :class:`Disposable` open + :meth:`acquire_io`
        in one call. ``mode`` accepts a :class:`yggdrasil.io.enums.Mode`,
        a stdlib :func:`open` mode string (``"rb"``, ``"rb+"``,
        ``"wb"``, …), or ``None`` for the default ``"rb+"``. Memory
        holders ignore the mode (no separate IO state); :class:`Path`
        holders translate it into the fd / transaction-buffer
        acquire.
        """
        if not self._acquired:
            Disposable.open(self)
        os_mode = _resolve_mode_string(mode)
        self.acquire_io(os_mode)
        return self

    def acquire_io(self, mode: "str | None" = None) -> "Holder":
        """Open per-open IO state. Default no-op; returns ``self``.

        :class:`yggdrasil.io.fs.Path` overrides to mint a transaction
        buffer (remote) or :func:`os.open` an fd (local). Memory-style
        holders have no separate per-open state, so this is a no-op.
        """
        del mode
        return self

    def close_io(self) -> None:
        """Release per-open IO state. Default no-op.

        Mirrors :meth:`acquire_io`: subclasses with per-open state
        flush + tear it down here.
        """

    def flush(self) -> None:
        """Push buffered writes to the durable backing. Default no-op."""

    @property
    def dirty(self) -> bool:
        """True when there are uncommitted writes. Default ``False``."""
        return False

    @property
    def is_local(self) -> bool:
        """True for in-process / local-fs holders. Default ``True``."""
        return True

    # ------------------------------------------------------------------
    # Cursorless I/O — the canonical surface :class:`BytesIO` consumes
    # ------------------------------------------------------------------

    def pread(self, n: int, pos: int) -> bytes:
        """Positional read. Returns at most ``n`` bytes at *pos*.

        ``n < 0`` reads to end of holder. Returns a fresh
        :class:`bytes` so callers can hold onto it past further I/O
        against the holder. Default impl wraps :meth:`read_mv`.
        """
        return bytes(self.read_mv(n, pos))

    def pwrite(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
    ) -> int:
        """Positional write. Returns bytes actually written.

        Default impl normalises bytes-like ``data`` to a 1-D unsigned-
        byte memoryview and forwards to :meth:`write_mv`.
        """
        mv = memoryview(data)
        if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
            mv = mv.cast("B")
        if not mv.c_contiguous:
            mv = memoryview(bytes(mv))
        return self.write_mv(mv, pos)

    def memoryview(self) -> memoryview:
        """View over the holder's visible bytes.

        Default impl materialises via :meth:`read_mv` so every backend
        gets a consistent shape; engines with cheaper paths (mmap on
        local fd, alias of an in-memory bytearray) override.
        """
        return self.read_mv(-1, 0)

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
