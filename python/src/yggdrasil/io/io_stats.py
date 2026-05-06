"""Unified I/O stats — backend ``stat`` quad plus content-level ``media_type``.

Every :class:`yggdrasil.io.holder.Holder` (concretely :class:`Memory`
and :class:`yggdrasil.io.fs.Path`) exposes :meth:`stats` returning an
:class:`IOStats`. It's the single shape downstream code reads when it
needs "what kind, how big, how fresh, what is it" without caring
whether the backing is a local file, an S3 object, an in-memory
buffer, or something else.

The ``size`` / ``mtime`` / ``kind`` / ``mode`` quad mirrors
:class:`os.stat_result` so existing callers can keep using the
familiar ``st_size`` / ``st_mtime`` / ``st_mode`` aliases or the
positional tuple shape. ``media_type`` extends the picture with the
single most useful piece for content-level dispatch — codec
selection, format inference, content-negotiation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from yggdrasil.io.enums import MediaType


__all__ = ["IOStats", "IOKind"]


class IOKind(IntEnum):
    """What a backend reports a path/holder entry is.

    Integer-backed so the value compares cheaply and round-trips
    through binary protocols / cache keys without string overhead.
    """

    MISSING = 0
    FILE = 1
    DIRECTORY = 2
    SYMLINK = 3
    SOCKET = 4
    FIFO = 5
    CHAR_DEVICE = 6
    BLOCK_DEVICE = 7


@dataclass(slots=True)
class IOStats:
    """Stat-like quad (``size`` / ``mtime`` / ``kind`` / ``mode``) + ``media_type``.

    All fields are best-effort:

    - ``size`` — visible byte count. ``0`` is a legitimate value for a
      freshly-created empty holder or a directory.
    - ``mtime`` — last modification time as a Unix timestamp. ``0.0``
      when the backing has no meaningful mtime (memory holders,
      newly-minted spills) — callers that need "now" should use
      :attr:`mtime_or_now`.
    - ``kind`` — :class:`IOKind` enum classifying the entry.
      Defaults to :attr:`IOKind.MISSING` so an "empty" stats object
      reads as "nothing here".
    - ``mode`` — POSIX permission bits, ``0`` when the backend has no
      meaningful concept of mode (S3, Databricks REST).
    - ``media_type`` — :class:`MediaType` inferred from the holder's
      identity (URL extension, registered mime, sniffed magic
      bytes). ``None`` when no honest answer is available; never
      guess :class:`MimeTypes.OCTET_STREAM` here — let the caller
      decide.

    Backends with richer metadata (ETag, content-type, owner…) should
    subclass and extend rather than cram extras into ``mode``.
    """

    size: int = 0
    mtime: float = 0.0
    kind: IOKind = IOKind.MISSING
    mode: int = 0
    media_type: "Optional[MediaType]" = None

    # ------------------------------------------------------------------
    # ``os.stat_result`` compatibility — drop-in for legacy callers
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Any:
        # Mirrors the positional layout of ``os.stat_result``:
        # (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime).
        return (self.mode, 0, 0, 0, 0, 0, self.size, 0, self.mtime, 0)[idx]

    @property
    def st_size(self) -> int:
        return self.size

    @property
    def st_mtime(self) -> float:
        return self.mtime

    @property
    def st_mode(self) -> int:
        return self.mode

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    @property
    def mtime_or_now(self) -> float:
        """``mtime`` if set, else :func:`time.time` — for callers that
        always want a non-zero timestamp without sprinkling fallbacks."""
        return self.mtime if self.mtime else time.time()

    @property
    def has_media_type(self) -> bool:
        return self.media_type is not None

    @property
    def exists(self) -> bool:
        return self.kind != IOKind.MISSING

    @property
    def is_file(self) -> bool:
        return self.kind == IOKind.FILE

    @property
    def is_dir(self) -> bool:
        return self.kind == IOKind.DIRECTORY

    def copy(
        self,
        *,
        size: Any = ...,
        mtime: Any = ...,
        kind: Any = ...,
        mode: Any = ...,
        media_type: Any = ...,
    ) -> "IOStats":
        """Return a fresh :class:`IOStats` with selected fields overridden.

        Each kwarg uses the ``...`` sentinel so the caller can pass
        ``None`` (or any other value) to override without colliding
        with "leave unchanged". Any field left at ``...`` carries the
        value over from ``self``.
        """
        return IOStats(
            size=self.size if size is ... else size,
            mtime=self.mtime if mtime is ... else mtime,
            kind=self.kind if kind is ... else kind,
            mode=self.mode if mode is ... else mode,
            media_type=self.media_type if media_type is ... else media_type,
        )

    def with_(
        self,
        *,
        size: Any = ...,
        mtime: Any = ...,
        kind: Any = ...,
        mode: Any = ...,
        media_type: Any = ...,
        inplace: bool = False,
    ) -> "IOStats":
        """Return a stats object with the given fields overridden.

        ``inplace=False`` (default) returns a fresh :class:`IOStats`
        via :meth:`copy`. ``inplace=True`` mutates ``self`` and
        returns ``self`` for chaining. Each field kwarg uses the
        ``...`` sentinel — pass any other value (including ``None``)
        to override.
        """
        if not inplace:
            return self.copy(
                size=size,
                mtime=mtime,
                kind=kind,
                mode=mode,
                media_type=media_type,
            )
        if size is not ...:
            self.size = size
        if mtime is not ...:
            self.mtime = mtime
        if kind is not ...:
            self.kind = kind
        if mode is not ...:
            self.mode = mode
        if media_type is not ...:
            self.media_type = media_type
        return self

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __iter__(self):
        yield self.size
        yield self.mtime
        yield self.kind
        yield self.mode
        yield self.media_type

    def __repr__(self) -> str:
        mt = self.media_type
        mt_repr = repr(mt) if mt is not None else "None"
        return (
            f"IOStats(size={self.size}, mtime={self.mtime!r}, "
            f"kind={self.kind.name}, mode={self.mode!r}, "
            f"media_type={mt_repr})"
        )
