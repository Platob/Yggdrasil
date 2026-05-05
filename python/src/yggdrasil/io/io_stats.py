"""Global I/O stats — the canonical ``size`` / ``mtime`` / ``media_type`` triple.

Every :class:`yggdrasil.io.holder.Holder` (concretely :class:`Memory`
and :class:`yggdrasil.io.fs.Path`) exposes :meth:`stats` returning an
:class:`IOStats`. It's the single shape downstream code reads when it
needs "how big, how fresh, what is it" without caring whether the
backing is a file, an S3 object, a buffer in memory, or something
else.

Distinct from :class:`yggdrasil.io.path_stat.PathStats` (the stat-like
``kind`` / ``mode`` / ``size`` / ``mtime`` quad backends synthesize
from native ``stat()``): :class:`IOStats` is *content*-focused.
``media_type`` is the most useful piece for I/O dispatch — codec
selection, format inference, content-negotiation — and PathStats
deliberately doesn't carry it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from yggdrasil.io.enums import MediaType


__all__ = ["IOStats"]


@dataclass(slots=True)
class IOStats:
    """Size / mtime / media-type triple for any byte holder.

    All three fields are best-effort:

    - ``size`` — visible byte count. Always populated; ``0`` is a
      legitimate value for a freshly-created empty holder.
    - ``mtime`` — last modification time as a Unix timestamp. ``0.0``
      when the backing has no meaningful mtime (memory holders,
      newly-minted spills) — callers that need "now" should fall
      back to :func:`time.time` themselves.
    - ``media_type`` — :class:`MediaType` inferred from the holder's
      identity (URL extension, registered mime, sniffed magic
      bytes). ``None`` when no honest answer is available; never
      guess :class:`MimeTypes.OCTET_STREAM` here — let the caller
      decide.
    """

    size: int = 0
    mtime: float = 0.0
    media_type: "Optional[MediaType]" = None

    # ------------------------------------------------------------------
    # Convenience views — keep IOStats interchangeable with the bare
    # tuple shape callers used to pass around.
    # ------------------------------------------------------------------

    @property
    def mtime_or_now(self) -> float:
        """``mtime`` if set, else :func:`time.time` — for callers that
        always want a non-zero timestamp without sprinkling fallbacks."""
        return self.mtime if self.mtime else time.time()

    @property
    def has_media_type(self) -> bool:
        return self.media_type is not None

    def with_(
        self,
        *,
        size: Optional[int] = None,
        mtime: Optional[float] = None,
        media_type: Any = ...,
        copy: bool = False,
    ) -> "IOStats":
        """Mutate in place (default) or return a copy with the given fields set.

        ``media_type`` uses the ``...`` sentinel so callers can
        explicitly clear it by passing ``media_type=None``.
        """
        if copy:
            return IOStats(
                size=self.size if size is None else size,
                mtime=self.mtime if mtime is None else mtime,
                media_type=(
                    self.media_type if media_type is ... else media_type
                ),
            )
        if size is not None:
            self.size = size
        if mtime is not None:
            self.mtime = mtime
        if media_type is not ...:
            self.media_type = media_type
        return self

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __iter__(self):
        yield self.size
        yield self.mtime
        yield self.media_type

    def __repr__(self) -> str:
        mt = self.media_type
        mt_repr = repr(mt) if mt is not None else "None"
        return f"IOStats(size={self.size}, mtime={self.mtime!r}, media_type={mt_repr})"
