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
import datetime

import time
import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from yggdrasil.data.cast import any_to_datetime
from yggdrasil.enums.io_kind import IOKind

if TYPE_CHECKING:
    from yggdrasil.enums import MediaType


__all__ = ["IOStats", "IOKind", "TimeLike"]


TimeLike = Union[dt.datetime, dt.date, dt.timedelta, str, float, int]


@dataclass(slots=True, repr=False, eq=False)
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
    - ``metadata`` — free-form backend metadata as a flat
      ``dict[str, str]`` (S3 response + ``x-amz-meta-*`` headers,
      Databricks Files headers, ETag, version id, …). ``None`` when the
      backend exposes nothing extra; a single home for "everything else
      the backend told us" without subclassing :class:`IOStats`.

    Backends with richer metadata (ETag, content-type, owner…) populate
    :attr:`metadata` rather than cramming extras into ``mode``.
    """

    size: int = 0
    mtime: float = 0.0
    kind: IOKind = IOKind.MISSING
    mode: int = 0
    media_type: "Optional[MediaType]" = None
    metadata: "Optional[dict[str, str]]" = None

    # ------------------------------------------------------------------
    # ``os.stat_result`` compatibility — drop-in for legacy callers
    # ------------------------------------------------------------------

    def __repr__(self):
        dt_ = datetime.datetime.fromtimestamp(self.mtime, datetime.timezone.utc).isoformat()
        return f"<IOStats size={self.size} mtime={dt_!r} kind={self.kind.name} mode={self.mode} media_type={self.media_type!r}>"

    def __str__(self):
        return repr(self)

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
    def has_mtime(self) -> bool:
        """Whether the backing reported a real modification time.

        ``False`` for memory holders, freshly-minted spills, and any
        backend that doesn't expose mtime — the sentinel ``0.0``
        means "unknown", not "epoch".
        """
        return self.mtime != 0.0

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
    # mtime filtering
    # ------------------------------------------------------------------
    #
    # All mtime predicates follow one rule: an unknown timestamp
    # (``self.mtime == 0.0``) is **excluded** from every range
    # check. Returning ``False`` rather than guessing keeps the
    # sentinel honest — a memory holder is never "fresher than
    # yesterday" *or* "older than yesterday", it's simply unknown,
    # and callers that want to include it must check ``has_mtime``
    # explicitly.

    @staticmethod
    def normalize_timestamp(when: TimeLike | None, default: Any = ...) -> float:
        """Resolve a flexible time spec to a Unix timestamp.

        ``None`` propagates as ``None`` (caller uses it as "no
        bound"). Numbers pass through as raw Unix timestamps.

        :class:`datetime.timedelta` is resolved as
        ``now(UTC) - delta`` — a *past* wall-clock moment. This
        matches how callers naturally phrase mtime filters: ``stats
        .is_fresher_than(timedelta(hours=1))`` reads as "modified in
        the last hour", and ``modified_between(start=timedelta(days
        =7))`` as "modified in the last week". A zero/negative
        timedelta is allowed and resolves to "now" or a future
        instant respectively — useful as a sentinel but uncommon in
        practice.

        Everything else (datetime, ISO string) goes through
        :func:`any_to_datetime` for consistent parsing.
        """
        if when is None:
            if default is ...:
                raise ValueError("Default value required for None to create timestamp.")
            return default
        if isinstance(when, (int, float)):
            return float(when)
        if isinstance(when, dt.timedelta):
            return (
                datetime.datetime.now(datetime.timezone.utc) - when
            ).timestamp()
        return any_to_datetime(when, tz=datetime.timezone.utc).timestamp()

    def is_fresher_than(self, mtime: TimeLike) -> bool:
        """Whether this holder was modified strictly after ``mtime``.

        Unknown mtime (``self.mtime == 0.0``) returns ``False``.
        """
        ts = self.normalize_timestamp(mtime, default=None)
        assert ts is not None  # mtime is required here
        return self.is_fresher_than_timestamp(ts)

    def is_fresher_than_timestamp(self, mtime: float) -> bool:
        if self.mtime == 0.0:
            return False
        return self.mtime > mtime

    def is_older_than(self, mtime: TimeLike) -> bool:
        """Whether this holder was modified strictly before ``mtime``.

        Unknown mtime (``self.mtime == 0.0``) returns ``False``.
        """
        ts = self.normalize_timestamp(mtime, default=None)
        assert ts is not None
        return self.is_older_than_timestamp(ts)

    def is_older_than_timestamp(self, mtime: float) -> bool:
        if self.mtime == 0.0:
            return False
        return self.mtime < mtime

    def modified_between(
        self,
        start: TimeLike | None = None,
        end: TimeLike | None = None,
    ) -> bool:
        """Whether mtime falls in the half-open window ``[start, end)``.

        Either bound may be ``None`` for "unbounded on that side", so
        ``modified_between(start=yesterday)`` reads "modified since
        yesterday" and ``modified_between(end=cutoff)`` reads
        "modified before ``cutoff``". Both ``None`` is allowed and
        accepts any known mtime (still rejects unknown).

        Bounds accept anything :func:`any_to_datetime` understands —
        ``datetime``, ISO strings, raw timestamps, or a
        :class:`datetime.timedelta` interpreted relative to now.

        Unknown mtime (``self.mtime == 0.0``) always returns
        ``False`` — there's no honest answer.
        """
        if self.mtime == 0.0:
            return False
        start_ts = self.normalize_timestamp(start, default=None)
        end_ts = self.normalize_timestamp(end, default=None)
        return self.is_between_timestamp(start_ts, end_ts)

    def is_between_timestamp(
        self,
        start: float | None,
        end: float | None,
    ) -> bool:
        """Raw-timestamp variant of :meth:`modified_between`.

        Half-open ``[start, end)``. ``None`` on either side means
        unbounded. Unknown mtime returns ``False``.
        """
        if self.mtime == 0.0:
            return False
        if start is not None and self.mtime < start:
            return False
        if end is not None and self.mtime >= end:
            return False
        return True

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __iter__(self):
        yield self.size
        yield self.mtime
        yield self.kind
        yield self.mode
        yield self.media_type