"""Sliding-window streaming :class:`Holder`.

:class:`MemoryStream` is a :class:`Holder` that pulls bytes from a
streaming source on demand into a fixed-size in-memory window.
Positions are absolute offsets from the start of the stream; the live
window covers ``[window_start, window_end)`` with ``window_end -
window_start <= byte_size``.

Reads past the current :attr:`window_end` pull more bytes from the
source. When a pull (or a manual write) would push the window past
:attr:`byte_size`, the oldest bytes are dropped — :attr:`window_start`
advances. Reads or writes that target a position behind the live
window raise — those bytes are gone.

The whole point is that consumers can keep ``size`` / ``mtime`` /
positional reads working on top of an unbounded source without
buffering the entire stream in memory; the trailing window is enough
for a cursor to backtrack a bounded amount and for downstream code
that wants ``BytesIO``-style addressability.

Sources accepted:

- File-like with ``.read(n)``: ``open(...)``, :class:`io.BytesIO`,
  ``urllib3.HTTPResponse``, ``requests.Response.raw``.
- Bytes-like (:class:`bytes` / :class:`bytearray` / :class:`memoryview`):
  wrapped internally as a finite source — useful for tests and for
  reusing the windowed-reader code on already-resident bytes.
- Iterable yielding bytes-like chunks (a generator, a list of frames).
- Callable ``f(n) -> bytes-like`` returning at most ``n`` bytes; an
  empty return signals EOF.
- ``None``: empty stream — only manual feeds via :meth:`write_bytes`
  add bytes.
"""

from __future__ import annotations

import io
import time
from typing import Any, Callable, Iterable, Iterator, Optional, Union

from yggdrasil.io.io_stats import IOKind, IOStats

from .holder import Holder, _resolve_pos

__all__ = ["MemoryStream"]


_DEFAULT_PULL_CHUNK = 64 * 1024


SourceLike = Union[
    None,
    bytes,
    bytearray,
    memoryview,
    "io.IOBase",
    Iterable[Union[bytes, bytearray, memoryview]],
    Callable[[int], Union[bytes, bytearray, memoryview]],
]


class MemoryStream(Holder):
    """In-memory sliding-window view over a streaming source.

    Construction::

        MemoryStream(source, *, byte_size=64 * 1024, pull_chunk=...)

    ``source`` is the upstream feed (see module docstring for accepted
    shapes). ``byte_size`` caps the live window; once the buffered
    bytes would exceed it, the oldest are evicted and
    :attr:`window_start` advances. ``pull_chunk`` is the default size
    of each pull from the source — defaults to ``min(64 KiB,
    byte_size)``.

    Implements the five :class:`Holder` primitives with absolute-offset
    semantics: :attr:`size` is the highest offset the stream has
    reached so far (= :attr:`window_end`), and reads/writes at any
    ``pos`` in ``[window_start, size]`` are valid.
    """

    __slots__ = (
        "_source",
        "_source_iter",
        "_read_chunk",
        "_byte_size",
        "_pull_chunk",
        "_buf",
        "_window_start",
        "_eof",
    )

    def __init__(
        self,
        source: SourceLike = None,
        *,
        byte_size: int,
        pull_chunk: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(byte_size, int) or isinstance(byte_size, bool):
            raise TypeError(
                f"byte_size must be an int, got {type(byte_size).__name__}"
            )
        if byte_size <= 0:
            raise ValueError(
                f"byte_size must be > 0, got {byte_size!r}"
            )
        if pull_chunk is not None:
            if not isinstance(pull_chunk, int) or pull_chunk <= 0:
                raise ValueError(
                    f"pull_chunk must be a positive int if provided, "
                    f"got {pull_chunk!r}"
                )

        self._byte_size: int = byte_size
        self._pull_chunk: int = (
            pull_chunk if pull_chunk is not None
            else min(_DEFAULT_PULL_CHUNK, byte_size)
        )
        self._buf: bytearray = bytearray()
        self._window_start: int = 0
        self._eof: bool = False
        self._source: Any = source
        self._source_iter: Optional[Iterator] = None
        self._read_chunk: Optional[Callable[[int], Any]] = None
        self._bind_source(source)

        # Skip Holder.__init__'s ``data`` routing — ``source`` is the
        # content feed, not a bytes/path/url seed. Pass only the
        # identity/stat kwargs through.
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Source binding — normalize the four accepted shapes into a single
    # ``read(n) -> bytes-like`` callable.
    # ------------------------------------------------------------------

    def _bind_source(self, source: Any) -> None:
        if source is None:
            self._read_chunk = None
            self._eof = True
            return
        if isinstance(source, (bytes, bytearray, memoryview)):
            self._read_chunk = io.BytesIO(bytes(source)).read
            return
        if hasattr(source, "read") and callable(source.read):
            self._read_chunk = source.read
            return
        if callable(source):
            self._read_chunk = source
            return
        if isinstance(source, str):
            raise TypeError(
                "MemoryStream source cannot be a str — pass bytes-like or "
                "a file-like object opened in binary mode."
            )
        try:
            self._source_iter = iter(source)
        except TypeError as exc:
            raise TypeError(
                f"MemoryStream source must be None, bytes-like, file-like "
                f"with .read(n), a callable f(n), or an iterable of "
                f"bytes-like; got {type(source).__name__}"
            ) from exc
        self._read_chunk = self._read_from_iter

    def _read_from_iter(self, n: int) -> bytes:
        # ``n`` is advisory for iterables — we can't ask a generator
        # for "exactly N bytes," so we return whatever the next chunk
        # is. The pull loop will keep calling until ``size`` reaches
        # the target or EOF hits.
        if self._source_iter is None:
            return b""
        try:
            chunk = next(self._source_iter)
        except StopIteration:
            return b""
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"MemoryStream iterable yielded "
                f"{type(chunk).__name__}, expected bytes-like"
            )
        return bytes(chunk)

    # ------------------------------------------------------------------
    # Window accessors
    # ------------------------------------------------------------------

    @property
    def byte_size(self) -> int:
        """Maximum bytes retained in the live window."""
        return self._byte_size

    @property
    def window_start(self) -> int:
        """Absolute offset of the first byte still in the window."""
        return self._window_start

    @property
    def window_end(self) -> int:
        """Absolute offset one past the last byte in the window
        (== :attr:`size`).
        """
        return self._window_start + len(self._buf)

    @property
    def eof(self) -> bool:
        """True once the source has signalled EOF."""
        return self._eof

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

    @property
    def is_streaming(self) -> bool:
        """True while EOF hasn't been reached on the source feed.

        :attr:`size` reflects only the bytes pulled so far, so
        cursor-anchored readers (:class:`IO.read`) need to
        bypass the standard ``cap at size`` clamp until the
        source signals EOF.
        """
        return not self._eof

    # ------------------------------------------------------------------
    # Pull machinery
    # ------------------------------------------------------------------

    def _pull_one(self, want: int) -> int:
        """Pull at most ``want`` bytes from the source. Returns the
        count actually appended to the buffer (``0`` on EOF).
        """
        if self._eof or self._read_chunk is None:
            self._eof = True
            return 0
        try:
            chunk = self._read_chunk(want)
        except StopIteration:
            self._eof = True
            return 0
        if not chunk:
            self._eof = True
            return 0
        # File-like ``.read(n)`` may return ``str`` on a text-mode
        # handle — caller's bug, but the error from ``extend`` is
        # opaque, so raise something honest.
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"MemoryStream source returned "
                f"{type(chunk).__name__}, expected bytes-like"
            )
        n = len(chunk)
        self._buf.extend(chunk)
        self._slide_window()
        self._touch_stat(size=self.size)
        return n

    def _slide_window(self) -> None:
        excess = len(self._buf) - self._byte_size
        if excess > 0:
            del self._buf[:excess]
            self._window_start += excess

    def _pull_until(self, target_offset: int) -> None:
        """Pull until :attr:`size` reaches ``target_offset`` or EOF."""
        while not self._eof and self.size < target_offset:
            want = max(self._pull_chunk, target_offset - self.size)
            if self._pull_one(want) == 0:
                break

    def _pull_to_eof(self) -> None:
        while not self._eof:
            if self._pull_one(self._pull_chunk) == 0:
                break

    # ------------------------------------------------------------------
    # Holder primitives — overridden to drive pulls before bounds checks.
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._window_start + len(self._buf)

    def _stat(self) -> IOStats:
        return IOStats(
            kind=IOKind.MEMORY,
            size=self.size,
            mtime=self._mtime,
            media_type=self.media_type,
        )

    def read_mv(self, size: int = -1, offset: int = 0) -> memoryview:
        """Read ``size`` bytes at absolute ``offset``, pulling from source
        as needed. ``size < 0`` reads to EOF.

        Raises if ``offset`` is behind the live window — those bytes have
        already slid out and are unrecoverable.
        """
        # Resolve negative offsets against the *current* size first
        # so the SEEK_END idiom (``offset = -1, size = 0``) lands at
        # window_end without forcing a pull.
        offset = _resolve_pos(offset, self.size)
        if offset < 0:
            raise ValueError(
                f"Offset {offset} is out of bounds for "
                f"MemoryStream of size {self.size}"
            )

        if size < 0:
            self._pull_to_eof()
            size = max(0, self.size - offset)
        else:
            target = offset + size
            if target > self.size:
                self._pull_until(target)
            # EOF may have hit before reaching target — cap to what's
            # actually available.
            size = min(size, max(0, self.size - offset))

        if offset < self._window_start:
            raise ValueError(
                f"Offset {offset} is behind the live window "
                f"[{self._window_start}, {self.window_end}); the window "
                f"holds at most {self._byte_size} bytes and has slid past."
            )
        if offset > self.size:
            raise ValueError(
                f"Offset {offset} is past EOF (size {self.size})."
            )

        return self._read_mv(size, offset)

    def _read_mv(self, n: int, pos: int) -> memoryview:
        local = pos - self._window_start
        return memoryview(self._buf)[local : local + n]

    def write_mv(
        self,
        data: memoryview,
        offset: int = 0,
        *,
        size: int = -1,
        overwrite: bool = False,
        update_stat: bool = True,
    ) -> int:
        """Splice bytes at ``offset``. Appends past current end extend the
        stream and may slide the window; in-window writes overwrite.

        ``size>=0`` slices the input buffer to at most ``size``
        bytes before the splice. ``overwrite=True`` truncates the
        tail past ``offset + len(data)`` after the splice (same
        contract as :meth:`Holder.write_mv`). Writes behind
        :attr:`window_start` raise — the target bytes have
        already been evicted.
        """
        if size >= 0 and len(data) > size:
            data = data[:size]
        total = self.size
        offset = _resolve_pos(offset, total)
        if offset < 0:
            raise ValueError(
                f"Offset {offset} is out of bounds for MemoryStream"
            )
        if offset < self._window_start:
            raise ValueError(
                f"Cannot write at offset {offset}: behind the live window "
                f"start {self._window_start}."
            )
        if offset > total:
            raise ValueError(
                f"Cannot write at offset {offset}: past current end {total} "
                f"(stream is append-or-overwrite, not sparse)."
            )

        n = len(data)
        if n == 0:
            return 0

        end = offset + n
        local_end = end - self._window_start
        if local_end > len(self._buf):
            # Grow exactly to local_end; the splice below overwrites
            # the new tail, so no zero-padding survives.
            self._buf.extend(b"\x00" * (local_end - len(self._buf)))

        written = self._write_mv(data, offset)
        if overwrite and end < self.size:
            # Drop trailing bytes past the spliced range — collapses
            # ``truncate + write`` into one call.
            self.truncate(end)
        if written > 0 and update_stat:
            self._slide_window()
            self._touch_stat(size=self.size)
            self.mark_dirty()
        return written

    def _write_mv(self, data: memoryview, pos: int) -> int:
        n = len(data)
        if n == 0:
            return 0
        local = pos - self._window_start
        memoryview(self._buf)[local : local + n] = data
        return n

    def reserve(self, n: int) -> None:
        """Pre-grow the underlying bytearray, capped at :attr:`byte_size`.

        Capacity beyond :attr:`byte_size` would be evicted on the very
        next :meth:`_slide_window` call, so the cap is the honest
        ceiling here.
        """
        if n < 0:
            raise ValueError(f"reserve size must be >= 0, got {n!r}")
        target_local = max(0, n - self._window_start)
        target_local = min(target_local, self._byte_size)
        cur = len(self._buf)
        if target_local <= cur:
            return
        self._buf.extend(b"\x00" * (target_local - cur))
        self._touch_stat(size=self.size)

    def truncate(self, n: int) -> int:
        """Set visible :attr:`size` to ``n``. Shrinks drop the tail;
        extends zero-pad. Truncating below :attr:`window_start` raises.
        """
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        if n < self._window_start:
            raise ValueError(
                f"Cannot truncate to {n}: behind the live window start "
                f"{self._window_start}."
            )
        local = n - self._window_start
        cur = len(self._buf)
        if local < cur:
            del self._buf[local:]
        elif local > cur:
            self._buf.extend(b"\x00" * (local - cur))
        self._slide_window()
        stats = self.stat()
        stats.size = self.size
        # ``mtime`` left alone — see :meth:`Memory.truncate` for the
        # rationale: per-call clock reads in the write hot path are
        # expensive enough to dominate tight loops. Use
        # :meth:`touch_mtime` post-loop when freshness matters.
        return self.size

    def _clear(self) -> None:
        """Drop the buffered window and reset :attr:`window_start` to 0.

        The bound source is left in place — subsequent reads can pull
        again from where the source was. To detach from the source
        entirely, drop the holder.
        """
        self._buf = bytearray()
        self._window_start = 0
        # ``_eof`` reflects the source state, not the buffer; clear()
        # of the buffer doesn't unsignal EOF on a drained source.
        stats = self.stat()
        stats.size = 0

    # ------------------------------------------------------------------
    # Convenience overrides
    # ------------------------------------------------------------------

    def memoryview(self) -> memoryview:
        """View over the live window only (not the full stream).

        Override of :meth:`Holder.memoryview` — that one would call
        ``self.read_mv(-1, 0)``, which raises once :attr:`window_start`
        > 0 because position 0 is no longer in the window. The window
        view is the honest answer for a sliding-window holder.
        """
        return memoryview(self._buf)
