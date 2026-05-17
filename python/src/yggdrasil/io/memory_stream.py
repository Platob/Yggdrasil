"""Spill-aware sliding-window streaming :class:`Holder`.

:class:`MemoryStream` is a :class:`Holder` that pulls bytes from a
streaming source on demand into an in-memory window plus an
optional on-disk spill file. Positions are absolute offsets from
the start of the stream.

Retention model
---------------

- :attr:`spill_threshold` (default 128 MiB) caps the live in-memory
  bytearray. Pulls that would push the buffer past this threshold
  spill the cold (oldest) bytes to a tempfile; the buffer holds
  only the recent window.
- :attr:`byte_size` (default 2 GiB) caps total retained bytes
  (memory + spill). Pulls that would push retention past this cap
  evict the oldest bytes — spilled bytes first, then in-memory.
- When ``byte_size <= spill_threshold``, no spill file is ever
  created; the holder falls back to the legacy single-window
  eviction shape so small-budget consumers don't pay tempfile
  overhead.

Reads valid in ``[spill_start, size)``; writes valid in
``[window_start, size]`` (append or in-place inside the live
in-memory portion). Reads or writes that target an evicted offset
raise — those bytes are gone.

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
import tempfile
import time
from typing import Any, BinaryIO, Callable, Iterable, Iterator, Optional, Union

from yggdrasil.io.io_stats import IOKind, IOStats

from .holder import Holder, _resolve_pos

__all__ = ["MemoryStream"]


_DEFAULT_PULL_CHUNK = 64 * 1024

#: In-memory window cap. Bytes beyond this spill to a tempfile.
_DEFAULT_SPILL_THRESHOLD = 128 * 1024 * 1024  # 128 MiB

#: Total retention cap (memory + spill). Beyond this, oldest bytes
#: are evicted (truly dropped — reads behind raise). Picked to keep
#: a comfortable headroom for typical multi-GB downloads while
#: still capping unbounded streams.
_DEFAULT_BYTE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GiB

#: Chunk size for in-place spill-file compaction (eviction-driven
#: rewrite). Picked to balance per-syscall overhead vs. peak
#: transient memory.
_SPILL_COMPACT_CHUNK = 1024 * 1024  # 1 MiB


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
    """Sliding-window streaming holder with optional spill-to-disk.

    Construction::

        MemoryStream(
            source,
            *,
            byte_size=2 GiB,
            spill_threshold=128 MiB,
            pull_chunk=64 KiB,
        )

    ``source`` is the upstream feed (see module docstring for accepted
    shapes). ``byte_size`` caps total retained bytes (memory + spill);
    when retention would exceed it, the oldest bytes are evicted.
    ``spill_threshold`` caps the in-memory live window; cold bytes
    above this go to a tempfile and stay readable until evicted.
    ``pull_chunk`` is the default size of each pull from the source.

    When ``byte_size <= spill_threshold`` the spill file is never
    created — the holder collapses to a pure in-memory eviction loop,
    matching the original single-window shape so small-budget
    consumers don't pay tempfile setup cost.

    Implements the :class:`Holder` primitives with absolute-offset
    semantics: :attr:`size` is the highest offset the stream has
    reached so far. Reads valid in ``[spill_start, size)``; writes
    valid in ``[window_start, size]``.
    """

    __slots__ = (
        "_source",
        "_source_iter",
        "_read_chunk",
        "_byte_size",
        "_spill_threshold",
        "_pull_chunk",
        "_buf",
        "_window_start",
        "_spill_start",
        "_spill_file",
        "_eof",
    )

    def __init__(
        self,
        source: SourceLike = None,
        *,
        byte_size: int = _DEFAULT_BYTE_SIZE,
        spill_threshold: int = _DEFAULT_SPILL_THRESHOLD,
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
        if not isinstance(spill_threshold, int) or isinstance(spill_threshold, bool):
            raise TypeError(
                f"spill_threshold must be an int, got "
                f"{type(spill_threshold).__name__}"
            )
        if spill_threshold <= 0:
            raise ValueError(
                f"spill_threshold must be > 0, got {spill_threshold!r}"
            )
        if pull_chunk is not None:
            if not isinstance(pull_chunk, int) or pull_chunk <= 0:
                raise ValueError(
                    f"pull_chunk must be a positive int if provided, "
                    f"got {pull_chunk!r}"
                )

        self._byte_size: int = byte_size
        self._spill_threshold: int = spill_threshold
        self._pull_chunk: int = (
            pull_chunk if pull_chunk is not None
            else min(_DEFAULT_PULL_CHUNK, byte_size)
        )
        self._buf: bytearray = bytearray()
        self._window_start: int = 0
        self._spill_start: int = 0
        self._spill_file: Optional[BinaryIO] = None
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
        """Maximum bytes retained (memory + spill)."""
        return self._byte_size

    @property
    def spill_threshold(self) -> int:
        """In-memory window cap; bytes above spill to disk."""
        return self._spill_threshold

    @property
    def window_start(self) -> int:
        """Absolute offset of the first byte in the in-memory window."""
        return self._window_start

    @property
    def window_end(self) -> int:
        """Absolute offset one past the last byte in the in-memory window
        (== :attr:`size`).
        """
        return self._window_start + len(self._buf)

    @property
    def spill_start(self) -> int:
        """Absolute offset of the first retained byte.

        Equal to :attr:`window_start` when no spill is active.
        Otherwise it sits at the start of the spill region; bytes
        before this have been evicted and reads behind it raise.
        """
        return self._spill_start

    @property
    def has_spill(self) -> bool:
        """True iff a spill tempfile is currently active."""
        return self._spill_file is not None

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
        """Spill cold in-memory bytes to disk; evict oldest spill +
        in-memory bytes when total retention would exceed
        :attr:`byte_size`.

        Layered budget:

        - When ``byte_size <= spill_threshold``, no spill file is
          ever opened — buf is the only retention layer, capped
          directly at ``byte_size``. Excess bytes are dropped from
          the front (legacy single-window shape).
        - Otherwise the in-memory cap is ``spill_threshold``; cold
          bytes above this go to a spill tempfile (lazy-created on
          first overflow). When memory + spill exceeds ``byte_size``
          the oldest bytes are evicted from spill first, then from
          memory.

        Eviction inside the spill file is implemented by rewriting
        the file's live tail to position 0 — done in
        :data:`_SPILL_COMPACT_CHUNK`-sized chunks so the rewrite
        doesn't materialise the whole spill into Python.
        """
        # Fast path: spill disabled (cap fits in memory). No
        # spill file ever exists in this mode, so retained =
        # in-memory and ``spill_start`` tracks ``window_start``.
        if self._byte_size <= self._spill_threshold:
            excess = len(self._buf) - self._byte_size
            if excess > 0:
                del self._buf[:excess]
                self._window_start += excess
                self._spill_start = self._window_start
            return

        # Spill cold in-memory bytes above the threshold.
        cold = len(self._buf) - self._spill_threshold
        if cold > 0:
            self._spill_append(bytes(self._buf[:cold]))
            del self._buf[:cold]
            self._window_start += cold

        # Enforce total retention budget — evict from spill (oldest),
        # then memory if spill alone wasn't enough.
        retained = self.size - self._spill_start
        if retained > self._byte_size:
            excess_total = retained - self._byte_size
            spilled = self._window_start - self._spill_start
            if spilled > 0:
                self._spill_evict(min(excess_total, spilled))
                excess_total = (self.size - self._spill_start) - self._byte_size
            if excess_total > 0:
                # Spill exhausted (or never existed) — fall through
                # to evicting the buf front.
                drop = min(excess_total, len(self._buf))
                if drop > 0:
                    del self._buf[:drop]
                    self._window_start += drop
                    self._spill_start = self._window_start

    # ------------------------------------------------------------------
    # Spill helpers — tempfile is lazy-created on first overflow.
    # ------------------------------------------------------------------

    def _ensure_spill_file(self) -> BinaryIO:
        if self._spill_file is None:
            self._spill_file = tempfile.TemporaryFile(
                prefix="ygg-memstream-spill-",
                mode="w+b",
            )
        return self._spill_file

    def _spill_append(self, data: bytes) -> None:
        fh = self._ensure_spill_file()
        fh.seek(0, io.SEEK_END)
        fh.write(data)

    def _spill_evict(self, n: int) -> None:
        """Drop oldest ``n`` bytes from the spill region.

        Rewrites the spill file's live tail to position 0 — the
        only way to free disk space the file already allocated.
        Eviction is rare (only on retention-cap pressure) and the
        chunked copy keeps per-call memory bounded.
        """
        if self._spill_file is None or n <= 0:
            return
        spilled = self._window_start - self._spill_start
        n = min(n, spilled)
        if n == spilled:
            # Entire spill gone — drop the file outright.
            self._spill_file.seek(0)
            self._spill_file.truncate()
            self._spill_start = self._window_start
            return
        live_size = spilled - n
        src = n
        dst = 0
        while src < spilled:
            self._spill_file.seek(src)
            chunk = self._spill_file.read(min(_SPILL_COMPACT_CHUNK, spilled - src))
            if not chunk:
                break
            self._spill_file.seek(dst)
            self._spill_file.write(chunk)
            src += len(chunk)
            dst += len(chunk)
        self._spill_file.truncate(live_size)
        self._spill_start += n

    def _spill_read(self, offset: int, n: int) -> bytes:
        """Read ``n`` bytes from the spill region starting at *offset*.

        Caller must ensure ``[offset, offset+n) ⊆ [spill_start,
        window_start)``.
        """
        if self._spill_file is None or n <= 0:
            return b""
        self._spill_file.seek(offset - self._spill_start)
        return self._spill_file.read(n)

    def _close_spill(self) -> None:
        """Drop the spill file. Called by :meth:`_clear` / dispose."""
        if self._spill_file is not None:
            try:
                self._spill_file.close()
            except Exception:
                pass
            self._spill_file = None
        self._spill_start = self._window_start

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

        Reads are valid in ``[spill_start, size)`` — anything
        behind :attr:`spill_start` has been evicted (truly dropped
        from both memory and spill) and raises.
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

        if offset < self._spill_start:
            raise ValueError(
                f"Offset {offset} is behind the retained region "
                f"[{self._spill_start}, {self.size}); the retention "
                f"budget is {self._byte_size} bytes."
            )
        if offset > self.size:
            raise ValueError(
                f"Offset {offset} is past EOF (size {self.size})."
            )

        return self._read_mv(size, offset)

    def _read_mv(self, n: int, pos: int) -> memoryview:
        if n == 0:
            return memoryview(b"")
        if pos >= self._window_start:
            # Wholly in-memory.
            local = pos - self._window_start
            return memoryview(self._buf)[local : local + n]
        # Spill region — read from disk. May span into memory.
        spill_end = min(pos + n, self._window_start)
        spill_part = self._spill_read(pos, spill_end - pos)
        if pos + n <= self._window_start:
            return memoryview(spill_part)
        # Cross-boundary: stitch spill + memory.
        mem_n = (pos + n) - self._window_start
        mem_part = bytes(memoryview(self._buf)[:mem_n])
        return memoryview(spill_part + mem_part)

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
        """Set visible :attr:`size` to ``n``. Shrinks drop the tail
        (in-memory and spill); extends zero-pad in memory.

        Truncating below :attr:`spill_start` raises — those bytes
        are evicted and unrecoverable. A truncate that lands inside
        the spill region drops the trailing spill bytes and the
        whole in-memory window.
        """
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        if n < self._spill_start:
            raise ValueError(
                f"Cannot truncate to {n}: behind the retained region "
                f"start {self._spill_start}."
            )
        if n < self._window_start:
            # Drop the in-memory window entirely; shrink spill.
            self._buf = bytearray()
            self._window_start = n
            if self._spill_file is not None:
                self._spill_file.truncate(n - self._spill_start)
            stats = self.stat()
            stats.size = self.size
            return self.size

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
        """Drop the buffered window + spill file and reset offsets to 0.

        The bound source is left in place — subsequent reads can pull
        again from where the source was. To detach from the source
        entirely, drop the holder.
        """
        self._buf = bytearray()
        self._window_start = 0
        self._spill_start = 0
        self._close_spill()
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
