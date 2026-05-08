"""Fully in-memory :class:`Holder` backed by a :class:`bytearray`.

:class:`Memory` is the simplest concrete :class:`Holder`: every
read/write hits an internally-managed :class:`bytearray`. No fd, no
spill file, no transaction layer. Capacity grows with the standard
1.5× amortization pattern so back-to-back appends stay cheap.

Composes with :class:`yggdrasil.io.buffer.BytesIO`: a memory-mode
``BytesIO`` is conceptually a Memory holder plus a cursor and the
Tabular read/write surface.

Auto-spill
----------
When constructed with ``spill_bytes=N`` (or its environment-variable
equivalent), the Memory transparently migrates its bytearray to an
mmap-backed temp file once the underlying capacity would exceed the
threshold. Reads/writes keep using the same memoryview path — only
the backing changes. The migration **drops the old bytearray
reference** so the resident bytes are eligible for GC and the
process does not carry both copies simultaneously.

Spill files live under ``spill_dir`` (default
:func:`tempfile.gettempdir`) and are named ``ygg-mem-*.spill``. They
are unlinked on :meth:`clear` and on holder release; the mmap and fd
are closed at the same time. Crashing without a clean release leaves
the file on disk — that's a deliberate hygiene tradeoff so a still-
mapped file isn't yanked out from under another reader.

Caller-managed spill location
-----------------------------
Pass ``spill_path=`` to pin the spill to a specific file the caller
owns. The Memory will write into that exact path (creating it if
missing, truncating to the needed size) and treat it as **borrowed**:
:meth:`clear` and :meth:`_release` close the mmap + fd but **do not
unlink** the file. This is the escape hatch for mmap'ing into a
shared scratch area, a tmpfs the caller pre-allocated, a Databricks
``/local_disk0`` partition, etc. Ownership is exclusive: the caller
controls path lifetime; Memory just uses it.

Metadata model
--------------
All IO metadata (visible size, mtime, media-type) lives on a single
mutable :class:`IOStats` instance — :meth:`stat`. Writes mutate it
in place via :meth:`Holder._touch_stat`.
"""

from __future__ import annotations

import mmap
import os
import tempfile
import time
from typing import Any, Optional, Union

from yggdrasil.disposable import Disposable
from yggdrasil.io.io_stats import IOKind, IOStats

from .holder import Holder


__all__ = ["Memory"]


# Minimum mmap size — POSIX mmap of an empty file is invalid, so an
# empty-but-spilled holder still has a 1-byte file underneath. The
# visible size on the IOStats stays accurate; this is purely a backing
# detail.
_MIN_MMAP_BYTES = 1


class Memory(Holder):
    """Fully memory-resident byte holder, with optional mmap auto-spill.

    Construction shapes (in addition to those inherited from
    :class:`Holder`):

    - ``Memory()``          — empty, zero capacity.
    - ``Memory(int n)``     — empty, capacity ``n`` reserved.
    - ``Memory(other_mem)`` — deep copy of another Memory.

    Plus three keyword-only knobs that turn on the spill path:

    - ``spill_bytes`` — threshold in bytes. Once the underlying
      capacity would exceed this value, the bytearray is migrated to
      an mmap-backed temp file and dropped. Accepts an int, a
      :class:`yggdrasil.data.enums.byteunit.ByteUnit` member, or a
      size string (``"128 MB"``). Defaults to ``None`` (no spill —
      pure bytearray mode, the legacy behavior).
    - ``spill_dir`` — directory the spill file is created in.
      Defaults to :func:`tempfile.gettempdir`. Ignored when
      ``spill_path`` is supplied.
    - ``spill_path`` — explicit path to use as the spill file. When
      set, the Memory writes into that exact location and treats it
      as caller-owned: :meth:`clear` and close release the mapping
      but do not unlink the file. Use this to pin spill to a
      caller-controlled scratch area (tmpfs, ``/local_disk0``, …).

    Visible :attr:`size` and underlying buffer capacity are tracked
    separately. ``reserve(n)`` grows capacity without moving
    :attr:`size`; :meth:`truncate` moves :attr:`size`, zero-padding on
    extend.

    Implements the five :class:`Holder` primitives — :meth:`_read_mv`,
    :meth:`_write_mv`, :meth:`reserve`, :meth:`truncate`, :meth:`clear`
    — plus :attr:`size` and the lazy :meth:`stat`. Everything else
    (positional normalization, bounds checking, pre-grow via
    :meth:`resize`, ``mark_dirty``, bytes/text convenience,
    append-at-end ``pos = -1`` semantics) comes from the base class.

    Lifecycle / closed-state access
    -------------------------------
    A :class:`Memory` is usable both **acquired** (inside a ``with``
    block or after :meth:`acquire`) and **closed**. The choice between
    the two is the caller's, and either is correct:

    - **Contextual opening (preferred for hot paths):** wrap usage in
      ``with mem.open() as bio:`` (or ``with mem:``). This pins the
      backing, lets :class:`BytesIO` accumulate writes in a single
      cursor, and gives the spill path a consistent acquire/release
      window. This is the performant pattern for a sequence of
      reads/writes.
    - **Direct access on a closed holder:** all of :meth:`read_bytes`,
      :meth:`write_bytes`, :meth:`pread`, :meth:`pwrite`,
      :meth:`memoryview`, :meth:`to_bytes`, etc. are still callable
      against a closed Memory — :class:`Memory`'s acquire/release are
      no-ops by design (the bytes survive a :meth:`close`). This is
      explicitly supported even though it skips the cursor's
      buffering: each call re-enters the holder cold. Use it for
      one-shot accesses or when integrating with code that doesn't
      manage a context.

    In short: **closed-state access is allowed and correct; contextual
    opening is the performant path.** The choice is the caller's, not
    the holder's, even if direct access is less performant.
    """

    __slots__ = (
        "_buf",
        "_spill_bytes",
        "_spill_dir",
        "_spill_path_request",
        "_spill_fd",
        "_spill_path",
        "_owns_spill_path",
    )

    def __init__(
        self,
        data: Any = None,
        *,
        spill_bytes: Any = None,
        spill_dir: Optional[str] = None,
        spill_path: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self._buf: Union[bytearray, mmap.mmap] = bytearray()
        # Normalize spill threshold via ByteUnit so callers can pass an
        # int (``128 * 1024 * 1024``), a ByteUnit member (``128 *
        # ByteUnit.MIB``), or a string (``"128 MB"``) interchangeably.
        # All three converge on the same byte count — the centralized
        # parser is the single place that decides "what does '128 MB'
        # mean" for the entire codebase.
        from yggdrasil.data.enums.byteunit import ByteUnit
        if spill_bytes is None:
            self._spill_bytes: Optional[int] = None
        else:
            try:
                self._spill_bytes = ByteUnit.parse_size(spill_bytes)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"spill_bytes must be a non-negative byte count "
                    f"(int / ByteUnit / size string), got {spill_bytes!r}"
                ) from exc
        self._spill_dir: Optional[str] = spill_dir
        # ``spill_path`` is the *request* — the caller-supplied target
        # path. We don't open / create it until the spill actually
        # fires; until then ``_spill_path`` (the active path) is None.
        self._spill_path_request: Optional[str] = (
            os.fspath(spill_path) if spill_path is not None else None
        )
        if (
            self._spill_path_request is not None
            and self._spill_dir is not None
        ):
            raise ValueError(
                "Pass either spill_dir or spill_path, not both — "
                "spill_path already pins the location."
            )
        self._spill_fd: Optional[int] = None
        self._spill_path: Optional[str] = None
        # Ownership of the active spill file: True for ones we
        # mkstemp'd (we unlink on clear/release), False for paths the
        # caller supplied (we leave them on disk).
        self._owns_spill_path: bool = True

        # ``int`` is Memory-specific (reserve capacity, no payload);
        # everything else routes through Holder's _init_from_* dispatch.
        # Validate up front; defer the actual reserve/spill until
        # after ``super().__init__`` has populated ``_cached_stat`` —
        # the spill path reads stat() to know how much payload to
        # carry over.
        int_capacity: Optional[int] = None
        if isinstance(data, int) and not isinstance(data, bool):
            if data < 0:
                raise ValueError(
                    f"Memory(int) capacity must be >= 0, got {data!r}"
                )
            int_capacity = data
            data = None

        super().__init__(data, **kwargs)

        if int_capacity is not None:
            # Honor spill_bytes at construction time too — a caller
            # that asks for a 1 GB capacity with spill_bytes=128 MB
            # should never see the bytearray materialize.
            if (
                self._spill_bytes is not None
                and int_capacity > self._spill_bytes
            ):
                self._spill_to_disk(target_capacity=int_capacity)
            else:
                self._buf = bytearray(int_capacity)

    # ``_acquire`` / ``_release`` inherited as no-ops from Disposable
    # for the in-memory case. The spilled case overrides ``_release``
    # below to release the fd/mmap/file — leaving an mmap dangling
    # past close would be a real fd leak, not just a GC oddity.

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

        ``view``-constructed instances cannot spill: the whole point
        of an alias is that ``buf`` stays the canonical storage.
        """
        m = cls.__new__(cls)
        Disposable.__init__(m)
        m._url = None
        m.temporary = False
        m._buf = buf
        m._spill_bytes = None
        m._spill_dir = None
        m._spill_path_request = None
        m._spill_fd = None
        m._spill_path = None
        m._owns_spill_path = True
        m._size = len(buf) if size is None else int(size)
        m._mtime = time.time()
        m._media_type = media_type
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

    @property
    def is_spilled(self) -> bool:
        """True when the backing has migrated to an mmap'd temp file."""
        return self._spill_path is not None

    @property
    def spill_path(self) -> Optional[str]:
        """Path to the spill file, or ``None`` if not spilled."""
        return self._spill_path

    # ------------------------------------------------------------------
    # Holder primitives
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    def _stat(self) -> IOStats:
        """Snapshot the in-memory metadata into a fresh :class:`IOStats`.

        Memory holders own their size / mtime / media_type directly —
        no backend round-trip — so :meth:`stat` is just a copy of the
        slot fields plus :data:`IOKind.MEMORY`.
        """
        return IOStats(
            kind=IOKind.MEMORY,
            size=self._size,
            mtime=self._mtime,
            media_type=self._media_type,
        )

    @property
    def capacity(self) -> int:
        """Current allocated capacity (length of the underlying buffer).

        For a spilled holder this is the size of the mapped file, which
        is always at least :data:`_MIN_MMAP_BYTES` even when visible
        :attr:`size` is zero — POSIX mmap of an empty file is invalid.
        """
        return len(self._buf)

    def _read_mv(self, n: int, pos: int) -> memoryview:
        # Holder.read_mv has already normalized pos and bounded n;
        # 0 <= pos <= size and 0 <= n <= size - pos. Both bytearray
        # and mmap.mmap support the buffer protocol identically here.
        return memoryview(self._buf)[pos : pos + n]

    def _write_mv(self, data: memoryview, pos: int) -> int:
        """Splice bytes at ``pos`` — size already grown by :meth:`write_mv`.

        :class:`Holder.write_mv` has pre-grown the visible size via
        :meth:`resize` (which delegates to :meth:`truncate` here),
        which in turn called :meth:`reserve` to grow the underlying
        buffer (bytearray or mmap). So at this point capacity is
        already guaranteed ≥ ``pos + len(data)`` and we just lay
        bytes down.

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

        # Crossing the spill threshold flips the backing once; after
        # that we just grow the file in place.
        if isinstance(self._buf, mmap.mmap):
            self._grow_mmap(new_cap)
            return

        if (
            self._spill_bytes is not None
            and new_cap > self._spill_bytes
        ):
            self._spill_to_disk(target_capacity=new_cap)
            return

        self._buf.extend(b"\x00" * (new_cap - cur))

    def truncate(self, n: int) -> int:
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        if n == self._size:
            return n
        if n > len(self._buf):
            # bytearray.extend(b"\x00"*…) gives zero-padding for free,
            # so reserve() does both the capacity grow and the zero-fill.
            self.reserve(n)
        self._size = n
        # ``mtime`` intentionally not bumped here — :meth:`truncate`
        # is in the bulk-write hot path (``write_mv`` calls
        # ``resize`` which calls us once per chunk), and a per-call
        # ``time.time()`` dominated tight loops. Callers that want
        # freshness call :meth:`touch_mtime` after the loop.
        return n

    def _clear(self) -> None:
        """Drop all bytes; reset capacity AND size to zero.

        Tears down the spill backing if present so a clear()-then-
        write cycle starts from a fresh in-memory bytearray rather
        than carrying the mmap forward. The spill threshold itself is
        retained — re-crossing it re-spills.
        """
        self._teardown_spill()
        self._buf = bytearray()
        self._size = 0

    # ------------------------------------------------------------------
    # Spill machinery
    # ------------------------------------------------------------------

    def _spill_to_disk(self, *, target_capacity: int) -> None:
        """Migrate the bytearray payload into an mmap'd temp file.

        Drops the bytearray reference *before* the mmap handle is
        published, so the GC can reclaim the in-memory copy on the
        next collection cycle. The visible :attr:`IOStats.size` is
        unchanged — this is a backing-shape switch, not a payload
        edit.

        When the caller pre-pinned a target via ``spill_path=``, the
        spill writes into that exact path and is marked as borrowed —
        :meth:`_teardown_spill` will release the mapping but won't
        unlink the file. Otherwise an anonymous tempfile is created
        and owned by this Memory instance.
        """
        cap = max(target_capacity, _MIN_MMAP_BYTES)
        if self._spill_path_request is not None:
            # Caller-pinned destination. Open with O_RDWR | O_CREAT so
            # the file is materialized on first spill, but we don't
            # truncate it until ftruncate below — that way an existing
            # file's bytes aren't blasted before the call to ftruncate
            # clearly states the new size.
            path = self._spill_path_request
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            owns = False
        else:
            fd, path = tempfile.mkstemp(
                prefix="ygg-mem-",
                suffix=".spill",
                dir=self._spill_dir,
            )
            owns = True
        try:
            os.ftruncate(fd, cap)
            # Copy the existing payload — only the visible part. The
            # zero-pad of the bytearray's tail (past stat().size) is
            # not user-visible, and the mmap is freshly zeroed by
            # ftruncate, so we don't bother carrying it forward.
            visible = self._size
            if visible > 0 and isinstance(self._buf, bytearray):
                os.lseek(fd, 0, os.SEEK_SET)
                # write() may short on huge buffers — loop until done.
                view = memoryview(self._buf)[:visible]
                offset = 0
                while offset < visible:
                    written = os.write(fd, view[offset:])
                    if written <= 0:
                        raise OSError(
                            "short write while spilling Memory to "
                            f"{path!r}: wrote {offset} of {visible} bytes"
                        )
                    offset += written
            mm = mmap.mmap(fd, cap, access=mmap.ACCESS_WRITE)
        except BaseException:
            os.close(fd)
            # Only clean up the file if we created it. Caller-supplied
            # paths stay where they are — we don't speculatively delete
            # a file the caller pinned.
            if owns:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            raise

        # Drop the bytearray BEFORE storing the mmap — explicit
        # ordering so that the only live ref to those bytes is gone
        # the moment _buf rebinds. (CPython would do this anyway, but
        # spelling it out makes the leak-test invariants crisp.)
        self._buf = bytearray()
        self._buf = mm
        self._spill_fd = fd
        self._spill_path = path
        self._owns_spill_path = owns

    def _grow_mmap(self, new_cap: int) -> None:
        """Resize the spill file and remap to ``new_cap`` bytes.

        Uses ``mmap.resize`` on platforms that support it; falls back
        to close+ftruncate+remap otherwise. Either way the visible
        bytes ≤ old capacity are preserved (POSIX ftruncate on extend
        zero-pads, which matches the bytearray behavior).
        """
        new_cap = max(new_cap, _MIN_MMAP_BYTES)
        mm = self._buf
        assert isinstance(mm, mmap.mmap)
        try:
            mm.resize(new_cap)
            os.ftruncate(self._spill_fd, new_cap)
            return
        except (SystemError, OSError, ValueError):
            # Some platforms / mmap modes don't support in-place
            # resize. Fall through to remap.
            pass

        mm.flush()
        mm.close()
        os.ftruncate(self._spill_fd, new_cap)
        self._buf = mmap.mmap(
            self._spill_fd, new_cap, access=mmap.ACCESS_WRITE,
        )

    def _teardown_spill(self) -> None:
        """Close mmap + fd and unlink the spill file. Idempotent.

        For caller-supplied (``spill_path=``) destinations the file is
        left on disk — Memory owns the mapping, the caller owns the
        file. For anonymous tempfiles we created we unlink eagerly.
        """
        if isinstance(self._buf, mmap.mmap):
            try:
                self._buf.close()
            except (BufferError, ValueError):
                # Outstanding memoryviews — caller bug; surface
                # later via use-after-free rather than swallow here.
                pass
        self._buf = bytearray()
        if self._spill_fd is not None:
            try:
                os.close(self._spill_fd)
            except OSError:
                pass
            self._spill_fd = None
        if self._spill_path is not None:
            if self._owns_spill_path:
                try:
                    os.unlink(self._spill_path)
                except OSError:
                    pass
            self._spill_path = None
        # Reset to "owned" default so the next spill (after a clear()
        # without a fresh spill_path_request) creates its own tempfile.
        self._owns_spill_path = True

    def _release(self) -> None:
        """Disposable hook — also tears down spill backing on release.

        Spill files are temp resources and shouldn't outlive a clean
        ``close()``. This runs *in addition to* the inherited
        :meth:`Holder._release` (which honors :attr:`temporary`); when
        ``temporary=True``, ``clear()`` already tore the spill down,
        and the call here is a no-op via the idempotent guard.
        """
        super()._release()
        self._teardown_spill()

    def __del__(self) -> None:
        """Final safety net — release spill resources on GC.

        :meth:`mmap.mmap.close` is called by CPython's mmap finalizer
        on its own, but the explicit fd we hold via
        :func:`tempfile.mkstemp` is *separate* from the mmap's
        internal reference and would leak without an explicit close.
        Same story for the on-disk spill file (``os.unlink``). This
        finalizer is the catch-all for callers that drop the holder
        without ``close()`` / ``with``.
        """
        try:
            self._teardown_spill()
        except Exception:
            # __del__ runs at unpredictable times, including during
            # interpreter shutdown when ``os`` may be partly torn
            # down. Swallow rather than emit a warning the caller
            # can't act on.
            pass

    # ------------------------------------------------------------------
    # Memory-specific zero-copy accessors
    # ------------------------------------------------------------------

    def memoryview(self) -> memoryview:
        """Memoryview over the visible payload (size-bounded).

        Override of :meth:`Holder.memoryview` that aliases the
        underlying buffer directly — no copy, no per-byte dispatch.
        Works equally on bytearray and mmap backings.
        """
        return memoryview(self._buf)[: self._size]

    def to_bytes(self) -> bytes:
        return bytes(self.memoryview())
