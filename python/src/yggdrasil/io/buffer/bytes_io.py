"""Spill-to-disk byte buffer with cursor and tabular dispatch.

A :class:`BytesIO` is a cursor + format dispatcher layered on top of
a single :class:`yggdrasil.io.holder.Holder` backing. The buffer
keeps **one** ``_holder`` slot that mutates between two shapes:

- :class:`yggdrasil.io.memory.Memory` (in-memory backed by
  :class:`bytearray`) — fast for small/medium payloads, used when
  there's no path bound and the payload fits under ``spill_bytes``.
- :class:`yggdrasil.io.fs.Path` (path-bound) — the path's
  :meth:`Path.acquire_io` sets up its own fd (local) or transaction
  :class:`BytesIO` (remote); the buffer forwards every positional op
  (``pread`` / ``pwrite`` / ``truncate``) straight to the holder.

The buffer never branches on ``isinstance(_holder, Memory|Path)`` for
I/O — every cursorless op funnels through ``_holder.pread`` /
``_holder.pwrite`` / ``_holder.truncate`` / ``_holder.size``. The
spill swap rebinds ``_holder`` in place; call sites stay unchanged.

Shape
-----

A :class:`BytesIO` has one of two backings at any moment:

- **memory** — ``_holder`` is a :class:`Memory`; the buffer
  auto-spills to a self-owned temp file once the payload crosses
  ``spill_bytes``.
- **path-bound** — ``_holder`` is a :class:`Path` whose
  :meth:`Path.acquire_io` sets up its own fd (local) or transaction
  :class:`BytesIO` (remote). The path IS the I/O state.

The path-bound file is either:

- **owned** (``_owns_holder = True``) — minted by us in
  :func:`tempfile.gettempdir`. Unlinked on close.
- **external** (``_owns_holder = False``) — supplied by the
  caller via the ``path=`` constructor kwarg. Never unlinked.

Lifecycle
---------

Inherits :class:`Disposable`. Constructed in the open state by
default (``open`` runs from ``__init__``), so callers who never
enter a ``with`` block still see :meth:`close` do the right thing.

- :meth:`_acquire` opens the holder's per-open IO state via
  :meth:`Holder.acquire_io` — a no-op for :class:`Memory`, and for
  :class:`Path` it brings up the fd (local) or transaction buffer
  (remote).
- :meth:`flush` calls ``_holder.flush()`` — a no-op for memory and
  fd-backed contexts (writes already in the kernel via
  :func:`os.pwrite`) and a path commit for buffer-backed contexts.
- :meth:`_release` flushes, releases per-open state via
  :meth:`Holder.close_io`, unlinks the spill file (only if owned).
- ``with bio:`` uses single-shot semantics — the outermost
  ``__exit__`` always closes.

Three primitives
----------------

Every public op composes from three internals:

- :meth:`_slice` — read N bytes at position
- :meth:`_write_at` — write N bytes at position, growing/spilling
- :meth:`_set_size` — extend or shrink

Each routes through ``_holder.pread`` / ``_holder.pwrite`` /
``_holder.truncate``. Memory-mode auto-spills on threshold; path-bound
buffers never auto-spill at this layer (the path is the durable
backing).

Modes
-----

Mode strings follow stdlib ``open()``: ``"rb"``, ``"wb"``,
``"ab"``, ``"xb"`` plus ``+`` variants. The mode primarily affects:

- the open fd's :func:`os.open` flags (local-fd contexts),
- the initial cursor position (``ab`` starts at EOF),
- whether the buffer-backed context truncates / requires existence.

Memory-mode buffers honour the mode for ``readable``/``writable``
reporting but otherwise treat all modes alike.
"""

from __future__ import annotations

import base64
import contextlib
import io
import mmap
import os
import pathlib
import struct
import tempfile
import time
from typing import IO, TYPE_CHECKING, Any, Optional, Union, Literal

import pyarrow as pa

import yggdrasil.pickle.json as json_module
from yggdrasil.data.options import CastOptions
from yggdrasil.disposable import Disposable
from yggdrasil.io.enums import Codec, MediaType, MimeType, MimeTypes, Mode, ZSTD
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.buffer._spill_cleanup import maybe_cleanup_stale_spill_files
from yggdrasil.io.buffer.base import TabularIO
from yggdrasil.io.types import BytesLike
from yggdrasil.io.url import URL
from yggdrasil.lazy_imports import local_path_class, path_class

if TYPE_CHECKING:
    import blake3
    import xxhash
    from collections.abc import Iterable, Iterator
    from yggdrasil.io.fs import Path
    from yggdrasil.io.holder import Holder


__all__ = ["BytesIO", "BufferLike"]

BufferLike = Union[
    bytes,
    bytearray,
    memoryview,
    io.BytesIO,
    "BytesIO",
    IO[bytes],
]

_HEAD_DEFAULT = 128
_PICKLE_COMPRESS_THRESHOLD_DEFAULT = 1 * 1024 * 1024
_COPY_CHUNK_SIZE = 4 * 1024 * 1024
def _as_contiguous_mv(mv: memoryview) -> memoryview:
    """Return a C-contiguous memoryview.

    ``os.write`` / ``os.pwrite`` accept buffer-protocol objects, but
    non-contiguous memoryviews are not consistently supported across
    versions and platforms. Materialize only when required so the
    fast path stays zero-copy.
    """
    return mv if mv.c_contiguous else memoryview(mv.tobytes())


# ===========================================================================
# BytesIO
# ===========================================================================


class BytesIO(TabularIO[CastOptions], IO[bytes]):
    """Spill-to-disk byte buffer with optional caller-owned path.

    Two construction shapes:

    - **Autonomous** — ``BytesIO(data)`` or ``BytesIO()``. Starts in
      memory, spills to a self-owned temp file once it crosses
      ``spill_bytes``. The spill file is unlinked on close.
    - **Path-bound** — ``BytesIO(path=p, mode="rb"|"wb"|...)``. Pinned
      in spilled mode against the caller's path. The path is not
      unlinked on close.

    Mechanically identical otherwise: same fd-backed positional I/O,
    same primitive set. The only behavioural difference is the
    cleanup decision in :meth:`_release`, gated on
    :attr:`_owns_holder`.

    A :class:`BytesIO` IS-A :class:`TabularIO`. Without a tabular
    media type set, the tabular hooks (``_read_arrow_batches`` /
    ``_write_arrow_batches`` / ``_iter_children``) raise — the buffer
    on its own has no idea what format the bytes are in. Setting a
    media type (or calling :meth:`as_media`) routes through the
    registered concrete leaf (Parquet, IPC, CSV, …) which knows.
    """

    # No __slots__ — concrete subclasses (and per-instance back-pointers
    # like ZipEntryIO.parent) need free attribute setting. The class
    # tree is small enough that the per-instance dict overhead is not
    # a real cost.

    @classmethod
    def default_media_type(cls) -> "MimeType | None":
        """Don't auto-register against any mime type.

        :class:`TabularIO.__init_subclass__` registers concrete
        subclasses against their mime type. :class:`BytesIO` is the
        opaque-bytes layer — there's no single mime it owns, and
        registering against ``OCTET_STREAM`` would shadow the
        fallback resolution in :meth:`TabularIO.media_type_class`.
        Returning ``None`` opts out cleanly.
        """
        return None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def is_bytish(cls, obj: Any) -> bool:
        """Return whether *obj* is a BytesIO-like object."""
        if isinstance(obj, BytesIO):
            return True
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return True
        if (hasattr(obj, "read") or hasattr(obj, "write")):
            return True
        if path_class().is_pathish(obj):
            return True
        return False

    def __new__(
        cls,
        data: Union[
            IO[bytes],
            bytes,
            bytearray,
            memoryview,
            "BytesIO",
            io.BytesIO,
            None,
        ] = None,
        *,
        copy: bool = False,
        media_type: Optional["MediaType"] = None,
        path: "Path | None" = None,
        spill_bytes: int = 128 * 1024 * 1024,
        spill_ttl: int = 86400,
        auto_open: bool | None = None,
        mode: str = "rb+",
        metadata: dict | None = None,
        **kwargs
    ) -> None:
        if data is None and media_type is None and path is None:
            return object.__new__(cls)

        if media_type is None:
            P = path_class()

            if data is not None:
                if isinstance(data, P):
                    path = path if path is not None else data
                elif isinstance(data, str):
                    parsed_path = P.from_(data, default=None)
                    if path is None and parsed_path is not None:
                        path = parsed_path
                        media_type = parsed_path.url.media_type
                elif isinstance(data, BytesIO):
                    media_type = data._stats.media_type

            if media_type is None and path is not None:
                path = P.from_(path)
                media_type = path.media_type

        if media_type is not None:
            media_type = MediaType.from_(media_type)

            if not media_type.is_octet:
                # Defer to TabularIO.__new__'s registry lookup so a
                # tabular media_type lands on the registered leaf
                # (ParquetIO, CsvIO, ZipIO, …) for the requested mime.
                return TabularIO.__new__(
                    cls, data, media_type=media_type, path=path
                )

        # Opaque (or no) media type — stay a plain BytesIO.
        return object.__new__(cls)

    def __init__(
        self,
        data: Union[
            IO[bytes],
            bytes,
            bytearray,
            memoryview,
            "BytesIO",
            io.BytesIO,
            None,
        ] = None,
        *,
        copy: bool = False,
        media_type: Optional["MediaType"] = None,
        path: "Path | None" = None,
        spill_bytes: int = 128 * 1024 * 1024,
        spill_ttl: int = 86400,
        auto_open: bool | None = None,
        mode: str = "rb+",
        metadata: dict | None = None,
        **kwargs
    ) -> None:
        # Funnel cache slots and the spill-path placeholders through
        # the TabularIO base. We then refine the spill bindings below
        # for the buffer-specific cases. Note that TabularIO seeds
        # ``self._stats`` (the canonical IOStats holder) — every
        # ``self._stats.size`` / ``mtime`` / ``media_type`` reference
        # below mutates that single shared instance.
        TabularIO.__init__(self, media_type=media_type)

        # Single backing slot. Memory-mode buffers carry a
        # :class:`Memory` here; the same slot is mutated to a
        # :class:`Path` on spill or path-bind. ``None`` only for view
        # buffers that forward to ``parent`` instead.
        from yggdrasil.io.memory import Memory  # avoid import cycle
        self._holder: "Holder | None" = Memory(auto_open=True)
        self._owns_holder = True
        self._pos: int = 0
        self._mode: str = mode or "rb+"
        self._spill_bytes: int = int(spill_bytes)
        self._spill_ttl: int = int(spill_ttl)
        self._metadata = metadata or {}
        # View state — populated by :meth:`_make_view` only. When
        # ``parent`` is set AND ``_holder`` is ``None``, this BytesIO
        # is a window over the parent: reads/writes route through
        # ``parent.pread`` / ``parent.pwrite`` at
        # ``_view_offset + pos``, bounded by ``_view_max_size`` if
        # set. ZipEntryIO and other subclasses that set ``parent``
        # for navigation always have own backing, so :attr:`is_view`
        # stays False.
        self.parent: "BytesIO | None" = None
        self._view_offset: int = 0
        self._view_max_size: int | None = None

        if path is not None:
            # Path-bound. Caller owns the path; we don't unlink on close.
            self._holder = path_class().from_(path)
            self._owns_holder = False

        if data is not None:
            self._init_from(data, copy=copy)

        if auto_open is None:
            # Default policy:
            #
            # * Caller-owned path-bound buffers stay closed — the
            #   caller drives the lifecycle via ``with`` so we don't
            #   prematurely truncate someone else's file.
            # * Everything else (memory-only and self-owned spill
            #   paths) auto-opens. Memory-only buffers must report
            #   ``closed=False`` from construction so external
            #   writers (pyarrow ``NativeFile``, gzip, zipfile, ...)
            #   accept the handle. Their ``_acquire`` is a cheap
            #   no-op for the memory-only case.
            caller_owned_path = (
                self._is_path_holder() and not self._owns_holder
            )
            auto_open = caller_owned_path or not self._is_path_holder()

        if auto_open and not self._acquired:
            self.open()

    # ------------------------------------------------------------------
    # Input dispatch
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Holder helpers — single source of truth for the active backing
    # ------------------------------------------------------------------

    def _is_path_holder(self) -> bool:
        """True when ``_holder`` is path-bound (spilled or external)."""
        return isinstance(self._holder, path_class())

    def _is_memory_holder(self) -> bool:
        """True when ``_holder`` is the in-memory :class:`Memory` shape."""
        from yggdrasil.io.memory import Memory  # avoid import cycle
        return isinstance(self._holder, Memory)

    def _path_holder(self) -> "Path | None":
        """Return ``_holder`` if path-bound, else ``None``."""
        return self._holder if self._is_path_holder() else None

    def _memory_holder(self):
        """Return ``_holder`` if memory-backed, else ``None``."""
        return self._holder if self._is_memory_holder() else None

    def _set_memory(self, source=None) -> None:
        """Replace ``_holder`` with a :class:`Memory` seeded from *source*."""
        from yggdrasil.io.memory import Memory  # avoid import cycle
        self._holder = Memory(source, auto_open=True)
        self._owns_holder = True

    def _init_from(self, data: Any, *, copy: bool) -> None:
        """Memory-mode dispatcher. May immediately spill if input crosses threshold."""
        if isinstance(data, (bytes, bytearray, memoryview)):
            self._init_from_bytes(memoryview(data))
            return
        if isinstance(data, BytesIO):
            self._init_from_bytesio(data, copy=copy)
            return
        if isinstance(data, io.BytesIO):
            start = data.tell()
            mv = memoryview(data.getvalue())
            self._init_from_bytes(mv[start:] if start else mv)
            return

        # Path-ish positional input — bind to the path, no copy. This
        # is equivalent to passing ``path=data`` as a kwarg.
        p = path_class()
        if p.is_pathish(data):
            self._holder = p.from_(data)
            self._owns_holder = False
            return

        if hasattr(data, "read"):
            self._init_from_filelike(data)
            return

        raise TypeError(
            f"{type(self).__name__} does not accept data of type "
            f"{type(data)!r}. Pass bytes/bytearray/memoryview, "
            "io.BytesIO, BytesIO, or any file-like with .read()."
        )

    def _init_from_bytes(self, mv: memoryview) -> None:
        """In-memory bytes input. Spills if the payload crosses threshold."""
        n = len(mv)
        src = _as_contiguous_mv(mv)
        if n > self._spill_bytes:
            # Hand the bytes to a freshly minted local path — the
            # path's ``write_bytes`` owns the syscall side, the
            # buffer just supplies the payload.
            path = _mint_spill_path(self._ext_hint(), self._spill_ttl)
            path.write_bytes(src)
            self._holder = path
            self._owns_holder = True
            self._stats.size = n
            # Holder's per-open IO state opens lazily in _acquire.
        else:
            self._set_memory(bytes(src))
            self._stats.size = n
        self._pos = 0

    def _init_from_bytesio(
        self,
        src: "BytesIO",
        *,
        copy: bool
    ) -> None:
        """Accept another BytesIO as the source. Always deep-copies."""
        if copy:
            src = src.copy()

        super()._init_from_disposable(src)
        # Share the holder reference — each :class:`Disposable` instance
        # opens its own per-open state on _acquire, so two BytesIO
        # views over the same path get independent fds / transactions.
        self._holder = src._holder
        self._stats.size = src._stats.size
        self._owns_holder = src._owns_holder
        self._pos = src._pos
        self._stats.media_type = src._stats.media_type
        self._metadata = src._metadata
        self._mode = src._mode
        self._spill_bytes = src._spill_bytes
        self._spill_ttl = src._spill_ttl

    def _init_from_filelike(self, src: Any) -> None:
        """Drain a file-like from current cursor to end."""
        if hasattr(src, "seek") and hasattr(src, "tell"):
            start = src.tell()
            src.seek(0, io.SEEK_END)
            remaining = max(0, src.tell() - start)
            src.seek(start)

            if remaining <= self._spill_bytes:
                self._init_from_bytes(memoryview(src.read()))
                return

            # Mint a spill path, stream the source through the path's
            # own ``pwrite`` so the syscall side is the path's concern.
            path = _mint_spill_path(self._ext_hint(), self._spill_ttl)
            path.acquire_io("wb+")
            total = 0
            try:
                while True:
                    chunk = src.read(_COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    written = path.pwrite(chunk, total)
                    if written == 0:
                        break
                    total += written
                path.flush()
            finally:
                path.close_io()
            self._holder = path
            self._owns_holder = True
            self._stats.size = total
            self._pos = 0
            return

        # No seek/tell — drain blind, decide spill vs memory after.
        path = _mint_spill_path(self._ext_hint(), self._spill_ttl)
        path.acquire_io("wb+")
        total = 0
        try:
            while True:
                chunk = src.read(_COPY_CHUNK_SIZE)
                if not chunk:
                    break
                written = path.pwrite(chunk, total)
                if written == 0:
                    break
                total += written
            path.flush()
        finally:
            path.close_io()

        if total <= self._spill_bytes:
            payload = path.read_bytes()
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            self._set_memory(payload)
            self._stats.size = len(payload)
        else:
            self._holder = path
            self._owns_holder = True
            self._stats.size = total
        self._pos = 0

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_(
        cls,
        obj: Any,
        *,
        default: Any = ...,
        **kwargs
    ) -> "BytesIO":
        """Wrap *obj* as a BytesIO. Idempotent on existing instances."""
        if isinstance(obj, cls):
            return obj

        try:
            return cls(obj, **kwargs)
        except TypeError:
            if default is ...:
                raise ValueError(
                    f"Cannot make {cls.__name__} a BytesIO with {obj!r}"
                )
            return default

    @classmethod
    def from_path(
        cls,
        path: "Path",
        *,
        copy: bool = False,
        media_type: MediaType | None = None,
        default: Any = ...,
        **kwargs
    ):
        path = path_class().from_(path, default=None)

        if path is None:
            if default is ...:
                raise ValueError(
                    f"Cannot make {cls.__name__} a BytesIO with {path!r}"
                )
            return default

        return cls(
            path=path,
            copy=copy,
            media_type=media_type,
            **kwargs
        )

    @property
    def spilled(self) -> bool:
        """True when backed by a spill file (owned or external)."""
        return self._is_path_holder()

    @property
    def _owner(self) -> "Holder | None":
        """The active byte holder behind this buffer.

        - **Spilled / path-bound** → the bound :class:`Path` (already
          a :class:`Holder`). Mutates through the path's
          :meth:`acquire_io` flow.
        - **Memory mode** → the :class:`Memory` instance that owns
          the in-memory bytearray. Read/write operations through the
          Holder interface land on the same bytes the buffer's
          primitives do; spill swaps the slot over to the path
          without changing the call surface.
        - **View mode** (window over a parent buffer) → ``None``.
          Views forward to ``self.parent`` directly.
        """
        return self._holder

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    # ------------------------------------------------------------------
    # Disposable hooks — open per-open IO state on the holder
    # ------------------------------------------------------------------

    def _acquire(self) -> None:
        """Acquire the holder's per-open IO state.

        Memory holders are no-ops; :class:`Path` holders bring up an
        fd (local) or transaction buffer (remote) keyed by the
        buffer's mode. Always calls :meth:`Path.acquire_io` so a
        caller-supplied path opened in a different mode at
        construction is reopened in the buffer's mode.
        """
        path = self._path_holder()
        if path is None:
            return

        path.acquire_io(self._mode)
        self._stats.size = int(path.size)
        mt = path.mtime
        self._stats.mtime = float(mt) if mt is not None else 0.0

        if "a" in self._mode:
            self._pos = self._stats.size
        else:
            self._pos = 0

    def _release(self) -> None:
        """Close the holder's per-open IO state, unlink owned spill files."""
        # Tabular cache: clear first so a re-open after close starts cold.
        self.unpersist()

        # Views share their holder with a parent — never touch the
        # holder's per-open state or unlink it on close.
        if self.is_view:
            return

        path = self._path_holder()
        if path is not None:
            # Drop the path's I/O state — close_io flushes the
            # transaction buffer (remote) or closes the fd (local).
            try:
                path.close_io()
            except Exception:
                pass

            # Unlink the file only if we minted it. Caller-owned paths
            # are left alone.
            if self._owns_holder:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                # Drop the holder so later code observes "no path bound."
                self._holder = None

    def _ensure_holder_open(self) -> "Holder":
        """Lazily acquire the holder's per-open IO state; return the holder.

        Memory holders are no-ops; :class:`Path` holders open their fd
        / transaction buffer keyed by the buffer's mode.
        """
        holder = self._holder
        if holder is None:
            raise RuntimeError(
                f"{type(self).__name__} has no holder to open"
            )
        if isinstance(holder, path_class()) and not holder.io_open:
            holder.acquire_io(self._mode)
        return holder

    # ------------------------------------------------------------------
    # Three core primitives — single dispatch through ``_holder.pread`` /
    # ``_holder.pwrite`` / ``_holder.truncate``.
    # ------------------------------------------------------------------
    def _slice(self, pos: int, n: int) -> bytes:
        """Read *n* bytes at *pos*.

        - Path-bound → ``_holder.pread`` (uses ``os.pread`` for local
          paths and the transaction buffer for remote).
        - Memory mode → ``_holder.pread`` against the :class:`Memory`
          backing.
        - View → ``parent.pread`` at ``_view_offset + pos``, bounded
          by the view's tracked size.
        """
        if n <= 0:
            return b""
        if pos < 0:
            raise ValueError("slice position must be >= 0")

        # View mode: forward through the shared holder at the view's
        # window offset, bounded by the view's tracked size.
        if self.is_view:
            if pos >= self._stats.size:
                return b""
            n = min(n, self._stats.size - pos)
            return self._ensure_holder_open().pread(n, self._view_offset + pos)

        holder = self._holder
        if holder is None:
            return b""

        return self._ensure_holder_open().pread(n, pos)

    def _flag_dirty_for_commit(self) -> None:
        """Mark the buffer dirty when ``_commit`` would do real work.

        Hand-rolled instead of routing through :meth:`mark_dirty`
        because writes are valid against unacquired (autonomous,
        memory-mode) buffers, and ``mark_dirty`` raises in that
        state. The set mirrors the conditions in :meth:`_commit`:
        view buffers (parent flush needed) are the only shape with
        anything durable to push. Path-bound contexts already
        commit each ``_holder.pwrite`` straight to the path's
        :meth:`pwrite`, so there's nothing held back to flush;
        memory-mode buffers have nowhere to flush to.
        """
        if not self._acquired:
            return
        if self.is_view:
            self._dirty = True
            return
        if self._is_path_holder() and self._holder.dirty:
            self._dirty = True

    def _write_at(self, data: memoryview, pos: int) -> int:
        """Write *mv* at *pos*. Grows backing, auto-spills on threshold.

        Single dispatch through ``_holder.pwrite`` regardless of
        backing. Memory holders splice into their :class:`bytearray`;
        path holders forward to fd writes (local) or scratch-buffer
        splicing (remote). Auto-spill swaps ``_holder`` from
        :class:`Memory` to :class:`Path` when the projected size
        crosses ``spill_bytes`` — the same call then falls through
        to the path branch transparently.

        Flips the dirty bit on a successful non-zero write so that
        context-manager exit (or an explicit :meth:`flush`) commits
        the change to remote backings without callers having to
        call :meth:`mark_dirty` by hand.
        """
        if pos < 0:
            raise ValueError("write position must be >= 0")

        n = len(data)
        if n == 0:
            return 0

        # View mode: forward through the shared holder at the view's
        # window offset, capped by ``_view_max_size``. The view's
        # tracked size grows up to the cap; the holder's bytes past
        # ``_view_offset + size`` are left intact for the parent.
        if self.is_view:
            if self._view_max_size is not None:
                allowed = self._view_max_size - pos
                if allowed <= 0:
                    return 0
                if n > allowed:
                    data = data[:allowed]
                    n = len(data)
            holder = self._ensure_holder_open()
            written = holder.pwrite(data, self._view_offset + pos)
            if pos + written > self._stats.size:
                self._stats.size = pos + written
            if written:
                self._flag_dirty_for_commit()
                self._touch_mtime()
            return written

        # Auto-spill check fires only for memory holders. A buffer
        # already bound to a path never spills at this layer — the
        # path IS the durable backing.
        if self._is_memory_holder():
            projected = max(self._stats.size, pos + n)
            if projected > self._spill_bytes:
                self._spill()

        if self._holder is None:
            raise RuntimeError(f"Cannot write to {self!r}: no backing")

        holder = self._ensure_holder_open()
        written = holder.pwrite(data, pos)
        self._stats.size = max(self._stats.size, pos + written)
        if written:
            if self._is_path_holder() and holder.dirty:
                self._flag_dirty_for_commit()
            self._touch_mtime()
        return written

    def _set_size(self, n: int) -> int:
        """Truncate or extend backing to exactly *n* bytes.

        Single dispatch through ``_holder.truncate``. Local fd-backed
        path holders call ``os.ftruncate``; remote path holders
        forward to :meth:`Path.truncate`; memory holders resize the
        :class:`bytearray`. Views resize the parent so its bytes past
        ``_view_offset + n`` are dropped (or zero-extended on grow),
        then update the view's tracked size.

        Spill is one-way — truncate on a spilled buffer truncates
        the file, never demotes back to memory.
        """
        if n < 0:
            raise ValueError("Negative size value")

        if self.is_view:
            if self._view_max_size is not None:
                n = min(n, self._view_max_size)
            # Resize the shared holder so bytes past
            # ``_view_offset + n`` are dropped (or zero-extended on
            # grow). Other views over the same holder see the change.
            self._ensure_holder_open().truncate(self._view_offset + n)
            if n != self._stats.size:
                self._flag_dirty_for_commit()
            self._stats.size = n
            if self._pos > n:
                self._pos = n
            return n

        if self._holder is None:
            # Defensive — synthesize a memory backing of n zero bytes.
            # Reachable only after a botched reset.
            self._set_memory(b"\x00" * n)
            self._stats.size = n
            self._pos = min(self._pos, n)
            return n

        prev_size = self._stats.size
        holder = self._ensure_holder_open()
        holder.truncate(n)
        self._stats.size = n
        if self._is_path_holder() and n != prev_size and holder.dirty:
            self._flag_dirty_for_commit()
        if self._pos > n:
            self._pos = n
        return n

    # ------------------------------------------------------------------
    # Spill helper — mutates ``_holder`` from Memory to Path in place
    # ------------------------------------------------------------------

    def _spill(self) -> None:
        """Swap ``_holder`` from :class:`Memory` to :class:`Path`.

        The path owns the syscall side: we hand the payload to the
        path's :meth:`write_bytes` and acquire its IO state so post-
        spill positional ops continue to work without any additional
        bookkeeping at the BytesIO layer.

        After this returns, ``_holder`` is a path-bound :class:`Path`
        with its IO state already open in ``rb+`` mode.
        """
        memory = self._memory_holder()
        if memory is None:
            return  # Already spilled, or no backing.

        path = _mint_spill_path(self._ext_hint(), self._spill_ttl)
        size = self._stats.size
        if size:
            path.write_bytes(bytes(memory.read_mv(size, 0)))
        else:
            # write_bytes on b"" creates a zero-byte file — same shape
            # as the one we'd build via ``open(path, "wb")`` and close.
            path.write_bytes(b"")

        # Acquire the new path's IO state. ``rb+`` so subsequent reads
        # AND writes work regardless of the original mode — the spill
        # is internal scratch we own.
        path.acquire_io("rb+")
        self._holder = path
        self._owns_holder = True

    def _ext_hint(self) -> str:
        """File extension suggestion for spill files."""
        mt = self._stats.media_type
        if mt is not None:
            ext = mt.full_extension
            if ext:
                return ext
        return "bin"

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current size in bytes — the **buffer's** working size.

        Live-reads from the active path holder (fstat for local fd
        contexts, Path.stat for remote passthrough); for memory
        holders this returns the in-memory bytearray's size.

        For "what's on the remote right now", use :meth:`stat`,
        which is always live.
        """
        path = self._path_holder()
        if path is not None and path.io_open:
            self._stats.size = path.size
        return self._stats.size

    def is_empty(self):
        return self.size == 0

    def remaining_bytes(self):
        return self.size - self.tell()

    @property
    def mtime(self) -> float:
        """Modification time of the buffer's working backing.

        Same dispatch as :attr:`size`: live ``fstat`` for local
        fd-mode, ``Path.stat().mtime`` for remote-buffered (the
        remote file's mtime is exposed via :meth:`stat` if the
        caller wants it).
        """
        path = self._path_holder()
        if path is not None and path.io_open:
            return path.mtime or time.time()
        return self._stats.mtime or time.time()

    @property
    def path(self) -> "Path | None":
        """Spill file path, or ``None`` when in memory."""
        return self._path_holder()

    @property
    def url(self) -> URL:
        """URL of the buffer's working backing.

        Forwards to ``_holder.url``: ``file://`` / ``s3://`` / … for
        path holders, ``mem://<addr>`` for memory holders. Falls back
        to a buffer-keyed memory URL only when there's no holder yet
        (views before the parent populates a backing).
        """
        if self._holder is not None:
            return self._holder.url
        return URL.from_memory_address(self)

    @property
    def is_writing(self) -> bool:
        """True when the open mode includes any write semantics."""
        return any(c in self._mode for c in "wa+x")

    @property
    def name(self) -> str:
        path = self._path_holder()
        if path is not None:
            return path.full_path()
        return "<memory>"

    @property
    def is_local(self) -> bool:
        path = self._path_holder()
        if path is not None:
            return path.is_local
        return True

    @property
    def is_remote(self) -> bool:
        return not self.is_local

    def need_spill(self, size: int) -> bool:
        """Predicate: would *size* trigger a spill at current threshold?"""
        if not self._spill_bytes:
            return False
        if self._spill_bytes <= 0:
            return True
        return size >= self._spill_bytes

    # ------------------------------------------------------------------
    # View mode — bounded window sharing the parent's :class:`Holder`
    # ------------------------------------------------------------------

    @property
    def is_view(self) -> bool:
        """True iff this buffer is a window over :attr:`parent`.

        A view shares ``parent._holder`` rather than owning its own
        backing — every read / write routes through the same
        :class:`Holder`, just at ``_view_offset + cursor``. Views
        never unlink the underlying spill on close
        (``_owns_holder is False``). Subclasses (e.g.
        :class:`ZipEntryIO`) that set ``parent`` for navigation but
        allocate their own backing report ``False``.
        """
        parent = self.parent
        if parent is None or self._owns_holder:
            return False
        parent_holder = getattr(parent, "_holder", None)
        return parent_holder is not None and self._holder is parent_holder

    @property
    def start(self) -> int:
        """Absolute offset in :attr:`parent` where the view starts."""
        return self._view_offset

    @property
    def end(self) -> int:
        """Absolute end offset in :attr:`parent`."""
        return self._view_offset + self.size

    @property
    def remaining(self) -> int:
        """Bytes from the current cursor to the view's logical end."""
        return max(0, self.size - self._pos)

    @property
    def max_size(self) -> int | None:
        """Cap on view growth, or ``None`` for unbounded."""
        return self._view_max_size

    @classmethod
    def _make_view(
        cls,
        parent: "BytesIO",
        *,
        offset: int = 0,
        size: int = 0,
        pos: int = 0,
        max_size: "int | None" = None,
    ) -> "BytesIO":
        """Construct a view-mode :class:`BytesIO` over *parent*.

        The view shares ``parent._holder`` directly — every read /
        write reaches the same backing :class:`Holder`. The view
        owns nothing: closing the view does not touch the holder or
        unlink any spill file. ``offset`` / ``size`` / ``max_size``
        bound the visible window over the shared bytes.

        Bypasses :meth:`__new__`'s registry dispatch — a view is always
        a plain BytesIO regardless of the parent's concrete subclass.
        """
        if offset < 0:
            raise ValueError("view offset must be >= 0")
        if size < 0:
            raise ValueError("view size must be >= 0")
        if pos < 0:
            raise ValueError("view pos must be >= 0")
        if max_size is not None and max_size < 0:
            raise ValueError("view max_size must be >= 0")
        if max_size is not None and size > max_size:
            raise ValueError("view size cannot exceed max_size")

        view = object.__new__(BytesIO)
        TabularIO.__init__(view, media_type=parent._stats.media_type)
        # Share the parent's holder directly. The view never owns it,
        # so close-time unlinking is gated on ``_owns_holder``.
        view._holder = parent._holder
        view._owns_holder = False
        view._stats.size = int(size)
        view._stats.mtime = 0.0
        view._pos = int(pos)
        view._mode = parent._mode
        view._spill_bytes = 0
        view._spill_ttl = 0
        view._metadata = {}
        view.parent = parent
        view._view_offset = int(offset)
        view._view_max_size = None if max_size is None else int(max_size)
        Disposable.open(view)
        return view

    # ------------------------------------------------------------------
    # Cursorless I/O — public thin wrappers around the primitives
    # ------------------------------------------------------------------

    def pread(self, n: int, pos: int) -> bytes:
        """Read *n* bytes at absolute *pos*. Cursor untouched."""
        if pos < 0:
            raise ValueError("pread position must be >= 0")
        return self._slice(pos, n)

    def pwrite(self, b: memoryview | bytes | bytearray, pos: int) -> int:
        """Write *b* at absolute *pos*. Cursor untouched."""
        if pos < 0:
            raise ValueError("pwrite position must be >= 0")
        return self._write_at(memoryview(b), pos)

    # ------------------------------------------------------------------
    # Replace
    # ------------------------------------------------------------------

    def replace_with_payload(self, payload: Any) -> None:
        """Replace this buffer's content with *payload*, consuming it.

        Two-mode behaviour driven by ``_owns_holder``:

        - **Owned spill (or memory)** — tear down the old backing
          (close fd, unlink owned spill, drop bytearray), reset
          per-instance state, re-init from *payload*.
        - **External path binding** (path-bound buffer) — keep the
          binding. Truncate the bound file to zero, write the new
          payload's bytes through the fd, leave ``_spill_path`` /
          ``_owns_holder`` intact.

        Cursor resets to 0. ``_stats.media_type`` is intentionally NOT
        modified — callers that need to update it (compress /
        decompress) do so explicitly.

        ``payload is self`` raises ``ValueError``.
        ``payload=None`` clears.
        """
        if payload is self:
            raise ValueError(
                "Cannot replace_with_payload with self — would destroy the "
                "source mid-copy."
            )

        # ------------------------------------------------------------------
        # External path binding — preserve binding, write through it.
        # ------------------------------------------------------------------
        if self._is_path_holder() and not self._owns_holder:
            holder = self._ensure_holder_open()
            holder.truncate(0)
            self._stats.size = 0
            self._pos = 0
            if holder.dirty:
                # Truncate on a remote-buffered context is a real
                # mutation — flag so close/flush actually pushes the
                # zero-byte file to the remote.
                self._flag_dirty_for_commit()

            if payload is None:
                # Empty buffer; flush on close will push zero bytes
                # to remote, or local fd is already truncated.
                return

            # Funnel through _write_bytes_io so bytes-like and
            # BytesIO-like inputs share one path.
            if isinstance(payload, BytesIO):
                self._write_bytes_io(payload)
            else:
                with BytesIO(payload) as scratch:
                    self._write_bytes_io(scratch)

            self._pos = 0
            return

        # ------------------------------------------------------------------
        # Owned scratch — full teardown + reinit.
        # ------------------------------------------------------------------
        path = self._path_holder()
        if path is not None:
            try:
                path.close_io()
            except Exception:
                pass
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        self._set_memory()
        self._stats.size = 0
        self._stats.mtime = 0
        self._pos = 0

        if payload is None:
            return
        self._init_from(payload, copy=False)
        # If _init_from spilled, the holder swap happened in place. If
        # it stayed in memory, no IO state to open. Either way,
        # consistent state.

    # ------------------------------------------------------------------
    # Media type
    # ------------------------------------------------------------------

    @property
    def media_type(self) -> MediaType:
        if self._stats.media_type is None:
            path = self._path_holder()
            if path is not None:
                parsed = MediaType.from_path(path, default=None)
                if parsed is not None and not parsed.mime_type.is_any_bytes:
                    self._stats.media_type = parsed
                    return self._stats.media_type

            parsed = MediaType.from_io(self, default=None)
            if parsed is not None and not parsed.mime_type.is_any_bytes:
                self._stats.media_type = parsed
                return self._stats.media_type

            return MediaType(MimeTypes.OCTET_STREAM)
        elif self._stats.media_type.mime_type.is_any_bytes:
            self._stats.media_type = None
            return self.media_type
        return self._stats.media_type

    @media_type.setter
    def media_type(self, value: "MediaType") -> None:
        self.with_media_type(value, copy=False)

    @property
    def mime_type(self):
        return self.media_type.mime_type

    @property
    def codec(self):
        return self.media_type.codec

    def with_media_type(
        self,
        value: MediaType,
        *,
        copy: bool = False,
    ) -> "BytesIO":
        parsed = MediaType.from_(value)
        if copy:
            duplicate = type(self)(self, copy=True)
            duplicate.with_media_type(parsed, copy=False)
            return duplicate

        current = self._stats.media_type
        if current is None or self.size == 0:
            self._stats.media_type = parsed
            return self
        if current == parsed:
            return self
        if current.mime_type == parsed.mime_type:
            self._stats.media_type = parsed
            return self

        raise ValueError(
            f"Cannot change media type from {current!r} to {parsed!r} on "
            f"a non-empty buffer ({self.size} bytes). The mime type differs, "
            "which would reinterpret the byte content as a different structural "
            "format. Replace the buffer's content first or construct a new "
            "BytesIO with the desired media type."
        )

    def with_mime_type(self, mime_type: MimeType, *, copy: bool = False):
        if self._stats.media_type is None:
            media_type = MediaType.from_mime(mime_type)
        else:
            media_type = self._stats.media_type.with_mime_type(mime_type)
        return self.with_media_type(media_type, copy=copy)

    def stat(self) -> IOStats:
        """The mutable :class:`IOStats` carrying the buffer's metadata.

        Same instance for the buffer's lifetime — pin it once and
        observe live size / mtime / media_type updates as writes land.
        Path-bound buffers refresh ``size`` / ``mtime`` / ``kind`` /
        ``mode`` from the durable backing (open ctx fd, otherwise a
        ``Path.stat`` round-trip) before returning so a stale
        in-memory copy never lies about what's on disk.
        """
        path = self._path_holder()
        if path is not None:
            if path.io_open and path.is_local:
                # Local fd: fstat through the holder is the cheap path.
                try:
                    raw = os.fstat(path.fileno())
                    self._stats.size = raw.st_size
                    self._stats.mtime = raw.st_mtime
                    self._stats.kind = IOKind.FILE
                    self._stats.mode = raw.st_mode
                    return self._stats
                except OSError:
                    pass
            # Local without an open fd, or remote — go through the
            # path's stat. Live every call.
            backing = path.stat()
            self._stats.size = backing.size
            self._stats.mtime = backing.mtime
            self._stats.kind = backing.kind
            self._stats.mode = backing.mode
        return self._stats

    def _touch_mtime(self) -> None:
        """Stamp ``stats.mtime`` to now. Called by every mutator."""
        self._stats.mtime = time.time()

    # ``as_media`` lives on :class:`TabularIO` — it's the same logic
    # for every TabularIO subclass (final-leaf short-circuit, fall
    # through to the registered media class otherwise).

    # ``cached`` / ``persist`` / ``unpersist`` live on :class:`TabularIO`
    # — they just drive the shared ``_persisted_data`` slot.

    def _tabular_leaf_view(self) -> "TabularIO | None":
        """Return a registered leaf wrapping self for tabular dispatch.

        ``None`` when self has no tabular media type or self IS the
        registered leaf for its media type — both signal "no further
        dispatch available, the caller's _read/_write_arrow_batches
        is the leaf and must implement the format itself."
        """
        mt = self.media_type
        if mt is None or getattr(mt, "is_octet", False):
            return None
        target_cls = TabularIO.media_type_class(mt, default=None)
        if target_cls is None or target_cls is type(self):
            return None
        return target_cls.from_(self, media_type=mt)

    def _read_arrow_batches(self, options: CastOptions) -> "Iterator[pa.RecordBatch]":
        """Default — opaque buffer can't yield Arrow batches.

        When self carries a tabular media type but isn't itself the
        registered leaf (e.g. a :class:`ZipEntryIO` whose entry name
        ends in ``.arrow``), dispatch through the registered leaf
        wrapping self. Otherwise raise — there's no honest decode
        without a format.
        """
        view = self._tabular_leaf_view()
        if view is not None:
            # Sync the leaf's size to ours BEFORE the read — the leaf
            # was constructed without seeing self's payload size
            # (init copies _buf reference, not _stats.size).
            view._stats.size = self._stats.size
            yield from view._read_arrow_batches(view.check_options(options))
            return
        raise NotImplementedError(
            f"{type(self).__name__}: {self.synthetic_content()} has no tabular media type. "
            "Construct via the format leaf (ParquetIO, CsvIO, …) "
            "or pass media_type= to dispatch through the registry."
        )

    def _write_arrow_batches(
        self,
        batches: "Iterable[pa.RecordBatch]",
        options: CastOptions,
    ) -> None:
        """Default — opaque buffer can't accept Arrow batches.

        Same dispatch shape as :meth:`_read_arrow_batches`: a buffer
        with a tabular media type but no leaf override routes the
        write through the registered leaf wrapping self.
        """
        view = self._tabular_leaf_view()
        if view is not None:
            view._stats.size = self._stats.size
            view._pos = self._pos
            view._write_arrow_batches(batches, view.check_options(options))
            # Sync the leaf's tracking back — it appended into our
            # shared bytearray but has its own size/pos cursors.
            self._stats.size = view._stats.size
            self._pos = view._pos
            return
        raise NotImplementedError(
            f"{type(self).__name__} has no tabular media type. "
            "Construct via the format leaf (ParquetIO, CsvIO, …) "
            "or pass media_type= to dispatch through the registry."
        )

    # ==================================================================
    # Mode resolution — used by every single-buffer write path
    # ==================================================================

    def _resolve_save_mode(self, mode: Any) -> Mode:
        """Resolve any :class:`Mode` to one a writer can branch on.

        Returns one of:

        - :attr:`Mode.OVERWRITE` — truncate and write fresh.
          Includes AUTO/TRUNCATE, IGNORE-with-empty-buffer,
          ERROR_IF_EXISTS-with-empty-buffer.
        - :attr:`Mode.APPEND` — only when ``_SUPPORTED_APPEND``.
        - :attr:`Mode.IGNORE` — buffer non-empty, caller wants to
          skip.
        - :attr:`Mode.UPSERT` — only when ``_SUPPORTED_UPSERT``.

        Raises :class:`ValueError` for unsupported APPEND/UPSERT
        with a subclass-specific hint, :class:`FileExistsError` for
        ERROR_IF_EXISTS on a non-empty buffer.
        """
        m = Mode.from_(mode, default=Mode.AUTO)

        if m in (Mode.AUTO, Mode.OVERWRITE, Mode.TRUNCATE):
            return Mode.OVERWRITE

        if m is Mode.IGNORE:
            return Mode.IGNORE if not self.is_empty() else Mode.OVERWRITE

        if m is Mode.ERROR_IF_EXISTS:
            if not self.is_empty():
                raise FileExistsError(
                    f"{type(self).__name__} write with "
                    f"Mode.ERROR_IF_EXISTS but buffer is non-empty "
                    f"({self.size} bytes). Path: {self.path!r}"
                )
            return Mode.OVERWRITE

        return m

    # ==================================================================
    # Codec siblings
    # ==================================================================

    def _make_uncompressed_sibling(self) -> "BytesIO":
        """Build an uncompressed sibling carrying self's bytes decompressed.

        Same concrete class as ``self`` with ``default_media_type()``
        (no codec) as media — a downstream lookup of ``codec`` on the
        sibling returns ``None``, terminating recursion through the
        codec branch.
        """
        codec = self.codec
        if codec is None:
            raise RuntimeError(
                f"_make_uncompressed_sibling called on {type(self).__name__} "
                "with no codec; this is a bug in the caller."
            )

        decompressed_buf = codec.decompress(self, copy=True)
        return type(self)(
            decompressed_buf,
            media_type=type(self).default_media_type(),
        )

    def _make_empty_sibling(self) -> "BytesIO":
        """Empty sibling, no source bytes — same format minus the codec.

        Used by the write codec branch: the body fills the sibling
        with raw format bytes, then we compress on the way out.
        Deliberately not via :meth:`_make_uncompressed_sibling` —
        that decompresses self's current bytes, which for a write
        target are either empty or the previous compressed version
        we're about to overwrite.
        """
        return type(self)(media_type=type(self).default_media_type())

    # ==================================================================
    # Lifecycle context managers — open/seek/codec
    # ==================================================================

    @contextlib.contextmanager
    def _reading_context(self, options: CastOptions) -> "Iterator[BytesIO]":
        """Open an IO for reading; yield the IO the body should read from.

        Cursor-transparent: if ``self`` was already open on entry,
        the cursor is restored on exit regardless of where the body
        left it. This makes incidental reads (``collect_schema``,
        footer probes) safe to call mid-stream without disturbing
        an outer iteration.

        With a codec, yields a transient decompressed sibling whose
        lifetime is bounded by this context — the sibling is opened
        on entry and closed (scratch buffer unlinked) on exit.

        Driven by *options*:

        - ``options.read_seek`` — cursor to seek to before the body
          runs on the yielded IO. ``None`` leaves it untouched.
        """
        with contextlib.ExitStack() as stack:
            was_opened = self.opened

            try:
                if self.codec is not None:
                    target = stack.enter_context(self._make_uncompressed_sibling())
                else:
                    target = self
                    if not target.opened:
                        target.open()
                        stack.callback(target.close)
                    elif target.seekable():
                        stack.callback(target.seek, target.tell())

                if options.read_seek is not None and target.seekable():
                    target.seek(options.read_seek)

                yield target
            finally:
                if was_opened and self.closed:
                    self.open()

    @contextlib.contextmanager
    def _writing_context(self, options: CastOptions) -> "Iterator[BytesIO]":
        """Open an IO for writing; yield the IO the body should write to.

        With no codec, yields ``self``. With a codec, yields a
        transient uncompressed sibling — the body writes the raw
        format bytes into the sibling, and on successful exit the
        sibling's bytes are compressed back into ``self`` and ``self``
        is marked dirty so the bound path's write-back fires on
        close.

        On exception inside the body during the codec branch,
        ``self`` is left untouched (the sibling is discarded).

        Driven by *options*:

        - ``options.truncate_before_write`` — truncate the yielded
          IO to zero before the body. Set by OVERWRITE; cleared by
          APPEND.
        - ``options.write_seek`` — cursor on the yielded IO before
          the body. ``None`` leaves it untouched, ``0`` rewinds,
          ``-1`` seeks to end (SEEK_END). APPEND sets ``-1``.
        - ``options.reset_seek`` — restore the pre-entry cursor on
          exit (only when the IO stays open).
        """
        if self.codec is not None:
            yield from self._writing_context_compressed(options)
            return

        with contextlib.ExitStack() as stack:
            was_opened = self.opened
            if not was_opened:
                self.open()
                stack.callback(self.close, force=True)
            elif options.reset_seek and self.seekable():
                stack.callback(self.seek, self.tell())

            try:
                if options.mode is Mode.OVERWRITE:
                    self.truncate(0)

                if options.write_seek is not None and self.seekable():
                    self.seek(options.write_seek)

                yield self

                self.flush()
            finally:
                if was_opened and self.closed:
                    self.open()

    def _writing_context_compressed(
        self, options: CastOptions
    ) -> "Iterator[BytesIO]":
        """Codec branch of :meth:`_writing_context`."""
        codec = self.codec
        assert codec is not None

        if options.mode is Mode.APPEND:
            sibling = self.decompress(codec=codec, copy=True)
        else:
            sibling = self._make_empty_sibling()
        with sibling:
            if options.write_seek is not None and sibling.seekable():
                sibling.seek(options.write_seek)

            yield sibling

            sibling.seek(0)
            compressed = codec.compress(sibling)

        # Sibling closed and scratch unlinked. Replace self's payload.
        self.truncate(0)
        self.seek(0)
        self.replace_with_payload(compressed)
        self.mark_dirty()

    def copy(
        self,
        target_class: type[BytesIO] | None = None
    ) -> "BytesIO":
        """Return an independent copy with the same bytes and media type.

        Backing-aware behaviour:

        - **External path binding** (``_owns_holder=False``) — pass
          the path through to the new instance. Neither side owns the
          file, so sharing the binding is safe; both will pread/pwrite
          against the same durable backing, which is exactly the
          semantic of an external path.
        - **Memory mode** — clone ``_buf`` into a fresh bytearray
          (spilling to a new owned temp file if the payload crosses
          threshold).
        - **Owned spilled** — copy the spill file into a freshly minted
          owned spill via chunked pread→pwrite. The new instance opens
          its own fd lazily.
        - **Non-local owned** — would only happen via a stale transaction
          buffer; pull bytes through and produce an autonomous copy.

        Cursor, media_type, and metadata are preserved; spill_bytes /
        spill_ttl carry over.
        """
        # External path binding — share the binding. Neither instance
        # owns the file, so both can point at it without lifecycle
        # conflicts. The new instance will acquire its own context on
        # open.
        if target_class is None:
            target_class = type(self)

        if self._is_path_holder() and not self._owns_holder:
            return target_class(
                path=self._holder,
                mode=self._mode,
                media_type=self._stats.media_type,
                spill_bytes=self._spill_bytes,
                spill_ttl=self._spill_ttl,
                metadata=dict(self._metadata) if self._metadata else None,
            )

        new_instance = target_class(
            spill_bytes=self._spill_bytes,
            spill_ttl=self._spill_ttl,
            mode=self._mode,
            media_type=self._stats.media_type,
            metadata=dict(self._metadata) if self._metadata else None,
            auto_open=False,
        )

        size = self.size
        if size == 0:
            new_instance._pos = 0
            return new_instance

        memory = self._memory_holder()
        if memory is not None:
            payload = bytes(memory.read_mv(size, 0))
            if size > new_instance._spill_bytes:
                path = _mint_spill_path(new_instance._ext_hint(), new_instance._spill_ttl)
                path.write_bytes(payload)
                new_instance._holder = path
                new_instance._owns_holder = True
                new_instance._stats.size = size
            else:
                new_instance._set_memory(payload)
                new_instance._stats.size = size
            new_instance._pos = self._pos
            return new_instance

        # Owned spilled: copy the spill file via chunked pread→pwrite
        # against the source's acquired path and a fresh acquire on
        # the new path.
        src_holder = self._ensure_holder_open()
        new_path = _mint_spill_path(new_instance._ext_hint(), new_instance._spill_ttl)
        new_path.acquire_io("rb+")
        try:
            pos = 0
            while pos < size:
                want = min(_COPY_CHUNK_SIZE, size - pos)
                chunk = src_holder.pread(want, pos)
                if not chunk:
                    break
                written = new_path.pwrite(chunk, pos)
                if written == 0:
                    break
                pos += written
            new_path.flush()
        finally:
            new_path.close_io()

        new_instance._holder = new_path
        new_instance._owns_holder = True
        new_instance._stats.size = size
        new_instance._pos = self._pos
        return new_instance

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __bytes__(self) -> bytes:
        return self.to_bytes()

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        target_class = type(self)
        opened = "opened" if self.opened else "closed"
        internal = "internal" if self._owns_holder else "external"
        spilled = "spilled" if self.spilled else "memory"
        mt = f" media={self._stats.media_type.__repr__()}" if self._stats.media_type else ""
        path = self._path_holder()
        if path is not None:
            owner = path.url.to_string(encode=False)
        elif self._holder is not None:
            owner = str(id(self._holder))
        else:
            owner = "<not allocated>"
        return (
            f"<{target_class.__name__} [{spilled}/{internal}/{opened}] {owner!r} "
            f"size={self.size}, pos={self.tell()}, mode={self._mode}{mt}>"
        )

    def __getstate__(self):
        """Pickle differently for owned vs path-bound buffers.

        - **Owned** — snapshot the bytes (zstd-compress past
          threshold). The path is internal scratch, not durable.
        - **Path-bound** — pickle the path + mode. The path IS the
          data; snapshotting bytes would diverge from the file the
          moment any other writer touched it.
        """
        self.flush()
        path = self._path_holder()
        if not self._owns_holder and path is not None:
            # Pickle the path as a STRING, not the Path object. Path
            # subclasses inherit from Disposable, which holds a
            # WeakSet whose internal _remove callback is a closure
            # pickle can't serialize. Storing the str form sidesteps
            # the whole Disposable graph; we rebuild a Path on load
            # via path_class().from_().
            return {
                "kind": "path",
                "path": path.url.to_string(),
                "mode": self._mode,
                "media_type": self._stats.media_type,
            }

        if self.size > _PICKLE_COMPRESS_THRESHOLD_DEFAULT:
            blob = self.compress(codec=ZSTD, copy=True).to_bytes()
            codec = ZSTD.name
        else:
            blob = self.to_bytes()
            codec = None
        return {
            "kind": "bytes",
            "data": blob,
            "codec": codec,
            "media_type": self._stats.media_type,
        }

    def __setstate__(self, state):
        kind = state.get("kind", "bytes")  # default for legacy blobs
        media_type = state.get("media_type")
        if kind == "path":
            self.__init__(
                path=path_class().from_(state["path"]),
                mode=state.get("mode", "rb"),
                media_type=media_type,
            )
            return

        blob = state["data"]
        codec = state.get("codec")
        if codec is not None:
            blob = Codec.from_(codec).decompress_bytes(blob)
        self.__init__(blob, copy=False, media_type=media_type)

    # ------------------------------------------------------------------
    # IO[bytes] protocol
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def readable(self) -> bool:
        return "r" in self._mode or "+" in self._mode

    def writable(self) -> bool:
        return self.is_writing

    def seekable(self) -> bool:
        return True

    def _commit(self):
        if self.is_view:
            # Writes already landed on the parent via ``pwrite``; the
            # only remaining work is propagating the parent's flush.
            self.parent.flush()
            return

        path = self._path_holder()
        if path is None or not path.io_open:
            return  # Memory-mode (or unopened) — nothing durable to flush to.

        # Local fd holders: writes already in the kernel — flush is
        # a no-op. Remote holders: commit dirty bytes via path._pwrite.
        path.flush()
        self._stats.size = path.size
        self._stats.mtime = path.mtime
        self.clear_dirty()

    def flush(self) -> None:
        """Push buffered writes to the backing.

        Forwards to the holder's :meth:`Holder.flush`:

        - **Memory mode / view**: nothing durable to flush. (Views'
          flush propagates to the holder in :meth:`_commit`.)
        - **Local fd holder**: no-op. Writes already hit the kernel
          via :func:`os.pwrite`.
        - **Remote passthrough holder**: also a no-op. Every
          ``holder.pwrite`` already issued a :meth:`Path.pwrite`
          syscall, so there's nothing buffered between the buffer
          and the path.
        """
        return self.commit()

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        """Underlying fd. Raises if no holder has an fd to expose."""
        path = self._path_holder()
        if path is None:
            raise OSError("BytesIO has no underlying file descriptor")
        if not path.io_open:
            path.acquire_io(self._mode)
        return path.fileno()

    def readline(self, limit: int = -1) -> bytes:
        """Read one line up to the next ``\\n`` inclusive."""
        if self._pos >= self.size:
            return b""
        if limit is None or limit < 0:
            chunk_len = self.size - self._pos
        else:
            chunk_len = min(limit, self.size - self._pos)
        if chunk_len <= 0:
            return b""

        chunk = self._slice(self._pos, chunk_len)
        nl = chunk.find(b"\n")
        if nl == -1:
            self._pos += len(chunk)
            return chunk

        line = chunk[: nl + 1]
        self._pos += len(line)
        return line

    def readlines(self, hint: int = -1) -> list[bytes]:
        lines: list[bytes] = []
        total = 0
        while True:
            line = self.readline()
            if not line:
                break
            lines.append(line)
            total += len(line)
            if hint is not None and hint > 0 and total >= hint:
                break
        return lines

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def exists(self) -> bool:
        return self.size > 0

    def head(self, n: int = _HEAD_DEFAULT) -> bytes:
        if n <= 0 or self.size == 0:
            return b""
        return self._slice(0, min(n, self.size))

    def tail(self, n: int = _HEAD_DEFAULT) -> bytes:
        if n <= 0 or self.size == 0:
            return b""
        return self._slice(max(0, self.size - n), self.size)

    def synthetic_content(
        self,
        n: int = _HEAD_DEFAULT,
        encoding: str | None = "utf-8",
        errors: str | None = "replace",
    ) -> bytes | str:
        size = self.size
        if size == 0:
            return b"" if encoding is None else ""

        if size <= n:
            content = self.head(size)
        else:
            mid = max(1, n // 2)
            content = self.head(mid) + b"..." + self.tail(mid)

        return content if encoding is None else content.decode(encoding, errors=errors)
    
    def tell(self) -> int:
        return int(self._pos)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Seek to ``offset`` measured by ``whence``.

        Mirrors :meth:`io.IOBase.seek` so external writers (pandas,
        pyarrow, zipfile, gzip, ...) get the semantics they expect.
        Two intentional deviations from stdlib for ergonomics:

        * ``seek(-1, SEEK_SET)`` is a "go to end" sentinel mirroring
          the ``read(-1)`` / "read all" convention used elsewhere on
          this object. Any other negative ``SEEK_SET`` offset raises
          ``ValueError`` like ``io.BytesIO``.
        * ``SEEK_CUR`` / ``SEEK_END`` with an offset that would land
          before byte 0 clamps to 0 instead of raising — matches what
          most file-like consumers in this repo already rely on.
        """
        offset = int(offset)
        size = self.size
        if whence == io.SEEK_SET:
            if offset == -1:
                new_pos = size
            elif offset < 0:
                raise ValueError(
                    f"Negative SEEK_SET offset {offset!r} is invalid; "
                    f"only -1 is accepted as a 'seek to end' sentinel. "
                    f"Use seek({offset}, SEEK_END) to count from the end."
                )
            else:
                new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = max(0, self._pos + offset)
        elif whence == io.SEEK_END:
            new_pos = max(0, size + offset)
        else:
            raise ValueError(f"Invalid whence: {whence!r}")
        self._pos = new_pos
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = max(0, self.size - self._pos)
        out = self._slice(self._pos, size)
        self._pos += len(out)
        return out

    def readinto(self, b) -> int:
        mv = memoryview(b)
        n = len(mv)
        if n == 0:
            return 0

        chunk = self._slice(self._pos, n)
        got = len(chunk)
        if got:
            mv[:got] = chunk
            self._pos += got
        return got

    def readinto1(self, b) -> int:
        return self.readinto(b)

    def readall(self) -> bytes:
        """Read from the cursor to EOF, advancing the cursor.

        Matches :meth:`io.RawIOBase.readall`. For a cursorless full-
        buffer dump that doesn't move ``_pos``, use :meth:`to_bytes`.
        """
        return self.read(-1)

    def write(self, b: Any, *, batch_size: int = 1024 * 1024) -> int:
        if b is None:
            return 0
        if isinstance(b, str):
            return self.write_str(b)
        if isinstance(b, (bytes, bytearray, memoryview)):
            return self.write_bytes(b)
        if isinstance(b, (io.RawIOBase, io.BufferedIOBase)) or hasattr(b, "read"):
            total = 0
            while True:
                chunk = b.read(batch_size)
                if not chunk:
                    break
                total += self.write_bytes(chunk)
            return total
        # Buffer-protocol fallback (e.g. pyarrow.Buffer) — pyarrow's
        # IPC writers hand us native Buffer objects, which aren't
        # bytes/bytearray/memoryview but are memoryview-able.
        try:
            mv = memoryview(b)
        except TypeError:
            return self.write_stream(b)
        return self.write_bytes(mv)

    def write_stream(self, buffer: BytesIO):
        buffer = BytesIO.from_(buffer)
        return self._write_bytes_io(buffer)

    def write_bytes(self, b: bytes | bytearray | memoryview) -> int:
        mv = memoryview(b)
        # Normalize to a 1-D unsigned-byte view so internal splices
        # against bytearray-backed memoryviews don't trip on format /
        # itemsize mismatches (pa.Buffer hands us format='b').
        if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
            mv = mv.cast("B")
        if len(mv) == 0:
            return 0
        n = self._write_at(mv, self._pos)
        self._pos += n
        return n

    def write_bytes_io(
        self,
        buffer: "BytesIO",
        batch_size: int = 1024 * 1024,
    ) -> int:
        buffer = BytesIO.from_(buffer)
        return self._write_bytes_io(buffer, batch_size)

    def _write_bytes_io(
        self,
        buffer: "BytesIO",
        batch_size: int = 1024 * 1024,
    ) -> int:
        """Drain *buffer* into self at the current cursor.

        Single chunked pread → :meth:`_write_at` loop — every backing
        (memory and path) exposes the same :meth:`Holder.pread`
        contract, so there's no reason to dispatch on src/dst shape.
        :meth:`_write_at` handles auto-spill on the dst side when the
        projected size crosses ``spill_bytes``. Returns bytes written.
        Advances ``self._pos`` only; ``buffer._pos`` is untouched.
        ``buffer is self`` raises.
        """
        if buffer is self:
            raise ValueError("Cannot _write_bytes_io a BytesIO into itself")

        if not buffer.opened:
            with buffer:
                return self._write_bytes_io(buffer, batch_size=batch_size)

        src_size = buffer.size
        if src_size == 0:
            return 0

        total = 0
        src_pos = 0
        while src_pos < src_size:
            want = min(batch_size, src_size - src_pos)
            chunk = buffer.pread(want, src_pos)
            if not chunk:
                break
            written = self._write_at(memoryview(chunk), self._pos)
            if written == 0:
                break
            self._pos += written
            total += written
            src_pos += written
        return total

    def write_str(self, s: str, encoding: str = "utf-8") -> int:
        if not s:
            return 0
        return self.write_bytes(s.encode(encoding))

    def write_linebreak(self, newline: str = "\n") -> int:
        return self.write(newline)

    def truncate(self, size: int | None = None) -> int:
        if size is None:
            size = self._pos
        return self._set_size(int(size))

    # ------------------------------------------------------------------
    # write_into / to_path — drain to external sink
    # ------------------------------------------------------------------

    def write_into(
        self,
        dst: IO[bytes] | str | os.PathLike,
        *,
        batch_size: int = _COPY_CHUNK_SIZE,
        overwrite: bool = True,
    ) -> int:
        if isinstance(dst, (str, os.PathLike)):
            dst_path = os.fspath(dst)
            if os.path.exists(dst_path) and not overwrite:
                raise FileExistsError(f"Destination exists: {dst_path!r}")
            parent = os.path.dirname(dst_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(dst_path, "wb") as fh:
                return self._drain_into(fh, batch_size)

        if not hasattr(dst, "write"):
            raise TypeError(
                f"write_into() expected a writable IO or path-like destination, "
                f"got {type(dst)!r}"
            )
        writable = getattr(dst, "writable", None)
        if callable(writable) and not writable():
            raise ValueError("Destination IO is not writable")
        return self._drain_into(dst, batch_size)

    def write_into_path(
        self,
        path: "Path",
        batch_size: int = 1024 * 1024,
    ):
        p = path_class().from_(path)

        if not self.opened:
            with self as opened:
                return opened.write_into_path(p, batch_size=batch_size)

        with p.open(mode="wb") as fh:
            while True:
                chunk = self.read(batch_size)
                if not chunk:
                    break
                fh.write(chunk)

    def _drain_into(self, dst, batch_size: int) -> int:
        total = 0
        remaining = self.size
        pos = 0

        while remaining > 0:
            chunk = self._slice(pos, min(batch_size, remaining))
            if not chunk:
                break
            out = dst.write(chunk)
            written = len(chunk) if out is None else int(out)
            if written != len(chunk):
                raise io.BlockingIOError(
                    f"Short write: expected {len(chunk)}, got {written}"
                )
            total += written
            pos += written
            remaining -= written

        if remaining == 0 and total == 0:
            out = dst.write(b"")
            total = 0 if out is None else int(out)

        flush = getattr(dst, "flush", None)
        if callable(flush):
            flush()
        return total

    def to_path(self, path: str | os.PathLike, *, overwrite: bool = True) -> str:
        self.write_into(path, overwrite=overwrite)
        return os.fspath(path)

    # ------------------------------------------------------------------
    # Structured binary I/O
    # ------------------------------------------------------------------

    def _read_exact(self, n: int) -> bytes:
        data = self.read(n)
        if len(data) != n:
            raise EOFError(f"expected {n} bytes, got {len(data)}")
        return data

    def read_int8(self) -> int: return struct.unpack("<b", self._read_exact(1))[0]
    def write_int8(self, v: int) -> int: return self.write(struct.pack("<b", int(v)))
    def read_uint8(self) -> int: return struct.unpack("<B", self._read_exact(1))[0]
    def write_uint8(self, v: int) -> int: return self.write(struct.pack("<B", int(v)))
    def read_int16(self) -> int: return struct.unpack("<h", self._read_exact(2))[0]
    def write_int16(self, v: int) -> int: return self.write(struct.pack("<h", int(v)))
    def read_uint16(self) -> int: return struct.unpack("<H", self._read_exact(2))[0]
    def write_uint16(self, v: int) -> int: return self.write(struct.pack("<H", int(v)))
    def read_int32(self) -> int: return struct.unpack("<i", self._read_exact(4))[0]
    def write_int32(self, v: int) -> int: return self.write(struct.pack("<i", int(v)))
    def read_uint32(self) -> int: return struct.unpack("<I", self._read_exact(4))[0]
    def write_uint32(self, v: int) -> int: return self.write(struct.pack("<I", int(v)))
    def read_int64(self) -> int: return struct.unpack("<q", self._read_exact(8))[0]
    def write_int64(self, v: int) -> int: return self.write(struct.pack("<q", int(v)))
    def read_uint64(self) -> int: return struct.unpack("<Q", self._read_exact(8))[0]
    def write_uint64(self, v: int) -> int: return self.write(struct.pack("<Q", int(v)))
    def read_f32(self) -> float: return struct.unpack("<f", self._read_exact(4))[0]
    def write_f32(self, v: float) -> int: return self.write(struct.pack("<f", float(v)))
    def read_f64(self) -> float: return struct.unpack("<d", self._read_exact(8))[0]
    def write_f64(self, v: float) -> int: return self.write(struct.pack("<d", float(v)))
    def read_bool(self) -> bool: return bool(self.read_uint8())
    def write_bool(self, v: bool) -> int: return self.write_uint8(1 if v else 0)

    def read_bytes_u32(self) -> bytes:
        return self._read_exact(self.read_uint32())

    def write_bytes_u32(self, data: BytesLike) -> int:
        mv = memoryview(data)
        return self.write_uint32(len(mv)) + self.write_bytes(mv)

    def read_str_u32(self, encoding: str = "utf-8") -> str:
        return self.read_bytes_u32().decode(encoding)

    def write_str_u32(self, s: str, encoding: str = "utf-8") -> int:
        return self.write_bytes_u32(s.encode(encoding))

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def xxh3_64(self) -> "xxhash.xxh3_64":
        import xxhash
        h = xxhash.xxh3_64()
        mv = self.memoryview()
        if mv:
            h.update(mv)
        return h

    def xxh3_int64(self) -> int:
        u = self.xxh3_64().intdigest()
        return u if u < 2**63 else u - 2**64

    def blake3(self) -> "blake3.blake3":
        from blake3 import blake3
        h = blake3(max_threads=blake3.AUTO)

        # Local-path fast path: update_mmap reads the file via mmap
        # directly, no Python-level copy. Works whether or not the
        # holder is currently open.
        path = self._path_holder()
        if (
            path is not None
            and path.is_local
            and self._stats.size
        ):
            h.update_mmap(path.full_path())
            return h

        # Path-bound (remote) or memory: hash whatever the
        # ``memoryview`` accessor returns — for remote paths this
        # round-trips a fresh ``Path.pread`` of the whole payload.
        mv = self.memoryview()
        if mv:
            h.update(mv)
        return h

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def decode(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        if self.size == 0:
            return ""
        return self.to_bytes().decode(encoding, errors)

    def getvalue(self) -> bytes:
        return self.to_bytes()

    def view(
        self,
        *,
        pos: int | None = None,
        size: int | None = None,
        max_size: int | None = None,
    ) -> "BytesIO":
        """Return a view :class:`BytesIO` over this buffer's bytes.

        The view has its own cursor; reads/writes route through this
        buffer's :meth:`pread` / :meth:`pwrite` at
        ``view_offset + view_cursor``. The parent cursor is never
        touched. Use :attr:`is_view` to distinguish a view from a
        concrete buffer.
        """
        if pos is None:
            pos = self._pos if self._pos < self.size else 0
        pos = int(pos)
        if pos < 0:
            raise ValueError("view pos must be >= 0")
        if size is None:
            size = max(0, self.size - pos)
        else:
            size = int(size)
            if size < 0:
                raise ValueError("view length must be >= 0")
        return BytesIO._make_view(
            parent=self,
            offset=pos,
            size=size,
            pos=0,
            max_size=max_size,
        )

    def memoryview(self):
        """Return a ``memoryview`` over this buffer's bytes.

        Forwards to the active :class:`Holder.memoryview`. For an
        in-memory holder this is a zero-copy bytearray view; for a
        local fd-backed path it's an mmap; for a remote path it
        round-trips a fresh ``Path.pread`` of the whole payload.

        Views over a parent return a sized slice of the parent's
        memoryview at the view's offset.

        For local-spilled mode, the returned mmap-backed view's
        lifetime is tied to the GC of the underlying mmap object.
        Release the view before closing the BytesIO if your platform
        is strict about closing mapped files.
        """
        if self._holder is None:
            return memoryview(b"")
        if self.is_view:
            size = self._stats.size
            if size == 0:
                return memoryview(b"")
            return memoryview(self._slice(0, size))
        memory = self._memory_holder()
        if memory is not None:
            return memory.memoryview()
        return self._ensure_holder_open().memoryview()

    def to_bytes(self) -> bytes:
        if self.is_view:
            size = self._stats.size
            return b"" if size == 0 else self._slice(0, size)
        memory = self._memory_holder()
        if memory is not None:
            return memory.to_bytes()
        size = self.size
        if size == 0:
            return b""
        return self._slice(0, size)

    def to_base64(self, urlsafe: bool = True) -> str:
        b = self.to_bytes()
        if urlsafe:
            return base64.urlsafe_b64encode(b).decode("ascii")
        return base64.b64encode(b).decode("ascii")

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress(self, codec: Codec | str, *, copy: bool = False) -> "BytesIO":
        c = Codec.from_(codec)
        if c is None:
            raise ValueError(f"Unknown codec: {codec!r}")
        target_mt = self.media_type.with_codec(c) if self._stats.media_type else None
        payload = c.compress(self)
        if copy:
            payload._stats.media_type = target_mt
            return payload
        self.replace_with_payload(payload)
        self._stats.media_type = target_mt
        return self

    def decompress(
        self,
        codec: "Codec | str | None" = "infer",
        *,
        copy: bool = False,
    ) -> "BytesIO":
        if not codec:
            return self
        if codec == "infer":
            codec = self.media_type.codec
        else:
            codec = Codec.from_(codec)
        payload = codec.decompress(self)
        if copy:
            return payload
        self.replace_with_payload(payload)
        self._stats.media_type = payload._stats.media_type
        return self

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def json_load(
        self,
        orient: Optional[Literal["records", "split", "index", "columns", "values"]] = None,
        *,
        media_type: Optional[MediaType] = None
    ):
        media_type = media_type or self.media_type

        if media_type.mime_type is MimeTypes.JSON:
            if media_type is not None and media_type.codec is not None:
                with self.decompress(codec=media_type.codec, copy=True) as decompressed:
                    return decompressed.json_load(media_type=media_type.with_codec(None))

            with self.view(pos=0) as v:
                return json_module.load(v)
        else:
            with self.as_media(media_type) as reader:
                return reader.read_pylist(reset_seek=True)

    def reserve(self, n: int) -> "BytesIO":
        """Reserve capacity for a total size of *n* bytes.

        Capacity-reservation only — does NOT change ``_stats.size`` or
        ``_pos``. Use :meth:`truncate` if you want the visible size
        to change. Idempotent: if ``self.size >= n`` already, this
        is a no-op.

        Per-backing behaviour:

        - **Memory mode** — if ``n > _spill_bytes``, spill first and
          fall through to the path-bound branch. Otherwise grow the
          underlying ``bytearray`` to length ``n`` using the same
          1.5× amortized pattern as :meth:`_write_at`, leaving
          ``_stats.size`` untouched. Subsequent writes up to *n* bytes
          incur no further reallocation.
        - **Path-bound** — no-op. The path-side context grows lazily
          on positional write; ``posix_fallocate`` isn't portably
          exposed and remote backends don't have a "reserve" concept.

        Returns ``self`` for chaining.
        """
        if n < 0:
            raise ValueError(f"allocate size must be >= 0, got {n}")
        if n == 0:
            return self
        if n <= self.size:
            return self  # Already have at least n bytes of capacity.

        # Path-bound: nothing to pre-grow. The path grows lazily on
        # pwrite — no portable fallocate, remote backends would have
        # no use for it either.
        if self._is_path_holder():
            return self

        # Memory mode. If the target capacity crosses the spill
        # threshold, spill first and bail — the path-side holder
        # needs no pre-growth.
        if n > self._spill_bytes:
            self._spill()
            return self

        # Pre-grow the underlying :class:`Memory` holder. The Holder
        # primitive owns the 1.5× amortization so back-to-back
        # allocate + streaming writes don't fight each other.
        if self._memory_holder() is None:
            # Defensive — autonomous memory-mode buffer with no holder
            # shouldn't happen post-init, but synthesize one rather
            # than raise.
            self._set_memory()
        self._holder.reserve(n)
        return self

    def arrow_io(self, mode: str = "rb", size: int | None = None):
        path = self._path_holder()
        if path is not None and path.is_local:
            if size is not None:
                return pa.create_memory_map(path.full_path(), size)
            return pa.OSFile(path.full_path(), mode)

        if mode in {"a", "ab"}:
            self.seek(0, io.SEEK_END)
            mode = mode.replace("a", "w")
        return pa.PythonFile(self, mode=mode)

    def clear(self):
        self.close(force=True)

        path = self._path_holder()
        if path is not None:
            try:
                path.close_io()
            except Exception:
                pass

        self._set_memory()
        self._stats.size = 0
        self._pos = 0
        self._stats.media_type = None


# ===========================================================================
# Module-level helpers
# ===========================================================================


def _mint_spill_path(ext: str, ttl_seconds: int) -> "Path":
    """Mint a fresh temp file path under :func:`tempfile.gettempdir`.

    Filename layout (time-sortable): ``tmp-{start}-{end}-{seed}.{ext}``.
    Both timestamps are zero-padded to 12 digits so a lexical sort
    of the temp directory yields chronological order — useful for
    debugging, stream-tail tools, and the stale-cleanup sweep that
    walks files oldest-first. The file itself is not created here
    — the caller writes to it.

    Calls :func:`maybe_cleanup_stale_spill_files` before minting so
    the tempdir doesn't accumulate orphans from crashed workers. The
    cleanup is throttled in-process and serialised cross-process,
    so it never dominates the spill hot path.
    """
    maybe_cleanup_stale_spill_files()
    seed = os.urandom(8).hex()
    start = int(time.time())
    end = start + ttl_seconds
    name = f"tmp-{start:012d}-{end:012d}-{seed}.{ext}"
    return local_path_class().from_pathlib(
        pathlib.Path(os.path.join(tempfile.gettempdir(), name))
    )

def _mode_creates_missing_backing(mode: str) -> bool:
    """True when BytesIO path-backed mode should create a missing file."""
    return (
        "w" in mode
        or "a" in mode
        or "x" in mode
        or ("r" in mode and "+" in mode)
    )
