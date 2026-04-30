"""Spill-to-disk byte buffer — pure stdlib, fd-backed, fully self-contained.

Shape
-----

A :class:`BytesIO` has one of three backings at any moment:

- **memory** — a :class:`bytearray`. Fast for small/medium payloads.
  Used when there's no path bound and the payload fits under
  ``spill_bytes``.
- **local-spilled** — a local file backed by a real OS file
  descriptor. Direct positional I/O via :func:`os.pread` /
  :func:`os.pwrite`, growth via :func:`os.ftruncate`, mmap-backed
  :meth:`memoryview`.
- **transactional** — a non-local path (e.g. Databricks volume,
  S3) wrapped around an internal *transaction buffer* — itself a
  ``BytesIO`` (memory or local-spilled). On acquire, the path's
  bytes are loaded into the transaction buffer via ``path.pread``;
  on flush, they're committed back via ``path.pwrite`` +
  ``path.truncate``. All ops in between go against the inner
  buffer.

The local-spilled file is either:

- **owned** (``_owns_spill_path = True``) — minted by us in
  :func:`tempfile.gettempdir`. Unlinked on close.
- **external** (``_owns_spill_path = False``) — supplied by the
  caller via the ``path=`` constructor kwarg. Never unlinked.

Transactional backings are always external — the path is the
caller's destination.

Why a transaction buffer for non-local
--------------------------------------

Previously the non-local mode used a raw ``bytearray`` and
``path.read_bytes`` / ``path.write_bytes``. That worked but
duplicated spill machinery (the bytearray could grow unboundedly
in memory). Wrapping the working bytes in a regular ``BytesIO``
gives us spill-to-local-temp for free: a multi-GiB remote download
spills to a local temp file under the hood, the outer buffer
stays a thin coordinator. The composition is the simplification.

Lifecycle
---------

Inherits :class:`Disposable`. Constructed in the open state by
default (``open`` runs from ``__init__``), so callers who never
enter a ``with`` block still see :meth:`close` do the right thing.

- :meth:`_acquire` opens the spill backing per mode:
  * memory mode → no-op.
  * local path → :func:`os.open` an fd.
  * non-local path → build the transaction buffer, fill it via
    ``path.pread``.
- :meth:`flush` commits a transactional backing to the path
  via ``path.pwrite`` + ``path.truncate`` (no-op for local /
  memory).
- :meth:`_release` flushes (always), closes the transaction
  buffer, closes the local fd (if any), unlinks the spill file
  (only if owned).
- ``with bio:`` uses single-shot semantics — the outermost
  ``__exit__`` always closes.

Three primitives
----------------

Every public op composes from three internals:

- :meth:`_slice` — read N bytes at position
- :meth:`_write_at` — write N bytes at position, growing/spilling
- :meth:`_set_size` — extend or shrink

Each branches on ``_spill_path.is_local``: local goes direct to
the fd, non-local delegates to the transaction buffer, no-path
goes against the in-memory bytearray. Memory-mode auto-spills on
threshold; transactional / local-spilled never auto-spill at this
layer (the path is the durable backing).

Modes
-----

Mode strings follow stdlib ``open()``: ``"rb"``, ``"wb"``,
``"ab"``, ``"xb"`` plus ``+`` variants. The mode primarily affects:

- the spill fd's :func:`os.open` flags (local-spilled),
- the initial cursor position (``ab`` starts at EOF),
- whether ``_acquire`` truncates / requires existence
  (transactional).

Memory-mode buffers honour the mode for ``readable``/``writable``
reporting but otherwise treat all modes alike.
"""

from __future__ import annotations

import base64
import io
import mmap
import os
import pathlib
import struct
import tempfile
import time
from typing import IO, TYPE_CHECKING, Any, Optional, Union

import pyarrow as pa

import yggdrasil.pickle.json as json_module
from yggdrasil.disposable import Disposable
from yggdrasil.io.enums import Codec, MediaType, MimeType, MimeTypes, ZSTD
from yggdrasil.io.path_stat import PathStats, PathKind
from yggdrasil.io.types import BytesLike
from yggdrasil.io.url import URL
from yggdrasil.lazy_imports import tabular_io_class, local_path_class, path_class

if TYPE_CHECKING:
    import blake3
    import xxhash
    from yggdrasil.io.fs import Path
    from yggdrasil.io.tabular import TabularIO


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
_HAS_PWRITE = hasattr(os, "pwrite")
_HAS_PREAD = hasattr(os, "pread")


def _as_contiguous_mv(mv: memoryview) -> memoryview:
    """Return a C-contiguous memoryview.

    ``os.write`` / ``os.pwrite`` accept buffer-protocol objects, but
    non-contiguous memoryviews are not consistently supported across
    versions and platforms. Materialize only when required so the
    fast path stays zero-copy.
    """
    return mv if mv.c_contiguous else memoryview(mv.tobytes())


# ---------------------------------------------------------------------------
# Mode → os.O_* flags
# ---------------------------------------------------------------------------


def _flags_for_mode(mode: str) -> int:
    """Translate a stdlib-style mode string into ``os.open`` flags.

    Binary subset only — text-mode bits (``t``, ``U``, encoding) are
    handled at any wrapper layer above us. Raises :class:`ValueError`
    for nonsensical modes ("rw", "", multiple primaries) so the
    caller surfaces a clear error rather than an opaque ``OSError``.

    Adds ``O_BINARY`` on Windows and ``O_CLOEXEC`` where available.
    """
    has_r = "r" in mode
    has_w = "w" in mode
    has_a = "a" in mode
    has_x = "x" in mode
    has_plus = "+" in mode

    primary_count = sum((has_r, has_w, has_a, has_x))
    if primary_count != 1:
        raise ValueError(
            f"Invalid open mode {mode!r}: must contain exactly one of "
            "'r', 'w', 'a', 'x'."
        )

    if has_r and not has_plus:
        flags = os.O_RDONLY
    elif has_r and has_plus:
        flags = os.O_RDWR
    elif has_w and not has_plus:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    elif has_w and has_plus:
        flags = os.O_RDWR | os.O_CREAT | os.O_TRUNC
    elif has_a and not has_plus:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    elif has_a and has_plus:
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    elif has_x and not has_plus:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    else:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL

    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY  # Windows
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    return flags


# ===========================================================================
# BytesIO
# ===========================================================================


class BytesIO(Disposable, IO[bytes]):
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
    :attr:`_owns_spill_path`.
    """

    __slots__ = (
        "_buf",
        "_size",
        "_mtime",
        "_pos",
        "_spill_path",
        "_spill_fd",
        "_transaction_buffer",
        "_media_type",
        "_spill_bytes",
        "_spill_ttl",
        "_owns_spill_path",
        "_mode",
        "_metadata"
    )

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
        auto_open: bool = True,
        mode: str = "rb+",
        metadata: dict | None = None
    ) -> None:
        Disposable.__init__(self)

        # Per-instance state, all initialized before any helper runs.
        self._buf: bytearray | None = bytearray()
        self._size: int = 0
        self._mtime: float = 0
        self._pos: int = 0
        self._spill_fd: int | None = None
        # Transaction buffer for non-local paths. Filled on _acquire
        # via path.pread, committed on flush via path.pwrite + path.truncate.
        # None for memory-mode and local-path mode (those go direct).
        self._transaction_buffer: "BytesIO | None" = None
        self._mode: str = mode or "rb+"
        self._spill_bytes: int = int(spill_bytes)
        self._spill_ttl: int = int(spill_ttl)
        self._media_type: MediaType | None = (
            None if media_type is None else MediaType.from_(media_type)
        )
        self._metadata = metadata or {}

        if path is not None:
            # Path-bound. Caller owns the path; we don't unlink on close.
            self._spill_path = path_class().from_(path)
            self._owns_spill_path = False
            # Path-bound buffers don't keep an in-memory bytearray —
            # everything goes through the fd.
            self._buf = None
        else:
            self._spill_path = None
            self._owns_spill_path = True  # default for any spill we mint

        if data is not None:
            self._init_from(data, copy=copy)

        if auto_open:
            self.open()

    # ------------------------------------------------------------------
    # Input dispatch
    # ------------------------------------------------------------------

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
            self._spill_path = p.from_(data)
            self._owns_spill_path = False
            self._buf = None
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
            path = _mint_spill_path(self._ext_hint(), self._spill_ttl)
            with open(path.full_path(), "wb") as fh:
                fh.write(src)
            self._buf = None
            self._size = n
            self._spill_path = path
            self._owns_spill_path = True
            # fd opens lazily in _acquire / _ensure_spill_fd.
        else:
            self._buf = bytearray(src)
            self._size = n
        self._pos = 0

    def _init_from_bytesio(self, src: "BytesIO", *, copy: bool) -> None:
        """Accept another BytesIO as the source. Always deep-copies."""
        del copy
        self._init_from_bytes(memoryview(src.to_bytes()))
        if self._media_type is None:
            self._media_type = src._media_type
        self._pos = 0

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

            path = _mint_spill_path(self._ext_hint(), self._spill_ttl)
            total = 0
            with open(path.full_path(), "wb") as fh:
                while True:
                    chunk = src.read(_COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    fh.write(chunk)
                    total += len(chunk)
            self._buf = None
            self._size = total
            self._spill_path = path
            self._owns_spill_path = True
            self._pos = 0
            return

        # No seek/tell — drain blind, decide spill vs memory after.
        path = _mint_spill_path(self._ext_hint(), self._spill_ttl)
        total = 0
        with open(path.full_path(), "wb") as fh:
            while True:
                chunk = src.read(_COPY_CHUNK_SIZE)
                if not chunk:
                    break
                fh.write(chunk)
                total += len(chunk)

        if total <= self._spill_bytes:
            with open(path.full_path(), "rb") as fh:
                payload = fh.read()
            try:
                os.unlink(path.full_path())
            except OSError:
                pass
            self._buf = bytearray(payload)
            self._size = len(payload)
        else:
            self._buf = None
            self._size = total
            self._spill_path = path
            self._owns_spill_path = True
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
        return self._spill_path is not None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    # ------------------------------------------------------------------
    # Disposable hooks — fd lifecycle
    # ------------------------------------------------------------------

    def _acquire(self) -> None:
        """Open the spill backing."""
        if self._spill_path is None:
            return

        if not self._spill_path.is_local:
            self._acquire_transaction_buffer()
            return

        if self._spill_fd is not None:
            return

        local_str_path = self._spill_path.full_path()
        mode = self._mode

        flags = _flags_for_mode(mode)

        if _mode_creates_missing_backing(mode):
            parent_str = os.path.dirname(local_str_path)
            if parent_str:
                os.makedirs(parent_str, exist_ok=True)
            flags |= os.O_CREAT

        fd = os.open(local_str_path, flags, 0o644)
        self._spill_fd = fd

        # Seed size from fstat.
        try:
            self._size = os.fstat(fd).st_size
        except OSError:
            self._size = 0

        # Append modes start the visible cursor at EOF.
        if "a" in self._mode:
            self._pos = self._size
        else:
            self._pos = 0

    def _acquire_transaction_buffer(self) -> None:
        """Build and fill the transaction buffer for a non-local path.

        Uses ``path.pread`` to fill, mode-driven semantics same as
        before. The transaction buffer is a regular autonomous
        ``BytesIO`` — it'll spill to local temp if the remote payload
        is huge, all transparent.

        Mode mapping:

        - ``rb`` / ``rb+`` — pread the existing bytes; raise
          ``FileNotFoundError`` if missing.
        - ``wb`` / ``wb+`` — start empty (truncate semantics).
        - ``ab`` / ``ab+`` — pread (or empty if missing); cursor at
          EOF.
        - ``xb`` / ``xb+`` — raise ``FileExistsError`` if existing;
          otherwise start empty.
        """
        if self._transaction_buffer is not None:
            return  # Already acquired.

        path = self._spill_path
        mode = self._mode
        wants_truncate = "w" in mode
        wants_excl = "x" in mode

        # Probe existence for modes whose semantics depend on it.
        exists = None
        if wants_excl or (("r" in mode or "a" in mode) and not wants_truncate):
            try:
                exists = path.exists()
            except Exception:
                exists = None

        if wants_excl and exists:
            raise FileExistsError(
                f"Cannot exclusively create {path.full_path()!r}: file exists."
            )

        # Build an empty transaction buffer first; fill it below. The
        # inner BytesIO inherits our spill threshold so a multi-GiB
        # remote download spills locally rather than blowing memory.
        # Drop _buf because the outer's bytearray is no longer used —
        # the transaction buffer owns the working bytes.
        self._buf = None
        self._transaction_buffer = type(self)(
            spill_bytes=self._spill_bytes,
            spill_ttl=self._spill_ttl,
            mode="rb+",  # Internal scratch: needs r/w regardless of outer mode.
        )

        if wants_truncate or wants_excl:
            # Start empty regardless of file's current state.
            self._size = 0
            self._pos = 0
            return

        # Read modes ('rb', 'rb+') and append ('ab', 'ab+').
        # Fill via pread. ``n=-1`` reads to EOF.
        try:
            existing = path.pread(n=-1, pos=0)
        except FileNotFoundError:
            existing = None
        except OSError:
            existing = None

        if existing is None:
            # Append modes — empty starting buffer is fine.
            existing = b""

        if existing:
            # Push the bytes into the transaction buffer at offset 0.
            self._transaction_buffer.pwrite(existing, 0)

        self._size = len(existing)
        if "a" in mode:
            self._pos = self._size
        else:
            self._pos = 0

    def _release(self, committed: bool) -> None:
        """Flush transaction buffer, close the spill fd, unlink owned files.

        Order matters:

        1. **Flush first.** For non-local-path backings, commit the
           transaction buffer back via ``path.pwrite`` +
           ``path.truncate``. Skipped if the buffer was opened
           read-only.
        2. **Close the transaction buffer**, releasing any local
           spill it acquired.
        3. **Close the local fd** (if any).
        4. **Unlink the spill file** if owned. Caller-owned paths
           are left alone.

        ``committed`` is currently unused — the contract is "always
        flush dirty buffers on close." If finer-grained commit
        semantics ever ship, this is the hook.
        """
        del committed

        # Flush first, before tearing anything down. A flush failure
        # shouldn't prevent us from closing — but we surface it after
        # teardown so the caller sees the error.
        flush_error: BaseException | None = None
        try:
            self.flush()
        except BaseException as e:
            flush_error = e

        # Close the transaction buffer if we have one. It owns its
        # own spill machinery; closing it releases that.
        if self._transaction_buffer is not None:
            try:
                self._transaction_buffer.close()
            except Exception:
                pass
            self._transaction_buffer = None

        # Close the local fd unconditionally — it's always ours,
        # regardless of who owns the file.
        fd = self._spill_fd
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            self._spill_fd = None

        # Unlink the file only if we minted it. Caller-owned paths
        # are left alone.
        if self._spill_path is not None and self._owns_spill_path:
            try:
                self._spill_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._spill_path = None

        if flush_error is not None:
            raise flush_error

    def _ensure_spill_fd(self) -> int:
        """Open the spill fd lazily; return it.

        Raises if there's no local path to open against. Remote
        (non-local) paths don't have an fd backing — they're
        buffered in ``_buf`` and flushed via :meth:`flush`. Callers
        that need an fd against a remote path should rethink the
        operation; positional fd ops aren't representable.
        """
        if self._spill_fd is not None:
            return self._spill_fd
        if self._spill_path is None:
            raise RuntimeError("BytesIO has no spill path to open an fd against")
        if not self._spill_path.is_local:
            raise RuntimeError(
                f"Cannot open an fd against non-local path "
                f"{self._spill_path.full_path()!r}; the buffer is "
                "memory-backed and flushes via write_bytes on close."
            )
        # Reuse _acquire's open shape so flags / parents stay consistent.
        flags = _flags_for_mode(self._mode)
        fd = os.open(self._spill_path.full_path(), flags, 0o644)
        self._spill_fd = fd
        return fd

    # ------------------------------------------------------------------
    # Three core primitives — only branch sites for memory vs spilled
    # ------------------------------------------------------------------

    def _slice(self, pos: int, n: int) -> bytes:
        """Read *n* bytes at *pos*. Handles all three backings.

        - Local path → ``os.pread`` against the spill fd.
        - Non-local path → delegate to the transaction buffer,
          which is itself a ``BytesIO`` (memory or local-spilled).
        - No path → read from ``_buf``.
        """
        if n <= 0:
            return b""
        if pos < 0:
            raise ValueError("slice position must be >= 0")

        # Local-path fast path: positional read against the fd.
        if self._spill_path is not None and self._spill_path.is_local:
            if self._spill_fd is not None:
                return _pread_bounded(self._spill_fd, n, pos)
            # fd not yet open (e.g. _init_from_bytes spilled but
            # acquire hasn't run yet). Open lazily.
            fd = self._ensure_spill_fd()
            return _pread_bounded(fd, n, pos)

        # Non-local path: delegate to the transaction buffer.
        if self._transaction_buffer is not None:
            return self._transaction_buffer._slice(pos, n)

        # Pure memory: read from _buf.
        if self._buf is not None:
            end = min(pos + n, self._size)
            if pos >= end:
                return b""
            return bytes(memoryview(self._buf)[pos:end])

        return b""

    def _write_at(self, data: memoryview, pos: int) -> int:
        """Write *mv* at *pos*. Grows backing, auto-spills on threshold.

        Three dispatch shapes:

        - **Local path** → positional ``os.pwrite`` against the fd.
        - **Non-local path** → delegate to the transaction buffer.
          Auto-spill happens inside the transaction buffer if it
          crosses ITS threshold; the outer buffer doesn't auto-spill
          because the path is the durable backing.
        - **No path** → write into ``_buf``; auto-spill if the
          projected size crosses ``spill_bytes``.

        Critical fix vs. previous version: if a memory-mode write
        crosses ``spill_bytes`` and triggers ``_spill()``, the same
        call falls through to the spill-fd write branch.
        """
        if pos < 0:
            raise ValueError("write position must be >= 0")

        n = len(data)
        if n == 0:
            return 0

        # Auto-spill check fires only for autonomous (no-path) memory
        # buffers. A buffer already bound to a path — local or
        # non-local — never spills at this layer; the path IS the
        # durable backing. (For non-local, the inner transaction
        # buffer handles its own spill behavior.)
        if self._buf is not None and self._spill_path is None:
            projected = max(self._size, pos + n)
            if projected > self._spill_bytes:
                self._spill()

        # Local-path fast path: positional write through the fd.
        if self._spill_path is not None and self._spill_path.is_local:
            fd = self._ensure_spill_fd()
            written = _pwrite_bounded(fd, data, pos)
            self._size = max(self._size, pos + written)
            return written

        # Non-local path: delegate to the transaction buffer.
        if self._transaction_buffer is not None:
            written = self._transaction_buffer._write_at(data, pos)
            self._size = max(self._size, pos + written)
            return written

        # Pure memory: splice into _buf.
        if self._buf is not None:
            need = pos + n
            if need > len(self._buf):
                new_cap = max(need, int(len(self._buf) * 1.5) + 1)
                self._buf.extend(b"\x00" * (new_cap - len(self._buf)))
            memoryview(self._buf)[pos:need] = _as_contiguous_mv(data)
            self._size = max(self._size, need)
            return n

        raise RuntimeError(f"Cannot write to {self!r}: no backing")

    def _set_size(self, n: int) -> int:
        """Truncate or extend backing to exactly *n* bytes.

        - **Local path** → ``os.ftruncate`` on the fd.
        - **Non-local path** → resize the transaction buffer.
          Remote file gets rewritten on flush.
        - **No path** → resize ``_buf``.

        Spill is one-way — truncate on a spilled buffer truncates
        the file, never demotes back to memory.
        """
        if n < 0:
            raise ValueError("Negative size value")

        # Local-path fast path: ftruncate on the fd.
        if self._spill_path is not None and self._spill_path.is_local:
            fd = self._ensure_spill_fd()
            os.ftruncate(fd, n)
            self._size = n
            if self._pos > n:
                self._pos = n
            return n

        # Non-local path: delegate to the transaction buffer.
        if self._transaction_buffer is not None:
            self._transaction_buffer._set_size(n)
            self._size = n
            if self._pos > n:
                self._pos = n
            return n

        # Pure memory: resize _buf.
        if self._buf is not None:
            if n < self._size:
                self._size = n
            else:
                if n > len(self._buf):
                    self._buf.extend(b"\x00" * (n - len(self._buf)))
                self._size = n
            if self._pos > n:
                self._pos = n
            return n

        # Neither backing — synthesize a memory backing of n zero
        # bytes. Reachable only after a botched reset; defensive.
        self._buf = bytearray(b"\x00" * n)
        self._size = n
        self._pos = min(self._pos, n)
        return n

    # ------------------------------------------------------------------
    # Spill helper — also opens the fd so post-spill writes work
    # ------------------------------------------------------------------

    def _spill(self) -> None:
        """Move the in-memory payload to a fresh owned temp file.

        After this returns, ``_buf`` is None, ``_spill_path`` is set,
        ``_spill_fd`` is open. The caller can immediately use
        positional fd ops without a second open.
        """
        if self._buf is None:
            return  # Already spilled, or no backing.

        path = _mint_spill_path(self._ext_hint(), self._spill_ttl)
        # Write the in-memory payload to disk in one shot.
        with open(path.full_path(), "wb") as fh:
            if self._size:
                fh.write(memoryview(self._buf)[: self._size])

        self._buf = None
        self._spill_path = path
        self._owns_spill_path = True

        # Open the fd for the new file. Use rb+ so subsequent reads
        # AND writes work regardless of the original mode — the spill
        # is internal scratch we own. Mode preservation matters only
        # for the path-bound case, where the user picked the mode.
        flags = _flags_for_mode("rb+")
        self._spill_fd = os.open(path.full_path(), flags, 0o644)

    def _ext_hint(self) -> str:
        """File extension suggestion for spill files."""
        mt = self._media_type
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

        Three shapes:

        - **Local fd-mode** → fresh ``os.fstat`` on the fd. The fd
          reflects every write (positional and otherwise), so the
          fstat is authoritative.
        - **Remote-buffered** → ``_size`` from ``_buf``. Does NOT
          re-stat the remote path: the working in-memory copy is
          what this buffer holds; the remote file may differ until
          flush.
        - **Memory-mode** → ``_size``.

        For "what's on the remote right now", use :meth:`stat`,
        which is always live.
        """
        if (
            self._spill_path is not None
            and self._spill_path.is_local
            and self._spill_fd is not None
        ):
            try:
                self._size = os.fstat(self._spill_fd).st_size
            except OSError:
                pass
        return self._size
    
    def is_empty(self):
        return self.size == 0

    def remaining_bytes(self):
        return self.size - self.tell()

    @property
    def mtime(self) -> float:
        """Modification time of the buffer's working backing.

        Same dispatch as :attr:`size`: live ``fstat`` for local
        fd-mode, in-memory ``_mtime`` for remote-buffered (the
        remote file's mtime is exposed via :meth:`stat` if the
        caller wants it).
        """
        if (
            self._spill_path is not None
            and self._spill_path.is_local
            and self._spill_fd is not None
        ):
            try:
                return os.fstat(self._spill_fd).st_mtime
            except OSError:
                pass
        return self._mtime or time.time()

    @property
    def path(self) -> "Path | None":
        """Spill file path, or ``None`` when in memory."""
        return self._spill_path

    @property
    def url(self) -> URL:
        """URL of the buffer's working backing."""
        if self._spill_path is not None:
            return self._spill_path.url
        return URL.from_memory_address(self)

    @property
    def is_writing(self) -> bool:
        """True when the open mode includes any write semantics."""
        return any(c in self._mode for c in "wa+x")

    @property
    def name(self) -> str:
        if self._spill_path is not None:
            return self._spill_path.full_path()
        return "<memory>"

    @property
    def is_local(self) -> bool:
        if self._spill_path is not None:
            return self._spill_path.is_local
        return True

    @property
    def is_remote(self) -> bool:
        return False

    def need_spill(self, size: int) -> bool:
        """Predicate: would *size* trigger a spill at current threshold?"""
        if not self._spill_bytes:
            return False
        if self._spill_bytes <= 0:
            return True
        return size >= self._spill_bytes

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

        Two-mode behaviour driven by ``_owns_spill_path``:

        - **Owned spill (or memory)** — tear down the old backing
          (close fd, unlink owned spill, drop bytearray), reset
          per-instance state, re-init from *payload*.
        - **External path binding** (path-bound buffer) — keep the
          binding. Truncate the bound file to zero, write the new
          payload's bytes through the fd, leave ``_spill_path`` /
          ``_owns_spill_path`` intact.

        Cursor resets to 0. ``_media_type`` is intentionally NOT
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
        if self._spill_path is not None and not self._owns_spill_path:
            if self._spill_path.is_local:
                # Local: truncate via fd, rewind, write new payload
                # through the fd via _write_bytes_io.
                fd = self._ensure_spill_fd()
                os.ftruncate(fd, 0)
                self._size = 0
                self._pos = 0
                self._buf = None  # Local path-bound: no buf, fd does work.
            else:
                # Non-local: truncate the transaction buffer to 0.
                # Flush on close will commit the (eventually-written)
                # new payload to the path via pwrite + truncate.
                if self._transaction_buffer is None:
                    # Defensive — acquire should have made one.
                    self._transaction_buffer = type(self)(
                        spill_bytes=self._spill_bytes,
                        spill_ttl=self._spill_ttl,
                        mode="rb+",
                    )
                self._transaction_buffer._set_size(0)
                self._size = 0
                self._pos = 0

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
        # Tear down a transaction buffer if we have one — it may
        # have been left over from a prior path-bound shape.
        if self._transaction_buffer is not None:
            try:
                self._transaction_buffer.close()
            except Exception:
                pass
            self._transaction_buffer = None
        if self._spill_fd is not None:
            try:
                os.close(self._spill_fd)
            except OSError:
                pass
            self._spill_fd = None
        if self._spill_path is not None:
            try:
                self._spill_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._spill_path = None

        self._buf = bytearray()
        self._size = 0
        self._mtime = 0
        self._pos = 0
        self._owns_spill_path = True

        if payload is None:
            return
        self._init_from(payload, copy=False)
        # If _init_from spilled, the fd is open already. If it stayed
        # in memory, no fd to open. Either way, consistent state.

    # ------------------------------------------------------------------
    # Media type
    # ------------------------------------------------------------------

    @property
    def media_type(self) -> MediaType:
        if self._media_type is None:
            if self._spill_path is not None:
                parsed = MediaType.from_path(self._spill_path, default=None)
                if parsed is not None and not parsed.mime_type.is_any_bytes:
                    self._media_type = parsed
                    return self._media_type

            parsed = MediaType.from_io(self, default=None)
            if parsed is not None and not parsed.mime_type.is_any_bytes:
                self._media_type = parsed
                return self._media_type

            return MediaType(MimeTypes.OCTET_STREAM)
        elif self._media_type.mime_type.is_any_bytes:
            self._media_type = None
            return self.media_type
        return self._media_type

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

        current = self._media_type
        if current is None or self.size == 0:
            self._media_type = parsed
            return self
        if current == parsed:
            return self
        if current.mime_type == parsed.mime_type:
            self._media_type = parsed
            return self

        raise ValueError(
            f"Cannot change media type from {current!r} to {parsed!r} on "
            f"a non-empty buffer ({self.size} bytes). The mime type differs, "
            "which would reinterpret the byte content as a different structural "
            "format. Replace the buffer's content first or construct a new "
            "BytesIO with the desired media type."
        )

    def with_mime_type(self, mime_type: MimeType, *, copy: bool = False):
        if self._media_type is None:
            media_type = MediaType.from_mime(mime_type)
        else:
            media_type = self._media_type.with_mime_type(mime_type)
        return self.with_media_type(media_type, copy=copy)

    def stat(self) -> PathStats:
        """Live snapshot of the **backing's** state.

        Always re-fetches; never cached. Distinct from :attr:`size`
        / :attr:`mtime`, which reflect the **buffer's** working
        state. The two can differ for remote-buffered backings:
        ``stat()`` returns whatever the remote file looks like
        right now; ``size``/``mtime`` reflect the in-memory working
        copy.

        Three shapes:

        - **Local fd-mode** → fresh ``os.fstat`` on the fd. Equivalent
          to a ``stat()`` syscall against the path, but cheaper
          (no path resolution).
        - **Remote-buffered** → live ``self._spill_path.stat()`` —
          one round-trip to the remote per call. May report
          ``MISSING`` if the remote file hasn't been flushed yet.
        - **Memory-mode** → synthesize a SOCKET-kind PathStats from
          the buffer's own ``_size``/``_mtime``.
        """
        if self._spill_path is not None:
            if self._spill_path.is_local and self._spill_fd is not None:
                # Local fd: fstat is the cheap, authoritative path.
                try:
                    raw = os.fstat(self._spill_fd)
                    return PathStats(
                        size=raw.st_size,
                        mtime=raw.st_mtime,
                        kind=PathKind.FILE,
                        mode=raw.st_mode,
                    )
                except OSError:
                    pass
            # Local without an open fd, or remote — go through the
            # path's stat. Live every call.
            return self._spill_path.stat()

        # Memory-mode — synthesize.
        return PathStats(
            size=self._size,
            mtime=self._mtime or time.time(),
            kind=PathKind.SOCKET,
            mode=0,
        )

    def as_media(self, media_type: MediaType | None = None) -> "TabularIO":
        return tabular_io_class()(self, media_type=media_type, copy=False)

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
        owned = "owned" if self._owns_spill_path else "external"
        if self._spill_path is not None:
            handle = "open" if self._spill_fd is not None else "closed"
            return (
                f"<{type(self).__name__} [spilled/{owned}/{handle}] "
                f"size={self._size} pos={self._pos} "
                f"mode={self._mode!r} path={self._spill_path!r}>"
            )
        if self._buf is None:
            return f"<{type(self).__name__} [empty] size=0 pos=0>"
        return (
            f"<{type(self).__name__} [memory] size={self._size} "
            f"pos={self._pos}>"
        )

    def __getstate__(self):
        """Pickle differently for owned vs path-bound buffers.

        - **Owned** — snapshot the bytes (zstd-compress past
          threshold). The path is internal scratch, not durable.
        - **Path-bound** — pickle the path + mode. The path IS the
          data; snapshotting bytes would diverge from the file the
          moment any other writer touched it.
        """
        if not self._owns_spill_path and self._spill_path is not None:
            # Pickle the path as a STRING, not the Path object. Path
            # subclasses inherit from Disposable, which holds a
            # WeakSet whose internal _remove callback is a closure
            # pickle can't serialize. Storing the str form sidesteps
            # the whole Disposable graph; we rebuild a Path on load
            # via path_class().from_().
            return {
                "kind": "path",
                "path": self._spill_path.full_path(),
                "mode": self._mode,
                "media_type": self._media_type,
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
            "media_type": self._media_type,
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

    def flush(self) -> None:
        """Push buffered writes to the backing.

        - **Local path / no path**: no-op. Local writes already hit
          the kernel via ``os.pwrite``; in-memory buffers have
          nowhere to flush to.
        - **Non-local path**: commit the transaction buffer to the
          path via ``path.write_bytes(payload)`` — single round-trip,
          implicit truncate-and-replace semantics. Skipped when the
          buffer was opened read-only.

        Earlier drafts used ``pwrite(0) + truncate(size)`` here, which
        is correct but pessimistic for backends that implement
        ``pwrite`` as read-modify-write (Databricks, S3, …). For
        those, writing ``payload`` at offset 0 reads the whole remote
        file, splices, uploads — then truncate reads it again. The
        simpler ``write_bytes`` is one upload, naturally truncating,
        and matches the "the entire transaction buffer is the new
        file content" semantic that's actually being expressed.
        """
        if self._spill_path is None:
            return  # Memory-mode — nothing durable to flush to.

        if self._spill_path.is_local:
            return  # Local fd ops bypass any user-space buffer.

        # Non-local path. Flush the transaction buffer if writable.
        if not self.is_writing:
            return

        if self._transaction_buffer is None:
            # Defensive — non-local acquire always sets the buffer.
            return

        payload = self._transaction_buffer.to_bytes()
        self._spill_path.write_bytes(payload)

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        """Underlying spill fd. Raises if there's no fd to expose."""
        if self._spill_path is None:
            raise OSError("BytesIO has no underlying file descriptor")
        return self._ensure_spill_fd()

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

    def tell(self) -> int:
        return int(self._pos)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Seek. Yggdrasil extension: SEEK_SET with negative offset
        counts from end."""
        offset = int(offset)
        size = self.size
        if whence == io.SEEK_SET:
            if offset < 0:
                new_pos = size + offset + 1
                if new_pos < 0:
                    raise ValueError(
                        f"Negative SEEK_SET offset {offset!r} is past the "
                        f"start of a {size}-byte buffer"
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
        return self.write_bytes(bytes(b))

    def write_bytes(self, b: bytes | bytearray | memoryview) -> int:
        mv = memoryview(b)
        if len(mv) == 0:
            return 0
        n = self._write_at(mv, self._pos)
        self._pos += n
        return n

    def _write_bytes_io(
        self,
        buffer: "BytesIO",
        batch_size: int = 1024 * 1024,
    ) -> int:
        """Drain *buffer* into self at the current cursor.

        Dispatches on the (source, destination) backing pair to one
        of four helpers. Returns bytes written. Advances ``self._pos``
        only; ``buffer._pos`` is untouched. ``buffer is self`` raises.
        """
        if buffer is self:
            raise ValueError("Cannot _write_bytes_io a BytesIO into itself")

        if not buffer.opened:
            with buffer:
                return self._write_bytes_io(buffer, batch_size=batch_size)

        src_size = buffer.size
        if src_size == 0:
            return 0

        src_in_memory = buffer._buf is not None
        dst_in_memory = self._buf is not None

        # Pre-flight spill: a memory dst whose projected size crosses
        # threshold spills BEFORE the copy so helpers stay on a stable
        # backing throughout.
        if dst_in_memory:
            projected = max(self._size, self._pos + src_size)
            if projected > self._spill_bytes:
                self._spill()
                dst_in_memory = False

        if src_in_memory and dst_in_memory:
            return self._copy_mem_to_mem(buffer)
        if src_in_memory and not dst_in_memory:
            return self._copy_mem_to_spilled(buffer)
        if not src_in_memory and dst_in_memory:
            return self._copy_spilled_to_mem(buffer, batch_size=batch_size)
        return self._copy_spilled_to_spilled(buffer, batch_size=batch_size)

    def _copy_mem_to_mem(self, buffer: "BytesIO") -> int:
        src_size = buffer._size
        if src_size == 0:
            return 0
        mv = memoryview(buffer._buf)[:src_size]
        n = self._write_at(mv, self._pos)
        self._pos += n
        return n

    def _copy_mem_to_spilled(self, buffer: "BytesIO") -> int:
        src_size = buffer._size
        if src_size == 0:
            return 0
        mv = memoryview(buffer._buf)[:src_size]
        n = self._write_at(mv, self._pos)
        self._pos += n
        return n

    def _copy_spilled_to_mem(self, buffer: "BytesIO", *, batch_size: int) -> int:
        src_size = buffer.size
        src_fd = buffer._ensure_spill_fd()
        if src_size == 0:
            return 0

        total = 0
        src_pos = 0
        while src_pos < src_size:
            want = min(batch_size, src_size - src_pos)
            chunk = _pread_bounded(src_fd, want, src_pos)
            if not chunk:
                break
            written = self._write_at(memoryview(chunk), self._pos)
            if written == 0:
                break
            self._pos += written
            total += written
            src_pos += written
        return total

    def _copy_spilled_to_spilled(self, buffer: "BytesIO", *, batch_size: int) -> int:
        src_size = buffer.size
        if src_size == 0:
            return 0
        src_fd = buffer._ensure_spill_fd()
        dst_fd = self._ensure_spill_fd()

        total = 0
        src_pos = 0
        while src_pos < src_size:
            want = min(batch_size, src_size - src_pos)
            chunk = _pread_bounded(src_fd, want, src_pos)
            if not chunk:
                break
            written = _pwrite_bounded(dst_fd, chunk, self._pos)
            if written == 0:
                break
            self._size = max(self._size, self._pos + written)
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

        with p.open_io(mode="wb") as fh:
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
        h.update(self.memoryview())
        return h

    def xxh3_int64(self) -> int:
        u = self.xxh3_64().intdigest()
        return u if u < 2**63 else u - 2**64

    def blake3(self) -> "blake3.blake3":
        from blake3 import blake3
        h = blake3(max_threads=blake3.AUTO)

        # Non-local: delegate. The transaction buffer is itself a
        # BytesIO; its own blake3 picks the right primitive (mmap
        # for local-spilled, bytearray hash for memory).
        if self._transaction_buffer is not None:
            return self._transaction_buffer.blake3()

        # Local-spilled fast path: update_mmap reads via mmap on the
        # file directly, no Python-level copy.
        if (
            self._spill_path is not None
            and self._spill_path.is_local
            and self._buf is None
        ):
            h.update_mmap(self._spill_path.full_path())
            return h

        # Memory-mode: hash the bytearray.
        if self._buf is not None and self._size:
            h.update(memoryview(self._buf)[: self._size])
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
    ):
        from .bytes_view import BytesIOView

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
        return BytesIOView(parent=self, start=pos, size=size, pos=0, max_size=max_size)

    def memoryview(self):
        """Return a ``memoryview`` over this buffer's bytes.

        - Memory mode (no path) → direct bytearray view.
        - Non-local path → delegate to the transaction buffer's
          memoryview (the inner buffer's own dispatch handles its
          memory-vs-spilled case).
        - Local-spilled mode → read-only mmap over the fd.
        - Empty → empty memoryview.

        For local-spilled mode, the returned mmap-backed view's
        lifetime is tied to the GC of the underlying mmap object.
        Release the view before closing the BytesIO if your platform
        is strict about closing mapped files.
        """
        # Non-local: delegate to the transaction buffer.
        if self._transaction_buffer is not None:
            return self._transaction_buffer.memoryview()

        if self._buf is not None:
            return memoryview(self._buf)[: self._size]

        if self._spill_path is not None and self._spill_path.is_local:
            size = self.size
            if size == 0:
                return memoryview(b"")
            fd = self._ensure_spill_fd()
            return memoryview(mmap.mmap(fd, size, access=mmap.ACCESS_READ))

        return memoryview(b"")

    def to_bytes(self) -> bytes:
        if self._buf is not None:
            return bytes(memoryview(self._buf)[: self._size])
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
        target_mt = self.media_type.with_codec(c) if self._media_type else None
        payload = c.compress(self)
        if copy:
            payload._media_type = target_mt
            return payload
        self.replace_with_payload(payload)
        self._media_type = target_mt
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
        self._media_type = payload._media_type
        return self

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def json_load(self, *, media_type: Optional[MediaType] = None):
        media_type = media_type or self.media_type
        if media_type is not None and media_type.codec is not None:
            with self.decompress(codec=media_type.codec, copy=True) as decompressed:
                return decompressed.json_load(media_type=media_type.with_codec(None))
        with self.view(pos=0) as v:
            return json_module.load(v)

    def arrow_io(self, mode: str = "rb", size: int | None = None):
        if self._transaction_buffer is not None:
            return self._transaction_buffer.arrow_io(mode, size)

        if (
            self.spilled
            and self._spill_path is not None
            and self._spill_path.is_local
        ):
            if size is not None:
                return pa.create_memory_map(self._spill_path.full_path(), size)
            return pa.OSFile(self._spill_path.full_path(), mode)

        if mode in {"a", "ab"}:
            self.seek(0, io.SEEK_END)
            mode = mode.replace("a", "w")
        return pa.PythonFile(self, mode=mode)

    def clear(self):
        self.close(force=True)

        if self._transaction_buffer is not None:
            self._transaction_buffer.clear()
            self._transaction_buffer = None

        self._buf = bytearray()
        self._size = 0
        self._pos = 0
        self._spill_path = None
        self._media_type = None


# ===========================================================================
# Module-level helpers
# ===========================================================================


def _pread_bounded(fd: int, n: int, pos: int) -> bytes:
    """:func:`os.pread` with a Windows fallback.

    ``os.pread`` is POSIX-only and not exposed on the Windows build
    of CPython. Falls back to lseek + read, restoring the cursor
    afterward so concurrent positional ops on the same fd don't
    fight. Not thread-safe in the fallback path.
    """
    if _HAS_PREAD:
        return os.pread(fd, n, pos)
    # Fallback: lseek + read. Preserves cursor so concurrent ops on
    # the same fd don't fight, but is NOT thread-safe.
    saved = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, pos, os.SEEK_SET)
        return os.read(fd, n)
    finally:
        try:
            os.lseek(fd, saved, os.SEEK_SET)
        except OSError:
            pass


def _pwrite_bounded(fd: int, data, pos: int) -> int:
    """:func:`os.pwrite` with a Windows fallback.

    Symmetric to :func:`_pread_bounded`. ``os.pwrite`` is POSIX-only;
    on Windows we ``lseek`` + ``write`` and restore the cursor.
    Coerces *data* to a contiguous :class:`memoryview` here so call
    sites can pass any buffer-protocol input without pre-coercion.

    Not thread-safe in the fallback path — concurrent positional
    writes on the same fd race for the cursor.
    """
    mv = _as_contiguous_mv(memoryview(data))
    if _HAS_PWRITE:
        return os.pwrite(fd, mv, pos)
    # Fallback for Windows et al.
    saved = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, pos, os.SEEK_SET)
        return os.write(fd, mv)
    finally:
        try:
            os.lseek(fd, saved, os.SEEK_SET)
        except OSError:
            pass


def _mint_spill_path(ext: str, ttl_seconds: int) -> "Path":
    """Mint a fresh temp file path under :func:`tempfile.gettempdir`.

    The TTL is encoded in the filename so external cleanup workers
    can delete stale files lexically without re-stat'ing every entry.
    The file itself is not created here — the caller writes to it.
    """
    seed = os.urandom(8).hex()
    start = int(time.time())
    end = start + ttl_seconds
    name = f"tmp-{seed}-{start}-{end}.{ext}"
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
