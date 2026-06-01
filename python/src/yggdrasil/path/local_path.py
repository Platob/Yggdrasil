"""Local-filesystem :class:`Path` — fd-backed byte holder at a path.

A :class:`LocalPath` is a :class:`Path` over the ``file://`` scheme.
It owns a long-lived RDWR fd opened on :meth:`_acquire`, routing
positional I/O through portable :func:`os.pread` / :func:`os.pwrite`
helpers (POSIX-fast, Windows-safe via ``lseek + read/write`` with
saved-cursor restore). The fd-fast path overrides :class:`Path`'s
default whole-file primitives (which would round-trip through
:meth:`_bread` / :meth:`_bwrite`).

Lifecycle
---------

The fd is opened in :meth:`_acquire` (driven by ``Disposable.open`` /
``__enter__``) and closed in :meth:`_release`. Outside that window
the holder is a quiet inert URL — :attr:`size` reads via
:func:`os.stat`; positional read / write raise.

Smart-open semantics on first acquire:

- **Trailing-slash URL → directory.** A path that ends in ``/`` is
  unambiguously a directory; we stage a fresh file under it without
  touching :func:`os.open` first.
- **Path is a directory → stage a child.** :func:`os.open` raises
  :class:`IsADirectoryError` (POSIX) / :class:`PermissionError`
  (Windows); we mint a unique ``part-{epoch_ms}-{seed}`` file under
  the directory, take ownership via ``temporary=True``, retry.
- **Parent directory missing.** :class:`FileNotFoundError` triggers
  one ``makedirs`` + retry. Hot path stays at one syscall.

The URL rebind from the staging step is sticky: a close/reopen
cycle reuses the same staging file, not a new one.

Anonymous staging
-----------------

A bare ``LocalPath()`` (no path / url / data) auto-stages a
``part-{epoch_ms}-{seed}`` file under
``{tempfile.gettempdir()}/yggdrasil-staging/`` and flips
``temporary=True`` so the file is unlinked on close. Useful as a
spill target, scratch buffer, or atomic-write source.
:meth:`LocalPath.staging_path` is the explicit entry point.

Once per process, on the first staging-path mint, the staging dir
is swept of entries older than 24 hours — keeps the dir from
accumulating after crashes / dirty exits.

Filesystem-shape concerns (listing, mkdir, rename, mmap, copy
acceleration) are implemented per :class:`Path` contract via the
backend hooks below.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from typing import Any, ClassVar, Iterator

from yggdrasil.enums import Mode, Scheme
from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.io.holder import IO
from yggdrasil.path import Path
from yggdrasil.io.io_stats import IOKind, IOStats

__all__ = ["LocalPath"]


# ---------------------------------------------------------------------------
# Portable positional I/O — POSIX uses os.pread/os.pwrite directly,
# Windows falls back to lseek+read/write loops with saved-cursor restore.
# ---------------------------------------------------------------------------

_HAS_PREAD = hasattr(os, "pread")
_HAS_PWRITE = hasattr(os, "pwrite")


def _fd_pread(fd: int, n: int, pos: int) -> bytes:
    """Read up to *n* bytes at *pos* from *fd*. POSIX-fast, Windows-safe.

    On POSIX, loops :func:`os.pread` to handle short reads from signal
    interruption. On Windows (no ``os.pread``), saves the fd's current
    cursor, seeks to *pos*, reads, restores the cursor — so concurrent
    positional ops on the same fd don't see each other's seeks.
    Returns fewer than *n* bytes only on EOF.
    """
    if n <= 0:
        return b""
    if _HAS_PREAD:
        chunks: list[bytes] = []
        remaining = n
        offset = pos
        while remaining > 0:
            chunk = os.pread(fd, remaining, offset)
            if not chunk:
                break
            chunks.append(chunk)
            got = len(chunk)
            remaining -= got
            offset += got
        return b"".join(chunks)

    saved = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, pos, os.SEEK_SET)
        chunks2: list[bytes] = []
        remaining = n
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks2.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks2)
    finally:
        try:
            os.lseek(fd, saved, os.SEEK_SET)
        except OSError:
            pass


def _fd_pwrite(fd: int, mv: memoryview, pos: int) -> int:
    """Write exactly ``len(mv)`` bytes at *pos* to *fd*.

    POSIX uses :func:`os.pwrite`; Windows falls through to ``lseek +
    write`` with saved-cursor restore.
    """
    n = len(mv)
    if n == 0:
        return 0
    if _HAS_PWRITE:
        total = 0
        while total < n:
            written = os.pwrite(fd, bytes(mv[total:]), pos + total)
            if written == 0:
                raise BlockingIOError(f"Short write at offset {pos + total}")
            total += written
        return total

    saved = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, pos, os.SEEK_SET)
        total = 0
        while total < n:
            written = os.write(fd, bytes(mv[total:]))
            if written == 0:
                raise BlockingIOError(f"Short write at offset {pos + total}")
            total += written
        return total
    finally:
        try:
            os.lseek(fd, saved, os.SEEK_SET)
        except OSError:
            pass


def _default_open_flags() -> int:
    """Standard RDWR | CREAT flag set, plus CLOEXEC / BINARY when
    available.

    Centralized so :meth:`LocalPath._acquire`, :meth:`LocalPath.truncate`,
    and the transient-open paths in :meth:`_bread` / :meth:`_bwrite`
    all use the same flag bag.
    """
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_BINARY"):  # Windows
        flags |= os.O_BINARY
    return flags


def _open_with_mkdir_retry(path: str, flags: int, mode: int = 0o644) -> int:
    """:func:`os.open` with a single ``makedirs`` + retry on missing parent.

    Hot path is one syscall; cold path (parent doesn't exist) is
    ``mkdir`` + a second open. Any other error type — and any
    second-attempt failure — propagates unchanged.
    """
    try:
        return os.open(path, flags, mode)
    except FileNotFoundError:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return os.open(path, flags, mode)


# ---------------------------------------------------------------------------
# Staging directory — process-shared scratch space for anonymous /
# transactional file holders. One sweep of stale entries per process.
# ---------------------------------------------------------------------------

#: Subdirectory under :func:`tempfile.gettempdir` used for staging
#: files. Shared across all processes on the host; the once-per-
#: process sweep keeps it from accumulating indefinitely.
_STAGING_SUBDIR = "yggdrasil-staging"

#: Files older than this (in seconds) get unlinked on the first
#: :meth:`LocalPath.staging_path` call of a process.
_STAGING_TTL_SECONDS = 86_400  # 1 day

_SWEEP_LOCK = threading.Lock()
_SWEEP_DONE = False


def _staging_dir() -> str:
    """The yggdrasil staging directory under the system tmpdir.

    Cheap to call repeatedly: :func:`os.makedirs` with ``exist_ok=True``
    short-circuits when the directory already exists.
    """
    path = os.path.join(tempfile.gettempdir(), _STAGING_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _sweep_staging_once() -> None:
    """Unlink staging files older than :data:`_STAGING_TTL_SECONDS`.

    Runs at most once per process. Thread-safe via a module-level
    lock — concurrent first-callers serialize on the lock and the
    second-and-onwards see the ``_SWEEP_DONE`` flag and return.
    Failures (permission denied, race with another process) are
    swallowed: the sweep is opportunistic, never fatal.

    Filename convention: ``part-{epoch_ms}-{seed}``. The epoch_ms
    prefix is what we compare against the cutoff; entries that
    don't match the convention are left alone.
    """
    global _SWEEP_DONE
    if _SWEEP_DONE:
        return
    with _SWEEP_LOCK:
        if _SWEEP_DONE:
            return
        _SWEEP_DONE = True

        try:
            staging = _staging_dir()
            entries = os.listdir(staging)
        except OSError:
            return

        cutoff_ms = int((time.time() - _STAGING_TTL_SECONDS) * 1000)
        for name in entries:
            if not name.startswith("part-"):
                continue
            try:
                _, epoch_str, _seed = name.split("-", 2)
                if int(epoch_str) >= cutoff_ms:
                    continue
            except (ValueError, IndexError):
                continue
            try:
                os.remove(os.path.join(staging, name))
            except OSError:
                # Race with another process / permission denied —
                # skip and keep sweeping.
                continue


def _mint_staging_name() -> str:
    """``part-{epoch_ms}-{seed}`` — collision-free across concurrent writers.

    Millisecond timestamp gives lexical-time ordering for free
    (handy for debug listings); 8 bytes of urandom (~64 bits of
    entropy) makes within-millisecond collisions effectively
    impossible.
    """
    epoch_ms = int(time.time() * 1000)
    seed = os.urandom(8).hex()
    return f"part-{epoch_ms}-{seed}"


# ---------------------------------------------------------------------------
# LocalPath
# ---------------------------------------------------------------------------


class LocalPath(Path):
    """File-backed byte holder for the local filesystem.

    Construction shapes (in addition to those inherited from
    :class:`Path`):

    - ``LocalPath("/tmp/foo.bin")``      — bind to a path string.
    - ``LocalPath(pathlib.Path(...))``    — bind to a ``pathlib`` path.
    - ``LocalPath(url=URL("file:///…"))`` — bind to a ``file://`` URL.

    The fd is opened lazily on the first :meth:`open` /
    ``with``-block / ``acquire`` call. Until then the holder is a
    URL with a side of ``os.stat`` — :meth:`stat` works; read /
    write do not.
    """

    scheme: ClassVar[Scheme] = Scheme.FILE

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, data: Any = None, **kwargs: Any) -> None:
        # Slots must exist before super().__init__ runs — Disposable
        # may call _acquire (which reads self._fd) during construction.
        self._fd: int = -1

        # Auto-stage when no path / url / data is provided. Lets
        # ``LocalPath()`` produce an anonymous local file (handy as a
        # spill target or scratch buffer) and lets ``LocalPath(binary=...)``
        # work without a path argument. The minted URL goes into
        # ``kwargs["url"]`` so :class:`Holder.__init__` sees a bound URL
        # before its construction-priority dispatch reaches any
        # ``binary=`` / ``binary_io=`` initializer that might write.
        if (
            data is None
            and "path" not in kwargs
            and "url" not in kwargs
        ):
            kwargs["url"] = self._fresh_staging_path()
            kwargs.setdefault("temporary", True)

        super().__init__(data, **kwargs)

    @staticmethod
    def _fresh_staging_path() -> str:
        """Mint and return an absolute path under the staging dir.

        Triggers the once-per-process sweep on first call. Doesn't
        create the file — just reserves a name. The actual file
        materializes when the holder is acquired.
        """
        _sweep_staging_once()
        return os.path.join(_staging_dir(), _mint_staging_name())

    @classmethod
    def staging_path(cls) -> "LocalPath":
        """Return a fresh :class:`LocalPath` over a staging file.

        The returned holder is closed (un-acquired) and marked
        ``temporary=True`` — closing it unlinks the file. Triggers
        the once-per-process sweep of stale staging entries (>1 day
        old) on the first call of the process; subsequent calls
        skip straight to name minting.

        Useful as a one-liner for scratch buffers, atomic-write
        targets that get renamed into place on success, or any
        "give me a local file I can write to" pattern that doesn't
        care about the filename.
        """
        return cls(url=cls._fresh_staging_path(), temporary=True)

    # ------------------------------------------------------------------
    # Backing-shape predicates — Path provides the others
    # ------------------------------------------------------------------

    @property
    def is_local_path(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def os_path(self) -> str:
        """The local filesystem path as a string. Alias for :meth:`full_path`.

        Always returns the OS-native form — backslashes on Windows,
        forward slashes elsewhere — matching :meth:`pathlib.Path.__fspath__`
        and what tools that compare against ``str(pathlib.Path(...))``
        expect. The underlying ``URL.__fspath__`` is POSIX-style by
        contract; we apply :func:`os.path.normpath` here so callers
        of ``LocalPath`` get the platform-canonical shape.
        """
        return os.path.normpath(os.fspath(self.url))

    def full_path(self) -> str:
        """Backend-native string form — :class:`Path` hook."""
        return self.os_path

    @property
    def fd(self) -> int:
        """The currently-bound fd, or ``-1`` when closed."""
        return self._fd

    # ------------------------------------------------------------------
    # Disposable lifecycle — own the fd
    # ------------------------------------------------------------------

    def _acquire(self) -> None:
        """Open a long-lived RDWR fd at the bound path.

        Three corners worth calling out:

        - **Trailing-slash URL → directory.** A path that ends in
          ``/`` is unambiguously a directory by URL convention.
          We stage a fresh file under it (same as the directory
          branch below) without bothering to ``os.open`` it first.
        - **Path is an existing directory → stage a child.** We
          can't open a directory for byte I/O, but the user's
          intent ("give me a writable file at this location") is
          clear enough: mint a unique ``part-{epoch_ms}-{seed}``
          file under the directory, take ownership via
          ``temporary=True`` (so it gets unlinked on close), and
          rebind ``self.url`` to point at the staging file. The
          rebind sticks across close/reopen cycles — only the
          first :meth:`_acquire` does the staging dance.
        - **Parent directory missing → mkdir, retry.** Handled by
          :func:`_open_with_mkdir_retry`. Hot path is one syscall;
          cold path is two.

        Existing files are opened in place without truncation —
        truncation is the caller's job via :meth:`truncate(0)`
        or :meth:`clear`.
        """
        if self._fd >= 0:
            return

        path = self.os_path
        if path.endswith(("/", os.sep)):
            self._stage_under_directory(path)
            path = self.os_path  # rebound

        flags = _default_open_flags()
        try:
            self._fd = _open_with_mkdir_retry(path, flags)
        except (IsADirectoryError, PermissionError, OSError):
            # POSIX raises IsADirectoryError when you try to open a
            # directory for byte I/O; Windows raises PermissionError
            # for the same situation. Disambiguate with stat: real
            # errors propagate, directories route through staging.
            if not os.path.isdir(path):
                raise
            self._stage_under_directory(path)
            self._fd = _open_with_mkdir_retry(self.os_path, flags)

    def _stage_under_directory(self, dir_path: str) -> None:
        """Rebind :attr:`url` to a fresh staging file under *dir_path*.

        Uses the standard :func:`_mint_staging_name` shape
        (``part-{epoch_ms}-{seed}``) so staging files everywhere in
        the system follow the same convention. Ensures the directory
        exists, sets :attr:`temporary` so the staging file is
        unlinked on close.

        After this runs, :attr:`os_path` points at the new staging
        file, so the caller's normal :func:`os.open` path picks up
        the rebound URL.
        """
        clean = dir_path.rstrip("/").rstrip(os.sep) or dir_path
        os.makedirs(clean, exist_ok=True)
        staged = os.path.join(clean, _mint_staging_name())
        self.url = staged
        self.temporary = True

    def _release(self) -> None:
        """Close the fd; honor :attr:`temporary` via the base hook."""
        if self._fd >= 0:
            fd, self._fd = self._fd, -1
            try:
                os.close(fd)
            except OSError:
                pass
        # Path._release → Holder._release: calls clear() when temporary.
        super()._release()

    # ==================================================================
    # Path hooks — filesystem surface
    # ==================================================================

    def _stat(self) -> IOStats:
        """One round-trip: kind + size + mtime via fstat (when open)
        or :func:`os.stat` (when closed). Returns ``MISSING`` for
        non-existent paths; never raises :class:`FileNotFoundError`.

        Each call builds a fresh :class:`IOStats` from the live
        filesystem state. The holder's stamped
        :attr:`Holder.media_type` (or the URL-inferred default) is
        merged in so the result is self-describing without a second
        lookup.
        """
        # ``self.media_type`` resolves the lazy ``_media_type`` slot —
        # a path-only holder skips the URL-mime walk at construction
        # and pays for it here on the first ``stat()`` instead.
        media_type = self.media_type
        try:
            if self._fd >= 0:
                st = os.fstat(self._fd)
            else:
                st = os.stat(self.os_path)
        except (FileNotFoundError, NotADirectoryError):
            return IOStats(
                size=0, mtime=0.0, kind=IOKind.MISSING,
                media_type=media_type,
            )

        return IOStats(
            size=int(st.st_size),
            mtime=float(st.st_mtime),
            mode=int(st.st_mode),
            kind=(
                IOKind.DIRECTORY if os.path.isdir(self.os_path)
                else IOKind.FILE
            ),
            media_type=media_type,
        )

    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator["LocalPath"]:
        """Yield children. Empty when missing or not a directory.

        ``singleton_ttl`` is accepted for signature compatibility with
        the base :class:`Path` contract; :class:`LocalPath` is not a
        :class:`Singleton`, so the kwarg has no effect here. Each
        :class:`os.DirEntry` already carries ``is_dir`` / ``is_file``
        / ``stat`` cheaply, so seed every child's stat slot off the
        DirEntry — that's the fast path the base :class:`Path` uses
        on ``is_file`` / ``is_dir`` / ``size`` / ``mtime`` calls so
        an iterdir-and-classify loop doesn't fire an extra
        :func:`os.stat` per child.
        """
        del singleton_ttl
        try:
            entries = list(os.scandir(self.os_path))
        except (FileNotFoundError, NotADirectoryError):
            return

        for entry in entries:
            child = self._from_url(entry.path)
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                st = entry.stat(follow_symlinks=False)
                # Skip the eager ``child.media_type`` lookup: every
                # caller that needs it (``_leaf_for``, ``stat()``)
                # resolves it lazily off ``URL.infer_media_type`` the
                # first time they read it, with the result cached on
                # the child's ``_media_type`` slot. Pre-computing it
                # during ``_ls`` ran a URL extension parse for every
                # scandir entry — even non-tabular siblings the
                # ``Folder._leaf_for`` cache already short-circuits.
                child._persist_stat_cache(IOStats(
                    size=int(st.st_size),
                    mtime=float(st.st_mtime),
                    mode=int(st.st_mode),
                    kind=IOKind.DIRECTORY if is_dir else IOKind.FILE,
                    media_type=None,
                ))
            except OSError:
                # Race: the entry vanished between scandir and stat.
                # Skip seeding — :meth:`_stat` falls back to a live
                # probe, which will report MISSING.
                is_dir = False
            yield child
            if recursive and is_dir:
                yield from child._ls(recursive=True)

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        """Create directory at this path."""
        if parents:
            os.makedirs(self.os_path, exist_ok=exist_ok)
        else:
            try:
                os.mkdir(self.os_path)
            except FileExistsError:
                if not exist_ok:
                    raise

    def _close_fd(self) -> None:
        """Force-close the underlying fd and reset lifecycle state.

        Bypasses :class:`Disposable.close` (which has reentrance
        protection that fires when called from inside a method
        already holding the disposable's state) and goes straight
        at the fd. Required for Windows compatibility — can't unlink
        a file with an open handle there. Also flips ``_acquired``
        to ``False`` so a subsequent :meth:`open` actually runs
        :meth:`_acquire` instead of seeing the holder as still-open
        and short-circuiting.

        Idempotent: a closed fd is fine; an already-released
        disposable stays released.
        """
        if self._fd >= 0:
            fd, self._fd = self._fd, -1
            try:
                os.close(fd)
            except OSError:
                pass
        # Sync lifecycle state with reality. Without this, the next
        # ``open()`` call would see ``_acquired == True`` and skip
        # ``_acquire``, leaving ``_fd == -1`` for follow-up ops.
        try:
            self._acquired = False
        except AttributeError:
            pass

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        """Unlink the file at this path. Force-closes the fd first."""
        self._close_fd()
        try:
            os.remove(self.os_path)
        except FileNotFoundError:
            if not missing_ok:
                raise

    def _remove_dir(
        self, recursive: bool, missing_ok: bool, wait: WaitingConfig,
    ) -> None:
        """Remove the directory at this path. Force-closes the fd first
        so a holder pointing at the directory (rare, but possible) doesn't
        block the rmtree on Windows.
        """
        self._close_fd()
        try:
            if recursive:
                shutil.rmtree(self.os_path)
            else:
                os.rmdir(self.os_path)
        except FileNotFoundError:
            if not missing_ok:
                raise

    def _bread(self, n: int, pos: int, mode: Mode) -> "IO":
        """Positional read → fresh :class:`IO` over a Memory holder.

        ``n < 0`` reads to EOF. Caller owns the returned buffer
        (must close it). The :class:`Path` default would route
        through here; LocalPath's :meth:`_read_mv` short-circuits
        to :func:`_fd_pread` directly so this is mostly a fallback
        for callers that explicitly want an IO instead of bytes.
        """
        del mode  # read is mode-agnostic
        if self._fd >= 0:
            size = self.size if n < 0 else n
            data = _fd_pread(self._fd, size, pos)
        else:
            # Transient open — read everything at pos, then close.
            flags = os.O_RDONLY
            if hasattr(os, "O_BINARY"):
                flags |= os.O_BINARY
            fd = os.open(self.os_path, flags)
            try:
                size = (
                    os.fstat(fd).st_size - pos if n < 0 else n
                )
                data = _fd_pread(fd, max(0, size), pos)
            finally:
                os.close(fd)

        return IO(data)

    def _bwrite(self, data: "IO", pos: int, mode: Mode) -> int:
        """Splice *data* at *pos* on the backing.

        ``mode`` honors :attr:`Mode.APPEND` by ignoring *pos* and
        landing at EOF; everything else writes positionally.
        :class:`LocalPath`'s :meth:`_write_mv` short-circuits to
        :func:`_fd_pwrite` directly, so this is the fallback for
        callers that hand in a full IO at once.
        """
        payload = data.read_bytes() if hasattr(data, "read_bytes") else bytes(data.to_bytes())
        if not payload:
            return 0

        owns_fd = self._fd < 0
        if owns_fd:
            fd = _open_with_mkdir_retry(self.os_path, _default_open_flags())
        else:
            fd = self._fd

        try:
            if mode is Mode.APPEND:
                pos = os.fstat(fd).st_size
            return _fd_pwrite(fd, memoryview(payload), pos)
        finally:
            if owns_fd:
                os.close(fd)

    # ==================================================================
    # Holder primitives — fd-fast overrides of Path's defaults
    # ==================================================================

    def _read_mv(self, n: int, pos: int) -> memoryview:
        """Positional read, fd-direct (transient open when un-acquired).

        :class:`Holder.read_mv` has already normalized ``(n, pos)``:
        ``0 <= pos <= size`` and ``0 <= n <= size - pos``. Hot path
        is the long-lived fd opened by :meth:`_acquire`; if no fd is
        bound (``read_bytes`` / ``pread`` on a closed holder) we open
        an O_RDONLY fd just for this call so casual reads "just work".
        """
        if n == 0:
            return memoryview(b"")
        if self._fd >= 0:
            return memoryview(_fd_pread(self._fd, n, pos))

        flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        fd = os.open(self.os_path, flags)
        try:
            return memoryview(_fd_pread(fd, n, pos))
        finally:
            os.close(fd)

    def _write_mv(self, data: memoryview, pos: int) -> int:
        """Positional write, fd-direct (transient open when un-acquired).

        :class:`Holder.write_mv` has pre-grown the visible size via
        :meth:`resize` → :meth:`truncate`, so the file is already at
        least ``pos + len(data)`` bytes long when an fd is bound.
        Without an fd, we open one for this call (creating the file
        if needed); :meth:`truncate` is a no-op on the closed-holder
        path, but :func:`_fd_pwrite` extends the file naturally as
        bytes land past EOF.
        """
        n = len(data)
        if n == 0:
            return 0
        if self._fd >= 0:
            return _fd_pwrite(self._fd, data, pos)

        fd = _open_with_mkdir_retry(self.os_path, _default_open_flags())
        try:
            return _fd_pwrite(fd, data, pos)
        finally:
            os.close(fd)

    def reserve(self, n: int) -> None:
        """No-op — local filesystems have no useful capacity-vs-size
        distinction. Files grow on write; ``posix_fallocate`` is an
        optimization, and most filesystems no-op it anyway.
        """
        if n < 0:
            raise ValueError(f"reserve size must be >= 0, got {n!r}")

    def truncate(self, n: int) -> int:
        """Set the file size to exactly ``n`` bytes via :func:`os.ftruncate`.

        Shrinks drop the tail; extends zero-pad. Overrides the
        :class:`Path` default (read-truncate-rewrite via _bread/_bwrite)
        with the direct syscall. Transient fd when the holder isn't
        acquired so ``truncate`` is usable on a fresh path.
        """
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        if self._fd >= 0:
            # File is open ⇒ it exists. Only issue the ``ftruncate`` when
            # the size actually changes — a no-op ``truncate(n)`` (the
            # common ``truncate(0)`` overwrite-prelude on a freshly-opened,
            # already-empty target) shouldn't pay a write syscall. The
            # stat cache is refreshed either way.
            if os.fstat(self._fd).st_size != n:
                os.ftruncate(self._fd, n)
            self._touch_stat(size=n)
            return n

        # Un-acquired: read the real on-disk size first (0 when the file
        # is missing). A no-op truncate skips the
        # ``open(O_CREAT)+ftruncate+close`` entirely.
        try:
            current = os.stat(self.os_path).st_size
        except (FileNotFoundError, NotADirectoryError):
            # Missing file: ``truncate(0)`` is already satisfied (no
            # bytes) — don't force-create it just to zero it; the
            # overwrite-prelude's following write materialises the file.
            # A larger truncate must still create the n-byte file.
            if n == 0:
                return 0
            current = -1

        if current == n:
            # Already exactly ``n`` bytes — keep the listing-seeded stat
            # cache in step without touching the backing.
            self._touch_stat(size=n)
            return n

        fd = _open_with_mkdir_retry(self.os_path, _default_open_flags())
        try:
            os.ftruncate(fd, n)
        finally:
            os.close(fd)
        # Keep a listing-seeded stat cache in step with the new size.
        self._touch_stat(size=n)
        return n

    def _clear(self) -> None:
        """Drop the payload — close the fd and unlink the file.

        Idempotent: a missing file is fine. After :meth:`_clear` the
        holder is closed; reopen via :meth:`open` to start fresh
        (the next :meth:`_acquire` recreates the file).
        """
        self._close_fd()
        try:
            os.remove(self.os_path)
        except FileNotFoundError:
            pass
        # The file is gone — drop any listing-seeded stat snapshot so the
        # next probe re-checks the filesystem instead of reporting the
        # cleared file as still present at its old size.
        self._stat_cached = None
        self._stat_cached_at = 0.0
