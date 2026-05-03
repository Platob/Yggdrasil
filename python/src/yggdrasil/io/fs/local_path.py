"""Local-filesystem :class:`Path` implementation.

Default fallback when no remote backend's :meth:`Path.handles`
claims an input. Talks directly to ``os.*`` for everything; the
rest of the abstract surface comes from :class:`Path`.

What's overridden vs. inherited
-------------------------------

Inherited unchanged: pure-path navigation (``parent``, ``joinpath``,
``__truediv__``, …), :meth:`ls` filtering, :meth:`walk`,
:meth:`read_bytes` / :meth:`write_bytes`, :meth:`read_text` /
:meth:`write_text`, :meth:`infer_media_type`, the Disposable
lifecycle (``_acquire`` is already a no-op in the base).

Overridden because the local FS has a cheaper primitive:

- :meth:`pread` / :meth:`pwrite` — :func:`os.pread` /
  :func:`os.pwrite` on a transient fd, single syscall per call.
  Required (these are abstract on :class:`Path` now, so every
  backend must implement them).
- :meth:`memoryview` — real :class:`mmap.mmap`, not a bytes copy.
- :meth:`open_mmap` — real :class:`mmap.mmap`.
- :meth:`copy_to` — :func:`shutil.copyfile` for same-backend
  (uses ``sendfile`` / ``copy_file_range`` on Linux 3.8+), falls
  through to the base streaming copy when ``dest`` lives on a
  different backend.
- :meth:`rename` — :func:`os.replace` for atomic same-FS rename,
  falls back to copy+remove on ``EXDEV``.
- :meth:`touch` — :func:`os.utime` + ``O_CREAT``. The base default
  uses ``write_bytes(b"")`` which would truncate an existing file.
- :meth:`is_symlink` — :func:`os.path.islink`.
- :meth:`resolve`, :meth:`absolute` — wired to :func:`os.path.realpath`
  and :func:`os.path.abspath` respectively.
- :meth:`truncate` — :func:`os.truncate` direct.
"""

from __future__ import annotations

import io
import mmap
import os
import shutil
import stat as stat_module
from pathlib import PurePath
from typing import Any, ClassVar, Iterator, Optional, Union

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.fs.path import Path, register_path_class
from yggdrasil.io.path_stat import PathKind, PathStats
from yggdrasil.io.url import URL

__all__ = ["LocalPath"]


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------

_HAS_PREAD = hasattr(os, "pread")
_HAS_PWRITE = hasattr(os, "pwrite")
_IS_WINDOWS = os.name == "nt"

_COPY_CHUNK = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Windows long-path support — apply the ``\\?\`` extended-length prefix at
# the syscall boundary so partitioned cache layouts that exceed the legacy
# ``MAX_PATH`` 260-char limit keep working without a registry / manifest
# opt-in. No-op on non-Windows. The prefix only flows out via
# :meth:`LocalPath._os_path` to ``os.*`` / ``shutil.*`` / :func:`open`;
# :meth:`full_path` stays clean for display, error messages, and URL math.
# ---------------------------------------------------------------------------

_LONG_PATH_PREFIX = "\\\\?\\"
_LONG_PATH_UNC_PREFIX = "\\\\?\\UNC\\"


def _to_long_path(p: str) -> str:
    if not _IS_WINDOWS or not p:
        return p
    if p.startswith(_LONG_PATH_PREFIX):
        return p
    abs_p = os.path.abspath(p).replace("/", "\\")
    if abs_p.startswith("\\\\"):
        return _LONG_PATH_UNC_PREFIX + abs_p[2:]
    return _LONG_PATH_PREFIX + abs_p


def _strip_long_path(p: str) -> str:
    if not _IS_WINDOWS or not p:
        return p
    if p.startswith(_LONG_PATH_UNC_PREFIX):
        return "\\\\" + p[len(_LONG_PATH_UNC_PREFIX):]
    if p.startswith(_LONG_PATH_PREFIX):
        return p[len(_LONG_PATH_PREFIX):]
    return p


# ---------------------------------------------------------------------------
# Mode parsing — translate Python "rb+" / "ab" / "xb" into os.O_* flags
# ---------------------------------------------------------------------------


def _flags_for_mode(mode: str) -> int:
    """Return :func:`os.open` flags for *mode*."""
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
# LocalPath
# ===========================================================================


class LocalPath(Path):
    """:class:`Path` for the local filesystem."""

    scheme: ClassVar[str] = "file"

    @property
    def is_local(self) -> bool:
        return True

    # ==================================================================
    # Classification
    # ==================================================================

    @classmethod
    def handles(cls, obj: Any) -> bool:
        """Local paths claim every input that doesn't look remote."""
        if isinstance(obj, URL):
            return not obj.scheme
        if isinstance(obj, str):
            head, sep, _ = obj.partition(":")
            # "scheme:/..." with an alpha-only multi-char scheme is
            # for someone else. Plain Windows-style "C:/..." (single-
            # letter drive) is ours; URL parsing will canonicalize.
            if sep and head.isalpha() and len(head) > 1:
                return False
            return True
        if isinstance(obj, (PurePath, os.PathLike)):
            return True
        try:
            return not URL.from_(obj).scheme
        except (ValueError, TypeError):
            return False

    # ==================================================================
    # Abstract hooks — file metadata, listing, mutation
    # ==================================================================

    def full_path(self) -> str:
        return os.fspath(self.url)

    def _os_path(self) -> str:
        """:meth:`full_path` with the Windows long-path prefix applied.

        Use anywhere the path string is fed to :mod:`os` / :mod:`shutil` /
        :func:`open` / :class:`mmap.mmap`; keep :meth:`full_path` for
        display, error messages, and URL math. No-op on non-Windows.
        """
        return _to_long_path(self.full_path())

    def _stat(self) -> PathStats:
        path = self._os_path()
        try:
            st = os.stat(path)
        except FileNotFoundError:
            return PathStats(kind=PathKind.MISSING)
        except NotADirectoryError:
            return PathStats(kind=PathKind.MISSING)

        mode = st.st_mode
        if stat_module.S_ISDIR(mode):
            kind = PathKind.DIRECTORY
        elif stat_module.S_ISREG(mode):
            kind = PathKind.FILE
        else:
            kind = PathKind.FILE

        return PathStats(
            kind=kind,
            size=int(st.st_size),
            mtime=float(st.st_mtime),
            mode=int(mode),
        )

    def is_symlink(self) -> bool:
        try:
            return os.path.islink(self._os_path())
        except OSError:
            return False

    def _ls(
        self,
        recursive: bool = False,
        allow_not_found: bool = True,
    ) -> Iterator["Path"]:
        root_path = self._os_path()

        if not recursive:
            try:
                with os.scandir(root_path) as it:
                    for entry in it:
                        yield self._from_url(URL.from_(_strip_long_path(entry.path)))
            except (FileNotFoundError, NotADirectoryError):
                if not allow_not_found:
                    raise
            return

        stack: list[str] = [root_path]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        yield self._from_url(URL.from_(_strip_long_path(entry.path)))
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(entry.path)
                        except OSError:
                            continue
            except (FileNotFoundError, NotADirectoryError):
                if not allow_not_found:
                    raise

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        path = self._os_path()
        try:
            if parents:
                os.makedirs(path, exist_ok=exist_ok)
            else:
                os.mkdir(path)
        except FileExistsError:
            if not exist_ok:
                raise

    def _remove_file(self, allow_not_found: bool = True) -> None:
        try:
            os.remove(self._os_path())
        except FileNotFoundError:
            if not allow_not_found:
                raise
        except IsADirectoryError as exc:
            raise IsADirectoryError(
                f"{self.full_path()!r} is a directory, not a file. "
                "Use rmdir() or remove()."
            ) from exc

    def _remove_dir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> None:
        path = self._os_path()
        try:
            if recursive:
                if with_root:
                    shutil.rmtree(path)
                else:
                    with os.scandir(path) as it:
                        for entry in it:
                            if entry.is_dir(follow_symlinks=False):
                                shutil.rmtree(entry.path)
                            else:
                                os.remove(entry.path)
            else:
                if not with_root:
                    raise ValueError(
                        "_remove_dir(recursive=False, with_root=False) is "
                        "a no-op shape — pass recursive=True to empty a "
                        "directory in place."
                    )
                os.rmdir(path)
        except FileNotFoundError:
            if not allow_not_found:
                raise

    def _open(
        self,
        mode: str = "rb",
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        auto_open: bool = True,
        touch: bool = False,
    ) -> BytesIO:
        if touch and "r" in mode and "+" not in mode and not self.exists():
            self.touch(exist_ok=True, parents=True)

        return BytesIO(
            self,
            mode=mode,
            auto_open=auto_open,
        )

    # ==================================================================
    # Abstract hooks — random-access I/O
    # ==================================================================
    #
    # The base :class:`Path` declares :meth:`pread` and :meth:`pwrite`
    # as ``@abstractmethod``. LocalPath implements them with single
    # syscalls per call; remote backends would either use range-
    # requests or call :meth:`_pread_via_io` / :meth:`_pwrite_via_rmw`
    # from their overrides.

    def pread(
        self,
        n: int,
        pos: int,
        *,
        default: Any = ...,
    ) -> bytes:
        """Positional read via :func:`os.pread` (POSIX) or
        ``lseek+read`` fallback. Single syscall pair per call,
        opens a fresh fd, closes it before returning.

        ``n < 0`` reads from *pos* to EOF. Stat is done first to
        size the read; if stat fails and *default* is supplied
        (anything other than the ``...`` sentinel), returns
        *default* instead of raising.
        """
        if pos < 0:
            raise ValueError("pread position must be >= 0")

        if n < 0:
            try:
                size = self.size
            except OSError:
                if default is ...:
                    raise
                return default
            n = max(0, size - pos)

        if n == 0:
            return b""

        try:
            fd = os.open(self._os_path(), os.O_RDONLY)
        except OSError:
            if default is ...:
                raise
            return default

        try:
            if _HAS_PREAD:
                # Loop until satisfied or EOF.
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

            # Portable fallback for platforms without pread (Windows
            # before recent Python builds). Cursor mutation is fine
            # — we own the fd and close it before returning.
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
            os.close(fd)

    def pwrite(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
        *,
        parents: bool = True,
    ) -> int:
        """Positional write via :func:`os.pwrite` (POSIX) or
        ``lseek+write`` fallback. Single syscall pair per call.

        Honors ``parents``: when True, missing parent directories
        are created via :func:`os.makedirs`. When False, a missing
        parent surfaces as the underlying OSError. The previous
        version always created parents regardless of the flag —
        a real bug for callers explicitly requesting strict mode.
        """
        mv = memoryview(data)
        if not mv.c_contiguous:
            mv = memoryview(bytes(mv))
        n = len(mv)
        if n == 0:
            return 0
        if pos < 0:
            raise ValueError("pwrite position must be >= 0")

        # O_CREAT lets a fresh path grow from a positional write
        # without a separate touch step. No O_TRUNC — we patch in
        # place.
        flags = os.O_WRONLY | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC

        # Honor the parents flag. Fixed vs. previous version which
        # always created parents regardless of the kwarg.
        os_path = self._os_path()
        if parents:
            parent = os.path.dirname(os_path)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)

        fd = os.open(os_path, flags, 0o644)
        try:
            if _HAS_PWRITE:
                total = 0
                while total < n:
                    written = os.pwrite(fd, bytes(mv[total:]), pos + total)
                    if written == 0:
                        raise io.BlockingIOError(
                            f"Short write at offset {pos + total}"
                        )
                    total += written
                return total

            os.lseek(fd, pos, os.SEEK_SET)
            total = 0
            while total < n:
                written = os.write(fd, bytes(mv[total:]))
                if written == 0:
                    raise io.BlockingIOError(
                        f"Short write at offset {pos + total}"
                    )
                total += written
            return total
        finally:
            os.close(fd)

    # ==================================================================
    # Filesystem mutators — local-cheap overrides
    # ==================================================================

    def truncate(self, n: int, *, parents: bool = True) -> int:
        """Local fast path — single ``os.truncate`` syscall.

        POSIX ``truncate(2)`` already implements the contract:

        - shrink drops the tail
        - extend zero-pads
        - missing path → ``OSError(ENOENT)`` → ``FileNotFoundError``
        - directory target → ``IsADirectoryError`` (Linux EISDIR)

        ``parents`` is accepted for signature parity but unused —
        :func:`os.truncate` won't create parent directories, and
        silently auto-creating them here would diverge from POSIX.
        """
        del parents

        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")

        os.truncate(self._os_path(), n)
        return n

    def touch(
        self,
        mode: int = 0o666,
        exist_ok: bool = True,
        parents: bool = True,
    ) -> None:
        """Create empty / bump mtime — :func:`os.utime` semantics.

        Avoids the base default's ``write_bytes(b"")`` which would
        truncate an existing file.
        """
        path = self._os_path()
        if parents:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        flags = os.O_WRONLY | os.O_CREAT
        if not exist_ok:
            flags |= os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC

        try:
            fd = os.open(path, flags, mode)
        except FileExistsError:
            if not exist_ok:
                raise
            return

        try:
            os.utime(fd, None)
        finally:
            os.close(fd)

    def rename(self, target: Any) -> "Path":
        """Atomic same-FS rename via :func:`os.replace`, falling
        back to base copy+remove for cross-backend / cross-device."""
        target_path = Path.from_(target)
        if not isinstance(target_path, LocalPath):
            return super().rename(target_path)

        try:
            os.replace(self._os_path(), target_path._os_path())
            return target_path
        except OSError:
            return super().rename(target_path)

    def resolve(self, *, strict: bool = False) -> "Path":
        try:
            resolved = os.path.realpath(self._os_path(), strict=strict)
        except (OSError, ValueError):
            if strict:
                raise
            resolved = os.path.realpath(self._os_path())
        return self._from_url(URL.from_(_strip_long_path(resolved)))

    def absolute(self) -> "Path":
        path = self.full_path()
        if os.path.isabs(path):
            return self
        return self._from_url(URL.from_(os.path.abspath(path)))

    # ==================================================================
    # mmap-backed memoryview / open_mmap
    # ==================================================================

    def memoryview(
        self,
        *,
        offset: int = 0,
        size: Optional[int] = None,
        raise_error: bool = True,
    ) -> memoryview:
        """Real mmap-backed view, not a bytes copy.

        Zero-byte files would trip :class:`mmap.mmap` (it raises
        ``ValueError`` on zero-length mappings); we short-circuit
        before opening the fd.
        """
        if offset < 0:
            raise ValueError("memoryview offset must be >= 0")

        try:
            total = self.size
        except OSError:
            if raise_error:
                raise
            return memoryview(b"")

        if total == 0:
            return memoryview(b"")

        if size is None:
            length = max(0, total - offset)
        else:
            length = max(0, min(int(size), total - offset))

        if length == 0:
            return memoryview(b"")

        try:
            fd = os.open(self._os_path(), os.O_RDONLY)
        except OSError:
            if raise_error:
                raise
            return memoryview(b"")

        try:
            mm = mmap.mmap(fd, length, access=mmap.ACCESS_READ, offset=offset)
        finally:
            # Safe to close the fd: mmap holds its own reference to
            # the mapping. The kernel keeps the mapping live until
            # the mmap object itself is closed/GC'd.
            os.close(fd)
        return memoryview(mm)

    def open_mmap(self, mode: str = "r"):
        """Real :class:`mmap.mmap` over this file, or ``None`` if empty."""
        size = self.size
        if size == 0:
            return None

        flag = os.O_RDONLY if mode == "r" else os.O_RDWR
        access = mmap.ACCESS_READ if mode == "r" else mmap.ACCESS_WRITE

        fd = os.open(self._os_path(), flag)
        try:
            return mmap.mmap(fd, size, access=access)
        finally:
            os.close(fd)

    # ==================================================================
    # Streaming copy — let shutil dispatch sendfile / copy_file_range
    # ==================================================================

    def copy_to(
        self,
        dest: Any,
        *,
        batch_size: int = _COPY_CHUNK,
        parents: bool = True,
    ) -> int:
        # Same-path guard inherited from the base via comparison.
        dest_path = Path.from_(dest)
        if dest_path == self:
            return self.size

        if not isinstance(dest_path, LocalPath):
            # Cross-backend copy goes through the base streaming loop.
            return super().copy_to(dest_path, batch_size=batch_size, parents=parents)

        if parents:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        # shutil.copyfile uses sendfile / copy_file_range on Linux
        # 3.8+ and falls back to a read/write loop elsewhere.
        shutil.copyfile(self._os_path(), dest_path._os_path())
        try:
            return self.size
        except OSError:
            return 0


# Make sure the registry has us — covers both __init_subclass__
# auto-registration and any module-reload scenario.
register_path_class(LocalPath)