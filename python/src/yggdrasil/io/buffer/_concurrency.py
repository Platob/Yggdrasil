"""Cross-process concurrency helpers for the buffer layer.

Three concerns live here:

- :class:`FileLock` — an advisory, sidecar-file lock with proper
  reader/writer semantics. Shared (``LOCK_SH``) for read-only access
  so concurrent readers don't block each other; exclusive
  (``LOCK_EX``) for any mode that mutates. Backed by :mod:`fcntl` on
  POSIX and :mod:`msvcrt` on Windows; the OS releases the lock when
  the holding process dies, so callers can't deadlock by crashing.

- :func:`lock_path_for` — canonical sidecar naming. Mode-aware
  suffix (``-r.lock`` / ``-w.lock`` / ``-rw.lock``) so external
  tooling can quickly identify what kind of lock is held without
  reading the file's contents.

- :func:`cleanup_stale_spill_files` — a janitor for the temp files
  minted by :func:`yggdrasil.io.buffer.bytes_io._mint_spill_path`. The
  filename encodes a TTL (``tmp-<seed>-<start>-<end>.<ext>``); this
  walker decodes ``end``, compares to ``time.time()``, and unlinks
  anything that's expired. A directory-level lock keeps multiple
  cleaners from racing.

All three are best-effort. They tolerate partial failures (NFS
without flock support, Windows-style "file in use", torn writes)
because the path-bound writer's correctness is grounded in the OS
fd, not in the sidecar lock or the TTL filename.
"""

from __future__ import annotations

import errno
import os
import re
import sys
import tempfile
import time
from typing import Optional, Union


__all__ = [
    "FileLock",
    "lock_path_for",
    "lock_suffix_for",
    "cleanup_stale_spill_files",
    "maybe_cleanup_stale_spill_files",
]


_IS_WINDOWS = sys.platform.startswith("win")

try:
    if _IS_WINDOWS:
        import msvcrt  # type: ignore[import-not-found]
        _HAS_FCNTL = False
        _HAS_MSVCRT = True
    else:
        import fcntl  # type: ignore[import-not-found]
        _HAS_FCNTL = True
        _HAS_MSVCRT = False
except ImportError:  # pragma: no cover — exotic platform
    _HAS_FCNTL = False
    _HAS_MSVCRT = False


# Mirrors the format produced by ``_mint_spill_path``:
#   tmp-<16hex>-<start>-<end>.<ext>
_SPILL_FILENAME_RE = re.compile(
    r"^tmp-[0-9a-f]+-(?P<start>\d+)-(?P<end>\d+)\.[^/\\\s]+$"
)

# Sentinel filename for the directory-level cleanup lock.
_CLEANUP_LOCK_NAME = ".ygg-spill-cleanup.lock"

# In-process throttle so the cleanup probe can't dominate the hot
# path. Cross-process coordination is the directory lock; this is
# just to avoid re-listing tempdir on every spill in a busy worker.
_CLEANUP_INTERVAL_S = 600.0
_last_cleanup_at: float = 0.0


# ---------------------------------------------------------------------------
# FileLock
# ---------------------------------------------------------------------------


class FileLock:
    """Advisory cross-process lock keyed by a sidecar lock file.

    Usage::

        with FileLock(lock_path, timeout=30):
            # critical section — writes to the materialised path

    Shared vs exclusive:

    - ``shared=False`` (default) — :data:`LOCK_EX`. Only one writer
      at a time; blocks both other writers and shared readers on
      the same lock file.
    - ``shared=True`` — :data:`LOCK_SH`. Multiple readers coexist;
      blocked only by an exclusive writer on the same lock file.

    Note that two callers must share the *same* lock-file path for
    coordination. :func:`lock_path_for` produces mode-suffixed
    paths so that read-only access uses a different lock file than
    write access — this is intentional: it lets external tools
    distinguish kinds of locks at a glance, at the cost of not
    serialising readers against writers. If you need strict rwlock
    semantics across modes, point both callers at the same lock
    path and pass appropriate ``shared`` values.

    The OS releases the lock when the holding fd is closed
    (including on process death), so a crashed holder cannot block
    subsequent runs indefinitely. The on-disk file may persist as a
    visual orphan, but won't prevent future acquisition.

    On platforms without :mod:`fcntl` and :mod:`msvcrt` (very rare),
    :meth:`acquire` degrades to a no-op — better than failing the
    write outright; the caller still gets the durable backing.
    """

    __slots__ = ("_path", "_timeout", "_poll", "_shared", "_fd")

    def __init__(
        self,
        path: Union[str, "os.PathLike[str]"],
        *,
        shared: bool = False,
        timeout: Optional[float] = None,
        poll: float = 0.05,
    ) -> None:
        self._path: str = os.fspath(path)
        # ``None`` → wait forever; ``0`` → single attempt + raise on contention.
        self._timeout = timeout
        self._poll = max(0.001, float(poll))
        self._shared = bool(shared)
        self._fd: Optional[int] = None

    @property
    def shared(self) -> bool:
        return self._shared

    # -- lifecycle ----------------------------------------------------

    @property
    def held(self) -> bool:
        return self._fd is not None

    @property
    def path(self) -> str:
        return self._path

    def acquire(self) -> None:
        """Block (up to ``timeout``) until the lock is held.

        Raises :class:`TimeoutError` when ``timeout`` elapses without
        the lock being obtained. Re-entrant on the same instance:
        a second call while already held is a no-op.
        """
        if self._fd is not None:
            return
        if not (_HAS_FCNTL or _HAS_MSVCRT):
            # Locking unsupported on this platform; degrade silently.
            self._fd = -1
            return

        parent = os.path.dirname(self._path)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                # If we can't create the parent we likely can't write
                # the target either — surface the underlying error
                # later when the caller tries to write.
                pass

        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC

        deadline = (
            None if self._timeout is None
            else time.monotonic() + max(0.0, self._timeout)
        )

        while True:
            try:
                fd = os.open(self._path, flags, 0o644)
            except OSError as exc:
                # Parent dir gone (race with cleanup), permission
                # denied, etc. Retry briefly — the parent may have
                # just been recreated by another writer.
                if deadline is None or time.monotonic() < deadline:
                    time.sleep(self._poll)
                    continue
                raise TimeoutError(
                    f"Could not open lock file {self._path!r}: {exc}"
                ) from exc

            locked = self._try_lock(fd, shared=self._shared)
            if locked:
                self._fd = fd
                self._write_owner_info(fd)
                return

            try:
                os.close(fd)
            except OSError:
                pass

            # Probe staleness: if the on-disk owner PID is gone, drop
            # the orphaned lock file so a stat-only observer doesn't
            # see a "lock present" indicator forever. Best effort —
            # the actual flock release happened at process death.
            self._try_break_stale_lock()

            if self._timeout == 0:
                raise TimeoutError(
                    f"Lock {self._path!r} is held by another process"
                )
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Could not acquire lock on {self._path!r} within "
                    f"{self._timeout}s"
                )
            time.sleep(self._poll)

    def release(self) -> None:
        """Release the lock; idempotent."""
        fd = self._fd
        self._fd = None
        if fd is None or fd < 0:
            # Either never acquired or running on a platform without
            # OS locking (in which case there's no fd to close).
            return
        # Unlink before unlock — readers waiting on the path will
        # see "file gone" and re-create / re-lock cleanly. If the
        # unlink races with another acquirer who just opened the
        # same path, both fds still serialize through the kernel
        # lock; only the on-disk file may be doubly-created, which
        # is fine.
        try:
            os.unlink(self._path)
        except OSError:
            pass
        try:
            self._unlock(fd)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:  # pragma: no cover — defensive
        try:
            self.release()
        except Exception:
            pass

    # -- platform primitives ------------------------------------------

    @staticmethod
    def _try_lock(fd: int, *, shared: bool = False) -> bool:
        if _HAS_FCNTL:
            op = (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB
            try:
                fcntl.flock(fd, op)
                return True
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                    return False
                # NFS without flock support: degrade to "no lock,"
                # treat as acquired so the holder proceeds.
                if exc.errno in (errno.ENOLCK, errno.ENOSYS, errno.EOPNOTSUPP):
                    return True
                raise
        if _HAS_MSVCRT:
            # msvcrt.locking has no shared mode — exclusive only.
            # Shared callers degrade to exclusive (correct but pessimistic).
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        return True

    @staticmethod
    def _unlock(fd: int) -> None:
        if _HAS_FCNTL:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            return
        if _HAS_MSVCRT:
            try:
                # Rewind: msvcrt.locking is offset-based.
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass

    def _write_owner_info(self, fd: int) -> None:
        """Stamp PID + timestamp into the lock file for diagnostics."""
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(
                fd,
                f"{os.getpid()} {int(time.time())}\n".encode("ascii"),
            )
        except OSError:
            pass

    def _try_break_stale_lock(self) -> None:
        """If the on-disk owner PID is gone, unlink the lock file.

        The kernel-level lock is already gone (it was tied to the
        crashed process's fd); this just cleans up the visible
        sidecar so users / monitors don't see a perpetual
        ``.lock`` file.
        """
        try:
            with open(self._path, "rb") as fh:
                head = fh.read(64)
        except OSError:
            return
        try:
            pid_str = head.decode("ascii", errors="ignore").split()[0]
            pid = int(pid_str)
        except (ValueError, IndexError):
            return
        if pid <= 0 or pid == os.getpid():
            return
        try:
            os.kill(pid, 0)
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                try:
                    os.unlink(self._path)
                except OSError:
                    pass


def lock_suffix_for(*, read: bool, write: bool) -> str:
    """Return the mode-flag suffix used by :func:`lock_path_for`.

    ``read`` and ``write`` describe the *intended access*:

    - read-only access → ``-r``
    - write-only access → ``-w``
    - both → ``-rw``

    Callers with neither set are treated as ``write=True`` (the
    conservative default — locking implies mutation intent).
    """
    if not (read or write):
        write = True
    if read and write:
        return "-rw"
    if read:
        return "-r"
    return "-w"


def lock_path_for(
    target: Union[str, "os.PathLike[str]"],
    *,
    read: bool = False,
    write: bool = True,
) -> str:
    """Return the canonical sidecar lock-file path for *target*.

    Encoded as ``<dir>/.<basename>-{r|w|rw}.lock`` — hidden, scoped
    to the same directory so readers/writers without filesystem-level
    privileges still discover the lock; suffixed with the access
    intent so external tooling can identify what kind of lock is
    held without reading the file's contents.

    Read locks (``-r.lock``) are typically *skippable* by cleanup or
    monitoring tools — multiple of them coexist by design and don't
    indicate contention.
    """
    target = os.fspath(target)
    parent = os.path.dirname(target) or "."
    base = os.path.basename(target) or "_"
    suffix = lock_suffix_for(read=read, write=write)
    return os.path.join(parent, f".{base}{suffix}.lock")


# ---------------------------------------------------------------------------
# Stale spill-temp cleanup
# ---------------------------------------------------------------------------


def cleanup_stale_spill_files(
    directory: Optional[str] = None,
    *,
    now: Optional[float] = None,
    grace_seconds: float = 0.0,
) -> int:
    """Unlink expired spill temp files in *directory*.

    Files matching the
    ``tmp-<seed>-<start>-<end>.<ext>`` pattern (the convention
    produced by ``_mint_spill_path``) are inspected; those whose
    encoded ``end`` epoch second is below ``now - grace_seconds``
    are unlinked.

    A directory-level lock prevents two concurrent cleaners from
    fighting over the same scan. If the lock can't be taken
    immediately, the call returns ``0`` — the other cleaner will
    handle this round.

    Returns the count of files actually unlinked. Errors on
    individual files are swallowed (another process may have just
    deleted the file, or the file may be in use on Windows).
    """
    if directory is None:
        directory = tempfile.gettempdir()
    if now is None:
        now = time.time()

    lock_path = os.path.join(directory, _CLEANUP_LOCK_NAME)
    lock = FileLock(lock_path, timeout=0)
    try:
        try:
            lock.acquire()
        except TimeoutError:
            return 0
        except OSError:
            return 0
        return _scan_and_unlink(directory, now=now, grace_seconds=grace_seconds)
    finally:
        lock.release()


def _scan_and_unlink(
    directory: str,
    *,
    now: float,
    grace_seconds: float,
) -> int:
    try:
        entries = os.listdir(directory)
    except OSError:
        return 0

    threshold = now - grace_seconds
    removed = 0
    for name in entries:
        m = _SPILL_FILENAME_RE.match(name)
        if m is None:
            continue
        try:
            end_epoch = int(m.group("end"))
        except ValueError:
            continue
        if end_epoch > threshold:
            continue
        full = os.path.join(directory, name)
        try:
            os.unlink(full)
            removed += 1
        except OSError:
            # Another process beat us, or Windows holds the file
            # open — either way, no harm done.
            continue
    return removed


def maybe_cleanup_stale_spill_files(
    directory: Optional[str] = None,
    *,
    interval_s: Optional[float] = None,
) -> int:
    """Probe-and-cleanup, throttled to once per :data:`_CLEANUP_INTERVAL_S`.

    Safe to call from hot paths: the in-process throttle bounds
    the cost; the cross-process directory lock prevents duplicate
    work across workers.
    """
    global _last_cleanup_at
    period = _CLEANUP_INTERVAL_S if interval_s is None else float(interval_s)
    monotonic_now = time.monotonic()
    if monotonic_now - _last_cleanup_at < period:
        return 0
    _last_cleanup_at = monotonic_now
    try:
        return cleanup_stale_spill_files(directory)
    except Exception:
        return 0
