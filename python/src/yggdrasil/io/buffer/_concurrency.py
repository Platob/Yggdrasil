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
import socket
import sys
import tempfile
import time
from typing import Optional, Tuple, Union

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg


__all__ = [
    "AbstractLock",
    "AtomicLock",
    "FileLock",
    "OWNER_URL_ENV",
    "compute_identifier_url",
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


# Mirrors the time-sortable format produced by ``_mint_spill_path``:
#   tmp-<start>-<end>-<16hex>.<ext>
# ``start`` and ``end`` are zero-padded epoch seconds; the seed is
# 16 lowercase hex chars. Lexical sort matches chronological order.
_SPILL_FILENAME_RE = re.compile(
    r"^tmp-(?P<start>\d+)-(?P<end>\d+)-[0-9a-f]+\.[^/\\\s]+$"
)

# Sentinel filename for the directory-level cleanup lock.
_CLEANUP_LOCK_NAME = ".ygg-spill-cleanup.lock"

# In-process throttle so the cleanup probe can't dominate the hot
# path. Cross-process coordination is the directory lock; this is
# just to avoid re-listing tempdir on every spill in a busy worker.
_CLEANUP_INTERVAL_S = 600.0
_last_cleanup_at: float = 0.0

# Stale-lock probes are expensive — local for FileLock (one open +
# read) and remote for AtomicLock (a pread, possibly a stat). On a
# healthy contended lock the holder is alive and the probe is pure
# overhead. We always run it on the FIRST contention iteration so
# wait=0 callers can break a dead-holder sidecar immediately, then
# back off: every Nth iteration AND at most once per probe-window.
# With the default ~50 ms poll cadence, every-8th-iteration plus the
# 5 s window keeps wait=N callers responsive while cutting up to ~90%
# of the probe round-trips on a busy lock.
_STALE_PROBE_EVERY_N: int = 8
_STALE_PROBE_MIN_INTERVAL_S: float = 5.0


def _safe_hostname() -> str:
    """Return ``socket.gethostname()`` with whitespace sanitised.

    The sidecar payload is whitespace-delimited, so a hostname
    containing tabs/newlines/spaces would corrupt the parser's field
    boundaries. Such hostnames are vanishingly rare on real systems
    but cheap to defend against. Falls back to ``"unknown"`` on any
    OS-level failure (e.g. tightly sandboxed containers without
    a hostname configured).
    """
    try:
        host = socket.gethostname() or "unknown"
    except OSError:
        return "unknown"
    # Replace any whitespace with ``_`` so the field stays
    # split-on-whitespace-safe on the consumer side.
    return re.sub(r"\s+", "_", host)


# Cached at module load — hostname is process-stable AND fork-safe.
# ``socket.gethostname`` is cheap, but each lock-acquire shouldn't
# pay the syscall. PID, by contrast, MUST be re-read per call so a
# forked child gets its own identifier.
_HOSTNAME: str = _safe_hostname()


#: Env var that overrides :func:`compute_identifier_url`. When set, the
#: function returns the value verbatim — used to propagate a driver-
#: derived URL to Spark executors / subprocesses so every worker
#: writing on behalf of the same job records the *same* owner. The
#: name is exposed publicly so callers can plumb it into Spark conf
#: via ``spark.executorEnv.YGG_OWNER_URL`` or pass-through env.
OWNER_URL_ENV: str = "YGG_OWNER_URL"


def compute_identifier_url() -> str:
    """Return a URL-shaped identifier for the *current* compute owner.

    Used as the third field of the lock's owner-info payload so a
    sidecar on a remote backend (S3, GCS, ABFS, network FS, …) can
    name its holder unambiguously across machines and pipelines —
    PID alone collides cross-host, hostname alone hides the job /
    run / task running on it.

    Resolution order:

    1. ``$YGG_OWNER_URL`` if set (verbatim) — the propagation hook
       used by the Spark connector and other coordinators that need
       worker processes to share an id with their driver.
    2. **Databricks** (``DATABRICKS_RUNTIME_VERSION`` or
       ``DB_CLUSTER_ID`` set) →
       ``databricks://<cluster_id>/<pid>?host=<hostname>&job=<id>&run=<id>&task=<key>&notebook=<path>``
       with each query field omitted when the underlying env var
       is not set.
    3. **Fallback** → ``host://<hostname>/<pid>``.

    Pipeline detection is best-effort and env-var-only — we don't
    reach into Spark / dbutils, that would couple the lock primitive
    to heavyweight optional deps.

    Always called fresh — the PID changes after ``os.fork()``, so a
    cached URL would mis-attribute child processes.
    """
    from urllib.parse import quote, urlencode

    env = os.environ.get

    override = env(OWNER_URL_ENV)
    if override:
        # Sanitise: strip whitespace so the field stays
        # split-on-whitespace-safe in the lock payload.
        sanitised = re.sub(r"\s+", "_", override.strip())
        if sanitised:
            return sanitised

    pid = os.getpid()

    cluster = env("DB_CLUSTER_ID") or env("DATABRICKS_CLUSTER_ID")
    if cluster or env("DATABRICKS_RUNTIME_VERSION"):
        # ``host`` is always carried so the same-host stale-break
        # check still works for FileLocks held by a Databricks driver
        # against a local (DBFS-mounted) sidecar.
        params: list[tuple[str, str]] = [
            ("host", _HOSTNAME),
            ("pid", str(pid)),
        ]
        # Job / Run / Task / Notebook tags — Databricks exposes a
        # rotating cast of these depending on runtime version and
        # workflow surface. Take the union and let the sidecar carry
        # whichever ones happen to be set.
        for env_key, qs_key in (
            ("DATABRICKS_JOB_ID", "job"),
            ("DATABRICKS_JOB_RUN_ID", "run"),
            ("DATABRICKS_RUN_ID", "run"),
            ("DATABRICKS_TASK_KEY", "task"),
            ("DATABRICKS_TASK_RUN_ID", "task_run"),
            ("DB_NOTEBOOK_PATH", "notebook"),
            ("DB_NOTEBOOK_ID", "notebook_id"),
        ):
            value = env(env_key)
            if value:
                params.append((qs_key, value))
        encoded = urlencode(params)
        cluster_seg = quote(cluster or "cluster", safe="")
        return f"databricks://{cluster_seg}/{pid}?{encoded}"

    return f"host://{quote(_HOSTNAME, safe='')}/{pid}"


def _host_from_owner_url(url: Optional[str]) -> Optional[str]:
    """Extract the originating hostname from a compute-identifier URL.

    For ``host://<host>/<pid>`` the netloc IS the hostname. For
    ``databricks://<cluster>/<pid>?host=<host>&…`` the hostname rides
    in the query string (the netloc is the cluster id). Anything
    unknown or unparseable returns ``None`` so the caller skips the
    same-host probe rather than guessing.
    """
    if not url:
        return None
    from urllib.parse import parse_qsl, unquote, urlparse

    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if parsed.scheme == "host":
        netloc = parsed.netloc or parsed.path.lstrip("/")
        return unquote(netloc) or None
    if parsed.query:
        for key, value in parse_qsl(parsed.query):
            if key == "host" and value:
                return value
    return None


def _build_owner_payload() -> bytes:
    """Owner-info payload for both lock variants.

    Format: ``{pid} {epoch} {compute_url}\\n`` (ASCII). Whitespace-
    delimited so the existing positional parser still works — older
    payloads (PID-only or PID+epoch) parse on the same offsets, with
    the missing tail fields surfaced as ``None``.

    The trailing field is :func:`compute_identifier_url` — see its
    docstring for the schemes (``databricks://`` / ``host://``).
    """
    return f"{os.getpid()} {int(time.time())} {compute_identifier_url()}\n".encode("ascii")


def _parse_owner_payload(head: bytes) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    """Decode an owner-info payload into ``(pid, epoch, compute_url)``.

    Any field that's missing or unparseable comes back as ``None`` —
    callers degrade gracefully (skip the same-host check, fall back
    to mtime, etc.). Backward-compatible with the legacy PID-only
    and PID+epoch payloads from before compute URLs were written.
    """
    if not head:
        return None, None, None
    try:
        text = head.decode("ascii", errors="ignore")
    except Exception:
        return None, None, None
    parts = text.split()
    pid: Optional[int] = None
    epoch: Optional[float] = None
    compute_url: Optional[str] = None
    if parts:
        try:
            pid = int(parts[0])
        except ValueError:
            pid = None
    if len(parts) >= 2:
        try:
            epoch = float(parts[1])
        except ValueError:
            epoch = None
    if len(parts) >= 3:
        compute_url = parts[2]
    return pid, epoch, compute_url


# ---------------------------------------------------------------------------
# FileLock
# ---------------------------------------------------------------------------


class AbstractLock:
    """Common surface every lock variant exposes.

    Concrete subclasses ship the actual mechanics (kernel flock for
    local fs, atomic exclusive-create for remote backends). Callers
    only care about the four-method API: ``acquire`` / ``release`` /
    ``held`` / context-manager support. :meth:`Path.lock` returns
    whichever subclass fits the backend.
    """

    @property
    def held(self) -> bool:  # pragma: no cover — abstract
        raise NotImplementedError

    def acquire(self) -> None:  # pragma: no cover — abstract
        raise NotImplementedError

    def release(self) -> None:  # pragma: no cover — abstract
        raise NotImplementedError

    def __enter__(self) -> "AbstractLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:  # pragma: no cover — defensive
        try:
            self.release()
        except Exception:
            pass

    # -- shared throttling for stale-lock probes ---------------------

    def _should_probe_stale(self, iteration: int) -> bool:
        """Decide whether this iteration should run a stale-lock probe.

        Always probes on iteration 0 (so ``wait=0`` callers can break
        a dead sidecar immediately). Otherwise gates on every-Nth
        iteration *and* a minimum wall-clock interval; the wall-clock
        floor protects against fast polling configs that would otherwise
        pass the count gate too often.

        Subclasses must own a ``_last_stale_probe_at: float`` slot
        initialised to ``0.0`` and reset to ``0.0`` at the top of each
        :meth:`acquire` call so a fresh contention loop always gets
        the iteration-0 probe.
        """
        if iteration == 0:
            self._last_stale_probe_at = time.monotonic()
            return True
        if iteration % _STALE_PROBE_EVERY_N != 0:
            return False
        now = time.monotonic()
        if now - self._last_stale_probe_at < _STALE_PROBE_MIN_INTERVAL_S:
            return False
        self._last_stale_probe_at = now
        return True


class FileLock(AbstractLock):
    """Advisory cross-process lock keyed by a sidecar lock file.

    Usage::

        with FileLock(lock_path, wait=30):
            # critical section — writes to the materialised path

    Wait semantics are driven by a :class:`WaitingConfig`:

    - ``wait=None`` — wait forever, polling at a fixed cadence.
    - ``wait=N`` — wait up to ``N`` seconds (coerced via
      :meth:`WaitingConfig.from_`). Backoff between attempts.
    - ``wait=0`` — single attempt, raise :class:`TimeoutError` on
      contention.
    - ``wait=WaitingConfig(...)`` — full control: ``timeout``,
      ``interval``, ``backoff``, ``max_interval``, ``retries``.

    On contention the loop calls :meth:`WaitingConfig.sleep`,
    which applies exponential backoff capped at ``max_interval`` and
    raises :class:`TimeoutError` once the deadline is reached.

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

    __slots__ = ("_path", "_wait", "_shared", "_fd", "_last_stale_probe_at")

    # Default poll cadence when ``wait=None`` (wait forever).
    _DEFAULT_POLL_S: float = 0.05

    def __init__(
        self,
        path: Union[str, "os.PathLike[str]"],
        *,
        shared: bool = False,
        wait: "WaitingConfigArg | None" = None,
    ) -> None:
        self._path: str = os.fspath(path)
        self._wait: "WaitingConfig | None" = (
            WaitingConfig.from_(wait) if wait is not None else None
        )
        self._shared = bool(shared)
        self._fd: Optional[int] = None
        # Monotonic timestamp of the most recent staleness probe.
        # Reset to 0 each acquire so a fresh contention loop always
        # gets to run a first-iteration probe; thereafter it gates
        # by ``_STALE_PROBE_EVERY_N`` and ``_STALE_PROBE_MIN_INTERVAL_S``.
        self._last_stale_probe_at: float = 0.0

    @property
    def shared(self) -> bool:
        return self._shared

    @property
    def wait(self) -> "WaitingConfig | None":
        return self._wait

    # -- lifecycle ----------------------------------------------------

    @property
    def held(self) -> bool:
        return self._fd is not None

    @property
    def path(self) -> str:
        return self._path

    def acquire(self) -> None:
        """Block (up to the wait deadline) until the lock is held.

        Raises :class:`TimeoutError` when the configured
        :class:`WaitingConfig` deadline elapses without the lock
        being obtained. Re-entrant on the same instance: a second
        call while already held is a no-op.
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

        wait = self._wait
        # ``start`` activates the timeout-enforcement branch in
        # ``WaitingConfig.sleep``. Skipped when wait is None or has
        # ``timeout=0`` (= no timeout, poll forever) — except that
        # ``wait=WaitingConfig(timeout=0)`` is also the "single
        # attempt, raise on contention" shape, handled below.
        start = time.time() if wait is not None and wait.timeout > 0 else None
        iteration = 0
        # Reset so the first contention iteration always probes —
        # critical for ``wait=0`` callers facing a stale sidecar.
        self._last_stale_probe_at = 0.0

        while True:
            try:
                fd = os.open(self._path, flags, 0o644)
            except OSError as exc:
                # Parent dir gone (race with cleanup), permission
                # denied, etc. If we're under a wait budget, retry
                # briefly; otherwise surface the error.
                if wait is None or wait.timeout > 0:
                    self._sleep_or_raise(wait, iteration, start)
                    iteration += 1
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
            # Throttled so a busy lock doesn't pay for the open+read
            # on every poll iteration.
            if self._should_probe_stale(iteration):
                self._try_break_stale_lock()

            # ``wait=WaitingConfig(timeout=0)`` — single-attempt mode.
            if wait is not None and wait.timeout == 0:
                raise TimeoutError(
                    f"Lock {self._path!r} is held by another process"
                )

            # Retry with backoff, honouring the wait budget.
            self._sleep_or_raise(wait, iteration, start)
            iteration += 1

    @classmethod
    def _sleep_or_raise(
        cls,
        wait: "WaitingConfig | None",
        iteration: int,
        start: "float | None",
    ) -> None:
        """Sleep one polling iteration; raise when the deadline is hit."""
        if wait is None:
            time.sleep(cls._DEFAULT_POLL_S)
            return
        wait.sleep(iteration, start=start)

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
        """Stamp PID + timestamp + hostname into the lock file for diagnostics.

        Hostname disambiguates owners across machines for sidecars
        on shared / network filesystems where PID alone collides.
        """
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, _build_owner_payload())
        except OSError:
            pass

    def _try_break_stale_lock(self) -> None:
        """If the on-disk owner PID is gone, unlink the lock file.

        The kernel-level lock is already gone (it was tied to the
        crashed process's fd); this just cleans up the visible
        sidecar so users / monitors don't see a perpetual
        ``.lock`` file.

        Cross-host safety: ``os.kill(pid, 0)`` only proves liveness
        for *this host's* process table. If the recorded hostname
        doesn't match ours, we can't conclude the holder is dead —
        leave the sidecar alone and let mtime-based recovery (or
        the holder's own release) handle it.
        """
        try:
            with open(self._path, "rb") as fh:
                # 512 B is more than enough for a Databricks compute
                # URL even with all optional job/run/task/notebook
                # query params populated.
                head = fh.read(512)
        except OSError:
            return
        pid, _epoch, compute_url = _parse_owner_payload(head)
        if pid is None or pid <= 0 or pid == os.getpid():
            return
        # Cross-host safety: extract the originating hostname from
        # the compute identifier URL and skip the local liveness
        # probe when the holder lives on another machine. Legacy
        # payloads without a URL (compute_url is None) keep the old
        # same-host behaviour — those were local-fs-only by
        # construction since AtomicLock didn't write owner info
        # before either.
        host = _host_from_owner_url(compute_url)
        if host is not None and host != _HOSTNAME:
            return
        try:
            os.kill(pid, 0)
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                try:
                    os.unlink(self._path)
                except OSError:
                    pass


class AtomicLock(AbstractLock):
    """Cross-filesystem advisory lock via atomic exclusive-create.

    Works on *any* :class:`yggdrasil.io.fs.Path` backend that
    surfaces ``xb`` mode through :meth:`Path.open_io` (i.e. raises
    :class:`FileExistsError` when the file already exists). The
    sidecar's existence IS the lock: the first writer to create it
    wins; everyone else polls until it disappears.

    Use :class:`FileLock` for local-mount paths — it's faster (one
    fd, kernel-level enforcement, OS-released on death). Reach for
    :class:`AtomicLock` when the backend is remote (S3, GCS, ABFS,
    Databricks volumes, in-memory filesystem) and ``fcntl`` doesn't
    apply. :meth:`Path.lock` already picks the right one.

    Caveats vs :class:`FileLock`:

    - **No shared lock.** Atomic exclusive-create is binary; the
      ``shared=`` flag is accepted for API parity but ignored — the
      lock is always exclusive. Readers and writers using different
      sidecar names (``.r.lock`` / ``.w.lock``) won't interlock.
    - **Crash recovery is timestamp-based.** On a remote backend
      the kernel can't release the lock; if the holder crashed,
      the sidecar lingers until ``stale_after_seconds`` elapses, at
      which point any acquirer force-unlinks it.
    - **Eventual consistency.** On weakly-consistent backends an
      atomic-create can succeed for two callers within the
      consistency window. The lock degrades to "best effort" in
      that case — same as flock-on-NFS-without-lockd.
    """

    # Default age past which a lingering sidecar is presumed stale
    # and force-unlinked. Tuned high enough that an honest holder
    # heartbeating every minute is safe; tighten/loosen via the
    # ``stale_after_seconds`` constructor kwarg.
    DEFAULT_STALE_AFTER_S: float = 15 * 60.0

    __slots__ = (
        "_path", "_wait", "_shared", "_held",
        "_stale_after_s", "_owner_payload",
        "_last_stale_probe_at",
    )

    def __init__(
        self,
        path: Any,
        *,
        shared: bool = False,
        wait: "WaitingConfigArg | None" = None,
        stale_after_seconds: "float | None" = None,
    ) -> None:
        self._path = path  # yggdrasil Path
        self._wait: "WaitingConfig | None" = (
            WaitingConfig.from_(wait) if wait is not None else None
        )
        self._shared = bool(shared)  # accepted for parity; not enforced
        self._held: bool = False
        self._stale_after_s: float = (
            self.DEFAULT_STALE_AFTER_S
            if stale_after_seconds is None
            else float(stale_after_seconds)
        )
        self._owner_payload: bytes = b""
        # See ``FileLock._last_stale_probe_at`` — same throttling
        # rationale, even more important here because the probe
        # is one-or-two remote round-trips.
        self._last_stale_probe_at: float = 0.0

    @property
    def held(self) -> bool:
        return self._held

    @property
    def shared(self) -> bool:
        return self._shared

    @property
    def path(self) -> Any:
        return self._path

    @property
    def wait(self) -> "WaitingConfig | None":
        return self._wait

    def acquire(self) -> None:
        """Block (up to the wait deadline) until the sidecar can be created."""
        if self._held:
            return

        wait = self._wait
        start = time.time() if wait is not None and wait.timeout > 0 else None
        iteration = 0
        # Reset so the first iteration always probes — same rationale
        # as :meth:`FileLock.acquire`. The remote round-trip cost of
        # the probe makes the throttle materially more impactful here.
        self._last_stale_probe_at = 0.0
        # Hostname is part of the payload — for a remote sidecar
        # (S3 / GCS / ABFS / network FS) the holder is plausibly
        # on a different machine, and downstream tooling needs a
        # way to attribute the lock without guessing.
        payload = _build_owner_payload()

        while True:
            try:
                with self._path.open_io("xb") as io:
                    io.write_bytes(payload)
                self._held = True
                self._owner_payload = payload
                return
            except FileExistsError:
                # Stale-break runs inline so a crashed-holder sidecar
                # gets unlinked AND retried in the same iteration —
                # ``wait=0`` callers shouldn't have to know about
                # ghost sidecars to make progress against them.
                # Throttled to keep contended-but-healthy locks from
                # paying remote round-trips on every poll.
                if self._should_probe_stale(iteration) and self._maybe_break_stale():
                    continue
            except OSError as exc:
                # Backend transient — treat like contention so the
                # wait config can decide whether to keep retrying.
                if wait is not None and wait.timeout == 0:
                    raise TimeoutError(
                        f"Could not create lock {self._path!r}: {exc}"
                    ) from exc

            if wait is not None and wait.timeout == 0:
                raise TimeoutError(
                    f"Lock {self._path!r} is held by another process"
                )

            FileLock._sleep_or_raise(wait, iteration, start)
            iteration += 1

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            self._path.unlink(missing_ok=True)
        except Exception:
            # Backend rejected the unlink; the lock will eventually
            # age out via the staleness check on the next acquirer.
            pass
        self._owner_payload = b""

    def _maybe_break_stale(self) -> bool:
        """Force-unlink the sidecar if it's older than the staleness window.

        Returns ``True`` when a stale sidecar was actually unlinked,
        so the caller knows to retry the create. ``False`` when
        nothing needed doing.

        Remote backends don't release the lock on process death, so
        a crashed writer would otherwise wedge the lock forever.
        Two-stage check: prefer the *embedded* timestamp (written
        by the holder at acquire time, so it survives copies and
        mtime resets), fall back to ``stat.mtime`` if the embedded
        bytes are unparseable.
        """
        if self._stale_after_s <= 0:
            return False
        now = time.time()
        recorded_age: "float | None" = None

        # Embedded timestamp (preferred — survives backend mtime quirks).
        # Read 128 bytes — enough for ``{pid} {epoch} {hostname}\n``
        # even with a long FQDN, while still cheap on every backend.
        try:
            head = self._path.pread(128, 0)
        except Exception:
            head = None
        if head:
            _pid, epoch, _host = _parse_owner_payload(head)
            if epoch is not None:
                recorded_age = now - epoch

        # Mtime fallback.
        if recorded_age is None:
            try:
                stat = self._path.stat()
            except Exception:
                return False
            mtime = (
                getattr(stat, "mtime", None)
                or getattr(stat, "modified", None)
            )
            if isinstance(mtime, (int, float)) and mtime > 0:
                recorded_age = now - float(mtime)

        if recorded_age is None or recorded_age <= self._stale_after_s:
            return False

        try:
            self._path.unlink(missing_ok=True)
            return True
        except Exception:
            return False


def lock_suffix_for(*, read: bool, write: bool) -> str:
    """Return the mode-flag token used by :func:`lock_path_for`.

    ``read`` and ``write`` describe the *intended access*:

    - read-only access → ``r``
    - write-only access → ``w``
    - both → ``rw``

    Callers with neither set are treated as ``write=True`` (the
    conservative default — locking implies mutation intent).
    """
    if not (read or write):
        write = True
    if read and write:
        return "rw"
    if read:
        return "r"
    return "w"


def lock_path_for(
    target: Union[str, "os.PathLike[str]"],
    *,
    read: bool = False,
    write: bool = True,
) -> str:
    """Return the canonical sidecar lock-file path for *target*.

    Encoded as ``<dir>/.<basename>.{r|w|rw}.lock`` — hidden, scoped
    to the same directory so readers/writers without filesystem-level
    privileges still discover the lock; the mode-flag segment lets
    external tooling identify what kind of lock is held without
    reading the file's contents.

    Read locks (``.r.lock``) are typically *skippable* by cleanup or
    monitoring tools — multiple of them coexist by design and don't
    indicate contention.
    """
    target = os.fspath(target)
    parent = os.path.dirname(target) or "."
    base = os.path.basename(target) or "_"
    suffix = lock_suffix_for(read=read, write=write)
    return os.path.join(parent, f".{base}.{suffix}.lock")


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
    lock = FileLock(lock_path, wait=0)
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
