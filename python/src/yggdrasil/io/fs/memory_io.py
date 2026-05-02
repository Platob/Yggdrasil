"""In-memory filesystem — :class:`MemoryPath` + :class:`MemoryIO`.

Architecture
------------

Three pieces:

- :data:`REGISTRY` — ``ExpiringDict[URL, BytesIO]``, the in-memory
  filesystem. One :class:`BytesIO` per URL — that's the "file."
  1-hour default TTL. Subclassed to skip eviction of buffers that
  still have live :class:`MemoryIO` handles.
- :class:`MemoryIO` — per-handle wrapper around a registry
  :class:`BytesIO`. Carries its own cursor and mode, delegates byte
  storage to the wrapped buffer. Two opens on the same URL get two
  independent :class:`MemoryIO` instances over the same bytes —
  cursors don't collide, but writes from one are visible to the
  other (POSIX-shared-fd semantics).
- :class:`MemoryPath` — :class:`Path` for scheme ``memory``. Stat,
  ls, open, mkdir, remove all hit the registry.

Design choice: wrap, don't subclass
-----------------------------------

An earlier sketch made :class:`MemoryIO` a :class:`BytesIO`
subclass and reused the same instance across opens. That broke the
``mode`` accessor — two callers opening the same URL with different
modes clobber each other's mode flag. Wrapping a registry-owned
:class:`BytesIO` with a per-handle :class:`MemoryIO` gives each
caller its own ``mode`` / cursor / Disposable lifecycle without
duplicating bytes.

Eviction policy
---------------

:class:`MemoryRegistry` overrides both eviction paths in the base
:class:`ExpiringDict`:

- **Expiry sweep**: skip claimed buffers; bump their TTL by one
  default-window so they don't get re-considered every 15 min.
- **Capacity**: walk candidates in expiry order, pick the first
  *unclaimed* one. If everything is claimed, capacity eviction
  silently no-ops — exceeding ``max_size`` momentarily is safer
  than killing live state.

A buffer's claim count comes from the Disposable graph: every live
:class:`MemoryIO` ``add_owned`` s its registry buffer at
construction, so ``buffer._claimers`` reflects live handle count.
"""

from __future__ import annotations

from typing import IO, Any, ClassVar, Iterator, List, Optional

from yggdrasil.dataclasses.expiring import ExpiringDict, now_utc_ns
from yggdrasil.disposable import Disposable
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.url import URL
from .path import Path, register_path_class
from yggdrasil.io.path_stat import PathKind, PathStats

__all__ = ["MemoryIO", "MemoryPath", "MemoryRegistry", "REGISTRY"]


# Default TTL for unreferenced buffers — 1 hour. While a buffer
# has live :class:`MemoryIO` handles its claim count is non-zero
# and the registry skips it for eviction; the TTL clock effectively
# starts ticking when the last handle closes.
_DEFAULT_TTL_SECONDS = 60 * 60


# ===========================================================================
# MemoryRegistry
# ===========================================================================


class MemoryRegistry(ExpiringDict):
    """Claim-aware :class:`ExpiringDict` for the in-memory filesystem.

    Behaviourally an :class:`ExpiringDict[URL, BytesIO]` with two
    eviction overrides: expiry sweeps skip claimed buffers (and
    refresh their TTL), capacity eviction picks the soonest-expiring
    *unclaimed* buffer rather than the soonest-expiring one
    overall.

    Keeps the rest of the :class:`ExpiringDict` machinery — TTL
    parsing, ``on_evict`` notification outside the lock,
    :meth:`set_many`, :meth:`snapshot`, pickling — intact.
    """

    @staticmethod
    def _is_claimed(buffer: BytesIO) -> bool:
        """Does *buffer* have any active claimers in the Disposable graph?

        Defensive ``getattr`` — ``_claimers`` is a Disposable
        implementation detail, not a public API. If a buffer skipped
        Disposable's slot init (mocked, alternate Disposable layout),
        treat it as unclaimed rather than locking the entry into the
        registry forever.
        """
        return getattr(buffer, "_claimers", 0) > 0

    # ------------------------------------------------------------------
    # Override eviction paths
    # ------------------------------------------------------------------

    def _evict_expired_locked(self) -> List:
        """Evict expired-and-unclaimed entries; refresh expired-but-claimed
        ones forward by one default TTL window.

        The base :meth:`ExpiringDict._evict_expired_locked` drops every
        expired key. We split:

        - ``unclaimed and expired`` → really evict, return for ``on_evict``.
        - ``claimed and expired`` → push expiry forward so the next
          purge sweep doesn't re-process them. The TTL clock effectively
          starts fresh when the last handle drops the claim.
        - ``unexpired`` → ignored, same as base.
        """
        now = now_utc_ns()
        evicted = []
        refreshed: List = []
        for k, (v, exp) in list(self._store.items()):
            if exp is None or now < exp:
                continue
            if self._is_claimed(v):
                refreshed.append(k)
                continue
            self._store.pop(k)
            evicted.append((k, v))

        if refreshed and self._default_ttl_ns is not None:
            new_exp = now + self._default_ttl_ns
            for k in refreshed:
                v, _ = self._store[k]
                self._store[k] = (v, new_exp)

        return evicted

    def _evict_one_for_capacity_locked(self):
        """Pick the soonest-expiring unclaimed entry to evict.

        If every entry is claimed, return ``None`` — capacity
        eviction is a soft constraint here. The cache exceeds
        ``max_size`` momentarily; correctness > capacity. The next
        ``set`` retries.
        """
        if not self._store:
            return None

        # Sort by expiry ascending; treat None (non-expiring) as +inf.
        candidates = sorted(
            self._store.items(),
            key=lambda kv: (
                kv[1][1] if kv[1][1] is not None else float("inf")
            ),
        )
        for key, (value, _exp) in candidates:
            if self._is_claimed(value):
                continue
            self._store.pop(key)
            return (key, value)

        return None


# ===========================================================================
# Module-level singleton
# ===========================================================================


def _on_registry_evict(url: URL, buf: BytesIO) -> None:
    """Close evicted buffers so their resources (spill file, fd) drop.

    By the time the registry decides to evict, claim count is
    confirmed zero by :class:`MemoryRegistry`'s eviction overrides,
    so closing here doesn't pull the rug out from under any live
    handle. Exceptions are absorbed by the
    :meth:`ExpiringDict._notify_evicted` contract; we don't try to
    compensate.
    """
    del url
    try:
        if not buf.closed:
            buf.close()
    except Exception:
        pass


REGISTRY: MemoryRegistry = MemoryRegistry(
    default_ttl=float(_DEFAULT_TTL_SECONDS),
    on_evict=_on_registry_evict,
)


# ===========================================================================
# MemoryIO
# ===========================================================================


class MemoryIO(Disposable, IO[bytes]):
    """Per-handle wrapper over a registry :class:`BytesIO`.

    What :class:`MemoryIO` adds over the underlying :class:`BytesIO`:

    - Per-handle :attr:`mode` and cursor (the wrapped buffer's
      cursor is ignored; we maintain ``_pos`` ourselves).
    - :attr:`path` accessor pointing at the :class:`MemoryPath`.
    - Disposable-graph ownership of the buffer (``add_owned``)
      so the registry's eviction logic sees a non-zero claim
      count while this handle is live.
    - Mode-driven semantics — ``wb`` truncates the buffer on
      ``open()``, ``ab`` positions at end, ``xb`` would have
      already raised in :meth:`MemoryPath._open`.

    What it deliberately does NOT do:

    - Duplicate bytes. Two :class:`MemoryIO` handles over the same
      URL share the registry buffer; writes from one are visible
      to the other (POSIX-shared-fd semantics). Cursors are
      independent.
    - Implement spill. The wrapped :class:`BytesIO` may itself
      transition to spill mode; that's the buffer's concern.
    """

    __slots__ = ("_path", "_mode", "_buffer", "_pos")

    def __init__(
        self,
        path: "MemoryPath",
        buffer: BytesIO,
        mode: str = "rb",
        *,
        auto_open: bool = True,
    ) -> None:
        Disposable.__init__(self)
        self._path = path
        self._mode = mode
        self._buffer = buffer
        # Per-handle cursor. Initial position depends on mode:
        # ``ab`` positions at EOF; everything else at 0. We don't
        # call ``buffer.seek`` — the buffer's internal cursor is
        # shared with any other handles, and we maintain our own.
        self._pos = buffer.size if "a" in mode else 0

        if auto_open:
            self.open()

    # ------------------------------------------------------------------
    # PathIO surface
    # ------------------------------------------------------------------

    @property
    def path(self) -> "MemoryPath":
        return self._path

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_writing(self) -> bool:
        return any(c in self._mode for c in "wa+x")

    @property
    def is_local(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self._path.full_path()

    # ------------------------------------------------------------------
    # Disposable hooks
    # ------------------------------------------------------------------

    def _acquire(self) -> None:
        # Apply mode-driven content reset on the underlying buffer.
        # Truncate-on-open is shared across all handles by design:
        # POSIX ``open(O_TRUNC)`` on a path drops bytes even with
        # other fds on the same inode. We match that.
        if "w" in self._mode:
            self._buffer.truncate(0)
            self._pos = 0

    # ------------------------------------------------------------------
    # IO[bytes] protocol — delegate to buffer with our own cursor
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self.closed or self._buffer.closed:
            raise ValueError("I/O operation on closed MemoryIO")

    def read(self, n: int = -1) -> bytes:
        self._check_open()
        if n is None or n < 0:
            n = max(0, self._buffer.size - self._pos)
        data = self._buffer.pread(n, self._pos)
        self._pos += len(data)
        return data

    def readline(self, limit: int = -1) -> bytes:
        self._check_open()
        # Chunked scan from current cursor — bounded by ``limit``
        # or buffer size if unbounded, so reads are O(line length)
        # rather than O(file size).
        buf_size = self._buffer.size
        if self._pos >= buf_size:
            return b""
        if limit is None or limit < 0:
            chunk_len = buf_size - self._pos
        else:
            chunk_len = min(limit, buf_size - self._pos)
        if chunk_len <= 0:
            return b""

        chunk = self._buffer.pread(chunk_len, self._pos)
        nl = chunk.find(b"\n")
        if nl == -1:
            self._pos += len(chunk)
            return chunk
        line = chunk[: nl + 1]
        self._pos += len(line)
        return line

    def readlines(self, hint: int = -1) -> List[bytes]:
        out: List[bytes] = []
        total = 0
        while True:
            line = self.readline()
            if not line:
                break
            out.append(line)
            total += len(line)
            if hint is not None and hint > 0 and total >= hint:
                break
        return out

    def write(self, data: Any) -> int:
        self._check_open()
        if not self.is_writing:
            raise ValueError(
                f"MemoryIO opened in mode {self._mode!r} is not writable"
            )
        # Append mode: writes always land at end-of-buffer regardless
        # of where ``_pos`` sits, mirroring O_APPEND. Update ``_pos``
        # to the new EOF after the write so a subsequent read sees
        # the appended bytes.
        if "a" in self._mode:
            target = self._buffer.size
        else:
            target = self._pos
        if isinstance(data, str):
            data = data.encode("utf-8")
        n = self._buffer.pwrite(data, target)
        self._pos = target + n
        if n:
            self.mark_dirty()
        return n

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def seek(self, offset: int, whence: int = 0) -> int:
        self._check_open()
        size = self._buffer.size
        if whence == 0:
            new_pos = size + offset if offset < 0 else offset
            if new_pos < 0:
                raise ValueError(
                    f"Negative SEEK_SET offset {offset!r} past start of "
                    f"{size}-byte buffer"
                )
        elif whence == 1:
            new_pos = max(0, self._pos + offset)
        elif whence == 2:
            new_pos = max(0, size + offset)
        else:
            raise ValueError(f"Invalid whence: {whence!r}")
        self._pos = new_pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def truncate(self, size: Optional[int] = None) -> int:
        self._check_open()
        if not self.is_writing:
            raise ValueError(
                f"MemoryIO opened in mode {self._mode!r} is not writable"
            )
        if size is None:
            size = self._pos
        result = self._buffer.truncate(int(size))
        if self._pos > result:
            self._pos = result
        self.mark_dirty()
        return result

    # ------------------------------------------------------------------
    # Cursorless IO — straight delegation; the buffer's pread/pwrite
    # are already cursor-independent.
    # ------------------------------------------------------------------

    def pread(self, n: int, pos: int) -> bytes:
        self._check_open()
        return self._buffer.pread(n, pos)

    def pwrite(self, data: Any, pos: int) -> int:
        self._check_open()
        if not self.is_writing:
            raise ValueError(
                f"MemoryIO opened in mode {self._mode!r} is not writable"
            )
        n = self._buffer.pwrite(data, pos)
        if n:
            self.mark_dirty()
        return n

    @property
    def size(self) -> int:
        return self._buffer.size

    def __len__(self) -> int:
        return self._buffer.size

    def to_bytes(self) -> bytes:
        return self._buffer.to_bytes()

    def memoryview(self):
        return self._buffer.memoryview()

    # ------------------------------------------------------------------
    # IO[bytes] capabilities
    # ------------------------------------------------------------------

    def readable(self) -> bool:
        return "r" in self._mode or "+" in self._mode

    def writable(self) -> bool:
        return self.is_writing

    def seekable(self) -> bool:
        return True

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        # The wrapped buffer might be spilled — defer to it.
        # Memory-mode buffers raise ``OSError``, matching the
        # ``IO[bytes]`` contract for "no underlying fd."
        return self._buffer.fileno()

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    # ------------------------------------------------------------------
    # Context manager — single-shot, file-like idiom
    # ------------------------------------------------------------------

    def __enter__(self) -> "MemoryIO":
        if not self._acquired:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self._dirty = False
        self.close(force=True)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        state = "open" if not self.closed else "closed"
        return (
            f"<MemoryIO [{state}] mode={self._mode!r} "
            f"size={self._buffer.size} pos={self._pos} "
            f"path={self._path!r}>"
        )


# ===========================================================================
# MemoryPath
# ===========================================================================


class MemoryPath(Path):
    """:class:`Path` over the in-memory :data:`REGISTRY`.

    The "filesystem" is the registry: a key exists iff there's a
    :class:`BytesIO` registered for the URL. Directories are
    virtual — implied by the URL prefix structure of the actual
    file keys, never registered as separate entries.
    """

    scheme: ClassVar[str] = "memory"

    __slots__ = ()

    @property
    def is_local(self) -> bool:
        # Same-process bytes — no network. ``is_local`` is consulted
        # by IO-routing code to decide whether mmap / sendfile are
        # in scope; for memory the answer is "moot but yes."
        return True

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    def full_path(self) -> str:
        # Scheme-qualified rendering. MemoryPath URLs always carry
        # the ``memory:`` prefix in their string form so they
        # survive round-tripping through string-typed APIs (config,
        # logs, error messages) without being re-dispatched as
        # local paths.
        return f"memory:{self.url.path}"

    def _stat(self) -> PathStats:
        """Stat against the registry.

        - URL present and live → ``FILE`` with current size.
        - URL is a prefix of any registered key → ``DIRECTORY``.
        - Otherwise → ``MISSING``.

        Stat does not refresh the entry's TTL — that's a get-side
        semantic in :class:`ExpiringDict`. Filesystem navigation
        and handle ownership are kept on separate axes; the
        Disposable claim count is the right axis for "keep alive."
        """
        buf = REGISTRY.get(self.url, None)
        if buf is not None and not buf.closed:
            return PathStats(kind=PathKind.FILE, size=int(buf.size))

        if self._has_children():
            return PathStats(kind=PathKind.DIRECTORY, size=0)

        return PathStats(kind=PathKind.MISSING)

    def _has_children(self) -> bool:
        """True when at least one registered key sits under this URL prefix."""
        prefix = self.url.path.rstrip("/") + "/"
        for key in REGISTRY.keys():
            if isinstance(key, URL) and key.path.startswith(prefix):
                return True
        return False

    def _ls(
        self,
        recursive: bool = False,
        allow_not_found: bool = True,
    ) -> Iterator["Path"]:
        """List children. Directories are derived from key prefixes.

        Non-recursive: yield distinct one-level descendants only,
        deduplicating directory entries. Recursive: yield every
        descendant key.
        """
        prefix = self.url.path.rstrip("/") + "/"
        prefix_len = 1 if prefix == "/" else len(prefix)

        seen: set[str] = set()
        any_yielded = False

        for key in REGISTRY.keys():
            if not isinstance(key, URL):
                continue
            kp = key.path
            if not kp.startswith(prefix):
                continue
            any_yielded = True

            if recursive:
                yield self._from_url(key)
                continue

            tail = kp[prefix_len:]
            if not tail:
                continue
            first_seg, _, rest = tail.partition("/")
            child_path = prefix + first_seg + ("/" if rest else "")
            if child_path in seen:
                continue
            seen.add(child_path)

            child_url = URL.from_(f"memory:{child_path}")
            yield self._from_url(child_url)

        if not any_yielded and not allow_not_found:
            raise FileNotFoundError(self.full_path())

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        # Memory directories are virtual — implied by file keys
        # under their prefix. Nothing to materialize. We honor
        # ``exist_ok=False`` for parity with concrete backends:
        # mkdir over an existing file raises.
        del parents
        buf = REGISTRY.get(self.url, None)
        if buf is not None and not buf.closed and not exist_ok:
            raise FileExistsError(
                f"{self.full_path()!r} already exists as a file, "
                "cannot mkdir over it."
            )

    def _remove_file(self, allow_not_found: bool = True) -> None:
        # ``ExpiringDict.__delitem__`` fires ``on_evict``, which
        # closes the buffer. Note: this bypasses the eviction-time
        # claim-count check; explicit user delete is the one place
        # we let the caller break live handles, on the theory that
        # they asked for it. Live handles will start raising
        # ``ValueError`` (closed buffer) on their next op — same
        # as POSIX unlink-while-open, except POSIX preserves the
        # bytes on the inode and we don't.
        try:
            del REGISTRY[self.url]
        except KeyError:
            if not allow_not_found:
                raise FileNotFoundError(self.full_path())

    def _remove_dir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> None:
        prefix = self.url.path.rstrip("/") + "/"
        children = [
            k for k in REGISTRY.keys()
            if isinstance(k, URL) and k.path.startswith(prefix)
        ]

        if not children:
            if not allow_not_found:
                raise FileNotFoundError(self.full_path())
            return

        if not recursive:
            raise OSError(
                f"Directory {self.full_path()!r} is not empty "
                f"({len(children)} entries) and recursive=False"
            )

        REGISTRY.delete_many(children)
        del with_root  # No directory entry to remove for the root itself.

    def _open(
        self,
        mode: str = "rb",
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        auto_open: bool = True,
        touch: bool = False,
    ) -> MemoryIO:
        """Get-or-create the registry buffer, return a fresh :class:`MemoryIO`.

        Mode handling lives here, not in :class:`MemoryIO`, because
        the registry get-or-create flow needs to decide whether to
        fail-on-missing (``rb``, ``rb+``) or create-fresh (``wb``,
        ``ab``, ``xb``).

        Each call constructs a *new* :class:`MemoryIO` wrapping the
        registry's buffer. Cursors and modes are per-handle; bytes
        are shared.
        """
        del encoding, errors, newline  # MemoryIO is binary-only

        existing = REGISTRY.get(self.url, None)
        # A closed buffer that hasn't been swept out of the registry
        # yet should be treated as missing. Shouldn't happen often
        # (the on_evict path closes only after removing from the
        # store) but defensive.
        if existing is not None and existing.closed:
            existing = None

        if existing is None:
            if "r" in mode and "+" not in mode and not touch:
                raise FileNotFoundError(self.full_path())
            # Mint a fresh buffer and register. ``touch=True`` on
            # an ``rb`` open creates an empty entry, matching the
            # contract ``LocalPath._open`` honors via ``self.touch()``.
            buf = BytesIO()
            REGISTRY[self.url] = buf
        else:
            if "x" in mode:
                raise FileExistsError(self.full_path())
            buf = existing

        return MemoryIO(self, buf, mode=mode, auto_open=auto_open)


# Defensive registration — `__init_subclass__` already does this,
# but explicit registration covers module-reload edge cases.
register_path_class(MemoryPath)