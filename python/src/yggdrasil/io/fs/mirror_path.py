""":class:`MirrorPath` — proxy a remote :class:`Path` through its local mirror.

Wraps any remote Path. Reads ensure the local mirror is fresh (via
:meth:`Path.local_mirror`) and dispatch through the local file —
hot loops pay one network round-trip per ``ttl`` window.  Writes
hit the local mirror immediately and fire a daemon
:class:`~yggdrasil.concurrent.threading.ThreadJob` to upload the
bytes upstream; :meth:`flush` / :meth:`close` drain in-flight
uploads.

Stat, listing, and directory mutations pass through to the remote
so the metadata view stays canonical (size/mtime always reflect the
upstream object — never the local mirror's "in-flight" state).
This is important: a downstream caller that ``stat()``s after a
:class:`MirrorPath` write sees the post-upload size only after
:meth:`flush`. That's the explicit contract.

Identity
--------

A :class:`MirrorPath` carries the wrapped remote's URL — equality,
``full_path()``, ``parts``, ``parent`` etc. all render as the
remote. ``is_local`` stays ``False`` (the logical identity is
remote even though reads bypass the network).

Async upload semantics
----------------------

Each write spawns one daemon :class:`ThreadJob`. The instance keeps
a list of pending handles under a lock; :meth:`flush` joins every
pending upload (optionally with a timeout). :meth:`close(wait=True)`
flushes before the underlying :class:`Path` teardown.

Failures during upload are stored on the :class:`MirrorPath` as
``last_upload_error`` so a caller can surface them without polling
each :class:`ThreadJob`. The mirror is intentionally NOT rolled back
— the local copy stays put, and a follow-up ``write_bytes`` will
re-queue the upload.
"""

from __future__ import annotations

import logging
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterator,
    List,
    Optional,
    Union,
)

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.fs.path import Path
from yggdrasil.io.path_stat import PathStats
from yggdrasil.io.url import URL

if TYPE_CHECKING:
    from yggdrasil.concurrent.threading import ThreadJob


__all__ = ["MirrorPath"]


LOGGER = logging.getLogger(__name__)


class MirrorPath(Path):
    """Path proxy that serves reads from a local mirror and async-flushes writes.

    Construction::

        from yggdrasil.io.fs import LocalPath
        from yggdrasil.io.fs.mirror_path import MirrorPath

        remote = SomeRemotePath(...)
        cached = MirrorPath(remote)             # default 60s freshness window
        cached.read_bytes()                     # one download, served forever after

        cached.write_bytes(b"...")              # writes locally + queues upload
        cached.flush()                          # block on the upload

        with MirrorPath(remote) as cached:
            cached.write_text("v2")             # close() flushes by default

    See module docstring for the full contract.
    """

    scheme: ClassVar[str] = "mirror"
    __slots__ = (
        "_remote",
        "_mirror_root",
        "_mirror_ttl",
        "_pending",
        "_pending_lock",
        "_last_upload_error",
    )

    # ==================================================================
    # Construction / dispatch
    # ==================================================================

    @classmethod
    def handles(cls, obj: Any) -> bool:
        """:class:`MirrorPath` never claims a URL by itself — it's
        always built explicitly around an existing remote Path."""
        return False

    def __new__(cls, obj: Any = None, *args: Any, **kwargs: Any) -> "Path":
        # Bypass Path.__new__'s registry dispatch — MirrorPath wraps
        # whatever it's handed, it doesn't dispatch by URL scheme.
        del args, kwargs
        if isinstance(obj, MirrorPath):
            return obj
        return object.__new__(cls)

    def __init__(
        self,
        remote: Any,
        *,
        root: "Optional[Path]" = None,
        ttl: float = 60.0,
        temporary: bool = False,
        auto_open: bool = True,
    ) -> None:
        if isinstance(remote, MirrorPath):
            # Idempotent wrap: copy state over.
            inner = remote._remote
            root = remote._mirror_root if root is None else root
            ttl = remote._mirror_ttl if ttl == 60.0 else ttl
            remote = inner

        if isinstance(remote, Path):
            wrapped: Path = remote
        else:
            wrapped = Path.from_(remote)

        # Stamp the URL onto self so URL-derived properties (parent,
        # name, parts) match the remote's view.
        Path.__init__(
            self,
            url=wrapped.url,
            temporary=temporary,
            auto_open=auto_open,
        )
        self._remote: Path = wrapped
        self._mirror_root = root
        self._mirror_ttl = float(ttl)
        self._pending: List["ThreadJob"] = []
        self._pending_lock = threading.RLock()
        self._last_upload_error: Optional[BaseException] = None

    # ==================================================================
    # Identity / wrapping
    # ==================================================================

    @property
    def remote(self) -> Path:
        """The wrapped remote :class:`Path`. Read-only."""
        return self._remote

    @property
    def mirror_local(self) -> Path:
        """The on-disk mirror file. Pure mapping — no I/O."""
        return self._remote.mirror_path(root=self._mirror_root)

    @property
    def mirror_ttl(self) -> float:
        return self._mirror_ttl

    @property
    def is_local(self) -> bool:
        # Logical identity is the remote — even though reads bypass
        # the network. Callers checking ``is_local`` for routing
        # decisions (e.g. zero-copy local→local) get the right answer.
        return self._remote.is_local

    def full_path(self) -> str:
        return self._remote.full_path()

    def __repr__(self) -> str:
        return f"MirrorPath({self._remote!r}, ttl={self._mirror_ttl})"

    def _from_url(self, url: URL) -> "Path":
        """Build a same-typed proxy around the remote's URL-derived sibling.

        Navigation (``parent``, ``joinpath``) returns a new
        :class:`MirrorPath` wrapping the corresponding remote path so
        derived nodes inherit the cache settings.
        """
        derived_remote = self._remote._from_url(url)
        return MirrorPath(
            derived_remote,
            root=self._mirror_root,
            ttl=self._mirror_ttl,
        )

    # ==================================================================
    # Stat / list / mkdir / remove — pass through to remote
    # ==================================================================

    def _stat(self) -> PathStats:
        return self._remote._stat()

    def _ls(
        self,
        recursive: bool = False,
        allow_not_found: bool = True,
    ) -> Iterator["Path"]:
        for child in self._remote._ls(
            recursive=recursive, allow_not_found=allow_not_found,
        ):
            yield MirrorPath(
                child, root=self._mirror_root, ttl=self._mirror_ttl,
            )

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        self._remote._mkdir(parents=parents, exist_ok=exist_ok)

    def _remove_file(self, allow_not_found: bool = True) -> None:
        self._remote._remove_file(allow_not_found=allow_not_found)
        self._drop_local_mirror()

    def _remove_dir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> None:
        self._remote._remove_dir(
            recursive=recursive,
            allow_not_found=allow_not_found,
            with_root=with_root,
        )

    def _drop_local_mirror(self) -> None:
        """Best-effort removal of the on-disk mirror + sidecar."""
        try:
            local = self.mirror_local
            if local.exists():
                local._remove_file(allow_not_found=True)
            sidecar = local.parent / f".{local.name}.ygmirror.json"
            if sidecar.exists():
                sidecar._remove_file(allow_not_found=True)
        except Exception:
            LOGGER.debug(
                "Failed dropping local mirror for %s",
                self.full_path(), exc_info=True,
            )
        self._remote.invalidate_mirror()

    # ==================================================================
    # Reads — through the local mirror
    # ==================================================================

    def _ensure_mirror(self, *, force_refresh: bool = False) -> Path:
        """Refresh and return the local mirror file."""
        return self._remote.local_mirror(
            ttl=self._mirror_ttl,
            root=self._mirror_root,
            force_refresh=force_refresh,
        )

    def _pread(self):
        """Whole-file read through the local mirror.

        Refreshes the mirror first so callers see the upstream state
        as of the TTL window. Returns the mirror's BytesIO so reads
        ride the local-fd fast path.
        """
        return self._ensure_mirror()._pread()

    def _pwrite(self, data) -> int:
        """Whole-file write to the local mirror, queuing an async upload.

        Drops the verdict/sidecar so the next read after a remote
        round-trip notices the local change.
        """
        local = self.mirror_local
        local.parent.mkdir(parents=True, exist_ok=True)
        n = local._pwrite(data)
        payload = local.read_bytes()
        self._remote.invalidate_mirror()
        self._enqueue_upload(payload=payload)
        return n

    def read_bytes(self, *, raise_error: bool = True) -> bytes:
        try:
            return self._ensure_mirror().read_bytes(raise_error=raise_error)
        except FileNotFoundError:
            if raise_error:
                raise
            return b""

    def read_text(
        self,
        encoding: str = "utf-8",
        errors: str = "strict",
        raise_error: bool = True,
    ) -> str:
        return self.read_bytes(raise_error=raise_error).decode(
            encoding, errors=errors,
        )

    def pread(
        self,
        n: int,
        pos: int,
        *,
        default: Any = ...,
    ) -> bytes:
        if pos < 0:
            raise ValueError("pread position must be >= 0")
        if n == 0:
            return b""
        try:
            return self._ensure_mirror().pread(n=n, pos=pos, default=default)
        except FileNotFoundError:
            if default is ...:
                raise
            return default

    # ==================================================================
    # Writes — local first, async upstream
    # ==================================================================

    def write_bytes(
        self,
        data: Union[bytes, bytearray, memoryview],
        *,
        mode: str = "wb",
        parents: bool = True,
    ) -> int:
        local = self.mirror_local
        local.parent.mkdir(parents=True, exist_ok=True)
        n = local.write_bytes(data, mode=mode, parents=parents)
        # Drop the verdict/sidecar so the next read after a remote
        # round-trip notices the local change. The upload itself
        # restores upstream consistency.
        self._remote.invalidate_mirror()
        self._enqueue_upload(payload=bytes(data))
        return n

    def write_text(
        self,
        data: str,
        encoding: str = "utf-8",
        errors: str = "strict",
        newline: Optional[str] = None,
        parents: bool = True,
    ) -> int:
        del newline
        encoded = data.encode(encoding, errors=errors)
        return self.write_bytes(encoded, parents=parents)

    def pwrite(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
        *,
        parents: bool = True,
    ) -> int:
        # Ensure the local mirror reflects the remote before patching.
        try:
            self._ensure_mirror()
        except FileNotFoundError:
            # Fresh path — local will be created by pwrite below.
            pass

        local = self.mirror_local
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            local.touch(parents=parents, exist_ok=True)

        n = local.pwrite(data, pos, parents=parents)
        # Upload the WHOLE local file post-patch — most remote
        # backends don't support positional writes, so a write_bytes
        # of the full local content is the honest answer.
        payload = local.read_bytes()
        self._remote.invalidate_mirror()
        self._enqueue_upload(payload=payload)
        return n

    def truncate(self, n: int, *, parents: bool = True) -> int:
        local = self.mirror_local
        if not local.exists():
            raise FileNotFoundError(
                f"Cannot truncate non-mirrored path {self.full_path()!r}; "
                "write or read it first."
            )
        result = local.truncate(n, parents=parents)
        payload = local.read_bytes()
        self._remote.invalidate_mirror()
        self._enqueue_upload(payload=payload)
        return result

    def write_stream(
        self,
        src: Any,
        *,
        batch_size: int = 4 * 1024 * 1024,
        parents: bool = True,
    ) -> int:
        # Drain into the local mirror first, then queue one upload.
        local = self.mirror_local
        local.parent.mkdir(parents=True, exist_ok=True)
        total = local.write_stream(src, batch_size=batch_size, parents=parents)
        payload = local.read_bytes()
        self._remote.invalidate_mirror()
        self._enqueue_upload(payload=payload)
        return total

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
        """Open through the standard transaction-buffer machinery.

        The returned :class:`BytesIO` is bound to ``self``, so its
        acquire fills via :meth:`pread` (which routes through the
        local mirror) and its flush commits via :meth:`write_bytes`
        (which writes locally and queues the upload).
        """
        del encoding, errors, newline
        if touch and "r" in mode and "+" not in mode and not self.exists():
            raise FileNotFoundError(self.full_path())
        return BytesIO(path=self, mode=mode, auto_open=auto_open)

    # ==================================================================
    # Async upload tracking
    # ==================================================================

    def _enqueue_upload(self, *, payload: bytes) -> "ThreadJob":
        """Spawn a daemon ThreadJob to upload *payload* to the remote."""
        # Lazy import — concurrent.threading pulls in dependencies we
        # don't want to wire on module load.
        from yggdrasil.concurrent.job import Job

        def _upload(payload: bytes) -> int:
            try:
                return self._remote.write_bytes(payload, parents=True)
            except BaseException as exc:  # noqa: BLE001
                self._last_upload_error = exc
                LOGGER.warning(
                    "MirrorPath upload failed for %s: %r",
                    self.full_path(), exc,
                )
                raise

        handle = Job.make(_upload, payload).fire_and_forget()
        with self._pending_lock:
            self._pending.append(handle)
        return handle

    @property
    def pending_uploads(self) -> int:
        """Number of in-flight uploads (handles not yet completed).

        Note: a finished handle stays in the pending list until the
        next :meth:`flush` so its result/exception can still be
        observed; this property counts only handles that haven't yet
        finished, which is the operationally interesting number.
        """
        with self._pending_lock:
            return sum(1 for h in self._pending if not h.is_done)

    @property
    def last_upload_error(self) -> Optional[BaseException]:
        """The most recent upload exception, if any. Cleared by :meth:`flush`."""
        return self._last_upload_error

    def flush(
        self,
        *,
        wait: Any = None,
        raise_error: bool = False,
    ) -> int:
        """Wait for every pending upload to finish.

        ``wait``: ``None`` blocks indefinitely (the default), a number
        is a per-handle timeout in seconds, ``False`` is a non-blocking
        poll. Returns the number of handles that successfully completed.
        Drops finished handles from the pending list.
        """
        with self._pending_lock:
            handles = list(self._pending)

        completed = 0
        last_error: Optional[BaseException] = None
        for handle in handles:
            try:
                handle.wait(wait=wait, raise_error=False)
                if handle.is_done:
                    res = handle.result()
                    if res is not None and res.exception is not None:
                        last_error = res.exception
                    else:
                        completed += 1
            except BaseException as exc:  # noqa: BLE001
                last_error = exc

        with self._pending_lock:
            self._pending = [h for h in self._pending if not h.is_done]

        if last_error is not None:
            self._last_upload_error = last_error
            if raise_error:
                raise last_error
        else:
            # Clean slate after a fully successful drain.
            self._last_upload_error = None
        return completed

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def _release(self) -> None:
        # Drain pending uploads before the base teardown so the
        # remote object reflects every queued write by the time
        # ``close()`` returns.
        try:
            self.flush(wait=None, raise_error=False)
        except Exception:
            LOGGER.debug(
                "MirrorPath flush during release failed for %s",
                self.full_path(), exc_info=True,
            )
        super()._release()
