""":class:`WorkspacePath` — ``/Workspace/...`` via the Workspace API.

Workspace objects (notebooks, files, directories) are managed
through ``workspace.workspace.*``: ``download``, ``upload``,
``list``, ``mkdirs``, ``delete``, ``get_status``.

The Workspace API has a single download / upload pair (no range
reads, no positional writes), so the byte primitives map onto a
read-modify-rewrite scheme.

Cluster-mount fast path
-----------------------

Inside a Databricks runtime (DBR 13+) the workspace tree is mounted
as a FUSE filesystem at ``/Workspace/...``. Read / stat / listdir
calls against workspace **files** (``.py``, ``.txt``, ``.whl``, …)
go through the kernel mount, which is dramatically faster than the
Workspace REST API. Notebooks aren't backed by simple files; reads
that need their notebook source still go through ``workspace.download``.
The probe lives in :func:`_workspace_mount_available` and is gated on
``DatabricksClient.is_in_databricks_environment()`` AND
``os.path.isdir("/Workspace")``.
"""

from __future__ import annotations

import logging
import os
import stat as _stat
import time
from typing import Any, ClassVar, Iterator

from yggdrasil.dataclasses import ExpiringDict, WaitingConfig
from yggdrasil.enums import Scheme
from yggdrasil.enums.media_type import MediaType
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.path.remote_path import _STAT_CACHE_TTL
from yggdrasil.url import URL

from ..path import DatabricksPath
from ..workspaces.service import Workspaces

__all__ = ["WorkspacePath"]

logger = logging.getLogger(__name__)


# Process-wide cache of resolved usernames, keyed by ``id`` of the
# bound workspace client. One ``current_user.me()`` round-trip per
# client; cleared implicitly when the client is garbage-collected
# (the next caller gets a fresh ``id`` from the allocator).
_USER_NAME_CACHE: dict[int, str] = {}


# ---------------------------------------------------------------------------
# Local /Workspace FUSE mount fast path
# ---------------------------------------------------------------------------

_LOCAL_WS_MOUNT_PROBED: bool = False
_LOCAL_WS_MOUNT_AVAILABLE: bool = False


def _workspace_mount_available() -> bool:
    """``True`` when ``/Workspace/...`` is reachable via the kernel mount.

    Conjunction of "this process runs inside a Databricks runtime" and
    "``/Workspace`` exists on disk". Cached process-wide after the
    first probe; the result is logged once at INFO so an operator can
    see whether WorkspacePath is hitting the FUSE mount or paying
    Workspace-API round trips.
    """
    global _LOCAL_WS_MOUNT_PROBED, _LOCAL_WS_MOUNT_AVAILABLE
    if _LOCAL_WS_MOUNT_PROBED:
        return _LOCAL_WS_MOUNT_AVAILABLE
    try:
        from yggdrasil.databricks.client import DatabricksClient
        in_runtime = DatabricksClient.is_in_databricks_environment()
    except Exception:
        in_runtime = False
    has_mount = bool(in_runtime) and os.path.isdir("/Workspace")
    _LOCAL_WS_MOUNT_AVAILABLE = has_mount
    _LOCAL_WS_MOUNT_PROBED = True
    if has_mount:
        logger.info(
            "WorkspacePath: /Workspace kernel mount detected — "
            "short-circuiting stat/read/ls/mkdir/remove off the Workspace API.",
        )
    else:
        logger.debug(
            "WorkspacePath: /Workspace kernel mount unavailable "
            "(in_runtime=%s, /Workspace exists=%s) — routing through Workspace API.",
            in_runtime, os.path.isdir("/Workspace"),
        )
    return _LOCAL_WS_MOUNT_AVAILABLE


def _reset_workspace_mount_probe() -> None:
    """Test hook — drop the cached probe result."""
    global _LOCAL_WS_MOUNT_PROBED, _LOCAL_WS_MOUNT_AVAILABLE
    _LOCAL_WS_MOUNT_PROBED = False
    _LOCAL_WS_MOUNT_AVAILABLE = False


class WorkspacePath(DatabricksPath):
    """Path under ``/Workspace/...`` via the Workspace API."""

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_WORKSPACE
    NAMESPACE_PREFIX: ClassVar[str] = "/Workspace/"
    _SERVICE_CLASS: ClassVar[type] = Workspaces

    # Per-class singleton cache — partitioned away from DBFSPath /
    # VolumePath. No companion lock —
    # :class:`ExpiringDict.get_or_set` is GIL-atomic.
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=_STAT_CACHE_TTL,
        max_size=10_000,
    )

    # ==================================================================
    # Path rendering
    # ==================================================================

    def full_path(self) -> str:
        p = (self.url.path or "").lstrip("/")
        rendered = "/Workspace/" + p if p else "/Workspace"
        return self._resolve_me(rendered)

    @property
    def api_path(self) -> str:
        return self.full_path()

    # ------------------------------------------------------------------
    # ``<me>`` placeholder resolution
    # ------------------------------------------------------------------

    def _resolve_me(self, path: str) -> str:
        """Substitute ``<me>`` segments with the current user's name.

        Lets callers write portable shapes like
        ``/Workspace/Users/<me>/scratch`` and have them resolve to
        ``/Workspace/Users/<actual.user@example.com>/scratch`` on the
        bound workspace client. The lookup is cached per workspace
        client (one ``current_user.me()`` round-trip per session),
        and the placeholder is matched as a *segment* — substrings
        that happen to contain ``<me>`` won't be touched.
        """
        if "<me>" not in path:
            return path
        parts = path.split("/")
        if "<me>" not in parts:
            return path
        username = self._current_user_name()
        if not username:
            return path
        return "/".join(username if seg == "<me>" else seg for seg in parts)

    def _current_user_name(self) -> str:
        """Return the bound workspace client's current user name (cached).

        Cache is process-wide, keyed by ``id(workspace)``, so every
        :class:`WorkspacePath` derived from the same client reuses a
        single ``current_user.me()`` round-trip. Returns an empty
        string when the lookup fails or the SDK gives back a non-str
        username (e.g. a :class:`MagicMock` in tests) —
        :meth:`_resolve_me` then leaves the placeholder untouched
        rather than masking the real error at every callsite.
        """
        ws = self.workspace_client
        key = id(ws)
        cached = _USER_NAME_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            info = ws.current_user.me()
        except Exception:
            return ""
        name = getattr(info, "user_name", None) or getattr(
            info,
            "userName",
            None,
        )
        if not isinstance(name, str) or not name:
            return ""
        _USER_NAME_CACHE[key] = name
        return name

    # ==================================================================
    # Stat
    # ==================================================================

    def _stat_uncached(self) -> IOStats:
        api_path = self.api_path
        if _workspace_mount_available():
            try:
                st = os.stat(api_path)
            except FileNotFoundError:
                logger.debug(
                    "stat via /Workspace mount: %r -> MISSING", api_path,
                )
                return IOStats(kind=IOKind.MISSING, size=0, mtime=0.0)
            except OSError as exc:
                logger.debug(
                    "stat via /Workspace mount: %r -> OSError %r, "
                    "falling back to Workspace API", api_path, exc,
                )
            else:
                if _stat.S_ISDIR(st.st_mode):
                    logger.debug(
                        "stat via /Workspace mount: %r -> DIRECTORY",
                        api_path,
                    )
                    return IOStats(
                        kind=IOKind.DIRECTORY,
                        size=0,
                        mtime=st.st_mtime,
                    )
                logger.debug(
                    "stat via /Workspace mount: %r -> FILE size=%d",
                    api_path, st.st_size,
                )
                return IOStats(
                    kind=IOKind.FILE,
                    size=int(st.st_size),
                    mtime=st.st_mtime,
                )
        try:
            info = self._call(
                self.client.workspace_client().workspace.get_status,
                api_path,
            )
        except Exception:
            return IOStats(kind=IOKind.MISSING, size=0, mtime=0.0)

        ot = getattr(info, "object_type", None)
        is_dir = getattr(ot, "name", None) == "DIRECTORY" or str(ot).upper().endswith(
            "DIRECTORY"
        )
        size = int(getattr(info, "size", 0) or 0)
        mtime_ms = getattr(info, "modified_at", None) or 0
        return IOStats(
            kind=IOKind.DIRECTORY if is_dir else IOKind.FILE,
            size=size,
            mtime=float(mtime_ms) / 1000.0 if mtime_ms else 0.0,
        )

    @property
    def size(self) -> int:
        return int(self._stat().size)

    # ==================================================================
    # Listing
    # ==================================================================

    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator["WorkspacePath"]:
        if _workspace_mount_available():
            scan_root = self.api_path
            logical_root = self.full_path().rstrip("/")
            try:
                scan = os.scandir(scan_root)
            except FileNotFoundError:
                logger.debug(
                    "ls via /Workspace mount: %r -> not found", scan_root,
                )
                return
            except (NotADirectoryError, PermissionError) as exc:
                logger.warning(
                    "Cannot scan workspace directory %r: %r", self, exc,
                )
                return
            yielded = 0
            with scan as it:
                for entry in it:
                    child_logical = f"{logical_root}/{entry.name}"
                    child = type(self)(
                        child_logical,
                        service=self.service,
                        singleton_ttl=singleton_ttl,
                    )
                    try:
                        st = entry.stat(follow_symlinks=False)
                        is_dir = _stat.S_ISDIR(st.st_mode)
                        child._persist_stat_cache(
                            IOStats(
                                kind=(
                                    IOKind.DIRECTORY
                                    if is_dir
                                    else IOKind.FILE
                                ),
                                size=0 if is_dir else int(st.st_size),
                                mtime=st.st_mtime,
                            )
                        )
                    except OSError:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    yielded += 1
                    yield child
                    if recursive and is_dir:
                        yield from child._ls(
                            recursive=True,
                            singleton_ttl=singleton_ttl,
                        )
            logger.debug(
                "ls via /Workspace mount: %r -> %d entries (recursive=%s)",
                scan_root, yielded, recursive,
            )
            return
        try:
            entries = list(
                self._call(self.client.workspace_client().workspace.list, self.api_path)
            )
        except Exception:
            return
        logger.debug(
            "Listing workspace directory %r -> %d entries (recursive=%s)",
            self,
            len(entries),
            recursive,
        )
        for info in entries:
            child_path = getattr(info, "path", None)
            if not child_path:
                continue
            # Workspace.list returns paths under ``/Workspace/...``;
            # the URL path is the namespace-stripped suffix.
            url_path = child_path
            if url_path.startswith("/Workspace/"):
                url_path = url_path[len("/Workspace") :]
            elif url_path.startswith("/Workspace"):
                url_path = url_path[len("/Workspace") :] or "/"
            # ``singleton_ttl`` defaults to ``False`` so listing
            # children stay out of ``DatabricksPath._INSTANCES``;
            # callers wanting cached children pass it through ``ls``.
            child = type(self)(
                url=URL(scheme=self.scheme, path=url_path),
                service=self.service,
                singleton_ttl=singleton_ttl,
            )
            ot = getattr(info, "object_type", None)
            is_dir = getattr(ot, "name", None) == "DIRECTORY" or str(
                ot
            ).upper().endswith("DIRECTORY")
            # Seed from the listing entry so the caller's ``is_file()``
            # / ``size`` / ``exists()`` per child don't each issue a
            # follow-up ``workspace.get_status`` round trip.
            mtime_ms = getattr(info, "modified_at", None) or 0
            child._persist_stat_cache(
                IOStats(
                    kind=IOKind.DIRECTORY if is_dir else IOKind.FILE,
                    size=0 if is_dir else int(getattr(info, "size", 0) or 0),
                    mtime=float(mtime_ms) / 1000.0 if mtime_ms else 0.0,
                )
            )
            yield child
            if recursive and is_dir:
                yield from child._ls(recursive=True, singleton_ttl=singleton_ttl)

    # ==================================================================
    # Mutators
    # ==================================================================

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        logger.debug(
            "Creating workspace directory %r (parents=%s, exist_ok=%s)",
            self, parents, exist_ok,
        )
        if _workspace_mount_available():
            api_path = self.api_path
            try:
                if parents:
                    os.makedirs(api_path, exist_ok=exist_ok)
                else:
                    os.mkdir(api_path)
            except FileExistsError:
                if not exist_ok:
                    raise
            self._persist_stat_cache(IOStats(kind=IOKind.DIRECTORY))
            logger.debug("mkdir via /Workspace mount: %r", api_path)
            return
        del parents
        try:
            self._call(self.client.workspace_client().workspace.mkdirs, self.api_path)
        except Exception as exc:
            if _looks_like_already_exists(exc):
                if not exist_ok:
                    raise
                self._persist_stat_cache(IOStats(kind=IOKind.DIRECTORY))
                return
            if _looks_like_protected_parent(exc):
                # Hitting a protected ancestor (e.g. ``/Workspace/Users``)
                # is fine if the leaf already landed — fall through and
                # let downstream ops succeed.
                self._persist_stat_cache(IOStats(kind=IOKind.DIRECTORY))
                return
            raise
        self._persist_stat_cache(IOStats(kind=IOKind.DIRECTORY))

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        del wait
        logger.debug(
            "Deleting workspace file %r (missing_ok=%s)", self, missing_ok,
        )
        if _workspace_mount_available():
            try:
                os.remove(self.api_path)
                logger.debug("rm via /Workspace mount: %r", self.api_path)
            except FileNotFoundError:
                if not missing_ok:
                    raise
            except IsADirectoryError:
                raise
            self.invalidate_singleton()
            return
        try:
            self._call(
                self.client.workspace_client().workspace.delete,
                self.api_path,
                recursive=False,
            )
        except Exception:
            if not missing_ok:
                raise
        self.invalidate_singleton()

    def _remove_dir(
        self,
        recursive: bool,
        missing_ok: bool,
        wait: WaitingConfig,
    ) -> None:
        del wait
        logger.debug(
            "Deleting workspace directory %r (recursive=%s, missing_ok=%s)",
            self, recursive, missing_ok,
        )
        if _workspace_mount_available():
            import shutil
            api_path = self.api_path
            try:
                if recursive:
                    shutil.rmtree(api_path)
                else:
                    os.rmdir(api_path)
            except FileNotFoundError:
                if not missing_ok:
                    raise
            self.invalidate_singleton()
            logger.debug(
                "rmdir via /Workspace mount: %r (recursive=%s)",
                api_path, recursive,
            )
            return
        try:
            self._call(
                self.client.workspace_client().workspace.delete,
                self.api_path,
                recursive=recursive,
            )
        except Exception:
            if not missing_ok:
                raise
        self.invalidate_singleton()

    # ==================================================================
    # Holder I/O — download / upload
    # ==================================================================

    def _read_mv(self, n: int, pos: int) -> memoryview:
        if n == 0:
            return memoryview(b"")
        # Cluster fast path — read off the FUSE mount when available.
        # Workspace files (.whl, .txt, .py, …) are exposed as regular
        # files on disk; notebooks are exposed as their source/IPython
        # representation, matching what ``workspace.download(format=AUTO)``
        # returns. Any OSError falls through to the Workspace API.
        if _workspace_mount_available():
            api_path = self.api_path
            try:
                with open(api_path, "rb") as fh:
                    if pos:
                        fh.seek(pos)
                    data = fh.read() if n < 0 else fh.read(n)
            except FileNotFoundError as exc:
                logger.debug(
                    "read via /Workspace mount: %r -> NOT FOUND", api_path,
                )
                raise FileNotFoundError(self.full_path()) from exc
            except OSError as exc:
                logger.debug(
                    "read via /Workspace mount: %r -> OSError %r, "
                    "falling back to Workspace API",
                    api_path, exc,
                )
            else:
                logger.debug(
                    "read via /Workspace mount: %r -> %d bytes "
                    "(pos=%d, n=%s)",
                    api_path, len(data), pos, "EOF" if n < 0 else n,
                )
                if not self._stat_cached:
                    try:
                        st = os.stat(api_path)
                        self._persist_stat_cache(
                            IOStats(
                                size=int(st.st_size),
                                kind=IOKind.FILE,
                                mtime=st.st_mtime,
                            )
                        )
                    except OSError:
                        pass
                return memoryview(data)
        try:
            # ``format`` defaults to ``ExportFormat.SOURCE`` in the SDK,
            # which routes through the notebook export path and returns
            # an empty body for binary workspace files (``.whl``,
            # ``.zip``, ``.parquet``, …). ``AUTO`` mirrors what
            # :meth:`_upload` passes on the import side so the same
            # extension/content sniff drives the export — notebooks
            # come back as source, workspace files come back as their
            # raw bytes.
            response = self._call(
                self.client.workspace_client().workspace.download,
                self.api_path,
                format=_export_format_auto(),
            )
        except Exception as exc:
            if _looks_like_not_found(exc):
                raise FileNotFoundError(self.full_path()) from exc
            raise
        body = getattr(response, "contents", None) or response
        try:
            data = body.read()
        except AttributeError:
            data = bytes(body)
        logger.debug(
            "Downloaded workspace file %r -> %d bytes (slice pos=%d n=%s)",
            self,
            len(data),
            pos,
            "EOF" if n < 0 else n,
        )

        # ``workspace.download`` always returns the whole object, so its
        # length IS the file size. Seed the stat cache (or refresh the
        # ``size`` slot when the prior entry came from a metadata probe)
        # so the next ``size`` / ``exists`` lookup is local.
        media_type = _media_type_from_response(response)
        if self._stat_cached is None:
            self._persist_stat_cache(
                IOStats(
                    size=len(data),
                    kind=IOKind.FILE,
                    media_type=media_type,
                )
            )
        else:
            self._stat_cached.size = len(data)
            if media_type is not None and self._stat_cached.media_type is None:
                self._stat_cached.media_type = media_type
            # Re-stamp the TTL — this download IS the freshest size we
            # could observe; the entry should outlive the original
            # probe's window from this point on.
            self._persist_stat_cache(self._stat_cached)

        if pos:
            data = data[pos:]
        if n > 0:
            data = data[:n]
        return memoryview(data)

    def _write_stream(
        self,
        src: Any,
        *,
        offset: int,
        size: int = -1,
        **kwargs: Any,
    ) -> int:
        """Override the base chunked stream — Workspace wants one PUT.

        The Workspace API has no positional / range write, so a
        chunked :meth:`Holder._write_stream` would issue one RMW
        per chunk. Hand the live :class:`IO[bytes]` straight to
        :meth:`_upload`, which already seek-rewinds on retry and
        the SDK builds the multipart body lazily — multi-GB
        sources never materialise as a Python ``bytes`` object.
        The atomic PUT inherently replaces the object, so
        ``overwrite=True`` (the natural semantic of a single-shot
        upload) needs no extra round trip — that's the whole
        point of letting the caller signal it.

        ``size>=0`` (capped read) or non-zero ``offset`` fall
        back to the chunked base path because the API can't
        splice at a range and reads the full body without an
        upper bound. ``batch_size`` only matters for that
        fallback — the atomic upload doesn't chunk.
        """
        if offset != 0 or size >= 0:
            return super()._write_stream(src, offset=offset, size=size, **kwargs)
        return self._upload(src)

    def _upload(self, content: Any) -> int:
        """Upload *content* through ``workspace.upload`` with retry semantics.

        Accepts either a bytes-like payload (``bytes`` /
        ``bytearray`` / ``memoryview``) or a seekable binary stream.
        Streams ride through to the SDK verbatim — no eager
        ``read()`` into a buffer — and get rewound to origin on
        every retry so transient-error / parent-recovery re-tries
        POST the full body, not an empty tail. Bytes-like payloads
        are passed through directly; ``WorkspaceExt.upload`` builds
        a fresh multipart body per request from the same ``bytes``.

        ``format=AUTO`` is the import-side hint — the SDK default
        is ``SOURCE``, which routes raw bytes through the notebook
        importer and fails with ``BadRequest: The zip archive
        contains no items``.

        Returns the byte count when known (bytes-like input) or
        ``-1`` when the input is a stream of unknown length.
        """
        size = len(content) if hasattr(content, "__len__") else -1
        logger.debug(
            "Uploading workspace file %r (%s bytes)",
            self,
            size if size >= 0 else "?",
        )
        upload = self.client.workspace_client().workspace.upload
        api_path = self.api_path
        fmt = _import_format_auto()

        if hasattr(content, "seek"):
            stream = content

            def _do_upload() -> None:
                # IO inputs ride through unbuffered; rewind to
                # origin on every attempt so the multipart POST
                # reads the full body even on a retry.
                stream.seek(0)
                upload(path=api_path, content=stream, format=fmt, overwrite=True)

        else:

            def _do_upload() -> None:
                # Bytes-like input — ``WorkspaceExt.upload`` will
                # build a fresh ``IO`` per request, so no
                # cursor state crosses retry attempts.
                upload(path=api_path, content=content, format=fmt, overwrite=True)

        try:
            self._call_ensuring_parents(_do_upload)
        except Exception as exc:
            # ``overwrite=True`` covers same-type re-uploads, but the
            # Workspace import API still refuses to replace a node with
            # one of a *different* type — e.g. a previously-staged
            # SparkPythonTask source file at ``main-<digest>.py`` can't
            # be overwritten by an AUTO upload of the same path when
            # the content sniffs as a notebook (notebooks are stored
            # at the extension-stripped stem). Delete the conflicting
            # node and retry once so re-stagings that change task type
            # — the ``async_job(force=True)`` after-upgrade scenario —
            # don't leave the caller stuck on a stale workspace entry.
            if not _looks_like_already_exists(exc):
                raise
            logger.warning(
                "Workspace upload of %r hit %s despite overwrite=True — "
                "deleting conflicting node and retrying",
                self,
                type(exc).__name__,
            )
            try:
                self._call(
                    self.client.workspace_client().workspace.delete,
                    api_path,
                    recursive=False,
                )
            except Exception as del_exc:
                if not _looks_like_not_found(del_exc):
                    raise
            self._call_ensuring_parents(_do_upload)
        if size >= 0:
            self._persist_stat_cache(
                IOStats(
                    size=size,
                    kind=IOKind.FILE,
                    mtime=time.time(),
                    media_type=self.media_type,
                )
            )
            logger.info("Uploaded workspace file %r (size=%d)", self, size)
        else:
            logger.info("Uploaded workspace file %r (size=stream)", self)
        return size

    # ==================================================================
    # Module upload — stream directly through ``workspace.upload``
    # ==================================================================

    def upload_module(
        self,
        module: Any,
        *,
        name: str | None = None,
        overwrite: bool = True,
    ) -> "WorkspacePath":
        """Pack a local module / package and import it into the workspace.

        Mirrors the :meth:`Path.upload_module` contract but bypasses
        the generic ``read_bytes()`` round-trip: the produced
        ``.zip`` is streamed straight into ``workspace.upload`` from
        an open file handle, so a large archive doesn't get
        materialized into a Python ``bytes`` object before the
        upload.

        Destination resolution stays consistent with the base:
        ``self`` with a ``.zip`` / ``.whl`` suffix is taken
        verbatim; anything else gets ``self / <name or
        "<module>.zip">``. The format hint is :class:`ImportFormat.AUTO`
        — the Workspace API detects the binary content type and
        stores the object as a workspace file (not a notebook).
        """
        from yggdrasil.path._module_pack import (
            build_module_archive,
            resolve_module_root,
        )

        local_root = resolve_module_root(module)
        suffix = self.suffix.lower()
        archive_default = (
            name
            if name is not None
            else (
                local_root.name
                if local_root.is_file() and suffix in (".zip", ".whl")
                else f"{local_root.name}.zip"
            )
        )

        target: "WorkspacePath" = (
            self if suffix in (".zip", ".whl") else self / archive_default
        )

        if not overwrite and target.exists():
            raise FileExistsError(
                f"upload_module: destination {target.full_path()!r} "
                f"already exists. Pass overwrite=True to replace it."
            )

        archive_path = build_module_archive(local_root, dest=None)
        try:
            size = archive_path.stat().st_size
            with open(archive_path, "rb") as fh:
                # Hand the live file handle to ``_upload`` — it owns
                # the seek-on-retry contract, so a large archive
                # never gets read into a Python ``bytes`` object
                # before the upload.
                target._upload(fh)
        finally:
            if local_root != archive_path:
                try:
                    archive_path.unlink()
                except OSError:
                    pass

        target._persist_stat_cache(
            IOStats(
                size=int(size),
                kind=IOKind.FILE,
                mtime=time.time(),
                media_type=target.media_type,
            )
        )
        return target

    def _clear(self) -> None:
        self._remove_file(missing_ok=True, wait=WaitingConfig.from_(True))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _media_type_from_response(info) -> "MediaType | None":
    """Resolve a :class:`MediaType` from a Workspace download response.

    The SDK exposes the MIME type as ``content_type`` on the response
    object (when set on import). Returns ``None`` when absent — the
    caller falls back to URL-extension inference.
    """
    if info is None:
        return None
    mime = getattr(info, "content_type", None) or getattr(info, "mime_type", None)
    if not mime:
        return None
    return MediaType.from_(mime, default=None)


def _looks_like_not_found(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in (
        "NotFound",
        "ResourceDoesNotExist",
        "FileNotFoundError",
    ) or isinstance(exc, FileNotFoundError)


def _looks_like_already_exists(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("AlreadyExists", "ResourceAlreadyExists", "FileExistsError"):
        return True
    return "already exists" in str(exc).lower()


def _looks_like_protected_parent(exc: BaseException) -> bool:
    """``mkdirs`` can fail with ``BadRequest: Folder X is protected`` when
    an ancestor (typically ``/Workspace/Users``) refuses creation. The
    leaf the caller actually wants is independent of the protected
    ancestor — treat as non-fatal."""
    return "is protected" in str(exc).lower()


def _import_format_auto() -> Any:
    """Resolve the SDK's ``ImportFormat.AUTO`` enum, falling back to a string.

    The Databricks SDK accepts the enum or the literal ``"AUTO"``;
    the string fallback keeps the helper usable in test environments
    that mock the workspace client without the SDK installed.
    """
    try:
        from databricks.sdk.service.workspace import ImportFormat

        return ImportFormat.AUTO
    except Exception:
        return "AUTO"


def _export_format_auto() -> Any:
    """Resolve the SDK's ``ExportFormat.AUTO`` enum, falling back to a string.

    Mirror of :func:`_import_format_auto` for the download side —
    ``ExportFormat`` is a sibling enum on the workspace SDK and
    accepts ``"AUTO"`` as a literal string when the enum import is
    unavailable (mocked test environments).
    """
    try:
        from databricks.sdk.service.workspace import ExportFormat

        return ExportFormat.AUTO
    except Exception:
        return "AUTO"
