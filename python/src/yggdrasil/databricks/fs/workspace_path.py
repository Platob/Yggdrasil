""":class:`WorkspacePath` — ``/Workspace/...`` via the Workspace API.

Workspace objects (notebooks, files, directories) are managed
through ``workspace.workspace.*``. There's no FUSE counterpart;
every read / write is SDK-mediated.

The Workspace API has a single download / upload pair (no range
reads, no positional writes), so the byte primitives map onto a
read-modify-rewrite scheme.
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar, Iterator

from yggdrasil.data.enums import Scheme
from yggdrasil.data.enums.media_type import MediaType
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL

from ..path import DatabricksPath
from ..workspaces.service import Workspaces


__all__ = ["WorkspacePath"]
from ...dataclasses import WaitingConfig


logger = logging.getLogger(__name__)


# Process-wide cache of resolved usernames, keyed by ``id`` of the
# bound workspace client. One ``current_user.me()`` round-trip per
# client; cleared implicitly when the client is garbage-collected
# (the next caller gets a fresh ``id`` from the allocator).
_USER_NAME_CACHE: dict[int, str] = {}


class WorkspacePath(DatabricksPath):
    """Path under ``/Workspace/...`` via the Workspace API."""

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_WORKSPACE
    namespace_prefix: ClassVar[str] = "/Workspace/"
    _SERVICE_CLASS: ClassVar[type] = Workspaces

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
            info, "userName", None,
        )
        if not isinstance(name, str) or not name:
            return ""
        _USER_NAME_CACHE[key] = name
        return name

    # ==================================================================
    # Stat
    # ==================================================================

    def _stat_uncached(self) -> IOStats:
        try:
            info = self._call(
                self.client.workspace_client().workspace.get_status, self.api_path,
            )
        except Exception:
            return IOStats(kind=IOKind.MISSING, size=0, mtime=0.0)

        ot = getattr(info, "object_type", None)
        is_dir = (
            getattr(ot, "name", None) == "DIRECTORY"
            or str(ot).upper().endswith("DIRECTORY")
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
        try:
            entries = list(
                self._call(self.client.workspace_client().workspace.list, self.api_path)
            )
        except Exception:
            return
        logger.debug(
            "Listing workspace directory %r -> %d entries (recursive=%s)",
            self, len(entries), recursive,
        )
        for info in entries:
            child_path = getattr(info, "path", None)
            if not child_path:
                continue
            # Workspace.list returns paths under ``/Workspace/...``;
            # the URL path is the namespace-stripped suffix.
            url_path = child_path
            if url_path.startswith("/Workspace/"):
                url_path = url_path[len("/Workspace"):]
            elif url_path.startswith("/Workspace"):
                url_path = url_path[len("/Workspace"):] or "/"
            # ``singleton_ttl`` defaults to ``False`` so listing
            # children stay out of ``DatabricksPath._INSTANCES``;
            # callers wanting cached children pass it through ``ls``.
            child = type(self)(
                url=URL(scheme=self.scheme, path=url_path),
                service=self.service,
                singleton_ttl=singleton_ttl,
            )
            ot = getattr(info, "object_type", None)
            is_dir = (
                getattr(ot, "name", None) == "DIRECTORY"
                or str(ot).upper().endswith("DIRECTORY")
            )
            # Seed from the listing entry so the caller's ``is_file()``
            # / ``size`` / ``exists()`` per child don't each issue a
            # follow-up ``workspace.get_status`` round trip.
            mtime_ms = getattr(info, "modified_at", None) or 0
            child._seed_stat_cache(IOStats(
                kind=IOKind.DIRECTORY if is_dir else IOKind.FILE,
                size=0 if is_dir else int(getattr(info, "size", 0) or 0),
                mtime=float(mtime_ms) / 1000.0 if mtime_ms else 0.0,
            ))
            yield child
            if recursive and is_dir:
                yield from child._ls(recursive=True, singleton_ttl=singleton_ttl)

    # ==================================================================
    # Mutators
    # ==================================================================

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        del parents
        logger.debug("Creating workspace directory %r", self)
        try:
            self._call(self.client.workspace_client().workspace.mkdirs, self.api_path)
        except Exception as exc:
            if _looks_like_already_exists(exc):
                if not exist_ok:
                    raise
                self._seed_stat_cache(IOStats(kind=IOKind.DIRECTORY))
                return
            if _looks_like_protected_parent(exc):
                # Hitting a protected ancestor (e.g. ``/Workspace/Users``)
                # is fine if the leaf already landed — fall through and
                # let downstream ops succeed.
                self._seed_stat_cache(IOStats(kind=IOKind.DIRECTORY))
                return
            raise
        self._seed_stat_cache(IOStats(kind=IOKind.DIRECTORY))

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        del wait
        logger.debug("Deleting workspace file %r", self)
        try:
            self._call(
                self.client.workspace_client().workspace.delete,
                self.api_path, recursive=False,
            )
        except Exception:
            if not missing_ok:
                raise
        self.invalidate_singleton()

    def _remove_dir(
        self, recursive: bool, missing_ok: bool, wait: WaitingConfig,
    ) -> None:
        del wait
        logger.debug(
            "Deleting workspace directory %r (recursive=%s)",
            self, recursive,
        )
        try:
            self._call(
                self.client.workspace_client().workspace.delete,
                self.api_path, recursive=recursive,
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
            self, len(data), pos, "EOF" if n < 0 else n,
        )

        # ``workspace.download`` always returns the whole object, so its
        # length IS the file size. Seed the stat cache (or refresh the
        # ``size`` slot when the prior entry came from a metadata probe)
        # so the next ``size`` / ``exists`` lookup is local.
        media_type = _media_type_from_response(response)
        if self._stat_cached is None:
            self._seed_stat_cache(IOStats(
                size=len(data),
                kind=IOKind.FILE,
                media_type=media_type,
            ))
        else:
            self._stat_cached.size = len(data)
            if media_type is not None and self._stat_cached.media_type is None:
                self._stat_cached.media_type = media_type
            # Re-stamp the TTL — this download IS the freshest size we
            # could observe; the entry should outlive the original
            # probe's window from this point on.
            self._seed_stat_cache(self._stat_cached)

        if pos:
            data = data[pos:]
        if n > 0:
            data = data[:n]
        return memoryview(data)

    def _write_stream(self, src: Any, *, offset: int, size: int = -1) -> int:
        """Override the base chunked stream — Workspace wants one PUT.

        The Workspace API has no positional / range write, so a
        chunked :meth:`Holder._write_stream` would issue one RMW
        per chunk. Hand the live :class:`IO[bytes]` straight to
        :meth:`_upload`, which already seek-rewinds on retry and
        the SDK builds the multipart body lazily — multi-GB
        sources never materialise as a Python ``bytes`` object.

        ``size>=0`` (capped read) or non-zero ``offset`` fall
        back to the chunked base path because the API can't
        splice at a range and reads the full body without an
        upper bound.
        """
        if offset != 0 or size >= 0:
            return super()._write_stream(src, offset=offset, size=size)
        return self._upload(src)

    def _write_mv(self, data: memoryview, pos: int) -> int:
        n = len(data)
        if n == 0:
            return 0
        if pos == 0:
            payload = bytes(data)
        else:
            # Single ``workspace.download`` round trip — no preceding
            # ``get_status`` probe. The Workspace API delivers the
            # whole object on any download call, so asking for
            # "to EOF" is no more expensive than a sized read.
            try:
                existing = bytes(self._read_mv(-1, 0))
            except FileNotFoundError:
                existing = b""
            if pos > len(existing):
                existing = existing + b"\x00" * (pos - len(existing))
            payload = existing[:pos] + bytes(data) + existing[pos + n:]
        self._upload(payload)
        return n

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
            self, size if size >= 0 else "?",
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
                # build a fresh ``BytesIO`` per request, so no
                # cursor state crosses retry attempts.
                upload(path=api_path, content=content, format=fmt, overwrite=True)

        self._call_ensuring_parents(_do_upload)
        if size >= 0:
            self._seed_stat_cache(IOStats(
                size=size,
                kind=IOKind.FILE,
                mtime=time.time(),
                media_type=self.media_type,
            ))
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
        from yggdrasil.io.path._module_pack import (
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
            self
            if suffix in (".zip", ".whl")
            else self / archive_default
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

        target._seed_stat_cache(IOStats(
            size=int(size),
            kind=IOKind.FILE,
            mtime=time.time(),
            media_type=target.media_type,
        ))
        return target

    def truncate(self, n: int) -> int:
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        if n == 0:
            self._upload(b"")
            return 0
        # Single ``workspace.download`` round trip — no preceding
        # ``get_status``. A missing target surfaces as zero bytes and
        # we upload a fresh zero-padded head.
        try:
            existing = bytes(self._read_mv(-1, 0))
        except FileNotFoundError:
            existing = b""
        if n <= len(existing):
            head = existing[:n]
        else:
            head = existing + b"\x00" * (n - len(existing))
        self._upload(head)
        return n

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
    mime = (
        getattr(info, "content_type", None)
        or getattr(info, "mime_type", None)
    )
    if not mime:
        return None
    return MediaType.from_(mime, default=None)


def _looks_like_not_found(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in (
        "NotFound", "ResourceDoesNotExist", "FileNotFoundError",
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
