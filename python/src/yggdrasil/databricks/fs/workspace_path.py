""":class:`WorkspacePath` — ``/Workspace/...`` via the Workspace API.

Workspace objects (notebooks, files, directories) are managed
through ``workspace.workspace.*``. There's no FUSE counterpart;
every read / write is SDK-mediated.

The Workspace API has a single download / upload pair (no range
reads, no positional writes), so the byte primitives map onto a
read-modify-rewrite scheme.
"""

from __future__ import annotations

import io as _stdio
from typing import Any, ClassVar, Iterator

from yggdrasil.data.enums import Scheme
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL

from .path import DatabricksPath


__all__ = ["WorkspacePath"]


# Process-wide cache of resolved usernames, keyed by ``id`` of the
# bound workspace client. One ``current_user.me()`` round-trip per
# client; cleared implicitly when the client is garbage-collected
# (the next caller gets a fresh ``id`` from the allocator).
_USER_NAME_CACHE: dict[int, str] = {}


class WorkspacePath(DatabricksPath):
    """Path under ``/Workspace/...`` via the Workspace API."""

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_WORKSPACE
    namespace_prefix: ClassVar[str] = "/Workspace/"

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

    def _ls(self, recursive: bool = False) -> Iterator["WorkspacePath"]:
        try:
            entries = list(
                self._call(self.client.workspace_client().workspace.list, self.api_path)
            )
        except Exception:
            return
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
            child = type(self)(
                url=URL(scheme=self.scheme, path=url_path),
                client=self._client,
            )
            yield child
            ot = getattr(info, "object_type", None)
            is_dir = (
                getattr(ot, "name", None) == "DIRECTORY"
                or str(ot).upper().endswith("DIRECTORY")
            )
            if recursive and is_dir:
                yield from child._ls(recursive=True)

    # ==================================================================
    # Mutators
    # ==================================================================

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        try:
            self._call(self.client.workspace_client().workspace.mkdirs, self.api_path)
        except Exception as exc:
            if _looks_like_already_exists(exc):
                if not exist_ok:
                    raise
                self._invalidate_stat_cache()
                return
            if _looks_like_protected_parent(exc):
                # Hitting a protected ancestor (e.g. ``/Workspace/Users``)
                # is fine if the leaf already landed — fall through and
                # let downstream ops succeed.
                self._invalidate_stat_cache()
                return
            raise
        self._invalidate_stat_cache()

    def _remove_file(self, missing_ok: bool = True) -> None:
        try:
            self._call(
                self.client.workspace_client().workspace.delete,
                self.api_path, recursive=False,
            )
        except Exception:
            if not missing_ok:
                raise
        self._invalidate_stat_cache()

    def _remove_dir(
        self, recursive: bool = True, missing_ok: bool = True,
    ) -> None:
        try:
            self._call(
                self.client.workspace_client().workspace.delete,
                self.api_path, recursive=recursive,
            )
        except Exception:
            if not missing_ok:
                raise
        self._invalidate_stat_cache()

    # ==================================================================
    # Holder I/O — download / upload
    # ==================================================================

    def _read_mv(self, n: int, pos: int) -> memoryview:
        if n == 0:
            return memoryview(b"")
        try:
            response = self._call(
                self.client.workspace_client().workspace.download, self.api_path,
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
        if pos:
            data = data[pos:]
        if n > 0:
            data = data[:n]
        return memoryview(data)

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

    def _upload(self, payload: bytes) -> None:
        # ``format`` defaults to ``ImportFormat.SOURCE`` in the Databricks
        # SDK, which routes through the notebook importer — non-notebook
        # bytes then fail with ``BadRequest: The zip archive contains
        # no items``. ``AUTO`` lets the server inspect the extension and
        # content to decide between workspace file and notebook.
        self._call_ensuring_parents(
            self.client.workspace_client().workspace.upload,
            path=self.api_path,
            content=_stdio.BytesIO(payload),
            format=_import_format_auto(),
            overwrite=True,
        )
        self._invalidate_stat_cache()

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
        self._remove_file(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
