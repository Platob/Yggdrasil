"""Abstract :class:`DatabricksPath` — Databricks-aware :class:`Path` base.

What this class adds on top of :class:`yggdrasil.io.fs.path.Path`
---------------------------------------------------------------

- Legacy POSIX path parsing (``/dbfs/…``, ``/Workspace/…``,
  ``/Volumes/…``, ``/Tables/…``) so ``Path("/Volumes/cat/sch/vol/x")``
  routes to :class:`VolumePath` without the caller having to write
  the URL form.
- Client binding: :attr:`client`, :meth:`with_client`,
  :meth:`connect`, :meth:`connected`. The client is set late and
  doesn't participate in identity (two paths with the same URL but
  different clients still compare equal).
- A directory-aware :meth:`copy_to` override — the base streaming
  copy opens both endpoints as files, which 400s on the SDK for
  bare directory paths. We branch on :meth:`is_dir` and walk the
  tree.

Big design change vs. the legacy
--------------------------------

**No more ``DatabricksIO`` subclasses.** The previous architecture
had four IO classes (``DatabricksIO`` abstract base, plus
``DBFSIO``, ``WorkspaceIO``, ``VolumeIO``) each managing buffer
plumbing, mode handling, fingerprint dedup, commit-on-close.

All of that is now folded into :class:`yggdrasil.io.buffer.bytes_io.BytesIO`
itself: when a ``BytesIO`` is constructed with a non-local path,
it builds an internal *transaction buffer* (another ``BytesIO``,
autonomous), fills it via ``path.pread`` on acquire, commits via
``path.write_bytes`` on flush. Mode handling, fingerprint dedup,
and commit semantics live there.

What's left for Databricks paths to implement is the
positional-IO contract that ``Path`` declares abstract:

- :meth:`pread` — read N bytes at offset (range-read or
  download-and-slice).
- :meth:`pwrite` — write N bytes at offset (read-modify-write).
- :meth:`read_bytes` — full download.
- :meth:`write_bytes` — full upload.

The last two override the base ``Path``'s default implementations
(which go through ``open_io``) to call the SDK directly. Otherwise
opening a path-bound BytesIO would recurse: BytesIO._acquire calls
path.pread → which would call open_io → which would construct a
new BytesIO bound to the path → which would try to acquire …

So Databricks paths bypass ``open_io`` for their own bytes I/O,
and ``_open`` returns a plain ``BytesIO(path=self, mode=mode)``
with no special subclass — the BytesIO machinery handles
everything.

Per-subclass responsibilities
-----------------------------

Each concrete subclass (DBFSPath / WorkspacePath / VolumePath /
TablePath) implements:

- The abstract ``Path`` hooks: ``full_path``, ``_stat``, ``_ls``,
  ``_mkdir``, ``_remove_file``, ``_remove_dir``, ``_open``.
- Two new SDK transport hooks: ``_remote_download`` (whole-object
  GET → bytes) and ``_remote_upload`` (whole-object PUT, takes
  bytes).
- Optionally fast paths for ``pread``/``pwrite`` if the SDK has
  range-IO primitives (DBFS FUSE does; the others don't).

The base class wires ``read_bytes`` → ``_remote_download``,
``write_bytes`` → ``_remote_upload``, and provides default
``pread``/``pwrite`` that go through download-and-slice /
read-modify-write.
"""

from __future__ import annotations

import logging
import threading
from abc import abstractmethod
from dataclasses import replace as _dc_replace
from typing import IO, TYPE_CHECKING, Any, ClassVar, Optional, Tuple, Union

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.fs.path import Path, _select_path_class
from yggdrasil.io.url import URL
from ._errors import retry_sdk_call
from .path_kind import DatabricksPathKind
from ...lazy_imports import databricks_client_class

if TYPE_CHECKING:
    from ..client import DatabricksClient


__all__ = [
    "DatabricksPath",
    "DatabricksPathKind",
]


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy POSIX path parsing
# ---------------------------------------------------------------------------

_NAMESPACES = {"dbfs+fuse", "dbfs+workspace", "dbfs+volumes", "dbfs+tables"}


def _parse_legacy(raw: str) -> Tuple[str, str]:
    """Parse a POSIX Databricks path → ``(scheme, url_path)``."""
    raw = URL.from_(raw).path

    if not raw.startswith("/"):
        raise ValueError(f"Not a POSIX Databricks path: {raw!r}")
    parts = raw.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Not a POSIX Databricks path: {raw!r}")
    namespace = parts[1].lower()
    if namespace not in _NAMESPACES:
        raise ValueError(
            f"Not a POSIX Databricks path: {raw!r} "
            f"(namespace {parts[1]!r} not in {sorted(_NAMESPACES)})"
        )
    rest = "/" + parts[2] if len(parts) == 3 else "/"
    return namespace, rest


def _looks_like_legacy_databricks(s: str) -> bool:
    if not isinstance(s, str) or not s.startswith("/"):
        return False
    parts = s.split("/", 2)
    return len(parts) >= 2 and parts[1].lower() in _NAMESPACES


# ===========================================================================
# DatabricksPath
# ===========================================================================


class DatabricksPath(Path):
    """Abstract :class:`Path` for Databricks namespaces.

    Concrete subclasses (DBFSPath / WorkspacePath / VolumePath /
    TablePath) implement two SDK-transport hooks, plus the abstract
    ``Path`` surface. The buffered-IO machinery is provided by
    :class:`BytesIO` itself; this class no longer instantiates a
    custom IO subclass.
    """

    __slots__ = ("_client",)

    scheme: ClassVar[str] = "dbfs"

    #: Canonical POSIX prefix for the legacy string shape
    #: (``/dbfs/``, ``/Workspace/``, …). Empty on the abstract base.
    _NAMESPACE_PREFIX: ClassVar[Optional[str]] = None

    # ==================================================================
    # Construction — chain through Path.__init__
    # ==================================================================

    def __new__(cls, obj: Any = None, *args: Any, **kwargs: Any) -> "Path":
        del args, kwargs

        if cls is DatabricksPath:
            for sub in cls.__subclasses__():
                if sub.handles(obj):
                    return sub.__new__(obj, obj)

            target = _select_path_class(obj)
            return target.__new__(target, obj)
        return super().__new__(cls, obj)

    def __init__(
        self,
        obj: Any = None,
        *,
        url: Optional[URL] = None,
        temporary: bool = False,
        auto_open: bool = True,
        client: Optional["DatabricksClient"] = None,
    ) -> None:
        coerced = self._coerce_legacy(obj)

        super().__init__(
            coerced,
            url=url,
            temporary=temporary,
            auto_open=auto_open,
        )

        # Stamp the class scheme onto a URL that came in without one.
        scheme = getattr(self, "scheme", "") or ""
        if scheme and not self.url.scheme:
            self.url = _dc_replace(self.url, scheme=scheme)

        # Strip our canonical namespace prefix if a URL-form input
        # retained it.
        prefix = self._NAMESPACE_PREFIX
        if prefix:
            stripped = self._strip_namespace_prefix(self.url.path)
            if stripped != self.url.path:
                self.url = _dc_replace(self.url, path=stripped)

        self._client = client

    @classmethod
    def _coerce_legacy(cls, obj: Any) -> Any:
        """Pre-coerce inputs the new :class:`Path` base doesn't handle.

        Only intervenes for strings that look like canonical
        Databricks POSIX paths; everything else passes through.
        """
        if isinstance(obj, str) and _looks_like_legacy_databricks(obj):
            scheme, path = _parse_legacy(obj)
            return f"{scheme}://{path}"
        return obj

    # ==================================================================
    # Classification — extend Path.handles for legacy POSIX
    # ==================================================================

    @classmethod
    def handles(cls, obj: Any) -> bool:
        if cls is DatabricksPath:
            for sub in cls.__subclasses__():
                if sub.handles(obj):
                    return True
            return False

        if not cls.scheme:
            return False
        if not cls._NAMESPACE_PREFIX:
            return False

        if Path.is_pathish(obj):
            url = URL.from_(
                obj,
                default_scheme=cls.scheme,
                default=None
            )

            if url is not None:
                return url.scheme == cls.scheme or (
                    url.path and url.path.startswith(cls._NAMESPACE_PREFIX)
                )

        ns, _ = _parse_legacy(obj)
        return ns == cls.scheme

    @classmethod
    def _strip_namespace_prefix(cls, raw: str) -> str:
        if not raw.startswith("/"):
            raw = "/" + raw
        prefix = cls._NAMESPACE_PREFIX
        if not prefix:
            return raw
        if raw.startswith(prefix):
            tail = raw[len(prefix):].lstrip("/")
            return "/" + tail if tail else "/"
        if raw == prefix.rstrip("/"):
            return "/"
        return raw

    # ==================================================================
    # Coercion entry points
    # ==================================================================

    @classmethod
    def from_(
        cls,
        obj: Any,
        default: Any = ...,
        *,
        temporary: bool = False,
        client: "DatabricksClient | None" = None,
    ) -> "DatabricksPath":
        if isinstance(obj, DatabricksPath):
            same_type = type(obj) is cls or cls is DatabricksPath
            if same_type:
                if temporary:
                    obj.temporary = True
                if client is not None:
                    obj._client = client
                return obj
            return cls.from_url(
                obj.url,
                default=default, temporary=temporary,
                client=client
            )

        if Path.is_pathish(obj):
            return cls.from_url(
                URL.from_(obj),
                default=default,
                temporary=temporary,
                client=client
            )

        return super().from_(obj, default=default, temporary=temporary)

    @classmethod
    def from_url(
        cls,
        url: URL,
        default: Any = ...,
        *,
        temporary: bool = False,
        client: "DatabricksClient | None" = None,
    ) -> "Path":
        """Build a :class:`Path` from a URL, dispatching by scheme.

        Dead-branch removed vs. previous version: ``URL.from_``
        either returns a :class:`URL` or raises, never ``None``,
        so the ``if resolved is None`` check never fired.
        """
        try:
            resolved = URL.from_(url)
        except (ValueError, TypeError):
            if default is ...:
                raise
            return default

        if cls is DatabricksPath:
            for sub in cls.__subclasses__():
                if sub.handles(resolved):
                    client = databricks_client_class().current() if client is None else client
                    return sub(
                        url=resolved, temporary=temporary,
                        client=client
                    )

        return super().from_url(resolved, default=default, temporary=temporary)

    @property
    def is_local(self) -> bool:
        # Default False — subclasses (notably DBFSPath) override
        # when a local-FS view exists (FUSE mount).
        return False

    def to_sql_string_location(self, file_format: str):
        return f"{file_format}.`{self.full_path()}`"

    def _from_url(self, url: URL) -> "Path":
        """Build a same-typed :class:`Path` from *url*.

        Stays on ``type(self)`` so subclass overrides flow through.
        Does NOT propagate :attr:`temporary` — navigation produces
        a new identity, and the temp lifecycle is anchored to the
        original Path that claimed it.
        """
        return type(self)(
            url=url,
            client=self.client,
        )

    # ==================================================================
    # Client binding
    # ==================================================================

    @property
    def client(self) -> "DatabricksClient":
        if self._client is None:
            raise RuntimeError(
                f"{self!r} has no Databricks client bound. Use "
                "with_client(...) or pass client= at construction."
            )
        return self._client

    @property
    def connected(self) -> bool:
        return self._client is not None and getattr(
            self._client, "connected", False,
        )

    @property
    def workspace(self):
        """SDK :class:`WorkspaceClient` shortcut."""
        return self.client.workspace_client()

    def _sdk(self):
        return self.workspace

    def with_client(self, client: "DatabricksClient") -> "DatabricksPath":
        self._client = client
        return self

    def connect(self) -> "DatabricksPath":
        if self._client is None:
            raise RuntimeError(
                f"{self!r} has no Databricks client bound. Use "
                "with_client(...) before calling connect()."
            )
        if not self.connected:
            self._client.connect()
        return self

    def _unsafe_remove(self) -> None:
        try:
            self.remove(recursive=True, allow_not_found=True)
        except BaseException:
            try:
                LOGGER.debug(
                    "Shutdown cleanup of %s failed",
                    self.full_path(), exc_info=True,
                )
            except Exception:
                pass

    def close(self, wait: bool = True) -> None:
        """Close the path. ``wait=False`` spawns a daemon thread so the
        remote-remove can't block interpreter shutdown."""
        if not self.temporary or wait:
            super().close()
            return

        threading.Thread(
            target=self._unsafe_remove,
            name=f"databricks-path-close-{self.name or 'root'}",
            daemon=True,
        ).start()

    # ==================================================================
    # I/O — _open returns a plain BytesIO bound to this path
    # ==================================================================
    #
    # The major change vs. the legacy: we no longer construct a
    # custom DatabricksIO subclass. The base BytesIO already knows
    # how to handle non-local paths via its transaction-buffer
    # machinery — it'll call self.pread to fill on acquire and
    # self.write_bytes to commit on flush. That's all we need.

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
        """Path-bound :class:`BytesIO`, with lazy SDK connect.

        Same shape as :meth:`Path._open` (the new concrete default)
        but ensures the SDK client is connected and normalises text
        modes to binary at the buffer layer.
        """
        # Bind the client lazily on first IO if it isn't connected.
        if self._client is not None and not self.connected:
            self.connect()

        # Promote text mode to binary internally.
        if "b" not in mode:
            mode = mode.replace("t", "")
            if "b" not in mode:
                mode += "b"

        return super()._open(
            mode=mode,
            encoding=encoding,
            errors=errors,
            newline=newline,
            auto_open=auto_open,
            touch=touch,
        )

    def open(
        self,
        mode: str = "rb",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        *,
        auto_open: bool = True,
        touch: bool = False,
    ) -> IO[bytes]:
        """``builtins.open``-shaped open. Binary or text handle.

        Provided for parity with the legacy ``FileSystem.open``
        signature. ``buffering`` is forwarded for stdlib parity but
        ignored — the underlying :class:`BytesIO` does its own
        sizing.

        Text-mode (``mode`` without ``b``) returns a
        :class:`io.TextIOWrapper` over the binary handle.
        """
        del buffering

        if "b" in mode:
            return self.open_io(
                mode=mode,
                encoding=encoding,
                auto_open=auto_open,
                touch=touch,
            )

        binary_mode = mode.replace("t", "") + "b"
        binary_mode = binary_mode.replace("bb", "b")
        binary_handle = self.open_io(
            mode=binary_mode,
            encoding=None,
            auto_open=auto_open,
            touch=touch,
        )

        import io as _stdio
        return _stdio.TextIOWrapper(
            binary_handle,
            encoding=encoding or "utf-8",
            errors=errors or "strict",
            newline=newline,
            write_through=True,
        )

    # ==================================================================
    # Whole-file primitives — direct SDK transport
    # ==================================================================
    #
    # :class:`Path` requires :meth:`_pread` (download → BytesIO) and
    # :meth:`_pwrite` (upload from BytesIO).  Wired straight onto the
    # subclass-supplied ``_remote_download`` / ``_remote_upload``
    # SDK calls, with retry on the transport flakes.  All other
    # I/O (read_bytes, write_bytes, pread, pwrite, truncate, …)
    # falls out of the base implementations on top.

    def _pread(self) -> BytesIO:
        """Whole-file download via :meth:`_remote_download`.

        Retried on transient transport errors; semantic errors
        (``FileNotFoundError`` / ``NotFound`` / ``BadRequest``)
        propagate on the first attempt. Returns the project's
        spill-capable :class:`BytesIO` so large downloads stream to
        local temp rather than hogging RAM.
        """
        content = retry_sdk_call(
            self._remote_download, allow_not_found=False,
        )

        if isinstance(content, BytesIO):
            if not content.opened:
                content.open()
            content.seek(0)
            return content

        bio = BytesIO()
        bio.open()
        if content:
            bio.write(content)
            bio.seek(0)
        return bio

    def _pwrite(self, data: BytesIO) -> int:
        """Whole-file upload via :meth:`_remote_upload`.

        Most Databricks SDKs don't expose positional writes — the
        streaming upload truncates the destination. Auto-creates
        parent directories on :class:`FileNotFoundError`.  The
        retry wrapper seeks the buffer back to its starting offset
        before each replay so partial transport failures upload the
        exact same bytes.
        """
        if not data.opened:
            data.open()
        start = data.tell()
        size = data.size

        def _seek_back(_attempt: int, _exc: BaseException) -> None:
            try:
                data.seek(start)
            except Exception:
                LOGGER.debug(
                    "Could not seek payload back to %s before retry "
                    "(buffer may not be seekable); continuing.",
                    start,
                )

        try:
            retry_sdk_call(self._remote_upload, data, on_retry=_seek_back)
        except FileNotFoundError:
            self.parent.mkdir(parents=True, exist_ok=True)
            data.seek(start)
            retry_sdk_call(self._remote_upload, data, on_retry=_seek_back)
        self.invalidate_mirror()
        return int(size)

    # ==================================================================
    # SDK transport — abstract, subclasses fill in
    # ==================================================================

    @abstractmethod
    def _remote_download(self, allow_not_found: bool = False) -> BytesIO:
        """Whole-object GET. Returns the project's :class:`BytesIO`.

        Implementations must drain the SDK response into a
        :class:`yggdrasil.io.buffer.bytes_io.BytesIO` so large
        downloads can spill to disk transparently rather than
        balloon in memory. The buffer is left positioned at the
        start so callers can ``read()`` / ``to_bytes()`` directly.

        Raises :class:`FileNotFoundError` on a missing object when
        ``allow_not_found=False``; returns an empty buffer when
        True. Other backend errors propagate.
        """

    @abstractmethod
    def _remote_upload(self, payload: BytesIO) -> None:
        """Whole-object PUT. Takes the project's :class:`BytesIO`.

        Implementations should ``seek(0)`` (or honour the buffer's
        current position) and stream the contents to the SDK so
        spill-backed payloads upload without a full in-memory
        copy. The base ``write_bytes`` resets the position to its
        original value before each retry so a transport flake
        replays the same bytes.

        Implementations should overwrite if the object already
        exists. ``FileNotFoundError`` on a missing parent directory
        is allowed to propagate; the caller (:meth:`write_bytes`)
        will retry once after creating the parent.
        """

    # ==================================================================
    # copy_to — directory-aware override
    # ==================================================================

    def copy_to(
        self,
        dest: Any,
        *,
        batch_size: int = 4 * 1024 * 1024,
        parents: bool = True,
    ) -> int:
        """Copy this path to *dest*; recurses for directories.

        The base streaming copy opens both endpoints as files —
        fine when ``self`` is a leaf, but the SDK reads on a
        directory path return ``BadRequest``. Branch on
        :meth:`is_dir` and walk children for the directory case,
        rebuilding the relative tree under ``dest``.
        """
        dest_path = Path.from_(dest)

        if not self.is_dir():
            return super().copy_to(
                dest_path, batch_size=batch_size, parents=parents,
            )

        if parents:
            dest_path.mkdir(parents=True, exist_ok=True)

        total = 0
        for child in self.ls(recursive=True, allow_not_found=True):
            try:
                rel = child.relative_to(self)
            except ValueError:
                continue
            target = dest_path.joinpath(*rel.parts)
            if child.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            total += Path.copy_to(
                child, target,
                batch_size=batch_size, parents=False,
            )
        return total

    # ==================================================================
    # Cross-scheme helpers
    # ==================================================================

    @property
    @abstractmethod
    def kind(self) -> DatabricksPathKind:
        """The :class:`DatabricksPathKind` enum tag for this scheme."""

    @property
    def local_os_path(self) -> Optional[str]:
        """Local-FS path when this remote is OS-mounted, else ``None``.

        Default ``None``; :class:`DBFSPath` overrides for FUSE.
        """
        return None

    @property
    def is_local_fs(self) -> bool:
        return self.local_os_path is not None

    def sql_volume_or_table_parts(self) -> Tuple[
        Optional[str], Optional[str], Optional[str], list,
    ]:
        return (None, None, None, [])

    def sql_engine(self):
        from yggdrasil.databricks.sql import SQLEngine
        return SQLEngine(client=self.client)

    # ==================================================================
    # Abstract — concrete subclasses must implement
    # ==================================================================

    @abstractmethod
    def full_path(self) -> str:
        """POSIX rendering with the namespace prefix.

        ``/dbfs/foo/bar``, ``/Workspace/x``, ``/Volumes/cat/sch/vol/x``,
        ``/Tables/cat/sch/tbl``.
        """