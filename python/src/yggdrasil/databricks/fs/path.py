"""Abstract :class:`DatabricksPath` — Databricks-aware :class:`RemotePath`.

Concrete subclasses (:class:`DBFSPath`, :class:`VolumePath`,
:class:`WorkspacePath`) plug the SDK transport into the four
abstract :class:`Path` hooks (``_stat``, ``_ls``, ``_mkdir``,
``_remove_*``) plus the :class:`Holder` byte primitives
(``_read_mv`` / ``_write_mv`` / ``truncate`` / ``_clear``).

The base owns:

- **Legacy POSIX coercion** — strings like ``/dbfs/...``,
  ``/Workspace/...``, ``/Volumes/...`` are pre-coerced into the
  canonical ``dbfs://`` / ``workspace://`` / ``volumes://`` URL form
  so callers don't have to know the URL syntax.
- **Workspace-client binding** — :attr:`workspace` is whatever the
  caller injected, typically a ``databricks.sdk.WorkspaceClient``
  in production or a :class:`unittest.mock.Mock` in tests. No
  imports of ``databricks.sdk`` happen at module load — concrete
  subclasses pass any quack-compatible client through.

What's intentionally NOT here
-----------------------------

The legacy code had a ``DatabricksClient`` aggregator that owned a
workspace client, retry policies, account-level routing, and more.
That layer is decoupled in this rewrite — :class:`DatabricksPath`
just needs the workspace client itself, so:

- Tests pass a :class:`Mock` directly.
- Production callers pass ``DatabricksClient.current().workspace_client()``.

The aggregator can still exist; it just isn't required for the path
layer to work.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Optional, Tuple

from yggdrasil.data.enums import Scheme
from yggdrasil.io.path import RemotePath
from yggdrasil.io.path._retry import retry_sdk_call
from yggdrasil.io.url import URL


__all__ = ["DatabricksPath"]


# ---------------------------------------------------------------------------
# Legacy POSIX coercion
# ---------------------------------------------------------------------------

#: Map POSIX namespace prefix → canonical URL scheme. Keys are the
#: leading directory the user types (case-insensitive); values are the
#: scheme each subclass registers under (the ``dbfs+<surface>``
#: convention — see :class:`Scheme`).
_POSIX_NAMESPACES: dict[str, str] = {
    "dbfs":      Scheme.DATABRICKS_DBFS.value,
    "Volumes":   Scheme.DATABRICKS_VOLUME.value,
    "Workspace": Scheme.DATABRICKS_WORKSPACE.value,
}


def _looks_like_posix(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith("/"):
        return False
    parts = value.split("/", 2)
    if len(parts) < 2:
        return False
    return parts[1] in _POSIX_NAMESPACES or parts[1].lower() == "dbfs"


def _parse_posix(value: str) -> Tuple[str, str]:
    """``/dbfs/x`` → ``("dbfs", "/x")``, etc.

    Case-insensitive on the namespace; the rest of the path is
    preserved verbatim.
    """
    parts = value.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Not a Databricks POSIX path: {value!r}")
    raw_ns = parts[1]
    ns = raw_ns if raw_ns in _POSIX_NAMESPACES else raw_ns.lower()
    if ns not in _POSIX_NAMESPACES:
        raise ValueError(
            f"Not a Databricks POSIX path: {value!r} "
            f"(namespace {raw_ns!r} not in {sorted(_POSIX_NAMESPACES)})"
        )
    rest = "/" + parts[2] if len(parts) == 3 else "/"
    return _POSIX_NAMESPACES[ns], rest


def _coerce_to_url_str(value: Any) -> Any:
    """Convert a recognized POSIX Databricks path string into a URL.

    Pass-through for anything that doesn't match — :class:`Holder`
    handles the rest of the coercion (URL parsing, holder type
    dispatch).
    """
    if isinstance(value, str) and _looks_like_posix(value):
        scheme, path = _parse_posix(value)
        return f"{scheme}://{path}"
    return value


# ===========================================================================
# DatabricksPath
# ===========================================================================


class DatabricksPath(RemotePath):
    """Abstract :class:`RemotePath` for Databricks namespaces.

    Registers under :attr:`Scheme.DBFS` (the ``dbfs://`` family root)
    and acts as the dispatcher: :meth:`from_url` inspects the URL and
    forwards to the right concrete subclass (DBFS, Volumes,
    Workspace) based on the compound ``dbfs+<surface>://`` scheme,
    or — for the legacy un-prefixed ``dbfs://`` form — on the URL
    path's leading namespace (``/Volumes/...`` →
    :class:`VolumePath`, ``/Workspace/...`` → :class:`WorkspacePath`,
    everything else → :class:`DBFSPath`).
    """

    __slots__ = ("_workspace", "_retry_sleep")

    scheme: ClassVar[Scheme] = Scheme.DBFS

    #: Canonical POSIX prefix for the legacy string shape
    #: (``/dbfs/``, ``/Workspace/``, ``/Volumes/``). Empty on the
    #: abstract base; concrete subclasses override.
    namespace_prefix: ClassVar[Optional[str]] = None

    # ==================================================================
    # Construction
    # ==================================================================

    def __init__(
        self,
        data: Any = None,
        *,
        url: URL | None = None,
        workspace: Any = None,
        temporary: bool = False,
        retry_sleep: Optional[Callable[[float], None]] = None,
        **kwargs: Any,
    ) -> None:
        # Pre-coerce ``/dbfs/...`` / ``/Volumes/...`` / ``/Workspace/...``
        # into the canonical URL form so the URL parser sees a real
        # scheme.
        if data is not None:
            data = _coerce_to_url_str(data)

        if url is None and isinstance(data, str):
            url = URL.from_(data)
            data = None
        if url is None and isinstance(data, URL):
            url = data
            data = None
        if url is not None:
            url = URL.from_(url)
            target_scheme = self.scheme
            if target_scheme is not None:
                target_token = target_scheme.value if isinstance(
                    target_scheme, Scheme,
                ) else str(target_scheme)
                if not url.scheme:
                    url = url.with_scheme(target_token)
                elif url.scheme != target_token:
                    # Collapse legacy aliases (``volumes://`` for
                    # ``dbfs+volume://``, ``workspace://`` for
                    # ``dbfs+workspace://``, ``dbfs://`` for
                    # ``dbfs+dbfs://``) onto the canonical compound
                    # form so the URL round-trips cleanly.
                    resolved = Scheme.from_(url.scheme, default=None)
                    if resolved is target_scheme:
                        url = url.with_scheme(target_token)

        super().__init__(data=data, url=url, temporary=temporary, **kwargs)

        self._workspace = workspace
        self._retry_sleep: Optional[Callable[[float], None]] = retry_sleep

    # ==================================================================
    # URLBased dispatch — ``dbfs://`` resolves here and forwards
    # ==================================================================

    @classmethod
    def from_url(cls, url: "URL | str", **kwargs: Any) -> "DatabricksPath":
        """Construct the right concrete subclass from a Databricks URL.

        Two URL shapes are supported:

        - ``dbfs+dbfs://``, ``dbfs+volume://``, ``dbfs+workspace://``
          — the compound :class:`Scheme` form, dispatched by URL
          scheme alone.
        - ``dbfs://`` — legacy / un-qualified family URL. Dispatched
          by the URL path's leading namespace: ``/Volumes/...``
          →  :class:`VolumePath`, ``/Workspace/...`` →
          :class:`WorkspacePath`, anything else → :class:`DBFSPath`.

        Concrete subclasses (DBFSPath / VolumePath / WorkspacePath)
        bypass the dispatcher and forward straight to ``cls(url=url)``.
        """
        u = URL.from_(url)
        if cls is not DatabricksPath:
            return cls(url=u, **kwargs)

        from .dbfs_path import DBFSPath
        from .volume_path import VolumePath
        from .workspace_path import WorkspacePath

        scheme = (u.scheme or "").lower()
        if scheme == Scheme.DATABRICKS_DBFS.value:
            return DBFSPath(url=u, **kwargs)
        if scheme == Scheme.DATABRICKS_VOLUME.value:
            return VolumePath(url=u, **kwargs)
        if scheme == Scheme.DATABRICKS_WORKSPACE.value:
            return WorkspacePath(url=u, **kwargs)
        # Legacy ``dbfs://`` family URL — peek at the path. The
        # surface subclasses each carry their own POSIX prefix
        # (``/dbfs/``, ``/Volumes/``, ``/Workspace/``) which they
        # re-attach in :meth:`full_path`, so the URL path we hand
        # them must be the *suffix* below that prefix.
        raw_path = u.path or "/"
        path = raw_path.lstrip("/")
        head = path.split("/", 1)[0] if path else ""
        rest = path[len(head) + 1:] if "/" in path else ""
        suffix = "/" + rest if rest else "/"
        if head == "Volumes":
            return VolumePath(
                url=u.with_scheme(Scheme.DATABRICKS_VOLUME.value)._replace(path=suffix),
                **kwargs,
            )
        if head == "Workspace":
            return WorkspacePath(
                url=u.with_scheme(Scheme.DATABRICKS_WORKSPACE.value)._replace(path=suffix),
                **kwargs,
            )
        return DBFSPath(
            url=u.with_scheme(Scheme.DATABRICKS_DBFS.value),
            **kwargs,
        )

    # ==================================================================
    # Workspace client binding
    # ==================================================================

    @property
    def workspace(self) -> Any:
        """The workspace client (``databricks.sdk.WorkspaceClient`` or
        a mock).

        Lazily resolves through the aggregator client when none was
        injected. Tests inject mocks at construction so this branch
        never fires.
        """
        if self._workspace is None:
            from yggdrasil.lazy_imports import databricks_client_class
            self._workspace = databricks_client_class().current().workspace_client()
        return self._workspace

    def with_workspace(self, workspace: Any) -> "DatabricksPath":
        """Replace the bound workspace client. Returns *self*."""
        self._workspace = workspace
        return self

    # ==================================================================
    # Retry policy
    # ==================================================================

    def _call(self, func, *args, **kwargs):
        """Invoke *func(*args, **kwargs)* under the standard retry policy.

        Transient errors (BadRequest, InternalError, throttling,
        5xx, connect timeouts) get up to 4 retries with 1 / 2 / 4 /
        8 s sleeps. Permission errors get exactly one retry to
        absorb credential-refresh races. Anything else (NotFound,
        AlreadyExists, deterministic SDK errors) propagates.
        """
        if self._retry_sleep is not None:
            return retry_sdk_call(func, *args, sleep=self._retry_sleep, **kwargs)
        return retry_sdk_call(func, *args, **kwargs)

    # ==================================================================
    # Sibling paths inherit the same workspace client
    # ==================================================================

    def _from_url(self, url: URL) -> "DatabricksPath":
        return type(self)(
            url=url,
            workspace=self._workspace,
            retry_sleep=self._retry_sleep,
        )

    # ==================================================================
    # Holder primitives — defaults that work without a fast path
    # ==================================================================

    def reserve(self, n: int) -> None:
        """No-op — Databricks backends have no capacity layer."""
        if n < 0:
            raise ValueError(f"reserve size must be >= 0, got {n!r}")

    def _bread(self, n: int, pos: int, mode):  # pragma: no cover - thin shim
        """Fallback whole-file read into a fresh :class:`BytesIO`."""
        from yggdrasil.io.bytes_io import BytesIO
        del mode
        size = n if n >= 0 else max(0, int(self._stat().size) - pos)
        if size <= 0:
            return BytesIO()
        try:
            data = bytes(self._read_mv(size, pos))
        except FileNotFoundError:
            data = b""
        return BytesIO(data)

    def _bwrite(self, data, pos: int, mode) -> int:  # pragma: no cover
        """Fallback whole-file write from a :class:`BytesIO`."""
        del mode
        if hasattr(data, "to_bytes"):
            payload = data.to_bytes()
        elif hasattr(data, "read"):
            payload = data.read()
        else:
            payload = bytes(data)
        if not payload:
            return 0
        return self._write_mv(memoryview(payload), pos)

    # ==================================================================
    # Repr
    # ==================================================================

    def __repr__(self) -> str:
        marker = ", temporary=True" if self.temporary else ""
        return f"{type(self).__name__}({self.full_path()!r}{marker})"
