"""Abstract :class:`DatabricksPath` — Databricks-aware :class:`RemotePath`.

Concrete subclasses (:class:`DBFSPath`, :class:`VolumePath`,
:class:`WorkspacePath`) plug the SDK transport into the four
abstract :class:`Path` hooks (``_stat``, ``_ls``, ``_mkdir``,
``_remove_*``) plus the :class:`Holder` byte primitives
(``_read_mv`` / ``_write_mv`` / ``truncate`` / ``_clear``).

The base owns:

- **POSIX coercion** — strings like ``/dbfs/...``,
  ``/Workspace/...``, ``/Volumes/...`` are pre-coerced into the
  canonical ``dbfs+dbfs://`` / ``dbfs+workspace://`` /
  ``dbfs+volume://`` URL form so callers don't have to know the URL
  syntax.
- **Subclass dispatch** — :meth:`__new__`, :meth:`from_`, and
  :meth:`from_url` all route ``DatabricksPath(...)`` calls to the
  right concrete subclass (DBFS / Volumes / Workspace / Unity
  Catalog :class:`Table`) based on the URL scheme or the POSIX
  namespace in the path. Construction via the abstract base
  "just works" — no need for callers to pick a subclass up front.
- **Client binding** — :attr:`client` is a
  :class:`yggdrasil.databricks.client.DatabricksClient` aggregator.
  The SDK workspace handle is reached through ``client.workspace_client()``
  — :class:`DatabricksPath` never holds a bare workspace client
  directly. No imports of ``databricks.sdk`` happen at module load.

Tests pass a :class:`DatabricksClient` (or a mock shaped like one
— typically ``MagicMock()`` with ``workspace_client.return_value``
wired to a workspace-shaped mock). There is no alternate
"workspace-only" entry point: every caller routes through
:class:`DatabricksClient`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, Tuple

from yggdrasil.data.enums import Scheme
from yggdrasil.io.path import RemotePath
from yggdrasil.io.path._retry import retry_sdk_call
from yggdrasil.io.url import URL


if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient


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


# ---------------------------------------------------------------------------
# Subclass dispatch — shared by ``__new__`` / ``from_`` / ``from_url``
# ---------------------------------------------------------------------------


def _strip_dbfs_family_prefix(url: "URL") -> "URL":
    """Rewrite an un-qualified ``dbfs://`` URL into its concrete shape.

    ``dbfs:///Volumes/cat/sch/vol/x`` → ``dbfs+volume:///cat/sch/vol/x``
    ``dbfs:///Workspace/Users/me/x``  → ``dbfs+workspace:///Users/me/x``
    ``dbfs:///path``                  → ``dbfs+dbfs:///path``

    Any URL whose scheme is already a compound ``dbfs+<surface>`` (or
    something unrelated) is returned unchanged. Used by both the
    :class:`Holder`-level dispatch (``__new__`` / ``from_url``) so the
    leading namespace becomes part of the scheme rather than the path.
    """
    scheme = (url.scheme or "").lower()
    if scheme != Scheme.DBFS.value:
        return url
    raw_path = url.path or "/"
    path = raw_path.lstrip("/")
    head = path.split("/", 1)[0] if path else ""
    rest = path[len(head) + 1:] if "/" in path else ""
    suffix = "/" + rest if rest else "/"
    if head == "Volumes":
        return url.with_scheme(Scheme.DATABRICKS_VOLUME.value)._replace(path=suffix)
    if head == "Workspace":
        return url.with_scheme(Scheme.DATABRICKS_WORKSPACE.value)._replace(path=suffix)
    # Everything else stays under DBFS — flip the scheme but keep the
    # path verbatim so ``dbfs:///tmp/x`` and ``dbfs+dbfs:///tmp/x``
    # produce identical paths.
    return url.with_scheme(Scheme.DATABRICKS_DBFS.value)


def _resolve_databricks_subclass(
    *,
    data: Any = None,
    url: "URL | None" = None,
) -> Tuple[type, "URL | None"]:
    """Pick the concrete subclass for a Databricks-family URL/string.

    Returns ``(target_class, normalized_url)``. The normalized URL has
    the un-qualified ``dbfs://`` family prefix expanded into a concrete
    ``dbfs+<surface>://`` form (or is ``None`` when no URL information
    is available and the caller intends to fall through to a default
    subclass).

    Dispatch order:

    1. Explicit ``url=`` keyword.
    2. ``data`` already shaped like a :class:`URL`.
    3. ``data`` is a POSIX string (``/Volumes/...`` / ``/Workspace/...`` /
       ``/dbfs/...``) — coerced into a URL via :func:`_coerce_to_url_str`.
    4. ``data`` is a ``dbfs[+...]://`` URL string.

    Unknown shapes fall back to :class:`DBFSPath` — the DBFS surface
    is the historical home of ``dbfs://`` URLs that don't carry a
    leading namespace.
    """
    from .dbfs_path import DBFSPath
    from .volume_path import VolumePath
    from .workspace_path import WorkspacePath

    candidate: "URL | None" = url
    if candidate is None and isinstance(data, URL):
        candidate = data
    elif candidate is None and isinstance(data, str):
        coerced = _coerce_to_url_str(data)
        if isinstance(coerced, str) and "://" in coerced:
            try:
                candidate = URL.from_(coerced)
            except Exception:
                candidate = None

    if candidate is None:
        return DBFSPath, None

    candidate = _strip_dbfs_family_prefix(candidate)
    scheme = (candidate.scheme or "").lower()

    if scheme == Scheme.DATABRICKS_VOLUME.value:
        return VolumePath, candidate
    if scheme == Scheme.DATABRICKS_WORKSPACE.value:
        return WorkspacePath, candidate
    if scheme == Scheme.DATABRICKS_DBFS.value:
        return DBFSPath, candidate
    if scheme == Scheme.DATABRICKS_TABLE.value:
        # :class:`Table` lives in ``yggdrasil.databricks.sql.table``
        # and is *not* a :class:`DatabricksPath` subclass — it's a
        # logical Unity Catalog resource on the same ``dbfs+...``
        # family. The dispatcher still surfaces it here so callers
        # that go through ``DatabricksPath(...)`` don't have to know
        # the SQL module exists.
        from yggdrasil.databricks.sql.table import Table
        return Table, candidate

    # Unknown scheme (or empty) — let the caller's intended class
    # decide. DBFSPath is the safe default for the un-qualified
    # ``dbfs://`` family root.
    return DBFSPath, candidate


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

    __slots__ = ("_client", "_retry_sleep")

    scheme: ClassVar[Scheme] = Scheme.DBFS

    #: Canonical POSIX prefix for the legacy string shape
    #: (``/dbfs/``, ``/Workspace/``, ``/Volumes/``). Empty on the
    #: abstract base; concrete subclasses override.
    namespace_prefix: ClassVar[Optional[str]] = None

    # ==================================================================
    # Construction — dispatch on the abstract base, allocate on subclasses
    # ==================================================================

    def __new__(
        cls,
        data: Any = None,
        *,
        url: "URL | None" = None,
        **kwargs: Any,
    ):
        """Allocate the right concrete subclass when called on the base.

        ``DatabricksPath`` itself is abstract: instantiating it directly
        would fail on the abstract :class:`Path` hooks. Instead, peek at
        the inputs, pick the concrete subclass, and forward there. Python
        will call :meth:`__init__` on the returned instance using its
        actual class — so ``DatabricksPath("/Volumes/cat/sch/vol/x")``
        ends up running :meth:`VolumePath.__init__` with the original
        args.

        For a Unity Catalog :class:`Table` (which is *not* a
        :class:`DatabricksPath` subclass) we fully construct via
        :meth:`Table.from_url`; the returned object is not a
        :class:`DatabricksPath`, so Python skips the auto-``__init__``
        pass after ``__new__``.

        On a concrete subclass (``DBFSPath`` / ``VolumePath`` /
        ``WorkspacePath``) the call forwards up the MRO so
        :meth:`RemotePath.__new__` can apply its singleton-by-URL
        cache before the eventual ``object.__new__`` allocation —
        normalize the seed into a URL kwarg first so a POSIX-string
        construction (``VolumePath("/Volumes/cat/sch/vol/x")``)
        collapses to the same singleton as the URL-shaped one.
        """
        if cls is not DatabricksPath:
            if url is None and data is not None:
                _, normalized = _resolve_databricks_subclass(data=data)
                if normalized is not None:
                    url = normalized
                    data = None
            return super().__new__(cls, data=data, url=url, **kwargs)

        target, normalized = _resolve_databricks_subclass(data=data, url=url)
        # When dispatching to a DatabricksPath subclass, hand back
        # control to Python so the subclass's ``__init__`` fires with
        # the caller's original (data, url, **kwargs). Coerce ``data``
        # to ``None`` once we've folded its information into the URL,
        # so the subclass init doesn't double-parse it.
        if isinstance(target, type) and issubclass(target, DatabricksPath):
            if normalized is not None:
                data = None
                url = normalized
            return target.__new__(target, data=data, url=url, **kwargs)

        # Off-family target (Unity Catalog :class:`Table`) — construct
        # eagerly via ``from_url`` so the returned object is fully
        # initialized. Python won't auto-call ``__init__`` because the
        # result isn't a :class:`DatabricksPath` instance.
        if normalized is None:
            raise TypeError(
                f"{cls.__name__} cannot dispatch to {target.__name__} "
                f"without a URL or recognizable data shape; "
                f"got data={data!r}, url={url!r}."
            )
        return target.from_url(normalized, **kwargs)

    def __init__(
        self,
        data: Any = None,
        *,
        url: URL | None = None,
        client: "DatabricksClient | None" = None,
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
            # Un-qualified ``dbfs://`` family URLs whose path leads with
            # ``/Volumes/`` / ``/Workspace/`` belong to a concrete
            # surface — rewrite once so the URL path is the *suffix*
            # below the namespace prefix that ``full_path()`` re-adds.
            url = _strip_dbfs_family_prefix(url)
            target_scheme = self.scheme
            if target_scheme is not None:
                target_token = target_scheme.value if isinstance(
                    target_scheme, Scheme,
                ) else str(target_scheme)
                if not url.scheme:
                    url = url.with_scheme(target_token)

        super().__init__(data=data, url=url, temporary=temporary, **kwargs)

        # :class:`DatabricksClient` is the single source of truth.
        # The SDK workspace handle is always reached through
        # ``self.client.workspace_client()`` — the path holds no bare
        # workspace client. ``client=None`` is allowed; the path then
        # lazy-resolves to :meth:`DatabricksClient.current` on first
        # touch.
        self._client: Any = client
        self._retry_sleep: Optional[Callable[[float], None]] = retry_sleep

    # ==================================================================
    # URLBased dispatch — ``dbfs://`` resolves here and forwards
    # ==================================================================

    @classmethod
    def from_url(cls, url: "URL | str", **kwargs: Any) -> "DatabricksPath":
        """Construct the right concrete subclass from a Databricks URL.

        Four URL shapes are supported on :class:`DatabricksPath`
        itself:

        - ``dbfs+dbfs://``, ``dbfs+volume://``, ``dbfs+workspace://``
          — the compound :class:`Scheme` form, dispatched by URL
          scheme alone.
        - ``dbfs+table://[creds@]host/<cat>/<sch>/<tbl>?…`` — Unity
          Catalog logical table; dispatches to
          :class:`yggdrasil.databricks.sql.table.Table` (not a
          :class:`DatabricksPath`, but on the same scheme family).
        - ``dbfs://`` — un-qualified family URL. Dispatched by the
          URL path's leading namespace: ``/Volumes/...`` →
          :class:`VolumePath`, ``/Workspace/...`` →
          :class:`WorkspacePath`, anything else → :class:`DBFSPath`.

        Concrete subclasses (DBFSPath / VolumePath / WorkspacePath)
        bypass the dispatcher and forward straight to ``cls(url=url)``.
        """
        u = URL.from_(url)
        if cls is not DatabricksPath:
            return cls(url=u, **kwargs)

        target, normalized = _resolve_databricks_subclass(url=u)
        if normalized is None:
            normalized = u
        return target.from_url(normalized, **kwargs)

    @classmethod
    def from_(cls, obj: Any, **kwargs: Any) -> "DatabricksPath":
        """Coerce *obj* (string / URL / :class:`Path` / dict) into the
        right concrete subclass.

        On the abstract :class:`DatabricksPath`, this is the friendly
        entry point: POSIX strings like ``/Volumes/cat/sch/vol/x`` are
        coerced through :func:`_coerce_to_url_str` and routed by
        scheme to :class:`VolumePath` / :class:`WorkspacePath` /
        :class:`DBFSPath`, while compound ``dbfs+...://`` URLs
        dispatch by scheme alone (including ``dbfs+table://`` →
        :class:`Table`). On a concrete subclass, the call returns an
        instance of that subclass without redispatching — the standard
        :meth:`Path.from_` contract.
        """
        if isinstance(obj, DatabricksPath):
            if cls is DatabricksPath or isinstance(obj, cls):
                return obj
            obj = obj.url

        # POSIX-string fast path — convert ``/Volumes/...`` etc. into a
        # canonical ``dbfs+volume://...`` URL up front so the rest of
        # the pipeline sees a real scheme.
        if isinstance(obj, str):
            obj = _coerce_to_url_str(obj)

        if cls is DatabricksPath:
            return cls.from_url(URL.from_(obj), **kwargs)
        return cls(url=URL.from_(obj), **kwargs)

    # ==================================================================
    # Client binding — DatabricksClient is the single point of access
    # ==================================================================

    @property
    def client(self) -> "DatabricksClient":
        """The bound :class:`DatabricksClient` aggregator.

        Lazily resolves to :meth:`DatabricksClient.current` when none
        was injected at construction — production callers usually let
        this fire so the active workspace selection follows the
        process-wide singleton. Tests inject a :class:`DatabricksClient`
        mock at construction (typically a :class:`MagicMock` with
        ``workspace_client.return_value`` wired to a workspace-shaped
        mock) so this branch is a no-op.
        """
        if self._client is None:
            from yggdrasil.lazy_imports import databricks_client_class
            self._client = databricks_client_class().current()
        return self._client

    def with_client(self, client: "DatabricksClient") -> "DatabricksPath":
        """Replace the bound :class:`DatabricksClient`. Returns *self*."""
        self._client = client
        return self

    @property
    def workspace_client(self) -> Any:
        """Shortcut for ``self.client.workspace_client()`` — the live
        Databricks SDK workspace handle every SDK call routes through."""
        return self.client.workspace_client()

    # ==================================================================
    # Retry policy
    # ==================================================================

    def _call(self, func, *args, **kwargs):
        """Invoke *func(*args, **kwargs)* under the standard retry policy.

        Transient errors (InternalError, throttling, 5xx, connect
        timeouts) get up to 4 retries with 1 / 2 / 4 / 8 s sleeps.
        Permission errors get exactly one retry to absorb
        credential-refresh races. Anything else (NotFound,
        AlreadyExists, deterministic ``BadRequest`` messages like
        "Folder X is protected") propagates immediately.
        """
        if self._retry_sleep is not None:
            return retry_sdk_call(func, *args, sleep=self._retry_sleep, **kwargs)
        return retry_sdk_call(func, *args, **kwargs)

    def _call_ensuring_parents(self, func, *args, **kwargs):
        """Like :meth:`_call`, but auto-creates missing parents on NotFound.

        Used by mutating ops (``upload``, ``mkdirs``, ``create_directory``)
        where the request can fail purely because an intermediate
        directory — or, for Volume paths, the Unity Catalog volume
        itself — does not exist yet. On the first NotFound-shaped
        failure we hand off to :meth:`_ensure_parents` (subclass hook
        for catalog/schema/volume creation, then a recursive parent
        ``mkdir``) and retry exactly once. Other errors propagate.
        """
        try:
            return self._call(func, *args, **kwargs)
        except Exception as exc:
            if not _looks_like_parent_missing(exc):
                raise
            if not self._ensure_parents():
                raise
            return self._call(func, *args, **kwargs)

    def _ensure_parents(self) -> bool:
        """Best-effort create of every directory above *self*.

        Returns ``True`` if any creation actually happened (so the
        caller knows a retry is worth attempting). Subclasses extend
        this — :class:`VolumePath` first creates the catalog / schema
        / managed volume, then recurses into ``parent.mkdir``.
        """
        parent = self.parent
        if parent.url == self.url:
            return False
        try:
            parent._mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    # ==================================================================
    # Sibling paths inherit the same workspace client
    # ==================================================================

    def _from_url(self, url: URL) -> "DatabricksPath":
        return type(self)(
            url=url,
            client=self._client,
            retry_sleep=self._retry_sleep,
        )

    # ==================================================================
    # Holder primitives — defaults that work without a fast path
    # ==================================================================

    def reserve(self, n: int) -> None:
        """No-op — Databricks backends have no capacity layer."""
        if n < 0:
            raise ValueError(f"reserve size must be >= 0, got {n!r}")

    def read_mv(self, n: int, pos: int) -> memoryview:
        """Range read with an aggressive whole-file fast path.

        The base :meth:`Holder.read_mv` runs ``self.size`` (an
        :meth:`_stat` probe) to convert ``n < 0`` into a concrete byte
        count and to bounds-check the requested window. On Databricks
        backends that probe costs a Unity Catalog / Workspace round
        trip every read — wasted for ``read_bytes()`` /
        ``read_arrow_table()`` and other "give me everything" calls,
        because each backend's :meth:`_read_mv` already handles EOF
        natively (chunked-until-short-page on DBFS, full-object
        download on Volumes / Workspace).

        Whole-file shape (``n < 0`` and ``pos == 0``) skips the size
        probe entirely. Partial / positional reads keep the base
        bounds check so out-of-range windows still raise.
        """
        if n < 0 and pos == 0:
            # ``FileNotFoundError`` propagates — semantics match the
            # base ``Holder.read_mv`` which would raise on a stat
            # probe against a missing object. The :meth:`_bread`
            # fallback (used by base ``Path`` methods like
            # :meth:`truncate`) is the only place that swallows it
            # into an empty buffer.
            return self._read_mv(-1, 0)
        return super().read_mv(n, pos)

    def _bread(self, n: int, pos: int, mode):  # pragma: no cover - thin shim
        """Fallback whole-file read into a fresh :class:`BytesIO`.

        Aggressive path: ``n`` is forwarded straight to :meth:`_read_mv`,
        which handles ``n < 0`` as "read to EOF". The previous version
        gated this on a ``_stat()`` probe to compute the size — that's
        one extra round trip per ``read_bytes`` / Arrow open on every
        Databricks surface, and the backends each download the whole
        object anyway. Catching :class:`FileNotFoundError` on the real
        call gives the same "missing → empty buffer" semantics without
        the precondition.
        """
        from yggdrasil.io.bytes_io import BytesIO
        del mode
        if n == 0:
            return BytesIO()
        try:
            data = bytes(self._read_mv(n, pos))
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


# ---------------------------------------------------------------------------
# Error duck-typing — module-private; subclasses keep their own variants
# ---------------------------------------------------------------------------


_PARENT_MISSING_NAMES = frozenset({
    "NotFound", "ResourceDoesNotExist", "FileNotFoundError",
})

_PARENT_MISSING_MESSAGES = (
    "does not exist",
    "no such file or directory",
    "no such directory",
    "parent directory",
    "path does not exist",
)


def _looks_like_parent_missing(exc: BaseException) -> bool:
    """True when *exc* looks like ``parent dir / volume not found``.

    Both ``NotFound``-typed errors and ``BadRequest``/etc. carrying a
    "does not exist" message qualify — the Databricks SDK isn't fully
    consistent about which class it raises for missing-parent cases.
    """
    if type(exc).__name__ in _PARENT_MISSING_NAMES:
        return True
    if isinstance(exc, FileNotFoundError):
        return True
    msg = str(exc).lower()
    return any(pat in msg for pat in _PARENT_MISSING_MESSAGES)
