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
- **Service binding** — the path holds a :class:`DatabricksService`
  (subclass) via ``self.service``; :attr:`client` (inherited from
  :class:`DatabricksResource`) returns ``self.service.client``.
  The SDK workspace handle is reached through
  ``self.client.workspace_client()`` — :class:`DatabricksPath` never
  holds a bare workspace client directly. No imports of
  ``databricks.sdk`` happen at module load.

Tests pass a service-shaped mock (e.g. ``MagicMock(spec=Volumes)``
with ``service.client`` wired to a :class:`DatabricksClient`-shaped
mock that itself exposes ``workspace_client``). There is no
alternate "workspace-only" entry point: every caller routes through
the service.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Optional, Tuple

from yggdrasil.data.enums import Scheme
from yggdrasil.io.path import RemotePath
from yggdrasil.io.path._retry import retry_sdk_call
from yggdrasil.io.url import URL
from .resource import DatabricksResource
from .service import DatabricksService


__all__ = ["DatabricksPath"]


# ---------------------------------------------------------------------------
# Cross-method stash for the URL ``__new__`` normalizes
# ---------------------------------------------------------------------------

# Keyed by ``id(instance)``: lives only between the call to
# :meth:`DatabricksPath.__new__` and the matching :meth:`__init__`
# (Python guarantees the latter runs synchronously on the same thread
# immediately after). ``__init__`` always pops what ``__new__``
# stashed, so the dict stays bounded by the count of currently
# in-flight constructions.
#
# The stash deliberately lives outside ``self.__dict__``: writing the
# slot into the instance dict and then popping it leaves a "dummy"
# entry in CPython's internal hash table that every subsequent
# attribute lookup probes past — measured at +30% on the hot
# ``_stat_cached_fresh`` / ``size`` / ``exists`` reads.
_PENDING_URL_STASH: dict[int, "URL"] = {}


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
    from .fs.dbfs_path import DBFSPath
    from .fs.volume_path import VolumePath
    from .fs.workspace_path import WorkspacePath

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
        # Dispatch by path-segment count under the ``/Volumes/`` namespace
        # so the Unity Catalog hierarchy maps cleanly onto the URL:
        #
        #   ``/Volumes``                      → :class:`VolumePath` (root)
        #   ``/Volumes/cat``                  → :class:`Catalog`
        #   ``/Volumes/cat/sch``              → :class:`Schema`
        #   ``/Volumes/cat/sch/vol``          → :class:`Volume`
        #   ``/Volumes/cat/sch/vol/<rest>``   → :class:`VolumePath`
        #
        # Catalog / Schema / Volume are off-family resources (not
        # :class:`DatabricksPath` subclasses) — :meth:`DatabricksPath.__new__`
        # routes them through their own :meth:`from_url` constructor.
        parts = [p for p in (candidate.path or "/").lstrip("/").split("/") if p]
        depth = len(parts)
        if depth == 1:
            from yggdrasil.databricks.catalog.catalog import Catalog
            return Catalog, candidate
        if depth == 2:
            from yggdrasil.databricks.schema.schema import Schema
            return Schema, candidate
        if depth == 3:
            from yggdrasil.databricks.volume.volume import Volume
            return Volume, candidate
        return VolumePath, candidate
    if scheme == Scheme.DATABRICKS_WORKSPACE.value:
        return WorkspacePath, candidate
    if scheme == Scheme.DATABRICKS_DBFS.value:
        return DBFSPath, candidate
    if scheme == Scheme.DATABRICKS_TABLE.value:
        # :class:`Table` lives in ``yggdrasil.databricks.table.table``
        # and is *not* a :class:`DatabricksPath` subclass — it's a
        # logical Unity Catalog resource on the same ``dbfs+...``
        # family. The dispatcher still surfaces it here so callers
        # that go through ``DatabricksPath(...)`` don't have to know
        # the SQL module exists.
        from yggdrasil.databricks.table.table import Table
        return Table, candidate
    if scheme == Scheme.DATABRICKS_CATALOG.value:
        # ``dbfs+catalog:///cat`` — explicit catalog URL form. Routes
        # to the same :class:`Catalog` resource the volume-path dispatch
        # picks for ``dbfs+volume:///cat``.
        from yggdrasil.databricks.catalog.catalog import Catalog
        return Catalog, candidate
    if scheme == Scheme.DATABRICKS_SCHEMA.value:
        # ``dbfs+schema:///cat/sch`` — explicit schema URL form.
        from yggdrasil.databricks.schema.schema import Schema
        return Schema, candidate

    # Unknown scheme (or empty) — let the caller's intended class
    # decide. DBFSPath is the safe default for the un-qualified
    # ``dbfs://`` family root.
    return DBFSPath, candidate


# ===========================================================================
# DatabricksPath
# ===========================================================================


class DatabricksPath(DatabricksResource, RemotePath):
    """Abstract :class:`RemotePath` for Databricks namespaces.

    Mutualizes the :class:`DatabricksResource` surface — ``self.service``,
    the inherited ``client`` / ``sql`` properties, generic resource
    pickling — with :class:`RemotePath`'s URL-keyed singleton machinery.
    A path is a resource: it has a service, the service has a client,
    the client routes SDK calls. Subclasses (:class:`DBFSPath`,
    :class:`VolumePath`, :class:`WorkspacePath`) get the client/sql
    accessors for free.

    Registers under :attr:`Scheme.DBFS` (the ``dbfs://`` family root)
    and acts as the dispatcher: :meth:`from_url` inspects the URL and
    forwards to the right concrete subclass (DBFS, Volumes,
    Workspace) based on the compound ``dbfs+<surface>://`` scheme,
    or — for the legacy un-prefixed ``dbfs://`` form — on the URL
    path's leading namespace (``/Volumes/...`` →
    :class:`VolumePath`, ``/Workspace/...`` → :class:`WorkspacePath`,
    everything else → :class:`DBFSPath`).

    Singleton identity caching, the 5-minute default TTL, and the
    bounded ``_INSTANCES`` dict all come from :class:`RemotePath` —
    see its docstring for the policy.
    """

    scheme: ClassVar[Scheme] = Scheme.DBFS

    #: Canonical POSIX prefix for the legacy string shape
    #: (``/dbfs/``, ``/Workspace/``, ``/Volumes/``). Empty on the
    #: abstract base; concrete subclasses override.
    NAMESPACE_PREFIX: ClassVar[Optional[str]] = None

    #: :class:`DatabricksService` subclass to use as the default when
    #: a path is constructed without an explicit ``service=`` / ``client=``.
    #: Each concrete subclass declares its typed service
    #: (:class:`DBFSService` / :class:`Volumes` / :class:`Workspaces`)
    #: so ``self.service`` is always the right collection-level handle.
    _SERVICE_CLASS: ClassVar[type] = DatabricksService

    @classmethod
    def _singleton_key(
        cls,
        data: Any = None,
        *,
        url: "URL | None" = None,
        service: Any = None,
        client: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Identity = (subclass, canonical URL string, service-or-client).

        ``data`` collapses into ``url`` before keying so ``"/Volumes/x"``
        and ``URL.from_("/Volumes/x")`` map to the same singleton.
        ``service`` is part of the key because the same URL can be
        backed by different workspaces; passing ``service=None``
        collapses to the lazy-resolved
        :meth:`DatabricksService.current`. Two paths bound to
        distinct services (cross-workspace fixtures) stay distinct
        singletons. ``client`` is accepted as the convenience shortcut
        for ``service=cls._SERVICE_CLASS(client=client)`` and folds into
        the same identity slot so ``DatabricksPath.from_(url, client=A)``
        and ``DatabricksPath.from_(url, client=B)`` don't collide.
        """
        if url is None:
            if isinstance(data, URL):
                url = data
            elif isinstance(data, str):
                url = URL.from_(_coerce_to_url_str(data))
            else:
                # Without a URL there's no canonical identity — fall
                # through to ``id``-based hashing by returning a unique
                # sentinel.
                return (cls, object())
        # ``URL`` is hashable (``hash(self.to_string())``) and compares
        # field-by-field — it works as a dict key directly. Drop the
        # ``str(url)`` round trip that the hot parent-walk loop was
        # paying on every step.
        return (cls, url, service if service is not None else client)

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
        ``WorkspacePath``) the call forwards up the MRO to
        :class:`RemotePath` / :class:`Path` / :class:`Holder` for
        the eventual ``object.__new__`` allocation — normalize the
        seed into a URL kwarg first so a POSIX-string construction
        (``VolumePath("/Volumes/cat/sch/vol/x")``) lands on the
        same canonical URL as the URL-shaped one. The normalized
        URL is then stashed on the instance so :meth:`__init__`
        can skip the second parse.
        """
        if cls is not DatabricksPath:
            # Hot path: explicit ``url=`` kwarg with no ``data=`` to
            # resolve. Sibling construction (``_from_url`` /
            # ``_url_parent`` / ``joinpath``) and the ``_ls`` listing
            # loop all hit this shape, so the parent-walk inner loop
            # skips the dispatcher, the stash bookkeeping, and the
            # ``getattr(instance, "_initialized")`` probe entirely —
            # straight through to ``Holder.__new__`` →
            # ``Singleton.__new__``.
            if url is not None and data is None:
                return super().__new__(cls, data=None, url=url, **kwargs)
            normalized: "URL | None" = None
            if data is not None:
                _, normalized = _resolve_databricks_subclass(data=data)
                if normalized is not None:
                    url = normalized
                    data = None
            instance = super().__new__(cls, data=data, url=url, **kwargs)
            # Stash the normalized URL so the upcoming ``__init__``
            # skips the redundant ``_coerce_to_url_str`` +
            # ``URL.from_`` + ``_strip_dbfs_family_prefix`` chain that
            # would otherwise re-derive the same value from the same
            # POSIX seed. Only stash when we actually computed the
            # normalization here — explicit URL kwargs still need
            # ``__init__`` to run ``_strip_dbfs_family_prefix`` on
            # them. On a singleton cache hit ``_initialized`` is
            # already True; ``__init__`` short-circuits and never
            # reads the slot.
            if normalized is not None and not getattr(instance, "_initialized", False):
                _PENDING_URL_STASH[id(instance)] = normalized
            return instance

        target, normalized = _resolve_databricks_subclass(data=data, url=url)
        # Byte-shaped Path subclasses (DBFSPath / VolumePath /
        # WorkspacePath) accept ``url=`` straight through ``__init__``,
        # so we can hand control back to Python — their auto-fired
        # ``__init__`` re-runs with the caller's kwargs and the
        # normalized URL.
        #
        # Resource-shaped subclasses (Catalog / Schema / Volume / Table)
        # go through ``from_url`` because their constructors take
        # ``(service, catalog_name=…, schema_name=…, …)``, not
        # ``(data, url, …)`` — the URL needs to be parsed into Unity
        # Catalog coordinates before the constructor sees it.
        if isinstance(target, type) and issubclass(target, DatabricksPath):
            from .fs.dbfs_path import DBFSPath
            from .fs.volume_path import VolumePath
            from .fs.workspace_path import WorkspacePath
            if target in (DBFSPath, VolumePath, WorkspacePath):
                if normalized is not None:
                    data = None
                    url = normalized
                instance = target.__new__(target, data=data, url=url, **kwargs)
                # The dispatcher just normalized the seed via
                # ``_resolve_databricks_subclass``; stash it on the
                # target instance so the auto-fired ``__init__``
                # skips the re-parse. ``target.__new__`` itself
                # only re-stashes when it had to do the work
                # locally — here ``url=`` arrived pre-normalized
                # so the inner ``__new__`` doesn't see the
                # POSIX seed.
                if normalized is not None and not getattr(
                    instance, "_initialized", False,
                ):
                    _PENDING_URL_STASH[id(instance)] = normalized
                return instance

        # Resource-shaped or off-family target — construct eagerly via
        # ``from_url`` so the returned object is fully initialized.
        # Python's auto-``__init__`` only fires when the result is an
        # instance of the originating ``cls`` (DatabricksPath); resource
        # targets like ``Table`` aren't, so they need eager init here.
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
        service: Optional[DatabricksService] = None,
        client: Optional["DatabricksClient"] = None,
        temporary: bool = False,
        retry_sleep: Optional[Callable[[float], None]] = None,
        singleton_ttl: Any = ...,
        **kwargs: Any,
    ) -> None:
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return

        if service is None:
            if client is not None:
                # ``client=`` is the user-facing shortcut for "bind this
                # path to the workspace this client speaks to". Wrap
                # the client in the typed collection-level service
                # (:class:`Volumes` / :class:`Workspaces` / :class:`DBFSService`)
                # so ``self.service.client`` returns the caller's client
                # verbatim — and every parent/child the path mints via
                # :meth:`_from_url` inherits the same service.
                service = self._SERVICE_CLASS(client=client)
            else:
                service = self._SERVICE_CLASS.current()

        # ``__new__`` already normalized POSIX-string seeds into a
        # canonical URL and stashed the result on the instance — pick
        # it up and skip the ``_coerce_to_url_str`` + ``URL.from_`` +
        # ``_strip_dbfs_family_prefix`` chain we'd otherwise repeat.
        # The stash is keyed to the seed that produced it, so callers
        # passing an explicit ``url=`` kwarg still go through the
        # parse path below (no stash was made for that case). The
        # stash lives in a process-wide id-keyed dict, not on
        # ``self.__dict__``, so popping it doesn't leave a dummy
        # slot that every later ``_stat_cached_fresh`` / ``size`` /
        # ``exists`` hit has to probe past — CPython retains dummy
        # entries from popped keys until the dict is rebuilt.
        pending = _PENDING_URL_STASH.pop(id(self), None)
        if pending is not None:
            url = pending
            data = None
        else:
            # Pre-coerce ``/dbfs/...`` / ``/Volumes/...`` /
            # ``/Workspace/...`` into the canonical URL form so the URL
            # parser sees a real scheme.
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
                url = _strip_dbfs_family_prefix(url)
                target_scheme = self.scheme
                if target_scheme is not None:
                    target_token = target_scheme.value if isinstance(
                        target_scheme, Scheme,
                    ) else str(target_scheme)
                    if not url.scheme:
                        url = url.with_scheme(target_token)

        super().__init__(
            service=service, data=data, url=url, temporary=temporary,
            **kwargs,
        )

        self._retry_sleep: Optional[Callable[[float], None]] = retry_sleep
        self._initialized = True

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
          :class:`yggdrasil.databricks.table.table.Table` (not a
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
    # Client binding — inherited from DatabricksResource
    # ==================================================================
    #
    # ``self.client`` and ``self.sql`` come from :class:`DatabricksResource`
    # (``self.service.client`` / ``self.client.sql``). Production callers
    # let ``service`` default to :meth:`DatabricksService.current` and
    # the active workspace selection follows the process-wide singleton.
    # Tests inject a service-shaped mock whose ``client`` attribute is
    # the mock :class:`DatabricksClient` so ``self.client`` returns the
    # mock directly.

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

        Used by mutating ops (``upload``, ``mkdirs``, …) where the
        request can fail purely because an intermediate directory —
        or, for :class:`VolumePath`, the Unity Catalog volume itself
        — does not exist yet. On the first NotFound-shaped failure
        we hand off to :meth:`_ensure_parents` (subclass hook for
        catalog / schema / volume creation, then a recursive parent
        ``mkdir``) and retry exactly once. Other errors propagate.
        """
        try:
            return self._call(func, *args, **kwargs)
        except Exception as exc:
            if not _looks_like_parent_missing(exc):
                raise
            if not self._ensure_parents(exc):
                raise
            return self._call(func, *args, **kwargs)

    def _ensure_parents(self, exc: "BaseException | None" = None) -> bool:
        """Default recovery: best-effort ``mkdir`` of the parent directory.

        Subclasses (:class:`VolumePath`) override to handle missing
        catalog / schema / volume cases first.
        """
        parent = self.parent
        if parent is None or parent == self:
            return False
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return False
        return True

    # ==================================================================
    # Sibling paths inherit the same workspace client
    # ==================================================================

    def _from_url(self, url: URL) -> "DatabricksPath":
        return type(self)(
            url=url,
            service=self.service,
            retry_sleep=self._retry_sleep,
        )

    # ==================================================================
    # Pickling — Singleton-style, filtering transient stat-cache slots
    # ==================================================================

    def __getstate__(self) -> dict[str, Any]:
        # Bypass :class:`DatabricksResource`'s non-filtering version so
        # ``_stat_cached`` / ``_stat_cached_at`` (declared in
        # ``Path._TRANSIENT_STATE_ATTRS``) actually stay out of the
        # payload — same convention :class:`Singleton` enforces.
        return {
            k: v for k, v in self.__dict__.items()
            if k not in self._TRANSIENT_STATE_ATTRS
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        if getattr(self, "_initialized", False):
            return
        self.__dict__.update(state)
        for attr in self._TRANSIENT_STATE_ATTRS:
            self.__dict__.setdefault(attr, None)

    # ==================================================================
    # Holder primitives — defaults that work without a fast path
    # ==================================================================

    def reserve(self, n: int) -> None:
        """No-op — Databricks backends have no capacity layer."""
        if n < 0:
            raise ValueError(f"reserve size must be >= 0, got {n!r}")

    def read_mv(
        self,
        size: int = -1,
        offset: int = 0,
        *,
        cursor: bool = False,
    ) -> memoryview:
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

        Buffered state (dirty pages from an in-flight write, or a
        cached buffered tip) routes through the page-cache path on
        :class:`RemotePath` instead — the dirty pages aren't on the
        backend yet, and the fast path would silently miss them.
        """
        if cursor:
            offset = self._pos
        buffered = (
            self._buffersize is not None
            and (self._dirty_pages or self._buffered_size is not None)
        )
        if size < 0 and offset == 0 and not buffered:
            # ``FileNotFoundError`` propagates — semantics match the
            # base ``Holder.read_mv`` which would raise on a stat
            # probe against a missing object. The :meth:`_bread`
            # fallback (used by base ``Path`` methods like
            # :meth:`truncate`) is the only place that swallows it
            # into an empty buffer.
            out = self._read_mv(-1, 0)
            if cursor:
                self._pos = len(out)
            return out
        return super().read_mv(size, offset, cursor=cursor)

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
