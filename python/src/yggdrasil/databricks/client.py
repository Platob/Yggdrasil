import logging
import os
import re
import time
from pathlib import Path
from threading import RLock
from typing import (
    Optional,
    Any,
    Type,
    ClassVar,
    TypeVar,
    TYPE_CHECKING,
    Callable,
    Union,
)

from databricks.sdk import AccountClient as DAC, WorkspaceClient as DWC
from databricks.sdk.client_types import ClientType
from databricks.sdk.config import Config

from yggdrasil.concurrent.threading import Job
from yggdrasil.dataclasses import (
    WaitingConfigArg,
    WaitingConfig,
    ExpiringDict,
    Singleton,
)
from yggdrasil.io.holder import IO
from yggdrasil.enums import MimeTypes, Scheme
from yggdrasil.url import URL, URLBased
from yggdrasil.version import __version__ as ygg_version

if TYPE_CHECKING:
    from .iam import IAM
    from .job.service import Jobs, JobRuns
    from .sql.engine import SQLEngine
    from .table.tables import Tables
    from .column.columns import Columns
    from .catalog.catalogs import Catalogs
    from .schema.schemas import Schemas
    from .volume.volumes import Volumes
    from .warehouse.service import Warehouses
    from .compute.service import Compute
    from .secrets.service import Secrets
    from .workspaces import Workspaces, Workspace
    from .path import DatabricksPath
    from .ai import DatabricksAI
    from .tags.service import EntityTags

__all__ = ["DatabricksClient", "DatabricksService", "DatabricksResource"]

LOGGER = logging.getLogger(__name__)
CURRENT_BASE_CLIENT: Optional["DatabricksClient"] = None
CURRENT_BASE_CLIENT_LOCK: RLock = RLock()

T = TypeVar("T")
TC = TypeVar("TC", bound="DatabricksClient")

_ALLOWED = re.compile(r"^[\d \w\+\-=\.:/@]*$")  # noqa
_SANITIZE = re.compile(r"[^\d \w\+\-=\.:/@]+")  # noqa

# Cache the collapse-repeats regex per ``repl`` so ``safe_tag_value`` doesn't
# pay ``re.escape`` + ``re.compile`` on every dirty input. Real workloads only
# ever pass the default ``"-"``, but keep the mapping general for the rare
# caller that overrides it.
_SAFE_TAG_COLLAPSE_CACHE: dict[str, re.Pattern[str]] = {}


def _safe_tag_collapse(repl: str) -> re.Pattern[str]:
    pattern = _SAFE_TAG_COLLAPSE_CACHE.get(repl)
    if pattern is None:
        pattern = re.compile(re.escape(repl) + r"{2,}")
        _SAFE_TAG_COLLAPSE_CACHE[repl] = pattern
    return pattern


def _is_ygg_dep(dep: Any) -> bool:
    """True when *dep* refers to the ``ygg`` / ``yggdrasil`` project."""
    if isinstance(dep, str):
        head = dep.strip().split("[", 1)[0]
        for op in ("==", ">=", "<=", "!=", "~=", ">", "<"):
            head = head.split(op, 1)[0]
        return head.strip().lower() in ("ygg", "yggdrasil")
    name = getattr(dep, "__module__", None) or getattr(dep, "__name__", "")
    return isinstance(name, str) and name.split(".", 1)[0] in ("ygg", "yggdrasil")


def getenv(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else None


# Init kwarg name → environment variable name. Used by ``__init__`` to
# fill defaults from the environment when the caller passes ``...`` for
# a slot, and by :meth:`DatabricksClient._resolve_init_kwargs` to build
# the canonical kwargs view that drives the singleton key.
_ENV_DEFAULTS: dict[str, str] = {
    "host": "DATABRICKS_HOST",
    "account_id": "DATABRICKS_ACCOUNT_ID",
    "workspace_id": "DATABRICKS_WORKSPACE_ID",
    "token": "DATABRICKS_TOKEN",
    "client_id": "DATABRICKS_CLIENT_ID",
    "client_secret": "DATABRICKS_CLIENT_SECRET",
    "token_audience": "DATABRICKS_TOKEN_AUDIENCE",
    "cluster_id": "DATABRICKS_CLUSTER_ID",
    "serverless_compute_id": "DATABRICKS_SERVERLESS_COMPUTE_ID",
    "azure_workspace_resource_id": "ARM_RESOURCE_ID",
    "azure_client_secret": "ARM_CLIENT_SECRET",
    "azure_client_id": "ARM_CLIENT_ID",
    "azure_tenant_id": "ARM_TENANT_ID",
    "azure_environment": "ARM_ENVIRONMENT",
    "google_credentials": "GOOGLE_CREDENTIALS",
    "google_service_account": "DATABRICKS_GOOGLE_SERVICE_ACCOUNT",
    "profile": "DATABRICKS_CONFIG_PROFILE",
    "config_file": "DATABRICKS_CONFIG_FILE",
}


# Default urllib3 pool sizing applied to every SDK ``Config`` we build.
#
# The SDK's own defaults (`max_connection_pools=20`,
# `max_connections_per_pool=20`, `pool_block=True`) are too tight for
# the typical yggdrasil workload: volume IO (``files.upload`` /
# ``dbfs.read``) and SQL statement-execution traffic share a single
# :class:`requests.Session` inside the workspace client (see
# ``databricks/sdk/_base_client.py``), so a handful of in-flight volume
# transfers pin every connection slot and ``execute_statement`` queues
# behind them. Bumping the per-pool size gives both traffic classes
# enough headroom that statement submission doesn't stall on volume IO.
#
# Callers can override via ``DatabricksClient(max_connection_pools=...,
# max_connections_per_pool=...)`` if they have a different workload
# shape.
_DEFAULT_MAX_CONNECTION_POOLS = 32
_DEFAULT_MAX_CONNECTIONS_PER_POOL = 64


# Static defaults applied when the caller passes ``...`` and the slot
# has no environment fallback.
_STATIC_DEFAULTS: dict[str, Any] = {
    "azure_use_msi": None,
    "auth_type": None,
    "http_timeout_seconds": None,
    "retry_timeout_seconds": None,
    "debug_truncate_bytes": None,
    "debug_headers": None,
    "rate_limit": None,
    "max_connection_pools": _DEFAULT_MAX_CONNECTION_POOLS,
    "max_connections_per_pool": _DEFAULT_MAX_CONNECTIONS_PER_POOL,
    "product": "yggdrasil",
    "product_version": ygg_version,
    "skip_verify": False
}


# Lazy snapshot of resolved env-default values. The env values get baked
# into both the singleton-cache key and the instance's resolved fields,
# so re-reading 17 ``DATABRICKS_*`` / ``ARM_*`` / ``GOOGLE_*`` vars on
# every constructor call dominated singleton-hit cost (~14 us of
# ``os.getenv`` per build). Snapshot once on first use; long-running
# processes that rotate credentials and tests that mutate the env can
# call :func:`invalidate_env_defaults` to drop the cache.
_ENV_DEFAULTS_SNAPSHOT: Optional[dict[str, Any]] = None
_ENV_DEFAULTS_LOCK: RLock = RLock()


def _env_defaults_snapshot() -> dict[str, Any]:
    """Process-lifetime snapshot of the DATABRICKS_* env-default values."""
    global _ENV_DEFAULTS_SNAPSHOT
    cached = _ENV_DEFAULTS_SNAPSHOT
    if cached is not None:
        return cached
    with _ENV_DEFAULTS_LOCK:
        if _ENV_DEFAULTS_SNAPSHOT is None:
            _ENV_DEFAULTS_SNAPSHOT = {
                name: getenv(env_var) for name, env_var in _ENV_DEFAULTS.items()
            }
        return _ENV_DEFAULTS_SNAPSHOT


def invalidate_env_defaults() -> None:
    """Drop the env-default snapshot so the next constructor re-reads.

    Call after rotating ``DATABRICKS_*`` / ``ARM_*`` / ``GOOGLE_*`` env vars
    so a subsequent :class:`DatabricksClient` build picks them up. The
    in-process singleton cache is *not* cleared — entries already keyed off
    the previous snapshot stay live until they're replaced or evicted.
    """
    global _ENV_DEFAULTS_SNAPSHOT
    with _ENV_DEFAULTS_LOCK:
        _ENV_DEFAULTS_SNAPSHOT = None


def _normalize_host(host: Optional[str]) -> Optional[str]:
    """Mirror :meth:`DatabricksClient.__init__`'s host canonicalisation.

    Strips any scheme + path so two callers passing ``"x.com"`` and
    ``"https://x.com/foo"`` collapse onto the same singleton key.
    """
    if not host:
        return host
    normalized = host.split("://")[-1].split("/")[0]
    return f"https://{normalized}"


class DatabricksClient(Singleton, URLBased):
    """
    Thin wrapper around databricks.sdk.config.Config.

    URL-addressable through the :class:`URLBased` base: ``cls.scheme``
    is :attr:`Scheme.DATABRICKS` (``dbks``), so a single
    ``dbks://[client_id[:secret]@]<host>[?profile=...&account_id=...]``
    URL round-trips a client through :meth:`from_url` /
    :meth:`to_url`. The userinfo carries the credential — a bare
    ``:<token>@`` for a PAT, ``<client_id>:<client_secret>@`` for an
    OAuth service principal — and the query string carries every
    other ``DatabricksClient`` field that ``__init__`` accepts.
    Sensitive fields (``host``, ``token``, ``client_id``,
    ``client_secret``) are kept out of the query so they don't get
    duplicated when the URL is logged or persisted.

    Public state is intentionally minimal:
      - config: Config

    Extra wrapper-only metadata is kept separately.

    Cross-process / cross-host serialization is supported via
    :meth:`__getstate__` / :meth:`__setstate__`. SDK clients, Configs,
    and lazy sub-service caches are dropped (all rebuild on demand from
    the dataclass fields). A best-effort *session token snapshot* is
    carried alongside so the receiving side can warm-start without
    re-running the auth dance (browser flow, MSI probe, gcloud, etc).

    If the deserializing host is itself a Databricks runtime (driver
    node), the carried credentials are discarded and ``auth_type`` is
    forced to ``"runtime"`` — DBR's notebook-scoped auth is short-lived,
    identity-correct, and the right default in that environment.
    """

    # URLBased registration: ``dbks://`` URLs route to this class.
    scheme: ClassVar[Scheme] = Scheme.DATABRICKS

    # ---- type hints (init kwargs; defaults applied in ``__init__``) -------

    host: Optional[str]
    account_id: Optional[str]
    workspace_id: Optional[str]
    token: Optional[str]
    client_id: Optional[str]
    client_secret: Optional[str]
    token_audience: Optional[str]
    cluster_id: Optional[str]
    serverless_compute_id: Optional[str]
    azure_workspace_resource_id: Optional[str]
    azure_use_msi: bool | None
    azure_client_secret: Optional[str]
    azure_client_id: Optional[str]
    azure_tenant_id: Optional[str]
    azure_environment: Optional[str]
    google_credentials: Optional[str]
    google_service_account: Optional[str]
    profile: Optional[str]
    config_file: Optional[str]
    auth_type: str | None
    http_timeout_seconds: Optional[int]
    retry_timeout_seconds: Optional[int]
    debug_truncate_bytes: Optional[int]
    debug_headers: bool | None
    rate_limit: Optional[int]
    max_connection_pools: Optional[int]
    max_connections_per_pool: Optional[int]
    product: Optional[str]
    product_version: Optional[str]

    # ---- private singleton cache -----------------------------------------

    # Per-class singleton cache. Two clients constructed with the
    # same canonicalised init kwargs (env defaults applied, host
    # normalised) collapse to one instance — same SDK clients, same
    # lazy service caches, same cached Config. No companion lock —
    # :class:`ExpiringDict.get_or_set` (used by
    # :class:`Singleton.__new__`) is GIL-atomic.
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(default_ttl=None)
    # Cache every constructed client for the process lifetime — SDK
    # handles, lazy service slots, and the cached Config aren't worth
    # rebuilding when the same identity comes back.
    _SINGLETON_TTL: ClassVar[Any] = None

    # ---- init field bookkeeping ------------------------------------------

    # Names of the kwargs ``__init__`` accepts. Materialised once so
    # downstream consumers (Workspace cloning, ``DatabricksService.check_client``,
    # the pickle serializer) don't have to walk the signature.
    _INIT_NAMES: ClassVar[tuple[str, ...]] = tuple(_ENV_DEFAULTS) + tuple(
        _STATIC_DEFAULTS
    )

    # ----- transport policy -------------------------------------------------

    # Attributes that must NOT cross a process / host boundary verbatim.
    # SDK clients hold sockets and threads; cached Configs hold credential
    # providers bound to the *source* host's filesystem (token-cache.json,
    # ~/.databrickscfg, IMDS, gcloud DAC, etc). Sub-service caches hold
    # back-references to ``self`` and re-hydrate lazily via @property.
    _TRANSIENT_STATE: ClassVar[frozenset[str]] = frozenset(
        {
            "_workspace_client",
            "_account_client",
            "_workspace_config",
            "_account_config",
            "_was_connected",
            "_base_url_cached",
            "_workspace",
            "_sql",
            "_entity_tags",
            "_warehouses",
            "_compute",
            "_secrets",
            "_iam",
            "_tables",
            "_columns_svc",
            "_catalogs",
            "_schemas",
            "_volumes",
            "_filesystem",
        }
    )

    # Config attributes worth snapshotting for warm restart on another host.
    # PAT / OAuth secret fields already live on DatabricksClient itself and
    # round-trip via normal state — no need to duplicate them here.
    _SESSION_TOKEN_KEYS: ClassVar[tuple[str, ...]] = (
        "token",
        "token_audience",
        "auth_type",
    )

    @classmethod
    def _resolve_init_kwargs(cls, **kwargs: Any) -> dict[str, Any]:
        """Apply env / static defaults and host normalisation.

        Used by both :meth:`_singleton_key` (to canonicalise the
        identity) and :meth:`__init__` (to populate instance state)
        so the two never disagree about what a given call should
        produce.

        Env defaults come from :func:`_env_defaults_snapshot` — a lazy
        process-lifetime cache — so the per-call cost is one dict
        lookup per name rather than one ``os.getenv`` per name.
        """
        env_defaults = _env_defaults_snapshot()
        resolved: dict[str, Any] = {}
        for name in _ENV_DEFAULTS:
            value = kwargs.get(name, ...)
            resolved[name] = env_defaults[name] if value is ... else value
        for name, default in _STATIC_DEFAULTS.items():
            value = kwargs.get(name, ...)
            resolved[name] = default if value is ... else value

        # ``account_id`` without an explicit ``host`` lands on the
        # central accounts endpoint — mirrors the legacy
        # ``__post_init__`` behaviour.
        if resolved["account_id"] and not resolved["host"]:
            resolved["host"] = "https://accounts.cloud.databricks.com"

        resolved["host"] = _normalize_host(resolved["host"])
        return resolved

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        """Identity key: class + canonicalised init kwargs.

        ``args`` is ignored because :meth:`__init__` is keyword-only;
        any positional input is a programmer error and would key
        differently from the equivalent kwarg call anyway.

        ``resolved`` keys land in the fixed insertion order of
        ``_ENV_DEFAULTS + _STATIC_DEFAULTS``, and every value is
        either a primitive or ``None``, so the items tuple is
        already hashable and stable without an extra ``sorted()``
        per call.
        """
        return (cls, tuple(cls._resolve_init_kwargs(**kwargs).items()))

    def __init__(
        self,
        *,
        host: Any = ...,
        account_id: Any = ...,
        workspace_id: Any = ...,
        token: Any = ...,
        client_id: Any = ...,
        client_secret: Any = ...,
        token_audience: Any = ...,
        cluster_id: Any = ...,
        serverless_compute_id: Any = ...,
        azure_workspace_resource_id: Any = ...,
        azure_use_msi: Any = ...,
        azure_client_secret: Any = ...,
        azure_client_id: Any = ...,
        azure_tenant_id: Any = ...,
        azure_environment: Any = ...,
        google_credentials: Any = ...,
        google_service_account: Any = ...,
        profile: Any = ...,
        config_file: Any = ...,
        auth_type: Any = ...,
        http_timeout_seconds: Any = ...,
        retry_timeout_seconds: Any = ...,
        debug_truncate_bytes: Any = ...,
        debug_headers: Any = ...,
        rate_limit: Any = ...,
        max_connection_pools: Any = ...,
        max_connections_per_pool: Any = ...,
        product: Any = ...,
        product_version: Any = ...,
        skip_verify: Any = ...,
        singleton_ttl: "int | None" = ...,
    ) -> None:
        # Singleton-cached instances are re-entered on every constructor
        # call (Python always invokes ``__init__`` after ``__new__``);
        # skip the second pass so live SDK handles + lazy caches survive.
        # ``singleton_ttl`` is consumed by ``Singleton.__new__``; accept
        # it here so the auto-init pass doesn't trip on an unknown kwarg.
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return

        resolved = self._resolve_init_kwargs(
            host=host,
            account_id=account_id,
            workspace_id=workspace_id,
            token=token,
            client_id=client_id,
            client_secret=client_secret,
            token_audience=token_audience,
            cluster_id=cluster_id,
            serverless_compute_id=serverless_compute_id,
            azure_workspace_resource_id=azure_workspace_resource_id,
            azure_use_msi=azure_use_msi,
            azure_client_secret=azure_client_secret,
            azure_client_id=azure_client_id,
            azure_tenant_id=azure_tenant_id,
            azure_environment=azure_environment,
            google_credentials=google_credentials,
            google_service_account=google_service_account,
            profile=profile,
            config_file=config_file,
            auth_type=auth_type,
            http_timeout_seconds=http_timeout_seconds,
            retry_timeout_seconds=retry_timeout_seconds,
            debug_truncate_bytes=debug_truncate_bytes,
            debug_headers=debug_headers,
            rate_limit=rate_limit,
            max_connection_pools=max_connection_pools,
            max_connections_per_pool=max_connections_per_pool,
            product=product,
            product_version=product_version,
            skip_verify=skip_verify
        )
        for name, value in resolved.items():
            self.__dict__[name] = value

        self._was_connected = False
        self._workspace_config: Optional[Config] = None
        self._workspace_client: Optional[DWC] = None
        self._account_config: Optional[Config] = None
        self._account_client: Optional[DAC] = None
        # Cached parsed ``base_url`` — ``URL.from_str`` is ~6 us per call,
        # and the host doesn't change after ``__init__``. Cleared on
        # transport so the receiving process re-parses against its host.
        self._base_url_cached: Optional[URL] = None
        # Serializable session-token snapshot, populated on
        # ``__getstate__`` and consumed on ``__setstate__``. Off-cluster
        # only — DBR runtime ignores it.
        self._session_token: Optional[dict[str, Any]] = None

        self._initialized = True

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def __getnewargs_ex__(self):
        """Route unpickling through ``__new__`` with the originating kwargs.

        Pickle calls ``cls.__new__(cls, **kwargs)`` first, so the
        singleton machinery can collapse a cross-process restore onto
        the live in-process instance whose key matches.
        """
        kwargs = {name: getattr(self, name, None) for name in self._INIT_NAMES}
        return (), kwargs

    def __getstate__(self):
        """
        Serialize for transport across processes / hosts.

        Drops SDK clients, Configs, and lazy sub-service caches (they all
        rebuild from the init kwargs on demand). Captures a session
        token snapshot from the live Config so the receiving side can warm
        start without re-running an interactive or environment-bound auth.
        """
        state = self.__dict__.copy()

        for key in self._TRANSIENT_STATE:
            state.pop(key, None)
        # ``_singleton_key_`` is rebuilt by ``__new__`` from the
        # ``__getnewargs_ex__`` payload — don't ship it in the
        # pickle.
        state.pop("_singleton_key_", None)

        # Best-effort. We don't *construct* a Config here — pickling must
        # never trigger a browser flow or network round-trip.
        state["_session_token"] = self._snapshot_session_token()

        return state

    def __setstate__(self, state):
        """
        Rehydrate after transport.

        Inside a Databricks runtime (driver node), DBR's notebook-scoped
        credentials win unconditionally: we force ``auth_type="runtime"``
        and clear any sender-bound credential fields that would otherwise
        bias ``_make_base_config`` toward the wrong path.

        Off-cluster, the carried session token is restored onto ``self``
        so the next ``make_config()`` call picks it up via the normal
        field path — no Config exists yet at unpickle time.
        """
        # ``__new__`` may have returned a live singleton already
        # populated by an earlier construction or unpickle in this
        # process — keep its in-flight state (SDK handles, lazy
        # caches, init-time config) untouched.
        if getattr(self, "_initialized", False):
            return

        # Initialize transient slots so attribute access is safe even if
        # the pickle pre-dates a field addition.
        for key in self._TRANSIENT_STATE:
            self.__dict__[key] = None
        self.__dict__["_session_token"] = None

        self.__dict__.update(state)

        if self.is_in_databricks_environment():
            self.auth_type = "runtime"
            self.token = None
            self.client_id = None
            self.client_secret = None
            self.profile = None
            self.config_file = None
            self._session_token = None
            self._initialized = True
            return

        self._restore_session_token(state.get("_session_token"))
        self._initialized = True

    def _snapshot_session_token(self) -> Optional[dict[str, Any]]:
        """
        Capture a minimal, transportable view of the active session token.

        Returns the previously stored snapshot (if any) when no Config has
        been materialized yet — pickling must not trigger Config construction.
        """
        config = self._workspace_config or self._account_config
        if config is None:
            return self._session_token

        snap: dict[str, Any] = {}
        for key in self._SESSION_TOKEN_KEYS:
            value = getattr(config, key, None)
            if value is not None:
                snap[key] = value

        # Some SDK versions cache the resolved Authorization header on the
        # Config; grab it if present, but never call authenticate() — that
        # could hit the network mid-pickle.
        try:
            headers = getattr(config, "_inner", None)
            if isinstance(headers, dict) and "Authorization" in headers:
                snap["authorization"] = headers["Authorization"]
        except Exception:  # noqa: BLE001 - snapshot is best-effort
            pass

        return snap or None

    def _restore_session_token(self, snap: Optional[dict[str, Any]]) -> None:
        """
        Apply a snapshot produced by :meth:`_snapshot_session_token`.

        Only fills fields that are not already set on ``self`` —
        explicit constructor-provided values win over a stale carried token.
        """
        if not snap:
            return

        if not self.token and "token" in snap:
            self.token = snap["token"]
        if not self.token_audience and "token_audience" in snap:
            self.token_audience = snap["token_audience"]
        if not self.auth_type and "auth_type" in snap:
            self.auth_type = snap["auth_type"]

        # Forward the full snapshot so a downstream re-pickle can pass it on
        # even before make_config() runs on this host.
        self._session_token = snap

    # -------------------------------------------------------------------------
    # URLBased — round-trip through ``dbks://`` URLs
    # -------------------------------------------------------------------------

    @property
    def base_url(self):
        cached = self._base_url_cached
        if cached is not None:
            return cached
        if not self.host:
            # Don't cache the make_config() path — config resolution can
            # change ``host`` after the auth dance lands.
            return URL.from_str(self.make_config().host)
        parsed = URL.from_str(self.host)
        object.__setattr__(self, "_base_url_cached", parsed)
        return parsed

    @property
    def explore_url(self) -> URL:
        """Workspace UI root for the Catalog Explorer (``/explore/data``).

        Mirrors :attr:`Catalog.explore_url` / :attr:`Schema.explore_url` so
        the whole resource hierarchy advertises a deep-link in one place.
        """
        return self.base_url.with_path("/explore/data")

    def to_url(self, scheme: str | None = None) -> URL:
        """Render this client as a ``dbks://...`` URL.

        Pack everything ``__init__`` would need to rebuild the client
        into the URL: the workspace host as the URL host, the
        credential (PAT or OAuth client_id/secret) as userinfo, and
        every other non-default field as query items. Sensitive
        fields (``host``, ``token``, ``client_id``,
        ``client_secret``) are intentionally kept *out* of the query
        so they don't get duplicated alongside the userinfo.

        ``scheme`` overrides :attr:`scheme` for callers that want a
        different URL scheme (e.g. ``"https"`` for the bare workspace
        URL); defaults to :attr:`Scheme.DATABRICKS`.
        """
        query: dict[str, Any] = {}
        for key in _TO_URL_QUERY_KEYS:
            value = getattr(self, key)
            if value is not None:
                query[key] = value

        if self.token:
            user, password = None, self.token
        elif self.client_id and self.client_secret:
            user, password = self.client_id, self.client_secret
        else:
            user, password = None, None

        return (
            self.base_url.with_scheme(scheme or type(self).scheme.value)
            .with_query_items(query)
            .with_user_password(user=user, password=password)
        )

    @classmethod
    def from_url(cls: Type[TC], url: "URL | str", **kwargs: Any) -> TC:
        """Build a client from a ``dbks://...`` URL.

        Reads:

        - the workspace host from ``url.host`` (preferred) or a
          ``host=`` query param;
        - credentials from ``url.userinfo`` —
          ``<client_id>:<client_secret>@`` for OAuth, ``:<token>@``
          (or anything-as-password) for a PAT;
        - every other init field of :class:`DatabricksClient` from
          the query string (``profile``, ``auth_type``,
          ``account_id``, ``workspace_id``, ``http_timeout_seconds``,
          …).

        ``kwargs`` overrides anything the URL provides so callers can
        layer programmatic overrides on top of a parsed URL without
        an extra ``replace`` call.
        """
        u = URL.from_(url)

        parsed: dict[str, Any] = {}
        for key, value in u.query_items():
            parsed[key] = value

        host = parsed.pop("host", None) or (f"https://{u.host}/" if u.host else None)
        if not host:
            raise ValueError(
                f"Host is required for {cls.__name__} URL: {u!r}. "
                f"Pass it as the URL host (``dbks://workspace.example/``) "
                f"or as a ``host=`` query item."
            )

        user, password = u.user, u.password
        if user and password:
            parsed["client_id"] = user
            parsed["client_secret"] = password
        elif password:
            parsed["token"] = password

        # Caller-supplied ``kwargs`` win over URL-derived values.
        merged = {"host": host, **parsed, **kwargs}
        return cls(**merged)

    #: Legacy alias for :meth:`from_url`. Kept so existing callers
    #: (notably :class:`DatabricksService.from_parsed_url`) keep
    #: working without an audit pass.
    from_parsed_url = from_url

    @classmethod
    def from_(cls, obj: Any):
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, URL):
            return cls.from_url(obj)
        elif isinstance(obj, str):
            url = URL.from_str(obj, default_scheme="https")
            return cls.from_url(url)
        elif isinstance(obj, dict):
            return cls(**obj)
        else:
            raise ValueError(
                f"Cannot parse {cls.__name__} from object of type {type(obj)}: {obj!r}"
            )

    # -------------------------------------------------------------------------
    # Singleton helpers
    # -------------------------------------------------------------------------

    @classmethod
    def current(cls, reset: bool = False, **overrides: Any) -> "DatabricksClient":
        global CURRENT_BASE_CLIENT

        if reset or CURRENT_BASE_CLIENT is None:
            with CURRENT_BASE_CLIENT_LOCK:
                if reset or CURRENT_BASE_CLIENT is None:
                    client = cls(**overrides)
                    CURRENT_BASE_CLIENT = client

        return CURRENT_BASE_CLIENT

    @classmethod
    def set_current(cls, workspace: Optional["DatabricksClient"]) -> None:
        global CURRENT_BASE_CLIENT
        with CURRENT_BASE_CLIENT_LOCK:
            CURRENT_BASE_CLIENT = workspace

    # -------------------------------------------------------------------------
    # Repr / context manager
    # -------------------------------------------------------------------------

    def __repr__(self):
        return f"{self.__class__.__name__}(host={self.host!r}, auth_type={self.auth_type!r})"

    def __str__(self):
        return self.__repr__()

    def __enter__(self) -> TC:
        object.__setattr__(self, "_was_connected", self.connected)
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._was_connected:
            self.close()

    # -------------------------------------------------------------------------
    # Connection lifecycle
    # -------------------------------------------------------------------------

    @property
    def config(self):
        if self._workspace_config is None or self._account_config is None:
            config = self.make_config()
            ct = config.client_type

            if ct == ClientType.WORKSPACE:
                object.__setattr__(self, "_workspace_config", config)
            elif ct == ClientType.ACCOUNT:
                object.__setattr__(self, "_account_config", config)
            else:
                try:
                    return self.workspace_config
                except Exception:
                    return self.account_config
        return self._workspace_config or self._account_config

    @property
    def workspace_config(self) -> Config:
        if self._workspace_config is None:
            config = self.make_config(client_type=ClientType.WORKSPACE)
            object.__setattr__(self, "_workspace_config", config)
        return self._workspace_config

    @property
    def account_config(self) -> Config:
        if self._account_config is None:
            config = self.make_config(client_type=ClientType.ACCOUNT)
            object.__setattr__(self, "_account_config", config)
        return self._account_config

    @property
    def default_client_type(self) -> ClientType:
        return self.config.client_type

    def _make_base_config(
        self,
        client_type: Optional[ClientType] = None,
    ):
        if client_type == ClientType.ACCOUNT:
            host = "https://accounts.cloud.databricks.com"

            if not self.account_id:
                self.account_id = self.get_account_id()
        else:
            host = self.host

        try:
            config = Config(
                host=host,
                token=self.token,
                client_id=self.client_id,
                client_secret=self.client_secret,
                account_id=self.account_id,
                cluster_id=self.cluster_id,
                serverless_compute_id=self.serverless_compute_id,
                token_audience=self.token_audience,
                azure_workspace_resource_id=self.azure_workspace_resource_id,
                azure_use_msi=self.azure_use_msi,
                azure_client_secret=self.azure_client_secret,
                azure_client_id=self.azure_client_id,
                azure_tenant_id=self.azure_tenant_id,
                azure_environment=self.azure_environment,
                google_credentials=self.google_credentials,
                google_service_account=self.google_service_account,
                profile=self.profile,
                config_file=self.config_file,
                auth_type=self.auth_type,
                http_timeout_seconds=self.http_timeout_seconds,
                retry_timeout_seconds=self.retry_timeout_seconds,
                debug_truncate_bytes=self.debug_truncate_bytes,
                debug_headers=self.debug_headers,
                rate_limit=self.rate_limit,
                max_connection_pools=self.max_connection_pools,
                max_connections_per_pool=self.max_connections_per_pool,
                product=self.product,
                product_version=self.product_version,
                skip_verify=self.skip_verify,
            )
        except Exception as e:
            raise ValueError(self._diagnose_config_error(e, client_type, host)) from e

        return config

    def _diagnose_config_error(
        self,
        error: Exception,
        client_type: Optional[ClientType],
        host: Optional[str],
    ) -> str:
        """Build an actionable error message explaining why Config() failed."""
        target = "account" if client_type == ClientType.ACCOUNT else "workspace"
        lines = [f"Failed to build Databricks {target} config: {error}"]

        if not host:
            lines.append(
                "  - host is missing. Set DATABRICKS_HOST "
                "(e.g. 'https://adb-1234567890123456.7.azuredatabricks.net') "
                "or pass host=... to DatabricksClient(...)."
            )

        if client_type == ClientType.ACCOUNT and not self.account_id:
            lines.append(
                "  - account_id is missing. Set DATABRICKS_ACCOUNT_ID "
                "or pass account_id=... (required for account-level clients)."
            )

        has_pat = bool(self.token)
        has_oauth = bool(self.client_id and self.client_secret)
        has_azure_sp = bool(
            self.azure_client_id and self.azure_client_secret and self.azure_tenant_id
        )
        has_azure_msi = self.azure_use_msi is True
        has_gcp = bool(self.google_credentials or self.google_service_account)
        has_profile = bool(self.profile)
        in_runtime = self.is_in_databricks_environment()

        if not any(
            (
                has_pat,
                has_oauth,
                has_azure_sp,
                has_azure_msi,
                has_gcp,
                has_profile,
                in_runtime,
            )
        ):
            lines.append(
                "  - no credentials found. Provide one of:\n"
                "      * DATABRICKS_TOKEN (PAT)\n"
                "      * DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET (OAuth M2M)\n"
                "      * ARM_CLIENT_ID + ARM_CLIENT_SECRET + ARM_TENANT_ID (Azure SP)\n"
                "      * GOOGLE_CREDENTIALS or DATABRICKS_GOOGLE_SERVICE_ACCOUNT (GCP)\n"
                "      * DATABRICKS_CONFIG_PROFILE pointing to a profile in ~/.databrickscfg"
            )

        if self.client_id and not self.client_secret:
            lines.append(
                "  - client_id is set but client_secret is missing (OAuth requires both)."
            )
        if self.client_secret and not self.client_id:
            lines.append(
                "  - client_secret is set but client_id is missing (OAuth requires both)."
            )

        if self.auth_type:
            lines.append(
                f"  - auth_type explicitly set to {self.auth_type!r} — "
                f"clear it to let the SDK auto-detect."
            )

        if self.profile and self.config_file:
            lines.append(
                f"  - using profile {self.profile!r} from {self.config_file!r}; "
                f"verify the profile exists and has valid fields."
            )
        elif self.profile:
            lines.append(
                f"  - using profile {self.profile!r} from ~/.databrickscfg; "
                f"verify the profile exists."
            )

        return "\n".join(lines)

    def make_config(self, client_type: Optional[ClientType] = None):
        try:
            config = self._make_base_config(client_type=client_type)
        except Exception as first_err:
            if self.auth_type is not None:
                raise

            fallback_auth = (
                "runtime" if self.is_in_databricks_environment() else "external-browser"
            )
            object.__setattr__(self, "auth_type", fallback_auth)

            try:
                config = self._make_base_config(client_type=client_type)
            except Exception as second_err:
                raise ValueError(
                    f"Failed to build Databricks config with auto-detected auth.\n"
                    f"  initial attempt (auth_type=None): {first_err}\n"
                    f"  fallback attempt (auth_type={fallback_auth!r}): {second_err}"
                ) from second_err

        for key in (
            "auth_type",
            "token",
            "client_id",
            "client_secret",
            "account_id",
            "token_audience",
            "azure_workspace_resource_id",
            "azure_use_msi",
            "azure_client_secret",
            "azure_client_id",
            "azure_tenant_id",
            "azure_environment",
            "google_credentials",
            "google_service_account",
            "http_timeout_seconds",
            "retry_timeout_seconds",
            "debug_truncate_bytes",
            "debug_headers",
            "rate_limit",
            "max_connection_pools",
            "max_connections_per_pool",
        ):
            value = getattr(config, key, None)
            if value is not None:
                object.__setattr__(self, key, value)

        return config

    @property
    def local_config_folder(self):
        return Path.home() / ".config" / "databricks-sdk-py"

    @property
    def connected(self) -> bool:
        return self._workspace_client is not None or self._account_client is not None

    def connect(self, *, reset: bool = False) -> "DatabricksClient":
        if reset:
            self.close()

        config = self.config

        if config.client_type == ClientType.ACCOUNT:
            self.account_client()
        else:
            self.workspace_client()

        return self

    def close(self) -> None:
        for key in ("_workspace_client", "_account_client"):
            object.__setattr__(self, key, None)

        object.__setattr__(self, "_was_connected", False)

    def workspace_client(self) -> DWC:
        if self._workspace_client is None:
            object.__setattr__(
                self, "_workspace_client", DWC(config=self.workspace_config)
            )
        return self._workspace_client

    def account_client(self) -> DAC:
        if self._account_client is None:
            object.__setattr__(self, "_account_client", DAC(config=self.account_config))
        return self._account_client

    def files_session(self) -> "HTTPSession":
        """Authenticated :class:`HTTPSession` bound to this workspace host.

        Volume / Files-API traffic routes through yggdrasil's own HTTP
        transport instead of the SDK's ``requests``-based client: the
        :class:`HTTPSession` owns a per-host keep-alive connection pool,
        status-aware tiered retry (429 / 5xx with backoff), and — via the
        :class:`HTTPStream` response body — transparent resume-on-disconnect
        for SSL ``UNEXPECTED_EOF`` / connection-reset mid-download, the
        failure modes the SDK's Files client handles poorly.

        :class:`HTTPSession` is itself a process-wide singleton keyed by
        ``(base_url, verify, …)``, so repeated calls collapse onto one
        shared pool. ``skip_verify`` flows through to ``verify=False``.
        """
        from yggdrasil.http_ import HTTPSession

        return HTTPSession(base_url=self.base_url, verify=not self.skip_verify)

    def files_authorization(self) -> str:
        """Fresh ``Authorization`` header value for Files-API requests.

        Delegates to the SDK Config's auth flow
        (:meth:`databricks.sdk.config.Config.authenticate`) so every
        supported credential type — PAT, OAuth M2M, Azure SP, GCP — and the
        SDK's own token-refresh caching apply unchanged; only the wire
        transport is ours. Raises when the resolved auth produces no bearer
        header (e.g. a misconfigured profile)."""
        header = self.workspace_config.authenticate().get("Authorization")
        if not header:
            raise ValueError(
                "Databricks auth produced no Authorization header for "
                f"{self.host!r}; check credentials / profile."
            )
        return header

    def get_workspace_id(self) -> int:
        if self.workspace_id:
            return self.workspace_id

        self.workspace_id = int(self.workspace_client().get_workspace_id())
        return self.workspace_id

    def get_account_id(self) -> str:
        from yggdrasil.lazy_imports import polars as pl

        if self.account_id:
            return self.account_id

        local_cache = self.local_config_folder / "workspaces_latest.parquet"
        buff = IO(local_cache, copy=False)
        workspace_id = self.get_workspace_id()

        if local_cache.exists() and local_cache.stat().st_size > 0:
            # TODO: fix
            existing_data = buff.as_media(MimeTypes.PARQUET).read_polars_frame()
            filtered = existing_data.filter(pl.col("workspace_id") == workspace_id)

            for record in filtered.to_dicts():
                self.set_account_id(record.get("account_id"))
                return self.account_id
        else:
            existing_data = None

        new_data = self.sql.execute(
            "select distinct account_id, cast(workspace_id as bigint) as workspace_id, workspace_url"
            " from system.access.workspaces_latest"
            " where status = 'RUNNING'"
        ).to_polars()

        if existing_data is not None:
            new_data = pl.concat([existing_data, new_data]).unique()

        buff.truncate()
        buff.seek(0)
        buff.as_media(MimeTypes.PARQUET).write_arrow_table(new_data.to_arrow())
        buff.close()

        filtered = new_data.filter(pl.col("workspace_id") == workspace_id)

        for record in filtered.to_dicts():
            self.set_account_id(record.get("account_id"))
            return self.account_id

        raise ValueError(
            f"Could not find account_id for workspace_id {workspace_id!r} in "
            f"system.access.workspaces_latest. Verify the current identity has "
            f"SELECT on system.access.workspaces_latest, that the workspace is "
            f"in RUNNING status, or set DATABRICKS_ACCOUNT_ID explicitly."
        )

    def set_account_id(self, account_id: str) -> "DatabricksClient":
        if account_id:
            object.__setattr__(self, "account_id", account_id)

            if self._workspace_config is not None:
                object.__setattr__(self._workspace_config, "account_id", account_id)

            if self._account_client is not None:
                self._account_client.config.account_id = account_id

        return self

    @staticmethod
    def is_in_databricks_environment() -> bool:
        return getenv("DATABRICKS_RUNTIME_VERSION") is not None

    def default_tags(self, update: bool = True):
        """Return default resource tags for Databricks assets.

        On create (``update=False``) the tag set is enriched with
        environment-derived owner metadata pulled from
        :class:`~yggdrasil.environ.UserInfo`:

        - ``Product`` / ``ProductVersion`` from the client config.
        - ``Owner`` — UserInfo email when available.
        - ``Hostname`` — local hostname so per-user pools / clusters are
          distinguishable in shared workspaces.
        - ``User`` — ``whoami`` key, useful when no email is reachable.

        Returns:
            A dict of default tags.
        """
        if update:
            return {}
        return {k: self.safe_tag_value(v) for k, v in self._owner_tag_pairs() if v}

    def _owner_tag_pairs(self) -> list[tuple[str, Optional[str]]]:
        """Return the ``(key, value)`` tag pairs sourced from environment + client.

        Split out so subclasses / tests can extend the set without
        rewriting :meth:`default_tags`.
        """
        from yggdrasil.environ import UserInfo

        try:
            info = UserInfo.current()
        except Exception:  # noqa: BLE001 - userinfo is best-effort
            info = None

        pairs: list[tuple[str, Optional[str]]] = [
            ("Product", self.product),
            ("ProductVersion", self.product_version),
        ]
        if info is not None:
            pairs.extend(
                [
                    ("Owner", info.email),
                    ("Hostname", info.hostname or None),
                    ("User", info.key or None),
                ]
            )
        return pairs

    def user_scoped_name(
        self,
        base: str,
        *,
        separator: str = "-",
        max_length: int = 80,
    ) -> str:
        """Return ``base`` suffixed with a stable per-user slug.

        Resolution order for the slug:

        1. ``UserInfo.email`` local part (``alice@example.com`` → ``alice``).
        2. ``UserInfo.key`` (``whoami``).
        3. ``UserInfo.hostname``.

        Each candidate is sanitized with :meth:`safe_tag_value` so the result
        is a legal Databricks resource name. Falls back to ``base`` unchanged
        when no candidate is available — useful in test harnesses where the
        environment carries no identity. The result is truncated to
        ``max_length`` characters (default ``80``, well under the Databricks
        cluster / pool name cap of 100).
        """
        from yggdrasil.environ import UserInfo

        try:
            info = UserInfo.current()
        except Exception:  # noqa: BLE001 - best-effort
            return self.safe_tag_value(base)[:max_length]

        candidates: list[Optional[str]] = []
        email = info.email
        if email and "@" in email:
            candidates.append(email.split("@", 1)[0])
        candidates.extend([info.key, info.hostname])

        slug: Optional[str] = None
        for candidate in candidates:
            cleaned = self.safe_tag_value(candidate) if candidate else ""
            if cleaned and cleaned.lower() not in {"unknown", "none", ""}:
                slug = cleaned
                break

        sanitized_base = self.safe_tag_value(base) or base
        if not slug:
            return sanitized_base[:max_length]

        return f"{sanitized_base}{separator}{slug}"[:max_length]

    @staticmethod
    def safe_tag_value(value: str, *, repl: str = "-") -> str:
        """
        Sanitize a tag string to match: ^[\\d \\w\\+\\-=:.:/@]*$
        Replaces any illegal characters with `repl` and collapses repeats.
        """
        if value is None:
            return ""

        s = str(value).strip()

        # fast path
        if _ALLOWED.fullmatch(s):
            return s

        # replace illegal chars (e.g. '#', '?', '&', '%') with repl
        s = _SANITIZE.sub(repl, s)

        # collapse repeated repl and trim edges
        if repl:
            s = _safe_tag_collapse(repl).sub(repl, s).strip(repl + " ")

        return s

    # ------------------------------------------------------------------ Path

    def dbfs_path(
        self,
        parts: Union[list[str], str],
        temporary: bool = False,
    ):
        """Create a DatabricksPath in this workspace.

        .. deprecated:: 0.8.31
           Use :meth:`open` for byte IO
           (``client.open("/Volumes/cat/sch/vol/x", "rb")``), or build
           the path directly via
           ``DatabricksPath.from_(parts, client=self)`` when you need
           the resource itself.

        Args:
            parts: Path parts or string to parse.
            temporary: Temporary path

        Returns:
            A DatabricksPath instance.
        """
        import warnings
        warnings.warn(
            "DatabricksClient.dbfs_path is deprecated; use "
            "DatabricksClient.open(path, mode) for IO or "
            "DatabricksPath.from_(path, client=...) for the path object.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .path import DatabricksPath

        return DatabricksPath.from_(
            obj=parts,
            client=self,
            temporary=temporary,
        )

    def open(
        self,
        path: Any,
        mode: "Mode | str | None" = None,
        **kwargs: Any,
    ) -> "IO":
        """Open ``path`` against this workspace and return an :class:`IO` cursor.

        Defaults to a :class:`DatabricksPath` bound to this client —
        strings like ``/Volumes/cat/sch/vol/x`` or
        ``dbfs+volume:///cat/sch/vol/x`` dispatch to the right concrete
        subclass (DBFS / Volumes / Workspace). A pre-built
        :class:`~yggdrasil.path.Path` is opened verbatim so callers can
        mix in S3/HTTP/local paths without losing the workspace binding.

        ``mode`` and ``**kwargs`` ride straight through to
        :meth:`Path.open` (which forwards to :meth:`IO.open`).
        """
        from yggdrasil.path import Path
        from .path import DatabricksPath

        target = (
            path
            if isinstance(path, Path)
            else DatabricksPath.from_(obj=path, client=self)
        )
        return target.open(mode=mode, **kwargs)

    @staticmethod
    def _base_tmp_path(
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
    ) -> str:
        if catalog_name and schema_name:
            base_path = "/Volumes/%s/%s/%s" % (
                catalog_name,
                schema_name,
                volume_name or "tmp",
            )
        else:
            base_path = "/Workspace/Shared/.ygg/tmp"

        return base_path

    def tmp_path(
        self,
        suffix: str | None = None,
        extension: str | None = None,
        max_lifetime: float | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
        base_path: str | None = None,
    ) -> "DatabricksPath":
        """
        Shared cache base under Volumes for the current user.

        Args:
            suffix: Optional suffix
            extension: Optional extension suffix to append.
            max_lifetime: Max lifetime of temporary path
            catalog_name: Unity catalog name for volume path
            schema_name: Unity schema name for volume path
            volume_name: Unity volume name for volume path
            base_path: Base temporary path

        Returns:
            A DatabricksPath pointing at the shared cache location.
        """
        start = int(time.time())
        max_lifetime = int(max_lifetime or 48 * 3600)
        end = max(0, int(start + max_lifetime))

        base_path = base_path or self._base_tmp_path(
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name,
        )

        rnd = os.urandom(4).hex()
        temp_path = f"tmp-{start}-{end}-{rnd}"

        if suffix:
            temp_path += suffix

        if extension:
            temp_path += ".%s" % extension

        self.clean_tmp_folder(
            raise_error=False,
            wait=False,
            base_path=base_path,
        )

        from .path import DatabricksPath
        return DatabricksPath.from_(f"{base_path}/{temp_path}", client=self)

    def clean_tmp_folder(
        self,
        raise_error: bool = True,
        wait: WaitingConfigArg = True,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        volume_name: str | None = None,
        base_path: str | None = None,
    ):
        wait = WaitingConfig.from_(wait)

        base_path = base_path or self._base_tmp_path(
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name,
        )

        if is_checked_tmp_path(
            host=self.base_url.to_string(),
            base_path=base_path,
        ):
            return self

        if wait.timeout:
            from .path import DatabricksPath
            base_path = DatabricksPath.from_(base_path, client=self)

            LOGGER.debug("Cleaning temp path %s", base_path)

            for path in base_path.ls(recursive=False, allow_not_found=True):
                if path.name.startswith("tmp"):
                    parts = path.name.split("-")

                    if (
                        len(parts) > 2
                        and parts[0] == "tmp"
                        and parts[1].isdigit()
                        and parts[2].isdigit()
                    ):
                        end = int(parts[2])

                        if end and time.time() > end:
                            path.remove(recursive=True)

            LOGGER.info("Cleaned temp path %s", base_path)
        else:
            (
                Job.make(
                    self.clean_tmp_folder, raise_error=raise_error, base_path=base_path
                ).fire_and_forget()
            )

        return self

    # Sub services with optional caching for default workspace

    @staticmethod
    def lazy_property(
        self,
        *,
        cache_attr: str,
        factory: Callable[[], T],
        use_cache: bool,
    ) -> T:
        """Public helper kept for backwards compatibility.

        New properties on :class:`DatabricksClient` inline the
        lookup directly through ``self.__dict__`` (one dict get,
        one dict set on miss) — the function-call + lambda overhead
        of routing every sub-service through this helper used to
        cost ~1.3 us per access on the hot path. Callers outside
        this class keep the same surface.
        """
        if use_cache:
            d = self.__dict__
            cached = d.get(cache_attr)
            if cached is not None:
                return cached

            created = factory()
            d[cache_attr] = created
            return created

        return factory()

    @property
    def workspaces(self) -> "Workspaces":
        from .workspaces.service import Workspaces

        return Workspaces(client=self)

    @property
    def workspace(self) -> "Workspace":
        cached = self.__dict__.get("_workspace")
        if cached is not None:
            return cached
        from .workspaces import Workspace

        cached = Workspace(**{name: getattr(self, name) for name in self._INIT_NAMES})
        self.__dict__["_workspace"] = cached
        return cached

    @property
    def sql(self) -> "SQLEngine":
        cached = self.__dict__.get("_sql")
        if cached is not None:
            return cached
        from .sql.engine import SQLEngine

        cached = SQLEngine(client=self)
        self.__dict__["_sql"] = cached
        return cached

    def dataset(
        self,
        sql_or_table: str,
        *,
        schema: Any = None,
    ):
        """Return a :class:`~yggdrasil.spark.tabular.Dataset` from SQL or a table name.

        Resolves the Spark session via :meth:`spark` (Databricks
        Connect) and builds a :class:`Dataset` directly — no
        intermediate executor hop::

            dbc = DatabricksClient()
            ds = dbc.dataset("SELECT * FROM main.sales.orders")
            ds = dbc.dataset("main.sales.orders")

        The result is a full :class:`Dataset` — call ``.map``,
        ``.filter``, ``.to_table``, ``.toArrow``, etc. on it.
        """
        from yggdrasil.data.statement import PreparedStatement
        from yggdrasil.spark.tabular import SparkDataset

        session = self.spark()
        if PreparedStatement.looks_like_query(sql_or_table):
            return SparkDataset.from_sql(sql_or_table, spark_session=session, schema=schema)
        return SparkDataset.from_table(sql_or_table, spark_session=session, schema=schema)

    def parallelize(
        self,
        inputs: "Any",
        function: "Callable | None" = None,
        *,
        schema: Any = None,
        byte_size: int = 128 * 1024 * 1024,
    ):
        """Distribute *function* over *inputs* via Spark executors, or
        create a :class:`~yggdrasil.spark.tabular.Dataset` directly
        from *inputs* when no function is given::

            dbc = DatabricksClient()
            # With function
            results = dbc.parallelize(urls, fetch, schema=output_schema)
            # Without function — just wrap inputs as a Dataset
            ds = dbc.parallelize(rows, schema=output_schema)
        """
        from yggdrasil.spark.tabular import SparkDataset

        return SparkDataset.parallelize(
            inputs,
            function,
            schema=schema,
            spark_session=self.spark(),
            byte_size=byte_size,
        )

    def deploy(
        self,
        bundle: "str | Path",
        *,
        target: "str | None" = None,
    ) -> int:
        """Deploy a Databricks Asset Bundle to this workspace.

        Parses the bundle YAML, resolves the target, syncs workspace
        files, and upserts every resource defined under ``resources:``.

        *bundle* is a path to a ``databricks.yml`` file or a directory
        containing one. When a directory is given, the standard bundle
        filenames are probed (``databricks.yml``, ``databricks.yaml``,
        ``bundle.yml``, ``bundle.yaml``).

        Returns an exit code (0 on success).
        """
        from .cli.bundle.deploy import deploy as _deploy

        bundle_path = Path(bundle) if not isinstance(bundle, Path) else bundle

        if bundle_path.is_dir():
            _BUNDLE_FILENAMES = (
                "databricks.yml", "databricks.yaml",
                "bundle.yml", "bundle.yaml",
            )
            for name in _BUNDLE_FILENAMES:
                candidate = bundle_path / name
                if candidate.exists():
                    bundle_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"No bundle file found in {bundle_path}. "
                    f"Expected one of: {', '.join(_BUNDLE_FILENAMES)}. "
                    f"Pass the path to the YAML file explicitly."
                )

        return _deploy(bundle_path, target, client=self)

    @property
    def entity_tags(self) -> "EntityTags":
        cached = self.__dict__.get("_entity_tags")
        if cached is not None:
            return cached
        from .tags.service import EntityTags

        cached = EntityTags(client=self)
        self.__dict__["_entity_tags"] = cached
        return cached

    @property
    def warehouses(self) -> "Warehouses":
        cached = self.__dict__.get("_warehouses")
        if cached is not None:
            return cached
        from yggdrasil.databricks.warehouse.service import Warehouses

        cached = Warehouses(client=self)
        self.__dict__["_warehouses"] = cached
        return cached

    @property
    def compute(self) -> "Compute":
        """Default cluster helper for this client."""
        cached = self.__dict__.get("_compute")
        if cached is not None:
            return cached
        from .compute.service import Compute

        cached = Compute(client=self)
        self.__dict__["_compute"] = cached
        return cached

    @property
    def clusters(self):
        return self.compute.clusters

    @property
    def jobs(self) -> "Jobs":
        cached = self.__dict__.get("_jobs")
        if cached is not None:
            return cached
        from .job.service import Jobs

        cached = Jobs(client=self)
        self.__dict__["_jobs"] = cached
        return cached

    @property
    def job_runs(self) -> "JobRuns":
        cached = self.__dict__.get("_job_runs")
        if cached is not None:
            return cached
        from .job.service import JobRuns

        cached = JobRuns(client=self)
        self.__dict__["_job_runs"] = cached
        return cached

    @property
    def secrets(self) -> "Secrets":
        """Default secrets helper for this client."""
        cached = self.__dict__.get("_secrets")
        if cached is not None:
            return cached
        from .secrets.service import Secrets

        cached = Secrets(client=self)
        self.__dict__["_secrets"] = cached
        return cached

    @property
    def iam(self) -> "IAM":
        cached = self.__dict__.get("_iam")
        if cached is not None:
            return cached
        from .iam import IAM

        cached = IAM(client=self)
        self.__dict__["_iam"] = cached
        return cached

    @property
    def tables(self) -> "Tables":
        """Collection-level Unity Catalog table service for this client."""
        cached = self.__dict__.get("_tables")
        if cached is not None:
            return cached
        from .table.tables import Tables

        cached = Tables(client=self)
        self.__dict__["_tables"] = cached
        return cached

    @property
    def views(self) -> "Tables":
        """Alias for :attr:`tables` — Unity Catalog stores views in the
        same ``tables`` API, and :class:`Table` handles both shapes.
        """
        return self.tables

    @property
    def columns(self) -> "Columns":
        """Collection-level Unity Catalog column service for this client."""
        cached = self.__dict__.get("_columns_svc")
        if cached is not None:
            return cached
        from .column.columns import Columns

        cached = Columns(client=self)
        self.__dict__["_columns_svc"] = cached
        return cached

    @property
    def catalogs(self) -> "Catalogs":
        """Collection-level Unity Catalog hierarchy service for this client.

        Provides dict-like access to catalogs, schemas, and tables::

            client.catalogs["main"]                   # Catalog
            client.catalogs["main"]["sales"]          # Schema
            client.catalogs["main"]["sales"]["orders"]  # Table
        """
        cached = self.__dict__.get("_catalogs")
        if cached is not None:
            return cached
        from .catalog.catalogs import Catalogs

        cached = Catalogs(client=self)
        self.__dict__["_catalogs"] = cached
        return cached

    @property
    def schemas(self) -> "Schemas":
        """Collection-level Unity Catalog schema service for this client.

        Provides dict-like access to schemas and tables::

            client.schemas["main.sales"]             # Schema
            client.schemas["main.sales.orders"]      # Table
            client.schemas(catalog_name="main")      # Schemas scoped to "main"
        """
        cached = self.__dict__.get("_schemas")
        if cached is not None:
            return cached
        from .schema.schemas import Schemas

        cached = Schemas(client=self)
        self.__dict__["_schemas"] = cached
        return cached

    @property
    def volumes(self) -> "Volumes":
        """Collection-level Unity Catalog volume service for this client.

        Provides dict-like access to volumes::

            client.volumes["main.sales.uploads"]                # Volume
            client.volumes(catalog_name="main", schema_name="sales")["uploads"]
            client.volumes.list(catalog_name="main")            # Iterator[Volume]
        """
        cached = self.__dict__.get("_volumes")
        if cached is not None:
            return cached
        from .volume.volumes import Volumes

        cached = Volumes(client=self)
        self.__dict__["_volumes"] = cached
        return cached

    @property
    def ai(self) -> "DatabricksAI":
        """Databricks AI umbrella service (vector search today, serving/registry next).

        Reach the concrete services through it::

            client.ai.vector_search.endpoint("rag").ensure_created()
            client.ai.vector_search.index("main.rag.docs").query(
                query_text="…", columns=["id", "text"],
            )
        """
        cached = self.__dict__.get("_ai")
        if cached is not None:
            return cached
        from .ai import DatabricksAI

        cached = DatabricksAI(client=self)
        self.__dict__["_ai"] = cached
        return cached

    @property
    def is_serverless_compute(self) -> bool:
        """True when this client explicitly targets serverless compute.

        Only returns ``True`` when ``serverless_compute_id`` was set
        by the caller. A bare client with no ``cluster_id`` and no
        ``serverless_compute_id`` is NOT serverless — it simply has
        no compute target and will resolve one lazily when needed.
        """
        return bool(self.serverless_compute_id)

    def spark(
        self,
        *dependencies: "Any",
        registry: "Optional[Any]" = None,
        check_public: bool = False,
        cache_dir: "Optional[Union[str, os.PathLike]]" = None,
        cluster: "str | Cluster | None" = None
    ):
        """Open a Databricks Connect session with auto-resolved deps.

        Returns a live :class:`pyspark.sql.SparkSession` (Spark
        Connect variant) configured against this client's workspace
        host and credentials. The bound :class:`DatabricksClient`
        is stashed on the session as ``session.ygg_client`` so
        downstream helpers (UDFs, :class:`Dataset` extensions,
        ad-hoc resource lookups) can reach the same auth without an
        extra ``DatabricksClient.current()`` call.

        Each *dependency* is classified once via
        :func:`classify_dependency`:

        - Public PyPI specs (``"ygg[data,databricks]==0.7.85"``,
          ``"numpy>=1.0"``, …) ride straight to the cluster via
          :meth:`DatabricksEnv.withDependencies`. ``ygg`` is
          always declared via :meth:`default_ygg_spec` — pinned
          to the driver's installed version with the
          ``[data, databricks]`` extras so the cluster sees the
          exact same runtime + ``pandas`` / ``numpy`` /
          ``databricks-sdk`` surface the driver is using.
          Override by passing an explicit ``ygg`` spec
          (e.g. ``client.spark("ygg==0.7.0")`` or
          ``client.spark("ygg")`` for an editable-mode rebuild).
        - Editable installs (``pip install -e .``) get their
          local working copy built into a wheel whose version
          carries the local hostname
          (``0.7.85+host.<host>``). The wheel lives at
          ``/Workspace/Users/<me>/.ygg/pypi/simple/<pkg>/<wheel>``
          (overridable via *registry*) and is re-uploaded on every
          load so the registry slot tracks the developer's
          working code.
        - Private / non-PyPI installs get the same wheel-build +
          workspace-upload treatment, but lazily — the upload
          is skipped when the workspace slot already exists, so
          a team sharing one registry path only re-uploads on
          version bumps.

        Both editable and private wheels are then handed to
        :meth:`DatabricksEnv.withDependencies` via the
        ``local:<path>`` prefix Databricks Connect understands;
        the wheel itself is read back from the workspace into a
        local cache so the spec is reachable from the driver
        process.

        Serverless compute (the default — no ``cluster_id``) wires
        deps through ``DatabricksEnv`` + ``withEnvironment``;
        classic compute falls back to
        :meth:`SparkSession.addArtifacts` with ``pyfile=True``
        since classic clusters don't honour the declarative
        environment.

        Arguments:

        - *dependencies* — variadic. Each entry is anything
          :func:`classify_dependency` accepts (PyPI spec string,
          bare module name, :class:`os.PathLike`, or any object
          with ``__module__``). ``client.spark("polars",
          "my_internal", Path("/some/pkg"))`` is the headline
          shape; ``ygg[data,databricks]`` is appended
          automatically unless the caller already provides their
          own ``ygg`` spec.
        - *registry* — a :class:`WorkspacePyPIRegistry` (or any
          shape its constructor accepts) to use as the shared
          wheel cache. Defaults to a registry rooted at
          ``/Workspace/Users/<me>/.ygg/pypi/simple`` so a
          single-user setup needs no configuration.
        - *check_public* — when ``True``, an HTTPS probe to
          ``pypi.org`` decides whether an installed dist counts
          as public. Off by default so an offline registry stays
          fast; turn on when shipping mixed public / private
          deps.
        - *cache_dir* — local scratch dir used by the classic
          compute fallback (and for wheel materialization when
          no explicit *registry* is passed).

        When a :class:`pyspark.sql.SparkSession` is already active
        in the process (notebook driver, an outer
        ``client.spark()`` call, a Databricks Job task), that
        session is returned as-is — dependency classification and
        wheel publishing are skipped, since the active session's
        environment is already fixed. The client is still stashed
        on it as ``session.ygg_client`` so downstream helpers find
        the same auth.
        """
        try:
            from pyspark.sql import SparkSession  # noqa

            active = SparkSession.getActiveSession()
        except Exception:
            active = None
        if active is not None:
            LOGGER.debug(
                "Reusing active Spark Connect session for %r",
                self,
            )
            return self._bind_spark_session(active)

        from databricks.connect import DatabricksSession  # noqa

        deps = list(dependencies)
        if not any(_is_ygg_dep(d) for d in deps):
            deps.insert(0, f"ygg[data,databricks]=={ygg_version}")

        mode = "serverless" if self.is_serverless_compute else "classic"
        LOGGER.debug(
            "Resolving Spark Connect dependencies for %r (mode=%s, deps=%r)",
            self,
            mode,
            deps,
        )
        registry_obj = self._resolve_registry(registry, cache_dir=cache_dir)
        specs, _remotes = registry_obj.publish_many(deps, check_public=check_public)

        cluster = self.cluster_id if cluster is None else cluster

        if cluster is not None and cluster != "serverless":
            cluster = self.clusters.get_or_create(cluster)
            self.cluster_id = cluster.cluster_id

        LOGGER.debug(
            "Creating Spark Connect session %r (mode=%s, cluster_id=%r, "
            "serverless_compute_id=%r, install_specs=%d)",
            self,
            mode,
            self.cluster_id,
            self.serverless_compute_id,
            len(specs),
        )
        builder = DatabricksSession.builder.sdkConfig(self.workspace_config)

        if self.is_serverless_compute:
            env = self._build_databricks_env(install_specs=specs)
            if env is not None:
                builder = builder.withEnvironment(env)
            session = builder.getOrCreate()
        else:
            # Classic compute — the cluster won't read the
            # environment, so attach the local wheels as
            # ``addArtifacts(pyfile=True)`` instead.
            session = builder.getOrCreate()
            local_paths = [
                spec[len("local:") :] for spec in specs if spec.startswith("local:")
            ]
            if local_paths:
                LOGGER.debug(
                    "Attaching wheel artifacts to Spark Connect session %r "
                    "(count=%d, paths=%r)",
                    self,
                    len(local_paths),
                    local_paths,
                )
                session.addArtifacts(*local_paths, pyfile=True)

        LOGGER.info(
            "Created Spark Connect session %r (mode=%s, install_specs=%d)",
            self,
            mode,
            len(specs),
        )
        return self._bind_spark_session(session)

    def _bind_spark_session(self, session: "Any") -> "Any":
        """Stash ``self`` on *session* as ``ygg_client`` and return it.

        Downstream ``client.spark(...)`` consumers (UDFs,
        :class:`Dataset` extensions, ad-hoc resource lookups)
        pull the bound client off the session instead of
        re-resolving :meth:`DatabricksClient.current`.
        """
        try:
            session.ygg_client = self  # type: ignore[attr-defined]
        except Exception:
            pass

        from yggdrasil.environ import PyEnv

        PyEnv.set_spark_session(session)

        return session

    def _resolve_registry(
        self,
        registry: "Optional[Any]",
        *,
        cache_dir: "Optional[Union[str, os.PathLike]]",
    ):
        """Coerce *registry* into a :class:`WorkspacePyPIRegistry`.

        ``None`` → fresh registry at the default workspace path
        with this client bound. Any other shape (string,
        :class:`WorkspacePath`) is forwarded to the
        ``WorkspacePyPIRegistry.__init__`` keyword ``base_path``.
        An already-constructed registry is rebound to this
        client (no-op if it was already pointing here) and
        returned as-is so callers can preconfigure the local
        cache or workspace root once and reuse it.
        """
        from yggdrasil.databricks.registry import WorkspacePyPIRegistry

        if isinstance(registry, WorkspacePyPIRegistry):
            registry.client = self
            return registry

        local_cache = Path(os.fspath(cache_dir)) if cache_dir is not None else None
        return WorkspacePyPIRegistry(
            client=self,
            base_path=registry,
            local_cache=local_cache,
        )

    def _build_databricks_env(
        self,
        *,
        install_specs: "list[str]",
    ):
        """Build a :class:`DatabricksEnv` for the serverless path.

        *install_specs* is the merged list produced by
        :meth:`WorkspacePyPIRegistry.publish_many` — a mix of
        ``"name==version"`` (public PyPI) and ``"local:<path>"``
        entries. Returns ``None`` when nothing to install so the
        builder can skip ``withEnvironment`` entirely.
        """
        if not install_specs:
            return None

        from databricks.connect import DatabricksEnv

        env = DatabricksEnv()
        env = env.withDependencies(list(install_specs))
        return env


DATABRICKS_CLIENT_INIT_NAMES = frozenset(DatabricksClient._INIT_NAMES)
# Fields that ``to_url`` emits as query items: every init field except the
# credentials / host that already ride in the URL host + userinfo. Computed
# once at import so the hot path doesn't walk the init-name tuple per call.
_TO_URL_QUERY_KEYS: tuple[str, ...] = tuple(
    name
    for name in DatabricksClient._INIT_NAMES
    if name not in ("host", "token", "client_id", "client_secret")
)
CHECKED_TMP_WORKSPACES: ExpiringDict[str, set[str]] = ExpiringDict()


def is_checked_tmp_path(host: str, base_path: str):
    existing = CHECKED_TMP_WORKSPACES.get(host)

    if existing is None:
        # Seed with the path itself — ``set(base_path)`` would iterate the
        # string into a set of characters, defeating the cache and forcing
        # every other call to re-walk the Volumes listing.
        CHECKED_TMP_WORKSPACES[host] = {base_path}
        return False

    if base_path in existing:
        return True

    existing.add(base_path)
    return False


# ``DatabricksService`` and ``DatabricksResource`` now live in
# sibling modules. Re-export here for backward-compatible
# ``from yggdrasil.databricks.client import DatabricksService /
# DatabricksResource`` imports.
from .service import DatabricksService  # noqa: E402
from .resource import DatabricksResource  # noqa: E402
