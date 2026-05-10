"""AWS client / service / resource trio.

Mirrors the Databricks pattern: one client owns the session and
mints boto3 service clients on demand; service objects bind to a
client; resource objects bind to a service. The split lets a
single set of credentials cover an entire tree of objects without
each one re-resolving auth.

Class summary
-------------

- :class:`AWSClient` — the analog of :class:`DatabricksClient`.
  Wraps an :class:`AWSConfig`, owns a lazily-built boto3
  :class:`Session`, exposes per-service client factories
  (``s3_client``, ``sts_client``), and per-service
  *service objects* (``self.s3`` returns :class:`S3Service`).
  Has a ``current()`` singleton + URL round-trip.

- :class:`AWSService` — abstract base for service objects. Holds an
  :class:`AWSClient`, defers shared concerns (session, config,
  region) to it. Subclasses (:class:`S3Service`, future
  :class:`DynamoService`, …) layer their own client + behavior on
  top.

- :class:`AWSResource` — abstract base for individual entities
  (an S3 object, a DynamoDB row). Holds a service; reaches the
  client via ``self.service.client``.

Singleton vs explicit
---------------------

``AWSClient.current()`` returns a process-global default. Service
defaults flow from there: ``S3Service.current()`` builds against
``AWSClient.current()`` automatically. Pass an explicit client to
any service / path constructor to escape the singleton (different
account, different role, etc.).
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import uuid
from abc import ABC
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from yggdrasil.io.url import URL
from yggdrasil.lazy_imports import boto3_module, botocore_module

from .config import AwsCredentials, AWSConfig

if TYPE_CHECKING:
    import boto3  # noqa: F401
    from botocore.client import BaseClient  # type: ignore[import-untyped]

    from .fs.service import S3Service


__all__ = [
    "AWSClient",
    "AWSService",
    "AWSResource",
]

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")
TC = TypeVar("TC", bound="AWSClient")
TS = TypeVar("TS", bound="AWSService")


# ---------------------------------------------------------------------------
# Process-global current client
# ---------------------------------------------------------------------------

_CURRENT_CLIENT: Optional["AWSClient"] = None
_CURRENT_CLIENT_LOCK: threading.RLock = threading.RLock()


# ===========================================================================
# AWSClient
# ===========================================================================


class AWSClient:
    """Top-level AWS client.

    Owns:

    - a :class:`AWSConfig` (the static config — what credentials
      and where);
    - a lazily-built :class:`boto3.Session` (refreshable when
      ``config.role_arn`` is set);
    - a per-service boto-client cache (one client per ``(service,
      overrides)`` tuple);
    - a per-service service-object cache (``self.s3`` returns one
      :class:`S3Service` per :class:`AWSClient`).

    Construction:

        >>> AWSClient()                      # default chain
        >>> AWSClient(AWSConfig(profile="prod"))
        >>> AWSClient.from_credentials(creds, region="us-east-1")

    URL round-trip:

        >>> client.to_url()                    # aws://...
        >>> AWSClient.from_parsed_url(url)

    Identity & singleton caching
    ----------------------------

    Instances are cached per ``(class, config)`` in :attr:`_INSTANCES`
    so two callers that build a client with the same config share
    one boto :class:`Session`, one connection pool, and one boto
    :class:`BaseClient` cache. ``__init__`` is idempotent — Python
    always invokes it after :meth:`__new__` returns the cached
    instance, so the second pass skips reinitialization. Pickling
    routes through :meth:`__getnewargs__` + :meth:`__setstate__` so
    a client unpickled in the same process collapses to the live
    singleton instead of cloning its session and pool.
    """

    # Per-(cls, config) singleton cache. Two AWSClients built with the
    # same config share the boto session, connection pool, and the
    # per-service boto-client cache. Subclasses inherit this slot.
    _INSTANCES: ClassVar[dict[Tuple[type, "AWSConfig"], "AWSClient"]] = {}
    _INSTANCES_LOCK: ClassVar[threading.Lock] = threading.Lock()

    # Instance attributes that don't survive pickling — excluded by the
    # generic :meth:`__getstate__` and rebuilt by :meth:`__setstate__`.
    # ``_session`` and ``_client_cache`` carry live boto handles; the
    # rest are cheap rebuilds whose lazy state would just be wrong on
    # the receiver side.
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({
        "_session", "_client_cache", "_s3", "_account_id", "_was_connected",
    })

    def __new__(
        cls: Type[TC],
        config: Optional[AWSConfig] = None,
    ) -> TC:
        if config is None:
            config = AWSConfig()
        key = (cls, config)
        with cls._INSTANCES_LOCK:
            cached = cls._INSTANCES.get(key)
            if cached is not None:
                return cached  # type: ignore[return-value]
            instance = super().__new__(cls)
            cls._INSTANCES[key] = instance
            return instance  # type: ignore[return-value]

    def __init__(self, config: Optional[AWSConfig] = None) -> None:
        # Singleton-cached instances are re-entered on every constructor
        # call (Python always invokes __init__ after __new__); skip the
        # second pass so the live session / boto-client cache survive.
        if getattr(self, "_initialized", False):
            return
        self.config = config if config is not None else AWSConfig()
        self._session: Any = None
        self._client_cache: dict = {}
        self._s3: Optional["S3Service"] = None
        self._account_id: Optional[str] = None
        self._was_connected: bool = False
        self._initialized = True

    # ------------------------------------------------------------------
    # Pickling — drop cached handles, route unpickle through __new__
    # ------------------------------------------------------------------

    def __getnewargs__(self):
        # Route unpickling through __new__ so a client reconstructed in
        # the same process with the same config collapses to the live
        # singleton instead of cloning the boto session / pool.
        return (self.config,)

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in self._TRANSIENT_STATE_ATTRS
        }

    def __setstate__(self, state):
        # __new__ may have returned a live singleton — leave its session,
        # boto-client cache, and lazy service handles untouched.
        if getattr(self, "_initialized", False):
            return
        self.__dict__.update(state)
        self._session = None
        self._client_cache = {}
        self._s3 = None
        self._account_id = None
        self._was_connected = False
        self._initialized = True

    # ==================================================================
    # URL contract — mirrors DatabricksClient.url_scheme / to_url
    # ==================================================================

    @classmethod
    def url_scheme(cls) -> str:
        return "aws"

    def to_url(self, scheme: Optional[str] = None) -> URL:
        """Render this client as a URL.

        Format: ``aws://[creds@]region/?profile=...&role_arn=...``

        - Region goes in the host slot (a region is the closest AWS
          analog to a "host" — it parameterizes every endpoint).
        - Static creds go in user:password (when both set).
        - Everything else goes in the query string.

        Sensitive fields (``secret_access_key``, ``session_token``)
        are emitted but the rendered URL is intended for
        config-as-URL plumbing, not logging — :class:`AWSConfig` has
        ``repr=False`` on those fields for log safety.
        """
        query: dict[str, Any] = {}
        for f in dataclasses.fields(self.config):
            if not f.init:
                continue
            if f.name in (
                "access_key_id",
                "secret_access_key",
                "session_token",
                "region",
            ):
                continue
            value = getattr(self.config, f.name)
            if value is not None:
                query[f.name] = value

        # Region in host slot.
        host = self.config.region or ""

        url = URL.from_str(f"{scheme or self.url_scheme()}://{host}/")
        url = url.with_query_items(query)

        # Static creds → userinfo. Skip when empty.
        if self.config.has_static_credentials():
            url = url.with_user_password(
                user=self.config.access_key_id,
                password=self.config.secret_access_key,
            )

        return url

    @classmethod
    def parse(cls: Type[TC], obj: Any) -> TC:
        """Coerce *obj* (str / URL / dict / AWSClient / AWSConfig) to a client."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, AWSConfig):
            return cls(config=obj)
        if isinstance(obj, URL):
            return cls.from_parsed_url(obj)
        if isinstance(obj, str):
            return cls.from_parsed_url(URL.from_str(obj))
        if isinstance(obj, dict):
            return cls(config=AWSConfig(**obj))
        raise ValueError(
            f"Cannot parse {cls.__name__} from {type(obj).__name__}: {obj!r}"
        )

    @classmethod
    def from_parsed_url(cls: Type[TC], url: URL) -> TC:
        """Parse an ``aws://`` URL back into an :class:`AWSClient`."""
        kwargs: dict[str, Any] = {}
        for key, value in url.query_items():
            kwargs[key] = value

        if url.host:
            kwargs.setdefault("region", url.host)

        if url.user:
            kwargs["access_key_id"] = url.user
        if url.password:
            kwargs["secret_access_key"] = url.password

        return cls(config=AWSConfig(**kwargs))

    @classmethod
    def from_credentials(
        cls: Type[TC],
        creds: AwsCredentials,
        *,
        region: Optional[str] = None,
        **kwargs: Any,
    ) -> TC:
        """Construct from a static :class:`AwsCredentials`."""
        return cls(config=AWSConfig.from_credentials(creds, region=region, **kwargs))

    # ==================================================================
    # Singleton
    # ==================================================================

    @classmethod
    def current(cls: Type[TC], *, reset: bool = False, **overrides: Any) -> TC:
        """Process-global default :class:`AWSClient`.

        ``reset=True`` rebuilds; ``overrides`` are passed to a fresh
        :class:`AWSConfig`. Mirrors :meth:`DatabricksClient.current`.

        Note: per-config singleton caching in :meth:`__new__` means a
        rebuilt default landing on the same config still reuses the
        live boto :class:`Session` / connection pool from the previous
        ``current()``. Pass ``reset=True`` *and* drop the cached
        instance via :meth:`AWSClient._INSTANCES.pop` if you need a
        truly fresh boto handle.
        """
        global _CURRENT_CLIENT

        if reset or _CURRENT_CLIENT is None:
            with _CURRENT_CLIENT_LOCK:
                if reset or _CURRENT_CLIENT is None:
                    config = (
                        AWSConfig(**overrides) if overrides else AWSConfig()
                    )
                    _CURRENT_CLIENT = cls(config=config)

        return _CURRENT_CLIENT  # type: ignore[return-value]

    @classmethod
    def set_current(cls, client: Optional["AWSClient"]) -> None:
        """Replace the process-global current client. Pass ``None`` to clear."""
        global _CURRENT_CLIENT
        with _CURRENT_CLIENT_LOCK:
            _CURRENT_CLIENT = client

    # ==================================================================
    # Repr / context manager
    # ==================================================================

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"region={self.config.region!r}, "
            f"profile={self.config.profile!r}, "
            f"role_arn={self.config.role_arn!r})"
        )

    def __enter__(self) -> "AWSClient":
        object.__setattr__(self, "_was_connected", self.connected)
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._was_connected:
            self.close()

    # ==================================================================
    # Connection lifecycle
    # ==================================================================

    @property
    def connected(self) -> bool:
        return self._session is not None

    def connect(self, *, reset: bool = False) -> "AWSClient":
        """Eagerly build the boto session. Idempotent."""
        if reset:
            self.close()
        # Touching the property triggers the lazy build.
        _ = self.session
        return self

    def close(self) -> None:
        """Drop the cached session, all per-service clients, all service objects.

        Boto3 sessions don't have a real ``close()`` — the underlying
        HTTP connections live in connection pools that GC will reap.
        Our ``close`` is "let go of references"; subsequent calls
        rebuild on demand.
        """
        self._session = None
        self._client_cache = {}
        self._s3 = None
        self._account_id = None
        self._was_connected = False

    # ==================================================================
    # Session / boto-client factory
    # ==================================================================

    @property
    def session(self) -> "boto3.Session":
        """Lazily-built boto3 :class:`Session`. Cached.

        - ``config.has_assume_role()`` → :class:`RefreshableCredentials`
          driven by STS AssumeRole.
        - Otherwise → static / profile / default-chain creds.
        """
        if self._session is None:
            self._session = self._build_session()
        return self._session

    def client(self, service: str, **overrides: Any) -> "BaseClient":
        """Get a boto3 client for *service*. Cached per (service, overrides).

        ``overrides`` are forwarded to :meth:`Session.client` — used
        for one-off endpoint or signature overrides without rebuilding
        the whole config.
        """
        cache_key = (
            service,
            tuple(sorted(overrides.items())) if overrides else (),
        )
        cached = self._client_cache.get(cache_key)
        if cached is not None:
            return cached

        kwargs: dict[str, Any] = {}
        if self.config.region:
            kwargs["region_name"] = self.config.region
        if self.config.endpoint_url:
            kwargs["endpoint_url"] = self.config.endpoint_url
        if service == "s3" and self.config.s3_addressing_style:
            botocore = botocore_module()
            kwargs["config"] = botocore.config.Config(
                s3={"addressing_style": self.config.s3_addressing_style}
            )
        kwargs.update(overrides)

        boto_client = self.session.client(service, **kwargs)
        self._client_cache[cache_key] = boto_client
        return boto_client

    # Convenience accessors for the boto clients we use most.

    def s3_client(self, **overrides: Any) -> "BaseClient":
        return self.client("s3", **overrides)

    def sts_client(self, **overrides: Any) -> "BaseClient":
        return self.client("sts", **overrides)

    # ==================================================================
    # Lazy-cached sub-service factories
    # ==================================================================

    @property
    def s3(self) -> "S3Service":
        """The :class:`S3Service` bound to this client. Lazy + cached.

        :class:`AWSService` subclasses are themselves singleton-cached
        per ``(cls, client)`` — the local ``_s3`` slot just dodges the
        ``_INSTANCES`` lookup on the hot path.
        """
        if self._s3 is None:
            from .fs.service import S3Service
            self._s3 = S3Service(client=self)
        return self._s3

    # ==================================================================
    # Identity helper — useful for default tags / debugging
    # ==================================================================

    def caller_identity(self) -> dict[str, Any]:
        """Wrap STS GetCallerIdentity. Returns dict with Account / Arn / UserId.

        Network call; not cached. Use sparingly.
        """
        return self.sts_client().get_caller_identity()

    @property
    def account_id(self) -> str:
        """Resolve the account ID via STS. Cached on the instance.

        Not part of the configured field set on purpose: it's
        derivable from credentials, and stamping it eagerly would
        force a network call at construction. The cached value is
        excluded from pickling — the receiver re-resolves on demand.
        """
        if self._account_id is None:
            identity = self.caller_identity()
            self._account_id = identity["Account"]
        return self._account_id

    @property
    def region(self) -> Optional[str]:
        """Effective region: explicit config first, then session default."""
        if self.config.region:
            return self.config.region
        try:
            return self.session.region_name
        except Exception:
            return None

    # ==================================================================
    # Session construction internals
    # ==================================================================

    def _build_session(self) -> "boto3.Session":
        boto3 = boto3_module()
        if self.config.has_refresher():
            return self._build_refresher_session(boto3)
        if self.config.has_assume_role():
            return self._build_assume_role_session(boto3)
        return self._build_simple_session(boto3)

    def _build_refresher_session(self, boto3) -> "boto3.Session":
        """Build a Session whose credentials auto-refresh via
        :attr:`AWSConfig.refresher`.

        This is the path taken when credentials are vended by an
        external service (Databricks
        ``temporary_path_credentials`` / ``temporary_table_credentials``,
        an STS broker, a custom credential service). The refresher
        callback is invoked once for the seed metadata and then again
        on every botocore refresh cycle (~5 min before token expiry),
        exactly the same wiring as
        :meth:`_build_assume_role_session` but with a caller-supplied
        callback rather than a built-in STS AssumeRole driver.
        """
        botocore = botocore_module()
        refresher = self.config.refresher
        assert refresher is not None  # gated by has_refresher() above

        from .config import _refresher_to_metadata

        def refresh():
            return _refresher_to_metadata(refresher)

        refreshable = (
            botocore.credentials.RefreshableCredentials
            .create_from_metadata(
                metadata=refresh(),
                refresh_using=refresh,
                method="ygg-refresher",
            )
        )

        botocore_session = botocore.session.get_session()
        botocore_session._credentials = refreshable
        if self.config.region:
            botocore_session.set_config_variable("region", self.config.region)

        return boto3.Session(botocore_session=botocore_session)

    def _build_simple_session(self, boto3) -> "boto3.Session":
        """Static / profile / default-chain session."""
        kwargs: dict[str, Any] = {}
        if self.config.profile:
            kwargs["profile_name"] = self.config.profile
        if self.config.access_key_id:
            kwargs["aws_access_key_id"] = self.config.access_key_id
        if self.config.secret_access_key:
            kwargs["aws_secret_access_key"] = self.config.secret_access_key
        if self.config.session_token:
            kwargs["aws_session_token"] = self.config.session_token
        if self.config.region:
            kwargs["region_name"] = self.config.region
        return boto3.Session(**kwargs)

    def _build_assume_role_session(self, boto3) -> "boto3.Session":
        """Build a Session whose credentials auto-refresh via STS AssumeRole.

        Botocore calls our refresh callback ~5 min before token
        expiry. The base session — using whatever creds the user
        gave us (static or profile or default-chain) — drives the
        STS AssumeRole calls.
        """
        botocore = botocore_module()
        base_session = self._build_simple_session(boto3)

        role_session_name = (
            self.config.role_session_name
            or f"yggdrasil-{uuid.uuid4().hex[:12]}"
        )

        def refresh():
            sts = base_session.client("sts")
            assume_kwargs: dict[str, Any] = {
                "RoleArn": self.config.role_arn,
                "RoleSessionName": role_session_name,
                "DurationSeconds": int(self.config.duration_seconds),
            }
            if self.config.external_id:
                assume_kwargs["ExternalId"] = self.config.external_id
            response = sts.assume_role(**assume_kwargs)
            creds = response["Credentials"]
            expiration = creds["Expiration"]
            return {
                "access_key": creds["AccessKeyId"],
                "secret_key": creds["SecretAccessKey"],
                "token": creds["SessionToken"],
                "expiry_time": (
                    expiration.isoformat()
                    if hasattr(expiration, "isoformat")
                    else str(expiration)
                ),
            }

        refreshable = (
            botocore.credentials.RefreshableCredentials
            .create_from_metadata(
                metadata=refresh(),
                refresh_using=refresh,
                method="sts-assume-role",
            )
        )

        botocore_session = botocore.session.get_session()
        # Direct hook: botocore exposes _credentials as the slot
        # boto3.Session reads on construction. Documented but
        # private; if the internals shift, this is the line to
        # adapt.
        botocore_session._credentials = refreshable
        if self.config.region:
            botocore_session.set_config_variable("region", self.config.region)

        return boto3.Session(botocore_session=botocore_session)

    # ==================================================================
    # Dunder
    # ==================================================================

    def __hash__(self) -> int:
        # Hash on config — two clients with the same config behave
        # identically, even if their cached sessions differ.
        return hash((type(self), self.config))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AWSClient):
            return NotImplemented
        return type(self) is type(other) and self.config == other.config


# ===========================================================================
# AWSService
# ===========================================================================


class AWSService(ABC):
    """Abstract base for AWS service objects.

    A service object is a thin wrapper that binds an
    :class:`AWSClient` to a particular AWS service (S3, DynamoDB,
    SQS, …). Subclasses expose the boto client for their service
    plus any service-specific helpers.

    Mirrors :class:`DatabricksService`: holds a ``client``,
    delegates shared concerns (region, account_id, session)
    upstream, supports a ``current()`` singleton, and round-trips
    via URL.

    Subclass contract
    -----------------

    Subclasses must define :attr:`service_name` (e.g. ``"s3"``).
    The default :meth:`client` resolves to
    ``self.client.client(self.service_name())``. Subclass-specific
    state (caches, lazy handles) goes in :meth:`__init__` after the
    ``super().__init__(client=client)`` call and is gated by the
    inherited ``_initialized`` guard so the singleton's lazy state
    survives idempotent re-entry from ``cls(client=...)``.

    Identity & singleton caching
    ----------------------------

    Instances are cached per ``(class, client)`` in :attr:`_INSTANCES`,
    so ``S3Service(client=c)`` always returns the same handle for a
    given client. Pickling routes through :meth:`__getnewargs__` so
    a service unpickled in the same process collapses to the live
    singleton. Subclasses should add their non-picklable handles to
    :attr:`_TRANSIENT_STATE_ATTRS` to keep the generic getstate clean.
    """

    # Per-(cls, client) singleton cache, mirrors AWSClient._INSTANCES.
    # Subclasses inherit this slot — there's one shared dict across
    # every AWSService subclass so the (cls, client) tuple disambiguates
    # S3Service, DynamoService, etc. against the same underlying client.
    _INSTANCES: ClassVar[dict[Tuple[type, "AWSClient"], "AWSService"]] = {}
    _INSTANCES_LOCK: ClassVar[threading.Lock] = threading.Lock()

    # Generic getstate excludes these attrs from the pickle payload.
    # Subclasses extend by overriding the frozenset (use union with the
    # base set to avoid losing the inherited transients).
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset()

    _current: ClassVar[Optional["AWSService"]] = None

    def __new__(
        cls: Type[TS],
        client: Optional[AWSClient] = None,
    ) -> TS:
        if client is None:
            client = AWSClient.current()
        key = (cls, client)
        with cls._INSTANCES_LOCK:
            cached = cls._INSTANCES.get(key)
            if cached is not None:
                return cached  # type: ignore[return-value]
            instance = super().__new__(cls)
            cls._INSTANCES[key] = instance
            return instance  # type: ignore[return-value]

    def __init__(self, client: Optional[AWSClient] = None) -> None:
        # Singleton-cached instances are re-entered on every constructor
        # call (Python always invokes __init__ after __new__); skip the
        # second pass so subclass-side caches survive.
        if getattr(self, "_initialized", False):
            return
        self.client: AWSClient = client if client is not None else AWSClient.current()
        self._initialized = True

    # ==================================================================
    # Subclass identity
    # ==================================================================

    @classmethod
    def service_name(cls) -> str:
        """The AWS service name (e.g. ``"s3"``, ``"dynamodb"``).

        Default: lowercase the class name with ``Service`` stripped.
        Subclasses override when the convention doesn't match.
        """
        name = cls.__name__
        if name.endswith("Service"):
            name = name[:-len("Service")]
        return name.lower()

    @classmethod
    def url_scheme(cls) -> str:
        return f"aws+{cls.service_name()}"

    # ==================================================================
    # Boto client passthrough
    # ==================================================================

    @property
    def boto_client(self) -> "BaseClient":
        """The boto3 client for this service. Cached on the AWSClient."""
        return self.client.client(self.service_name())

    # ==================================================================
    # Singleton
    # ==================================================================

    @classmethod
    def current(cls: Type[TS], *, reset: bool = False) -> TS:
        """Process-global default service against
        :meth:`AWSClient.current`."""
        if reset or cls._current is None:
            cls._current = cls(client=AWSClient.current())
        return cls._current  # type: ignore[return-value]

    @classmethod
    def set_current(cls, service: Optional["AWSService"]) -> None:
        cls._current = service

    # ==================================================================
    # URL round-trip
    # ==================================================================

    def to_url(self, scheme: Optional[str] = None) -> URL:
        return (
            self.client
            .to_url(scheme=scheme or self.url_scheme())
            .with_path(f"/{self.service_name()}")
        )

    @classmethod
    def from_parsed_url(cls: Type[TS], url: URL) -> TS:
        return cls(client=AWSClient.from_parsed_url(url))

    # ==================================================================
    # Context manager — defer to the client
    # ==================================================================

    def __enter__(self) -> "AWSService":
        self.client.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.client.__exit__(exc_type, exc_val, exc_tb)

    def connect(self) -> "AWSService":
        self.client.connect()
        return self

    def close(self) -> None:
        # Service close is a no-op by default; the client owns the
        # session, and individual services don't carry separate
        # state. Subclasses with caches override.
        return

    # ==================================================================
    # Shared-config passthrough
    # ==================================================================

    @property
    def config(self) -> AWSConfig:
        return self.client.config

    @property
    def region(self) -> Optional[str]:
        return self.client.region

    @property
    def account_id(self) -> str:
        return self.client.account_id

    # ==================================================================
    # Pickling — generic state, route unpickle through __new__
    # ==================================================================

    def __getnewargs__(self):
        # Route unpickling through __new__ so a service reconstructed in
        # the same process with the same client collapses to the live
        # singleton instead of cloning subclass-side caches.
        return (self.client,)

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in self._TRANSIENT_STATE_ATTRS
        }

    def __setstate__(self, state):
        # __new__ may have returned a live singleton — leave its caches
        # untouched.
        if getattr(self, "_initialized", False):
            return
        self.__dict__.update(state)
        self._initialized = True


# ===========================================================================
# AWSResource
# ===========================================================================


class AWSResource(ABC):
    """Abstract base for AWS-backed entities.

    A resource binds to a *service*, not directly to a client. A
    resource (an S3 object, a DynamoDB row) reaches the boto client
    through ``self.service.boto_client``. Reaches the AWS client
    through ``self.client`` shorthand.

    Subclasses define ``__init__`` accepting a ``service=`` kwarg
    (defaulting to the appropriate service's ``current()``) and
    whatever fields identify the resource (bucket+key for S3,
    table+pk for DynamoDB, …).
    """

    service: AWSService

    def __init__(self, service: Optional[AWSService] = None, *args, **kwargs) -> None:
        if service is None:
            # Resolve the default through whatever AWSService subclass
            # the resource is keyed against. Subclasses are expected
            # to override this constructor with their own default;
            # this base default falls through to the bare AWSService
            # current — useful for resources that aren't tied to one
            # service in particular (rare).
            service = AWSService.current()
        self.service = service
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Pickling — resource subclasses typically chain to super()
    # ------------------------------------------------------------------

    def __getstate__(self):
        # Subclasses with their own state should override and merge
        # this dict with their own.
        return {"service": self.service}

    def __setstate__(self, state):
        self.service = state["service"]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def client(self) -> AWSClient:
        return self.service.client

    @property
    def boto_client(self) -> "BaseClient":
        return self.service.boto_client