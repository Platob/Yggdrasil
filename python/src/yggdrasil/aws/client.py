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


@dataclasses.dataclass
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
    """

    config: AWSConfig = dataclasses.field(default_factory=AWSConfig)

    # --- Cached lazy state, all init=False / repr=False / not-compared ----

    _session: Any = dataclasses.field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _client_cache: dict = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False,
    )
    _service_cache: dict = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False,
    )
    _was_connected: bool = dataclasses.field(
        default=False, init=False, repr=False, compare=False, hash=False,
    )

    # ------------------------------------------------------------------
    # Pickling — drop cached objects
    # ------------------------------------------------------------------

    def __getstate__(self):
        return {"config": self.config}

    def __setstate__(self, state):
        self.config = state["config"]
        self._session = None
        self._client_cache = {}
        self._service_cache = {}
        self._was_connected = False

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
        self._service_cache = {}
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

    @staticmethod
    def _lazy_property(
        self: "AWSClient",
        *,
        cache_attr: str,
        factory,
    ):
        """Internal: cached lazy-property pattern. Mirrors Databricks'
        ``lazy_property``."""
        cached = self._service_cache.get(cache_attr)
        if cached is not None:
            return cached
        created = factory()
        self._service_cache[cache_attr] = created
        return created

    @property
    def s3(self) -> "S3Service":
        """The :class:`S3Service` bound to this client. Lazy + cached."""
        from .fs.service import S3Service

        return self._lazy_property(
            self,
            cache_attr="s3",
            factory=lambda: S3Service(client=self),
        )

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

        Not part of the dataclass field set on purpose: it's
        derivable from credentials, and stamping it eagerly would
        force a network call at construction.
        """
        cached = self._service_cache.get("_account_id")
        if cached is not None:
            return cached
        identity = self.caller_identity()
        account_id = identity["Account"]
        self._service_cache["_account_id"] = account_id
        return account_id

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


@dataclasses.dataclass
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
    ``self.client.client(self.service_name())``.
    """

    client: AWSClient = dataclasses.field(
        default_factory=AWSClient.current,
        repr=False,
    )

    _current: ClassVar[Optional["AWSService"]] = None

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
    # Pickling — drop nothing; service is just a (client) wrapper
    # ==================================================================

    def __getstate__(self):
        return {"client": self.client}

    def __setstate__(self, state):
        self.client = state["client"]


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