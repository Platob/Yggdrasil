"""AWS client / service / resource trio.

Mirrors the Databricks pattern: one client owns the configuration
**and** the session, mints boto3 service clients on demand;
service objects bind to a client; resource objects bind to a
service. The split lets a single set of credentials cover an
entire tree of objects without each one re-resolving auth.

Class summary
-------------

- :class:`AWSClient` — the analog of :class:`DatabricksClient`.
  Holds every configuration knob directly (no separate
  ``AWSConfig`` class — :class:`AWSClient` *is* the config).
  Owns a lazily-built boto3 :class:`Session`, exposes per-service
  client factories (``s3_client``, ``sts_client``), and per-service
  *service objects* (``self.s3`` returns :class:`S3Service`).
  Has a ``current()`` singleton + URL round-trip.

- :class:`AWSService` — abstract base for service objects. Holds an
  :class:`AWSClient`, defers shared concerns (session, region) to
  it. Subclasses (:class:`S3Service`, future
  :class:`DynamoService`, …) layer their own client + behavior on
  top.

- :class:`AWSResource` — abstract base for individual entities
  (an S3 object, a DynamoDB row). Holds a service; reaches the
  client via ``self.service.client``.

Singleton & runtime defaults
----------------------------

:class:`AWSClient` is a :class:`Singleton` keyed on every
identity-bearing init kwarg, so two callers building a client with
the same auth share one boto :class:`Session`. Bare ``AWSClient()``
scrapes the managed-runtime context (AWS Lambda / Batch / ECS /
EKS env vars: ``AWS_ACCESS_KEY_ID``, ``AWS_REGION``, ``AWS_PROFILE``,
``AWS_ROLE_ARN``, ``AWS_ROLE_SESSION_NAME``, ``AWS_ENDPOINT_URL``,
``AWS_S3_ADDRESSING_STYLE``); anything still unset falls through to
boto3's own credential chain at session-build time.

``AWSClient.current()`` returns a process-global default. Service
defaults flow from there: ``S3Service.current()`` builds against
``AWSClient.current()`` automatically. Pass explicit kwargs to any
service / path constructor to escape the singleton (different
account, different role, etc.).
"""

from __future__ import annotations

import logging
import os
import re
import socket
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

from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.url import URL
from yggdrasil.url.explore import ExploreUrlRepr
from yggdrasil.lazy_imports import boto3_module, botocore_module

from .config import (
    AwsCredentials,
    CredentialsRefresher,
    DATABRICKS_SQL_CREDENTIAL_COLUMNS,
    DatabricksSQLCredentialsRefresher,
    _coerce_refresher_output,
    _refresher_to_metadata,
)

if TYPE_CHECKING:
    import boto3  # noqa: F401
    from botocore.client import BaseClient  # type: ignore[import-untyped]

    from .fs.service import S3Service
    from yggdrasil.databricks.client import DatabricksClient


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


# ---------------------------------------------------------------------------
# Env / runtime helpers
# ---------------------------------------------------------------------------


def _env(name: str) -> Optional[str]:
    """Return env var if set and non-empty, else ``None``."""
    value = os.environ.get(name)
    return value if value else None


# STS session names are constrained to ``[\w+=,.@-]{2,64}`` per AWS;
# strip anything else (Lambda function names allow ``:`` for versions,
# hostnames can carry dots which are already legal, etc.) so the
# synthesized name doesn't get rejected at AssumeRole time.
_SESSION_NAME_INVALID_RE = re.compile(r"[^\w+=,.@-]+")


def _runtime_session_name() -> str:
    """Build a CloudTrail-friendly STS session name from the active runtime.

    Falls through the most informative env vars before landing on
    ``socket.gethostname()``. ``ygg-`` prefix keeps the synthesized
    name visibly distinct from user-supplied ones in audit logs.
    """
    function_name = _env("AWS_LAMBDA_FUNCTION_NAME")
    if function_name:
        raw = f"ygg-lambda-{function_name}"
    else:
        batch_job = _env("AWS_BATCH_JOB_ID")
        if batch_job:
            raw = f"ygg-batch-{batch_job}"
        else:
            try:
                host = socket.gethostname() or "unknown"
            except Exception:
                host = "unknown"
            raw = f"ygg-{host}"
    cleaned = _SESSION_NAME_INVALID_RE.sub("-", raw).strip("-")
    # STS caps session names at 64 chars; truncate from the tail so
    # the ``ygg-<source>`` prefix stays readable.
    return cleaned[:64] or "ygg-session"


# ===========================================================================
# AWSClient
# ===========================================================================


class AWSClient(Singleton):
    """Merged AWS configuration + session + per-service client factory.

    Holds every knob needed to mint a boto3 :class:`Session` directly
    on the instance — there is no separate ``AWSConfig`` class.
    Equality and hashing follow :meth:`_singleton_key`, which excludes
    :attr:`refresher` (callables aren't comparable) and lazy / transient
    session state. Use :attr:`refresher_key` as the discriminator when
    distinct refreshers must mint distinct clients.

    Construction shapes:

    - **Static credentials**: pass ``access_key_id`` /
      ``secret_access_key`` / optional ``session_token``.
    - **Profile**: pass ``profile`` (matches ``AWS_PROFILE``); the
      session resolves through ``~/.aws/credentials``.
    - **Assume-role**: pass ``role_arn``, optionally with
      ``role_session_name`` / ``external_id`` / ``duration_seconds``.
      A refreshable credential provider drives STS AssumeRole on
      demand.
    - **SSO (boto3-native)**: pass ``sso_start_url`` / ``sso_region`` /
      ``sso_account_id`` / ``sso_role_name`` for IAM Identity Center
      with external-browser device-code auth — boto3's
      ``SSOTokenProvider`` handles the device-code dance and token
      cache (typically primed by ``aws sso login``).
    - **Default chain**: pass nothing. Runtime env vars
      (``AWS_ACCESS_KEY_ID`` / ``AWS_REGION`` / ``AWS_PROFILE`` /
      ``AWS_ROLE_ARN`` / ``AWS_ROLE_SESSION_NAME`` / ``AWS_ENDPOINT_URL``
      / ``AWS_S3_ADDRESSING_STYLE``) are auto-detected so Lambda /
      Batch / ECS / EKS land with reasonable defaults; anything still
      empty falls through to boto3's own chain at session-build time.
    """

    # ------------------------------------------------------------------
    # Singleton wiring
    # ------------------------------------------------------------------

    _SINGLETON_TTL: ClassVar[Any] = None

    # Identity-bearing init kwargs in canonical order; matches the
    # iteration order of :meth:`_singleton_key`'s tuple. ``refresher``
    # is intentionally absent — callables aren't hashable in a stable
    # way; carry a ``refresher_key`` instead when two refresher-backed
    # configs must mint distinct clients.
    _IDENTITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "access_key_id", "secret_access_key", "session_token",
        "region", "profile",
        "role_arn", "role_session_name", "external_id", "duration_seconds",
        "endpoint_url", "s3_addressing_style",
        "sso_start_url", "sso_region", "sso_account_id", "sso_role_name",
        "refresher_key",
    )

    # Live boto handles and lazy caches — excluded from pickling.
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({
        "_session", "_client_cache", "_s3", "_account_id", "_was_connected",
    })

    # Snapshot of the default Databricks SQL credential-column aliases,
    # kept here so callers can reach it as ``AWSClient.DATABRICKS_SQL_CREDENTIAL_COLUMNS``.
    DATABRICKS_SQL_CREDENTIAL_COLUMNS = DATABRICKS_SQL_CREDENTIAL_COLUMNS

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_init_kwargs(cls, **kwargs: Any) -> dict[str, Any]:
        """Canonicalise init kwargs against runtime env so
        :meth:`_singleton_key` and :meth:`__init__` agree on identity.

        ``...`` for a field means "caller didn't pass this — fall
        through to the env var(s)". Explicit ``None`` keeps ``None``.
        Mirrors :meth:`DatabricksClient._resolve_init_kwargs`.
        """
        def pull(name: str, env: tuple[str, ...]) -> Optional[str]:
            value = kwargs.get(name, ...)
            if value is not ...:
                return value
            for env_name in env:
                env_value = _env(env_name)
                if env_value is not None:
                    return env_value
            return None

        resolved: dict[str, Any] = {
            "access_key_id":     pull("access_key_id",     ("AWS_ACCESS_KEY_ID",)),
            "secret_access_key": pull("secret_access_key", ("AWS_SECRET_ACCESS_KEY",)),
            "session_token":     pull("session_token",     ("AWS_SESSION_TOKEN",)),
            "region":            pull("region",            ("AWS_REGION", "AWS_DEFAULT_REGION")),
            "profile":           pull("profile",           ("AWS_PROFILE", "AWS_DEFAULT_PROFILE")),
            "role_arn":          pull("role_arn",          ("AWS_ROLE_ARN",)),
            "role_session_name": pull("role_session_name", ("AWS_ROLE_SESSION_NAME",)),
            "external_id":       kwargs.get("external_id", None),
            "duration_seconds":  kwargs.get("duration_seconds", 3600),
            "endpoint_url":      pull("endpoint_url",      ("AWS_ENDPOINT_URL",)),
            "s3_addressing_style": pull("s3_addressing_style", ("AWS_S3_ADDRESSING_STYLE",)),
            # SSO fields — picked up from the same boto3-native env
            # convention so a Lambda / EKS task with these set
            # auto-routes through IAM Identity Center.
            "sso_start_url":     pull("sso_start_url",     ("AWS_SSO_START_URL",)),
            "sso_region":        pull("sso_region",        ("AWS_SSO_REGION",)),
            "sso_account_id":    pull("sso_account_id",    ("AWS_SSO_ACCOUNT_ID",)),
            "sso_role_name":     pull("sso_role_name",     ("AWS_SSO_ROLE_NAME",)),
            "refresher_key":     kwargs.get("refresher_key", None),
        }

        # When an assume-role is in play but neither the caller nor
        # the env named the STS session, synthesize one from the
        # active runtime so CloudTrail records the calling workload.
        if resolved["role_arn"] and not resolved["role_session_name"]:
            resolved["role_session_name"] = _runtime_session_name()

        return resolved

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        # ``__init__`` is keyword-only; positional args would key
        # differently from the equivalent kwarg call. ``refresher``
        # is excluded — callables aren't comparable.
        del args
        kwargs.pop("refresher", None)
        kwargs.pop("singleton_ttl", None)
        resolved = cls._resolve_init_kwargs(**kwargs)
        return (cls, tuple(resolved[name] for name in cls._IDENTITY_FIELDS))

    def __init__(
        self,
        *,
        access_key_id: Any = ...,
        secret_access_key: Any = ...,
        session_token: Any = ...,
        region: Any = ...,
        profile: Any = ...,
        role_arn: Any = ...,
        role_session_name: Any = ...,
        external_id: Optional[str] = None,
        duration_seconds: int = 3600,
        endpoint_url: Any = ...,
        s3_addressing_style: Any = ...,
        sso_start_url: Any = ...,
        sso_region: Any = ...,
        sso_account_id: Any = ...,
        sso_role_name: Any = ...,
        refresher_key: Optional[str] = None,
        refresher: Optional[CredentialsRefresher] = None,
        singleton_ttl: Any = ...,
    ) -> None:
        # ``Singleton.__new__`` may return a cached instance, in which
        # case ``__init__`` runs a second time — guard so we don't
        # clobber the live session, lazy boto-client cache, or the
        # refresher the original caller bound.
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return

        resolved = self._resolve_init_kwargs(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            region=region,
            profile=profile,
            role_arn=role_arn,
            role_session_name=role_session_name,
            external_id=external_id,
            duration_seconds=duration_seconds,
            endpoint_url=endpoint_url,
            s3_addressing_style=s3_addressing_style,
            sso_start_url=sso_start_url,
            sso_region=sso_region,
            sso_account_id=sso_account_id,
            sso_role_name=sso_role_name,
            refresher_key=refresher_key,
        )
        for name, value in resolved.items():
            setattr(self, name, value)
        self.refresher = refresher

        # Lazy / transient state.
        self._session: Any = None
        self._client_cache: dict = {}
        self._s3: Optional["S3Service"] = None
        self._account_id: Optional[str] = None
        self._was_connected: bool = False

        self._initialized = True

    def __getnewargs_ex__(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Route unpickle through ``__new__`` with the identity kwargs.

        ``refresher`` rides along so the receiver can invoke it; it's
        just not part of the singleton key (callables aren't
        comparable).
        """
        kwargs = {name: getattr(self, name) for name in self._IDENTITY_FIELDS}
        kwargs["refresher"] = self.refresher
        return (), kwargs

    def __setstate__(self, state: dict[str, Any]) -> None:
        # ``Singleton.__setstate__`` would seed transient slots to
        # ``None``; we instead want a fully reset session / client
        # cache so the receiver builds its own boto handles. Mirror
        # the dunder but use the proper sentinels for our slots.
        if getattr(self, "_initialized", False):
            return
        self.__dict__.update(state)
        self._session = None
        self._client_cache = {}
        self._s3 = None
        self._account_id = None
        self._was_connected = False
        self._initialized = True

    # ------------------------------------------------------------------
    # Backwards-compat: ``client.config`` once pointed at a separate
    # AWSConfig instance. Merged class is its own config; alias so
    # the existing callers (``AWSService.config``, debugging code,
    # tests) keep working.
    # ------------------------------------------------------------------

    @property
    def config(self) -> "AWSClient":
        return self

    # ==================================================================
    # Inspection helpers
    # ==================================================================

    def has_assume_role(self) -> bool:
        return bool(self.role_arn)

    def has_static_credentials(self) -> bool:
        return bool(self.access_key_id and self.secret_access_key)

    def has_refresher(self) -> bool:
        """True iff a :attr:`refresher` callback is wired up.

        Drives :meth:`_build_session` to mint a
        :class:`RefreshableCredentials`-backed session instead of a
        static one.
        """
        return self.refresher is not None

    def has_sso(self) -> bool:
        """True iff this client is configured for IAM Identity Center.

        Either :attr:`sso_start_url` alone (token-cache flow primed by
        ``aws sso login``) or the full role triple
        (``sso_account_id`` + ``sso_role_name`` + ``sso_region``) is
        enough — boto3 picks up whichever set is present once the
        profile is materialised at session-build time.
        """
        return bool(
            self.sso_start_url
            or (self.sso_account_id and self.sso_role_name)
        )

    def refresh_metadata(self) -> dict[str, Any]:
        """Invoke :attr:`refresher` and return botocore-shaped metadata."""
        if self.refresher is None:
            raise RuntimeError(
                "AWSClient.refresh_metadata() requires a refresher; "
                "none is set. Build the client via "
                "AWSClient.from_refresher(...) or assign client.refresher."
            )
        return dict(_refresher_to_metadata(self.refresher))

    def to_credentials(self) -> AwsCredentials:
        """Snapshot the static credentials into an :class:`AwsCredentials`.

        Returns the configured static fields; does NOT materialize
        assumed-role tokens — exporting a live STS token would defeat
        the auto-refresh that's the whole point of using a role.
        """
        return AwsCredentials(
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            session_token=self.session_token,
        )

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
        - Everything else identity-bearing goes in the query string;
          the secret credential fields are emitted only via userinfo.
        """
        query: dict[str, Any] = {}
        for name in self._IDENTITY_FIELDS:
            # Region rides the host slot, secrets ride userinfo.
            if name in (
                "access_key_id", "secret_access_key", "session_token",
                "region",
            ):
                continue
            value = getattr(self, name)
            if value is not None:
                query[name] = value

        host = self.region or ""
        url = URL.from_str(f"{scheme or self.url_scheme()}://{host}/")
        url = url.with_query_items(query)

        if self.has_static_credentials():
            url = url.with_user_password(
                user=self.access_key_id,
                password=self.secret_access_key,
            )

        return url

    @classmethod
    def from_(cls: Type[TC], obj: Any) -> TC:
        """Coerce *obj* (str / URL / dict / AWSClient) to a client."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, URL):
            return cls.from_parsed_url(obj)
        if isinstance(obj, str):
            return cls.from_parsed_url(URL.from_str(obj))
        if isinstance(obj, dict):
            return cls(**obj)
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

        return cls(**kwargs)

    # ==================================================================
    # Coercion entry points (formerly on AWSConfig)
    # ==================================================================

    @classmethod
    def from_credentials(
        cls: Type[TC],
        creds: AwsCredentials,
        *,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        refresher: Optional[CredentialsRefresher] = None,
        **kwargs: Any,
    ) -> TC:
        """Construct from a static :class:`AwsCredentials`.

        Pass ``refresher`` for self-renewing temporary credentials.
        """
        return cls(
            access_key_id=creds.access_key_id,
            secret_access_key=creds.secret_access_key,
            session_token=creds.session_token,
            region=region,
            endpoint_url=endpoint_url,
            refresher=refresher,
            **kwargs,
        )

    @classmethod
    def from_refresher(
        cls: Type[TC],
        refresher: CredentialsRefresher,
        *,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        refresher_key: Optional[str] = None,
        **kwargs: Any,
    ) -> TC:
        """Build a self-refreshing :class:`AWSClient` from a credentials callback."""
        seed = _coerce_refresher_output(refresher())
        if isinstance(seed, AwsCredentials):
            access_key_id = seed.access_key_id
            secret_access_key = seed.secret_access_key
            session_token = seed.session_token
        else:
            access_key_id = seed.get("access_key")
            secret_access_key = seed.get("secret_key")
            session_token = seed.get("token")

        return cls(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            region=region,
            endpoint_url=endpoint_url,
            refresher=refresher,
            refresher_key=refresher_key,
            **kwargs,
        )

    @classmethod
    def from_databricks_sql(
        cls: Type[TC],
        query: str,
        *,
        client: Optional["DatabricksClient"] = None,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        columns: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> TC:
        """Build an :class:`AWSClient` that vends credentials from a Databricks SQL row."""
        refresher = DatabricksSQLCredentialsRefresher(
            query=query,
            client=client,
            columns=dict(columns) if columns else None,
            column_aliases=dict(cls.DATABRICKS_SQL_CREDENTIAL_COLUMNS),
        )
        return cls.from_refresher(
            refresher,
            region=region,
            endpoint_url=endpoint_url,
            **kwargs,
        )

    # ==================================================================
    # Singleton — process-global default
    # ==================================================================

    @classmethod
    def current(cls: Type[TC], *, reset: bool = False, **overrides: Any) -> TC:
        """Process-global default :class:`AWSClient`.

        ``reset=True`` rebuilds; ``overrides`` are passed to a fresh
        constructor. Mirrors :meth:`DatabricksClient.current`.
        """
        global _CURRENT_CLIENT

        if reset or _CURRENT_CLIENT is None:
            with _CURRENT_CLIENT_LOCK:
                if reset or _CURRENT_CLIENT is None:
                    _CURRENT_CLIENT = cls(**overrides)

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
            f"region={self.region!r}, "
            f"profile={self.profile!r}, "
            f"role_arn={self.role_arn!r})"
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
        _ = self.session
        return self

    def close(self) -> None:
        """Drop the cached session, all per-service clients, all service objects."""
        if self._session is not None:
            LOGGER.debug("Closing AWS client %r", self)
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

        - :meth:`has_refresher` → :class:`RefreshableCredentials`
          driven by the user-supplied callback.
        - :meth:`has_assume_role` → :class:`RefreshableCredentials`
          driven by STS AssumeRole.
        - :meth:`has_sso` → boto3-native SSO token provider via
          a transient profile.
        - Otherwise → static / profile / default-chain creds.
        """
        if self._session is None:
            self._session = self._build_session()
        return self._session

    def client(self, service: str, **overrides: Any) -> "BaseClient":
        """Get a boto3 client for *service*. Cached per (service, overrides)."""
        cache_key = (
            service,
            tuple(sorted(overrides.items())) if overrides else (),
        )
        cached = self._client_cache.get(cache_key)
        if cached is not None:
            return cached

        LOGGER.debug(
            "Building boto client for service %r on %r (overrides=%r)",
            service, self, overrides or None,
        )
        kwargs: dict[str, Any] = {}
        if self.region:
            kwargs["region_name"] = self.region
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if service == "s3" and self.s3_addressing_style:
            botocore = botocore_module()
            kwargs["config"] = botocore.config.Config(
                s3={"addressing_style": self.s3_addressing_style}
            )
        kwargs.update(overrides)

        boto_client = self.session.client(service, **kwargs)
        self._client_cache[cache_key] = boto_client
        return boto_client

    def s3_client(self, **overrides: Any) -> "BaseClient":
        return self.client("s3", **overrides)

    def sts_client(self, **overrides: Any) -> "BaseClient":
        return self.client("sts", **overrides)

    # ==================================================================
    # Lazy-cached sub-service factories
    # ==================================================================

    @property
    def s3(self) -> "S3Service":
        """The :class:`S3Service` bound to this client. Lazy + cached."""
        if self._s3 is None:
            from .fs.service import S3Service
            self._s3 = S3Service(client=self)
        return self._s3

    # ==================================================================
    # Identity helper — useful for default tags / debugging
    # ==================================================================

    def caller_identity(self) -> dict[str, Any]:
        """Wrap STS GetCallerIdentity. Network call; not cached."""
        LOGGER.debug("Fetching STS caller identity for %r", self)
        return self.sts_client().get_caller_identity()

    @property
    def account_id(self) -> str:
        """Resolve the account ID via STS. Cached on the instance."""
        if self._account_id is None:
            identity = self.caller_identity()
            self._account_id = identity["Account"]
        return self._account_id

    @property
    def effective_region(self) -> Optional[str]:
        """Configured region first, then boto session default."""
        if self.region:
            return self.region
        try:
            return self.session.region_name
        except Exception:
            return None

    @property
    def explore_url(self) -> URL:
        """AWS Console home for this client's region — clickable from code."""
        from yggdrasil.aws.console import account_console_url

        return account_console_url(self.effective_region)

    @property
    def account(self) -> "AWSAccount":
        """The :class:`AWSAccount` resource for this client (STS-backed)."""
        from yggdrasil.aws.account import AWSAccount, AccountService

        return AWSAccount(service=AccountService(client=self))

    @property
    def batch(self) -> "AWSBatch":
        """The :class:`AWSBatch` runtime resource for this client.

        Pure ``os.environ`` read — no network, no credentials. ``.is_batch``
        gates whether the job-id / queue / array-index fields are meaningful.
        """
        from yggdrasil.aws.batch import AWSBatch, BatchService

        return AWSBatch(service=BatchService(client=self))

    # ==================================================================
    # Session construction internals
    # ==================================================================

    def _build_session(self) -> "boto3.Session":
        boto3 = boto3_module()
        if self.has_refresher():
            mode = "refresher"
        elif self.has_assume_role():
            mode = "assume-role"
        elif self.has_sso():
            mode = "sso"
        else:
            mode = "simple"
        LOGGER.debug("Building boto3 session for %r (mode=%s)", self, mode)
        if mode == "refresher":
            session = self._build_refresher_session(boto3)
        elif mode == "assume-role":
            session = self._build_assume_role_session(boto3)
        elif mode == "sso":
            session = self._build_sso_session(boto3)
        else:
            session = self._build_simple_session(boto3)
        LOGGER.info("Built boto3 session for %r (mode=%s)", self, mode)
        return session

    def _build_refresher_session(self, boto3) -> "boto3.Session":
        """Build a Session whose credentials auto-refresh via :attr:`refresher`."""
        botocore = botocore_module()
        refresher = self.refresher
        assert refresher is not None  # gated by has_refresher() above

        def refresh():
            LOGGER.debug("Refreshing credentials via refresher %r", refresher)
            metadata = _refresher_to_metadata(refresher)
            LOGGER.info(
                "Refreshed credentials via refresher %r (expiry=%s)",
                refresher, metadata.get("expiry_time"),
            )
            return metadata

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
        if self.region:
            botocore_session.set_config_variable("region", self.region)

        return boto3.Session(botocore_session=botocore_session)

    def _build_simple_session(self, boto3) -> "boto3.Session":
        """Static / profile / default-chain session."""
        kwargs: dict[str, Any] = {}
        if self.profile:
            kwargs["profile_name"] = self.profile
        if self.access_key_id:
            kwargs["aws_access_key_id"] = self.access_key_id
        if self.secret_access_key:
            kwargs["aws_secret_access_key"] = self.secret_access_key
        if self.session_token:
            kwargs["aws_session_token"] = self.session_token
        if self.region:
            kwargs["region_name"] = self.region
        return boto3.Session(**kwargs)

    def _build_sso_session(self, boto3) -> "boto3.Session":
        """Build a Session that auths through IAM Identity Center (SSO).

        boto3 reads SSO config from named profiles in
        ``~/.aws/config``. To avoid mutating the user's on-disk
        config we materialise the SSO knobs into an in-memory
        botocore session via ``set_config_variable`` and bind a
        fresh profile name to it. The ``SSOTokenProvider`` /
        ``SSOCredentialFetcher`` chain then handles the device-code
        + external-browser dance (re-using the SSO token cache that
        ``aws sso login`` populates).
        """
        botocore = botocore_module()
        botocore_session = botocore.session.get_session()

        if self.region:
            botocore_session.set_config_variable("region", self.region)

        # Materialise the SSO knobs as a synthetic profile so
        # boto3's SSOTokenProvider picks them up exactly as if they
        # were in ~/.aws/config. The profile name is namespaced
        # under ``ygg-sso-`` so it doesn't collide with user profiles.
        profile_name = self.profile or f"ygg-sso-{uuid.uuid4().hex[:8]}"
        sso_profile: dict[str, Any] = {}
        if self.sso_start_url:
            sso_profile["sso_start_url"] = self.sso_start_url
        if self.sso_region:
            sso_profile["sso_region"] = self.sso_region
        if self.sso_account_id:
            sso_profile["sso_account_id"] = self.sso_account_id
        if self.sso_role_name:
            sso_profile["sso_role_name"] = self.sso_role_name
        if self.region:
            sso_profile["region"] = self.region

        # ``_build_profile_map`` is the cached profile lookup
        # botocore's loaders use; injecting our synthetic profile
        # there is the supported in-memory route.
        full_config = botocore_session.full_config
        full_config.setdefault("profiles", {})[profile_name] = sso_profile
        botocore_session.set_config_variable("profile", profile_name)

        return boto3.Session(botocore_session=botocore_session)

    def _build_assume_role_session(self, boto3) -> "boto3.Session":
        """Build a Session whose credentials auto-refresh via STS AssumeRole."""
        botocore = botocore_module()
        base_session = self._build_simple_session(boto3)

        role_session_name = (
            self.role_session_name
            or _runtime_session_name()
            or f"yggdrasil-{uuid.uuid4().hex[:12]}"
        )

        def refresh():
            LOGGER.debug(
                "Refreshing STS AssumeRole credentials (role_arn=%r, session_name=%r, duration=%ds)",
                self.role_arn, role_session_name, int(self.duration_seconds),
            )
            sts = base_session.client("sts")
            assume_kwargs: dict[str, Any] = {
                "RoleArn": self.role_arn,
                "RoleSessionName": role_session_name,
                "DurationSeconds": int(self.duration_seconds),
            }
            if self.external_id:
                assume_kwargs["ExternalId"] = self.external_id
            response = sts.assume_role(**assume_kwargs)
            creds = response["Credentials"]
            expiration = creds["Expiration"]
            expiry_str = (
                expiration.isoformat()
                if hasattr(expiration, "isoformat")
                else str(expiration)
            )
            LOGGER.info(
                "Refreshed STS AssumeRole credentials (role_arn=%r, expiry=%s)",
                self.role_arn, expiry_str,
            )
            return {
                "access_key": creds["AccessKeyId"],
                "secret_key": creds["SecretAccessKey"],
                "token": creds["SessionToken"],
                "expiry_time": expiry_str,
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
        botocore_session._credentials = refreshable
        if self.region:
            botocore_session.set_config_variable("region", self.region)

        return boto3.Session(botocore_session=botocore_session)


# ===========================================================================
# AWSService
# ===========================================================================


class AWSService(ABC):
    """Abstract base for AWS service objects.

    A service object binds an :class:`AWSClient` to a particular AWS
    service (S3, DynamoDB, …). Mirrors :class:`DatabricksService`:
    holds a ``client``, delegates shared concerns (region,
    account_id, session) upstream, supports a ``current()``
    singleton, and round-trips via URL.

    Identity & singleton caching
    ----------------------------

    Instances are cached per ``(class, client)`` in
    :attr:`_INSTANCES`. Pickling routes through
    :meth:`__getnewargs__` so a service unpickled in the same
    process collapses to the live singleton. Subclasses add
    non-picklable handles to :attr:`_TRANSIENT_STATE_ATTRS`.
    """

    _INSTANCES: ClassVar[dict[Tuple[type, "AWSClient"], "AWSService"]] = {}

    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset()

    _current: ClassVar[Optional["AWSService"]] = None

    def __new__(
        cls: Type[TS],
        client: Optional[AWSClient] = None,
    ) -> TS:
        if client is None:
            client = AWSClient.current()
        key = (cls, client)
        cached = cls._INSTANCES.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        # ``dict.setdefault`` is GIL-atomic: under contention two
        # threads may both allocate, but only the first writer's
        # instance ends up in the cache and is returned to all
        # callers. No external mutex needed.
        instance = super().__new__(cls)
        return cls._INSTANCES.setdefault(key, instance)  # type: ignore[return-value]

    def __init__(self, client: Optional[AWSClient] = None) -> None:
        if getattr(self, "_initialized", False):
            return
        self.client: AWSClient = client if client is not None else AWSClient.current()
        self._initialized = True

    @classmethod
    def service_name(cls) -> str:
        name = cls.__name__
        if name.endswith("Service"):
            name = name[:-len("Service")]
        return name.lower()

    @classmethod
    def url_scheme(cls) -> str:
        return f"aws+{cls.service_name()}"

    @property
    def boto_client(self) -> "BaseClient":
        return self.client.client(self.service_name())

    @classmethod
    def current(cls: Type[TS], *, reset: bool = False) -> TS:
        if reset or cls._current is None:
            cls._current = cls(client=AWSClient.current())
        return cls._current  # type: ignore[return-value]

    @classmethod
    def set_current(cls, service: Optional["AWSService"]) -> None:
        cls._current = service

    def to_url(self, scheme: Optional[str] = None) -> URL:
        return (
            self.client
            .to_url(scheme=scheme or self.url_scheme())
            .with_path(f"/{self.service_name()}")
        )

    @classmethod
    def from_parsed_url(cls: Type[TS], url: URL) -> TS:
        return cls(client=AWSClient.from_parsed_url(url))

    def __enter__(self) -> "AWSService":
        self.client.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.client.__exit__(exc_type, exc_val, exc_tb)

    def connect(self) -> "AWSService":
        self.client.connect()
        return self

    def close(self) -> None:
        return

    @property
    def config(self) -> AWSClient:
        # Back-compat: ``service.config`` used to return the
        # AWSConfig instance separate from the client. The merged
        # class IS the config, so just hand back the client.
        return self.client

    @property
    def region(self) -> Optional[str]:
        return self.client.effective_region

    @property
    def account_id(self) -> str:
        return self.client.account_id

    def __getnewargs__(self):
        return (self.client,)

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items()
            if k not in self._TRANSIENT_STATE_ATTRS
        }

    def __setstate__(self, state):
        if getattr(self, "_initialized", False):
            return
        self.__dict__.update(state)
        self._initialized = True


# ===========================================================================
# AWSResource
# ===========================================================================


class AWSResource(ExploreUrlRepr, ABC):
    """Abstract base for AWS-backed entities.

    Concrete resources (:class:`~yggdrasil.aws.account.AWSAccount`,
    :class:`~yggdrasil.aws.fs.path.S3Bucket`, …) override :attr:`explore_url`
    to return a Console deep-link; the inherited :class:`ExploreUrlRepr` then
    gives a clickable repr / ``_repr_html_`` for free."""

    service: AWSService

    def __init__(self, service: Optional[AWSService] = None, *args, **kwargs) -> None:
        if service is None:
            service = AWSService.current()
        self.service = service
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        return {"service": self.service}

    def __setstate__(self, state):
        self.service = state["service"]

    @property
    def client(self) -> AWSClient:
        return self.service.client

    @property
    def boto_client(self) -> "BaseClient":
        return self.service.boto_client
