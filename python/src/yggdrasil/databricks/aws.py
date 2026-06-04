"""Databricks-vended AWS credentials providers.

Two :class:`AwsCredentialsProvider` subclasses backed by Unity
Catalog's temporary-credentials APIs:

- :class:`AWSDatabricksVolumeCredentials` — vended through
  ``temporary_volume_credentials.generate_temporary_volume_credentials``,
  scoped to a volume id.
- :class:`AWSDatabricksTableCredentials` — vended through
  ``temporary_table_credentials.generate_temporary_table_credentials``,
  scoped to a table id.
- :class:`AWSDatabricksPathCredentials` — vended through
  ``temporary_path_credentials.generate_temporary_path_credentials``,
  scoped to a storage URL. This is the only endpoint that vends
  creds for an external location's **storage** credential (the
  service-credential endpoint rejects those), so it backs
  :class:`~yggdrasil.databricks.external.location.resource.ExternalLocation`.

Each provider is a process-wide singleton per resource id and
handles **both read and write modes internally** —
:meth:`get_credentials(mode=...)` resolves the requested
:class:`Mode` into the right UC operation and returns the matching
credentials. The per-region :class:`AWSClient` cache is keyed by
``(mode, region)`` so reads and writes mint distinct boto sessions
while still sharing one provider, one bound :class:`DatabricksClient`,
and the singleton-by-resource_id guarantee.

The bound :class:`DatabricksClient` is mutable: every constructor
call updates the live binding so refreshes that follow a client
rotation pick up the fresh workspace auth.

Vended credentials are also cached in a Databricks **secret scope, one
per resource** — ``aws.<kind>.<resource>`` (e.g.
``aws.volume.cat.sch.vol`` / ``aws.table.cat.sch.tbl``), with the
credentials stored under a single ``credentials`` key (a read/write map).
A later resolution — in this process, on a Spark executor, or in a fresh
run — reuses a still-valid cached credential instead of re-vending it from
Unity Catalog; an in-process memo short-circuits repeat calls without even
reading the secret. The ``aws`` prefix is overridable via
``YGG_DATABRICKS_CREDS_SECRET_PREFIX`` (set it empty to disable
persistence). The whole layer is best-effort: any Secrets-API failure
falls back to a fresh UC vend, so it never blocks credential resolution.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import os
import re
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional

from yggdrasil.aws.config import AwsCredentials
from yggdrasil.aws.provider import AwsCredentialsProvider
from yggdrasil.enums import Mode, ModeLike

if TYPE_CHECKING:
    from databricks.sdk.service.catalog import (
        PathOperation, TableOperation, VolumeOperation,
    )

    from yggdrasil.aws.client import AWSClient
    from yggdrasil.databricks.client import DatabricksClient


__all__ = [
    "AWSDatabricksVolumeCredentials",
    "AWSDatabricksTableCredentials",
    "AWSDatabricksPathCredentials",
]


LOGGER = logging.getLogger(__name__)


# Matches the UC error message shape:
#   "User does not have EXTERNAL USE SCHEMA on Schema 'cat.sch'"
# Captures the two-part schema name (with or without surrounding
# quotes / backticks). Case-insensitive — UC's wording has drifted
# between releases.
_EXTERNAL_USE_SCHEMA_RE = re.compile(
    r"EXTERNAL\s+USE\s+SCHEMA\s+on\s+Schema\s+['\"`]?([^'\"`\s]+)['\"`]?",
    re.IGNORECASE,
)


def _iso_or_str(value: Any) -> Optional[str]:
    """Coerce an expiration timestamp into the ISO-8601 string botocore
    wants for ``RefreshableCredentials``' ``expiry_time``.

    SDK shapes vary: ``datetime`` (volumes), ms-since-epoch ``int``
    (tables), or already-stringified ISO.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        import datetime as _dt
        return _dt.datetime.fromtimestamp(
            float(value) / 1000.0, tz=_dt.timezone.utc,
        ).isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


@dataclasses.dataclass(frozen=True)
class _ModeBoundRefresher:
    """Picklable per-mode adapter exposed as a no-arg refresher.

    Botocore's :class:`RefreshableCredentials` re-invokes its refresher
    with no arguments; the bound mode lets one provider drive both
    read and write botocore sessions from the same singleton.
    """

    provider: "_DatabricksCredentialsBase"
    mode: Mode

    def __call__(self) -> AwsCredentials:
        return self.provider.get_credentials(self.mode)


class _DatabricksCredentialsBase(AwsCredentialsProvider):
    """Common plumbing for the volume / table UC-vended providers.

    Subclasses define :attr:`_RESOURCE_NAME` (used in error messages),
    :meth:`_operation_for` (Mode → UC operation), and
    :meth:`_generate` (the SDK call).
    """

    _RESOURCE_NAME: ClassVar[str] = ""
    #: Short, human-readable kind stamped into the per-resource secret scope
    #: name (``aws.<kind>.<resource>``). Overridden per subclass.
    _RESOURCE_KIND: ClassVar[str] = "resource"
    DEFAULT_MODE: ClassVar[Mode] = Mode.READ_ONLY

    def __init__(
        self,
        key: str,
        *,
        client: Any = None,
        resource_url: "str | None" = None,
    ) -> None:
        # ``AwsCredentialsProvider.__init__`` is idempotent — re-entry
        # just rebinds the live client so refreshes after a workspace
        # rotation pick up the new auth context.
        if getattr(self, "_initialized", False):
            if client is not None:
                self._client = client
            if resource_url and not getattr(self, "_resource_url", None):
                self._resource_url = str(resource_url)
            return
        super().__init__(key)
        self._client: Any = client
        # Readable resource identity (the volume / table UC name, or the
        # storage URL) used to name the per-resource secret scope; falls
        # back to ``key`` (the UC id) when not supplied.
        self._resource_url: "Optional[str]" = str(resource_url) if resource_url else None
        # Cache key is the *resolved* mode + region.
        self._client_cache: "dict[tuple[Mode, Optional[str]], AWSClient]" = {}

    @property
    def client(self) -> "DatabricksClient":
        """Bound :class:`DatabricksClient`. Lazily resolves to
        :meth:`DatabricksClient.current` when no client was passed."""
        if self._client is None:
            from yggdrasil.lazy_imports import databricks_client_class
            self._client = databricks_client_class().current()
        return self._client

    @property
    def workspace(self) -> Any:
        """Shortcut for ``self.client.workspace_client()``."""
        return self.client.workspace_client()

    def with_client(self, client: Any) -> "_DatabricksCredentialsBase":
        """Replace the bound client. Returns *self*."""
        self._client = client
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_credentials(self, mode: ModeLike = None) -> AwsCredentials:
        """Return fresh credentials scoped to *mode*.

        ``mode`` accepts a :class:`Mode` enum, a mode-like string
        (``"read"``, ``"overwrite"``, …), or ``None`` (uses
        :attr:`DEFAULT_MODE`). Read-only modes vend the UC "read"
        operation; everything else vends the writable operation.

        If the SDK rejects the call with
        ``PermissionDenied: ... EXTERNAL USE SCHEMA on Schema 'cat.sch'``,
        we make exactly one attempt to self-grant
        ``EXTERNAL_USE_SCHEMA`` on the offending schema and retry.
        Owners of UC schemas commonly forget this grant — when they
        own the schema they have permission to fix it, and silently
        succeeding is dramatically less surprising than asking them
        to read the error and run a follow-up SQL. If the recovery
        itself fails (non-owner, network error, …) the *original*
        PermissionDenied propagates so the failure mode stays
        obvious.
        """
        resolved = Mode.from_(mode, default=self.DEFAULT_MODE)
        cached = self._load_persisted(resolved)
        if cached is not None:
            return cached
        operation = self._operation_for(resolved)
        try:
            resp = self._generate(operation)
        except Exception as exc:
            if not self._try_self_grant_external_use_schema(exc):
                raise
            resp = self._generate(operation)
        aws = getattr(resp, "aws_temp_credentials", None)
        if aws is None:
            raise RuntimeError(
                f"{type(self).__name__}: temporary credentials for "
                f"{self._RESOURCE_NAME}={self.key!r} carry no "
                f"``aws_temp_credentials`` — the {self._RESOURCE_NAME} "
                f"is likely backed by Azure or GCP, not S3."
            )
        creds = AwsCredentials(
            access_key_id=aws.access_key_id,
            access_point=getattr(aws, "access_point", None),
            secret_access_key=aws.secret_access_key,
            session_token=aws.session_token,
            expiration=_iso_or_str(getattr(resp, "expiration_time", None)),
        )
        self._persist(resolved, creds)
        return creds

    # ------------------------------------------------------------------
    # Secrets-backed persistence of vended credentials
    # ------------------------------------------------------------------
    #
    # Temporary credentials are cached in a Databricks secret scope, **one per
    # resource** (``aws.<kind>.<resource>``), under a single ``credentials``
    # key whose value is a ``{mode: creds}`` map. A later resolution — in this
    # process or another (a Spark executor, a fresh run) — reuses a still-valid
    # credential instead of re-vending it from Unity Catalog. An in-process
    # memo short-circuits repeat calls without even reading the secret.
    #
    # All of it is **best-effort**: any Secrets-API failure (scope missing, no
    # permission, network) silently falls back to a fresh UC vend, so the
    # backing can never break credential resolution. Set
    # ``YGG_DATABRICKS_CREDS_SECRET_PREFIX`` to override the ``aws`` scope
    # prefix, or to an empty string to disable persistence entirely.

    #: Don't reuse a persisted credential within this many seconds of its
    #: expiry — leave the caller enough runway to actually use it before a
    #: refresh is forced.
    _PERSIST_EXPIRY_MARGIN: ClassVar[float] = 600.0
    _DEFAULT_SECRET_PREFIX: ClassVar[str] = "aws"
    #: The single secret key each per-resource scope stores its read/write
    #: credential map under.
    _SECRET_KEY: ClassVar[str] = "credentials"

    @classmethod
    def _secret_prefix(cls) -> "Optional[str]":
        """The scope-name prefix (``aws`` by default), or ``None`` when
        persistence is disabled via an empty override."""
        prefix = os.environ.get(
            "YGG_DATABRICKS_CREDS_SECRET_PREFIX", cls._DEFAULT_SECRET_PREFIX,
        )
        return prefix.strip() or None

    def _secret_scope(self) -> "Optional[str]":
        """Per-resource secret scope name (``<prefix>.<kind>.<resource>``), or
        ``None`` when persistence is disabled.

        The resource slug is the readable :attr:`_resource_url` (the volume /
        table UC name, or storage URL) when known, else the UC id. Databricks
        scope names allow ``[A-Za-z0-9_.@-]`` and cap at 128 chars, so the slug
        is sanitised and, when the whole name would overflow, truncated with a
        stable xxhash suffix so distinct resources never collide.
        """
        prefix = self._secret_prefix()
        if prefix is None:
            return None
        raw = str(self._resource_url or self.key)
        slug = re.sub(r"[^A-Za-z0-9_.@-]", "_", raw).strip("._@-") or "x"
        scope = f"{prefix}.{self._RESOURCE_KIND}.{slug}"
        if len(scope) > 128:
            import xxhash
            digest = xxhash.xxh64(raw.encode("utf-8")).hexdigest()
            head = f"{prefix}.{self._RESOURCE_KIND}."
            keep = max(1, 128 - len(head) - len(digest) - 1)
            scope = f"{head}{slug[:keep]}.{digest}"
        return scope

    @staticmethod
    def _creds_to_entry(creds: AwsCredentials) -> dict:
        return {
            "access_key_id": creds.access_key_id,
            "access_point": creds.access_point,
            "secret_access_key": creds.secret_access_key,
            "session_token": creds.session_token,
            "expiration": creds.expiration,
        }

    @staticmethod
    def _remaining_seconds(creds: AwsCredentials) -> float:
        """Seconds until *creds* expire; ``-1`` when there's no usable expiry
        (a credential with no expiration is never trusted as cached state)."""
        if not creds.expiration:
            return -1.0
        try:
            from yggdrasil.data.cast import any_to_datetime
            expires = any_to_datetime(creds.expiration, tz=dt.timezone.utc)
        except Exception:
            return -1.0
        return (expires - dt.datetime.now(dt.timezone.utc)).total_seconds()

    def _read_secret_map(self) -> dict:
        """The ``{mode: creds}`` map stored under the resource's ``credentials``
        secret, or ``{}`` (best-effort — any failure yields an empty map)."""
        scope = self._secret_scope()
        if scope is None:
            return {}
        try:
            secret = self.client.secrets.secret(self._SECRET_KEY, scope=scope)
            data = secret.refresh(raise_error=False).object
        except Exception:
            LOGGER.debug(
                "reading persisted credentials for %s=%r failed",
                self._RESOURCE_NAME, self.key, exc_info=True,
            )
            return {}
        return dict(data) if isinstance(data, Mapping) else {}

    def _load_persisted(self, mode: Mode) -> "Optional[AwsCredentials]":
        """Return a still-valid cached credential for *mode*, or ``None``.

        Checks the in-process memo first, then the resource's ``credentials``
        secret (picking the *mode* entry from its read/write map). Anything
        within :attr:`_PERSIST_EXPIRY_MARGIN` of expiry is treated as a miss so
        the caller vends fresh.
        """
        memo = self.__dict__.setdefault("_persisted_cache", {})
        hit = memo.get(mode)
        if hit is not None and self._remaining_seconds(hit) > self._PERSIST_EXPIRY_MARGIN:
            return hit

        entry = self._read_secret_map().get(mode.name)
        if not isinstance(entry, Mapping):
            return None
        creds = AwsCredentials(
            access_key_id=entry.get("access_key_id"),
            access_point=entry.get("access_point"),
            secret_access_key=entry.get("secret_access_key"),
            session_token=entry.get("session_token"),
            expiration=entry.get("expiration"),
        )
        if not creds.is_complete() or self._remaining_seconds(creds) <= self._PERSIST_EXPIRY_MARGIN:
            return None
        memo[mode] = creds
        LOGGER.debug(
            "reusing persisted credentials for %s=%r (mode=%s)",
            self._RESOURCE_NAME, self.key, mode.name,
        )
        return creds

    def _persist(self, mode: Mode, creds: AwsCredentials) -> None:
        """Best-effort: stash *creds* in the in-process memo and write the
        resource's ``credentials`` secret as a ``{mode: creds}`` map.

        The map is built from the in-process memo, so once a process has vended
        both read and write the single secret carries both. Only credentials
        that carry an expiry are persisted (so the loader's freshness check is
        meaningful)."""
        memo = self.__dict__.setdefault("_persisted_cache", {})
        memo[mode] = creds
        scope = self._secret_scope()
        if scope is None or not creds.expiration:
            return
        value = {
            m.name: self._creds_to_entry(c)
            for m, c in memo.items()
            if c.expiration
        }
        try:
            self.client.secrets.create_secret(
                key=self._SECRET_KEY, value=value, scope=scope,
            )
        except Exception:
            LOGGER.debug(
                "persisting credentials for %s=%r (mode=%s) failed",
                self._RESOURCE_NAME, self.key, mode.name, exc_info=True,
            )

    # ------------------------------------------------------------------
    # AWSClient binding — one per (mode, region)
    # ------------------------------------------------------------------

    def aws_client(
        self,
        *,
        mode: ModeLike = None,
        region: Optional[str] = None,
    ) -> "AWSClient":
        """Return the cached :class:`AWSClient` for this provider /
        mode / region.

        First call per ``(mode, region)`` seeds a botocore
        :class:`RefreshableCredentials`-backed session whose refresher
        is bound to *mode*; subsequent calls with the same key return
        the same client and reuse the connection pool, boto-client
        cache, and in-flight refresh state.
        """
        resolved = Mode.from_(mode, default=self.DEFAULT_MODE)
        cache_key = (resolved, region)
        with self._client_cache_lock:
            existing = self._client_cache.get(cache_key)
            if existing is not None:
                return existing
            from yggdrasil.aws.client import AWSClient
            refresher = _ModeBoundRefresher(provider=self, mode=resolved)
            # Discriminator so the AWSClient singleton cache mints a
            # distinct session per (provider, resource, mode) — without
            # it, read and write configs would collapse to one client
            # (refresher itself is excluded from equality).
            refresher_key = f"{type(self).__name__}:{self.key}:{resolved.name}"
            client = AWSClient.from_refresher(
                refresher, region=region, refresher_key=refresher_key,
            )
            self._client_cache[cache_key] = client
            return client

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _operation_for(self, mode: Mode) -> Any:
        raise NotImplementedError

    def _generate(self, operation: Any) -> Any:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Recovery — auto self-grant EXTERNAL_USE_SCHEMA on UC schema
    # ------------------------------------------------------------------

    def _try_self_grant_external_use_schema(self, exc: BaseException) -> bool:
        """One-shot recovery for ``PermissionDenied: EXTERNAL USE SCHEMA …``.

        Returns ``True`` when a fresh ``EXTERNAL_USE_SCHEMA`` grant was
        successfully applied to ``self.client.iam.users.current_user``
        on the schema named in *exc*, so the caller can retry the
        credential mint. Returns ``False`` (without raising) on any
        of:

        - *exc* isn't a permission error;
        - the error message doesn't carry an ``EXTERNAL USE SCHEMA on
          Schema 'cat.sch'`` clause we can parse;
        - the current user lookup fails or produces no usable
          principal (email / username);
        - the grant itself fails (caller isn't owner — propagating
          the *original* PermissionDenied is more informative).
        """
        if type(exc).__name__ != "PermissionDenied":
            return False
        match = _EXTERNAL_USE_SCHEMA_RE.search(str(exc))
        if not match:
            return False
        full = match.group(1).strip("'\"`")
        parts = [p for p in full.split(".") if p]
        if len(parts) != 2:
            return False
        catalog_name, schema_name = parts

        try:
            current = self.client.iam.users.current_user
        except Exception:
            LOGGER.debug(
                "Self-granting EXTERNAL_USE_SCHEMA: current_user lookup failed",
                exc_info=True,
            )
            return False
        principal = (
            getattr(current, "email", None)
            or getattr(current, "username", None)
            or getattr(current, "name", None)
        )
        if not principal:
            return False

        try:
            schema = self.client.schemas.schema(
                catalog_name=catalog_name, schema_name=schema_name,
            )
            schema.grant(principal, "EXTERNAL_USE_SCHEMA")
        except Exception:
            LOGGER.debug(
                "Self-granting EXTERNAL_USE_SCHEMA on %s.%s for principal %r failed",
                catalog_name, schema_name, principal, exc_info=True,
            )
            return False

        LOGGER.info(
            "Self-granted EXTERNAL_USE_SCHEMA on schema %r to principal %r; retrying credential mint",
            schema, principal,
        )
        return True


class AWSDatabricksVolumeCredentials(_DatabricksCredentialsBase):
    """Refreshable AWS creds for a Unity Catalog volume.

    Singleton-cached per ``volume_id``. One provider serves both
    read (``READ_VOLUME``) and write (``WRITE_VOLUME``) modes — the
    requested mode is resolved at :meth:`get_credentials` / :meth:`aws_client`
    time.
    """

    _RESOURCE_NAME = "volume_id"
    _RESOURCE_KIND = "volume"

    def __new__(
        cls,
        volume_id: str,
        *,
        client: Any = None,
        resource_url: "str | None" = None,
    ) -> "AWSDatabricksVolumeCredentials":
        return super().__new__(cls, str(volume_id))

    def __init__(
        self,
        volume_id: str,
        *,
        client: Any = None,
        resource_url: "str | None" = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            if client is not None:
                self._client = client
            if resource_url and not getattr(self, "_resource_url", None):
                self._resource_url = str(resource_url)
            return
        super().__init__(str(volume_id), client=client, resource_url=resource_url)
        self.volume_id: str = str(volume_id)

    def __getnewargs__(self):
        return (self.volume_id,)

    def __getstate__(self):
        return {"volume_id": self.volume_id}

    def __setstate__(self, state):
        if getattr(self, "_initialized", False):
            return
        volume_id = state["volume_id"]
        super().__setstate__({"key": str(volume_id)})
        self.volume_id = str(volume_id)
        self._client = None
        self._resource_url = None
        self._client_cache = {}

    def _operation_for(self, mode: Mode) -> "VolumeOperation":
        from databricks.sdk.service.catalog import VolumeOperation
        if mode is Mode.READ_ONLY:
            return VolumeOperation.READ_VOLUME
        return VolumeOperation.WRITE_VOLUME

    def _generate(self, operation: "VolumeOperation") -> Any:
        return (
            self.workspace.temporary_volume_credentials
            .generate_temporary_volume_credentials(
                volume_id=self.volume_id,
                operation=operation,
            )
        )


class AWSDatabricksTableCredentials(_DatabricksCredentialsBase):
    """Refreshable AWS creds for a Unity Catalog table.

    Singleton-cached per ``table_id``. One provider serves both
    read (``READ``) and write (``READ_WRITE``) modes — the requested
    mode is resolved at :meth:`get_credentials` / :meth:`aws_client`
    time.
    """

    _RESOURCE_NAME = "table_id"
    _RESOURCE_KIND = "table"

    def __new__(
        cls,
        table_id: str,
        *,
        client: Any = None,
        resource_url: "str | None" = None,
    ) -> "AWSDatabricksTableCredentials":
        return super().__new__(cls, str(table_id))

    def __init__(
        self,
        table_id: str,
        *,
        client: Any = None,
        resource_url: "str | None" = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            if client is not None:
                self._client = client
            if resource_url and not getattr(self, "_resource_url", None):
                self._resource_url = str(resource_url)
            return
        super().__init__(str(table_id), client=client, resource_url=resource_url)
        self.table_id: str = str(table_id)

    def __getnewargs__(self):
        return (self.table_id,)

    def __getstate__(self):
        return {"table_id": self.table_id}

    def __setstate__(self, state):
        if getattr(self, "_initialized", False):
            return
        table_id = state["table_id"]
        super().__setstate__({"key": str(table_id)})
        self.table_id = str(table_id)
        self._client = None
        self._resource_url = None
        self._client_cache = {}

    def _operation_for(self, mode: Mode) -> "TableOperation":
        from databricks.sdk.service.catalog import TableOperation
        if mode is Mode.READ_ONLY:
            return TableOperation.READ
        return TableOperation.READ_WRITE

    def _generate(self, operation: "TableOperation") -> Any:
        return (
            self.workspace.temporary_table_credentials
            .generate_temporary_table_credentials(
                table_id=self.table_id,
                operation=operation,
            )
        )


class AWSDatabricksPathCredentials(_DatabricksCredentialsBase):
    """Refreshable AWS creds for a Unity Catalog storage URL.

    Singleton-cached per ``url``. Vends through
    ``temporary_path_credentials.generate_temporary_path_credentials``,
    which — unlike the service-credential endpoint — works for the
    **storage** credential backing an external location. One provider
    serves read (``PATH_READ``) and write (``PATH_READ_WRITE``) modes;
    the requested mode resolves at :meth:`get_credentials` /
    :meth:`aws_client` time.

    The URL is normalised to a directory prefix (trailing ``/``) so
    every key under the same external-location prefix collapses to one
    provider, one refresh cycle, and one boto session.
    """

    _RESOURCE_NAME = "url"
    _RESOURCE_KIND = "path"

    @staticmethod
    def _normalize(url: str) -> str:
        # UC vends per-prefix; a trailing slash keeps siblings on one
        # provider and matches how the location addresses its base.
        text = str(url)
        return text if text.endswith("/") else text + "/"

    def __new__(
        cls,
        url: str,
        *,
        client: Any = None,
    ) -> "AWSDatabricksPathCredentials":
        return super().__new__(cls, cls._normalize(url))

    def __init__(
        self,
        url: str,
        *,
        client: Any = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            if client is not None:
                self._client = client
            return
        normalized = self._normalize(url)
        super().__init__(normalized, client=client)
        self.url: str = normalized

    def __getnewargs__(self):
        return (self.url,)

    def __getstate__(self):
        return {"url": self.url}

    def __setstate__(self, state):
        if getattr(self, "_initialized", False):
            return
        url = self._normalize(state["url"])
        super().__setstate__({"key": url})
        self.url = url
        self._client = None
        self._resource_url = None
        self._client_cache = {}

    def _operation_for(self, mode: Mode) -> "PathOperation":
        from databricks.sdk.service.catalog import PathOperation
        if mode is Mode.READ_ONLY:
            return PathOperation.PATH_READ
        return PathOperation.PATH_READ_WRITE

    def _generate(self, operation: "PathOperation") -> Any:
        return (
            self.workspace.temporary_path_credentials
            .generate_temporary_path_credentials(
                url=self.url,
                operation=operation,
            )
        )
