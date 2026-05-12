"""Databricks-vended AWS credentials providers.

Two :class:`AwsCredentialsProvider` subclasses backed by Unity
Catalog's temporary-credentials APIs:

- :class:`AWSDatabricksVolumeCredentials` — vended through
  ``temporary_volume_credentials.generate_temporary_volume_credentials``,
  scoped to a volume id.
- :class:`AWSDatabricksTableCredentials` — vended through
  ``temporary_table_credentials.generate_temporary_table_credentials``,
  scoped to a table id.

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
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from yggdrasil.aws.config import AwsCredentials
from yggdrasil.aws.provider import AwsCredentialsProvider
from yggdrasil.data.enums import Mode, ModeLike

if TYPE_CHECKING:
    from databricks.sdk.service.catalog import TableOperation, VolumeOperation

    from yggdrasil.aws.client import AWSClient
    from yggdrasil.databricks.client import DatabricksClient


__all__ = [
    "AWSDatabricksVolumeCredentials",
    "AWSDatabricksTableCredentials",
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
    DEFAULT_MODE: ClassVar[Mode] = Mode.READ_ONLY

    def __init__(
        self,
        key: str,
        *,
        client: Any = None,
    ) -> None:
        # ``AwsCredentialsProvider.__init__`` is idempotent — re-entry
        # just rebinds the live client so refreshes after a workspace
        # rotation pick up the new auth context.
        if getattr(self, "_initialized", False):
            if client is not None:
                self._client = client
            return
        super().__init__(key)
        self._client: Any = client
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
        return AwsCredentials(
            access_key_id=aws.access_key_id,
            access_point=getattr(aws, "access_point", None),
            secret_access_key=aws.secret_access_key,
            session_token=aws.session_token,
            expiration=_iso_or_str(getattr(resp, "expiration_time", None)),
        )

    def __call__(self) -> AwsCredentials:
        return self.get_credentials()

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
            from yggdrasil.aws.config import AWSConfig
            refresher = _ModeBoundRefresher(provider=self, mode=resolved)
            # Discriminator so the AWSClient singleton cache mints a
            # distinct session per (provider, resource, mode) — without
            # it, read and write configs would collapse to one client
            # (refresher itself is excluded from equality).
            refresher_key = f"{type(self).__name__}:{self.key}:{resolved.name}"
            client = AWSConfig.from_refresher(
                refresher, region=region, refresher_key=refresher_key,
            ).to_client()
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
                "EXTERNAL_USE_SCHEMA self-grant: current_user lookup failed",
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
                "EXTERNAL_USE_SCHEMA self-grant: grant() failed on %s.%s for %s",
                catalog_name, schema_name, principal, exc_info=True,
            )
            return False

        LOGGER.info(
            "Self-granted EXTERNAL_USE_SCHEMA on %s.%s to %s; retrying credential mint",
            catalog_name, schema_name, principal,
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

    def __new__(
        cls,
        volume_id: str,
        *,
        client: Any = None,
    ) -> "AWSDatabricksVolumeCredentials":
        return super().__new__(cls, str(volume_id))

    def __init__(
        self,
        volume_id: str,
        *,
        client: Any = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            if client is not None:
                self._client = client
            return
        super().__init__(str(volume_id), client=client)
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

    def __new__(
        cls,
        table_id: str,
        *,
        client: Any = None,
    ) -> "AWSDatabricksTableCredentials":
        return super().__new__(cls, str(table_id))

    def __init__(
        self,
        table_id: str,
        *,
        client: Any = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            if client is not None:
                self._client = client
            return
        super().__init__(str(table_id), client=client)
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
