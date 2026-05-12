"""Databricks-vended AWS credentials providers.

Two :class:`AwsCredentialsProvider` subclasses backed by Unity
Catalog's temporary-credentials APIs:

- :class:`AWSDatabricksVolumeCredentials` — vended through
  ``temporary_volume_credentials.generate_temporary_volume_credentials``,
  scoped to a volume id + :class:`VolumeOperation`.
- :class:`AWSDatabricksTableCredentials` — vended through
  ``temporary_table_credentials.generate_temporary_table_credentials``,
  scoped to a table id + :class:`TableOperation`.

Both inherit the provider's process-wide singleton-by-key behavior
and per-region :class:`AWSClient` cache, so any number of
:class:`VolumePath` / :class:`Table` instances on the same scope
share one STS vend, one boto session, and one connection pool.

The bound :class:`DatabricksClient` is mutable: every constructor
call updates the live binding so refreshes that follow a client
rotation pick up the fresh workspace auth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.aws.config import AwsCredentials
from yggdrasil.aws.provider import AwsCredentialsProvider

if TYPE_CHECKING:
    from databricks.sdk.service.catalog import TableOperation, VolumeOperation

    from yggdrasil.databricks.client import DatabricksClient


__all__ = [
    "AWSDatabricksVolumeCredentials",
    "AWSDatabricksTableCredentials",
]


def _op_token(operation: Any) -> str:
    """Stable wire token for an enum-or-string operation."""
    return (
        getattr(operation, "value", None)
        or getattr(operation, "name", None)
        or str(operation)
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


class _DatabricksCredentialsBase(AwsCredentialsProvider):
    """Common plumbing for the volume / table UC-vended providers.

    Concrete subclasses define :attr:`_RESOURCE_NAME` and
    :meth:`_generate` (the actual SDK call).
    """

    _RESOURCE_NAME: str = ""

    def __init__(
        self,
        key: str,
        *,
        client: Any = None,
    ) -> None:
        # ``AwsCredentialsProvider.__init__`` is idempotent — re-entry
        # only updates the bound client so refreshes after a client
        # rotation pick up the new workspace auth.
        if getattr(self, "_initialized", False):
            if client is not None:
                self._client = client
            return
        super().__init__(key)
        self._client: Any = client

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

    # Refresh-time storage location returned by the SDK is irrelevant
    # for the credential refresh, but subclasses surface it via the
    # raw SDK response.

    def get_credentials(self) -> AwsCredentials:
        resp = self._generate()
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

    def _generate(self) -> Any:
        raise NotImplementedError


class AWSDatabricksVolumeCredentials(_DatabricksCredentialsBase):
    """Refreshable AWS creds for a Unity Catalog volume.

    Singleton-cached per ``(volume_id, operation)``. Subsequent
    constructions return the same instance and just re-bind the
    :class:`DatabricksClient` if one is passed.
    """

    _RESOURCE_NAME = "volume_id"

    def __new__(
        cls,
        volume_id: str,
        operation: "VolumeOperation",
        *,
        client: Any = None,
    ) -> "AWSDatabricksVolumeCredentials":
        key = f"{volume_id}:{_op_token(operation)}"
        return super().__new__(cls, key)

    def __init__(
        self,
        volume_id: str,
        operation: "VolumeOperation",
        *,
        client: Any = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            if client is not None:
                self._client = client
            return
        key = f"{volume_id}:{_op_token(operation)}"
        super().__init__(key, client=client)
        self.volume_id: str = str(volume_id)
        self.operation: Any = operation

    def __getnewargs__(self):
        return (self.volume_id, self.operation)

    def __getstate__(self):
        return {"volume_id": self.volume_id, "operation": self.operation}

    def __setstate__(self, state):
        if getattr(self, "_initialized", False):
            return
        # Re-derive the key from the identity fields so the parent
        # class' internal slot stays consistent with the singleton key.
        volume_id = state["volume_id"]
        operation = state["operation"]
        super().__setstate__({"key": f"{volume_id}:{_op_token(operation)}"})
        self.volume_id = str(volume_id)
        self.operation = operation
        self._client = None

    def _generate(self) -> Any:
        return (
            self.workspace.temporary_volume_credentials
            .generate_temporary_volume_credentials(
                volume_id=self.volume_id,
                operation=self.operation,
            )
        )


class AWSDatabricksTableCredentials(_DatabricksCredentialsBase):
    """Refreshable AWS creds for a Unity Catalog table.

    Singleton-cached per ``(table_id, operation)``.
    """

    _RESOURCE_NAME = "table_id"

    def __new__(
        cls,
        table_id: str,
        operation: "TableOperation",
        *,
        client: Any = None,
    ) -> "AWSDatabricksTableCredentials":
        key = f"{table_id}:{_op_token(operation)}"
        return super().__new__(cls, key)

    def __init__(
        self,
        table_id: str,
        operation: "TableOperation",
        *,
        client: Any = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            if client is not None:
                self._client = client
            return
        key = f"{table_id}:{_op_token(operation)}"
        super().__init__(key, client=client)
        self.table_id: str = str(table_id)
        self.operation: Any = operation

    def __getnewargs__(self):
        return (self.table_id, self.operation)

    def __getstate__(self):
        return {"table_id": self.table_id, "operation": self.operation}

    def __setstate__(self, state):
        if getattr(self, "_initialized", False):
            return
        table_id = state["table_id"]
        operation = state["operation"]
        super().__setstate__({"key": f"{table_id}:{_op_token(operation)}"})
        self.table_id = str(table_id)
        self.operation = operation
        self._client = None

    def _generate(self) -> Any:
        return (
            self.workspace.temporary_table_credentials
            .generate_temporary_table_credentials(
                table_id=self.table_id,
                operation=self.operation,
            )
        )
