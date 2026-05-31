"""Per-credential resource: metadata + refreshable cloud credentials.

A :class:`Credential` wraps one Unity Catalog credential (a named binding to a
cloud identity — for AWS, an IAM role). Its headline feature is turning that
into **refreshable AWS credentials**: ``credential.aws_client()`` hands back a
self-refreshing :class:`~yggdrasil.aws.client.AWSClient` (and ``.aws_credentials()``
a one-shot STS token). Collection ops live on
:class:`~yggdrasil.databricks.credentials.credentials.Credentials`.

    cred = client.credentials["prod_s3"]
    s3 = cred.aws_client(region="us-east-1").s3      # auto-refreshing
    s3.path("s3://bucket/key").read_bytes()
"""
from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from databricks.sdk.service.catalog import CredentialInfo

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.dataclasses.singleton import Singleton
from yggdrasil.databricks.resource import DatabricksResource
from yggdrasil.url import URL

if TYPE_CHECKING:
    from databricks.sdk.service.catalog import AwsIamRole, TemporaryCredentials
    from yggdrasil.aws.client import AWSClient
    from yggdrasil.aws.config import AwsCredentials
    from yggdrasil.databricks.credentials.credentials import Credentials

__all__ = ["Credential"]

#: UC credentials / external locations are near-static config — cache the
#: resource handle (and its fetched info) for 30 min before re-resolving.
_RESOURCE_TTL: float = 30 * 60.0


def _epoch_ms_to_iso(ms: Optional[int]) -> Optional[str]:
    if not ms:
        return None
    return _dt.datetime.fromtimestamp(ms / 1000, tz=_dt.timezone.utc).isoformat()


class Credential(DatabricksResource, Singleton):
    """A single Unity Catalog credential.

    Cached as a singleton per ``(service, name)`` for 30 min (``_SINGLETON_TTL``)
    — credentials are near-static, so repeated ``client.credentials[name]`` share
    one handle (and its fetched info) without re-resolving.
    """

    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(default_ttl=_RESOURCE_TTL, max_size=4096)
    _SINGLETON_TTL: ClassVar[Any] = _RESOURCE_TTL

    @classmethod
    def _singleton_key(cls, name: Any = None, *, service: Any = None, **kwargs: Any) -> Any:
        return (cls, service, name)

    def __init__(
        self,
        name: str,
        *,
        service: "Optional[Credentials]" = None,
        info: Optional[CredentialInfo] = None,
        singleton_ttl: Any = ...,
    ) -> None:
        del singleton_ttl  # consumed by Singleton.__new__
        if getattr(self, "_initialized", False):
            # Re-init on the cached singleton: refresh its info when the caller
            # eagerly fetched a fresh one (e.g. via ``get`` / ``create``).
            if info is not None:
                self._info = info
            return
        if service is None:
            from yggdrasil.databricks.credentials.credentials import Credentials

            service = Credentials.current()
        super().__init__(service=service)
        self.name = name
        self._info = info
        self._initialized = True

    def __getstate__(self) -> dict:
        return {"service": self.service, "name": self.name, "info": self._info}

    def __setstate__(self, state: dict) -> None:
        self.service = state["service"]
        self.name = state["name"]
        self._info = state.get("info")
        self._initialized = True

    # -- metadata (lazy fetch + cache) ---------------------------------
    @property
    def info(self) -> CredentialInfo:
        if self._info is None:
            self._info = self.service.get_info(self.name)
        return self._info

    def refresh(self) -> "Credential":
        self._info = self.service.get_info(self.name)
        return self

    @property
    def id(self) -> Optional[str]:
        return self.info.id

    @property
    def purpose(self) -> Any:
        return self.info.purpose

    @property
    def aws_iam_role(self) -> "Optional[AwsIamRole]":
        return self.info.aws_iam_role

    @property
    def aws_role_arn(self) -> Optional[str]:
        role = self.info.aws_iam_role
        return role.role_arn if role else None

    @property
    def comment(self) -> Optional[str]:
        return self.info.comment

    @property
    def owner(self) -> Optional[str]:
        return self.info.owner

    @property
    def read_only(self) -> bool:
        return bool(self.info.read_only)

    @property
    def isolation_mode(self) -> Any:
        return self.info.isolation_mode

    # -- refreshable AWS credentials -----------------------------------
    def temporary_credentials(self) -> "TemporaryCredentials":
        """Raw SDK :class:`TemporaryCredentials` from one
        ``generate_temporary_service_credential`` call."""
        return self.service.generate_temporary(self.name)

    def aws_credentials(self) -> "AwsCredentials":
        """A one-shot :class:`~yggdrasil.aws.config.AwsCredentials` (STS token)
        vended by this credential. Used by the refresher each cycle."""
        from yggdrasil.aws.config import AwsCredentials

        temp = self.temporary_credentials()
        aws = temp.aws_temp_credentials
        if aws is None:
            raise RuntimeError(
                f"credential {self.name!r} did not vend AWS temporary credentials "
                "(is it an AWS, SERVICE-purpose credential?)"
            )
        return AwsCredentials(
            access_key_id=aws.access_key_id,
            secret_access_key=aws.secret_access_key,
            session_token=aws.session_token,
            access_point=aws.access_point,
            expiration=_epoch_ms_to_iso(temp.expiration_time),
        )

    def aws_provider(self) -> "Any":
        """A singleton, self-refreshing AWS credentials provider for this
        credential."""
        from yggdrasil.databricks.credentials.provider import DatabricksCredentialAwsProvider

        host = getattr(self.client, "host", None) or "default"
        return DatabricksCredentialAwsProvider(f"{host}|{self.name}").bind(self)

    def aws_client(self, *, region: Optional[str] = None) -> "AWSClient":
        """A refreshable :class:`~yggdrasil.aws.client.AWSClient` backed by this
        credential — botocore re-mints the STS token before expiry, so the
        client (and its ``.s3``) stays valid indefinitely."""
        return self.aws_provider().aws_client(region=region)

    # -- lifecycle ------------------------------------------------------
    def update(self, **changes: Any) -> "Credential":
        updated = self.service.update(self.name, **changes)
        self._info = updated._info
        return self

    def delete(self, *, force: bool = False) -> None:
        self.service.delete(self.name, force=force)

    # -- explore --------------------------------------------------------
    @property
    def explore_url(self) -> URL:
        """Catalog Explorer deep-link to this credential."""
        return self.client.base_url.with_path(f"/explore/credentials/{self.name}")
