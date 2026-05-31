"""Collection-level service for Unity Catalog **credentials**.

Wraps the Databricks ``credentials`` workspace API
(https://docs.databricks.com/api/workspace/credentials) — list / get / create /
update / delete + temporary-credential generation — and adds AWS ergonomics:
:meth:`create_aws` (one call from a role ARN) and :meth:`aws_client` (a
refreshable client from an existing credential name). Reach it as
``client.credentials``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    AwsIamRole,
    CredentialInfo,
    CredentialPurpose,
    TemporaryCredentials,
)

from yggdrasil.databricks.client import DatabricksService
from yggdrasil.databricks.credentials.credential import Credential

if TYPE_CHECKING:
    from yggdrasil.aws.client import AWSClient

__all__ = ["Credentials"]

logger = logging.getLogger(__name__)


def _purpose(value: "str | CredentialPurpose") -> CredentialPurpose:
    return value if isinstance(value, CredentialPurpose) else CredentialPurpose(value)


class Credentials(DatabricksService):
    """Service over a workspace's Unity Catalog credentials."""

    @property
    def _api(self):
        return self.client.workspace_client().credentials

    # -- reads ----------------------------------------------------------
    def get_info(self, name: str) -> CredentialInfo:
        return self._api.get_credential(name)

    def credential(self, name: str) -> Credential:
        """A lazy :class:`Credential` handle (no API call until used)."""
        return Credential(name, service=self)

    __getitem__ = credential

    def get(self, name: str) -> Credential:
        return Credential(name, service=self, info=self.get_info(name))

    def list(self, *, purpose: "Optional[str | CredentialPurpose]" = None, **kwargs: Any) -> Iterator[Credential]:
        if purpose is not None:
            kwargs["purpose"] = _purpose(purpose)
        for info in self._api.list_credentials(**kwargs):
            yield Credential(info.name, service=self, info=info)

    def names(self, **kwargs: Any) -> "list[str]":
        return [c.name for c in self.list(**kwargs)]

    def exists(self, name: str) -> bool:
        try:
            self.get_info(name)
            return True
        except NotFound:
            return False

    # -- writes ---------------------------------------------------------
    def create_aws(
        self,
        name: str,
        role_arn: str,
        *,
        purpose: "str | CredentialPurpose" = CredentialPurpose.SERVICE,
        external_id: Optional[str] = None,
        comment: Optional[str] = None,
        read_only: bool = False,
        **kwargs: Any,
    ) -> Credential:
        """Create an AWS credential from an IAM role ARN — the easy path.

        Defaults to ``purpose=SERVICE`` so :meth:`Credential.aws_client` can
        vend refreshable STS tokens. ``**kwargs`` forward to the SDK
        (``skip_validation`` / ``isolation_mode`` / …).
        """
        info = self._api.create_credential(
            name=name,
            aws_iam_role=AwsIamRole(role_arn=role_arn, external_id=external_id),
            purpose=_purpose(purpose),
            comment=comment,
            read_only=read_only,
            **kwargs,
        )
        return Credential(name, service=self, info=info)

    def update(self, name: str, **changes: Any) -> Credential:
        info = self._api.update_credential(name, **changes)
        return Credential(info.name or name, service=self, info=info)

    def delete(self, name: str, *, force: bool = False) -> None:
        self._api.delete_credential(name, force=force)

    def generate_temporary(self, name: str) -> TemporaryCredentials:
        """One ``generate_temporary_service_credential`` call for *name*."""
        return self._api.generate_temporary_service_credential(name)

    # -- AWS convenience ------------------------------------------------
    def aws_client(self, name: str, *, region: Optional[str] = None) -> "AWSClient":
        """A refreshable :class:`~yggdrasil.aws.client.AWSClient` backed by the
        existing credential *name*."""
        return self.credential(name).aws_client(region=region)
