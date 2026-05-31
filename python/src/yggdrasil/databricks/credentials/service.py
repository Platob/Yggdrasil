"""Collection-level service for Unity Catalog **credentials**.

Wraps the Databricks ``credentials`` workspace API
(https://docs.databricks.com/api/workspace/credentials) as a dict-like
:class:`~yggdrasil.databricks.securable.SecurableMapping`::

    client.credentials["prod_s3"]                       # fetch (KeyError if absent)
    client.credentials["prod_s3"] = {"role_arn": arn}   # create (or update)
    del client.credentials["prod_s3"]                   # delete
    "prod_s3" in client.credentials                     # exists
    list(client.credentials)                            # names

Plus AWS ergonomics — :meth:`create_aws` and :meth:`aws_client` — and a
flexible :meth:`credential` finder that takes a handle, a name, or a credential
**id** (UUID, auto-detected).
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Iterator, Optional

from databricks.sdk.service.catalog import (
    AwsIamRole,
    CredentialInfo,
    CredentialPurpose,
    TemporaryCredentials,
)

from yggdrasil.databricks.credentials.resource import Credential
from yggdrasil.databricks.securable import SecurableMapping

if TYPE_CHECKING:
    from yggdrasil.aws.client import AWSClient

__all__ = ["Credentials"]

logger = logging.getLogger(__name__)

#: UC credential ids are UUIDs — used to dispatch a bare string to id vs name.
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _purpose(value: "str | CredentialPurpose") -> CredentialPurpose:
    return value if isinstance(value, CredentialPurpose) else CredentialPurpose(value)


class Credentials(SecurableMapping):
    """Dict-like service over a workspace's Unity Catalog credentials."""

    @property
    def _api(self):
        return self.client.workspace_client().credentials

    # -- flexible finder ------------------------------------------------
    def resolve(
        self,
        obj: "Credential | str | None" = None,
        *,
        credential_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Credential:
        """Coerce to a lazy :class:`Credential` handle.

        ``obj`` may be a :class:`Credential` (returned as-is), a name, or a
        credential **id** (a UUID, auto-detected by regex). ``credential_id`` /
        ``name`` pin the interpretation explicitly.
        """
        if obj is not None:
            if isinstance(obj, Credential):
                return obj
            if isinstance(obj, str):
                if _UUID_RE.match(obj):
                    credential_id = obj
                else:
                    name = obj
            else:
                raise TypeError(f"expected Credential | str | None, got {type(obj).__name__}")
        if credential_id is not None and name is None:
            name = self._name_for_id(credential_id)
        if name is None:
            raise ValueError("provide a Credential, a name, or a credential_id")
        return Credential(name, service=self)

    credential = resolve  # ergonomic alias

    def _name_for_id(self, credential_id: str) -> str:
        for info in self._infos():
            if info.id == credential_id:
                return info.name
        raise KeyError(f"no credential with id {credential_id!r}")

    # -- SecurableMapping hooks ----------------------------------------
    def _infos(self) -> Iterator[CredentialInfo]:
        return self._api.list_credentials()

    def get_info(self, name: str) -> CredentialInfo:
        return self._api.get_credential(name)

    def _resource(self, name: str, info: Any = None) -> Credential:
        return Credential(name, service=self, info=info)

    def delete(self, name: str, *, force: bool = False) -> None:
        self._api.delete_credential(name, force=force)

    def _apply(self, name: str, spec: Any, *, exists: bool) -> Credential:
        # ``svc[name] = arn_string`` is shorthand for ``{"role_arn": arn}``.
        spec = {"role_arn": spec} if isinstance(spec, str) else self._as_spec(spec)
        if exists:
            if "role_arn" in spec:
                spec["aws_iam_role"] = AwsIamRole(role_arn=spec.pop("role_arn"))
            return self.update(name, **spec)
        role_arn = spec.pop("role_arn", None) or spec.pop("aws_role_arn", None)
        if role_arn is None:
            raise ValueError(f"creating credential {name!r} needs a 'role_arn'")
        return self.create_aws(name, role_arn, **spec)

    # -- reads (typed override of the base list) -----------------------
    def list(self, *, purpose: "Optional[str | CredentialPurpose]" = None, **kwargs: Any) -> Iterator[Credential]:
        if purpose is not None:
            kwargs["purpose"] = _purpose(purpose)
        for info in self._api.list_credentials(**kwargs):
            yield Credential(info.name, service=self, info=info)

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
        vend refreshable STS tokens. ``**kwargs`` forward to the SDK.
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

    def generate_temporary(self, name: str) -> TemporaryCredentials:
        """One ``generate_temporary_service_credential`` call for *name*."""
        return self._api.generate_temporary_service_credential(name)

    # -- AWS convenience ------------------------------------------------
    def aws_client(self, name: str, *, region: Optional[str] = None) -> "AWSClient":
        """A refreshable :class:`~yggdrasil.aws.client.AWSClient` backed by the
        existing credential *name* (or id)."""
        return self.credential(name).aws_client(region=region)
