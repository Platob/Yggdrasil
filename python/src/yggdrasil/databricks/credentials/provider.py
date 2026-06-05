"""Refreshable AWS credentials backed by a Databricks UC credential.

A :class:`Credential` whose backing cloud identity is an AWS IAM role vends
short-lived STS credentials via ``generate_temporary_service_credential``. This
provider wraps that call in the standard
:class:`~yggdrasil.aws.provider.AwsCredentialsProvider` refresh cycle, so a
botocore session re-mints the token ~5 min before expiry — giving a long-lived,
self-refreshing :class:`~yggdrasil.aws.client.AWSClient` that never holds a
stale key.
"""
from __future__ import annotations

from typing import Any, Optional

from yggdrasil.aws.config import AwsCredentials
from yggdrasil.aws.provider import AwsCredentialsProvider
from yggdrasil.enums.mode import Mode

__all__ = ["DatabricksCredentialAwsProvider"]


class DatabricksCredentialAwsProvider(AwsCredentialsProvider):
    """Refresh AWS creds from a UC credential's temporary-credential endpoint.

    Singleton-cached per ``host|credential_name`` (inherited from the base),
    so every caller asking for the same credential shares one refresh cycle +
    one per-region :class:`AWSClient`.
    """

    def __init__(self, key: str) -> None:
        super().__init__(key)
        if not hasattr(self, "_credential"):
            self._credential: Any = None

    def bind(self, credential: Any) -> "DatabricksCredentialAwsProvider":
        """Point this provider at *credential* (a
        :class:`~yggdrasil.databricks.credentials.resource.Credential`)."""
        self._credential = credential
        return self

    def get_credentials(self, mode: Optional[Mode] = None) -> AwsCredentials:
        if self._credential is None:
            raise RuntimeError(
                "DatabricksCredentialAwsProvider is not bound to a credential; "
                "build it via Credential.aws_provider()."
            )
        return self._credential.aws_credentials()
