"""AWS account/identity service (STS-backed)."""
from __future__ import annotations

from yggdrasil.aws.client import AWSService

__all__ = ["AccountService"]


class AccountService(AWSService):
    """AWS account/identity service — STS-backed (``GetCallerIdentity``)."""

    @classmethod
    def service_name(cls) -> str:
        return "sts"

    def caller_identity(self) -> dict:
        return self.client.caller_identity()
