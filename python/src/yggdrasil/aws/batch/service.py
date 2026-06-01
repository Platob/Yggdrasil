"""AWS Batch service binding."""
from __future__ import annotations

from yggdrasil.aws.client import AWSService

__all__ = ["BatchService"]


class BatchService(AWSService):
    """AWS Batch service binding (the boto ``batch`` client lives on
    :attr:`boto_client` for future control-plane calls)."""

    @classmethod
    def service_name(cls) -> str:
        return "batch"
