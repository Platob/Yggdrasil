"""AWS account resource — the account a client authenticates as.

:class:`AWSAccount` is the :class:`~yggdrasil.aws.client.AWSResource` for the
account: account id, region, STS caller identity, and a clickable
:attr:`explore_url` to the AWS Console home. Its :class:`AccountService` (in
``service.py``) is the thin STS-flavored binding. Reach it as
``AWSClient.current().account``.
"""
from __future__ import annotations

from typing import Any, Optional

from yggdrasil.aws.account.service import AccountService
from yggdrasil.aws.client import AWSResource, AWSService
from yggdrasil.aws.console import account_console_url
from yggdrasil.url import URL

__all__ = ["AWSAccount"]


class AWSAccount(AWSResource):
    """The AWS account a client authenticates as.

        >>> AWSClient.current().account
        AWSAccount(URL('https://us-east-1.console.aws.amazon.com/console/home?region=us-east-1'))
        >>> AWSClient.current().account.account_id
        '123456789012'
    """

    def __init__(self, service: Optional[AWSService] = None, **kwargs: Any) -> None:
        super().__init__(service=service if service is not None else AccountService.current(), **kwargs)

    @property
    def account_id(self) -> str:
        return self.client.account_id

    @property
    def region(self) -> Optional[str]:
        return self.service.region

    def caller_identity(self) -> dict:
        """STS ``GetCallerIdentity`` — account, ARN, and user id of the caller."""
        return self.client.caller_identity()

    @property
    def explore_url(self) -> URL:
        """AWS Console home for this account, pinned to its region."""
        return account_console_url(self.region)
