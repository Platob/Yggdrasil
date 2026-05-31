"""AWS account resource + service (folder module: resource in ``account.py``,
service in ``service.py``)."""
from __future__ import annotations

from .account import AWSAccount
from .service import AccountService

__all__ = ["AWSAccount", "AccountService"]
