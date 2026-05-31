"""AWS account resource + service (folder module: resource in ``resource.py``,
service in ``service.py``)."""
from __future__ import annotations

from .resource import AWSAccount
from .service import AccountService

__all__ = ["AWSAccount", "AccountService"]
